"""
POST /recommend  — full recommendation pipeline
POST /analyze-profile — profile analysis without full recommendations

Pipeline:
  StudentProfile
    → Persona Detection
    → Fetch eligible cutoff rows from DuckDB
    → Score all rows (rank fit + interest + institute + career + home state + flexibility)
    → Risk classify (Dream / Target / Safe / Very Safe)
    → Bucket + sort by overall score
    → For top N: compute compatibility (8-dim) + reason codes + explanation
    → Return structured response with session_id for sharing
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import db, knowledge_loader, rag_pipeline
from app.engine.compatibility import compute_compatibility
from app.engine.explainer import (
    StructuredExplanation,
    build_counselor_insight,
    build_student_summary,
    explain,
)
from app.engine.interest_matcher import compute_interest_match
from app.engine.persona import get_all_persona_scores, infer_persona
from app.engine.reason_codes import compute_reason_codes
from app.engine.scorer import bucket_scored, score_all
from app.models.request import StudentProfile
from app.services.knowledge_loader import KnowledgeLoader
from app.utils.cache import cached
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

# How many recs per bucket to fully explain (LLM is expensive — cap it)
EXPLAIN_LIMIT = {"dream": 3, "target": 5, "safe": 4, "very_safe": 3}
MAX_DB_ROWS   = 50_000   # safety cap on DuckDB fetch


def _fetch_eligible_rows(student: StudentProfile, conn, max_rank_buffer: float = 1.35) -> list[dict]:
    """
    Fetch cutoff rows that could plausibly be eligible for this student.
    Uses a generous rank buffer so the scorer can make fine distinctions.
    Buffer of 1.35 = fetch rows with closing_rank up to 35% beyond student's rank.
    """
    rank = student.effective_rank
    max_closing = max(int(rank * max_rank_buffer), 5000)  # floor: always fetch min 5000 closing rank

    # Gender filter: males can't have Female-Only seats
    gender_filter = ""
    if student.gender == "male":
        gender_filter = "AND gender != 'Female-Only'"

    # Exam type filter
    exam_filter = ""
    if student.preferred_exam == "JEE_ADVANCED":
        exam_filter = "AND exam_type = 'JEE_ADVANCED'"
    elif student.preferred_exam == "JEE_MAIN":
        exam_filter = "AND exam_type = 'JEE_MAIN'"

    # Category filter — only show rows for this student's category + OPEN
    # A student can always apply to OPEN seats if their rank qualifies
    category = student.category.upper()
    if category == "OPEN":
        category_filter = "AND category = 'OPEN'"
    else:
        # Reserved category students can compete in OPEN + their own category
        category_filter = f"AND category IN ('OPEN', '{category}')"

    sql = f"""
        SELECT institute, program, branch, category, gender,
               opening_rank, closing_rank, round, year, exam_type, state_quota
        FROM cutoffs
        WHERE closing_rank <= ?
          {gender_filter}
          {exam_filter}
          {category_filter}
        ORDER BY closing_rank
        LIMIT {MAX_DB_ROWS}
    """
    rows = conn.execute(sql, [max_closing]).fetchall()
    cols = ["institute","program","branch","category","gender",
            "opening_rank","closing_rank","round","year","exam_type","state_quota"]
    return [dict(zip(cols, r)) for r in rows]


def _build_recommendation_dict(
    rec,
    expl:    StructuredExplanation,
    comp,
    student: StudentProfile,
) -> dict:
    """Serialize one ScoredRecommendation + explanation into API dict."""
    return {
        # Identity
        "institute":          rec.institute,
        "branch":             rec.branch,
        "program":            rec.program,
        "institute_type":     rec.institute_type,
        "city":               rec.city,
        "state":              rec.state,

        # Cutoff data
        "opening_rank":       rec.opening_rank,
        "closing_rank":       rec.closing_rank,
        "year":               rec.year,
        "round":              rec.round,
        "category":           rec.category,
        "exam_type":          rec.exam_type,
        "state_quota":        rec.state_quota,
        "home_state_advantage": rec.home_state_advantage,

        # Risk
        "admission_probability": rec.risk.admission_probability,
        "risk_level":          rec.risk.risk_level,
        "probability_label":   rec.risk.probability_label,
        "safety_margin":       rec.risk.safety_margin,
        "counselor_note":      rec.risk.counselor_note,

        # Scores
        "scores": {
            "overall":            rec.scores.overall,
            "rank_fit":           rec.scores.rank_fit,
            "interest_match":     rec.scores.interest_match,
            "institute_strength": rec.scores.institute_strength,
            "career_alignment":   rec.scores.career_alignment,
            "home_state_bonus":   rec.scores.home_state_bonus,
            "flexibility":        rec.scores.flexibility,
        },

        # 8-dim compatibility
        "compatibility": comp.to_dict(),

        # Reason codes (machine + human readable)
        "reason_codes": expl.reason_codes.to_dict(),

        # Full structured explanation
        "explanation": {
            "why_institute":       expl.why_institute,
            "why_branch":          expl.why_branch,
            "pros":                expl.pros,
            "cons":                expl.cons,
            "career_paths":        expl.career_paths,
            "career_roadmap":      expl.career_roadmap,
            "higher_studies":      expl.higher_studies,
            "recruiters":          expl.recruiters,
            "expected_salary":     expl.expected_salary,
            "risks":               expl.risks,
            "counselor_narrative": expl.counselor_narrative,
        },

        # Institute metadata
        "nirf_rank":           rec.nirf_rank,
        "placement_median_lpa":rec.placement_median_lpa,
        "research_score":      rec.research_score,
        "coding_culture_score":rec.coding_culture_score,
        "known_for":           rec.known_for,
    }


def _save_session(session_id: str, student: StudentProfile, response: dict, conn) -> None:
    """Persist session to DuckDB for share-link feature."""
    try:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, input_json, output_json, created_at) VALUES (?,?,?,?)",
            [
                session_id,
                student.model_dump_json(),
                json.dumps({
                    "student_summary":   response["student_summary"],
                    "counselor_insight": response["counselor_insight"],
                    "bucket_counts":     response["bucket_counts"],
                    "persona":           response["persona"]["label"],
                }),
                datetime.now(timezone.utc).isoformat(),
            ]
        )
        conn.commit()
    except Exception as e:
        log.warning(f"Session save failed: {e}")


@router.post("/recommend")
def recommend(
    student:  StudentProfile,
    explain_all: bool  = Query(False, description="Explain every recommendation (slow)"),
    conn     = Depends(db),
    kl       = Depends(knowledge_loader),
    pipeline = Depends(rag_pipeline),
):
    """
    Core recommendation endpoint.

    Returns Dream / Target / Safe / Very Safe recommendations with:
    - Multi-factor scores
    - 8-dimension compatibility profile
    - Machine-readable reason codes
    - Structured explanation (why_institute, why_branch, pros, cons, roadmap...)
    - Optional LLM counselor narrative (grounded via RAG)
    - Shareable session_id
    """
    session_id = str(uuid.uuid4())
    log.info(f"[{session_id}] Recommend: rank={student.effective_rank} cat={student.category} state={student.home_state}")

    # ── 1. Persona ────────────────────────────────────────────────────────────
    persona        = infer_persona(student)
    persona_scores = get_all_persona_scores(student)

    # ── 2. Fetch eligible cutoff rows ─────────────────────────────────────────
    rows = _fetch_eligible_rows(student, conn)
    log.info(f"[{session_id}] Fetched {len(rows)} candidate rows")

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No cutoff data found. Please ensure the dataset is loaded."
        )

    # ── 3. Score all rows ─────────────────────────────────────────────────────
    scored = score_all(student, rows)

    # ── 4. Bucket by risk level ───────────────────────────────────────────────
    buckets  = bucket_scored(scored)
    limits   = kl.max_results_per_bucket()

    bucket_counts = {k: len(v) for k, v in buckets.items()}
    log.info(f"[{session_id}] Buckets: {bucket_counts}")

    # ── 5. Explain top N per bucket ───────────────────────────────────────────
    response_buckets = {}

    for level in ["dream", "target", "safe", "very_safe"]:
        limit = len(buckets[level]) if explain_all else limits.get(level, 5)
        top   = buckets[level][:limit]
        recs  = []

        for rec in top:
            # Get interest match details for compatibility
            im = compute_interest_match(student, rec.branch)

            # 8-dim compatibility
            comp = compute_compatibility(
                student=student,
                risk=rec.risk,
                branch_name=rec.branch,
                branch_domain=im.get("matched_profile", "CS"),
                suits_goals=kl.get_branch(rec.branch).get("suits_goals", []),
                median_lpa=im.get("median_lpa", 10.0),
                avg_salary_lpa=im.get("median_lpa", 10.0) * 1.3,
                coding_intensity=im.get("coding_intensity", 3),
                research_scope=im.get("research_scope", 3),
                inst_type=rec.institute_type or "GFTI",
                inst_tier=(kl.get_institute(rec.institute).get("tier") or 3),
                inst_city=rec.city or "",
                inst_state=rec.state or "",
                nirf_rank=rec.nirf_rank,
                inst_research_score=rec.research_score or 2,
                inst_coding_score=rec.coding_culture_score or 2,
                inst_placement_lpa=rec.placement_median_lpa or 8.0,
                home_state_advantage=rec.home_state_advantage,
                flexibility_score=rec.scores.flexibility,
            )

            # Full structured explanation (with RAG + optional LLM)
            expl = explain(student, rec, persona, comp, pipeline=pipeline)

            recs.append(_build_recommendation_dict(rec, expl, comp, student))

        response_buckets[level] = recs

    # ── 6. Build top-level summaries ──────────────────────────────────────────
    student_summary    = build_student_summary(student, persona)
    counselor_insight  = build_counselor_insight(student, persona, len(scored), bucket_counts)

    response = {
        "session_id":       session_id,
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "share_url":        f"/sessions/{session_id}",

        # Student summary
        "student_summary":  student_summary,
        "counselor_insight":counselor_insight,

        # Persona
        "persona": {
            "id":          persona.id,
            "label":       persona.label,
            "icon":        persona.icon,
            "confidence":  persona.confidence,
            "secondary":   persona.secondary,
            "description": persona.description,
            "all_scores":  persona_scores,
        },

        # Buckets
        "buckets":      response_buckets,
        "bucket_counts":bucket_counts,
        "total_scored": len(scored),
        "total_fetched":len(rows),
    }

    # ── 7. Save session ───────────────────────────────────────────────────────
    _save_session(session_id, student, response, conn)

    return response


@router.post("/analyze-profile")
def analyze_profile(student: StudentProfile, kl=Depends(knowledge_loader)):
    """
    Lightweight profile analysis — no full recommendation run.
    Returns persona, top branch matches, strategic advice.
    Fast — no DB query, no scoring, no LLM.
    """
    from app.engine.interest_matcher import get_top_branches_for_student

    persona        = infer_persona(student)
    persona_scores = get_all_persona_scores(student)
    top_branches   = get_top_branches_for_student(student, top_n=5)

    # Rank context
    rank = student.effective_rank
    if rank <= 500:
        rank_band = "Top 500 — IIT Bombay/Delhi/Madras CSE is within reach"
    elif rank <= 2000:
        rank_band = "Top 2000 — Old IITs (non-CSE) or new IIT CSE"
    elif rank <= 5000:
        rank_band = "Top 5000 — IIT (new) CSE / NIT Trichy-Warangal top branches"
    elif rank <= 15000:
        rank_band = "Top 15000 — Good NITs, IIITs across all branches"
    elif rank <= 50000:
        rank_band = "Top 50000 — NITs (state quota advantage), IIITs, GFTIs"
    else:
        rank_band = "Consider state engineering colleges + CSAB counselling"

    weights = kl.scorer_weights()

    return {
        "rank":              rank,
        "rank_band":         rank_band,
        "category":          student.category,
        "home_state":        student.home_state,

        "persona": {
            "primary":     {"id": persona.id, "label": persona.label,
                            "icon": persona.icon, "confidence": persona.confidence},
            "secondary":   persona.secondary,
            "all_scores":  persona_scores,
            "advice":      {
                "opener":         persona.counselor_opener,
                "branch_advice":  persona.branch_advice,
                "inst_advice":    persona.institute_advice,
                "career_horizon": persona.career_horizon,
            }
        },

        "top_branch_matches": [
            {
                "branch":      b["branch"],
                "score":       b["score"],
                "explanation": b["explanation"],
                "career_paths":b.get("career_paths", [])[:3],
            }
            for b in top_branches
        ],

        "active_goals":      student.active_goals,
        "scoring_weights":   weights,

        "strategic_advice": (
            f"With rank {rank} ({student.category}), your strongest path is: "
            f"{rank_band}. {persona.branch_advice}"
        ),
    }


@router.get("/sessions/{session_id}")
def get_session(session_id: str, conn=Depends(db)):
    """Retrieve a saved recommendation session by ID (for share links)."""
    try:
        row = conn.execute(
            "SELECT input_json, output_json, created_at FROM sessions WHERE id = ?",
            [session_id]
        ).fetchone()
    except Exception:
        raise HTTPException(status_code=500, detail="Session lookup failed")

    if not row:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return {
        "session_id":  session_id,
        "created_at":  row[2],
        "input":       json.loads(row[0]),
        "summary":     json.loads(row[1]),
    }
