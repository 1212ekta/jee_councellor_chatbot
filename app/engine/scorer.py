"""
Scorer — Phase 1, Step 3

The brain of the recommendation engine.

Takes every eligible cutoff row and computes a multi-factor score
that balances rank fit, interest match, institute quality, career
alignment, home state advantage, and future flexibility.

Score formula:
  overall = (
      0.40 * rank_fit
    + 0.25 * interest_match
    + 0.15 * institute_strength
    + 0.12 * career_alignment
    + 0.05 * home_state_bonus
    + 0.03 * flexibility
  )

Weights rationale:
  - rank_fit (40%): Most important — if student can't get in, nothing else matters.
  - interest_match (25%): Second — a mismatched branch leads to 4 years of struggle.
  - institute_strength (15%): Brand, placements, research all bundled here.
  - career_alignment (12%): Goals-to-branch fit (startup → CSE, govt → Civil, etc).
  - home_state_bonus (5%): Small but real — HS quota gives genuine seat advantage.
  - flexibility (3%): Future optionality — can switch? MBA possible? Dual degree?
"""

import json
from dataclasses import dataclass, field

from app.engine.interest_matcher import compute_interest_match
from app.engine.risk_classifier import RiskResult, assess_risk, classify_risk
from app.etl.loader import get_db
from app.models.request import StudentProfile
from app.services.knowledge_loader import KnowledgeLoader
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Weight configuration ──────────────────────────────────────────────────────
WEIGHTS = {
    "rank_fit":           0.40,
    "interest_match":     0.25,
    "institute_strength": 0.15,
    "career_alignment":   0.12,
    "home_state_bonus":   0.05,
    "flexibility":        0.03,
}

# Home State Quota codes that indicate state-level advantage
HOME_STATE_QUOTA_CODES = {"HS", "GO", "JK", "LA"}

# Institute type tier scores (0-1)
INSTITUTE_TYPE_BASE = {
    "IIT":   1.00,
    "IIIT":  0.75,
    "NIT":   0.65,
    "GFTI":  0.45,
    "State": 0.40,
}


@dataclass
class ScoreBreakdown:
    """Transparent score breakdown for one recommendation."""
    overall:           float = 0.0
    rank_fit:          float = 0.0
    interest_match:    float = 0.0
    institute_strength:float = 0.0
    career_alignment:  float = 0.0
    home_state_bonus:  float = 0.0
    flexibility:       float = 0.0
    # Extra context (not weighted, used for explainability)
    interest_explanation: str = ""
    career_paths:         list = field(default_factory=list)
    median_lpa:           float = 0.0
    coding_intensity:     int = 3
    research_scope:       int = 3


@dataclass
class ScoredRecommendation:
    """A fully scored and explained recommendation."""
    # Identity
    institute:     str = ""
    branch:        str = ""
    program:       str = ""
    city:          str = ""
    state:         str = ""
    institute_type:str = ""

    # Cutoff data
    opening_rank:  int = 0
    closing_rank:  int = 0
    year:          int = 2025
    round:         int = 6
    category:      str = ""
    exam_type:     str = ""
    state_quota:   str = ""
    gender:        str = ""
    home_state_advantage: bool = False

    # Scoring
    scores:        ScoreBreakdown = field(default_factory=ScoreBreakdown)
    risk:          RiskResult = None  # type: ignore

    # Explainability (filled by explainer later)
    why_institute: str = ""
    why_branch:    str = ""
    risks:         list = field(default_factory=list)
    opportunities: list = field(default_factory=list)
    career_roadmap:list = field(default_factory=list)

    # Institute metadata
    nirf_rank:            int | None = None
    placement_median_lpa: float | None = None
    research_score:       int | None = None
    coding_culture_score: int | None = None
    known_for:            str | None = None


# ── Institute metadata cache ──────────────────────────────────────────────────
_institute_cache: dict = {}

def _get_institute_meta(institute_name: str) -> dict:
    """Fetch institute metadata from DB. Cached in memory."""
    global _institute_cache
    if not _institute_cache:
        try:
            db = get_db()
            rows = db.execute(
                "SELECT name, type, city, state, tier, nirf_rank, "
                "research_score, placement_median_lpa, coding_culture_score, "
                "strengths, known_for FROM institutes"
            ).fetchall()
            for row in rows:
                _institute_cache[row[0]] = {
                    "type": row[1], "city": row[2], "state": row[3],
                    "tier": row[4], "nirf_rank": row[5],
                    "research_score": row[6], "placement_median_lpa": row[7],
                    "coding_culture_score": row[8],
                    "strengths": json.loads(row[9]) if row[9] else [],
                    "known_for": row[10],
                }
        except Exception as e:
            log.warning(f"Could not load institute metadata: {e}")

    # Try exact match, then partial match
    if institute_name in _institute_cache:
        return _institute_cache[institute_name]

    name_lower = institute_name.lower()
    for key, meta in _institute_cache.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return meta

    # Infer type from name for unknown institutes
    inferred_type = "GFTI"
    if "indian institute of technology" in name_lower or "iit " in name_lower:
        inferred_type = "IIT"
    elif "national institute of technology" in name_lower or "nit " in name_lower:
        inferred_type = "NIT"
    elif "indian institute of information technology" in name_lower or "iiit" in name_lower:
        inferred_type = "IIIT"

    return {
        "type": inferred_type, "city": None, "state": None,
        "tier": 3, "nirf_rank": None,
        "research_score": 2, "placement_median_lpa": 8.0,
        "coding_culture_score": 2, "strengths": [], "known_for": None,
    }


# ── Individual factor scorers ─────────────────────────────────────────────────

def _score_rank_fit(admission_probability: float) -> float:
    """
    Rank fit IS the admission probability.
    A higher probability means better rank match.
    We apply a small boost for ranks well inside the cutoff
    to reward genuinely strong fits over borderline ones.
    """
    return round(admission_probability, 4)


def _score_institute_strength(meta: dict) -> float:
    """
    Composite institute quality score.
    Combines: type tier + NIRF rank + research score + placement median.
    """
    # Base from institute type
    inst_type = meta.get("type", "GFTI")
    base = INSTITUTE_TYPE_BASE.get(inst_type, 0.40)

    # NIRF rank adjustment: rank 1 = +0.15, rank 50 = 0, rank 100+ = -0.05
    nirf = meta.get("nirf_rank")
    nirf_boost = 0.0
    if nirf:
        if nirf <= 5:
            nirf_boost = 0.15
        elif nirf <= 15:
            nirf_boost = 0.10
        elif nirf <= 30:
            nirf_boost = 0.05
        elif nirf <= 50:
            nirf_boost = 0.0
        else:
            nirf_boost = -0.03

    # Research score (1-5 → 0 to +0.05)
    research = meta.get("research_score", 2)
    research_boost = (research - 1) / 4 * 0.05

    # Placement median (LPA) — normalised to 0-0.10
    placement = meta.get("placement_median_lpa", 8.0)
    placement_boost = min(0.10, placement / 400)  # 40 LPA max = 0.10

    score = base + nirf_boost + research_boost + placement_boost
    return round(min(1.0, max(0.0, score)), 4)


def _score_interest_match(student: StudentProfile, branch_name: str) -> tuple[float, dict]:
    """
    Returns (score, full_match_result) from the interest matcher.
    """
    result = compute_interest_match(student, branch_name)
    return result["score"], result


def _score_career_alignment(student: StudentProfile, branch_name: str, interest_result: dict) -> float:
    """
    How well does this branch serve the student's career goals?

    Checks if student's active goals match the branch's 'suits_goals' list.
    Each matching goal adds to the score. Also considers:
    - Salary priority vs branch median LPA
    - Research goal vs research_scope
    - Startup goal vs coding_intensity
    """
    from app.etl.loader import get_db
    import json

    # Get branch suits_goals from interest_result or DB
    career_paths = interest_result.get("career_paths", [])
    suits_goals_str = ""

    try:
        db = get_db()
        row = db.execute(
            "SELECT suits_goals, coding_intensity, research_scope, median_lpa "
            "FROM branches WHERE name = ? OR short_name = ?",
            [branch_name, branch_name]
        ).fetchone()
        if row:
            suits_goals_str = row[0] or "[]"
            coding_intensity = row[1] or 3
            research_scope = row[2] or 3
            median_lpa = row[3] or 10.0
        else:
            suits_goals_str = "[]"
            coding_intensity = interest_result.get("coding_intensity", 3)
            research_scope = interest_result.get("research_scope", 3)
            median_lpa = interest_result.get("median_lpa", 10.0)
    except Exception:
        suits_goals_str = "[]"
        coding_intensity = interest_result.get("coding_intensity", 3)
        research_scope = interest_result.get("research_scope", 3)
        median_lpa = interest_result.get("median_lpa", 10.0)

    try:
        suits_goals = json.loads(suits_goals_str) if isinstance(suits_goals_str, str) else suits_goals_str
    except Exception:
        suits_goals = []

    active_goals = student.active_goals
    if not active_goals:
        return 0.6  # neutral if no goals stated

    score = 0.0
    max_score = 0.0

    # Goal matching
    goal_weights = {
        "startup": 0.20,
        "higher_studies_abroad": 0.15,
        "research": 0.20,
        "salary_priority": 0.15,
        "govt_job": 0.15,
        "wants_mba": 0.15,
    }

    for goal, weight in goal_weights.items():
        # H8-FIX: Only add to max_score when the student actually has that goal,
        # preventing focused students from being unfairly penalized
        if goal == "research" and student.wants_research:
            max_score += weight
            score += weight * (research_scope / 5.0)
        elif goal == "startup" and student.wants_startup:
            max_score += weight
            score += weight * (coding_intensity / 5.0)
        elif goal == "salary_priority" and student.salary_priority > 0.6:
            max_score += weight
            score += weight * min(1.0, median_lpa / 30.0)
        elif goal == "govt_job" and student.wants_govt_job:
            max_score += weight
            score += weight * (1.0 if "govt_job" in suits_goals else 0.2)
        elif goal == "wants_mba" and student.wants_mba:
            max_score += weight
            score += weight * (1.0 if "wants_mba" in suits_goals else 0.4)
        elif goal == "higher_studies_abroad" and student.wants_higher_studies_abroad:
            max_score += weight
            score += weight * (research_scope / 5.0)

    if max_score == 0:
        return 0.6

    return round(min(1.0, score / max_score + 0.4), 4)  # +0.4 base so never 0


def _score_home_state(student: StudentProfile, state_quota: str, institute_state: str) -> float:
    """
    Home state advantage score.
    Returns 1.0 if student benefits from home state quota, else 0.0.
    """
    student_state = student.home_state.strip().lower()
    quota = (state_quota or "").strip().upper()

    if quota in HOME_STATE_QUOTA_CODES:
        if institute_state and student_state in institute_state.lower():
            return 1.0
    return 0.0


def _score_flexibility(branch_name: str, interest_result: dict) -> float:
    """
    Future flexibility score.
    High flexibility = branch keeps many options open.
    CSE/MnC → high (can go to any sector)
    Mining/Textile → low (niche sector)
    """
    branch_lower = branch_name.lower()

    # High flexibility branches
    if any(k in branch_lower for k in [
        "computer science", "mathematics and computing", "data science",
        "artificial intelligence", "electrical engineering"
    ]):
        return 0.95

    # Medium-high
    if any(k in branch_lower for k in [
        "electronics", "mechanical", "engineering physics", "economics"
    ]):
        return 0.75

    # Medium
    if any(k in branch_lower for k in [
        "civil", "chemical", "biotechnology", "aerospace"
    ]):
        return 0.55

    # Low flexibility (niche)
    if any(k in branch_lower for k in [
        "mining", "textile", "ceramic", "metallurg", "naval", "ocean",
        "agricultural", "dairy", "carpet", "printing", "rubber"
    ]):
        return 0.30

    return 0.55  # default moderate


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_recommendation(
    student: StudentProfile,
    row: dict,
) -> ScoredRecommendation | None:
    """
    Score a single cutoff row for a student.

    Args:
        student: StudentProfile with rank, interests, goals
        row:     Dict from DuckDB cutoff query

    Returns:
        ScoredRecommendation or None if not eligible
    """
    institute  = row.get("institute", "")
    branch     = row.get("branch", "")
    program    = row.get("program", branch)
    closing    = row.get("closing_rank", 999999)
    opening    = row.get("opening_rank", closing)
    seat_gender= row.get("gender", "Gender-Neutral")
    state_quota= row.get("state_quota", "AI")
    round_num  = row.get("round", 6)
    exam_type  = row.get("exam_type", "JEE_ADVANCED")
    category   = row.get("category", "OPEN")
    year       = row.get("year", 2025)

    # Use appropriate rank based on exam type
    effective_rank = (
        student.jee_advanced_rank
        if exam_type == "JEE_ADVANCED" and student.jee_advanced_rank
        else student.jee_main_rank or student.effective_rank
    )

    if not effective_rank:
        return None

    # ── Risk assessment ───────────────────────────────────────────────────────
    risk = assess_risk(
        student_rank=effective_rank,
        opening_rank=opening,
        closing_rank=closing,
        student_gender=student.gender,
        seat_gender=seat_gender,
        round_number=round_num,
    )

    if not risk.is_eligible:
        return None

    # ── Institute metadata ────────────────────────────────────────────────────
    meta = _get_institute_meta(institute)
    inst_type  = meta.get("type", "GFTI")
    city       = meta.get("city") or ""
    inst_state = meta.get("state") or ""

    # ── Individual factor scores ──────────────────────────────────────────────
    rank_fit         = _score_rank_fit(risk.admission_probability)
    interest_score, interest_result = _score_interest_match(student, branch)
    inst_strength    = _score_institute_strength(meta)
    career_align     = _score_career_alignment(student, branch, interest_result)
    home_state       = _score_home_state(student, state_quota, inst_state)
    flexibility      = _score_flexibility(branch, interest_result)

    # ── Weighted overall score ────────────────────────────────────────────────
    weights = KnowledgeLoader().scorer_weights()
    overall = (
        weights.get("rank_fit", 0.40)           * rank_fit
        + weights.get("interest_match", 0.25)   * interest_score
        + weights.get("institute_strength", 0.15)* inst_strength
        + weights.get("career_alignment", 0.12) * career_align
        + weights.get("home_state_bonus", 0.05) * home_state
        + weights.get("flexibility", 0.03)      * flexibility
    )

    scores = ScoreBreakdown(
        overall=round(overall, 4),
        rank_fit=round(rank_fit, 4),
        interest_match=round(interest_score, 4),
        institute_strength=round(inst_strength, 4),
        career_alignment=round(career_align, 4),
        home_state_bonus=round(home_state, 4),
        flexibility=round(flexibility, 4),
        interest_explanation=interest_result.get("explanation", ""),
        career_paths=interest_result.get("career_paths", []),
        median_lpa=interest_result.get("median_lpa", 0.0),
        coding_intensity=interest_result.get("coding_intensity", 3),
        research_scope=interest_result.get("research_scope", 3),
    )

    return ScoredRecommendation(
        institute=institute,
        branch=branch,
        program=program,
        city=city,
        state=inst_state,
        institute_type=inst_type,
        opening_rank=opening,
        closing_rank=closing,
        year=year,
        round=round_num,
        category=category,
        exam_type=exam_type,
        state_quota=state_quota,
        gender=seat_gender,
        home_state_advantage=(home_state > 0),
        scores=scores,
        risk=risk,
        nirf_rank=meta.get("nirf_rank"),
        placement_median_lpa=meta.get("placement_median_lpa"),
        research_score=meta.get("research_score"),
        coding_culture_score=meta.get("coding_culture_score"),
        known_for=meta.get("known_for"),
    )


def score_all(
    student: StudentProfile,
    cutoff_rows: list[dict],
) -> list[ScoredRecommendation]:
    """
    Score all cutoff rows for a student and return sorted by overall score.
    Filters out ineligible rows automatically.
    """
    results = []
    skipped = 0

    for row in cutoff_rows:
        rec = score_recommendation(student, row)
        if rec is not None:
            results.append(rec)
        else:
            skipped += 1

    results.sort(key=lambda r: r.scores.overall, reverse=True)

    log.info(
        f"Scored {len(cutoff_rows)} rows → "
        f"{len(results)} eligible, {skipped} filtered"
    )
    return results


def bucket_scored(
    scored: list[ScoredRecommendation],
) -> dict[str, list[ScoredRecommendation]]:
    """Group scored recommendations by risk level."""
    buckets: dict[str, list[ScoredRecommendation]] = {
        "dream": [], "target": [], "safe": [], "very_safe": []
    }
    for rec in scored:
        key = rec.risk.risk_level.lower().replace(" ", "_")
        if key in buckets:
            buckets[key].append(rec)
    return buckets
