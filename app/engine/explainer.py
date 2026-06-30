"""
Explainability Engine — refactored to use RAG pipeline

Returns fully structured explanation objects (not a single paragraph).
Two-layer design:
  Layer 1: deterministic structured fields from scores + knowledge
  Layer 2: optional LLM narrative via RAG pipeline
"""

import json
from dataclasses import dataclass, field

from app.engine.compatibility import CompatibilityProfile
from app.engine.persona import CareerPersona
from app.engine.rag import RAGPipeline, RecommendationContext, get_pipeline
from app.engine.reason_codes import ReasonCodes, compute_reason_codes
from app.engine.scorer import ScoredRecommendation
from app.models.request import StudentProfile
from app.services.knowledge_loader import knowledge
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class StructuredExplanation:
    """
    Fully structured explanation for one recommendation.
    Every field is independently useful — frontend can render each section.
    """
    why_institute:      str = ""
    why_branch:         str = ""
    pros:               list[str] = field(default_factory=list)
    cons:               list[str] = field(default_factory=list)
    career_paths:       list[str] = field(default_factory=list)
    career_roadmap:     list[str] = field(default_factory=list)
    higher_studies:     list[str] = field(default_factory=list)
    recruiters:         list[str] = field(default_factory=list)
    expected_salary:    str = ""
    risks:              list[str] = field(default_factory=list)
    counselor_narrative:str = ""
    reason_codes:       ReasonCodes = field(default_factory=ReasonCodes)
    rag_context:        dict = field(default_factory=dict)   # raw context for frontend/debug

    def to_dict(self) -> dict:
        return {
            "why_institute":       self.why_institute,
            "why_branch":          self.why_branch,
            "pros":                self.pros,
            "cons":                self.cons,
            "career_paths":        self.career_paths,
            "career_roadmap":      self.career_roadmap,
            "higher_studies":      self.higher_studies,
            "recruiters":          self.recruiters,
            "expected_salary":     self.expected_salary,
            "risks":               self.risks,
            "counselor_narrative": self.counselor_narrative,
            "reason_codes":        self.reason_codes.to_dict(),
        }


# ── Structured field builders (deterministic, no LLM) ────────────────────────

def _why_institute(rec: ScoredRecommendation, student: StudentProfile,
                   ctx: RecommendationContext) -> str:
    parts = []
    if rec.scores.institute_strength >= 0.80:
        known = ctx.inst_known_for or ""
        parts.append(
            f"{rec.institute} is one of India's premier {rec.institute_type} institutions"
            + (f", known for {known.lower()}" if known else "") + "."
        )
    else:
        parts.append(f"{rec.institute} offers a solid {rec.institute_type} education.")

    gap = rec.risk.rank_gap
    if gap <= 0:
        pct = abs(gap / rec.closing_rank * 100) if rec.closing_rank else 0
        parts.append(
            f"Your rank is {abs(gap):,} places ({pct:.0f}%) ahead of last year's "
            f"closing rank — a comfortable buffer."
        )
    else:
        parts.append(
            f"Your rank exceeds last year's closing rank by {gap:,} places, "
            f"making this a stretch goal worth applying in early rounds."
        )

    if rec.home_state_advantage:
        parts.append(
            f"As a student from {student.home_state}, the Home State quota "
            f"meaningfully improves your admission chances here."
        )

    if ctx.inst_placement_median and ctx.inst_placement_median >= 15:
        parts.append(
            f"The median placement package here is approximately "
            f"₹{ctx.inst_placement_median:.0f} LPA."
        )
    return " ".join(parts)


def _why_branch(rec: ScoredRecommendation, student: StudentProfile,
                ctx: RecommendationContext) -> str:
    parts = []
    score = rec.scores.interest_match
    quality = ("excellent" if score >= 0.85 else
               "good" if score >= 0.65 else
               "moderate" if score >= 0.50 else "limited")

    parts.append(
        f"{rec.branch} is a {quality} interest match for your profile "
        f"({score:.0%} alignment with what this branch demands)."
    )

    if ctx.career_paths:
        parts.append(
            f"Common career paths include: {', '.join(ctx.career_paths[:3])}."
        )

    if ctx.median_lpa and ctx.median_lpa >= 12:
        parts.append(f"Median placements run around ₹{ctx.median_lpa:.0f} LPA.")

    active = student.active_goals
    if "startup" in active and rec.scores.career_alignment >= 0.70:
        parts.append("The coding culture here strongly supports startup ambitions.")
    if "higher_studies_abroad" in active and (rec.research_score or 0) >= 4:
        parts.append("Strong research groups will help your MS/PhD applications.")
    if "research" in active and (rec.research_score or 0) >= 4:
        parts.append("Publication opportunities exist from as early as Year 2.")

    return " ".join(parts)


def _build_pros(rec: ScoredRecommendation, comp: CompatibilityProfile,
                reason_codes: ReasonCodes) -> list[str]:
    pros = []
    label_to_pro = {
        "✓ Strong Rank Match":       f"Your rank fits comfortably — {rec.risk.probability_label}",
        "🏠 Home State Advantage":   "Home State quota gives a meaningful seat advantage",
        "✓ Excellent Placements":    f"Strong placement record — median ~₹{rec.placement_median_lpa or 'N/A'} LPA",
        "✓ Career Goal Aligned":     "Branch directly supports your stated career goals",
        "✓ Strong Research Fit":     "Active research groups for academic/PhD aspirations",
        "✓ Great Startup Ecosystem": "E-Cell and startup culture are well-established",
        "✓ Strong Coding Culture":   "Strong competitive programming and hackathon culture",
        "✓ Top Institute Brand":     f"{rec.institute_type} brand opens doors across all sectors",
        "✓ High Career Flexibility": "Branch keeps many career paths open",
        "✓ Excellent Interest Match":"Your stated interests align closely with this branch",
        "✓ Very High Admission Confidence": "Near-certain admission — reliable safety option",
        "✓ MBA-Friendly Branch":     "Strong MBA admit record from this branch",
        "✓ Strong Higher Studies Profile": "Institute name strengthens MS/PhD applications",
        "✓ Strong GATE / PSU Pathway": "Excellent pathway to PSUs via GATE",
    }
    for label in reason_codes.labels:
        if label in label_to_pro:
            pros.append(label_to_pro[label])
    if rec.scores.flexibility >= 0.85 and "✓ High Career Flexibility" not in reason_codes.labels:
        pros.append("High career optionality — many sectors recruit from here")
    return pros[:6]


def _build_cons(rec: ScoredRecommendation, comp: CompatibilityProfile,
                student: StudentProfile, reason_codes: ReasonCodes) -> list[str]:
    cons = []
    for w in reason_codes.warnings:
        if "Stretch Rank" in w:
            cons.append(f"Admission risk: rank is {rec.risk.rank_gap:+,} beyond cutoff ({rec.risk.probability_label})")
        elif "Interest Mismatch" in w:
            cons.append(f"Interest alignment is only {rec.scores.interest_match:.0%} — daily work may feel misaligned")
        elif "Weak Placements" in w:
            cons.append("Placement record is below average for this tier")
        elif "Limited Research" in w:
            cons.append("Limited research output may weaken PhD applications — seek external internships")
        elif "Low Flexibility" in w:
            cons.append("Niche branch — fewer sectors to pivot into later")
        elif "Niche Branch" in w:
            cons.append("Specialised area with fewer mainstream recruiters")

    if not cons:
        cons.append("No significant concerns identified for this choice")
    return cons[:5]


def _build_risks(rec: ScoredRecommendation, student: StudentProfile) -> list[str]:
    risks = []
    if rec.risk.risk_level == "Dream":
        risks.append(
            f"Cutoff risk: closing rank was {rec.closing_rank:,} last year. "
            "Cutoffs vary — apply in Round 1 but have solid backups."
        )
    if rec.scores.interest_match < 0.55:
        risks.append(
            "Interest mismatch can hurt motivation, GPA, and internship performance over 4 years."
        )
    if student.wants_startup and rec.scores.flexibility < 0.50:
        risks.append("This branch limits the startup sectors you can target.")
    if student.wants_higher_studies_abroad and (rec.research_score or 0) <= 2:
        risks.append(
            "Weak research infrastructure here — compensate with external research internships."
        )
    if not risks:
        risks.append("No significant risks beyond normal academic challenges.")
    return risks


def _fallback_narrative(rec: ScoredRecommendation, persona: CareerPersona,
                        comp: CompatibilityProfile) -> str:
    risk  = rec.risk.risk_level
    score = comp.overall_compatibility
    if risk == "Very Safe" and score >= 0.75:
        return (
            f"{rec.branch} at {rec.institute} is a strong, well-matched choice. "
            f"Your rank gives you {rec.risk.probability_label}, and the branch aligns "
            f"well with your {persona.label.lower()} goals. Make this a high priority."
        )
    elif risk == "Very Safe":
        # M15-FIX: Handle Very Safe with lower compatibility score
        return (
            f"{rec.branch} at {rec.institute} is a very safe admission choice at "
            f"{rec.risk.probability_label}. Your rank is comfortably within the cutoff. "
            f"Consider this a reliable fallback while pursuing better-matched options."
        )
    elif risk == "Safe":
        # M15-FIX: 'Safe' is the most common bucket — was missing a dedicated template
        return (
            f"{rec.branch} at {rec.institute} is a solid, confident choice with "
            f"{rec.risk.probability_label}. The branch aligns {score:.0%} with your "
            f"{persona.label.lower()} profile. Include this as a strong primary option "
            f"in your preference list."
        )
    elif risk == "Target":
        return (
            f"{rec.branch} at {rec.institute} is a realistic target — {rec.risk.probability_label}. "
            f"Apply in early rounds and keep solid backups ready. "
            f"The branch suits your {persona.label.lower()} orientation well."
        )
    elif risk == "Dream":
        return (
            f"{rec.branch} at {rec.institute} is a genuine stretch at {rec.risk.probability_label}. "
            f"Include it in Round 1 — cutoffs fluctuate yearly — but don't plan around it. "
            f"Focus energy on your Target choices."
        )
    return (
        f"{rec.branch} at {rec.institute} offers a {risk.lower()} admission "
        f"prospect with good overall compatibility ({score:.0%}). A solid option."
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def explain(
    student:       StudentProfile,
    rec:           ScoredRecommendation,
    persona:       CareerPersona,
    comp:          CompatibilityProfile,
    pipeline:      RAGPipeline | None = None,
) -> StructuredExplanation:
    """
    Generate the complete structured explanation for one recommendation.
    Uses RAG pipeline for LLM narrative; all other fields are deterministic.
    """
    if pipeline is None:
        pipeline = get_pipeline()

    # Stage 1: retrieve context via pipeline
    ctx = pipeline.get_context(rec.institute, rec.branch)

    # Stage 2: compute reason codes
    reason_codes = compute_reason_codes(student, rec, comp)

    # Stage 3: build structured fields
    why_institute = _why_institute(rec, student, ctx)
    why_branch    = _why_branch(rec, student, ctx)
    pros          = _build_pros(rec, comp, reason_codes)
    cons          = _build_cons(rec, comp, student, reason_codes)
    risks         = _build_risks(rec, student)
    higher_studies= ctx.top_ms_programs + ctx.top_phd_programs
    recruiters    = (ctx.inst_top_recruiters or ctx.tier1_recruiters)[:6]

    salary_med = ctx.inst_placement_median or ctx.median_lpa or 0
    expected_salary = (
        f"₹{round(salary_med*0.6):.0f}–{round(salary_med*2.0):.0f} LPA "
        f"(median ~₹{salary_med:.0f} LPA)"
        if salary_med else "₹8–20 LPA (estimated)"
    )

    # Stage 4: LLM narrative (with RAG-grounded context)
    narrative = pipeline.generate_narrative(
        context=ctx,
        student_rank=student.effective_rank,
        student_category=student.category,
        student_home_state=student.home_state,
        persona_label=persona.label,
        active_goals=student.active_goals,
        prob_label=rec.risk.probability_label,
        risk_level=rec.risk.risk_level,
        compatibility_pct=comp.overall_compatibility,
        pros=pros,
        cons=cons,
    )
    if not narrative:
        narrative = _fallback_narrative(rec, persona, comp)

    return StructuredExplanation(
        why_institute=why_institute,
        why_branch=why_branch,
        pros=pros,
        cons=cons,
        career_paths=ctx.career_paths,
        career_roadmap=ctx.roadmap,
        higher_studies=higher_studies[:4],
        recruiters=recruiters,
        expected_salary=expected_salary,
        risks=risks,
        counselor_narrative=narrative,
        reason_codes=reason_codes,
        rag_context=ctx.to_dict(),
    )


# ── Student-level summaries ───────────────────────────────────────────────────

def build_student_summary(student: StudentProfile, persona: CareerPersona) -> str:
    rank_desc = (
        f"JEE Advanced rank {student.jee_advanced_rank:,}"
        if student.jee_advanced_rank
        else f"JEE Main rank {student.jee_main_rank:,}"
    )
    goals = student.active_goals
    goal_str = f"with goals around {', '.join(goals[:3])}" if goals else "with open goals"
    hs = (f" As a {student.home_state} student, you may benefit from "
          f"Home State quota at NITs." if student.home_state else "")
    return (
        f"Based on your {rank_desc} ({student.category} category) {goal_str}, "
        f"you've been identified as a {persona.icon} {persona.label}. "
        f"{persona.counselor_opener}{hs}"
    )


def build_counselor_insight(
    student:       StudentProfile,
    persona:       CareerPersona,
    total_matches: int,
    bucket_counts: dict,
) -> str:
    d = bucket_counts.get("dream", 0)
    t = bucket_counts.get("target", 0)
    s = bucket_counts.get("safe", 0)
    v = bucket_counts.get("very_safe", 0)

    strategy = (
        f"Your rank gives you access to {total_matches} realistic options — "
        f"{d} dream, {t} target, {s} safe, and {v} very safe choices. "
    )
    if d == 0:
        strategy += "Focus your energy on Target choices — they're genuinely achievable. "
    elif t == 0:
        strategy += "Apply to 2–3 dream choices in Round 1, then secure safe options. "
    else:
        strategy += f"Prioritise your top {min(3, t)} Target choices while locking in 2 Safe options. "

    return strategy + persona.institute_advice
