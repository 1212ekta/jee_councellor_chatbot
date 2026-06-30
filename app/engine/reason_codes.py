"""
Recommendation Reason Codes

Machine-readable codes attached to every recommendation.
Frontend uses these for badges; LLM uses them as context.
"""

from dataclasses import dataclass, field
from app.engine.scorer import ScoredRecommendation
from app.engine.compatibility import CompatibilityProfile
from app.models.request import StudentProfile


@dataclass
class ReasonCodes:
    """Machine-readable reason codes with human labels."""
    codes:   list[str] = field(default_factory=list)   # e.g. ["RANK_MATCH", "HOME_STATE"]
    labels:  list[str] = field(default_factory=list)   # e.g. ["✓ Strong Rank Match"]
    warnings:list[str] = field(default_factory=list)   # e.g. ["⚠ Stretch Rank"]

    def to_dict(self) -> dict:
        return {"codes": self.codes, "labels": self.labels, "warnings": self.warnings}


# Code → human label map
CODE_LABELS = {
    "RANK_MATCH":        "✓ Strong Rank Match",
    "HOME_STATE":        "🏠 Home State Advantage",
    "HIGH_PLACEMENT":    "✓ Excellent Placements",
    "CAREER_MATCH":      "✓ Career Goal Aligned",
    "RESEARCH":          "✓ Strong Research Fit",
    "STARTUP":           "✓ Great Startup Ecosystem",
    "CODING_CULTURE":    "✓ Strong Coding Culture",
    "BRAND_VALUE":       "✓ Top Institute Brand",
    "FLEXIBILITY":       "✓ High Career Flexibility",
    "INTEREST_MATCH":    "✓ Excellent Interest Match",
    "SAFE_CHOICE":       "✓ Very High Admission Confidence",
    "GATE_PATH":         "✓ Strong GATE / PSU Pathway",
    "MBA_FRIENDLY":      "✓ MBA-Friendly Branch",
    "ABROAD_STRONG":     "✓ Strong Higher Studies Profile",
}

WARNING_CODES = {
    "STRETCH_RANK":      "⚠ Rank Beyond Cutoff",
    "INTEREST_MISMATCH": "⚠ Low Interest Match",
    "WEAK_PLACEMENTS":   "⚠ Below-Average Placements",
    "LIMITED_RESEARCH":  "⚠ Limited Research Output",
    "LOW_FLEXIBILITY":   "⚠ Limited Career Flexibility",
    "NICHE_BRANCH":      "⚠ Niche Branch — Few Recruiters",
}


def compute_reason_codes(
    student: StudentProfile,
    rec: ScoredRecommendation,
    comp: CompatibilityProfile,
) -> ReasonCodes:
    """Compute all applicable reason codes for a recommendation."""
    codes, labels, warnings = [], [], []

    def add(code: str) -> None:
        codes.append(code)
        labels.append(CODE_LABELS[code])

    def warn(code: str) -> None:
        warnings.append(WARNING_CODES[code])

    # ── Positive codes ────────────────────────────────────────────────────────
    if rec.risk.admission_probability >= 0.80:
        add("RANK_MATCH")
    if rec.risk.admission_probability >= 0.90:
        add("SAFE_CHOICE")
    if rec.home_state_advantage:
        add("HOME_STATE")
    if comp.placement_strength >= 0.75:
        add("HIGH_PLACEMENT")
    if comp.career_match >= 0.75:
        add("CAREER_MATCH")
    if comp.research_match >= 0.75 and student.interest_research >= 0.6:
        add("RESEARCH")
    if student.wants_startup and (rec.coding_culture_score or 0) >= 4:
        add("STARTUP")
    if comp.coding_match >= 0.78 and student.interest_coding >= 0.6:
        add("CODING_CULTURE")
    if comp.institute_reputation >= 0.80:
        add("BRAND_VALUE")
    if rec.scores.flexibility >= 0.80:
        add("FLEXIBILITY")
    if rec.scores.interest_match >= 0.80:
        add("INTEREST_MATCH")
    if student.wants_mba and rec.scores.career_alignment >= 0.70:
        add("MBA_FRIENDLY")
    if student.wants_higher_studies_abroad and (rec.research_score or 0) >= 4:
        add("ABROAD_STRONG")
    if student.wants_govt_job and rec.scores.career_alignment >= 0.70:
        add("GATE_PATH")

    # ── Warning codes ─────────────────────────────────────────────────────────
    if rec.risk.risk_level == "Dream":
        warn("STRETCH_RANK")
    if rec.scores.interest_match < 0.50:
        warn("INTEREST_MISMATCH")
    if comp.placement_strength < 0.40:
        warn("WEAK_PLACEMENTS")
    if comp.research_match < 0.40 and student.interest_research >= 0.6:
        warn("LIMITED_RESEARCH")
    if rec.scores.flexibility < 0.40:
        warn("LOW_FLEXIBILITY")
    if comp.placement_strength < 0.35 and comp.institute_reputation < 0.50:
        warn("NICHE_BRANCH")

    return ReasonCodes(codes=codes, labels=labels, warnings=warnings)
