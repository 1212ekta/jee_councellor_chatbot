"""
Branch Compatibility Score — Phase 3

Computes an 8-dimension compatibility profile for every recommendation.
Each dimension is independently scored 0.0–1.0 and returned as a breakdown.

Dimensions:
  1. admission_probability  — Can the student realistically get this seat?
  2. career_match           — Does this branch serve their career goals?
  3. research_match         — Does the institute/branch support research ambitions?
  4. coding_match           — Coding culture alignment (for CS-oriented students)
  5. lifestyle_match        — Work-life balance, campus culture, location preference
  6. future_growth          — Long-term optionality and salary trajectory
  7. placement_strength     — Quality and consistency of placements
  8. institute_reputation   — Brand value and global recognition

These 8 scores are:
  - Returned in every recommendation response
  - Used to generate confidence badges (Phase 9)
  - Visualised as a radar/spider chart in the frontend
  - Fed into the LLM explainer as context (Phase 4)
"""

from dataclasses import dataclass, field

from app.engine.risk_classifier import RiskResult
from app.models.request import StudentProfile
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CompatibilityProfile:
    """8-dimension compatibility breakdown for one recommendation."""

    # Core dimensions (0.0 – 1.0)
    admission_probability: float = 0.0
    career_match:          float = 0.0
    research_match:        float = 0.0
    coding_match:          float = 0.0
    lifestyle_match:       float = 0.0
    future_growth:         float = 0.0
    placement_strength:    float = 0.0
    institute_reputation:  float = 0.0

    # Derived
    overall_compatibility: float = 0.0

    # Confidence badges (auto-generated)
    badges:  list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "admission_probability": self.admission_probability,
            "career_match":          self.career_match,
            "research_match":        self.research_match,
            "coding_match":          self.coding_match,
            "lifestyle_match":       self.lifestyle_match,
            "future_growth":         self.future_growth,
            "placement_strength":    self.placement_strength,
            "institute_reputation":  self.institute_reputation,
            "overall_compatibility": self.overall_compatibility,
            "badges":                self.badges,
            "caveats":               self.caveats,
        }


# ── Dimension scorers ─────────────────────────────────────────────────────────

def _dim_admission(risk: RiskResult) -> float:
    """Direct from admission probability."""
    return round(risk.admission_probability, 4)


def _dim_career_match(
    student: StudentProfile,
    suits_goals: list[str],
    branch_domain: str,
    median_lpa: float,
    coding_intensity: int,
    research_scope: int,
) -> float:
    """
    How well does this branch serve the student's stated career goals?
    Checks each active goal against branch characteristics.
    """
    if not student.active_goals:
        return 0.65  # neutral baseline

    score = 0.0
    total_weight = 0.0

    goal_checks = [
        # (goal_flag, weight, score_fn)
        (student.wants_startup,              0.20, coding_intensity / 5.0),
        (student.wants_research,             0.20, research_scope / 5.0),
        (student.wants_higher_studies_abroad,0.15, research_scope / 5.0),
        (student.wants_mba,                  0.15, 1.0 if "wants_mba" in suits_goals else 0.35),
        (student.wants_govt_job,             0.15, 1.0 if "govt_job" in suits_goals else 0.20),
        (student.salary_priority > 0.6,      0.15, min(1.0, median_lpa / 35.0)),
    ]

    for flag, weight, goal_score in goal_checks:
        if flag:
            score += weight * goal_score
            total_weight += weight

    if total_weight == 0:
        return 0.65

    raw = score / total_weight
    # Add baseline so even mismatches aren't 0
    return round(min(1.0, raw * 0.7 + 0.30), 4)


def _dim_research_match(
    student: StudentProfile,
    research_scope: int,         # 1–5 from branch profile
    inst_research_score: int,    # 1–5 from institute metadata
) -> float:
    """
    Research compatibility — combines student research interest,
    branch research depth, and institute research output.
    """
    student_interest = (student.interest_research + (0.3 if student.wants_research else 0.0))
    student_interest = min(1.0, student_interest)

    branch_research  = research_scope / 5.0
    inst_research    = (inst_research_score or 2) / 5.0

    # If student doesn't care about research, give neutral score
    if student_interest < 0.3:
        return 0.55

    score = (
        0.40 * student_interest
        + 0.35 * branch_research
        + 0.25 * inst_research
    )
    return round(min(1.0, score), 4)


def _dim_coding_match(
    student: StudentProfile,
    coding_intensity: int,       # 1–5 from branch profile
    inst_coding_score: int,      # 1–5 from institute metadata
) -> float:
    """
    Coding culture alignment.
    For high-coding students: high intensity branch + institute = great match.
    For low-coding students: inverse — they'd prefer branches without heavy coding.
    """
    branch_coding = coding_intensity / 5.0
    inst_coding   = (inst_coding_score or 2) / 5.0
    student_pref  = (student.interest_coding + student.interest_ai_ml) / 2.0

    # Alignment = 1 - |student_pref - branch_coding|
    branch_alignment = 1.0 - abs(student_pref - branch_coding)
    inst_alignment   = 1.0 - abs(student_pref - inst_coding)

    score = 0.60 * branch_alignment + 0.40 * inst_alignment
    return round(min(1.0, score), 4)


def _dim_lifestyle_match(
    student: StudentProfile,
    inst_state: str,
    inst_city: str,
    inst_tier: int,
    branch_domain: str,
    coding_intensity: int,
) -> float:
    """
    Lifestyle compatibility.
    Considers:
    - Location preference (home state vs far away)
    - Work-life balance (inverse of coding intensity for work-life students)
    - Campus life (tier 1 IITs have richer campus culture)
    """
    score = 0.0

    # Location alignment
    student_state = student.home_state.strip().lower()
    inst_state_lower = (inst_state or "").lower()
    location_score = 0.5  # default: fine with any location

    if student.location_flexibility < 0.3:
        # Prefers nearby
        location_score = 0.9 if student_state in inst_state_lower else 0.3
    elif student.location_flexibility > 0.7:
        # Flexible — location doesn't matter much
        location_score = 0.75
    else:
        location_score = 0.65 if student_state in inst_state_lower else 0.55

    # Work-life balance vs coding intensity
    # Low salary priority → prefers balanced life
    wlb_pref = 1.0 - student.salary_priority
    wlb_score = 1.0 - abs(wlb_pref - (1.0 - coding_intensity / 5.0))

    # Campus life score (based on tier and type)
    campus_score = max(0.3, 1.0 - (inst_tier - 1) * 0.15)

    score = (
        0.40 * location_score
        + 0.35 * wlb_score
        + 0.25 * campus_score
    )
    return round(min(1.0, score), 4)


def _dim_future_growth(
    branch_domain: str,
    median_lpa: float,
    avg_salary_lpa: float,
    flexibility_score: float,
    research_scope: int,
) -> float:
    """
    Long-term growth potential.
    Combines: salary trajectory, flexibility, and research optionality.
    """
    # Salary growth (normalised: 40 LPA median = 1.0)
    salary_norm = min(1.0, avg_salary_lpa / 40.0)

    # Flexibility (already 0–1 from scorer)
    flex = min(1.0, flexibility_score)

    # Research optionality (PhD pathway open?)
    research_opt = research_scope / 5.0

    score = (
        0.40 * salary_norm
        + 0.35 * flex
        + 0.25 * research_opt
    )
    return round(min(1.0, score), 4)


def _dim_placement_strength(
    inst_placement_median: float,  # LPA
    inst_type: str,
    branch_domain: str,
    inst_tier: int,
) -> float:
    """
    Placement quality score.
    Combines institute placement median, type, and branch-specific placement history.
    """
    # Institute base
    placement_norm = min(1.0, (inst_placement_median or 8.0) / 35.0)

    # Type premium
    type_premium = {
        "IIT": 0.20, "IIIT": 0.10, "NIT": 0.05, "GFTI": 0.00, "State": -0.05
    }.get(inst_type, 0.0)

    # Branch placement modifier
    high_placement_domains = {"CS", "MnC"}
    mid_placement_domains  = {"EE", "ECE", "EP"}
    low_placement_domains  = {"CIVIL", "MECH", "CHEM"}

    branch_mod = 0.0
    if branch_domain in high_placement_domains:
        branch_mod = 0.10
    elif branch_domain in mid_placement_domains:
        branch_mod = 0.05
    elif branch_domain in low_placement_domains:
        branch_mod = -0.03

    score = placement_norm + type_premium + branch_mod
    return round(min(1.0, max(0.0, score)), 4)


def _dim_institute_reputation(
    inst_type: str,
    nirf_rank: int | None,
    inst_tier: int,
) -> float:
    """
    Global and national reputation of the institute.
    """
    # Base from type
    type_base = {
        "IIT": 0.85, "IIIT": 0.65, "NIT": 0.60, "GFTI": 0.40, "State": 0.35
    }.get(inst_type, 0.40)

    # NIRF adjustment
    nirf_adj = 0.0
    if nirf_rank:
        if nirf_rank <= 3:
            nirf_adj = 0.15
        elif nirf_rank <= 10:
            nirf_adj = 0.10
        elif nirf_rank <= 25:
            nirf_adj = 0.05
        elif nirf_rank > 75:
            nirf_adj = -0.05

    score = type_base + nirf_adj
    return round(min(1.0, max(0.0, score)), 4)


# ── Badge generator ───────────────────────────────────────────────────────────

BADGE_THRESHOLDS = {
    "admission_probability": (0.80, "✓ Strong Rank Match",      0.40, "⚠ Stretch Rank"),
    "career_match":          (0.75, "✓ Career Goal Aligned",    0.45, "⚠ Career Mismatch"),
    "research_match":        (0.75, "✓ Strong Research Fit",    0.40, "⚠ Limited Research"),
    "coding_match":          (0.78, "✓ Great Coding Culture",   0.40, "⚠ Coding Mismatch"),
    "lifestyle_match":       (0.70, "✓ Good Lifestyle Fit",     0.40, "⚠ Lifestyle Concerns"),
    "future_growth":         (0.72, "✓ High Growth Potential",  0.45, "⚠ Limited Growth"),
    "placement_strength":    (0.75, "✓ Excellent Placements",   0.40, "⚠ Weak Placements"),
    "institute_reputation":  (0.80, "✓ Top Institute Brand",    0.50, "⚠ Limited Brand Value"),
}

HOME_STATE_BADGE = "🏠 Home State Advantage"


def _generate_badges(
    profile: CompatibilityProfile,
    home_state_advantage: bool,
) -> tuple[list[str], list[str]]:
    """Generate positive badges and warning caveats."""
    badges, caveats = [], []

    dim_values = {
        "admission_probability": profile.admission_probability,
        "career_match":          profile.career_match,
        "research_match":        profile.research_match,
        "coding_match":          profile.coding_match,
        "lifestyle_match":       profile.lifestyle_match,
        "future_growth":         profile.future_growth,
        "placement_strength":    profile.placement_strength,
        "institute_reputation":  profile.institute_reputation,
    }

    for dim, value in dim_values.items():
        high_thresh, high_badge, low_thresh, low_badge = BADGE_THRESHOLDS[dim]
        if value >= high_thresh:
            badges.append(high_badge)
        elif value <= low_thresh:
            caveats.append(low_badge)

    if home_state_advantage:
        badges.insert(0, HOME_STATE_BADGE)

    return badges, caveats


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_compatibility(
    student:             StudentProfile,
    risk:                RiskResult,
    branch_name:         str,
    branch_domain:       str,
    suits_goals:         list[str],
    median_lpa:          float,
    avg_salary_lpa:      float,
    coding_intensity:    int,
    research_scope:      int,
    inst_type:           str,
    inst_tier:           int,
    inst_city:           str,
    inst_state:          str,
    nirf_rank:           int | None,
    inst_research_score: int,
    inst_coding_score:   int,
    inst_placement_lpa:  float,
    home_state_advantage:bool,
    flexibility_score:   float = 0.55,
) -> CompatibilityProfile:
    """
    Compute the full 8-dimension compatibility profile.

    All inputs come from the ScoredRecommendation + branch knowledge base.
    This function is pure (no DB calls) — all data must be passed in.
    """
    # ── Compute each dimension ────────────────────────────────────────────────
    adm  = _dim_admission(risk)
    car  = _dim_career_match(student, suits_goals, branch_domain, median_lpa, coding_intensity, research_scope)
    res  = _dim_research_match(student, research_scope, inst_research_score)
    cod  = _dim_coding_match(student, coding_intensity, inst_coding_score)
    life = _dim_lifestyle_match(student, inst_state, inst_city, inst_tier, branch_domain, coding_intensity)
    grow = _dim_future_growth(branch_domain, median_lpa, avg_salary_lpa, flexibility_score, research_scope)
    plac = _dim_placement_strength(inst_placement_lpa, inst_type, branch_domain, inst_tier)
    rep  = _dim_institute_reputation(inst_type, nirf_rank, inst_tier)

    # ── Overall compatibility (equal-weight average of all 8) ─────────────────
    dims = [adm, car, res, cod, life, grow, plac, rep]
    overall = round(sum(dims) / len(dims), 4)

    profile = CompatibilityProfile(
        admission_probability=adm,
        career_match=car,
        research_match=res,
        coding_match=cod,
        lifestyle_match=life,
        future_growth=grow,
        placement_strength=plac,
        institute_reputation=rep,
        overall_compatibility=overall,
    )

    # ── Generate badges and caveats ───────────────────────────────────────────
    badges, caveats = _generate_badges(profile, home_state_advantage)
    profile.badges  = badges
    profile.caveats = caveats

    log.debug(
        f"Compatibility [{branch_name[:30]}]: "
        f"overall={overall:.2f} "
        f"badges={len(badges)} caveats={len(caveats)}"
    )

    return profile
