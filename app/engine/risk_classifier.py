"""
Risk Classifier — Phase 1, Step 2

Converts rank + cutoff data into admission probability and risk category.

Design: Sigmoid-based probability model
  - NOT binary (rank <= closing → yes/no)
  - Smooth S-curve centered on closing rank
  - Accounts for year-to-year cutoff variance (~8-12%)
  - Accounts for round number (Round 1 is conservative, Round 6 is final)

Risk Buckets:
  Dream     → 15–45% probability  (stretch goal, apply in Round 1)
  Target    → 45–75% probability  (realistic aim, main focus)
  Safe      → 75–90% probability  (strong backup)
  Very Safe → 90%+  probability  (near certain)
  Filtered  → <15%  probability  (don't recommend)
"""

import math
from dataclasses import dataclass

from app.services.knowledge_loader import KnowledgeLoader
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
DREAM_MIN     = 0.15
DREAM_MAX     = 0.45
TARGET_MIN    = 0.45
TARGET_MAX    = 0.75
SAFE_MIN      = 0.75
SAFE_MAX      = 0.90
VERY_SAFE_MIN = 0.90

# Sigmoid steepness — controls how sharp the probability curve is around closing rank
# Higher = steeper cliff, Lower = more gradual
SIGMOID_STEEPNESS = 8.0

# Historical cutoff variance — cutoffs shift ±10% across years
CUTOFF_VARIANCE_FACTOR = 0.10


@dataclass
class RiskResult:
    """Full risk assessment for one student-cutoff combination."""
    admission_probability: float     # 0.0 to 1.0
    risk_level: str                  # Dream / Target / Safe / Very Safe
    rank_gap: int                    # student_rank - closing_rank (negative = buffer)
    rank_gap_pct: float              # gap as % of closing rank
    is_eligible: bool                # False = filtered out (prob < 15%)
    safety_margin: str               # Human-readable margin description
    probability_label: str           # "~30% chance" etc.
    counselor_note: str              # One-line honest note


def _sigmoid(x: float, steepness: float = SIGMOID_STEEPNESS) -> float:
    """
    Sigmoid function: maps any real number to (0, 1).
    sigmoid(0) = 0.5 → at exactly closing rank = 50% chance
    sigmoid(+inf) → 0  (rank worse than closing → low chance)
    sigmoid(-inf) → 1  (rank much better than closing → high chance)
    """
    try:
        return 1.0 / (1.0 + math.exp(steepness * x))
    except OverflowError:
        return 0.0 if x > 0 else 1.0


def compute_admission_probability(
    student_rank: int,
    opening_rank: int,
    closing_rank: int,
    round_number: int = 6,
    gender_match: bool = True,
) -> float:
    """
    Compute probability of admission given ranks.

    Core formula:
      gap_normalized = (student_rank - closing_rank) / closing_rank
      probability = sigmoid(gap_normalized * steepness)

    Examples (closing_rank = 1000):
      student_rank = 500  → gap = -0.50 → prob ≈ 0.98  (Very Safe)
      student_rank = 900  → gap = -0.10 → prob ≈ 0.69  (Target)
      student_rank = 1000 → gap =  0.00 → prob ≈ 0.50  (risky Target)
      student_rank = 1100 → gap = +0.10 → prob ≈ 0.31  (Dream)
      student_rank = 1300 → gap = +0.30 → prob ≈ 0.09  (filtered)

    Args:
        student_rank:  Student's CRL rank (lower = better)
        opening_rank:  Historical opening rank for this seat
        closing_rank:  Historical closing rank (main reference)
        round_number:  JoSAA round (1-6). Round 6 = final, most reliable
        gender_match:  False if gender pool mismatches (slight penalty)
    """
    if closing_rank <= 0:
        return 0.0

    # Normalised gap: negative = student is better than cutoff (good)
    gap_normalized = (student_rank - closing_rank) / closing_rank

    # Base probability from sigmoid
    prob = _sigmoid(gap_normalized)

    # ── Adjustments ───────────────────────────────────────────────────────────

    # 1. Round adjustment: Round 1 closing ranks are conservative.
    #    By Round 6, seats expand. Since our data is final allotment,
    #    we slightly boost probability (more seats filled = more realistic).
    #    For Round 1 use (future feature), we'd reduce by 0.05.
    round_boost = 0.0  # data is final round, no adjustment needed

    # 2. Variance buffer: cutoffs shift ±10% year to year.
    #    If student is within the variance buffer zone, apply a small
    #    correction to avoid false confidence.
    variance_band = closing_rank * CUTOFF_VARIANCE_FACTOR
    if abs(student_rank - closing_rank) < variance_band:
        # In the uncertainty zone — reduce confidence slightly
        uncertainty_penalty = 0.05 * (1 - abs(gap_normalized) / CUTOFF_VARIANCE_FACTOR)
        prob = max(0.10, prob - uncertainty_penalty)

    # 3. Gender pool mismatch penalty (minor)
    if not gender_match:
        prob *= 0.85

    prob = round(min(0.99, max(0.01, prob + round_boost)), 4)
    return prob


def classify_risk(probability: float) -> str:
    """Map admission probability to risk category."""
    t = KnowledgeLoader().risk_thresholds()
    
    if probability >= t.get("very_safe_min", 0.90):
        return "Very Safe"
    elif probability >= t.get("safe_min", 0.75):
        return "Safe"
    elif probability >= t.get("target_min", 0.45):
        return "Target"
    elif probability >= t.get("dream_min", 0.15):
        return "Dream"
    else:
        return "Filtered"


def _safety_margin_label(rank_gap: int, closing_rank: int) -> str:
    """Human readable margin: 'You are 500 ranks ahead of cutoff'."""
    # C4-FIX: Guard against closing_rank == 0 to prevent ZeroDivisionError
    if closing_rank <= 0:
        return "Cutoff rank unavailable"
    if rank_gap < 0:
        pct = abs(rank_gap) / closing_rank * 100
        return f"You are {abs(rank_gap)} ranks ({pct:.0f}%) ahead of the cutoff"
    elif rank_gap == 0:
        return "Your rank is exactly at the closing rank — borderline"
    else:
        pct = rank_gap / closing_rank * 100
        return f"Your rank is {rank_gap} ranks ({pct:.0f}%) beyond the cutoff"


def _probability_label(prob: float) -> str:
    """Convert probability to a readable label."""
    pct = int(prob * 100)
    if pct >= 95:
        return f"~{pct}% chance (near certain)"
    elif pct >= 80:
        return f"~{pct}% chance (strong)"
    elif pct >= 60:
        return f"~{pct}% chance (good)"
    elif pct >= 40:
        return f"~{pct}% chance (moderate)"
    elif pct >= 20:
        return f"~{pct}% chance (low — stretch goal)"
    else:
        return f"~{pct}% chance (very unlikely)"


def _counselor_note(risk_level: str, rank_gap: int, prob: float) -> str:
    """One-line honest counselor note for this risk level."""
    if risk_level == "Very Safe":
        return "This is a strong safety option. Lock it in and use better choices for ambition."
    elif risk_level == "Safe":
        return "Solid choice with high confidence. Good to include as a reliable backup."
    elif risk_level == "Target":
        if prob > 0.60:
            return "Realistic target. Apply with full seriousness — good chance of getting in."
        else:
            return "Possible but competitive. Have backups ready. Apply in early rounds."
    elif risk_level == "Dream":
        return (
            "Stretch goal — rank is beyond the cutoff but worth applying in Round 1. "
            "Cutoffs can vary year to year. Don't count on it."
        )
    else:
        return "Rank is significantly beyond cutoff. Not recommended."


def assess_risk(
    student_rank: int,
    opening_rank: int,
    closing_rank: int,
    student_gender: str,
    seat_gender: str,
    round_number: int = 6,
) -> RiskResult:
    """
    Full risk assessment for one student × cutoff combination.

    Args:
        student_rank:   Student's effective rank
        opening_rank:   Seat opening rank
        closing_rank:   Seat closing rank
        student_gender: 'male' / 'female' / 'other'
        seat_gender:    'Gender-Neutral' / 'Female-Only'
        round_number:   Allotment round

    Returns:
        RiskResult with all fields populated
    """
    # Gender eligibility check
    # Female-Only seats: male students cannot apply
    is_female_seat = seat_gender == "Female-Only"
    is_female_student = student_gender == "female"

    if is_female_seat and not is_female_student:
        return RiskResult(
            admission_probability=0.0,
            risk_level="Filtered",
            rank_gap=student_rank - closing_rank,
            rank_gap_pct=0.0,
            is_eligible=False,
            safety_margin="Female-Only seat — not applicable for your gender",
            probability_label="0% chance",
            counselor_note="This seat is reserved for female students only.",
        )

    gender_match = True  # gender-neutral seats are open to all

    prob = compute_admission_probability(
        student_rank, opening_rank, closing_rank, round_number, gender_match
    )

    risk_level = classify_risk(prob)
    is_eligible = risk_level != "Filtered"
    rank_gap = student_rank - closing_rank
    rank_gap_pct = round((rank_gap / closing_rank) * 100, 1) if closing_rank > 0 else 0.0

    return RiskResult(
        admission_probability=prob,
        risk_level=risk_level,
        rank_gap=rank_gap,
        rank_gap_pct=rank_gap_pct,
        is_eligible=is_eligible,
        safety_margin=_safety_margin_label(rank_gap, closing_rank),
        probability_label=_probability_label(prob),
        counselor_note=_counselor_note(risk_level, rank_gap, prob),
    )


def filter_and_bucket(
    cutoff_rows: list[dict],
    student_rank: int,
    student_gender: str,
) -> dict[str, list]:
    """
    Process a list of cutoff rows and return them bucketed by risk level.

    Args:
        cutoff_rows:    List of dicts with keys: opening_rank, closing_rank,
                        gender, round (all from DuckDB)
        student_rank:   Student's effective rank
        student_gender: 'male' / 'female' / 'other'

    Returns:
        {
            "dream": [...],
            "target": [...],
            "safe": [...],
            "very_safe": [...],
        }
        Each item is the original row dict enriched with a 'risk' key.
    """
    buckets: dict[str, list] = {
        "dream": [], "target": [], "safe": [], "very_safe": []
    }

    for row in cutoff_rows:
        # C3-FIX: Use .get() with safe fallback to avoid KeyError on malformed DB rows
        closing_rank_val = row.get("closing_rank", 999999)
        result = assess_risk(
            student_rank=student_rank,
            opening_rank=row.get("opening_rank", closing_rank_val),
            closing_rank=closing_rank_val,
            student_gender=student_gender,
            seat_gender=row.get("gender", "Gender-Neutral"),
            round_number=row.get("round", 6),
        )

        if not result.is_eligible:
            continue

        enriched = {**row, "risk": result}
        bucket_key = result.risk_level.lower().replace(" ", "_")
        if bucket_key in buckets:
            buckets[bucket_key].append(enriched)

    log.debug(
        f"Bucketed {len(cutoff_rows)} rows → "
        f"dream={len(buckets['dream'])} "
        f"target={len(buckets['target'])} "
        f"safe={len(buckets['safe'])} "
        f"very_safe={len(buckets['very_safe'])}"
    )
    return buckets
