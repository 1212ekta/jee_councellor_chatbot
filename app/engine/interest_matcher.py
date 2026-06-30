"""
Interest Matcher — Phase 1, Step 1

Maps a student's interest vector to branch profiles using Cosine Similarity.

Design:
  - Each branch has an 8-dimensional interest vector (from branch_profiles.json)
  - Each student has an 8-dimensional interest vector (from StudentProfile)
  - Cosine similarity measures the angular alignment between them (0 to 1)
  - A score of 1.0 = perfect direction match, 0.0 = completely orthogonal interests

Why cosine similarity?
  - Magnitude-invariant: a student who rates [0.9, 0.8, ...] vs [0.5, 0.4, ...]
    with the same relative pattern gets the same score.
  - We care about WHAT they prefer relative to each other, not how strongly
    they filled the sliders.
  - Works naturally in 0-1 bounded space.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.models.request import StudentProfile
from app.utils.logger import get_logger

log = get_logger(__name__)

# Interest vector dimension order — must match StudentProfile.interest_vector
INTEREST_DIMS = [
    "coding",
    "ai_ml",
    "research",
    "core_engineering",
    "electronics",
    "mechanical",
    "civil",
    "chemical",
]

# Branch domain → canonical interest vector key mapping
# Handles the 241 real branch names from the dataset
DOMAIN_KEYWORDS: dict[str, str] = {
    # CS / AI / Data
    "computer science":             "Computer Science and Engineering",
    "cse":                          "Computer Science and Engineering",
    "artificial intelligence":      "Computer Science and Engineering",
    "data science":                 "Computer Science and Engineering",
    "data engineering":             "Computer Science and Engineering",
    "machine learning":             "Computer Science and Engineering",
    "information technology":       "Computer Science and Engineering",
    "cyber security":               "Computer Science and Engineering",
    "computer engineering":         "Computer Science and Engineering",
    "computational":                "Mathematics and Computing",
    "mathematics and computing":    "Mathematics and Computing",
    "mathematics & computing":      "Mathematics and Computing",
    "mnc":                          "Mathematics and Computing",
    "math":                         "Mathematics and Computing",
    "statistics":                   "Mathematics and Computing",
    "quantitative":                 "Mathematics and Computing",
    # EE / ECE
    "electrical engineering":       "Electrical Engineering",
    "power":                        "Electrical Engineering",
    "electronics and communication":"Electronics and Communication Engineering",
    "ece":                          "Electronics and Communication Engineering",
    "electronics and electrical":   "Electronics and Communication Engineering",
    "vlsi":                         "Electronics and Communication Engineering",
    "microelectronics":             "Electronics and Communication Engineering",
    "instrumentation":              "Electronics and Communication Engineering",
    "electronic engineering":       "Electronics and Communication Engineering",
    "telecom":                      "Electronics and Communication Engineering",
    "avionics":                     "Electronics and Communication Engineering",
    # Mechanical / Aerospace
    "mechanical":                   "Mechanical Engineering",
    "aerospace":                    "Mechanical Engineering",
    "aeronautical":                 "Mechanical Engineering",
    "production":                   "Mechanical Engineering",
    "manufacturing":                "Mechanical Engineering",
    "industrial engineering":       "Mechanical Engineering",
    "mechatronics":                 "Mechanical Engineering",
    "robotics":                     "Mechanical Engineering",
    "naval":                        "Mechanical Engineering",
    "ocean engineering":            "Mechanical Engineering",
    # Civil
    "civil":                        "Civil Engineering",
    "structural":                   "Civil Engineering",
    "environmental engineering":    "Civil Engineering",
    "planning":                     "Civil Engineering",
    "geotechnical":                 "Civil Engineering",
    "transportation":               "Civil Engineering",
    "infrastructure":               "Civil Engineering",
    # Chemical / Bio
    "chemical engineering":         "Chemical Engineering",
    "biochemical":                  "Chemical Engineering",
    "petroleum":                    "Chemical Engineering",
    "pharmaceutical":               "Chemical Engineering",
    "food":                         "Chemical Engineering",
    "textile":                      "Chemical Engineering",
    "polymer":                      "Chemical Engineering",
    "biotechnology":                "Chemical Engineering",
    "biomedical":                   "Chemical Engineering",
    "biological":                   "Chemical Engineering",
    "bioengineering":               "Chemical Engineering",
    # Physics / Research
    "engineering physics":          "Engineering Physics",
    "physics":                      "Engineering Physics",
    "applied geophysics":           "Engineering Physics",
    "exploration geophysics":       "Engineering Physics",
    "space science":                "Engineering Physics",
    "energy engineering":           "Engineering Physics",
    # Mining / Metallurgy / Materials
    "metallurg":                    "Mechanical Engineering",
    "materials":                    "Mechanical Engineering",
    "mining":                       "Civil Engineering",
    "ceramic":                      "Chemical Engineering",
    "mineral":                      "Civil Engineering",
    # Design / Architecture
    "design":                       "Computer Science and Engineering",
    "architecture":                 "Civil Engineering",
}


@lru_cache(maxsize=1)
def _load_branch_profiles() -> dict:
    """Load branch profiles from JSON. Cached after first call."""
    settings = get_settings()
    path = settings.knowledge_dir / "branch_profiles.json"
    if not path.exists():
        log.warning(f"branch_profiles.json not found at {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    log.info(f"Loaded {len(data)} branch profiles")
    return data


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two equal-length vectors.
    Returns 0.0 if either vector is all zeros.
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _resolve_branch_profile(branch_name: str, profiles: dict) -> dict | None:
    """
    Find the matching branch profile for a given raw branch name.
    Strategy:
      1. Exact match on profile name
      2. Keyword match via DOMAIN_KEYWORDS
      3. Return None (caller handles gracefully)
    """
    # 1. Exact match
    if branch_name in profiles:
        return profiles[branch_name]

    # 2. Case-insensitive exact match
    branch_lower = branch_name.lower()
    for profile_name, profile in profiles.items():
        if profile_name.lower() == branch_lower:
            return profile

    # 3. Keyword match — find the best matching domain keyword
    best_profile_key = None
    best_match_len = 0
    for keyword, profile_key in DOMAIN_KEYWORDS.items():
        if keyword in branch_lower and len(keyword) > best_match_len:
            best_profile_key = profile_key
            best_match_len = len(keyword)

    if best_profile_key and best_profile_key in profiles:
        return profiles[best_profile_key]

    return None


def compute_interest_match(student: StudentProfile, branch_name: str) -> dict:
    """
    Compute interest match between a student and a branch.

    Returns:
        {
            "score": float (0-1),
            "matched_profile": str,
            "dimension_scores": dict,
            "explanation": str
        }
    """
    profiles = _load_branch_profiles()
    profile = _resolve_branch_profile(branch_name, profiles)

    if profile is None:
        # Unknown branch — return neutral score with explanation
        log.debug(f"No interest profile found for branch: '{branch_name}'")
        return {
            "score": 0.5,
            "matched_profile": "Unknown",
            "dimension_scores": {},
            "explanation": f"No interest profile available for '{branch_name}'. Neutral score applied.",
        }

    # Build branch interest vector in the same dimension order as student
    branch_vec = profile.get("interest_vector", {})
    branch_vector = [branch_vec.get(dim, 0.5) for dim in INTEREST_DIMS]
    student_vector = student.interest_vector  # already ordered

    score = _cosine_similarity(student_vector, branch_vector)

    # Per-dimension scores for explainability
    dimension_scores = {
        dim: {
            "student": round(student_vector[i], 2),
            "branch_needs": round(branch_vector[i], 2),
            "match": round(1.0 - abs(student_vector[i] - branch_vector[i]), 2),
        }
        for i, dim in enumerate(INTEREST_DIMS)
    }

    # Find dims where BOTH student AND branch score high (true alignment)
    high_align = [
        (dim, min(student_vector[i], branch_vector[i]))
        for i, dim in enumerate(INTEREST_DIMS)
        if student_vector[i] > 0.5 and branch_vector[i] > 0.5
    ]
    best_dims = [d for d, _ in sorted(high_align, key=lambda x: x[1], reverse=True)[:2]]
    if not best_dims:
        dim_matches = [(dim, abs(student_vector[i] - branch_vector[i])) for i, dim in enumerate(INTEREST_DIMS)]
        best_dims = [d for d, _ in sorted(dim_matches, key=lambda x: x[1])[:2]]
    # Worst: branch needs HIGH but student is LOW
    worst_dims = [
        dim for i, dim in enumerate(INTEREST_DIMS)
        if branch_vector[i] > 0.6 and student_vector[i] < 0.4
    ]

    explanation = _build_explanation(branch_name, score, best_dims, worst_dims, profile)

    return {
        "score": round(score, 4),
        "matched_profile": profile.get("short", branch_name),
        "dimension_scores": dimension_scores,
        "explanation": explanation,
        "career_paths": profile.get("career_paths", []),
        "coding_intensity": profile.get("coding_intensity", 3),
        "research_scope": profile.get("research_scope", 3),
        "median_lpa": profile.get("median_lpa", 10.0),
    }


def _build_explanation(
    branch: str,
    score: float,
    best_dims: list[str],
    worst_dims: list[str],
    profile: dict,
) -> str:
    """Build a human-readable explanation of the interest match."""
    dim_labels = {
        "coding": "coding",
        "ai_ml": "AI/ML",
        "research": "research",
        "core_engineering": "core engineering",
        "electronics": "electronics",
        "mechanical": "mechanical engineering",
        "civil": "civil engineering",
        "chemical": "chemical engineering",
    }

    if score >= 0.85:
        quality = "Excellent"
    elif score >= 0.70:
        quality = "Strong"
    elif score >= 0.55:
        quality = "Moderate"
    elif score >= 0.40:
        quality = "Partial"
    else:
        quality = "Weak"

    best_labels = " and ".join(dim_labels.get(d, d) for d in best_dims[:2])
    explanation = f"{quality} interest match ({score:.0%}). Your interest in {best_labels} aligns well with {branch}."

    if worst_dims:
        worst_labels = ", ".join(dim_labels.get(d, d) for d in worst_dims)
        explanation += f" Note: this branch requires {worst_labels} which is less aligned with your stated interests."

    return explanation


def rank_branches_by_interest(student: StudentProfile, branch_names: list[str]) -> list[dict]:
    """
    Given a list of branch names, rank them by interest match for a student.
    Returns list sorted by score descending, each item includes the branch name.
    """
    results = []
    for branch in branch_names:
        match = compute_interest_match(student, branch)
        match["branch"] = branch
        results.append(match)

    return sorted(results, key=lambda x: x["score"], reverse=True)


def get_top_branches_for_student(student: StudentProfile, top_n: int = 5) -> list[dict]:
    """
    Returns the top N branches from the knowledge base that best match
    the student's interest profile.
    """
    profiles = _load_branch_profiles()
    all_branches = list(profiles.keys())
    ranked = rank_branches_by_interest(student, all_branches)
    return ranked[:top_n]
