from typing import Literal, Optional
from pydantic import BaseModel, Field


RiskLevel = Literal["Dream", "Target", "Safe", "Very Safe"]


class ScoreBreakdown(BaseModel):
    """Transparent score components — shows WHY a score was given."""
    overall: float = Field(..., description="Final weighted score (0-1)")
    rank_fit: float = Field(..., description="How well rank matches cutoff (0-1)")
    interest_match: float = Field(..., description="Branch ↔ student interest alignment (0-1)")
    institute_strength: float = Field(..., description="Institute tier + placement quality (0-1)")
    career_alignment: float = Field(..., description="Branch → student goals match (0-1)")
    flexibility: float = Field(..., description="Future optionality of this choice (0-1)")


class Recommendation(BaseModel):
    """A single institute+branch recommendation with full explainability."""

    # ── What is being recommended ─────────────────────────────────────────────
    institute: str
    branch: str
    program: str = Field(..., description="Full program name e.g. B.Tech CSE")
    institute_type: str = Field(..., description="IIT / NIT / IIIT / GFTI")
    city: str
    state: str

    # ── Cutoff data ───────────────────────────────────────────────────────────
    opening_rank: int
    closing_rank: int
    year: int
    round: int
    category: str
    exam_type: str
    state_quota: Optional[str] = Field(None, description="HS / OS / AI")
    home_state_advantage: bool = Field(
        False, description="True if student gets home state quota benefit"
    )

    # ── Scoring ───────────────────────────────────────────────────────────────
    scores: ScoreBreakdown
    admission_probability: float = Field(..., ge=0, le=1, description="Estimated probability 0-1")
    risk_level: RiskLevel

    # ── Explainability (the key differentiator) ───────────────────────────────
    why_this_institute: str = Field(..., description="Why this institute suits this student")
    why_this_branch: str = Field(..., description="Why this branch matches interests/goals")
    risks: list[str] = Field(default_factory=list, description="Honest risks to be aware of")
    opportunities: list[str] = Field(default_factory=list, description="Key upsides")
    alternatives: list[str] = Field(default_factory=list, description="Other options to consider")
    career_roadmap: list[str] = Field(default_factory=list, description="Year-by-year path")

    # ── Institute metadata ────────────────────────────────────────────────────
    nirf_rank: Optional[int] = None
    median_placement_lpa: Optional[float] = None
    research_score: Optional[int] = Field(None, ge=1, le=5)
    coding_culture_score: Optional[int] = Field(None, ge=1, le=5)
    known_for: Optional[str] = None


class RecommendationBuckets(BaseModel):
    """Recommendations grouped into Dream / Target / Safe / Very Safe."""
    dream: list[Recommendation] = Field(default_factory=list, description="20-50% probability, stretch goals")
    target: list[Recommendation] = Field(default_factory=list, description="50-75% probability, realistic aim")
    safe: list[Recommendation] = Field(default_factory=list, description="75-90% probability, solid choices")
    very_safe: list[Recommendation] = Field(default_factory=list, description="90%+ probability, guaranteed options")


class RecommendationResponse(BaseModel):
    """Top-level API response for POST /recommend."""
    session_id: str = Field(..., description="Unique ID — use to share or revisit this recommendation")
    student_summary: str = Field(..., description="One-paragraph counselor summary of the student's profile")
    counselor_insight: str = Field(..., description="Strategic advice tailored to this student")
    buckets: RecommendationBuckets
    total_matches: int = Field(..., description="Total cutoff rows that matched before scoring")
    generated_at: str = Field(..., description="ISO timestamp")

    # Share link (populated by API layer)
    share_url: Optional[str] = None


class ProfileInsight(BaseModel):
    """Response for POST /analyze-profile — profile analysis without recommendations."""
    rank_percentile: float = Field(..., description="Approximate percentile (0-100)")
    realistic_institute_types: list[str] = Field(
        ..., description="Types of institutes realistically achievable"
    )
    best_fit_branches: list[str] = Field(
        ..., description="Top 3 branches based on interest profile"
    )
    goals_analysis: str = Field(..., description="Analysis of career goals compatibility")
    suggested_strategy: str = Field(..., description="What to prioritise in counselling")
    warnings: list[str] = Field(default_factory=list, description="Potential issues to be aware of")


class InstituteInfo(BaseModel):
    """Response item for GET /institutes."""
    name: str
    short_name: str
    type: str
    city: str
    state: str
    tier: int
    nirf_rank: Optional[int]
    research_score: int
    placement_median_lpa: float
    coding_culture_score: int
    strengths: list[str]
    known_for: str


class BranchInfo(BaseModel):
    """Response item for GET /branches."""
    name: str
    short_name: str
    domain: str
    career_paths: list[str]
    coding_intensity: int = Field(..., ge=1, le=5)
    research_scope: int = Field(..., ge=1, le=5)
    median_lpa: float
    avg_salary_lpa: float
    suits_goals: list[str]


class CutoffRow(BaseModel):
    """Single row from the cutoffs table."""
    institute: str
    program: str
    branch: str
    category: str
    gender: str
    opening_rank: int
    closing_rank: int
    round: int
    year: int
    exam_type: str
    state_quota: Optional[str] = None


class ComparisonResult(BaseModel):
    """Response for GET /compare."""
    option_a: dict = Field(..., description="Institute A full details")
    option_b: dict = Field(..., description="Institute B full details")
    verdict: str = Field(..., description="Which is better for this student and why")
    head_to_head: dict = Field(..., description="Side-by-side metric comparison")


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: Literal["ok", "degraded", "down"]
    version: str
    db_connected: bool
    cutoffs_loaded: bool
    cutoff_years: list[int]
    total_cutoff_rows: int
    llm_enabled: bool
    cache_entries: int
