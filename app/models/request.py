from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


CATEGORIES = Literal["OPEN", "OBC-NCL", "SC", "ST", "EWS", "OPEN-PwD", "OBC-NCL-PwD"]
GENDERS = Literal["male", "female", "other"]
EXAM_TYPES = Literal["JEE_ADVANCED", "JEE_MAIN", "BOTH"]


class StudentProfile(BaseModel):
    """
    Complete student profile used for recommendation.
    All interest fields are 0.0 (not interested) to 1.0 (very interested).
    """

    # ── Core identifiers ──────────────────────────────────────────────────────
    jee_advanced_rank: Optional[int] = Field(
        None, gt=0, le=500000, description="JEE Advanced CRL rank"
    )
    jee_main_rank: Optional[int] = Field(
        None, gt=0, le=1500000, description="JEE Main CRL rank"
    )
    gender: GENDERS = Field(..., description="Used to match gender-neutral vs female-only seats")
    category: CATEGORIES = Field("OPEN", description="Reservation category")
    home_state: str = Field(..., min_length=2, description="Student's home state (for state quota)")

    # ── Branch interests ──────────────────────────────────────────────────────
    interest_coding: float = Field(0.5, ge=0.0, le=1.0)
    interest_ai_ml: float = Field(0.5, ge=0.0, le=1.0)
    interest_research: float = Field(0.5, ge=0.0, le=1.0)
    interest_core_engineering: float = Field(0.5, ge=0.0, le=1.0)
    interest_electronics: float = Field(0.5, ge=0.0, le=1.0)
    interest_mechanical: float = Field(0.5, ge=0.0, le=1.0)
    interest_civil: float = Field(0.5, ge=0.0, le=1.0)
    interest_chemical: float = Field(0.5, ge=0.0, le=1.0)

    # ── Career goals ──────────────────────────────────────────────────────────
    wants_mba: bool = Field(False, description="Plans to do MBA later")
    wants_startup: bool = Field(False, description="Wants to start a company")
    wants_govt_job: bool = Field(False, description="Prefers government/PSU job")
    wants_higher_studies_abroad: bool = Field(False, description="Plans MS/PhD abroad")
    wants_research: bool = Field(False, description="Wants to go into academic research")

    # ── Priorities (0 = low priority, 1 = high priority) ─────────────────────
    salary_priority: float = Field(
        0.5, ge=0.0, le=1.0, description="0=work-life balance, 1=maximum salary"
    )
    brand_priority: float = Field(
        0.5, ge=0.0, le=1.0, description="0=doesn't care about institute name, 1=brand matters a lot"
    )
    location_flexibility: float = Field(
        0.5, ge=0.0, le=1.0, description="0=wants to stay close to home, 1=open to anywhere"
    )

    # ── Optional preferences ──────────────────────────────────────────────────
    preferred_states: list[str] = Field(
        default_factory=list, description="Preferred states for institute location"
    )
    preferred_exam: EXAM_TYPES = Field(
        "BOTH", description="Which exam rank to use for filtering"
    )
    max_results: int = Field(20, ge=1, le=50, description="Max recommendations to return")

    # ── Validators ────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def at_least_one_rank(self) -> "StudentProfile":
        if self.jee_advanced_rank is None and self.jee_main_rank is None:
            raise ValueError("At least one of jee_advanced_rank or jee_main_rank must be provided")
        return self

    @property
    def effective_rank(self) -> int:
        """
        Returns the best rank to use for matching.
        JEE Advanced rank is preferred when available (used for IITs).
        JEE Main rank is used for NITs/IIITs.
        """
        if self.jee_advanced_rank:
            return self.jee_advanced_rank
        return self.jee_main_rank  # type: ignore

    @property
    def interest_vector(self) -> list[float]:
        """Returns interests as ordered vector for cosine similarity."""
        return [
            self.interest_coding,
            self.interest_ai_ml,
            self.interest_research,
            self.interest_core_engineering,
            self.interest_electronics,
            self.interest_mechanical,
            self.interest_civil,
            self.interest_chemical,
        ]

    @property
    def active_goals(self) -> list[str]:
        """Returns list of active career goal keys."""
        goals = []
        if self.wants_mba:
            goals.append("wants_mba")
        if self.wants_startup:
            goals.append("startup")
        if self.wants_govt_job:
            goals.append("govt_job")
        if self.wants_higher_studies_abroad:
            goals.append("higher_studies_abroad")
        if self.wants_research:
            goals.append("research")
        if self.salary_priority > 0.7:
            goals.append("salary_priority")
        return goals

    model_config = {
        "json_schema_extra": {
            "example": {
                "jee_advanced_rank": 3500,
                "gender": "male",
                "category": "OPEN",
                "home_state": "Maharashtra",
                "interest_coding": 0.9,
                "interest_ai_ml": 0.8,
                "interest_research": 0.5,
                "interest_core_engineering": 0.2,
                "interest_electronics": 0.4,
                "interest_mechanical": 0.1,
                "interest_civil": 0.0,
                "interest_chemical": 0.1,
                "wants_startup": True,
                "wants_higher_studies_abroad": True,
                "salary_priority": 0.8,
                "brand_priority": 0.6,
            }
        }
    }


class CompareRequest(BaseModel):
    """Request to compare two institute+branch combinations."""
    institute_1: str = Field(..., description="First institute name")
    branch_1: str = Field(..., description="First branch name")
    institute_2: str = Field(..., description="Second institute name")
    branch_2: str = Field(..., description="Second branch name")
    student: Optional[StudentProfile] = Field(
        None, description="Optional student profile to personalise comparison"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "institute_1": "IIT Bombay",
                "branch_1": "Computer Science and Engineering",
                "institute_2": "IIT Delhi",
                "branch_2": "Computer Science and Engineering",
            }
        }
    }


class CutoffQueryParams(BaseModel):
    """Filters for querying the cutoff table."""
    institute: Optional[str] = None
    branch: Optional[str] = None
    year: Optional[int] = Field(None, ge=2010, le=2030)
    category: Optional[CATEGORIES] = None
    exam_type: Optional[EXAM_TYPES] = None
    max_closing_rank: Optional[int] = Field(None, gt=0)
    min_closing_rank: Optional[int] = Field(None, gt=0)
