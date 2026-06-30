"""
Custom exceptions for the JEE Counselor application.

Using a clear hierarchy lets callers catch at the right level:
  - JEECounselorError      — catch-all for any app error
  - DataError              — ETL / DB / knowledge base problems
  - EngineError            — scoring / persona / compatibility problems
  - ValidationError        — bad inputs caught before engine runs
"""


class JEECounselorError(Exception):
    """Base exception for all application errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


# ── Data layer ────────────────────────────────────────────────────────────────

class DataError(JEECounselorError):
    """Raised when data loading or DB operations fail."""


class ETLError(DataError):
    """Raised when Excel ingestion fails."""


class DatabaseError(DataError):
    """Raised when DuckDB operations fail."""


class KnowledgeBaseError(DataError):
    """Raised when knowledge JSON files are missing or malformed."""


# ── Engine layer ──────────────────────────────────────────────────────────────

class EngineError(JEECounselorError):
    """Raised when the recommendation engine encounters an unrecoverable error."""


class ScoringError(EngineError):
    """Raised when scoring produces invalid results."""


class PersonaError(EngineError):
    """Raised when persona inference fails."""


# ── API / Validation layer ────────────────────────────────────────────────────

class ValidationError(JEECounselorError):
    """Raised when API inputs fail domain validation beyond Pydantic."""


class NoCutoffDataError(DataError):
    """Raised when no cutoff rows are found for a student's rank range."""
    def __init__(self, rank: int, exam_type: str):
        super().__init__(
            f"No cutoff data found for rank {rank} ({exam_type}). "
            "Ensure the dataset is loaded via ETL.",
            details={"rank": rank, "exam_type": exam_type},
        )


class SessionNotFoundError(JEECounselorError):
    """Raised when a session ID does not exist in the DB."""
    def __init__(self, session_id: str):
        super().__init__(
            f"Session '{session_id}' not found.",
            details={"session_id": session_id},
        )
