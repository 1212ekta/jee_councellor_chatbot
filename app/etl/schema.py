"""
DuckDB table definitions.
Run once on startup via schema.initialize(conn).
"""

CUTOFFS_TABLE = """
CREATE TABLE IF NOT EXISTS cutoffs (
    id            INTEGER PRIMARY KEY,
    year          INTEGER NOT NULL,
    round         INTEGER,
    institute     VARCHAR NOT NULL,
    program       VARCHAR,
    branch        VARCHAR NOT NULL,
    category      VARCHAR NOT NULL,
    gender        VARCHAR,
    opening_rank  INTEGER,
    closing_rank  INTEGER NOT NULL,
    exam_type     VARCHAR NOT NULL,
    state_quota   VARCHAR,
    seat_type     VARCHAR DEFAULT 'REGULAR',
    loaded_at     TIMESTAMP DEFAULT current_timestamp
);
"""

INSTITUTES_TABLE = """
CREATE TABLE IF NOT EXISTS institutes (
    id                    INTEGER PRIMARY KEY,
    name                  VARCHAR UNIQUE NOT NULL,
    short_name            VARCHAR,
    type                  VARCHAR,
    city                  VARCHAR,
    state                 VARCHAR,
    tier                  INTEGER,
    nirf_rank             INTEGER,
    research_score        FLOAT,
    placement_median_lpa  FLOAT,
    coding_culture_score  FLOAT,
    strengths             VARCHAR,
    known_for             VARCHAR
);
"""

BRANCHES_TABLE = """
CREATE TABLE IF NOT EXISTS branches (
    id                INTEGER PRIMARY KEY,
    name              VARCHAR UNIQUE NOT NULL,
    short_name        VARCHAR,
    domain            VARCHAR,
    career_paths      VARCHAR,
    coding_intensity  INTEGER,
    research_scope    INTEGER,
    median_lpa        FLOAT,
    avg_salary_lpa    FLOAT,
    suits_goals       VARCHAR
);
"""

SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id          VARCHAR PRIMARY KEY,
    input_json  VARCHAR NOT NULL,
    output_json VARCHAR NOT NULL,
    created_at  TIMESTAMP DEFAULT current_timestamp
);
"""

ALL_TABLES = [CUTOFFS_TABLE, INSTITUTES_TABLE, BRANCHES_TABLE, SESSIONS_TABLE]

# ── Indexes for fast rank-range queries ───────────────────────────────────────
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_cutoffs_rank    ON cutoffs (closing_rank, opening_rank);",
    "CREATE INDEX IF NOT EXISTS idx_cutoffs_inst    ON cutoffs (institute);",
    "CREATE INDEX IF NOT EXISTS idx_cutoffs_branch  ON cutoffs (branch);",
    "CREATE INDEX IF NOT EXISTS idx_cutoffs_cat     ON cutoffs (category);",
    "CREATE INDEX IF NOT EXISTS idx_cutoffs_year    ON cutoffs (year);",
    "CREATE INDEX IF NOT EXISTS idx_cutoffs_exam    ON cutoffs (exam_type);",
]


def initialize(conn) -> None:
    """Create all tables and indexes. Safe to call multiple times."""
    for ddl in ALL_TABLES:
        conn.execute(ddl)
    for idx in INDEXES:
        conn.execute(idx)
    conn.commit()
