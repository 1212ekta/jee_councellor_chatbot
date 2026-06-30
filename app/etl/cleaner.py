"""
Cleans and normalizes raw cutoff DataFrames from Excel.

Real JEE 2025 Cutoffs.xlsx structure (confirmed):
  Columns: Institute | Academic Program Name | Quota | Seat Type | Gender | Opening Rank | Closing Rank
  Quota values: AI, GO, HS, JK, LA, OS
  Seat Type: always OPEN (this file is OPEN category only)
  Gender: 'Gender-Neutral' | 'Female-only (including Supernumerary)'
"""

import re
import pandas as pd
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Exact column map for the real dataset ─────────────────────────────────────
COLUMN_RENAME = {
    "Institute":                "institute",
    "Academic Program Name":    "program",
    "Quota":                    "state_quota",
    "Seat Type":                "category",
    "Gender":                   "gender",
    "Opening Rank":             "opening_rank",
    "Closing Rank":             "closing_rank",
}

# Additional aliases for future year files that may differ
COLUMN_ALIASES: dict[str, list[str]] = {
    "institute":      ["Institute", "College", "Institution Name", "INSTITUTE"],
    "program":        ["Academic Program Name", "Program", "Course", "Programme", "Branch Name Full"],
    "state_quota":    ["Quota", "State Quota", "QUOTA", "Allotment Quota"],
    "category":       ["Seat Type", "Category", "CATEGORY", "Quota Type"],
    "gender":         ["Gender", "GENDER", "Gender Pool"],
    "opening_rank":   ["Opening Rank", "OR", "Open Rank", "OPENING_RANK"],
    "closing_rank":   ["Closing Rank", "CR", "Close Rank", "CLOSING_RANK"],
    "round":          ["Round", "ROUND", "Round No"],
}

# Quota code meanings (for display and filtering)
QUOTA_LABELS = {
    "AI": "All India",
    "HS": "Home State",
    "OS": "Other State",
    "GO": "Goa State",
    "JK": "Jammu & Kashmir",
    "LA": "Ladakh",
}

# Gender standardization
GENDER_MAP = {
    "gender-neutral":                          "Gender-Neutral",
    "female-only (including supernumerary)":   "Female-Only",
    "female-only":                             "Female-Only",
    "female":                                  "Female-Only",
    "girls":                                   "Female-Only",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical names.
    Tries exact match first, then aliases, then fuzzy.
    """
    # Try direct rename (works for the real 2025 file)
    direct_matches = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    df = df.rename(columns=direct_matches)

    # For any canonical name still missing, try aliases
    already_mapped = set(df.columns)
    df_cols_lower = {c.lower().strip(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in already_mapped:
            continue
        for alias in aliases:
            match = df_cols_lower.get(alias.lower().strip())
            if match:
                df = df.rename(columns={match: canonical})
                break

    # If branch column doesn't exist, extract from program
    if "branch" not in df.columns and "program" in df.columns:
        log.info("Extracting branch from 'program' column")
        df["branch"] = df["program"].apply(extract_branch_from_program)

    return df


def extract_branch_from_program(program: str) -> str:
    """
    Extracts clean branch name from full program string.

    Examples:
      'Computer Science and Engineering (4 Years, Bachelor of Technology)'
        → 'Computer Science and Engineering'
      'B.Tech (CSE) - MBA (5 Years, ...)' → 'CSE - MBA'
      'Artificial Intelligence and Data Science (4 Years, B.Tech)'
        → 'Artificial Intelligence and Data Science'
    """
    if pd.isna(program):
        return "Unknown"
    p = str(program).strip()

    # Remove the trailing duration+degree part: ' (4 Years, Bachelor of Technology)'
    p = re.sub(r"\s*\(\d+\s+Years?[^)]*\)\s*$", "", p, flags=re.IGNORECASE).strip()

    # Remove leading degree prefix: 'B.Tech in ...', 'B. Tech. (...) -'
    p = re.sub(
        r"^(B\.?\s*Tech\.?|B\.?\s*E\.?|Bachelor of Technology)\s*(in\s+)?",
        "", p, flags=re.IGNORECASE
    ).strip()

    # Clean up trailing dashes or brackets
    p = p.strip(" -–()")

    return p if p else "Unknown"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline on a normalized DataFrame."""
    original = len(df)

    if "closing_rank" not in df.columns:
        raise ValueError(
            f"No closing_rank column found. Available: {list(df.columns)}"
        )

    # ── 1. Drop rows with non-numeric ranks ──────────────────────────────────
    df = df[pd.to_numeric(df["closing_rank"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["opening_rank"], errors="coerce").notna()]
    df["closing_rank"] = df["closing_rank"].astype(int)
    df["opening_rank"] = df["opening_rank"].astype(int)

    # ── 2. Standardize gender ────────────────────────────────────────────────
    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .astype(str).str.strip().str.lower()
            .map(lambda x: GENDER_MAP.get(x, "Gender-Neutral"))
        )
    else:
        df["gender"] = "Gender-Neutral"

    # ── 3. Category — this file uses Seat Type = 'OPEN' always.
    #       Real category info comes from Quota column (state quota).
    #       We'll use 'OPEN' as the seat category for all rows here.
    if "category" not in df.columns:
        df["category"] = "OPEN"
    else:
        df["category"] = df["category"].astype(str).str.strip().str.upper()

    # ── 4. Add round = 6 (final round, since this file is final allotment) ──
    if "round" not in df.columns:
        df["round"] = 6  # JoSAA final round

    # ── 5. Strip whitespace from string columns ──────────────────────────────
    for col in ["institute", "program", "branch", "state_quota"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ── 6. Drop full duplicates ──────────────────────────────────────────────
    df = df.drop_duplicates()

    # ── 7. Drop rows with empty institute / branch ───────────────────────────
    for col in ["institute", "branch"]:
        if col in df.columns:
            df = df[df[col].notna() & (df[col] != "") & (df[col] != "nan")]

    dropped = original - len(df)
    if dropped:
        log.info(f"Cleaned: removed {dropped} invalid rows → {len(df)} remain")

    return df.reset_index(drop=True)
