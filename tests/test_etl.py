"""Tests for the ETL pipeline (loader + cleaner)."""

import pandas as pd
import pytest

from app.etl.cleaner import (
    clean,
    extract_branch_from_program,
    normalize_columns,
)


# ── extract_branch_from_program ───────────────────────────────────────────────

class TestExtractBranch:
    @pytest.mark.parametrize("program,expected", [
        ("Civil Engineering (4 Years, Bachelor of Technology)",
         "Civil Engineering"),
        ("Computer Science and Engineering (4 Years, Bachelor of Technology)",
         "Computer Science and Engineering"),
        ("Artificial Intelligence and Data Science (4 Years, Bachelor of Technology)",
         "Artificial Intelligence and Data Science"),
        ("Mathematics and Computing (4 Years, Bachelor of Technology)",
         "Mathematics and Computing"),
        ("B.Tech Computer Science (4 Years, Bachelor of Technology)",
         "Computer Science"),
        ("", "Unknown"),
        (None, "Unknown"),
    ])
    def test_extraction(self, program, expected):
        assert extract_branch_from_program(program) == expected


# ── normalize_columns ─────────────────────────────────────────────────────────

class TestNormalizeColumns:
    def _real_df(self):
        """Build a DataFrame matching the real JEE 2025 xlsx structure."""
        return pd.DataFrame({
            "Institute": ["IIT Bombay", "IIT Delhi"],
            "Academic Program Name": [
                "Computer Science and Engineering (4 Years, Bachelor of Technology)",
                "Electrical Engineering (4 Years, Bachelor of Technology)",
            ],
            "Quota":       ["AI", "HS"],
            "Seat Type":   ["OPEN", "OPEN"],
            "Gender":      ["Gender-Neutral", "Female-only (including Supernumerary)"],
            "Opening Rank":[1, 300],
            "Closing Rank":[66, 700],
        })

    def test_columns_renamed(self):
        df = normalize_columns(self._real_df())
        assert "institute" in df.columns
        assert "closing_rank" in df.columns
        assert "state_quota" in df.columns

    def test_branch_extracted_from_program(self):
        df = normalize_columns(self._real_df())
        assert "branch" in df.columns
        assert "Computer Science and Engineering" in df["branch"].values

    def test_gender_standardized(self):
        df = normalize_columns(self._real_df())
        df = clean(df)
        assert "Female-Only" in df["gender"].values
        assert "Gender-Neutral" in df["gender"].values


# ── clean ─────────────────────────────────────────────────────────────────────

class TestClean:
    def _base_df(self):
        return pd.DataFrame({
            "institute":    ["IIT Bombay", "IIT Delhi", "  ", ""],
            "branch":       ["CSE", "EE", "Civil", "Mech"],
            "program":      ["B.Tech CSE", "B.Tech EE", "B.Tech Civil", "B.Tech Mech"],
            "category":     ["OPEN", "OPEN", "OPEN", "OPEN"],
            "gender":       ["Gender-Neutral", "female-only (including supernumerary)",
                            "Gender-Neutral", "Gender-Neutral"],
            "opening_rank": [1, 300, 100, 200],
            "closing_rank": [66, 700, 500, 800],
            "state_quota":  ["AI", "HS", "OS", "AI"],
        })

    def test_empty_institute_dropped(self):
        df = clean(self._base_df())
        assert "" not in df["institute"].values
        assert "  " not in df["institute"].values

    def test_gender_normalized(self):
        df = clean(self._base_df())
        assert "Female-Only" in df["gender"].values

    def test_closing_rank_is_int(self):
        df = clean(self._base_df())
        assert df["closing_rank"].dtype in (int, "int64", "int32")

    def test_non_numeric_rank_dropped(self):
        base = self._base_df()
        base["closing_rank"] = base["closing_rank"].astype(object)
        base.loc[0, "closing_rank"] = "P"   # PWD placeholder
        df = clean(base)
        assert 66 not in df["closing_rank"].values  # that row dropped

    def test_round_added_if_missing(self):
        df = clean(self._base_df())
        assert "round" in df.columns
        assert df["round"].iloc[0] == 6   # default final round

    def test_raises_if_no_closing_rank_column(self):
        from app.exceptions import ETLError
        bad = pd.DataFrame({"institute": ["IIT"], "branch": ["CSE"]})
        with pytest.raises(Exception):
            clean(bad)


# ── CutoffLoader ──────────────────────────────────────────────────────────────

class TestCutoffLoader:
    def test_detect_year_from_filename(self, tmp_path):
        from pathlib import Path
        from app.etl.loader import CutoffLoader
        loader = CutoffLoader()
        assert loader._detect_year(Path("JEE_2025_Cutoffs.xlsx")) == 2025
        assert loader._detect_year(Path("cutoffs_2024.xlsx")) == 2024
        assert loader._detect_year(Path("no_year.xlsx")) == 2025  # fallback

    def test_detect_exam_type(self, tmp_path):
        from pathlib import Path
        from app.etl.loader import CutoffLoader
        loader = CutoffLoader()
        assert loader._detect_exam_type(Path("JEE_Main_2025.xlsx")) == "JEE_MAIN"
        assert loader._detect_exam_type(Path("JEE_Advanced_2025.xlsx")) == "JEE_ADVANCED"
        assert loader._detect_exam_type(Path("JEE_2025_Cutoffs.xlsx")) == "JEE_ADVANCED"

    def test_get_stats_returns_dict(self):
        from app.etl.loader import CutoffLoader
        loader = CutoffLoader()
        stats = loader.get_stats()
        assert "total_rows" in stats
        assert "years" in stats
        assert stats["total_rows"] >= 0

    def test_data_already_loaded(self):
        """DB should have rows from previous test run or fixture."""
        from app.etl.loader import get_db
        db = get_db()
        count = db.execute("SELECT COUNT(*) FROM cutoffs").fetchone()[0]
        assert count > 0, "Cutoff data should be loaded in the test DB"
