"""
ETL Pipeline: Excel cutoff files → DuckDB

Drop any .xlsx into data/cutoffs/ and call CutoffLoader().load_all().
Year and exam type are auto-detected from the filename.

Connection strategy:
  - One DuckDB connection per process (singleton via module global)
  - Opens read-write; read-only fallback if another process holds the lock
  - Caller must NOT share the connection across threads without a lock
"""

import json
import re
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

from app.config import get_settings
from app.etl import cleaner, schema
from app.etl.category_expander import expand_categories
from app.exceptions import DatabaseError, ETLError
from app.utils.logger import get_logger

log = get_logger(__name__)

PRIMARY_DATASET_URL = (
    "https://raw.githubusercontent.com/atmabodha/OpenNLP"
    "/main/IIT-JEE/JEE_2025_Cutoffs.xlsx"
)

_db_conn: duckdb.DuckDBPyConnection | None = None


def get_db() -> duckdb.DuckDBPyConnection:
    """
    Return the shared DuckDB connection (lazy, singleton per process).

    Opens read-write first; falls back to read-only if another process
    holds the write lock (e.g. during development with two terminals).
    """
    global _db_conn
    if _db_conn is not None:
        return _db_conn

    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    for read_only in (False, True):
        try:
            _db_conn = duckdb.connect(str(settings.db_path), read_only=read_only)
            if not read_only:
                schema.initialize(_db_conn)
            mode = "ro" if read_only else "rw"
            log.info(f"DuckDB connected ({mode}): {settings.db_path}")
            return _db_conn
        except duckdb.IOException:
            if read_only:
                raise DatabaseError(
                    f"Cannot open DuckDB at {settings.db_path} — "
                    "file may be locked or corrupted."
                )
            log.warning("DuckDB write lock busy — retrying read-only")

    raise DatabaseError("Unreachable")  # mypy guard


def close_db() -> None:
    """Close the DuckDB connection (call on shutdown)."""
    global _db_conn
    if _db_conn is not None:
        try:
            _db_conn.close()
        except Exception:
            pass
        _db_conn = None


class CutoffLoader:
    """
    Orchestrates the full ETL pipeline:
      1. Download xlsx (if not present)
      2. Detect year + exam type from filename
      3. Normalize columns
      4. Clean data
      5. Upsert into DuckDB (idempotent)
      6. Load knowledge base (institutes + branches)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.db = get_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def download_primary_dataset(self) -> Path:
        """Download the JEE 2025 cutoff xlsx if not already present."""
        dest = self.settings.cutoffs_dir / "JEE_2025_Cutoffs.xlsx"
        if dest.exists():
            log.info(f"Dataset already present: {dest}")
            return dest

        log.info("Downloading primary dataset from GitHub...")
        self.settings.cutoffs_dir.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(PRIMARY_DATASET_URL, dest)
        except Exception as exc:
            raise ETLError(f"Download failed: {exc}") from exc

        log.info(f"Downloaded: {dest} ({dest.stat().st_size // 1024} KB)")
        return dest

    def load_all(self) -> dict[str, int]:
        """
        Load all .xlsx files in cutoffs_dir.
        Returns {filename: rows_inserted} — negative value means failure.
        """
        xlsx_files = list(self.settings.cutoffs_dir.glob("*.xlsx"))
        if not xlsx_files:
            self.download_primary_dataset()
            xlsx_files = list(self.settings.cutoffs_dir.glob("*.xlsx"))

        results: dict[str, int] = {}
        for path in xlsx_files:
            try:
                results[path.name] = self.load_file(path)
            except Exception as exc:
                log.error(f"Failed to load {path.name}: {exc}")
                results[path.name] = -1

        self._load_knowledge_base()
        return results

    def load_file(self, path: Path) -> int:
        """Load one xlsx into DuckDB. Returns number of rows inserted."""
        year      = self._detect_year(path)
        exam_type = self._detect_exam_type(path)
        log.info(f"Loading {path.name} | year={year} | exam={exam_type}")

        df = self._read_excel(path)
        log.info(f"  Raw rows: {len(df)}, columns: {list(df.columns)}")

        df = cleaner.normalize_columns(df)
        df = cleaner.clean(df)
        log.info(f"  Clean rows: {len(df)}")

        # Expand to all categories if only OPEN data present
        if df["category"].nunique() == 1 and df["category"].iloc[0] == "OPEN":
            log.info("  Only OPEN category found — deriving reserved category rows")
            df = expand_categories(df)
            log.info(f"  After expansion: {len(df)} rows across {df['category'].nunique()} categories")

        df["year"]      = year
        df["exam_type"] = exam_type
        df["seat_type"] = "REGULAR"
        if "program" not in df.columns:
            df["program"] = df.get("branch", "Unknown")
        if "state_quota" not in df.columns:
            df["state_quota"] = None

        db_cols = [
            "year", "round", "institute", "program", "branch",
            "category", "gender", "opening_rank", "closing_rank",
            "exam_type", "state_quota", "seat_type",
        ]
        df = df[[c for c in db_cols if c in df.columns]]

        self.db.execute(
            "DELETE FROM cutoffs WHERE year = ? AND exam_type = ?",
            [year, exam_type],
        )
        self.db.execute(f"""
            INSERT INTO cutoffs
                (id, {', '.join(df.columns)})
            SELECT
                row_number() OVER () + COALESCE((SELECT MAX(id) FROM cutoffs), 0),
                {', '.join(df.columns)}
            FROM df
        """)
        self.db.commit()

        count = self.db.execute(
            "SELECT COUNT(*) FROM cutoffs WHERE year=? AND exam_type=?",
            [year, exam_type],
        ).fetchone()[0]
        log.info(f"  Inserted {count} rows")
        return count

    def get_stats(self) -> dict:
        """Return summary statistics about loaded cutoff data."""
        try:
            rows  = self.db.execute("SELECT COUNT(*) FROM cutoffs").fetchone()[0]
            years = [
                r[0] for r in
                self.db.execute("SELECT DISTINCT year FROM cutoffs ORDER BY year").fetchall()
            ]
            insts = self.db.execute(
                "SELECT COUNT(DISTINCT institute) FROM cutoffs"
            ).fetchone()[0]
            return {"total_rows": rows, "years": years, "institute_count": insts}
        except Exception:
            return {"total_rows": 0, "years": [], "institute_count": 0}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _read_excel(self, path: Path) -> pd.DataFrame:
        xl   = pd.ExcelFile(path)
        best = max(
            xl.sheet_names,
            key=lambda s: len(pd.read_excel(path, sheet_name=s)),
        )
        return pd.read_excel(path, sheet_name=best)

    def _detect_year(self, path: Path) -> int:
        m = re.search(r"(20\d{2})", path.stem)
        return int(m.group(1)) if m else 2025

    def _detect_exam_type(self, path: Path) -> str:
        name = path.stem.lower()
        if "main" in name:
            return "JEE_MAIN"
        return "JEE_ADVANCED"  # default — covers JoSAA (both exams)

    def _load_knowledge_base(self) -> None:
        self._load_institutes()
        self._load_branches()

    def _load_institutes(self) -> None:
        path = self.settings.knowledge_dir / "institute_tiers.json"
        if not path.exists():
            log.warning("institute_tiers.json not found — skipping")
            return
        with open(path) as f:
            data = json.load(f)
        self.db.execute("DELETE FROM institutes")
        rows = [
            (i, name,
             m.get("short"), m.get("type"), m.get("city"), m.get("state"),
             m.get("tier"), m.get("nirf_rank"), m.get("research_score"),
             m.get("placement_median_lpa"), m.get("coding_culture_score"),
             json.dumps(m.get("strengths", [])), m.get("known_for"))
            for i, (name, m) in enumerate(data.items(), 1)
        ]
        self.db.executemany(
            "INSERT INTO institutes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        self.db.commit()
        log.info(f"Loaded {len(rows)} institutes")

    def _load_branches(self) -> None:
        path = self.settings.knowledge_dir / "branch_profiles.json"
        if not path.exists():
            log.warning("branch_profiles.json not found — skipping")
            return
        with open(path) as f:
            data = json.load(f)
        self.db.execute("DELETE FROM branches")
        rows = [
            (i, name,
             m.get("short"), m.get("domain"),
             json.dumps(m.get("career_paths", [])),
             m.get("coding_intensity"), m.get("research_scope"),
             m.get("median_lpa"), m.get("avg_salary_lpa"),
             json.dumps(m.get("suits_goals", [])))
            for i, (name, m) in enumerate(data.items(), 1)
        ]
        self.db.executemany(
            "INSERT INTO branches VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        self.db.commit()
        log.info(f"Loaded {len(rows)} branches")
