"""GET /cutoffs — query raw cutoff data from DuckDB."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import db

router = APIRouter()


@router.get("/cutoffs")
def get_cutoffs(
    institute:        Optional[str] = Query(None),
    branch:           Optional[str] = Query(None),
    category:         Optional[str] = Query(None, description="OPEN / OBC-NCL / SC / ST / EWS"),
    gender:           Optional[str] = Query(None, description="Gender-Neutral / Female-Only"),
    year:             Optional[int] = Query(None),
    state_quota:      Optional[str] = Query(None, description="AI / HS / OS"),
    max_closing_rank: Optional[int] = Query(None, gt=0),
    min_closing_rank: Optional[int] = Query(None, gt=0),
    limit:            int           = Query(100, ge=1, le=1000),
    conn=Depends(db),
):
    """
    Query cutoff data with filters.
    Returns raw rows from DuckDB — useful for debugging and data exploration.
    """
    conditions = []
    params     = []

    if institute:
        conditions.append("institute ILIKE ?")
        params.append(f"%{institute}%")
    if branch:
        conditions.append("branch ILIKE ?")
        params.append(f"%{branch}%")
    if category:
        conditions.append("category = ?")
        params.append(category.upper())
    if gender:
        conditions.append("gender ILIKE ?")
        params.append(f"%{gender}%")
    if year:
        conditions.append("year = ?")
        params.append(year)
    if state_quota:
        conditions.append("state_quota = ?")
        params.append(state_quota.upper())
    if max_closing_rank:
        conditions.append("closing_rank <= ?")
        params.append(max_closing_rank)
    if min_closing_rank:
        conditions.append("closing_rank >= ?")
        params.append(min_closing_rank)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql   = f"""
        SELECT institute, branch, category, gender, state_quota,
               opening_rank, closing_rank, round, year, exam_type
        FROM cutoffs
        {where}
        ORDER BY closing_rank
        LIMIT {limit}
    """

    rows = conn.execute(sql, params).fetchall()
    cols = ["institute","branch","category","gender","state_quota",
            "opening_rank","closing_rank","round","year","exam_type"]

    # Total count (without limit)
    count_sql = f"SELECT COUNT(*) FROM cutoffs {where}"
    total = conn.execute(count_sql, params).fetchone()[0]

    return {
        "total":  total,
        "shown":  len(rows),
        "cutoffs": [dict(zip(cols, r)) for r in rows],
    }


@router.get("/cutoffs/stats")
def cutoff_stats(conn=Depends(db)):
    """Summary statistics about loaded cutoff data."""
    stats = conn.execute("""
        SELECT
            COUNT(*)                        AS total_rows,
            COUNT(DISTINCT institute)       AS institutes,
            COUNT(DISTINCT branch)          AS branches,
            COUNT(DISTINCT category)        AS categories,
            MIN(closing_rank)               AS min_rank,
            MAX(closing_rank)               AS max_rank,
            MIN(year)                       AS earliest_year,
            MAX(year)                       AS latest_year
        FROM cutoffs
    """).fetchone()

    cols = ["total_rows","institutes","branches","categories",
            "min_rank","max_rank","earliest_year","latest_year"]

    years = [r[0] for r in conn.execute(
        "SELECT DISTINCT year FROM cutoffs ORDER BY year"
    ).fetchall()]

    top_institutes = conn.execute("""
        SELECT institute, COUNT(*) AS seat_count
        FROM cutoffs
        GROUP BY institute
        ORDER BY seat_count DESC
        LIMIT 10
    """).fetchall()

    return {
        **dict(zip(cols, stats)),
        "years_available":  years,
        "top_institutes":   [{"institute": r[0], "seat_rows": r[1]} for r in top_institutes],
    }
