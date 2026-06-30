"""GET /institutes — list institutes with optional filters."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import db, knowledge_loader

router = APIRouter()


@router.get("/institutes")
def list_institutes(
    type:      Optional[str] = Query(None, description="IIT / NIT / IIIT / GFTI"),
    state:     Optional[str] = Query(None, description="Filter by state"),
    min_tier:  Optional[int] = Query(None, ge=1, le=5),
    max_tier:  Optional[int] = Query(None, ge=1, le=5),
    conn=Depends(db),
    kl=Depends(knowledge_loader),
):
    """
    List all institutes in the knowledge base with optional filters.
    Returns metadata: tier, NIRF rank, placement median, known_for.
    """
    institutes_raw = kl._cache.get("institute_tiers", {})

    results = []
    for name, meta in institutes_raw.items():
        # Apply filters
        if type and meta.get("type", "").upper() != type.upper():
            continue
        if state and state.lower() not in (meta.get("state") or "").lower():
            continue
        if min_tier and (meta.get("tier") or 99) < min_tier:
            continue
        if max_tier and (meta.get("tier") or 99) > max_tier:
            continue

        # Enrich with live cutoff count from DB
        try:
            count = conn.execute(
                "SELECT COUNT(DISTINCT branch) FROM cutoffs WHERE institute LIKE ?",
                [f"%{name.split()[-1]}%"]
            ).fetchone()[0]
        except Exception:
            count = 0

        results.append({
            "name":                name,
            "short_name":          meta.get("short"),
            "type":                meta.get("type"),
            "city":                meta.get("city"),
            "state":               meta.get("state"),
            "tier":                meta.get("tier"),
            "nirf_rank":           meta.get("nirf_rank"),
            "research_score":      meta.get("research_score"),
            "placement_median_lpa":meta.get("placement_median_lpa"),
            "coding_culture_score":meta.get("coding_culture_score"),
            "strengths":           meta.get("strengths", []),
            "known_for":           meta.get("known_for"),
            "branches_in_db":      count,
        })

    # Sort by tier then NIRF rank
    results.sort(key=lambda x: (x["tier"] or 99, x["nirf_rank"] or 999))
    return {"total": len(results), "institutes": results}


@router.get("/institutes/{name}/placement")
def institute_placement(name: str, kl=Depends(knowledge_loader)):
    """Get detailed placement data for an institute."""
    placement = kl.get_institute_placement(name)
    startup   = kl.get_institute_startup(name)
    recruiters= kl.get_institute_recruiters(name)

    if not placement and not startup:
        return {"error": f"No placement data found for '{name}'"}

    return {
        "institute":      name,
        "placement":      placement,
        "startup":        startup,
        "top_recruiters": recruiters,
    }
