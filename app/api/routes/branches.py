"""GET /branches — list branches with career and interest data."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import knowledge_loader

router = APIRouter()


@router.get("/branches")
def list_branches(
    domain:     Optional[str] = Query(None, description="CS / EE / ECE / MECH / CIVIL / CHEM / EP"),
    min_salary: Optional[float] = Query(None, description="Min median LPA"),
    kl=Depends(knowledge_loader),
):
    """List all branches in knowledge base with career metadata."""
    branches_raw = kl._cache.get("branch_profiles", {})

    results = []
    for name, meta in branches_raw.items():
        if domain and meta.get("domain", "").upper() != domain.upper():
            continue
        if min_salary and (meta.get("median_lpa") or 0) < min_salary:
            continue

        iv = meta.get("interest_vector", {})
        results.append({
            "name":             name,
            "short_name":       meta.get("short"),
            "domain":           meta.get("domain"),
            "career_paths":     meta.get("career_paths", []),
            "coding_intensity": meta.get("coding_intensity"),
            "research_scope":   meta.get("research_scope"),
            "median_lpa":       meta.get("median_lpa"),
            "avg_salary_lpa":   meta.get("avg_salary_lpa"),
            "suits_goals":      meta.get("suits_goals", []),
            "top_recruiters":   meta.get("top_recruiters", []),
            "interest_vector":  iv,
        })

    results.sort(key=lambda x: x["median_lpa"] or 0, reverse=True)
    return {"total": len(results), "branches": results}


@router.get("/branches/{name}/details")
def branch_details(name: str, kl=Depends(knowledge_loader)):
    """Full details for one branch: profile + career + higher studies + recruiters."""
    profile  = kl.get_branch(name)
    career   = kl.get_branch_career(name)
    hs       = kl.get_branch_higher_studies(name)
    rec      = kl.get_branch_recruiters(name)
    startup  = kl.get_startup_branch_fit(name)

    if not profile:
        return {"error": f"Branch '{name}' not found"}

    return {
        "name":           name,
        "profile":        profile,
        "career":         career,
        "higher_studies": hs,
        "recruiters":     rec,
        "startup_fit":    startup,
    }
