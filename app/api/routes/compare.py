"""GET /compare — side-by-side branch or institute comparison."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import knowledge_loader

router = APIRouter()


@router.get("/compare/branches")
def compare_branches(
    branch_a: str = Query(..., description="First branch name"),
    branch_b: str = Query(..., description="Second branch name"),
    kl=Depends(knowledge_loader),
):
    """
    Side-by-side comparison of two branches.
    Returns metrics, verdict, and head-to-head breakdown.
    """
    profile_a = kl.get_branch(branch_a)
    profile_b = kl.get_branch(branch_b)
    career_a  = kl.get_branch_career(branch_a)
    career_b  = kl.get_branch_career(branch_b)
    hs_a      = kl.get_branch_higher_studies(branch_a)
    hs_b      = kl.get_branch_higher_studies(branch_b)
    rec_a     = kl.get_branch_recruiters(branch_a)
    rec_b     = kl.get_branch_recruiters(branch_b)

    # Try knowledge base comparison
    kb_comparison = kl.get_branch_comparison(branch_a, branch_b)

    def metric(a_val, b_val, higher_is_better=True) -> str:
        if a_val is None or b_val is None:
            return "equal"
        if higher_is_better:
            return branch_a if a_val > b_val else (branch_b if b_val > a_val else "equal")
        return branch_a if a_val < b_val else (branch_b if b_val < a_val else "equal")

    head_to_head = {
        "median_salary":     metric(profile_a.get("median_lpa"), profile_b.get("median_lpa")),
        "coding_intensity":  metric(profile_a.get("coding_intensity"), profile_b.get("coding_intensity")),
        "research_scope":    metric(profile_a.get("research_scope"), profile_b.get("research_scope")),
        "startup_friendly":  "check startup_ecosystem.json",
        "gate_relevance":    (
            branch_a if career_a.get("gate_relevant") and not career_b.get("gate_relevant")
            else branch_b if career_b.get("gate_relevant") and not career_a.get("gate_relevant")
            else "both" if career_a.get("gate_relevant") and career_b.get("gate_relevant")
            else "neither"
        ),
    }

    return {
        "branch_a": {
            "name":            branch_a,
            "profile":         profile_a,
            "career":          career_a,
            "higher_studies":  hs_a.get("top_ms_programs", [])[:3],
            "tier1_recruiters":rec_a.get("tier_1", [])[:5],
            "mba_transition":  career_a.get("mba_transition"),
            "startup_fit":     career_a.get("startup_friendliness"),
        },
        "branch_b": {
            "name":            branch_b,
            "profile":         profile_b,
            "career":          career_b,
            "higher_studies":  hs_b.get("top_ms_programs", [])[:3],
            "tier1_recruiters":rec_b.get("tier_1", [])[:5],
            "mba_transition":  career_b.get("mba_transition"),
            "startup_fit":     career_b.get("startup_friendliness"),
        },
        "head_to_head":   head_to_head,
        "kb_comparison":  kb_comparison,
        "verdict": (
            kb_comparison.get("verdict")
            if kb_comparison
            else f"No pre-built comparison available for {branch_a} vs {branch_b}. "
                 f"Use the metrics above to decide."
        ),
    }


@router.get("/compare/institutes")
def compare_institutes(
    inst_a:  str = Query(..., description="First institute name"),
    inst_b:  str = Query(..., description="Second institute name"),
    branch:  Optional[str] = Query(None, description="Optional: compare for a specific branch"),
    kl=Depends(knowledge_loader),
):
    """Side-by-side institute comparison, optionally for a specific branch."""
    meta_a     = kl.get_institute(inst_a)
    meta_b     = kl.get_institute(inst_b)
    place_a    = kl.get_institute_placement(inst_a)
    place_b    = kl.get_institute_placement(inst_b)
    startup_a  = kl.get_institute_startup(inst_a)
    startup_b  = kl.get_institute_startup(inst_b)

    def winner(a, b, key, higher=True):
        va = a.get(key)
        vb = b.get(key)
        if va is None or vb is None:
            return "unknown"
        if higher:
            return inst_a if va > vb else (inst_b if vb > va else "equal")
        return inst_a if va < vb else (inst_b if vb < va else "equal")

    return {
        "institute_a": {
            "name":              inst_a,
            "meta":              meta_a,
            "placement":         place_a,
            "startup_culture":   startup_a.get("ecell_rating"),
            "notable_startups":  startup_a.get("notable_startups", [])[:3],
        },
        "institute_b": {
            "name":              inst_b,
            "meta":              meta_b,
            "placement":         place_b,
            "startup_culture":   startup_b.get("ecell_rating"),
            "notable_startups":  startup_b.get("notable_startups", [])[:3],
        },
        "head_to_head": {
            "placement_median": winner(place_a, place_b, "median_lpa"),
            "nirf_rank":        winner(meta_a, meta_b, "nirf_rank", higher=False),
            "research_score":   winner(meta_a, meta_b, "research_score"),
            "startup_culture":  winner(startup_a, startup_b, "ecell_rating"),
            "tier":             winner(meta_a, meta_b, "tier", higher=False),
        },
        "branch_context": (
            f"For {branch}: "
            f"{meta_a.get('short','?')} strengths={meta_a.get('strengths',[])} | "
            f"{meta_b.get('short','?')} strengths={meta_b.get('strengths',[])}"
            if branch else None
        ),
    }
