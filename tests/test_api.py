"""
Integration tests for API routes.
Uses FastAPI TestClient — no live server needed.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_STUDENT = {
    "jee_advanced_rank": 3500,
    "gender": "male",
    "category": "OPEN",
    "home_state": "Maharashtra",
    "interest_coding": 0.9,
    "interest_ai_ml": 0.8,
    "interest_research": 0.4,
    "interest_core_engineering": 0.1,
    "interest_electronics": 0.3,
    "interest_mechanical": 0.0,
    "interest_civil": 0.0,
    "interest_chemical": 0.0,
    "wants_startup": True,
    "salary_priority": 0.8,
    "brand_priority": 0.6,
    "location_flexibility": 0.7,
}


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_required_fields(self):
        r = client.get("/health")
        d = r.json()
        assert "status" in d
        assert "total_cutoff_rows" in d
        assert "knowledge_base" in d
        assert d["total_cutoff_rows"] > 0

    def test_health_db_connected(self):
        r = client.get("/health")
        assert r.json()["db_connected"] is True


# ── GET /institutes ───────────────────────────────────────────────────────────

class TestInstitutes:
    def test_list_all_institutes(self):
        r = client.get("/institutes")
        assert r.status_code == 200
        d = r.json()
        assert d["total"] > 0
        assert len(d["institutes"]) > 0

    def test_filter_by_type_iit(self):
        r = client.get("/institutes?type=IIT")
        assert r.status_code == 200
        d = r.json()
        for inst in d["institutes"]:
            assert inst["type"] == "IIT"

    def test_filter_by_type_nit(self):
        r = client.get("/institutes?type=NIT")
        d = r.json()
        for inst in d["institutes"]:
            assert inst["type"] == "NIT"

    def test_institute_has_required_fields(self):
        r = client.get("/institutes")
        inst = r.json()["institutes"][0]
        for field in ["name", "type", "tier"]:
            assert field in inst

    def test_placement_endpoint(self):
        r = client.get("/institutes/IIT Bombay/placement")
        assert r.status_code == 200


# ── GET /branches ─────────────────────────────────────────────────────────────

class TestBranches:
    def test_list_all_branches(self):
        r = client.get("/branches")
        assert r.status_code == 200
        assert r.json()["total"] > 0

    def test_filter_by_min_salary(self):
        r = client.get("/branches?min_salary=20")
        d = r.json()
        for b in d["branches"]:
            assert (b["median_lpa"] or 0) >= 20

    def test_branch_details(self):
        r = client.get("/branches/Computer Science and Engineering/details")
        assert r.status_code == 200
        d = r.json()
        assert "profile" in d
        assert "career" in d

    def test_unknown_branch_details(self):
        r = client.get("/branches/Carpet Technology XYZ/details")
        d = r.json()
        assert "error" in d


# ── GET /cutoffs ──────────────────────────────────────────────────────────────

class TestCutoffs:
    def test_cutoffs_returns_data(self):
        r = client.get("/cutoffs")
        assert r.status_code == 200
        d = r.json()
        assert d["total"] > 0

    def test_filter_by_institute(self):
        r = client.get("/cutoffs?institute=IIT Bombay&limit=10")
        d = r.json()
        for row in d["cutoffs"]:
            assert "Bombay" in row["institute"]

    def test_filter_by_max_rank(self):
        r = client.get("/cutoffs?max_closing_rank=200&limit=20")
        d = r.json()
        for row in d["cutoffs"]:
            assert row["closing_rank"] <= 200

    def test_cutoffs_stats(self):
        r = client.get("/cutoffs/stats")
        assert r.status_code == 200
        d = r.json()
        assert d["total_rows"] > 0
        assert d["institutes"] > 0
        assert d["branches"] > 0
        assert 2025 in d["years_available"]


# ── POST /recommend ───────────────────────────────────────────────────────────

class TestRecommend:
    def test_recommend_returns_200(self):
        r = client.post("/recommend", json=VALID_STUDENT)
        assert r.status_code == 200

    def test_recommend_has_session_id(self):
        r = client.post("/recommend", json=VALID_STUDENT)
        d = r.json()
        assert "session_id" in d
        assert len(d["session_id"]) == 36  # UUID format

    def test_recommend_has_persona(self):
        r = client.post("/recommend", json=VALID_STUDENT)
        d = r.json()
        assert "persona" in d
        assert d["persona"]["id"] in [
            "software_engineer", "ai_researcher", "entrepreneur",
            "mba_aspirant", "core_engineer", "govt_psu_aspirant",
            "higher_studies_abroad", "electronics_vlsi", "undecided_explorer"
        ]

    def test_recommend_has_all_buckets(self):
        r = client.post("/recommend", json=VALID_STUDENT)
        d = r.json()
        assert "buckets" in d
        buckets = d["buckets"]
        for level in ["dream", "target", "safe", "very_safe"]:
            assert level in buckets

    def test_recommend_bucket_has_scores(self):
        r = client.post("/recommend", json=VALID_STUDENT)
        d = r.json()
        # Find first non-empty bucket
        for level in ["very_safe", "safe", "target", "dream"]:
            recs = d["buckets"].get(level, [])
            if recs:
                rec = recs[0]
                assert "scores" in rec
                s = rec["scores"]
                assert "overall" in s
                assert 0 <= s["overall"] <= 1
                break

    def test_recommend_has_compatibility(self):
        r = client.post("/recommend", json=VALID_STUDENT)
        d = r.json()
        for level in ["very_safe", "safe", "target"]:
            recs = d["buckets"].get(level, [])
            if recs:
                assert "compatibility" in recs[0]
                comp = recs[0]["compatibility"]
                assert "overall_compatibility" in comp
                assert "badges" in comp
                break

    def test_recommend_has_explanation(self):
        r = client.post("/recommend", json=VALID_STUDENT)
        d = r.json()
        for level in ["very_safe", "safe", "target"]:
            recs = d["buckets"].get(level, [])
            if recs:
                expl = recs[0]["explanation"]
                assert "why_institute" in expl
                assert "why_branch" in expl
                assert "pros" in expl
                assert "cons" in expl
                assert "career_roadmap" in expl
                break

    def test_recommend_has_reason_codes(self):
        r = client.post("/recommend", json=VALID_STUDENT)
        d = r.json()
        for level in ["very_safe", "safe", "target", "dream"]:
            recs = d["buckets"].get(level, [])
            if recs:
                rc = recs[0]["reason_codes"]
                assert "codes" in rc
                assert "labels" in rc
                assert isinstance(rc["codes"], list)
                break

    def test_recommend_total_scored_positive(self):
        r = client.post("/recommend", json=VALID_STUDENT)
        d = r.json()
        assert d["total_scored"] > 0

    def test_recommend_female_student(self):
        student = {**VALID_STUDENT, "gender": "female", "jee_advanced_rank": 5000}
        r = client.post("/recommend", json=student)
        assert r.status_code == 200
        d = r.json()
        # Female students should get Female-Only seat recommendations
        all_recs = []
        for level in d["buckets"].values():
            all_recs.extend(level)
        # At least some recs should exist
        assert len(all_recs) > 0

    def test_recommend_low_rank_student(self):
        """Very high rank (bad) should still return valid response."""
        student = {**VALID_STUDENT, "jee_advanced_rank": 200000}
        r = client.post("/recommend", json=student)
        assert r.status_code == 200

    def test_recommend_top_rank_student(self):
        """Rank 1 should return very safe options."""
        student = {**VALID_STUDENT, "jee_advanced_rank": 1}
        r = client.post("/recommend", json=student)
        assert r.status_code == 200
        d = r.json()
        assert len(d["buckets"]["very_safe"]) > 0


# ── POST /analyze-profile ─────────────────────────────────────────────────────

class TestAnalyzeProfile:
    def test_analyze_returns_200(self):
        r = client.post("/analyze-profile", json=VALID_STUDENT)
        assert r.status_code == 200

    def test_analyze_has_persona(self):
        r = client.post("/analyze-profile", json=VALID_STUDENT)
        d = r.json()
        assert "persona" in d
        assert d["persona"]["primary"]["id"]

    def test_analyze_has_top_branches(self):
        r = client.post("/analyze-profile", json=VALID_STUDENT)
        d = r.json()
        assert "top_branch_matches" in d
        assert len(d["top_branch_matches"]) > 0

    def test_analyze_has_rank_band(self):
        r = client.post("/analyze-profile", json=VALID_STUDENT)
        d = r.json()
        assert "rank_band" in d
        assert "rank" in d


# ── GET /compare ──────────────────────────────────────────────────────────────

class TestCompare:
    def test_compare_branches(self):
        r = client.get(
            "/compare/branches"
            "?branch_a=Computer Science and Engineering"
            "&branch_b=Mechanical Engineering"
        )
        assert r.status_code == 200
        d = r.json()
        assert "branch_a" in d
        assert "branch_b" in d
        assert "verdict" in d

    def test_compare_institutes(self):
        r = client.get(
            "/compare/institutes"
            "?inst_a=IIT Bombay&inst_b=IIT Delhi"
        )
        assert r.status_code == 200
        d = r.json()
        assert "head_to_head" in d


# ── POST /counselor/chat ──────────────────────────────────────────────────────

class TestCounselorChat:
    def test_chat_returns_answer(self):
        r = client.post("/counselor/chat", json={
            "student": VALID_STUDENT,
            "question": "Should I choose CSE or ECE?",
        })
        assert r.status_code == 200
        d = r.json()
        assert "answer" in d
        assert len(d["answer"]) > 20

    def test_chat_returns_sources(self):
        r = client.post("/counselor/chat", json={
            "student": VALID_STUDENT,
            "question": "Is CSE better than ECE for placements?",
        })
        d = r.json()
        assert "sources" in d
        assert isinstance(d["sources"], list)

    def test_chat_faq_question(self):
        r = client.post("/counselor/chat", json={
            "student": VALID_STUDENT,
            "question": "quota",
        })
        d = r.json()
        assert d["faq_match"] is not None


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_missing_rank_rejected(self):
        bad = {k: v for k, v in VALID_STUDENT.items()
               if k not in ("jee_advanced_rank", "jee_main_rank")}
        r = client.post("/recommend", json=bad)
        assert r.status_code == 422  # Pydantic validation error

    def test_invalid_category(self):
        bad = {**VALID_STUDENT, "category": "INVALID_CAT"}
        r = client.post("/recommend", json=bad)
        assert r.status_code == 422

    def test_interest_out_of_range(self):
        bad = {**VALID_STUDENT, "interest_coding": 1.5}
        r = client.post("/recommend", json=bad)
        assert r.status_code == 422

    def test_session_not_found(self):
        r = client.get("/sessions/nonexistent-id-12345")
        assert r.status_code == 404

    def test_session_roundtrip(self):
        """Create a session via /recommend and retrieve it."""
        r1 = client.post("/recommend", json=VALID_STUDENT)
        session_id = r1.json()["session_id"]

        r2 = client.get(f"/sessions/{session_id}")
        assert r2.status_code == 200
        d = r2.json()
        assert d["session_id"] == session_id
        assert "input" in d
        assert "summary" in d
