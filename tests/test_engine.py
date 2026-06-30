"""
Unit tests for the JEE Counselor intelligence layer.

Run with:  pytest tests/test_engine.py -v
"""

import pytest
from app.models.request import StudentProfile


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cse_student():
    return StudentProfile(
        jee_advanced_rank=2000, gender="male", category="OPEN",
        home_state="Maharashtra",
        interest_coding=0.95, interest_ai_ml=0.90, interest_research=0.5,
        interest_core_engineering=0.1, interest_electronics=0.3,
        interest_mechanical=0.0, interest_civil=0.0, interest_chemical=0.0,
        wants_startup=True, salary_priority=0.8, brand_priority=0.7,
        location_flexibility=0.7,
    )

@pytest.fixture
def mech_student():
    return StudentProfile(
        jee_advanced_rank=8000, gender="male", category="OPEN",
        home_state="Gujarat",
        interest_coding=0.2, interest_ai_ml=0.1, interest_research=0.3,
        interest_core_engineering=0.95, interest_electronics=0.3,
        interest_mechanical=0.9, interest_civil=0.4, interest_chemical=0.3,
        wants_govt_job=False, salary_priority=0.5, brand_priority=0.5,
    )

@pytest.fixture
def research_student():
    return StudentProfile(
        jee_advanced_rank=1500, gender="female", category="OPEN",
        home_state="Karnataka",
        interest_coding=0.7, interest_ai_ml=0.9, interest_research=0.95,
        interest_core_engineering=0.2, interest_electronics=0.5,
        interest_mechanical=0.1, interest_civil=0.0, interest_chemical=0.2,
        wants_research=True, wants_higher_studies_abroad=True, salary_priority=0.3,
    )

@pytest.fixture
def govt_student():
    return StudentProfile(
        jee_advanced_rank=12000, gender="male", category="OBC-NCL",
        home_state="Bihar",
        interest_coding=0.2, interest_ai_ml=0.1, interest_research=0.2,
        interest_core_engineering=0.8, interest_electronics=0.3,
        interest_mechanical=0.7, interest_civil=0.9, interest_chemical=0.3,
        wants_govt_job=True, salary_priority=0.3, brand_priority=0.4,
    )


# ── StudentProfile model tests ────────────────────────────────────────────────

class TestStudentProfile:
    def test_effective_rank_advanced(self, cse_student):
        assert cse_student.effective_rank == 2000

    def test_effective_rank_main_only(self):
        s = StudentProfile(
            jee_main_rank=50000, gender="male", category="OPEN",
            home_state="Delhi",
        )
        assert s.effective_rank == 50000

    def test_requires_at_least_one_rank(self):
        with pytest.raises(ValueError, match="At least one"):
            StudentProfile(gender="male", category="OPEN", home_state="Delhi")

    def test_interest_vector_length(self, cse_student):
        assert len(cse_student.interest_vector) == 8

    def test_active_goals_startup(self, cse_student):
        assert "startup" in cse_student.active_goals

    def test_active_goals_research(self, research_student):
        assert "research" in research_student.active_goals

    def test_active_goals_empty_when_none_set(self):
        s = StudentProfile(
            jee_advanced_rank=5000, gender="male", category="OPEN",
            home_state="Delhi", salary_priority=0.5,
        )
        assert s.active_goals == []


# ── Interest Matcher tests ────────────────────────────────────────────────────

class TestInterestMatcher:
    def test_cse_student_matches_cse(self, cse_student):
        from app.engine.interest_matcher import compute_interest_match
        result = compute_interest_match(cse_student, "Computer Science and Engineering")
        assert result["score"] >= 0.85, f"CSE should score ≥0.85, got {result['score']}"

    def test_cse_student_low_on_mechanical(self, cse_student):
        from app.engine.interest_matcher import compute_interest_match
        result = compute_interest_match(cse_student, "Mechanical Engineering")
        assert result["score"] <= 0.55, f"Mech should score ≤0.55 for CSE student, got {result['score']}"

    def test_mech_student_matches_mechanical(self, mech_student):
        from app.engine.interest_matcher import compute_interest_match
        result = compute_interest_match(mech_student, "Mechanical Engineering")
        assert result["score"] >= 0.80

    def test_unknown_branch_returns_neutral(self, cse_student):
        from app.engine.interest_matcher import compute_interest_match
        result = compute_interest_match(cse_student, "Carpet Technology")
        assert result["score"] == 0.5
        assert result["matched_profile"] == "Unknown"

    def test_ai_branch_matches_ai_student(self, cse_student):
        from app.engine.interest_matcher import compute_interest_match
        result = compute_interest_match(cse_student, "Artificial Intelligence and Data Science")
        assert result["score"] >= 0.85

    def test_top_branches_returns_sorted_list(self, cse_student):
        from app.engine.interest_matcher import get_top_branches_for_student
        top = get_top_branches_for_student(cse_student, top_n=5)
        assert len(top) == 5
        scores = [b["score"] for b in top]
        assert scores == sorted(scores, reverse=True), "Should be sorted descending"

    def test_cosine_similarity_symmetric(self, cse_student, mech_student):
        """CSE student should prefer CSE; mech student should prefer Mech."""
        from app.engine.interest_matcher import compute_interest_match
        cse_on_cse  = compute_interest_match(cse_student, "Computer Science and Engineering")["score"]
        mech_on_cse = compute_interest_match(mech_student, "Computer Science and Engineering")["score"]
        assert cse_on_cse > mech_on_cse


# ── Risk Classifier tests ─────────────────────────────────────────────────────

class TestRiskClassifier:
    def test_rank_better_than_opening_is_very_safe(self):
        from app.engine.risk_classifier import assess_risk
        result = assess_risk(100, 500, 1000, "male", "Gender-Neutral")
        assert result.risk_level == "Very Safe"
        assert result.admission_probability >= 0.90

    def test_rank_at_closing_is_target(self):
        from app.engine.risk_classifier import assess_risk
        result = assess_risk(1000, 500, 1000, "male", "Gender-Neutral")
        assert result.risk_level in ("Target", "Dream")
        assert 0.30 <= result.admission_probability <= 0.65

    def test_rank_beyond_closing_is_dream_or_filtered(self):
        from app.engine.risk_classifier import assess_risk
        result = assess_risk(1300, 500, 1000, "male", "Gender-Neutral")
        assert result.risk_level in ("Dream", "Filtered")

    def test_rank_far_beyond_is_filtered(self):
        from app.engine.risk_classifier import assess_risk
        result = assess_risk(3000, 500, 1000, "male", "Gender-Neutral")
        assert result.risk_level == "Filtered"
        assert result.is_eligible is False

    def test_male_student_female_only_seat_ineligible(self):
        from app.engine.risk_classifier import assess_risk
        result = assess_risk(500, 100, 1000, "male", "Female-Only")
        assert result.is_eligible is False
        assert result.admission_probability == 0.0

    def test_female_student_female_only_seat_eligible(self):
        from app.engine.risk_classifier import assess_risk
        result = assess_risk(500, 100, 1000, "female", "Female-Only")
        assert result.is_eligible is True

    def test_probability_monotonic_with_rank(self):
        """Better rank should always give higher probability."""
        from app.engine.risk_classifier import compute_admission_probability
        closing = 1000
        probs = [
            compute_admission_probability(r, 500, closing)
            for r in [200, 500, 800, 900, 1000, 1100]
        ]
        assert probs == sorted(probs, reverse=True), "Probability must decrease as rank worsens"

    def test_classify_risk_thresholds(self):
        from app.engine.risk_classifier import classify_risk
        assert classify_risk(0.95) == "Very Safe"
        assert classify_risk(0.80) == "Safe"
        assert classify_risk(0.60) == "Target"
        assert classify_risk(0.30) == "Dream"
        assert classify_risk(0.10) == "Filtered"


# ── Scorer tests ──────────────────────────────────────────────────────────────

class TestScorer:
    def test_score_returns_none_for_ineligible(self, cse_student):
        from app.engine.scorer import score_recommendation
        row = {
            "institute": "IIT Bombay", "branch": "Computer Science and Engineering",
            "program": "B.Tech CSE", "category": "OPEN",
            "gender": "Gender-Neutral", "opening_rank": 1, "closing_rank": 66,
            "round": 6, "year": 2025, "exam_type": "JEE_ADVANCED", "state_quota": "AI",
        }
        # Rank 2000 vs closing 66 — should be filtered
        result = score_recommendation(cse_student, row)
        assert result is None

    def test_score_returns_recommendation_for_eligible(self, cse_student):
        from app.engine.scorer import score_recommendation
        row = {
            "institute": "IIT Ropar", "branch": "Computer Science and Engineering",
            "program": "B.Tech CSE", "category": "OPEN",
            "gender": "Gender-Neutral", "opening_rank": 1500, "closing_rank": 3000,
            "round": 6, "year": 2025, "exam_type": "JEE_ADVANCED", "state_quota": "AI",
        }
        result = score_recommendation(cse_student, row)
        assert result is not None
        assert 0 < result.scores.overall <= 1.0

    def test_score_weights_sum_to_one(self):
        from app.engine.scorer import WEIGHTS
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Weights should sum to 1.0, got {total}"

    def test_cse_branch_scores_higher_for_cse_student(self, cse_student):
        from app.engine.scorer import score_recommendation
        cse_row = {
            "institute": "IIT Ropar", "branch": "Computer Science and Engineering",
            "program": "B.Tech CSE", "category": "OPEN",
            "gender": "Gender-Neutral", "opening_rank": 1500, "closing_rank": 3000,
            "round": 6, "year": 2025, "exam_type": "JEE_ADVANCED", "state_quota": "AI",
        }
        mech_row = {**cse_row, "branch": "Mechanical Engineering"}
        cse_rec  = score_recommendation(cse_student, cse_row)
        mech_rec = score_recommendation(cse_student, mech_row)
        assert cse_rec is not None and mech_rec is not None
        assert cse_rec.scores.interest_match > mech_rec.scores.interest_match

    def test_home_state_advantage_detected(self):
        from app.engine.scorer import score_recommendation
        student = StudentProfile(
            jee_main_rank=15000, gender="female", category="OPEN",
            home_state="Karnataka",
            interest_coding=0.8, interest_ai_ml=0.6, interest_research=0.4,
            interest_core_engineering=0.2, interest_electronics=0.5,
            interest_mechanical=0.1, interest_civil=0.1, interest_chemical=0.1,
        )
        row = {
            "institute": "NIT Surathkal", "branch": "Computer Science and Engineering",
            "program": "B.Tech CSE", "category": "OPEN",
            "gender": "Gender-Neutral", "opening_rank": 10000, "closing_rank": 20000,
            "round": 6, "year": 2025, "exam_type": "JEE_ADVANCED", "state_quota": "HS",
        }
        result = score_recommendation(student, row)
        # Home state bonus should be positive for Karnataka student at NIT Surathkal
        assert result is not None
        assert result.scores.home_state_bonus > 0 or result.home_state_advantage


# ── Persona tests ─────────────────────────────────────────────────────────────

class TestPersona:
    def test_startup_student_gets_entrepreneur(self, cse_student):
        from app.engine.persona import infer_persona
        persona = infer_persona(cse_student)
        assert persona.id in ("entrepreneur", "software_engineer", "ai_researcher")

    def test_research_student_gets_research_persona(self, research_student):
        from app.engine.persona import infer_persona
        persona = infer_persona(research_student)
        assert persona.id in ("ai_researcher", "higher_studies_abroad")

    def test_govt_student_gets_govt_persona(self, govt_student):
        from app.engine.persona import infer_persona
        persona = infer_persona(govt_student)
        assert persona.id in ("govt_psu_aspirant", "core_engineer")

    def test_undecided_gets_undecided(self):
        from app.engine.persona import infer_persona
        s = StudentProfile(
            jee_advanced_rank=10000, gender="male", category="OPEN", home_state="Bihar",
            interest_coding=0.5, interest_ai_ml=0.5, interest_research=0.5,
            interest_core_engineering=0.5, interest_electronics=0.5,
            interest_mechanical=0.5, interest_civil=0.5, interest_chemical=0.5,
        )
        persona = infer_persona(s)
        assert persona.id == "undecided_explorer"

    def test_confidence_between_0_and_1(self, cse_student):
        from app.engine.persona import infer_persona
        persona = infer_persona(cse_student)
        assert 0.0 <= persona.confidence <= 1.0

    def test_all_persona_scores_returns_9_items(self, cse_student):
        from app.engine.persona import get_all_persona_scores
        scores = get_all_persona_scores(cse_student)
        assert len(scores) == 9

    def test_persona_has_counselor_fields(self, cse_student):
        from app.engine.persona import infer_persona
        persona = infer_persona(cse_student)
        assert persona.counselor_opener
        assert persona.branch_advice
        assert persona.institute_advice
        assert persona.career_horizon


# ── Compatibility tests ───────────────────────────────────────────────────────

class TestCompatibility:
    def _make_risk(self, prob: float):
        from app.engine.risk_classifier import RiskResult, classify_risk
        return RiskResult(
            admission_probability=prob,
            risk_level=classify_risk(prob),
            rank_gap=-500, rank_gap_pct=-50.0,
            is_eligible=True,
            safety_margin="Well ahead",
            probability_label=f"~{int(prob*100)}% chance",
            counselor_note="Good choice",
        )

    def test_all_dimensions_between_0_and_1(self, cse_student):
        from app.engine.compatibility import compute_compatibility
        risk = self._make_risk(0.85)
        comp = compute_compatibility(
            student=cse_student, risk=risk,
            branch_name="Computer Science and Engineering",
            branch_domain="CS", suits_goals=["startup"],
            median_lpa=25.0, avg_salary_lpa=32.0,
            coding_intensity=5, research_scope=4,
            inst_type="IIT", inst_tier=2, inst_city="Ropar", inst_state="Punjab",
            nirf_rank=None, inst_research_score=3, inst_coding_score=4,
            inst_placement_lpa=18.0, home_state_advantage=False, flexibility_score=0.95,
        )
        for dim in ["admission_probability", "career_match", "research_match",
                    "coding_match", "lifestyle_match", "future_growth",
                    "placement_strength", "institute_reputation"]:
            val = getattr(comp, dim)
            assert 0.0 <= val <= 1.0, f"{dim}={val} out of range"

    def test_home_state_badge_present(self, cse_student):
        from app.engine.compatibility import compute_compatibility
        risk = self._make_risk(0.85)
        comp = compute_compatibility(
            student=cse_student, risk=risk,
            branch_name="CSE", branch_domain="CS", suits_goals=[],
            median_lpa=15.0, avg_salary_lpa=18.0,
            coding_intensity=5, research_scope=3,
            inst_type="NIT", inst_tier=3, inst_city="Nagpur", inst_state="Maharashtra",
            nirf_rank=None, inst_research_score=2, inst_coding_score=3,
            inst_placement_lpa=10.0, home_state_advantage=True, flexibility_score=0.9,
        )
        assert "🏠 Home State Advantage" in comp.badges

    def test_overall_is_mean_of_8_dims(self, cse_student):
        from app.engine.compatibility import compute_compatibility
        risk = self._make_risk(0.75)
        comp = compute_compatibility(
            student=cse_student, risk=risk,
            branch_name="CSE", branch_domain="CS", suits_goals=[],
            median_lpa=15.0, avg_salary_lpa=18.0,
            coding_intensity=5, research_scope=3,
            inst_type="NIT", inst_tier=3, inst_city="X", inst_state="Y",
            nirf_rank=None, inst_research_score=2, inst_coding_score=3,
            inst_placement_lpa=10.0, home_state_advantage=False, flexibility_score=0.9,
        )
        dims = [comp.admission_probability, comp.career_match, comp.research_match,
                comp.coding_match, comp.lifestyle_match, comp.future_growth,
                comp.placement_strength, comp.institute_reputation]
        expected = round(sum(dims) / len(dims), 4)
        assert abs(comp.overall_compatibility - expected) < 0.001


# ── KnowledgeLoader tests ─────────────────────────────────────────────────────

class TestKnowledgeLoader:
    def test_all_files_loaded(self):
        from app.services.knowledge_loader import KnowledgeLoader
        kl = KnowledgeLoader()
        stats = kl.stats()
        assert stats["files_loaded"] == stats["total_files"], \
            f"Some files failed to load: {stats}"

    def test_branch_exact_match(self):
        from app.services.knowledge_loader import KnowledgeLoader
        kl = KnowledgeLoader()
        result = kl.get_branch("Computer Science and Engineering")
        assert result.get("median_lpa", 0) > 0

    def test_institute_fuzzy_full_name(self):
        from app.services.knowledge_loader import KnowledgeLoader
        kl = KnowledgeLoader()
        r1 = kl.get_institute("IIT Bombay")
        r2 = kl.get_institute("Indian Institute of Technology Bombay")
        assert r1 == r2, "Short and full name should resolve to same result"

    def test_nit_full_name_resolves(self):
        from app.services.knowledge_loader import KnowledgeLoader
        kl = KnowledgeLoader()
        result = kl.get_institute("National Institute of Technology Tiruchirappalli")
        assert result.get("tier") == 3

    def test_scoring_config_has_weights(self):
        from app.services.knowledge_loader import KnowledgeLoader
        kl = KnowledgeLoader()
        weights = kl.scorer_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_faq_search_returns_results(self):
        from app.services.knowledge_loader import KnowledgeLoader
        kl = KnowledgeLoader()
        results = kl.search_faq("branch")
        assert len(results) > 0

    def test_get_branch_recruiters(self):
        from app.services.knowledge_loader import KnowledgeLoader
        kl = KnowledgeLoader()
        rec = kl.get_branch_recruiters("Computer Science and Engineering")
        assert "Google" in rec.get("tier_1", [])

    def test_rag_context_has_key_fields(self):
        from app.services.knowledge_loader import KnowledgeLoader
        kl = KnowledgeLoader()
        ctx = kl.build_rag_context("IIT Bombay", "Computer Science and Engineering")
        assert ctx["career_paths"], "career_paths should not be empty"
        assert ctx["inst_placement_med"] is not None


# ── RAG Pipeline tests ────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_context_built_for_known_combo(self):
        from app.engine.rag import RAGPipeline
        p = RAGPipeline()
        ctx = p.get_context("IIT Bombay", "Computer Science and Engineering")
        assert ctx.institute == "IIT Bombay"
        assert ctx.career_paths

    def test_context_graceful_for_unknown(self):
        from app.engine.rag import RAGPipeline
        p = RAGPipeline()
        ctx = p.get_context("Some Unknown College", "Carpet Technology")
        assert ctx.institute == "Some Unknown College"
        assert ctx.career_paths == []   # empty, not exception

    def test_context_to_llm_string_non_empty_for_known(self):
        from app.engine.rag import RAGPipeline
        p = RAGPipeline()
        ctx = p.get_context("IIT Madras", "Electrical Engineering")
        s = ctx.to_llm_string()
        assert len(s) > 20

    def test_null_provider_returns_none(self):
        from app.engine.rag import NullProvider
        provider = NullProvider()
        result = provider.generate("any prompt")
        assert result is None


# ── Reason Codes tests ────────────────────────────────────────────────────────

class TestReasonCodes:
    def _make_comp(self, **overrides):
        from app.engine.compatibility import CompatibilityProfile
        defaults = dict(
            admission_probability=0.85, career_match=0.80, research_match=0.70,
            coding_match=0.85, lifestyle_match=0.70, future_growth=0.80,
            placement_strength=0.80, institute_reputation=0.85,
            overall_compatibility=0.80,
        )
        defaults.update(overrides)
        return CompatibilityProfile(**defaults)

    def _make_rec(self, risk_level="Safe", prob=0.85, home_state=False):
        from app.engine.risk_classifier import RiskResult, classify_risk
        from app.engine.scorer import ScoredRecommendation, ScoreBreakdown
        risk = RiskResult(
            admission_probability=prob, risk_level=risk_level,
            rank_gap=-200, rank_gap_pct=-20.0, is_eligible=True,
            safety_margin="", probability_label="", counselor_note="",
        )
        scores = ScoreBreakdown(
            overall=0.75, rank_fit=prob, interest_match=0.85,
            institute_strength=0.80, career_alignment=0.75,
            home_state_bonus=1.0 if home_state else 0.0, flexibility=0.85,
        )
        rec = ScoredRecommendation(
            institute="IIT Test", branch="CSE",
            institute_type="IIT", scores=scores, risk=risk,
            home_state_advantage=home_state,
            research_score=4, coding_culture_score=4,
            placement_median_lpa=20.0,
        )
        return rec

    def test_rank_match_code_present_for_high_prob(self, cse_student):
        from app.engine.reason_codes import compute_reason_codes
        rec  = self._make_rec(prob=0.90)
        comp = self._make_comp(admission_probability=0.90)
        codes = compute_reason_codes(cse_student, rec, comp)
        assert "RANK_MATCH" in codes.codes

    def test_home_state_code_present(self, cse_student):
        from app.engine.reason_codes import compute_reason_codes
        rec  = self._make_rec(home_state=True)
        comp = self._make_comp()
        codes = compute_reason_codes(cse_student, rec, comp)
        assert "HOME_STATE" in codes.codes

    def test_stretch_rank_warning_for_dream(self, cse_student):
        from app.engine.reason_codes import compute_reason_codes
        rec  = self._make_rec(risk_level="Dream", prob=0.30)
        comp = self._make_comp(admission_probability=0.30)
        codes = compute_reason_codes(cse_student, rec, comp)
        assert any("Stretch" in w or "Beyond" in w for w in codes.warnings)

    def test_no_duplicate_codes(self, cse_student):
        from app.engine.reason_codes import compute_reason_codes
        rec  = self._make_rec()
        comp = self._make_comp()
        codes = compute_reason_codes(cse_student, rec, comp)
        assert len(codes.codes) == len(set(codes.codes)), "Duplicate codes found"
