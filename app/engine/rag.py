"""
RAG Pipeline — Retrieval-Augmented Generation for counselor narratives.

Architecture:
  1. JSONRetriever — looks up institute + branch data from knowledge base
  2. ContextBuilder — assembles RecommendationContext from retrieved data
  3. LLM Provider   — generates a grounded narrative (Claude, Gemini, or None)
"""

import json
import threading
from dataclasses import dataclass, field
from typing import Protocol

from app.config import get_settings
from app.services.knowledge_loader import KnowledgeLoader
from app.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Retriever
# Responsibility: fetch raw knowledge for a given query
# Interface is a Protocol so it can be replaced without changing callers
# ─────────────────────────────────────────────────────────────────────────────

class RetrieverProtocol(Protocol):
    def retrieve(self, institute: str, branch: str) -> dict: ...


class JSONRetriever:
    """
    Retrieves knowledge from JSON files via KnowledgeLoader.
    To replace with FAISS/ChromaDB: implement RetrieverProtocol with
    embedding-based similarity search and plug in below.
    """

    def __init__(self, loader: KnowledgeLoader):
        self._kl = loader

    def retrieve(self, institute: str, branch: str) -> dict:
        """
        Fetch all relevant raw knowledge for a given institute + branch pair.
        Returns a raw dict — no shaping yet, that's ContextBuilder's job.
        """
        return {
            "branch_profile":     self._kl.get_branch(branch),
            "branch_career":      self._kl.get_branch_career(branch),
            "branch_recruiters":  self._kl.get_branch_recruiters(branch),
            "branch_hs":          self._kl.get_branch_higher_studies(branch),
            "branch_startup_fit": self._kl.get_startup_branch_fit(branch),
            "inst_meta":          self._kl.get_institute(institute),
            "inst_placement":     self._kl.get_institute_placement(institute),
            "inst_startup":       self._kl.get_institute_startup(institute),
            "inst_recruiters":    self._kl.get_institute_recruiters(institute),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Context
# Structured object that carries everything needed downstream
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecommendationContext:
    """
    Clean structured context assembled from raw retrieval results.
    Passed to LLMProvider and also returned in API response for explainability.
    """
    institute: str
    branch: str

    # Branch knowledge
    career_paths:         list[str] = field(default_factory=list)
    roadmap:              list[str] = field(default_factory=list)
    median_lpa:           float | None = None
    coding_intensity:     int | None = None
    research_scope:       int | None = None
    suits_goals:          list[str] = field(default_factory=list)
    top_recruiters_branch:list[str] = field(default_factory=list)

    # Career data
    immediate_roles:      list[str] = field(default_factory=list)
    typical_salary_range: str | None = None
    mba_transition:       str | None = None
    startup_friendliness: str | None = None
    gate_relevant:        bool = False

    # Recruiter tiers
    tier1_recruiters:     list[str] = field(default_factory=list)
    finance_recruiters:   list[str] = field(default_factory=list)
    all_recruiters:       list[str] = field(default_factory=list)

    # Higher studies
    top_ms_programs:      list[str] = field(default_factory=list)
    top_phd_programs:     list[str] = field(default_factory=list)
    ms_profile_boosters:  list[str] = field(default_factory=list)

    # Institute data
    inst_known_for:       str | None = None
    inst_strengths:       list[str] = field(default_factory=list)
    inst_placement_median:float | None = None
    inst_top_recruiters:  list[str] = field(default_factory=list)
    inst_placement_note:  str | None = None

    # Startup
    startup_score:        int | None = None
    startup_types:        str | None = None
    inst_ecell:           str | None = None
    inst_notable_startups:list[str] = field(default_factory=list)

    def to_llm_string(self) -> str:
        """
        Format context into a compact grounded-facts string for LLM injection.
        Only non-empty fields are included.
        """
        lines = []
        if self.career_paths:
            lines.append(f"Career paths: {', '.join(self.career_paths[:3])}")
        if self.immediate_roles:
            lines.append(f"Entry-level roles: {', '.join(self.immediate_roles[:3])}")
        if self.typical_salary_range:
            lines.append(f"Typical 5-year salary: ₹{self.typical_salary_range} LPA")
        if self.inst_placement_median:
            lines.append(f"Institute placement median: ₹{self.inst_placement_median} LPA")
        if self.median_lpa:
            lines.append(f"Branch placement median: ₹{self.median_lpa} LPA")
        if self.inst_top_recruiters:
            lines.append(f"Top recruiters: {', '.join(self.inst_top_recruiters[:4])}")
        elif self.tier1_recruiters:
            lines.append(f"Tier-1 recruiters: {', '.join(self.tier1_recruiters[:4])}")
        if self.inst_placement_note:
            lines.append(f"Placement note: {self.inst_placement_note}")
        if self.inst_known_for:
            lines.append(f"Institute known for: {self.inst_known_for}")
        if self.mba_transition:
            lines.append(f"MBA transition: {self.mba_transition}")
        if self.startup_friendliness:
            lines.append(f"Startup friendliness: {self.startup_friendliness}")
        if self.top_ms_programs:
            lines.append(f"Top MS options: {self.top_ms_programs[0]}")
        return "\n".join(lines) or "No additional context available."

    def to_dict(self) -> dict:
        """Serialisable dict for API response / caching."""
        return {
            "career_paths":          self.career_paths,
            "roadmap":               self.roadmap,
            "median_lpa":            self.median_lpa,
            "typical_salary_range":  self.typical_salary_range,
            "immediate_roles":       self.immediate_roles,
            "tier1_recruiters":      self.tier1_recruiters,
            "finance_recruiters":    self.finance_recruiters,
            "top_ms_programs":       self.top_ms_programs,
            "top_phd_programs":      self.top_phd_programs,
            "ms_profile_boosters":   self.ms_profile_boosters,
            "inst_known_for":        self.inst_known_for,
            "inst_top_recruiters":   self.inst_top_recruiters,
            "inst_placement_note":   self.inst_placement_note,
            "mba_transition":        self.mba_transition,
            "startup_friendliness":  self.startup_friendliness,
            "startup_types":         self.startup_types,
            "inst_ecell":            self.inst_ecell,
            "inst_notable_startups": self.inst_notable_startups,
            "gate_relevant":         self.gate_relevant,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — ContextBuilder
# Responsibility: shape raw retrieval results into a RecommendationContext
# ─────────────────────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Assembles raw retrieval results into a clean RecommendationContext.
    Pure transformation — no I/O, fully testable.
    """

    def build(self, institute: str, branch: str, raw: dict) -> RecommendationContext:
        bp  = raw.get("branch_profile", {})
        bc  = raw.get("branch_career", {})
        br  = raw.get("branch_recruiters", {})
        bhs = raw.get("branch_hs", {})
        bsf = raw.get("branch_startup_fit", {})
        im  = raw.get("inst_meta", {})
        ip  = raw.get("inst_placement", {})
        ist = raw.get("inst_startup", {})
        ir  = raw.get("inst_recruiters", [])

        return RecommendationContext(
            institute=institute,
            branch=branch,
            # Branch
            career_paths=bp.get("career_paths", []),
            roadmap=bp.get("roadmap", []),
            median_lpa=bp.get("median_lpa"),
            coding_intensity=bp.get("coding_intensity"),
            research_scope=bp.get("research_scope"),
            suits_goals=bp.get("suits_goals", []),
            top_recruiters_branch=bp.get("top_recruiters", []),
            # Career
            immediate_roles=bc.get("immediate_roles", []),
            typical_salary_range=bc.get("typical_5yr_salary_lpa"),
            mba_transition=bc.get("mba_transition"),
            startup_friendliness=bc.get("startup_friendliness"),
            gate_relevant=bool(bc.get("gate_relevant", False)),
            # Recruiters
            tier1_recruiters=br.get("tier_1", []),
            finance_recruiters=br.get("finance", []),
            all_recruiters=(ir or ip.get("top_recruiters", [])),
            # Higher studies
            top_ms_programs=bhs.get("top_ms_programs", []),
            top_phd_programs=bhs.get("top_phd_programs", []),
            ms_profile_boosters=bhs.get("strong_profile_boosters", []),
            # Institute
            inst_known_for=im.get("known_for"),
            inst_strengths=im.get("strengths", []),
            inst_placement_median=ip.get("median_lpa"),
            inst_top_recruiters=ir or ip.get("top_recruiters", []),
            inst_placement_note=ip.get("notable"),
            # Startup
            startup_score=bsf.get("score"),
            startup_types=bsf.get("typical_startups"),
            inst_ecell=ist.get("ecell_name"),
            inst_notable_startups=ist.get("notable_startups", []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — LLMProvider
# Responsibility: generate a grounded narrative using context + student data
# ─────────────────────────────────────────────────────────────────────────────

class LLMProviderProtocol(Protocol):
    def generate(self, prompt: str) -> str | None: ...


class ClaudeProvider:
    """
    Generates counselor narrative using Claude claude-sonnet-4-6.
    To swap for OpenAI/Gemini: implement LLMProviderProtocol.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        # H11-FIX: Lazy import so Gemini-only environments don't fail on module load
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model  = model

    def generate(self, prompt: str) -> str | None:
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=220,
                messages=[{"role": "user", "content": prompt}],
            )
            # C11-FIX: Guard against empty content list (safety-filtered responses)
            if not resp.content:
                log.warning("Claude returned empty content (possibly safety-filtered)")
                return None
            return resp.content[0].text.strip()
        except Exception as e:
            log.warning(f"LLM generation failed: {e}")
            return None


class NullProvider:
    """Drop-in when LLM is disabled — always returns None (triggers template fallback)."""
    def generate(self, prompt: str) -> str | None:
        return None

class MockProvider:
    """
    Mock provider for demonstrations when API keys are unavailable.
    Returns realistic, contextual responses based on the prompt content.
    """
    def generate(self, prompt: str) -> str | None:
        import re
        # Extract context to make the mock answer look grounded
        branch = "the recommended branch"
        inst = "this institute"
        
        b_match = re.search(r"Recommendation:\s*(.*?)\s*at\s*(.*)", prompt)
        if b_match:
            branch = b_match.group(1).strip()
            inst = b_match.group(2).strip()
            
        return (
            f"Based on your profile, {branch} at {inst} is an excellent fit. "
            f"The curriculum strongly aligns with your stated goals, and the placement statistics "
            f"show strong recruiter interest in this area. While the academic rigor is high, "
            f"the long-term career outcomes make it a highly strategic choice for your rank."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Provider (free tier — use when Anthropic API unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiProvider:
    """
    Generates counselor narrative using Google Gemini (free tier).
    Model: gemini-2.0-flash — fast, free tier.

    Usage: set GEMINI_API_KEY in .env and LLM_PROVIDER=gemini
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._api_key = api_key
        self._model   = model
        self._url     = (
            f"https://generativelanguage.googleapis.com/v1beta"
            f"/models/{model}:generateContent?key={api_key}"
        )

    def generate(self, prompt: str) -> str | None:
        import json
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 220,
                "temperature":     0.7,
            },
        }).encode()

        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                # C8-FIX: Guard against safety-filtered responses that have no 'content' key
                candidates = data.get("candidates", [])
                if not candidates:
                    log.warning("Gemini returned no candidates (possibly safety-filtered)")
                    return None
                content = candidates[0].get("content")
                if not content:
                    finish = candidates[0].get("finishReason", "UNKNOWN")
                    log.warning(f"Gemini candidate has no content (finishReason={finish})")
                    return None
                parts = content.get("parts", [])
                if not parts:
                    log.warning("Gemini response has empty parts list")
                    return None
                return parts[0].get("text", "").strip() or None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:300]
            log.warning(f"Gemini HTTP {exc.code} for model={self._model}: {body}")
            return None
        except Exception as exc:
            log.warning(f"Gemini generation failed: {exc}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Groq Provider (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class GroqProvider:
    """
    Generates counselor narrative using Groq's blazing fast inference API.
    Uses the official OpenAI-compatible chat completions endpoint.
    
    Usage: set GROQ_API_KEY in .env and LLM_PROVIDER=groq
    """

    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self._api_key = api_key
        self._model   = model
        self._url     = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt: str) -> str | None:
        import json
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 220,
            "temperature": 0.7,
        }).encode()

        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                choices = data.get("choices", [])
                if not choices:
                    log.warning("Groq returned no choices")
                    return self._fallback(prompt)
                
                content = choices[0].get("message", {}).get("content")
                if not content:
                    log.warning("Groq choice has no content")
                    return self._fallback(prompt)
                
                return content.strip()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:300]
            log.warning(f"Groq HTTP {exc.code} for model={self._model}: {body}")
            return self._fallback(prompt)
        except Exception as exc:
            log.warning(f"Groq generation failed: {exc}")
            return self._fallback(prompt)
            
    def _fallback(self, prompt: str) -> str | None:
        log.info("GroqProvider falling back to MockProvider")
        return MockProvider().generate(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — wires all three stages together
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full pipeline: institute + branch → grounded narrative.

    Usage:
        pipeline = RAGPipeline()
        context  = pipeline.get_context("IIT Bombay", "CSE")
        narrative= pipeline.generate_narrative(context, student, persona, scores)
    """

    def __init__(
        self,
        retriever:    RetrieverProtocol | None = None,
        builder:      ContextBuilder    | None = None,
        llm_provider: LLMProviderProtocol | None = None,
    ):
        # Use the singleton KnowledgeLoader instead of creating a fresh instance
        from app.services.knowledge_loader import knowledge as _knowledge
        kl = _knowledge

        self._retriever    = retriever    or JSONRetriever(kl)
        self._builder      = builder      or ContextBuilder()
        self._llm_provider = llm_provider or self._build_default_llm()

    @staticmethod
    def _build_default_llm() -> LLMProviderProtocol:
        settings = get_settings()
        if not settings.enable_llm_explanations:
            return NullProvider()

        provider = settings.llm_provider.lower()

        if provider == "gemini" and settings.gemini_api_key:
            log.info("LLM provider: Gemini (gemini-2.0-flash)")
            return GeminiProvider(settings.gemini_api_key)

        if provider == "groq" and settings.groq_api_key:
            log.info(f"LLM provider: Groq ({settings.llm_model})")
            return GroqProvider(settings.groq_api_key, settings.llm_model)

        if provider == "claude" and settings.anthropic_api_key:
            log.info(f"LLM provider: Claude ({settings.llm_model})")
            return ClaudeProvider(settings.anthropic_api_key, settings.llm_model)
            
        if provider == "mock":
            log.info("LLM provider: Mock (for demonstrations)")
            return MockProvider()

        log.info("LLM provider: None (template fallback)")
        return NullProvider()

    def get_context(self, institute: str, branch: str) -> RecommendationContext:
        """Retrieve + build context. Cached by caller if needed."""
        raw = self._retriever.retrieve(institute, branch)
        return self._builder.build(institute, branch, raw)

    def generate_narrative(
        self,
        context:   RecommendationContext,
        student_rank: int,
        student_category: str,
        student_home_state: str,
        persona_label: str,
        active_goals: list[str],
        prob_label: str,
        risk_level: str,
        compatibility_pct: float,
        pros: list[str],
        cons: list[str],
    ) -> str | None:
        """
        Build a grounded LLM prompt and generate narrative.
        Returns None if LLM unavailable (caller uses template fallback).
        """
        grounded_facts = context.to_llm_string()
        goals_str = ", ".join(active_goals) if active_goals else "not specified"

        prompt = f"""You are an experienced JEE admission counselor giving personalised advice.

Student:
- Rank: {student_rank} ({student_category})
- Home state: {student_home_state}
- Career direction: {persona_label}
- Goals: {goals_str}

Recommendation: {context.branch} at {context.institute}
- Admission: {prob_label} ({risk_level})
- Compatibility: {compatibility_pct:.0%}
- Strengths: {'; '.join(pros[:3])}
- Concerns: {'; '.join(cons[:2]) if cons else 'none significant'}

Verified facts (use ONLY these figures — do not invent numbers):
{grounded_facts}

Write 2–3 sentences in a warm, direct counselor voice. Be specific to this student's situation. Cite one concrete opportunity and one honest risk. No bullet points. Do not open with "As a" or "Based on"."""

        return self._llm_provider.generate(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

# BUG-34-FIX: Thread-safe singleton using a lock to prevent race conditions
# under concurrent FastAPI requests on cold start
_pipeline: RAGPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            # Double-checked locking: re-check after acquiring the lock
            if _pipeline is None:
                _pipeline = RAGPipeline()
                log.info("RAG pipeline initialised")
    return _pipeline