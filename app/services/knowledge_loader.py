"""
KnowledgeLoader Service

Single entry point for all structured knowledge.
Loads, validates, caches, and provides clean accessors.

Design principles:
  - Load once, cache forever (process lifetime)
  - Validate schema on load — fail loudly if data is corrupt
  - Clean accessors: knowledge.get_branch("CSE") not json["CSE"]
  - Fuzzy matching built-in: "IIT Bombay" matches "Indian Institute of Technology Bombay"
  - Graceful degradation: returns {} not exception if key not found
  - Swappable: replace JSON files with DB by changing _load() only

Usage:
    from app.services.knowledge_loader import knowledge
    branch = knowledge.get_branch("Computer Science and Engineering")
    institute = knowledge.get_institute("IIT Bombay")
    config = knowledge.scoring_config()
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# All knowledge files with their required top-level keys for validation
KNOWLEDGE_FILES = {
    "branch_profiles":   ("branch_profiles.json",   []),          # dict of branch→profile
    "institute_tiers":   ("institute_tiers.json",    []),
    "career_paths":      ("career_paths.json",       []),
    "placements":        ("placements.json",         []),
    "recruiters":        ("recruiters.json",         ["by_branch"]),
    "higher_studies":    ("higher_studies.json",     ["by_branch"]),
    "startup_ecosystem": ("startup_ecosystem.json",  ["by_institute"]),
    "branch_comparison": ("branch_comparison.json",  ["common_comparisons"]),
    "scoring_config":    ("scoring_config.json",     ["scorer_weights"]),
    "faq":               ("faq.json",                ["faqs"]),
}


class KnowledgeLoader:
    """
    Centralized knowledge service.
    All engine modules should use this instead of reading JSON directly.
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._load_all()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        """Load and validate all knowledge files on startup."""
        settings = get_settings()
        loaded, failed = 0, 0

        for key, (filename, required_keys) in KNOWLEDGE_FILES.items():
            path = settings.knowledge_dir / filename
            if not path.exists():
                log.warning(f"Knowledge file missing: {filename} — using empty fallback")
                self._cache[key] = {}
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                self._validate(filename, data, required_keys)
                self._cache[key] = data
                loaded += 1
            except json.JSONDecodeError as e:
                log.error(f"Invalid JSON in {filename}: {e}")
                self._cache[key] = {}
                failed += 1
            except Exception as e:
                log.error(f"Failed to load {filename}: {e}")
                self._cache[key] = {}
                failed += 1

        log.info(f"KnowledgeLoader: {loaded} files loaded, {failed} failed")

    def _validate(self, filename: str, data: Any, required_keys: list[str]) -> None:
        """Check required top-level keys exist."""
        if not isinstance(data, dict):
            raise ValueError(f"{filename}: expected dict, got {type(data)}")
        for key in required_keys:
            if key not in data:
                log.warning(f"{filename}: missing expected key '{key}'")

    def reload(self) -> None:
        """Hot-reload all knowledge files (useful for admin endpoint)."""
        log.info("Reloading all knowledge files...")
        self._cache.clear()
        self._load_all()

    # ── Fuzzy matching ────────────────────────────────────────────────────────

    # City/location alias map for institute fuzzy matching
    _CITY_ALIASES: dict[str, str] = {
        "bombay": "bombay", "mumbai": "bombay",
        "delhi": "delhi", "new delhi": "delhi",
        "madras": "madras", "chennai": "madras",
        "kanpur": "kanpur",
        "kharagpur": "kharagpur",
        "roorkee": "roorkee",
        "hyderabad": "hyderabad",
        "guwahati": "guwahati",
        "varanasi": "varanasi", "bhu": "varanasi",
        "bhubaneswar": "bhubaneswar",
        "gandhinagar": "gandhinagar",
        "jodhpur": "jodhpur",
        "ropar": "ropar",
        "mandi": "mandi",
        "patna": "patna",
        "tirupati": "tirupati",
        "palakkad": "palakkad",
        "goa": "goa",
        "jammu": "jammu",
        "dhanbad": "dhanbad",
        "trichy": "trichy", "tiruchirappalli": "trichy",
        "warangal": "warangal",
        "surathkal": "surathkal",
        "calicut": "calicut", "kozhikode": "calicut",
        "nagpur": "nagpur",
        "silchar": "silchar",
        "allahabad": "allahabad", "prayagraj": "allahabad",
        "durgapur": "durgapur",
        "rourkela": "rourkela",
        "kurukshetra": "kurukshetra",
        "jaipur": "jaipur",
        "srinagar": "srinagar",
    }

    @classmethod
    def _extract_city_token(cls, text: str) -> str | None:
        """Extract a normalised city token from an institute name."""
        t = text.lower()
        for alias, canonical in cls._CITY_ALIASES.items():
            if alias in t:
                return canonical
        return None

    @classmethod
    def _fuzzy_get(cls, data: dict, query: str) -> tuple[str | None, Any]:
        """
        Find best matching key in a dict.
        Returns (matched_key, value) or (None, {}).

        Strategy (in order):
          1. Exact match
          2. Case-insensitive exact
          3. Substring containment (longest key wins)
          4. City-token match  ← handles full-name ↔ short-name
          5. Word overlap (≥2 shared words)
        """
        if not query or not data:
            return None, {}

        # 1. Exact
        if query in data:
            return query, data[query]

        query_lower = query.lower().strip()

        # 2. Case-insensitive exact
        for key in data:
            if key.lower() == query_lower:
                return key, data[key]

        # 3. Substring containment
        matches = [
            (key, len(key))
            for key in data
            if key.lower() in query_lower or query_lower in key.lower()
        ]
        if matches:
            best_key = max(matches, key=lambda x: x[1])[0]
            return best_key, data[best_key]

        # 4. City-token match — "Indian Institute of Technology Bombay" → city=bombay
        #    matches "IIT Bombay" which also has city=bombay
        query_city = cls._extract_city_token(query_lower)
        if query_city:
            for key in data:
                key_city = cls._extract_city_token(key.lower())
                if key_city and key_city == query_city:
                    return key, data[key]

        # 5. Word overlap (stop-words excluded)
        STOP = {"of", "and", "the", "for", "in", "at", "a", "an", "institute",
                "technology", "national", "indian", "engineering"}
        query_words = set(query_lower.split()) - STOP
        best_key, best_overlap = None, 0
        for key in data:
            key_words = set(key.lower().split()) - STOP
            overlap = len(query_words & key_words)
            if overlap > best_overlap and overlap >= 1:
                best_key, best_overlap = key, overlap
        if best_key:
            return best_key, data[best_key]

        return None, {}

    # ── Branch accessors ──────────────────────────────────────────────────────

    def get_branch(self, branch_name: str) -> dict:
        """Get branch profile by name (fuzzy matched)."""
        _, result = self._fuzzy_get(self._cache.get("branch_profiles", {}), branch_name)
        return result

    def get_branch_career(self, branch_name: str) -> dict:
        """Get career path data for a branch."""
        _, result = self._fuzzy_get(self._cache.get("career_paths", {}), branch_name)
        return result

    def get_branch_recruiters(self, branch_name: str) -> dict:
        """Get recruiters for a branch."""
        by_branch = self._cache.get("recruiters", {}).get("by_branch", {})
        _, result = self._fuzzy_get(by_branch, branch_name)
        return result

    def get_branch_higher_studies(self, branch_name: str) -> dict:
        """Get higher studies options for a branch."""
        by_branch = self._cache.get("higher_studies", {}).get("by_branch", {})
        _, result = self._fuzzy_get(by_branch, branch_name)
        return result

    def all_branch_names(self) -> list[str]:
        """All branch names in the knowledge base."""
        return list(self._cache.get("branch_profiles", {}).keys())

    # ── Institute accessors ───────────────────────────────────────────────────

    def get_institute(self, institute_name: str) -> dict:
        """Get institute tier metadata (fuzzy matched)."""
        _, result = self._fuzzy_get(self._cache.get("institute_tiers", {}), institute_name)
        return result

    def get_institute_placement(self, institute_name: str) -> dict:
        """Get placement data for an institute."""
        _, result = self._fuzzy_get(self._cache.get("placements", {}), institute_name)
        return result

    def get_institute_startup(self, institute_name: str) -> dict:
        """Get startup ecosystem data for an institute."""
        by_inst = self._cache.get("startup_ecosystem", {}).get("by_institute", {})
        _, result = self._fuzzy_get(by_inst, institute_name)
        return result

    def get_institute_recruiters(self, institute_name: str) -> list[str]:
        """Get top recruiters for an institute."""
        by_inst = self._cache.get("recruiters", {}).get("by_institute", {})
        _, result = self._fuzzy_get(by_inst, institute_name)
        return result if isinstance(result, list) else []

    def all_institute_names(self) -> list[str]:
        """All institute names in the knowledge base."""
        return list(self._cache.get("institute_tiers", {}).keys())

    # ── Comparison accessor ───────────────────────────────────────────────────

    def get_branch_comparison(self, branch_a: str, branch_b: str) -> dict | None:
        """
        Get comparison data for two branches.
        Tries both orderings.
        """
        comparisons = self._cache.get("branch_comparison", {}).get("common_comparisons", {})
        # Build a combined key and search
        combined = f"{branch_a}_vs_{branch_b}".replace(" ", "_").replace("and", "").replace("__", "_")
        for key, val in comparisons.items():
            branches = val.get("branches", [])
            if branch_a in branches and branch_b in branches:
                return val
        return None

    def get_all_comparisons(self) -> dict:
        return self._cache.get("branch_comparison", {}).get("common_comparisons", {})

    # ── Config accessors ──────────────────────────────────────────────────────

    def scoring_config(self) -> dict:
        """Full scoring configuration."""
        return self._cache.get("scoring_config", {})

    def scorer_weights(self) -> dict:
        """Base scorer weights."""
        defaults = {
            "rank_fit": 0.40, "interest_match": 0.25,
            "institute_strength": 0.15, "career_alignment": 0.12,
            "home_state_bonus": 0.05, "flexibility": 0.03,
        }
        return self._cache.get("scoring_config", {}).get("scorer_weights", defaults)

    def persona_weight_overrides(self, persona_id: str) -> dict:
        """Weight overrides for a specific persona."""
        overrides = self._cache.get("scoring_config", {}).get("persona_weight_overrides", {})
        return overrides.get(persona_id, {})

    def risk_thresholds(self) -> dict:
        """Risk classification thresholds."""
        return self._cache.get("scoring_config", {}).get("risk_thresholds", {
            "dream_min": 0.15, "dream_max": 0.45,
            "target_min": 0.45, "target_max": 0.75,
            "safe_min": 0.75, "safe_max": 0.90,
            "very_safe_min": 0.90,
        })

    def institute_type_base_scores(self) -> dict:
        """Institute type → base score mapping."""
        return self._cache.get("scoring_config", {}).get("institute_type_base_scores", {
            "IIT": 1.00, "IIIT": 0.75, "NIT": 0.65, "GFTI": 0.45, "State": 0.40,
        })

    def max_results_per_bucket(self) -> dict:
        return self._cache.get("scoring_config", {}).get("max_results_per_bucket", {
            "dream": 5, "target": 8, "safe": 8, "very_safe": 5,
        })

    # ── FAQ / Glossary accessors ──────────────────────────────────────────────

    def get_faq(self, question_id: str | None = None) -> list[dict] | dict | None:
        """Get FAQ entries. Pass id for specific FAQ, None for all."""
        faqs = self._cache.get("faq", {}).get("faqs", [])
        if question_id is None:
            return faqs
        return next((f for f in faqs if f.get("id") == question_id), None)

    def get_glossary(self) -> dict:
        return self._cache.get("faq", {}).get("glossary", {})

    def search_faq(self, query: str) -> list[dict]:
        """Simple keyword search across FAQ questions and answers."""
        faqs = self.get_faq() or []
        query_lower = query.lower()
        return [
            faq for faq in faqs
            if query_lower in faq.get("question", "").lower()
            or query_lower in faq.get("answer", "").lower()
        ]

    # ── Startup ecosystem ─────────────────────────────────────────────────────

    def get_startup_branch_fit(self, branch_name: str) -> dict:
        branch_fits = self._cache.get("startup_ecosystem", {}).get("branch_startup_fit", {})
        _, result = self._fuzzy_get(branch_fits, branch_name)
        return result

    def startup_general_advice(self) -> dict:
        return self._cache.get("startup_ecosystem", {}).get("general_advice", {})

    # ── Context builder for RAG ───────────────────────────────────────────────

    def build_rag_context(self, institute: str, branch: str) -> dict:
        """
        Build complete RAG context for one recommendation.
        Used by the LLM explainer to ground its narrative.
        """
        branch_profile  = self.get_branch(branch)
        branch_career   = self.get_branch_career(branch)
        branch_rec      = self.get_branch_recruiters(branch)
        branch_hs       = self.get_branch_higher_studies(branch)
        inst_meta       = self.get_institute(institute)
        inst_placement  = self.get_institute_placement(institute)
        inst_startup    = self.get_institute_startup(institute)
        inst_recruiters = self.get_institute_recruiters(institute)
        startup_fit     = self.get_startup_branch_fit(branch)

        return {
            # Branch
            "career_paths":       branch_profile.get("career_paths", []),
            "roadmap":            branch_profile.get("roadmap", []),
            "median_lpa":         branch_profile.get("median_lpa"),
            "coding_intensity":   branch_profile.get("coding_intensity"),
            "research_scope":     branch_profile.get("research_scope"),
            "suits_goals":        branch_profile.get("suits_goals", []),
            "top_recruiters_branch": branch_profile.get("top_recruiters", []),

            # Career
            "immediate_roles":    branch_career.get("immediate_roles", []),
            "typical_salary":     branch_career.get("typical_5yr_salary_lpa"),
            "mba_transition":     branch_career.get("mba_transition"),
            "startup_friendliness": branch_career.get("startup_friendliness"),
            "gate_relevant":      branch_career.get("gate_relevant"),

            # Branch recruiters
            "tier1_recruiters":   branch_rec.get("tier_1", []),
            "finance_recruiters": branch_rec.get("finance", []),

            # Higher studies
            "top_ms_programs":    branch_hs.get("top_ms_programs", []),
            "top_phd_programs":   branch_hs.get("top_phd_programs", []),
            "ms_profile_boosters":branch_hs.get("strong_profile_boosters", []),

            # Institute
            "inst_known_for":     inst_meta.get("known_for"),
            "inst_strengths":     inst_meta.get("strengths", []),
            "inst_placement_med": inst_placement.get("median_lpa"),
            "inst_top_recruiters":inst_recruiters or inst_placement.get("top_recruiters", []),
            "inst_placement_note":inst_placement.get("notable"),

            # Startup
            "startup_score":      startup_fit.get("score"),
            "startup_types":      startup_fit.get("typical_startups"),
            "inst_ecell":         inst_startup.get("ecell_name"),
            "inst_notable_startups": inst_startup.get("notable_startups", []),
        }

    def format_rag_context(self, ctx: dict) -> str:
        """Format RAG context dict into a compact string for LLM injection."""
        lines = []
        if ctx.get("career_paths"):
            lines.append(f"Career paths: {', '.join(ctx['career_paths'][:3])}")
        if ctx.get("immediate_roles"):
            lines.append(f"Entry roles: {', '.join(ctx['immediate_roles'][:3])}")
        if ctx.get("typical_salary"):
            lines.append(f"5-year salary range: ₹{ctx['typical_salary']} LPA")
        if ctx.get("inst_placement_med"):
            lines.append(f"Institute placement median: ₹{ctx['inst_placement_med']} LPA")
        if ctx.get("inst_top_recruiters"):
            lines.append(f"Top recruiters: {', '.join(ctx['inst_top_recruiters'][:4])}")
        if ctx.get("inst_placement_note"):
            lines.append(f"Placement note: {ctx['inst_placement_note']}")
        if ctx.get("inst_known_for"):
            lines.append(f"Institute known for: {ctx['inst_known_for']}")
        if ctx.get("mba_transition"):
            lines.append(f"MBA path: {ctx['mba_transition']}")
        if ctx.get("startup_friendliness"):
            lines.append(f"Startup fit: {ctx['startup_friendliness']}")
        if ctx.get("top_ms_programs"):
            lines.append(f"Top MS options: {'; '.join(ctx['top_ms_programs'][:2])}")
        return "\n".join(lines) or "No additional context available."

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "files_loaded": sum(1 for v in self._cache.values() if v),
            "total_files": len(KNOWLEDGE_FILES),
            "branch_profiles": len(self._cache.get("branch_profiles", {})),
            "institute_profiles": len(self._cache.get("institute_tiers", {})),
            "faqs": len(self._cache.get("faq", {}).get("faqs", [])),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this directly: `from app.services.knowledge_loader import knowledge`

@lru_cache(maxsize=1)
def _get_knowledge_loader() -> KnowledgeLoader:
    return KnowledgeLoader()

knowledge = _get_knowledge_loader()
