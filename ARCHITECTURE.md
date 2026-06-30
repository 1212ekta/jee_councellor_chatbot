# Architecture — JEE AI Counselor

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│          Browser  /  Mobile  /  API Consumer                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Layer                               │
│   /recommend  /analyze-profile  /compare  /counselor/chat        │
│   /institutes  /branches  /cutoffs  /health  /sessions           │
└───────┬─────────────┬──────────────┬───────────────────────────┘
        │             │              │
┌───────▼──────┐ ┌────▼──────┐ ┌────▼──────────┐
│  Intelligence │ │Knowledge  │ │   Data Layer  │
│    Engine     │ │  Loader   │ │    DuckDB     │
│               │ │(10 JSONs) │ │  2410 rows    │
│ ┌───────────┐ │ └────┬──────┘ └────┬──────────┘
│ │  Persona  │ │      │             │
│ │  Scorer   │ │ ┌────▼──────────── ▼──────────┐
│ │  Risk     │ │ │      RAG Pipeline             │
│ │  Compat   │ │ │  Retriever → Builder → LLM   │
│ │  Explain  │ │ └──────────────────────────────┘
│ └───────────┘ │
└───────────────┘
```

---

## Folder Structure

```
jee-counselor/
├── app/
│   ├── main.py                    # FastAPI app + lifespan + exception handlers
│   ├── config.py                  # pydantic-settings, reads .env
│   ├── exceptions.py              # Custom exception hierarchy
│   │
│   ├── api/
│   │   ├── dependencies.py        # FastAPI DI: db(), knowledge_loader(), rag_pipeline()
│   │   └── routes/
│   │       ├── recommend.py       # POST /recommend, /analyze-profile, /sessions/{id}
│   │       ├── counselor.py       # POST /counselor/chat
│   │       ├── compare.py         # GET /compare/branches|institutes
│   │       ├── institutes.py      # GET /institutes, /institutes/{name}/placement
│   │       ├── branches.py        # GET /branches, /branches/{name}/details
│   │       ├── cutoffs.py         # GET /cutoffs, /cutoffs/stats
│   │       └── health.py          # GET /health
│   │
│   ├── engine/
│   │   ├── interest_matcher.py    # Cosine similarity, 8-dim interest vectors
│   │   ├── risk_classifier.py     # Sigmoid probability → Dream/Target/Safe/Very Safe
│   │   ├── scorer.py              # 6-factor weighted scoring, institute metadata cache
│   │   ├── persona.py             # 9 career personas, rule-based inference
│   │   ├── compatibility.py       # 8-dimension radar chart + badge generation
│   │   ├── reason_codes.py        # Machine-readable signals (RANK_MATCH, HOME_STATE…)
│   │   ├── explainer.py           # Structured explanation assembly (RAG + optional LLM)
│   │   └── rag.py                 # JSONRetriever → ContextBuilder → LLMProvider pipeline
│   │
│   ├── etl/
│   │   ├── loader.py              # Excel → DuckDB (idempotent, auto year/exam detection)
│   │   ├── cleaner.py             # Column normalization, gender/category standardization
│   │   └── schema.py              # DuckDB DDL: cutoffs, institutes, branches, sessions
│   │
│   ├── models/
│   │   ├── request.py             # StudentProfile (Pydantic v2, validators, computed props)
│   │   └── response.py            # API response schemas
│   │
│   ├── services/
│   │   └── knowledge_loader.py    # Centralised JSON access, city-token fuzzy matching
│   │
│   └── utils/
│       ├── logger.py              # Structured named logger, log-level from config
│       └── cache.py               # diskcache TTL wrapper, @cached decorator
│
├── data/
│   ├── cutoffs/                   # Drop xlsx here → auto-loaded on restart
│   └── knowledge/                 # RAG knowledge base (10 JSON files)
│
├── tests/
│   ├── test_engine.py             # 53 unit tests: all engine modules
│   ├── test_etl.py                # ETL + cleaner + edge cases
│   └── test_api.py                # 45 API integration tests via TestClient
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── render.yaml                    # Render PaaS deployment config
└── README.md
```

---

## Recommendation Pipeline (Sequence)

```
POST /recommend
      │
      ▼
1. StudentProfile validated (Pydantic)
      │
      ▼
2. infer_persona(student)
   ├── Score 9 personas (rule-based weighted functions)
   ├── Return primary + secondary persona
   └── Persona drives weight_overrides + counselor narrative style
      │
      ▼
3. _fetch_eligible_rows(student, db)
   ├── DuckDB query: closing_rank <= rank * 1.35 (floor 5000)
   ├── Gender filter: male students excluded from Female-Only seats
   └── Exam type filter: Advanced / Main / Both
      │
      ▼
4. score_all(student, rows)
   For each row:
   ├── assess_risk()         → sigmoid probability, risk level
   ├── compute_interest()    → cosine similarity on 8-dim vectors
   ├── _score_institute()    → type + NIRF + placement composite
   ├── _score_career()       → goal matching against suits_goals
   ├── _score_home_state()   → HS quota detection
   ├── _score_flexibility()  → branch optionality heuristic
   └── weighted sum → ScoredRecommendation
      │
      ▼
5. bucket_scored(scored)
   ├── dream:     15–45% probability
   ├── target:    45–75% probability
   ├── safe:      75–90% probability
   └── very_safe: 90%+  probability
      │
      ▼
6. For top N per bucket:
   ├── compute_compatibility() → 8-dim CompatibilityProfile + badges
   ├── compute_reason_codes()  → RANK_MATCH, HOME_STATE, STARTUP…
   └── explain()
       ├── RAGPipeline.get_context()  → retrieve + build context
       ├── _why_institute()           → deterministic from scores
       ├── _why_branch()              → deterministic from scores
       ├── _build_pros/cons()         → from reason codes + badges
       ├── _build_risks()             → personalised warnings
       └── LLMProvider.generate()    → optional Claude narrative (RAG-grounded)
      │
      ▼
7. Build response + save session
   └── Return JSON with session_id, share_url, all buckets
```

---

## Database Schema (DuckDB)

```sql
-- Primary cutoff data (loaded from xlsx)
CREATE TABLE cutoffs (
    id            INTEGER PRIMARY KEY,
    year          INTEGER NOT NULL,        -- 2025, 2024, ...
    round         INTEGER,                 -- JoSAA round (1-6); 6 = final
    institute     VARCHAR NOT NULL,        -- Full name e.g. "Indian Institute of Technology Bombay"
    program       VARCHAR,                 -- Full program string from xlsx
    branch        VARCHAR NOT NULL,        -- Extracted e.g. "Computer Science and Engineering"
    category      VARCHAR NOT NULL,        -- OPEN, OBC-NCL, SC, ST, EWS, OPEN-PwD
    gender        VARCHAR,                 -- Gender-Neutral, Female-Only
    opening_rank  INTEGER,
    closing_rank  INTEGER NOT NULL,
    exam_type     VARCHAR NOT NULL,        -- JEE_ADVANCED, JEE_MAIN
    state_quota   VARCHAR,                 -- AI (All India), HS (Home State), OS (Other State)
    seat_type     VARCHAR DEFAULT 'REGULAR',
    loaded_at     TIMESTAMP DEFAULT current_timestamp
);

-- Institute metadata (from knowledge/institute_tiers.json)
CREATE TABLE institutes (
    id                    INTEGER PRIMARY KEY,
    name                  VARCHAR UNIQUE NOT NULL,
    short_name            VARCHAR,
    type                  VARCHAR,          -- IIT, NIT, IIIT, GFTI, State
    city                  VARCHAR,
    state                 VARCHAR,
    tier                  INTEGER,          -- 1=top IIT, 2=new IIT, 3=NIT, 4=IIIT, 5=GFTI
    nirf_rank             INTEGER,
    research_score        FLOAT,            -- 1-5
    placement_median_lpa  FLOAT,
    coding_culture_score  FLOAT,            -- 1-5
    strengths             VARCHAR,          -- JSON array
    known_for             VARCHAR
);

-- Branch metadata (from knowledge/branch_profiles.json)
CREATE TABLE branches (
    id                INTEGER PRIMARY KEY,
    name              VARCHAR UNIQUE NOT NULL,
    short_name        VARCHAR,
    domain            VARCHAR,             -- CS, EE, ECE, MECH, CIVIL, CHEM, EP
    career_paths      VARCHAR,             -- JSON array
    coding_intensity  INTEGER,             -- 1-5
    research_scope    INTEGER,             -- 1-5
    median_lpa        FLOAT,
    avg_salary_lpa    FLOAT,
    suits_goals       VARCHAR              -- JSON array
);

-- Session persistence (for share links)
CREATE TABLE sessions (
    id          VARCHAR PRIMARY KEY,        -- UUID
    input_json  VARCHAR NOT NULL,
    output_json VARCHAR NOT NULL,
    created_at  TIMESTAMP DEFAULT current_timestamp
);

-- Indexes (critical for rank-range query performance)
CREATE INDEX idx_cutoffs_rank   ON cutoffs (closing_rank, opening_rank);
CREATE INDEX idx_cutoffs_inst   ON cutoffs (institute);
CREATE INDEX idx_cutoffs_branch ON cutoffs (branch);
CREATE INDEX idx_cutoffs_cat    ON cutoffs (category);
CREATE INDEX idx_cutoffs_year   ON cutoffs (year);
CREATE INDEX idx_cutoffs_exam   ON cutoffs (exam_type);
```

---

## Knowledge Base Design

```
data/knowledge/
├── branch_profiles.json      # 8 canonical branch profiles
│                             # Each: interest_vector (8-dim), career_paths,
│                             # roadmap, median_lpa, coding_intensity, research_scope
│
├── institute_tiers.json      # 16 institutes with tier, NIRF, placement, known_for
│
├── career_paths.json         # Role progression by branch: immediate → senior
│                             # gate_relevant, mba_transition, startup_friendliness
│
├── placements.json           # Institute-specific: median_lpa, top_recruiters, notable
│
├── recruiters.json           # By branch (tier1/tier2/tier3/finance) + by institute
│
├── higher_studies.json       # MS/PhD programs by branch, stipends, CGPA expectations
│
├── startup_ecosystem.json    # E-cell rating, notable startups, branch startup fit
│
├── branch_comparison.json    # Pre-built verdicts: CSE vs ECE, Mech vs Civil, etc.
│
├── scoring_config.json       # All weights and thresholds — tune without code changes
│
└── faq.json                  # 10 canonical FAQs + full glossary (JoSAA, LPA, etc.)
```

**Design principle:** Each file is independently replaceable. Adding 2026 placement data means only updating `placements.json`. Tuning recommendation weights means only editing `scoring_config.json`.

---

## RAG Architecture

```
Recommendation Request
        │
        ▼
┌───────────────────┐
│  JSONRetriever    │  ← implements RetrieverProtocol
│  .retrieve(       │    swap for FAISSRetriever / ChromaRetriever
│    institute,     │    without changing anything downstream
│    branch         │
│  )                │
└────────┬──────────┘
         │ raw dict (8 knowledge sections)
         ▼
┌───────────────────┐
│  ContextBuilder   │  ← pure transformation, fully testable
│  .build(          │    shapes raw retrieval into RecommendationContext
│    raw            │    dataclass with to_llm_string() + to_dict()
│  )                │
└────────┬──────────┘
         │ RecommendationContext
         ▼
┌───────────────────┐
│  LLMProvider      │  ← implements LLMProviderProtocol
│                   │    ClaudeProvider (default)
│  .generate(       │    NullProvider (when disabled)
│    prompt         │    swap for OpenAIProvider, GeminiProvider
│  )                │
└────────┬──────────┘
         │ narrative string | None
         ▼
  StructuredExplanation
  (LLM narrative or template fallback)
```

**Swap cost:**
- JSON → ChromaDB: implement `RetrieverProtocol`, change one line in `RAGPipeline.__init__`
- Claude → OpenAI: implement `LLMProviderProtocol`, change one line in `RAGPipeline.__init__`

---

## API Flow

```
Client                FastAPI              Engine              DuckDB / LLM
  │                      │                   │                      │
  │  POST /recommend     │                   │                      │
  │─────────────────────>│                   │                      │
  │                      │  validate()       │                      │
  │                      │──────────────────>│                      │
  │                      │  infer_persona()  │                      │
  │                      │──────────────────>│                      │
  │                      │                   │  SELECT cutoffs      │
  │                      │                   │─────────────────────>│
  │                      │                   │  2410 rows           │
  │                      │                   │<─────────────────────│
  │                      │                   │  score_all()         │
  │                      │                   │  bucket_scored()     │
  │                      │                   │  compatibility()     │
  │                      │                   │  explain() + RAG     │
  │                      │                   │  [optional LLM call] │
  │                      │                   │─────────────────────>│
  │                      │                   │  narrative           │
  │                      │                   │<─────────────────────│
  │                      │  INSERT session   │                      │
  │                      │─────────────────────────────────────────>│
  │  JSON response        │                   │                      │
  │<─────────────────────│                   │                      │
```

---

## Scoring Weights

All weights live in `data/knowledge/scoring_config.json`:

```json
{
  "scorer_weights": {
    "rank_fit":            0.40,
    "interest_match":      0.25,
    "institute_strength":  0.15,
    "career_alignment":    0.12,
    "home_state_bonus":    0.05,
    "flexibility":         0.03
  }
}
```

**Rationale:**
- `rank_fit` (40%) — if the student can't get in, nothing else matters
- `interest_match` (25%) — wrong branch = 4 years of misery regardless of rank
- `institute_strength` (15%) — brand + placements + research quality
- `career_alignment` (12%) — does this branch serve stated goals?
- `home_state_bonus` (5%) — real but secondary advantage
- `flexibility` (3%) — future optionality, tie-breaker

---

## Deployment Architecture

```
┌────────────────────────────────────────┐
│            Render / Railway            │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │         Docker Container         │  │
│  │                                  │  │
│  │  uvicorn app.main:app            │  │
│  │  --host 0.0.0.0 --port $PORT     │  │
│  │                                  │  │
│  │  /app/data/  (persistent disk)   │  │
│  │  ├── cutoffs/    ← xlsx files    │  │
│  │  ├── knowledge/  ← JSON KB       │  │
│  │  └── jee_counselor.duckdb        │  │
│  └──────────────────────────────────┘  │
│                                        │
│  Environment variables:                │
│    ANTHROPIC_API_KEY                   │
│    ENABLE_LLM_EXPLANATIONS=true        │
└────────────────────────────────────────┘
```

---

## Performance Characteristics

| Operation | Target | Actual (no LLM) |
|---|---|---|
| `POST /recommend` (full) | < 500ms | ~180ms |
| `POST /analyze-profile` | < 100ms | ~40ms |
| `GET /health` | < 50ms | ~5ms |
| DuckDB rank query (2410 rows) | < 50ms | ~8ms |
| Interest matching (241 branches) | < 20ms | ~3ms |
| Knowledge base load (startup) | < 2s | ~0.3s |
| ETL (2410 rows xlsx) | < 5s | ~1.2s |

With LLM enabled, add ~800ms per LLM call (only top N recommendations get LLM narrative).

---

## Future Improvements

1. **Multi-year cutoff trends** — load 2022–2025 data, show rank trend graphs
2. **Vector DB** — replace JSON retriever with ChromaDB for semantic FAQ search
3. **Branch change probability** — model CGPA requirements + seat availability
4. **CSAB round support** — separate cutoffs for CSAB special rounds
5. **JEE Main NTA score** — convert percentile → rank → recommendations
6. **Real-time cutoff updates** — GitHub Actions auto-pulls new JoSAA files
7. **PDF report** — ReportLab-based counseling report download
8. **Collaborative filtering** — "students like you chose…" based on session history
9. **Mobile app** — React Native frontend using the same API
10. **GATE/PSU recommender** — separate recommendation flow for post-BTech paths
