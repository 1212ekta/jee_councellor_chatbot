"""GET /health — system status check."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import db, knowledge_loader, rag_pipeline
from app.config import get_settings
from app.utils.cache import get_cache

router = APIRouter()


@router.get("/health")
def health(
    conn=Depends(db),
    kl=Depends(knowledge_loader),
):
    settings = get_settings()

    # DB check
    try:
        row = conn.execute("SELECT COUNT(*), COUNT(DISTINCT year) FROM cutoffs").fetchone()
        total_rows, year_count = row
        years = [r[0] for r in conn.execute(
            "SELECT DISTINCT year FROM cutoffs ORDER BY year"
        ).fetchall()]
        db_ok = True
    except Exception:
        total_rows, year_count, years, db_ok = 0, 0, [], False

    # Cache check
    try:
        cache = get_cache()
        cache_entries = len(cache)
    except Exception:
        cache_entries = -1

    kb_stats = kl.stats()

    is_ok = db_ok and total_rows > 0

    # H12-FIX: Check either LLM provider (Gemini or Claude)
    llm_enabled = (
        settings.enable_llm_explanations
        and (
            bool(settings.anthropic_api_key)
            or bool(settings.gemini_api_key)
        )
    )

    payload = {
        "status":           "ok" if is_ok else "degraded",
        "version":          "1.0.0",
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "db_connected":     db_ok,
        "cutoffs_loaded":   total_rows > 0,
        "cutoff_years":     years,
        "total_cutoff_rows":total_rows,
        "llm_enabled":      llm_enabled,
        "llm_provider":     settings.llm_provider,
        "llm_model":        settings.llm_model,
        "cache_entries":    cache_entries,
        "knowledge_base":   kb_stats,
    }

    # H1-FIX: Return 503 when degraded so Docker HEALTHCHECK curl -f actually fails
    if not is_ok:
        return JSONResponse(status_code=503, content=payload)
    return payload
