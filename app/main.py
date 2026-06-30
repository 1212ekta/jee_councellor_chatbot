"""
JEE AI Counselor — FastAPI Application Entry Point

Startup sequence:
  1. Ensure data directories exist
  2. Connect to DuckDB (rw or ro fallback)
  3. Run ETL only if cutoffs table is empty
  4. Verify knowledge base loaded
  5. Register all API routers
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.etl.loader import close_db, get_db
from app.exceptions import JEECounselorError, NoCutoffDataError, SessionNotFoundError
from app.services.knowledge_loader import knowledge
from app.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    settings.ensure_dirs()

    log.info("Starting JEE AI Counselor...")

    # Connect DB and run ETL only if needed
    # C5-FIX: Re-raise so the container exits rather than serving 500s on every request
    try:
        conn = get_db()
        existing = conn.execute("SELECT COUNT(*) FROM cutoffs").fetchone()[0]
        if existing > 0:
            log.info(f"Cutoffs already loaded: {existing} rows — skipping ETL")
        else:
            from app.etl.loader import CutoffLoader
            loader  = CutoffLoader()
            results = loader.load_all()
            for fname, rows in results.items():
                if rows >= 0:
                    log.info(f"  ETL: {fname} → {rows} rows")
                else:
                    log.warning(f"  ETL: {fname} → FAILED")
    except Exception as exc:
        log.critical(f"Database startup FATAL error: {exc}")
        raise  # Let the process exit so the container restarts cleanly

    # Verify knowledge base — use .get() to survive schema changes in stats()
    try:
        kb = knowledge.stats()
        log.info(
            f"Knowledge base: {kb.get('files_loaded', '?')}/{kb.get('total_files', '?')} files | "
            f"branches={kb.get('branch_profiles', '?')} | institutes={kb.get('institute_profiles', '?')}"
        )
    except Exception as exc:
        log.warning(f"Knowledge base stats failed (non-fatal): {exc}")

    log.info("Ready to serve requests")
    yield

    log.info("Shutting down...")
    close_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title="JEE AI Counselor",
        description=(
            "An intelligent, explainable JEE college and branch recommendation system. "
            "Combines multi-factor scoring, career persona detection, "
            "RAG-grounded explanations, and optional Claude AI narration.\n\n"
            "**Source:** https://github.com/your-org/jee-counselor\n\n"
            "**Primary dataset:** JoSAA 2025 Final Allotment Cutoffs"
        ),
        version="1.0.0",
        contact={"name": "JEE Counselor", "url": "https://github.com/your-org/jee-counselor"},
        license_info={"name": "MIT"},
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Security headers middleware ──────────────────────────────────────────
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            return response

    app.add_middleware(SecurityHeadersMiddleware)

    # ── CORS ──────────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handlers ─────────────────────────────────────────────
    @app.exception_handler(NoCutoffDataError)
    async def no_cutoff_handler(request: Request, exc: NoCutoffDataError):
        return JSONResponse(status_code=404, content={"error": str(exc), "details": exc.details})

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
        return JSONResponse(status_code=404, content={"error": str(exc), "details": exc.details})

    @app.exception_handler(JEECounselorError)
    async def app_error_handler(request: Request, exc: JEECounselorError):
        log.error(f"Application error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "details": exc.details},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    from app.api.routes import (
        branches, compare, counselor, cutoffs, health, institutes, recommend
    )
    app.include_router(health.router,     tags=["System"])
    app.include_router(recommend.router,  tags=["Recommendations"])
    app.include_router(counselor.router,  tags=["AI Counselor"])
    app.include_router(compare.router,    tags=["Comparison"])
    app.include_router(institutes.router, tags=["Data"])
    app.include_router(branches.router,   tags=["Data"])
    app.include_router(cutoffs.router,    tags=["Data"])

    # Serve frontend at /ui to avoid shadowing /docs, /redoc, and API routes
    # Fix: mounting at "/" would intercept /docs and /redoc in some Starlette versions
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        log.info(f"Frontend served at /ui from {frontend_dir}")
    else:
        log.warning(f"Frontend directory not found at {frontend_dir} — UI not served")

    return app


app = create_app()
