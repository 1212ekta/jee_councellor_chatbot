FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    pydantic pydantic-settings \
    pandas openpyxl numpy \
    duckdb \
    anthropic \
    diskcache \
    httpx

# Copy app source
COPY app/ ./app/
COPY data/knowledge/ ./data/knowledge/
# C7-FIX: Copy frontend directory so UI is served in Docker
COPY frontend/ ./frontend/

# Create data dirs (cutoffs mounted as volume)
RUN mkdir -p data/cutoffs data/cache

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    DATA_DIR=/app/data \
    CUTOFFS_DIR=/app/data/cutoffs \
    DB_PATH=/app/data/jee_counselor.duckdb \
    CACHE_DIR=/app/data/cache \
    KNOWLEDGE_DIR=/app/data/knowledge

EXPOSE 8000

# H10-FIX: Increased start-period to 120s to allow time for first-run ETL
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
