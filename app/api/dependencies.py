"""
Shared FastAPI dependencies.
Injected via Depends() into route handlers.
"""

import duckdb
from fastapi import Depends

from app.engine.rag import RAGPipeline, get_pipeline
from app.etl.loader import get_db
from app.services.knowledge_loader import KnowledgeLoader, knowledge
from app.utils.logger import get_logger

log = get_logger(__name__)


def db() -> duckdb.DuckDBPyConnection:
    """DuckDB connection dependency."""
    return get_db()


def knowledge_loader() -> KnowledgeLoader:
    """KnowledgeLoader singleton dependency."""
    return knowledge


def rag_pipeline() -> RAGPipeline:
    """RAG pipeline singleton dependency."""
    return get_pipeline()
