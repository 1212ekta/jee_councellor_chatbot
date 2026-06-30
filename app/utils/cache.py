import hashlib
import json
from functools import wraps
from typing import Any, Callable

import diskcache

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
_cache: diskcache.Cache | None = None


def get_cache() -> diskcache.Cache:
    """Returns (and lazily initializes) the disk cache singleton."""
    global _cache
    if _cache is None:
        settings = get_settings()
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(settings.cache_dir))
        log.info(f"Cache initialized at {settings.cache_dir}")
    return _cache


def make_key(*args, **kwargs) -> str:
    """Deterministic cache key from any args/kwargs."""
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def cached(ttl: int | None = None) -> Callable:
    """
    Decorator that caches a function's return value to disk.

    Usage:
        @cached(ttl=3600)
        def expensive_query(rank: int, category: str):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            settings = get_settings()
            effective_ttl = ttl or settings.cache_ttl_seconds
            key = f"{fn.__module__}.{fn.__name__}:{make_key(*args, **kwargs)}"
            cache = get_cache()

            result = cache.get(key)
            if result is not None:
                log.debug(f"Cache HIT: {fn.__name__}")
                return result

            log.debug(f"Cache MISS: {fn.__name__}")
            result = fn(*args, **kwargs)
            cache.set(key, result, expire=effective_ttl)
            return result

        return wrapper
    return decorator


def clear_cache() -> int:
    """Clears all cached entries. Returns number of items cleared."""
    cache = get_cache()
    count = len(cache)
    cache.clear()
    log.info(f"Cache cleared: {count} entries removed")
    return count
