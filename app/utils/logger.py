import logging
import sys
from functools import lru_cache

from app.config import get_settings


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@lru_cache
def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger. Call this at the top of every module:
        log = get_logger(__name__)
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured, don't double-add handlers

    logger.setLevel(settings.log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False

    return logger
