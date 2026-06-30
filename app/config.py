from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Data paths
    data_dir: Path = Path("./data")
    cutoffs_dir: Path = Path("./data/cutoffs")
    db_path: Path = Path("./data/jee_counselor.duckdb")
    sessions_db_path: Path = Path("./data/sessions.db")
    knowledge_dir: Path = Path("./data/knowledge")

    # Cache
    cache_dir: Path = Path("./data/cache")
    cache_ttl_seconds: int = 3600

    # AI
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    llm_provider: str = "gemini"   # "gemini" | "claude" | "groq" | "mock" | "none"
    enable_llm_explanations: bool = True
    llm_model: str = "llama3-8b-8192"  # default groq model, or claude-sonnet-4-6

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def ensure_dirs(self) -> None:
        """Create all required data directories if they don't exist."""
        for d in [
            self.data_dir,
            self.cutoffs_dir,
            self.cache_dir,
            self.knowledge_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
