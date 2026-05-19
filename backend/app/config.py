"""Application configuration via pydantic-settings."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # OpenRouter AI
    openrouter_api_key: str = ""
    ai_review_model: str = "anthropic/claude-sonnet-4-6"

    # Storage paths
    data_dir: Path = Path("../data")
    uploads_dir: Path = Path("../data/uploads")
    artifacts_dir: Path = Path("../data/artifacts")
    results_dir: Path = Path("../data/results")
    db_path: Path = Path("../data/jobs.sqlite")

    # Storage cleanup
    cleanup_enabled: bool = True           # run automatic cleanup on startup + every 24 h
    cleanup_max_age_days: int = 30         # remove upload files for jobs older than this
    cleanup_uploads_only: bool = True      # if True, only raw uploads are removed (artifacts/results kept)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for path in [self.uploads_dir, self.artifacts_dir, self.results_dir]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
