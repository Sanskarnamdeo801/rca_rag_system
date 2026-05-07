"""Minimal application configuration for the RCA MVP."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Environment-driven settings with sensible defaults for local development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "RCA RAG MVP"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    data_dir: Path = Field(default=PROJECT_ROOT / "app" / "data")
    incidents_path: Path = Field(default=PROJECT_ROOT / "app" / "data" / "incidents.json")
    faiss_dir: Path = Field(default=PROJECT_ROOT / "app" / "data" / "faiss_index")
    faiss_index_path: Path = Field(default=PROJECT_ROOT / "app" / "data" / "faiss_index" / "index.faiss")
    embeddings_path: Path = Field(default=PROJECT_ROOT / "app" / "data" / "faiss_index" / "embeddings.npy")
    static_dir: Path = Field(default=PROJECT_ROOT / "app" / "static")
    sample_incidents_path: Path = Field(default=PROJECT_ROOT / "sample_logs" / "incidents.json")

    max_upload_size_mb: int = 10
    top_k_results: int = 5

    embedding_backend: str = "sentence-transformers"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_warmup: bool = True

    llm_provider: str = "llama3"
    llm_timeout_seconds: float = 8.0
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1/chat/completions"
    llama3_model: str = "llama3.1"
    llama3_base_url: str = "http://localhost:11434/api/generate"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings and ensure data directories exist."""

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)
    return settings
