"""
ARIA / Hermes — Central Configuration
All settings loaded from environment / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────
    app_name: str = "ARIA"
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = "change-me"

    @computed_field
    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"

    # ── Database ─────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "hermes"
    postgres_user: str = "hermes"
    postgres_password: str = "hermes_secret"
    database_url: str = (
        "postgresql+asyncpg://hermes:hermes_secret@localhost:5432/hermes"
    )

    @computed_field
    @property
    def sync_database_url(self) -> str:
        """Synchronous URL for Alembic migrations."""
        return self.database_url.replace("+asyncpg", "+psycopg2", 1).replace(
            "+asyncpg", ""
        )

    # ── LLM ──────────────────────────────────────────────────
    llm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    llm_device: str = "cuda"
    llm_torch_dtype: str = "bfloat16"
    llm_load_in_4bit: bool = True
    llm_max_new_tokens: int = 2048
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_cache_dir: str = "./models"

    # ── Embedding ────────────────────────────────────────────
    embed_model_id: str = "nomic-ai/nomic-embed-text-v1.5"
    embed_device: str = "cuda"
    embed_batch_size: int = 32
    embed_dimension: int = 768

    # ── Whisper STT ──────────────────────────────────────────
    whisper_model_size: str = "base"
    whisper_device: str = "cuda"
    whisper_language: str | None = "en"

    # ── Email ────────────────────────────────────────────────
    email_client: Literal["imap", "outlook"] = "imap"
    email_imap_host: str = "imap.gmail.com"
    email_imap_port: int = 993
    email_imap_use_ssl: bool = True
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_smtp_use_tls: bool = True
    email_address: str = ""
    email_password: str = ""
    email_poll_interval_seconds: int = 300

    # ── Agent ────────────────────────────────────────────────
    agent_max_iterations: int = 10
    agent_timeout_seconds: int = 120
    tool_sandbox_timeout_seconds: int = 30
    auto_project_confidence_threshold: float = 0.75

    # ── Scheduler ────────────────────────────────────────────
    weekly_report_day: str = "sunday"
    weekly_report_hour: int = 8
    nightly_refinement_hour: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
