"""
Subtext Configuration Management

Centralized configuration using pydantic-settings with environment variable support.
Integrates with Enclii secrets management.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ══════════════════════════════════════════════════════════════
    # Application
    # ══════════════════════════════════════════════════════════════
    app_name: str = "Subtext"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    api_version: str = "v1"
    cors_origins: list[str] = ["http://localhost:3000", "https://subtext.live"]

    # ══════════════════════════════════════════════════════════════
    # Database (PostgreSQL + TimescaleDB)
    # ══════════════════════════════════════════════════════════════
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://subtext:subtext@localhost:5432/subtext"
    )
    database_pool_size: int = 20
    database_max_overflow: int = 10
    database_echo: bool = False

    # ══════════════════════════════════════════════════════════════
    # Redis
    # ══════════════════════════════════════════════════════════════
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_job_queue_db: int = 1
    redis_cache_db: int = 2

    # ══════════════════════════════════════════════════════════════
    # Object Storage (S3/R2)
    # ══════════════════════════════════════════════════════════════
    s3_endpoint: str = "https://s3.amazonaws.com"
    s3_bucket: str = "subtext-audio"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region: str = "us-east-1"

    # ══════════════════════════════════════════════════════════════
    # Janua Authentication (https://github.com/madfam-io/janua)
    # ══════════════════════════════════════════════════════════════
    janua_base_url: str = "https://auth.madfam.io"
    janua_client_id: str = ""
    janua_client_secret: str = ""
    janua_audience: str = "https://api.subtext.live"
    janua_algorithms: list[str] = ["RS256"]

    # ══════════════════════════════════════════════════════════════
    # Stripe Billing
    # ══════════════════════════════════════════════════════════════
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_publishable_key: str = ""

    # Product IDs (configured in Stripe Dashboard)
    stripe_product_personal: str = "prod_personal"
    stripe_product_teams: str = "prod_teams"
    stripe_product_enterprise: str = "prod_enterprise"
    stripe_product_api: str = "prod_api"

    # Price IDs
    stripe_price_personal_monthly: str = "price_personal_monthly"
    stripe_price_teams_monthly: str = "price_teams_monthly"
    stripe_price_api_per_minute: str = "price_api_per_minute"

    # ══════════════════════════════════════════════════════════════
    # Resend Email
    # ══════════════════════════════════════════════════════════════
    resend_api_key: str = ""
    resend_from_email: str = "Subtext <hello@subtext.live>"
    resend_reply_to: str = "support@subtext.live"

    # ══════════════════════════════════════════════════════════════
    # ML Models
    # ══════════════════════════════════════════════════════════════
    model_cache_dir: str = "/models"
    whisper_model: str = "large-v3"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    wav2vec_model: str = "facebook/wav2vec2-large-xlsr-53"
    deepfilternet_model: str = "deepfilternet3"

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "ollama", "local"] = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gpt-4-turbo-preview"

    # ══════════════════════════════════════════════════════════════
    # Pipeline Configuration
    # ══════════════════════════════════════════════════════════════
    pipeline_max_audio_duration_seconds: int = 7200  # 2 hours
    pipeline_max_file_size_mb: int = 500
    pipeline_chunk_duration_ms: int = 30000  # 30 seconds
    pipeline_worker_concurrency: int = 3
    signal_confidence_threshold: float = 0.5

    # ══════════════════════════════════════════════════════════════
    # Rate Limiting
    # ══════════════════════════════════════════════════════════════
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 20

    # ══════════════════════════════════════════════════════════════
    # Feature Flags
    # ══════════════════════════════════════════════════════════════
    feature_voice_fingerprinting: bool = False
    feature_realtime_streaming: bool = True
    feature_esp_protocol: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def async_database_url(self) -> str:
        """Get async database URL for SQLAlchemy."""
        url = str(self.database_url)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://")
        return url


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()
