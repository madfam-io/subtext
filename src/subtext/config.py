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
    # ML Models - Best-in-Class Open Source (2025)
    # See MODEL_STACK.md for detailed rationale
    # ══════════════════════════════════════════════════════════════
    model_cache_dir: str = "/models"

    # Noise Suppression: DeepFilterNet (still SOTA for real-time)
    # Alternatives: GT-CRN, ULCNet (for edge)
    deepfilternet_model: str = "deepfilternet3"

    # Voice Activity Detection: Silero VAD (87.7% TPR vs WebRTC's 50%)
    # GitHub: snakers4/silero-vad
    silero_vad_model: str = "silero_vad"

    # Speaker Diarization: Pyannote 4.0 Community-1 (best open-source)
    # 10% DER, 2.5% RTF on GPU
    # Alternative: NeMo Sortformer (for NVIDIA production)
    pyannote_model: str = "pyannote/speaker-diarization-3.1"  # Update to 4.0 when released
    nemo_diarization_model: str = "nvidia/diar_sortformer_4spk-v1"

    # ASR/Transcription: Multiple options by use case
    # - Accuracy: NVIDIA Canary Qwen 2.5B (5.63% WER - SOTA)
    # - Speed: NVIDIA Parakeet TDT (2000+ RTFx)
    # - Multilingual: Whisper large-v3 (40+ languages)
    asr_model: str = "openai/whisper-large-v3"  # Default: multilingual
    asr_model_accuracy: str = "nvidia/canary-1b"  # Best English accuracy
    asr_model_speed: str = "nvidia/parakeet-tdt-1.1b"  # Fastest throughput
    whisper_model: str = "large-v3"  # Legacy compatibility

    # Speech Emotion Recognition: Emotion2Vec (purpose-built, SOTA on 9 datasets)
    # GitHub: ddlBoJack/emotion2vec
    # Alternatives: WavLM (better for noisy speech), DistilHuBERT (lightweight)
    emotion_model: str = "iic/emotion2vec_plus_large"
    emotion_model_lite: str = "iic/emotion2vec_plus_base"
    wav2vec_model: str = "facebook/wav2vec2-large-xlsr-53"  # Legacy fallback

    # Speaker Embedding: ECAPA-TDNN (1.71% EER - best accuracy)
    # Alternative: TitaNet (NVIDIA NeMo, 1.91% EER)
    speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_embedding_nemo: str = "nvidia/speakerverification_en_titanet_large"

    # LLM Configuration - Open Source Options
    # SOTA Open: Llama 3.1 70B, Mixtral 8x22B, Qwen2.5 72B
    llm_provider: Literal["openai", "anthropic", "ollama", "vllm", "local"] = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8000"
    llm_model: str = "gpt-4-turbo-preview"  # Cloud default
    llm_model_local: str = "meta-llama/Llama-3.1-70B-Instruct"  # Self-hosted SOTA
    llm_model_fast: str = "meta-llama/Llama-3.1-8B-Instruct"  # Fast local option

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
    # Worker Configuration (ARQ)
    # ══════════════════════════════════════════════════════════════
    worker_max_jobs: int = 10
    worker_job_timeout: int = 3600  # 1 hour max per job
    worker_preload_models: bool = True
    worker_concurrency: int = 3

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
