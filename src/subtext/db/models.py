"""
SQLAlchemy ORM Models

Database models for Subtext entities. These map to the PostgreSQL + TimescaleDB
schema and are used for persistence operations.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from subtext.core.models import (
    SessionStatus,
    SignalType,
    SubscriptionTier,
    UserRole,
)
from subtext.db import Base


# ══════════════════════════════════════════════════════════════
# Mixins
# ══════════════════════════════════════════════════════════════


class TimestampMixin:
    """Mixin adding created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=datetime.utcnow,
        nullable=True,
    )


class UUIDMixin:
    """Mixin adding UUID primary key."""

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )


# ══════════════════════════════════════════════════════════════
# Organization & User Models
# ══════════════════════════════════════════════════════════════


class OrganizationModel(Base, UUIDMixin, TimestampMixin):
    """Organization (tenant) in the multi-tenant system."""

    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    subscription_tier: Mapped[SubscriptionTier] = mapped_column(
        SQLEnum(SubscriptionTier),
        default=SubscriptionTier.FREE,
        nullable=False,
    )

    # Stripe integration
    stripe_customer_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Settings and metadata
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    # Usage tracking
    monthly_minutes_limit: Mapped[int] = mapped_column(Integer, default=300, nullable=False)
    monthly_minutes_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    api_calls_limit: Mapped[int] = mapped_column(Integer, default=1000, nullable=False)
    api_calls_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Feature flags
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    # Relationships
    users: Mapped[list["UserModel"]] = relationship("UserModel", back_populates="organization")
    sessions: Mapped[list["SessionModel"]] = relationship("SessionModel", back_populates="organization")

    __table_args__ = (
        Index("ix_organizations_slug", "slug"),
        Index("ix_organizations_stripe_customer", "stripe_customer_id"),
    )


class UserModel(Base, UUIDMixin, TimestampMixin):
    """User account linked to Janua authentication."""

    __tablename__ = "users"

    org_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(
        SQLEnum(UserRole),
        default=UserRole.MEMBER,
        nullable=False,
    )

    # Janua integration
    janua_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Profile
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    # Voice fingerprint (opt-in for speaker recognition)
    voice_fingerprint_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    voice_fingerprint_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)

    # Relationships
    organization: Mapped["OrganizationModel"] = relationship("OrganizationModel", back_populates="users")
    created_sessions: Mapped[list["SessionModel"]] = relationship("SessionModel", back_populates="creator")

    __table_args__ = (
        UniqueConstraint("org_id", "email", name="uq_users_org_email"),
        Index("ix_users_email", "email"),
        Index("ix_users_janua_id", "janua_user_id"),
    )


# ══════════════════════════════════════════════════════════════
# Session & Analysis Models
# ══════════════════════════════════════════════════════════════


class SessionModel(Base, UUIDMixin, TimestampMixin):
    """Audio analysis session."""

    __tablename__ = "sessions"

    org_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_by: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[SessionStatus] = mapped_column(
        SQLEnum(SessionStatus),
        default=SessionStatus.PENDING,
        nullable=False,
    )

    # Audio metadata
    source_type: Mapped[str] = mapped_column(String(50), default="upload", nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Processing results
    speaker_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    signal_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ASR Backend used
    asr_backend: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Settings and metadata
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    session_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    organization: Mapped["OrganizationModel"] = relationship("OrganizationModel", back_populates="sessions")
    creator: Mapped["UserModel"] = relationship("UserModel", back_populates="created_sessions")
    speakers: Mapped[list["SpeakerModel"]] = relationship("SpeakerModel", back_populates="session", cascade="all, delete-orphan")
    segments: Mapped[list["TranscriptSegmentModel"]] = relationship("TranscriptSegmentModel", back_populates="session", cascade="all, delete-orphan")
    signals: Mapped[list["SignalModel"]] = relationship("SignalModel", back_populates="session", cascade="all, delete-orphan")
    prosodics: Mapped[list["ProsodicsModel"]] = relationship("ProsodicsModel", back_populates="session", cascade="all, delete-orphan")
    insights: Mapped[list["InsightModel"]] = relationship("InsightModel", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sessions_org_id", "org_id"),
        Index("ix_sessions_status", "status"),
        Index("ix_sessions_created_at", "created_at"),
    )


class SpeakerModel(Base, UUIDMixin):
    """Speaker identified in a session."""

    __tablename__ = "speakers"

    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    label: Mapped[str] = mapped_column(String(100), nullable=False)  # "Speaker A", "Speaker B"
    user_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Voice fingerprint reference
    fingerprint_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)

    # Speaker embedding (ECAPA-TDNN vector)
    embedding: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)

    # Aggregated metrics
    talk_time_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    segment_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    talk_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Psychometric scores (0-1 scale)
    engagement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    dominance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    stress_index: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    session: Mapped["SessionModel"] = relationship("SessionModel", back_populates="speakers")
    matched_user: Mapped["UserModel | None"] = relationship("UserModel")
    segments: Mapped[list["TranscriptSegmentModel"]] = relationship("TranscriptSegmentModel", back_populates="speaker")
    signals: Mapped[list["SignalModel"]] = relationship("SignalModel", back_populates="speaker")

    __table_args__ = (
        Index("ix_speakers_session_id", "session_id"),
        Index("ix_speakers_user_id", "user_id"),
    )


class TranscriptSegmentModel(Base, UUIDMixin):
    """Transcript segment with speaker attribution."""

    __tablename__ = "transcript_segments"

    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    speaker_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("speakers.id", ondelete="CASCADE"),
        nullable=False,
    )
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)

    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Word-level timestamps (stored as JSONB array)
    words: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)

    # Flags
    is_question: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    session: Mapped["SessionModel"] = relationship("SessionModel", back_populates="segments")
    speaker: Mapped["SpeakerModel"] = relationship("SpeakerModel", back_populates="segments")

    __table_args__ = (
        Index("ix_transcript_segments_session_id", "session_id"),
        Index("ix_transcript_segments_time", "session_id", "start_ms"),
    )


# ══════════════════════════════════════════════════════════════
# Signal & Prosodics Models (TimescaleDB hypertables)
# ══════════════════════════════════════════════════════════════


class SignalModel(Base, UUIDMixin):
    """Detected signal event from the Signal Atlas."""

    __tablename__ = "signals"

    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    speaker_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("speakers.id", ondelete="CASCADE"),
        nullable=True,
    )

    signal_type: Mapped[SignalType] = mapped_column(SQLEnum(SignalType), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    intensity: Mapped[float] = mapped_column(Float, nullable=False)

    # Raw metrics that triggered the signal
    metrics: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict, nullable=False)

    # Context (surrounding transcript)
    context: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    # Relationships
    session: Mapped["SessionModel"] = relationship("SessionModel", back_populates="signals")
    speaker: Mapped["SpeakerModel | None"] = relationship("SpeakerModel", back_populates="signals")

    __table_args__ = (
        Index("ix_signals_session_id", "session_id"),
        Index("ix_signals_type", "signal_type"),
        Index("ix_signals_time", "session_id", "timestamp_ms"),
    )


class ProsodicsModel(Base, UUIDMixin):
    """Prosodic (acoustic) features extracted from audio - time-series data."""

    __tablename__ = "prosodics"

    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    speaker_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("speakers.id", ondelete="CASCADE"),
        nullable=True,
    )
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Pitch features
    pitch_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_range: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    jitter: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Energy features
    energy_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    shimmer: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Temporal features
    speech_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    silence_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Voice quality
    hnr: Mapped[float | None] = mapped_column(Float, nullable=True)
    spectral_centroid: Mapped[float | None] = mapped_column(Float, nullable=True)
    spectral_flux: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Derived emotional state (VAD model)
    valence: Mapped[float | None] = mapped_column(Float, nullable=True)
    arousal: Mapped[float | None] = mapped_column(Float, nullable=True)
    dominance: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    session: Mapped["SessionModel"] = relationship("SessionModel", back_populates="prosodics")

    __table_args__ = (
        Index("ix_prosodics_session_time", "session_id", "timestamp_ms"),
    )


# ══════════════════════════════════════════════════════════════
# Insights & Timeline Models
# ══════════════════════════════════════════════════════════════


class InsightModel(Base, UUIDMixin, TimestampMixin):
    """AI-generated insight for a session."""

    __tablename__ = "insights"

    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    insight_type: Mapped[str] = mapped_column(String(50), nullable=False)  # summary, key_moment, risk_flag, recommendation
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)

    # Relationships
    session: Mapped["SessionModel"] = relationship("SessionModel", back_populates="insights")

    __table_args__ = (
        Index("ix_insights_session_id", "session_id"),
        Index("ix_insights_type", "insight_type"),
    )


class TimelineModel(Base, UUIDMixin):
    """Pre-computed timeline data point for visualization."""

    __tablename__ = "timeline"

    session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Emotional state
    valence: Mapped[float] = mapped_column(Float, nullable=False)
    arousal: Mapped[float] = mapped_column(Float, nullable=False)
    tension_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Active speaker
    active_speaker_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)

    # Active signals at this point
    active_signals: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)

    __table_args__ = (
        Index("ix_timeline_session_time", "session_id", "timestamp_ms"),
    )


# ══════════════════════════════════════════════════════════════
# Voice Fingerprint Model
# ══════════════════════════════════════════════════════════════


class VoiceFingerprintModel(Base, UUIDMixin, TimestampMixin):
    """Voice fingerprint for speaker recognition across sessions."""

    __tablename__ = "voice_fingerprints"

    org_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ECAPA-TDNN embedding (192-dimensional)
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)

    # Metadata
    sample_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_matched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_voice_fingerprints_org_id", "org_id"),
        Index("ix_voice_fingerprints_user_id", "user_id"),
    )


# ══════════════════════════════════════════════════════════════
# API Key Model (for programmatic access)
# ══════════════════════════════════════════════════════════════


class APIKeyModel(Base, UUIDMixin, TimestampMixin):
    """API key for programmatic access."""

    __tablename__ = "api_keys"

    org_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_by: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(10), nullable=False)  # "sk_live_" or "sk_test_"
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)  # Hashed key

    # Permissions
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)

    # Rate limits (per minute)
    rate_limit: Mapped[int] = mapped_column(Integer, default=100, nullable=False)

    # Usage tracking
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Expiration
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        Index("ix_api_keys_org_id", "org_id"),
        Index("ix_api_keys_prefix", "key_prefix"),
    )


# ══════════════════════════════════════════════════════════════
# Usage Tracking Model
# ══════════════════════════════════════════════════════════════


class UsageRecordModel(Base, UUIDMixin):
    """Usage record for billing and analytics."""

    __tablename__ = "usage_records"

    org_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    # Usage type
    usage_type: Mapped[str] = mapped_column(String(50), nullable=False)  # audio_minutes, api_call, storage_mb

    # Quantity
    quantity: Mapped[float] = mapped_column(Float, nullable=False)

    # Metadata
    usage_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # Stripe reporting
    stripe_reported: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    stripe_reported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_usage_records_org_time", "org_id", "recorded_at"),
        Index("ix_usage_records_type", "usage_type"),
    )


__all__ = [
    "OrganizationModel",
    "UserModel",
    "SessionModel",
    "SpeakerModel",
    "TranscriptSegmentModel",
    "SignalModel",
    "ProsodicsModel",
    "InsightModel",
    "TimelineModel",
    "VoiceFingerprintModel",
    "APIKeyModel",
    "UsageRecordModel",
]
