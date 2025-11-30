"""
Subtext Core Domain Models

Pydantic models representing the core domain entities.
These are used throughout the application for data validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════


class SignalType(str, Enum):
    """Signal types from the Signal Atlas."""

    # Temporal signals
    TRUTH_GAP = "truth_gap"
    STEAMROLL = "steamroll"
    DEAD_AIR = "dead_air"

    # Spectral signals
    MICRO_TREMOR = "micro_tremor"
    MONOTONE = "monotone"
    UPTICK = "uptick"

    # Contextual signals
    ECHO_CHAMBER = "echo_chamber"
    COFFEE_SHOP = "coffee_shop"

    # Composite signals
    STRESS_SPIKE = "stress_spike"
    DISENGAGEMENT = "disengagement"
    DECEPTION_MARKER = "deception_marker"
    ENTHUSIASM_SURGE = "enthusiasm_surge"
    AGREEMENT_SIGNAL = "agreement_signal"
    DISAGREEMENT_SIGNAL = "disagreement_signal"


class SessionStatus(str, Enum):
    """Session processing status."""

    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SubscriptionTier(str, Enum):
    """Subscription tiers for billing."""

    FREE = "free"
    PERSONAL = "personal"
    TEAMS = "teams"
    ENTERPRISE = "enterprise"
    API = "api"


class UserRole(str, Enum):
    """User roles within an organization."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


# ══════════════════════════════════════════════════════════════
# Base Models
# ══════════════════════════════════════════════════════════════


class SubtextModel(BaseModel):
    """Base model with common configuration."""

    model_config = {"from_attributes": True, "populate_by_name": True}


class TimestampMixin(BaseModel):
    """Mixin for created/updated timestamps."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None


# ══════════════════════════════════════════════════════════════
# Organization & User Models
# ══════════════════════════════════════════════════════════════


class Organization(SubtextModel, TimestampMixin):
    """Organization (tenant) in the multi-tenant system."""

    id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100)
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    # Usage limits based on tier
    monthly_minutes_limit: int = 300  # 5 hours for free tier
    monthly_minutes_used: int = 0
    api_calls_limit: int = 1000
    api_calls_used: int = 0


class User(SubtextModel, TimestampMixin):
    """User account linked to Janua authentication."""

    id: UUID
    org_id: UUID
    email: str = Field(..., max_length=255)
    name: str | None = Field(None, max_length=255)
    role: UserRole = UserRole.MEMBER

    # Janua integration
    janua_user_id: str | None = None

    # Profile
    avatar_url: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    # Voice fingerprint (opt-in)
    voice_fingerprint_enabled: bool = False


# ══════════════════════════════════════════════════════════════
# Session & Analysis Models
# ══════════════════════════════════════════════════════════════


class Session(SubtextModel, TimestampMixin):
    """Audio analysis session."""

    id: UUID
    org_id: UUID
    created_by: UUID
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    status: SessionStatus = SessionStatus.PENDING

    # Audio metadata
    source_type: str = "upload"  # upload, realtime, bot
    duration_ms: int | None = None
    language: str | None = None
    storage_path: str | None = None

    # Processing results
    speaker_count: int | None = None
    signal_count: int | None = None
    processing_time_ms: int | None = None
    error_message: str | None = None

    # Settings
    settings: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    completed_at: datetime | None = None


class Speaker(SubtextModel):
    """Speaker identified in a session."""

    id: UUID
    session_id: UUID
    label: str  # "Speaker A", "Speaker B", or matched name
    user_id: UUID | None = None  # If matched to a known user
    fingerprint_id: UUID | None = None  # Voice fingerprint reference

    # Aggregated metrics
    talk_time_ms: int = 0
    segment_count: int = 0
    talk_ratio: float = 0.0

    # Psychometric scores
    engagement_score: float | None = None
    dominance_score: float | None = None
    stress_index: float | None = None


class TranscriptSegment(SubtextModel):
    """Transcript segment with speaker attribution."""

    id: UUID
    session_id: UUID
    speaker_id: UUID
    segment_index: int

    start_ms: int
    end_ms: int
    text: str
    confidence: float = Field(ge=0.0, le=1.0)

    # Word-level timestamps
    words: list[dict[str, Any]] = Field(default_factory=list)

    # Associated signals
    signal_ids: list[UUID] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# Signal & Prosodics Models
# ══════════════════════════════════════════════════════════════


class Signal(SubtextModel):
    """Detected signal event from the Signal Atlas."""

    id: UUID
    session_id: UUID
    speaker_id: UUID | None = None
    signal_type: SignalType
    timestamp_ms: int
    duration_ms: int | None = None

    confidence: float = Field(ge=0.0, le=1.0)
    intensity: float = Field(ge=0.0, le=1.0)

    # Raw metrics that triggered the signal
    metrics: dict[str, float] = Field(default_factory=dict)

    # Context (surrounding transcript)
    context: dict[str, Any] = Field(default_factory=dict)


class ProsodicsFeatures(SubtextModel):
    """Prosodic (acoustic) features extracted from audio."""

    session_id: UUID
    speaker_id: UUID | None = None
    timestamp_ms: int

    # Pitch features
    pitch_mean: float | None = None
    pitch_std: float | None = None
    pitch_range: float | None = None
    pitch_slope: float | None = None
    jitter: float | None = None

    # Energy features
    energy_mean: float | None = None
    energy_std: float | None = None
    shimmer: float | None = None

    # Temporal features
    speech_rate: float | None = None  # Syllables per second
    pause_duration: float | None = None

    # Voice quality
    hnr: float | None = None  # Harmonic-to-noise ratio
    spectral_centroid: float | None = None

    # Derived emotional state
    valence: float | None = Field(None, ge=-1.0, le=1.0)
    arousal: float | None = Field(None, ge=0.0, le=1.0)
    dominance: float | None = Field(None, ge=0.0, le=1.0)


# ══════════════════════════════════════════════════════════════
# Insights & Timeline Models
# ══════════════════════════════════════════════════════════════


class TimelinePoint(SubtextModel):
    """Data point for the Tension Timeline visualization."""

    timestamp_ms: int
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    tension_score: float = Field(ge=0.0, le=1.0)
    active_speaker: str | None = None
    active_signals: list[str] = Field(default_factory=list)


class SessionInsight(SubtextModel):
    """AI-generated insight for a session."""

    id: UUID
    session_id: UUID
    insight_type: str  # summary, key_moment, risk_flag, recommendation
    content: dict[str, Any]
    importance: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KeyMoment(SubtextModel):
    """Significant moment identified in a session."""

    timestamp_ms: int
    moment_type: str  # tension_peak, breakthrough, disagreement, consensus
    description: str
    importance: float = Field(ge=0.0, le=1.0)
    speakers_involved: list[str] = Field(default_factory=list)


class RiskFlag(SubtextModel):
    """Risk flag for potential issues detected."""

    risk_type: str  # deception_risk, burnout_indicator, conflict_signal
    severity: str  # low, medium, high, critical
    description: str
    evidence: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# API Request/Response Models
# ══════════════════════════════════════════════════════════════


class SessionCreate(SubtextModel):
    """Request model for creating a session."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    language: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(SubtextModel):
    """Response model for session details."""

    id: UUID
    name: str
    status: SessionStatus
    duration_ms: int | None
    speaker_count: int | None
    signal_count: int | None
    created_at: datetime
    completed_at: datetime | None


class AnalysisResult(SubtextModel):
    """Complete analysis result for a session."""

    session: SessionResponse
    speakers: list[Speaker]
    transcript: list[TranscriptSegment]
    signals: list[Signal]
    timeline: list[TimelinePoint]
    insights: list[SessionInsight]
    key_moments: list[KeyMoment]
    risk_flags: list[RiskFlag]


# ══════════════════════════════════════════════════════════════
# ESP (Emotional State Protocol) Models
# ══════════════════════════════════════════════════════════════


class ESPMessage(SubtextModel):
    """Emotional State Protocol message for real-time broadcasting."""

    version: str = "1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    speaker_id: str | None = None

    # Core emotional state (VAD model)
    valence: float = Field(ge=-1.0, le=1.0)  # Negative to positive
    arousal: float = Field(ge=0.0, le=1.0)  # Calm to excited
    dominance: float = Field(ge=0.0, le=1.0)  # Submissive to dominant

    # Active signals
    signals: list[dict[str, Any]] = Field(default_factory=list)

    # Aggregate scores
    engagement_score: float = Field(ge=0.0, le=1.0)
    stress_index: float = Field(ge=0.0, le=1.0)
    authenticity_score: float | None = Field(None, ge=0.0, le=1.0)

    # Privacy controls
    consent_level: str = "self"  # self, team, org, public
    retention: str = "session"  # none, session, persistent

    @field_validator("signals", mode="before")
    @classmethod
    def validate_signals(cls, v: list[dict]) -> list[dict]:
        """Ensure signals have required fields."""
        for signal in v:
            if "type" not in signal or "confidence" not in signal:
                raise ValueError("Each signal must have 'type' and 'confidence'")
        return v
