"""
Realtime WebSocket Protocol

Defines the message types and data structures for real-time communication.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class RealtimeMessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    AUDIO_CHUNK = "audio.chunk"
    AUDIO_END = "audio.end"
    CONFIG_UPDATE = "config.update"
    PING = "ping"

    # Server -> Client
    SESSION_CREATED = "session.created"
    SESSION_CLOSED = "session.closed"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    SIGNAL_DETECTED = "signal.detected"
    ESP_UPDATE = "esp.update"
    SPEAKER_IDENTIFIED = "speaker.identified"
    PROSODICS_UPDATE = "prosodics.update"
    TIMELINE_UPDATE = "timeline.update"
    ERROR = "error"
    PONG = "pong"


class AudioConfig(BaseModel):
    """Audio stream configuration."""

    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    encoding: str = Field(default="pcm_s16le")  # pcm_s16le, pcm_f32le, opus
    chunk_duration_ms: int = Field(default=100, ge=20, le=1000)

    @property
    def bytes_per_sample(self) -> int:
        """Calculate bytes per sample based on encoding."""
        encoding_sizes = {
            "pcm_s16le": 2,
            "pcm_f32le": 4,
            "opus": 0,  # Variable
        }
        return encoding_sizes.get(self.encoding, 2)

    @property
    def chunk_samples(self) -> int:
        """Calculate samples per chunk."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def chunk_bytes(self) -> int:
        """Calculate expected bytes per chunk."""
        return self.chunk_samples * self.channels * self.bytes_per_sample


class SessionConfig(BaseModel):
    """Real-time session configuration."""

    name: str = Field(default="Realtime Session", max_length=255)
    language: str | None = None

    # Analysis settings
    enable_transcription: bool = True
    enable_diarization: bool = True
    enable_emotion: bool = True
    enable_prosodics: bool = True
    enable_signals: bool = True

    # ASR settings
    asr_backend: str = Field(default="whisperx")  # whisperx, canary, parakeet
    asr_partial_results: bool = True

    # Signal detection
    signal_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # ESP broadcasting
    esp_enabled: bool = True
    esp_interval_ms: int = Field(default=500, ge=100, le=5000)
    esp_consent_level: str = Field(default="self")  # self, team, org, public

    # Privacy
    retention: str = Field(default="session")  # none, session, persistent

    # Audio
    audio: AudioConfig = Field(default_factory=AudioConfig)


class RealtimeMessage(BaseModel):
    """Base WebSocket message format."""

    type: RealtimeMessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: UUID | None = None
    sequence: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


# ══════════════════════════════════════════════════════════════
# Specific Message Payloads
# ══════════════════════════════════════════════════════════════


class SessionStartPayload(BaseModel):
    """Payload for session.start message."""

    config: SessionConfig = Field(default_factory=SessionConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionCreatedPayload(BaseModel):
    """Payload for session.created response."""

    session_id: UUID
    audio_config: AudioConfig
    features_enabled: dict[str, bool]


class AudioChunkPayload(BaseModel):
    """Payload for audio.chunk message."""

    data: bytes | str  # Base64 encoded if string
    timestamp_ms: int
    is_speech: bool | None = None  # Optional client-side VAD hint


class TranscriptPayload(BaseModel):
    """Payload for transcript messages."""

    text: str
    speaker_id: str | None = None
    speaker_label: str | None = None
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    words: list[dict[str, Any]] = Field(default_factory=list)
    is_final: bool = False


class SignalPayload(BaseModel):
    """Payload for signal.detected message."""

    signal_type: str
    timestamp_ms: int
    duration_ms: int | None = None
    confidence: float
    intensity: float
    speaker_id: str | None = None
    speaker_label: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ESPPayload(BaseModel):
    """Payload for esp.update message (Emotional State Protocol)."""

    version: str = "1.0"
    speaker_id: str | None = None
    speaker_label: str | None = None

    # Core VAD state
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)

    # Active signals
    signals: list[dict[str, Any]] = Field(default_factory=list)

    # Aggregate scores
    engagement_score: float = Field(ge=0.0, le=1.0)
    stress_index: float = Field(ge=0.0, le=1.0)
    authenticity_score: float | None = None

    # Consent
    consent_level: str = "self"


class SpeakerPayload(BaseModel):
    """Payload for speaker.identified message."""

    speaker_id: str
    speaker_label: str
    is_new: bool = False
    matched_user_id: UUID | None = None
    matched_user_name: str | None = None
    confidence: float = 1.0


class ProsodicsPayload(BaseModel):
    """Payload for prosodics.update message."""

    timestamp_ms: int
    speaker_id: str | None = None

    # Pitch features
    pitch_mean: float | None = None
    pitch_std: float | None = None

    # Energy
    energy_mean: float | None = None
    energy_std: float | None = None

    # Temporal
    speech_rate: float | None = None

    # VAD derived
    valence: float | None = None
    arousal: float | None = None
    dominance: float | None = None


class TimelinePayload(BaseModel):
    """Payload for timeline.update message."""

    timestamp_ms: int
    valence: float
    arousal: float
    tension_score: float
    active_speaker: str | None = None
    active_signals: list[str] = Field(default_factory=list)


class ErrorPayload(BaseModel):
    """Payload for error message."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = True
