"""
Unit Tests for Realtime Module

Tests the WebSocket protocol, connection management, and broadcasting.
"""

import pytest
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from subtext.realtime.protocol import (
    RealtimeMessageType,
    AudioConfig,
    SessionConfig,
    RealtimeMessage,
    SessionStartPayload,
    SessionCreatedPayload,
    AudioChunkPayload,
    TranscriptPayload,
    SignalPayload,
    ESPPayload,
    SpeakerPayload,
    ProsodicsPayload,
    TimelinePayload,
    ErrorPayload,
)


# ══════════════════════════════════════════════════════════════
# Protocol Message Type Tests
# ══════════════════════════════════════════════════════════════


class TestRealtimeMessageTypes:
    """Test WebSocket message type enums."""

    def test_client_message_types(self):
        """Test client -> server message types."""
        client_types = [
            RealtimeMessageType.SESSION_START,
            RealtimeMessageType.SESSION_END,
            RealtimeMessageType.AUDIO_CHUNK,
            RealtimeMessageType.AUDIO_END,
            RealtimeMessageType.CONFIG_UPDATE,
            RealtimeMessageType.PING,
        ]
        assert all(t.value.startswith(("session.", "audio.", "config.", "ping")) for t in client_types)

    def test_server_message_types(self):
        """Test server -> client message types."""
        server_types = [
            RealtimeMessageType.SESSION_CREATED,
            RealtimeMessageType.SESSION_CLOSED,
            RealtimeMessageType.TRANSCRIPT_PARTIAL,
            RealtimeMessageType.TRANSCRIPT_FINAL,
            RealtimeMessageType.SIGNAL_DETECTED,
            RealtimeMessageType.ESP_UPDATE,
            RealtimeMessageType.SPEAKER_IDENTIFIED,
            RealtimeMessageType.PROSODICS_UPDATE,
            RealtimeMessageType.TIMELINE_UPDATE,
            RealtimeMessageType.ERROR,
            RealtimeMessageType.PONG,
        ]
        assert len(server_types) == 11

    def test_message_type_values(self):
        """Test message type string values."""
        assert RealtimeMessageType.SESSION_START.value == "session.start"
        assert RealtimeMessageType.ESP_UPDATE.value == "esp.update"
        assert RealtimeMessageType.SIGNAL_DETECTED.value == "signal.detected"


# ══════════════════════════════════════════════════════════════
# Audio Config Tests
# ══════════════════════════════════════════════════════════════


class TestAudioConfig:
    """Test audio configuration model."""

    def test_default_config(self):
        """Test default audio configuration."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.encoding == "pcm_s16le"
        assert config.chunk_duration_ms == 100

    def test_bytes_per_sample_pcm_s16le(self):
        """Test bytes per sample for PCM 16-bit."""
        config = AudioConfig(encoding="pcm_s16le")
        assert config.bytes_per_sample == 2

    def test_bytes_per_sample_pcm_f32le(self):
        """Test bytes per sample for PCM 32-bit float."""
        config = AudioConfig(encoding="pcm_f32le")
        assert config.bytes_per_sample == 4

    def test_bytes_per_sample_opus(self):
        """Test bytes per sample for Opus (variable)."""
        config = AudioConfig(encoding="opus")
        assert config.bytes_per_sample == 0

    def test_chunk_samples_calculation(self):
        """Test chunk samples calculation."""
        config = AudioConfig(sample_rate=16000, chunk_duration_ms=100)
        assert config.chunk_samples == 1600  # 16000 * 100 / 1000

    def test_chunk_bytes_calculation(self):
        """Test chunk bytes calculation."""
        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            encoding="pcm_s16le",
            chunk_duration_ms=100
        )
        # 1600 samples * 1 channel * 2 bytes
        assert config.chunk_bytes == 3200

    def test_stereo_chunk_bytes(self):
        """Test chunk bytes for stereo audio."""
        config = AudioConfig(
            sample_rate=16000,
            channels=2,
            encoding="pcm_s16le",
            chunk_duration_ms=100
        )
        # 1600 samples * 2 channels * 2 bytes
        assert config.chunk_bytes == 6400

    def test_sample_rate_validation(self):
        """Test sample rate validation bounds."""
        with pytest.raises(ValueError):
            AudioConfig(sample_rate=4000)  # Too low
        with pytest.raises(ValueError):
            AudioConfig(sample_rate=96000)  # Too high

    def test_channels_validation(self):
        """Test channel count validation."""
        AudioConfig(channels=1)  # Valid
        AudioConfig(channels=2)  # Valid
        with pytest.raises(ValueError):
            AudioConfig(channels=0)
        with pytest.raises(ValueError):
            AudioConfig(channels=8)


# ══════════════════════════════════════════════════════════════
# Session Config Tests
# ══════════════════════════════════════════════════════════════


class TestSessionConfig:
    """Test session configuration model."""

    def test_default_config(self):
        """Test default session configuration."""
        config = SessionConfig()
        assert config.name == "Realtime Session"
        assert config.language is None
        assert config.enable_transcription is True
        assert config.enable_emotion is True
        assert config.asr_backend == "whisperx"
        assert config.esp_enabled is True

    def test_custom_config(self):
        """Test custom session configuration."""
        config = SessionConfig(
            name="Test Session",
            language="en",
            enable_transcription=True,
            enable_diarization=False,
            asr_backend="canary",
        )
        assert config.name == "Test Session"
        assert config.language == "en"
        assert config.enable_diarization is False
        assert config.asr_backend == "canary"

    def test_signal_confidence_threshold(self):
        """Test signal confidence threshold validation."""
        config = SessionConfig(signal_confidence_threshold=0.7)
        assert config.signal_confidence_threshold == 0.7

        with pytest.raises(ValueError):
            SessionConfig(signal_confidence_threshold=-0.1)
        with pytest.raises(ValueError):
            SessionConfig(signal_confidence_threshold=1.5)

    def test_esp_interval_validation(self):
        """Test ESP interval validation."""
        config = SessionConfig(esp_interval_ms=1000)
        assert config.esp_interval_ms == 1000

        with pytest.raises(ValueError):
            SessionConfig(esp_interval_ms=50)  # Too low
        with pytest.raises(ValueError):
            SessionConfig(esp_interval_ms=10000)  # Too high

    def test_nested_audio_config(self):
        """Test nested audio configuration."""
        config = SessionConfig(
            audio=AudioConfig(sample_rate=44100, channels=2)
        )
        assert config.audio.sample_rate == 44100
        assert config.audio.channels == 2


# ══════════════════════════════════════════════════════════════
# Realtime Message Tests
# ══════════════════════════════════════════════════════════════


class TestRealtimeMessage:
    """Test base realtime message model."""

    def test_message_creation(self):
        """Test creating a realtime message."""
        msg = RealtimeMessage(type=RealtimeMessageType.PING)
        assert msg.type == RealtimeMessageType.PING
        assert msg.payload == {}
        assert isinstance(msg.timestamp, datetime)

    def test_message_with_session_id(self):
        """Test message with session ID."""
        session_id = uuid4()
        msg = RealtimeMessage(
            type=RealtimeMessageType.AUDIO_CHUNK,
            session_id=session_id,
            sequence=42,
            payload={"data": "test"}
        )
        assert msg.session_id == session_id
        assert msg.sequence == 42
        assert msg.payload["data"] == "test"

    def test_message_serialization(self):
        """Test message JSON serialization."""
        msg = RealtimeMessage(
            type=RealtimeMessageType.ESP_UPDATE,
            payload={"valence": 0.5}
        )
        data = msg.model_dump()
        assert data["type"] == "esp.update"  # Enum converted to value
        assert data["payload"]["valence"] == 0.5


# ══════════════════════════════════════════════════════════════
# Payload Tests
# ══════════════════════════════════════════════════════════════


class TestPayloads:
    """Test message payload models."""

    def test_session_start_payload(self):
        """Test session start payload."""
        payload = SessionStartPayload()
        assert isinstance(payload.config, SessionConfig)
        assert payload.metadata == {}

    def test_session_created_payload(self):
        """Test session created payload."""
        session_id = uuid4()
        payload = SessionCreatedPayload(
            session_id=session_id,
            audio_config=AudioConfig(),
            features_enabled={
                "transcription": True,
                "emotion": True,
            }
        )
        assert payload.session_id == session_id
        assert payload.features_enabled["transcription"] is True

    def test_audio_chunk_payload(self):
        """Test audio chunk payload."""
        payload = AudioChunkPayload(
            data=b"audio_bytes",
            timestamp_ms=1000,
            is_speech=True
        )
        assert payload.data == b"audio_bytes"
        assert payload.timestamp_ms == 1000
        assert payload.is_speech is True

    def test_transcript_payload(self):
        """Test transcript payload."""
        payload = TranscriptPayload(
            text="Hello world",
            speaker_id="spk_001",
            speaker_label="Speaker A",
            start_ms=0,
            end_ms=2000,
            confidence=0.95,
            is_final=True
        )
        assert payload.text == "Hello world"
        assert payload.speaker_label == "Speaker A"
        assert payload.is_final is True

    def test_signal_payload(self):
        """Test signal payload."""
        payload = SignalPayload(
            signal_type="hesitation",
            timestamp_ms=1500,
            duration_ms=500,
            confidence=0.8,
            intensity=0.6,
            speaker_id="spk_001"
        )
        assert payload.signal_type == "hesitation"
        assert payload.confidence == 0.8
        assert payload.intensity == 0.6

    def test_esp_payload(self):
        """Test ESP (Emotional State Protocol) payload."""
        payload = ESPPayload(
            valence=0.3,
            arousal=0.7,
            dominance=0.5,
            engagement_score=0.8,
            stress_index=0.4,
            signals=[{"type": "tension", "confidence": 0.7}]
        )
        assert payload.valence == 0.3
        assert payload.arousal == 0.7
        assert payload.dominance == 0.5
        assert payload.engagement_score == 0.8
        assert len(payload.signals) == 1

    def test_esp_payload_validation(self):
        """Test ESP payload validation."""
        # Valence must be between -1 and 1
        with pytest.raises(ValueError):
            ESPPayload(valence=-1.5, arousal=0.5, dominance=0.5, engagement_score=0.5, stress_index=0.5)

        # Arousal must be between 0 and 1
        with pytest.raises(ValueError):
            ESPPayload(valence=0.5, arousal=1.5, dominance=0.5, engagement_score=0.5, stress_index=0.5)

    def test_speaker_payload(self):
        """Test speaker identified payload."""
        payload = SpeakerPayload(
            speaker_id="spk_001",
            speaker_label="Speaker A",
            is_new=True,
            confidence=0.92
        )
        assert payload.speaker_id == "spk_001"
        assert payload.is_new is True

    def test_prosodics_payload(self):
        """Test prosodics update payload."""
        payload = ProsodicsPayload(
            timestamp_ms=5000,
            speaker_id="spk_001",
            pitch_mean=150.0,
            pitch_std=20.0,
            energy_mean=0.6,
            speech_rate=2.5,
            valence=0.4,
            arousal=0.6,
            dominance=0.5
        )
        assert payload.pitch_mean == 150.0
        assert payload.speech_rate == 2.5

    def test_timeline_payload(self):
        """Test timeline update payload."""
        payload = TimelinePayload(
            timestamp_ms=10000,
            valence=0.2,
            arousal=0.8,
            tension_score=0.7,
            active_speaker="Speaker A",
            active_signals=["hesitation", "tension"]
        )
        assert payload.tension_score == 0.7
        assert "hesitation" in payload.active_signals

    def test_error_payload(self):
        """Test error payload."""
        payload = ErrorPayload(
            code="INVALID_AUDIO",
            message="Audio format not supported",
            details={"expected": "pcm_s16le", "received": "mp3"},
            recoverable=False
        )
        assert payload.code == "INVALID_AUDIO"
        assert payload.recoverable is False


# ══════════════════════════════════════════════════════════════
# Broadcaster Tests
# ══════════════════════════════════════════════════════════════


class TestBroadcaster:
    """Test ESP broadcaster functionality."""

    def test_broadcaster_initialization(self):
        """Test broadcaster initialization."""
        from subtext.realtime.broadcaster import ESPBroadcaster

        broadcaster = ESPBroadcaster()
        assert broadcaster._channels == {}
        assert broadcaster._send_callbacks == {}

    def test_broadcaster_dataclasses(self):
        """Test broadcaster dataclasses."""
        from subtext.realtime.broadcaster import ESPSubscription, ESPChannel

        # Test ESPSubscription
        subscription = ESPSubscription(
            subscriber_id=uuid4(),
            source_session_id=uuid4(),
            consent_level="self"
        )
        assert subscription.updates_received == 0
        assert subscription.last_update is None

        # Test ESPChannel
        channel = ESPChannel(
            session_id=uuid4(),
            owner_id=uuid4(),
            org_id=uuid4(),
            consent_level="team"
        )
        assert channel.is_active is True
        assert channel.broadcast_count == 0
        assert channel.subscribers == {}

    @pytest.mark.asyncio
    async def test_broadcaster_start_stop(self):
        """Test broadcaster start and stop."""
        from subtext.realtime.broadcaster import ESPBroadcaster

        broadcaster = ESPBroadcaster()

        # Start creates cleanup task
        await broadcaster.start()
        assert broadcaster._cleanup_task is not None

        # Stop cancels cleanup task
        await broadcaster.stop()
        # After stop, cleanup task should be cancelled
        assert broadcaster._cleanup_task.cancelled() or broadcaster._cleanup_task.done()


# ══════════════════════════════════════════════════════════════
# Connection Manager Tests
# ══════════════════════════════════════════════════════════════


class TestConnectionManager:
    """Test WebSocket connection manager."""

    def test_connection_manager_initialization(self):
        """Test connection manager initialization."""
        from subtext.realtime.connection import ConnectionManager

        manager = ConnectionManager()
        assert manager._connections == {}
        assert manager._session_connections == {}
        assert manager.active_connection_count == 0
        assert manager.active_session_count == 0

    @pytest.mark.asyncio
    async def test_connection_manager_connect(self):
        """Test connecting a WebSocket."""
        from subtext.realtime.connection import ConnectionManager, RealtimeConnection

        manager = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()

        connection = await manager.connect(mock_ws)

        assert isinstance(connection, RealtimeConnection)
        assert connection.connection_id in manager._connections
        mock_ws.accept.assert_called_once()
        assert manager.active_connection_count == 1

    @pytest.mark.asyncio
    async def test_connection_manager_connect_with_user(self):
        """Test connecting a WebSocket with user context."""
        from subtext.realtime.connection import ConnectionManager

        manager = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()

        user_id = uuid4()
        org_id = uuid4()
        connection = await manager.connect(mock_ws, user_id=user_id, org_id=org_id)

        assert connection.user_id == user_id
        assert connection.org_id == org_id
        assert org_id in manager._org_connections
        assert connection.connection_id in manager._org_connections[org_id]

    @pytest.mark.asyncio
    async def test_connection_manager_disconnect(self):
        """Test disconnecting a WebSocket."""
        from subtext.realtime.connection import ConnectionManager

        manager = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()

        connection = await manager.connect(mock_ws)
        connection_id = connection.connection_id

        # disconnect expects the connection object, not UUID
        await manager.disconnect(connection)

        assert connection_id not in manager._connections
        assert manager.active_connection_count == 0

    def test_realtime_connection_dataclass(self):
        """Test RealtimeConnection dataclass."""
        from subtext.realtime.connection import RealtimeConnection
        from subtext.realtime.protocol import SessionConfig

        mock_ws = MagicMock()
        connection_id = uuid4()

        connection = RealtimeConnection(
            connection_id=connection_id,
            websocket=mock_ws,
        )

        assert connection.connection_id == connection_id
        assert connection.session_id is None
        assert connection.is_streaming is False
        assert connection.sequence_counter == 0
        assert isinstance(connection.config, SessionConfig)

    def test_realtime_connection_equality(self):
        """Test RealtimeConnection equality and hashing."""
        from subtext.realtime.connection import RealtimeConnection

        mock_ws = MagicMock()
        connection_id = uuid4()

        conn1 = RealtimeConnection(connection_id=connection_id, websocket=mock_ws)
        conn2 = RealtimeConnection(connection_id=connection_id, websocket=mock_ws)
        conn3 = RealtimeConnection(connection_id=uuid4(), websocket=mock_ws)

        assert conn1 == conn2
        assert conn1 != conn3
        assert hash(conn1) == hash(conn2)
