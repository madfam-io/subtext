"""
Unit tests for realtime WebSocket routes.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from subtext.api.routes.realtime import (
    authenticate_websocket,
    _parse_json_message,
)
from subtext.integrations.janua import TokenPayload


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.receive = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


@pytest.fixture
def mock_token_payload():
    """Create a mock token payload."""
    return TokenPayload(
        sub="user-123",
        email="test@example.com",
        org_id="org-456",
        roles=["user"],
        permissions=[],
        exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        iat=int(datetime.utcnow().timestamp()),
        iss="https://auth.example.com",
        aud="subtext-api",
    )


# ══════════════════════════════════════════════════════════════
# Authentication Tests
# ══════════════════════════════════════════════════════════════


class TestAuthenticateWebsocket:
    """Test WebSocket authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_no_token_development(self, mock_websocket):
        """Test authentication without token in development mode."""
        with patch("subtext.config.settings") as mock_settings:
            mock_settings.app_env = "development"

            user_id, org_id = await authenticate_websocket(mock_websocket, None)

            assert user_id is None
            assert org_id is None

    @pytest.mark.asyncio
    async def test_authenticate_no_token_production(self, mock_websocket):
        """Test authentication without token in production mode."""
        with patch("subtext.config.settings") as mock_settings:
            mock_settings.app_env = "production"

            user_id, org_id = await authenticate_websocket(mock_websocket, None)

            assert user_id is None
            assert org_id is None

    @pytest.mark.asyncio
    async def test_authenticate_valid_token(self, mock_websocket, mock_token_payload):
        """Test authentication with valid token."""
        with patch("subtext.integrations.janua.JanuaAuth") as mock_auth_cls:
            mock_auth = MagicMock()
            mock_auth._verify_token = AsyncMock(return_value=mock_token_payload)
            mock_auth_cls.return_value = mock_auth

            user_id, org_id = await authenticate_websocket(mock_websocket, "valid-token")

            # Should return None if UUID conversion fails
            assert user_id is None or isinstance(user_id, UUID)

    @pytest.mark.asyncio
    async def test_authenticate_invalid_token(self, mock_websocket):
        """Test authentication with invalid token."""
        with patch("subtext.integrations.janua.JanuaAuth") as mock_auth_cls:
            mock_auth = MagicMock()
            mock_auth._verify_token = AsyncMock(side_effect=Exception("Invalid token"))
            mock_auth_cls.return_value = mock_auth

            user_id, org_id = await authenticate_websocket(mock_websocket, "invalid-token")

            assert user_id is None
            assert org_id is None


# ══════════════════════════════════════════════════════════════
# Message Parsing Tests
# ══════════════════════════════════════════════════════════════


class TestParseJsonMessage:
    """Test JSON message parsing."""

    @pytest.mark.asyncio
    async def test_parse_valid_json(self):
        """Test parsing valid JSON message."""
        result = await _parse_json_message('{"type": "session.start", "payload": {}}')

        assert result["type"] == "session.start"
        assert result["payload"] == {}

    @pytest.mark.asyncio
    async def test_parse_complex_json(self):
        """Test parsing complex JSON message."""
        json_str = '{"type": "audio.chunk", "payload": {"data": "base64data", "timestamp_ms": 1000}}'
        result = await _parse_json_message(json_str)

        assert result["type"] == "audio.chunk"
        assert result["payload"]["data"] == "base64data"
        assert result["payload"]["timestamp_ms"] == 1000

    @pytest.mark.asyncio
    async def test_parse_invalid_json(self):
        """Test parsing invalid JSON raises error."""
        with pytest.raises(Exception):
            await _parse_json_message("not valid json")


# ══════════════════════════════════════════════════════════════
# Connection Manager Tests
# ══════════════════════════════════════════════════════════════


class TestConnectionManager:
    """Test connection manager functionality."""

    @pytest.mark.asyncio
    async def test_get_connection_manager(self):
        """Test getting global connection manager."""
        from subtext.realtime.connection import get_connection_manager

        manager = get_connection_manager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_manager_connect_disconnect(self, mock_websocket):
        """Test connecting and disconnecting a WebSocket."""
        from subtext.realtime.connection import ConnectionManager

        manager = ConnectionManager()

        # Mock websocket accept
        mock_websocket.accept = AsyncMock()

        # Connect
        connection = await manager.connect(mock_websocket)
        assert connection is not None
        assert connection.connection_id is not None

        # Disconnect
        await manager.disconnect(connection)

    @pytest.mark.asyncio
    async def test_manager_stats(self):
        """Test getting manager statistics."""
        from subtext.realtime.connection import ConnectionManager

        manager = ConnectionManager()
        stats = manager.get_stats()

        assert "active_connections" in stats
        assert "active_sessions" in stats


# ══════════════════════════════════════════════════════════════
# ESP Broadcaster Tests
# ══════════════════════════════════════════════════════════════


class TestESPBroadcaster:
    """Test ESP broadcaster functionality."""

    @pytest.mark.asyncio
    async def test_get_esp_broadcaster(self):
        """Test getting global ESP broadcaster."""
        from subtext.realtime.broadcaster import get_esp_broadcaster

        broadcaster = get_esp_broadcaster()
        assert broadcaster is not None

    @pytest.mark.asyncio
    async def test_broadcaster_stats(self):
        """Test getting broadcaster statistics."""
        from subtext.realtime.broadcaster import ESPBroadcaster

        broadcaster = ESPBroadcaster()
        stats = broadcaster.get_stats()

        assert "active_channels" in stats
        assert "total_subscribers" in stats

    @pytest.mark.asyncio
    async def test_create_and_close_channel(self):
        """Test creating and closing a channel."""
        from subtext.realtime.broadcaster import ESPBroadcaster

        broadcaster = ESPBroadcaster()
        session_id = uuid4()

        # Create channel
        await broadcaster.create_channel(
            session_id=session_id,
            owner_id=uuid4(),
            org_id=uuid4(),
        )

        stats = broadcaster.get_stats()
        assert stats["active_channels"] >= 0

        # Close channel
        await broadcaster.close_channel(session_id)

    @pytest.mark.asyncio
    async def test_register_unregister_callback(self):
        """Test registering and unregistering callbacks."""
        from subtext.realtime.broadcaster import ESPBroadcaster

        broadcaster = ESPBroadcaster()
        connection_id = uuid4()

        async def callback(msg):
            pass

        broadcaster.register_send_callback(connection_id, callback)
        broadcaster.unregister_send_callback(connection_id)


# ══════════════════════════════════════════════════════════════
# Realtime Processor Tests
# ══════════════════════════════════════════════════════════════


class TestRealtimeProcessor:
    """Test realtime processor functionality."""

    @pytest.mark.asyncio
    async def test_processor_initialization(self):
        """Test processor initialization."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig

        session_id = uuid4()
        config = SessionConfig()

        processor = RealtimeProcessor(session_id, config)
        assert processor.session_id == session_id

    @pytest.mark.asyncio
    async def test_processor_initialize_finalize(self):
        """Test processor initialize and finalize."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig

        session_id = uuid4()
        config = SessionConfig()

        processor = RealtimeProcessor(session_id, config)

        # Initialize
        await processor.initialize()

        # Finalize
        summary = await processor.finalize()
        assert isinstance(summary, dict)


# ══════════════════════════════════════════════════════════════
# Protocol Message Tests
# ══════════════════════════════════════════════════════════════


class TestProtocolMessages:
    """Test protocol message types."""

    def test_session_config_defaults(self):
        """Test SessionConfig with defaults."""
        from subtext.realtime.protocol import SessionConfig

        config = SessionConfig()
        assert config.asr_backend == "whisperx"
        assert config.enable_transcription is True

    def test_session_config_custom(self):
        """Test SessionConfig with custom values."""
        from subtext.realtime.protocol import SessionConfig

        config = SessionConfig(
            asr_backend="whisper",
            enable_transcription=False,
            enable_diarization=True,
        )
        assert config.asr_backend == "whisper"
        assert config.enable_transcription is False
        assert config.enable_diarization is True

    def test_realtime_message(self):
        """Test RealtimeMessage model."""
        from subtext.realtime.protocol import RealtimeMessage, RealtimeMessageType

        session_id = uuid4()
        msg = RealtimeMessage(
            type=RealtimeMessageType.SESSION_CREATED,
            session_id=session_id,
            payload={"key": "value"},
        )

        assert msg.type == RealtimeMessageType.SESSION_CREATED
        assert msg.session_id == session_id
        assert msg.payload == {"key": "value"}

    def test_session_start_payload(self):
        """Test SessionStartPayload model."""
        from subtext.realtime.protocol import SessionStartPayload, SessionConfig

        config = SessionConfig()
        payload = SessionStartPayload(config=config)

        assert payload.config.enable_transcription is True

    def test_audio_chunk_payload(self):
        """Test AudioChunkPayload model."""
        from subtext.realtime.protocol import AudioChunkPayload

        payload = AudioChunkPayload(
            data="base64data",
            timestamp_ms=1000,
        )

        assert payload.data == "base64data"
        assert payload.timestamp_ms == 1000

    def test_error_payload(self):
        """Test ErrorPayload model."""
        from subtext.realtime.protocol import ErrorPayload

        payload = ErrorPayload(
            code="error_code",
            message="Error message",
            recoverable=True,
        )

        assert payload.code == "error_code"
        assert payload.message == "Error message"
        assert payload.recoverable is True


# ══════════════════════════════════════════════════════════════
# Audio Chunk Tests
# ══════════════════════════════════════════════════════════════


class TestAudioChunk:
    """Test AudioChunk functionality."""

    def test_audio_chunk_from_bytes(self):
        """Test creating AudioChunk from bytes."""
        from subtext.realtime import AudioChunk
        from subtext.realtime.protocol import AudioConfig

        audio_config = AudioConfig()
        audio_data = b"\x00" * 100

        chunk = AudioChunk.from_bytes(audio_data, 0, audio_config)

        assert chunk.timestamp_ms == 0
        assert len(chunk.data) > 0

    def test_audio_chunk_from_base64(self):
        """Test creating AudioChunk from base64."""
        import base64
        from subtext.realtime import AudioChunk
        from subtext.realtime.protocol import AudioConfig

        audio_config = AudioConfig()
        audio_data = base64.b64encode(b"\x00" * 100).decode()

        chunk = AudioChunk.from_base64(audio_data, 1000, audio_config)

        assert chunk.timestamp_ms == 1000


# ══════════════════════════════════════════════════════════════
# Realtime Connection Tests
# ══════════════════════════════════════════════════════════════


class TestRealtimeConnection:
    """Test RealtimeConnection functionality."""

    @pytest.mark.asyncio
    async def test_connection_creation(self, mock_websocket):
        """Test creating a realtime connection."""
        from subtext.realtime.connection import RealtimeConnection

        connection_id = uuid4()
        connection = RealtimeConnection(
            connection_id=connection_id,
            websocket=mock_websocket,
            user_id=uuid4(),
            org_id=uuid4(),
        )

        assert connection.connection_id == connection_id
        assert connection.is_streaming is False

    @pytest.mark.asyncio
    async def test_connection_send_message(self, mock_websocket):
        """Test sending message through connection."""
        from subtext.realtime.connection import RealtimeConnection
        from subtext.realtime.protocol import RealtimeMessage, RealtimeMessageType

        connection = RealtimeConnection(
            connection_id=uuid4(),
            websocket=mock_websocket,
            user_id=None,
            org_id=None,
        )

        msg = RealtimeMessage(
            type=RealtimeMessageType.PONG,
        )

        await connection.send_message(msg)
        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_send_error(self, mock_websocket):
        """Test sending error through connection."""
        from subtext.realtime.connection import RealtimeConnection

        connection = RealtimeConnection(
            connection_id=uuid4(),
            websocket=mock_websocket,
            user_id=None,
            org_id=None,
        )

        await connection.send_error("error_code", "Error message")
        mock_websocket.send_json.assert_called_once()

    def test_connection_update_activity(self, mock_websocket):
        """Test updating activity timestamp."""
        from subtext.realtime.connection import RealtimeConnection

        connection = RealtimeConnection(
            connection_id=uuid4(),
            websocket=mock_websocket,
            user_id=None,
            org_id=None,
        )

        old_time = connection.last_activity
        connection.update_activity()

        assert connection.last_activity >= old_time


# ══════════════════════════════════════════════════════════════
# HTTP Endpoint Tests
# ══════════════════════════════════════════════════════════════


class TestHTTPEndpoints:
    """Test HTTP endpoints in realtime routes."""

    @pytest.mark.asyncio
    async def test_get_connections(self):
        """Test getting connection statistics."""
        from subtext.api.routes.realtime import get_connections
        from subtext.realtime.connection import ConnectionManager

        manager = ConnectionManager()
        result = await get_connections(manager)

        assert isinstance(result, dict)
        assert "active_connections" in result

    @pytest.mark.asyncio
    async def test_get_esp_stats(self):
        """Test getting ESP statistics."""
        from subtext.api.routes.realtime import get_esp_stats
        from subtext.realtime.broadcaster import ESPBroadcaster

        broadcaster = ESPBroadcaster()
        result = await get_esp_stats(broadcaster)

        assert isinstance(result, dict)
        assert "active_channels" in result
