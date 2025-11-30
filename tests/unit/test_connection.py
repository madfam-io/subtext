"""
Unit Tests for WebSocket Connection Module

Tests the WebSocket connection management for realtime streaming.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from subtext.core.models import ESPMessage
from subtext.realtime.connection import (
    RealtimeConnection,
    ConnectionManager,
    manager,
    get_connection_manager,
)
from subtext.realtime.protocol import (
    RealtimeMessage,
    RealtimeMessageType,
    SessionConfig,
)


# ============================================================
# RealtimeConnection Tests
# ============================================================


class TestRealtimeConnectionInit:
    """Test RealtimeConnection initialization."""

    def test_connection_creation(self):
        """Test creating a connection."""
        ws = MagicMock()
        conn_id = uuid4()

        conn = RealtimeConnection(
            connection_id=conn_id,
            websocket=ws,
        )

        assert conn.connection_id == conn_id
        assert conn.websocket == ws
        assert conn.session_id is None
        assert conn.user_id is None
        assert conn.is_streaming is False
        assert conn.sequence_counter == 0

    def test_connection_with_user(self):
        """Test creating connection with user info."""
        ws = MagicMock()
        user_id = uuid4()
        org_id = uuid4()

        conn = RealtimeConnection(
            connection_id=uuid4(),
            websocket=ws,
            user_id=user_id,
            org_id=org_id,
        )

        assert conn.user_id == user_id
        assert conn.org_id == org_id

    def test_connection_defaults(self):
        """Test connection default values."""
        ws = MagicMock()

        conn = RealtimeConnection(
            connection_id=uuid4(),
            websocket=ws,
        )

        assert isinstance(conn.config, SessionConfig)
        assert isinstance(conn.connected_at, datetime)
        assert conn.audio_buffer == b""
        assert conn.samples_received == 0


class TestRealtimeConnectionHashEquality:
    """Test connection hash and equality."""

    def test_connection_hash(self):
        """Test connection hash based on ID."""
        conn_id = uuid4()
        ws = MagicMock()

        conn = RealtimeConnection(
            connection_id=conn_id,
            websocket=ws,
        )

        assert hash(conn) == hash(conn_id)

    def test_connection_equality(self):
        """Test connection equality based on ID."""
        conn_id = uuid4()
        ws1 = MagicMock()
        ws2 = MagicMock()

        conn1 = RealtimeConnection(connection_id=conn_id, websocket=ws1)
        conn2 = RealtimeConnection(connection_id=conn_id, websocket=ws2)

        assert conn1 == conn2

    def test_connection_inequality(self):
        """Test connections with different IDs are not equal."""
        ws = MagicMock()

        conn1 = RealtimeConnection(connection_id=uuid4(), websocket=ws)
        conn2 = RealtimeConnection(connection_id=uuid4(), websocket=ws)

        assert conn1 != conn2

    def test_connection_not_equal_to_other_types(self):
        """Test connection not equal to non-connection types."""
        ws = MagicMock()
        conn = RealtimeConnection(connection_id=uuid4(), websocket=ws)

        assert conn != "not a connection"
        assert conn != 123


class TestRealtimeConnectionSendMessage:
    """Test sending messages."""

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending a message."""
        ws = AsyncMock()
        conn = RealtimeConnection(connection_id=uuid4(), websocket=ws)

        message = RealtimeMessage(
            type=RealtimeMessageType.SESSION_CREATED,
            session_id=uuid4(),
        )

        await conn.send_message(message)

        ws.send_json.assert_called_once()
        assert conn.sequence_counter == 1

    @pytest.mark.asyncio
    async def test_send_message_increments_sequence(self):
        """Test sequence counter increments."""
        ws = AsyncMock()
        conn = RealtimeConnection(connection_id=uuid4(), websocket=ws)

        for i in range(3):
            message = RealtimeMessage(
                type=RealtimeMessageType.SESSION_CREATED,
                session_id=uuid4(),
            )
            await conn.send_message(message)

        assert conn.sequence_counter == 3


class TestRealtimeConnectionSendError:
    """Test sending error messages."""

    @pytest.mark.asyncio
    async def test_send_error(self):
        """Test sending an error."""
        ws = AsyncMock()
        session_id = uuid4()
        conn = RealtimeConnection(
            connection_id=uuid4(),
            websocket=ws,
            session_id=session_id,
        )

        await conn.send_error(
            code="INVALID_CONFIG",
            message="Configuration is invalid",
            details={"field": "sample_rate"},
        )

        ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_recoverable(self):
        """Test sending recoverable error."""
        ws = AsyncMock()
        conn = RealtimeConnection(connection_id=uuid4(), websocket=ws)

        await conn.send_error(
            code="RATE_LIMIT",
            message="Too many requests",
            recoverable=True,
        )

        ws.send_json.assert_called_once()


class TestRealtimeConnectionUpdateActivity:
    """Test activity tracking."""

    def test_update_activity(self):
        """Test updating last activity."""
        ws = MagicMock()
        conn = RealtimeConnection(connection_id=uuid4(), websocket=ws)

        old_time = conn.last_activity
        conn.update_activity()
        new_time = conn.last_activity

        assert new_time >= old_time


# ============================================================
# ConnectionManager Tests
# ============================================================


class TestConnectionManagerInit:
    """Test ConnectionManager initialization."""

    def test_manager_init(self):
        """Test manager initializes correctly."""
        mgr = ConnectionManager()

        assert isinstance(mgr._connections, dict)
        assert len(mgr._connections) == 0
        assert mgr.active_connection_count == 0
        assert mgr.active_session_count == 0


class TestConnectionManagerConnect:
    """Test connection handling."""

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test accepting a new connection."""
        mgr = ConnectionManager()
        ws = AsyncMock()

        conn = await mgr.connect(ws)

        assert conn is not None
        assert conn.connection_id in mgr._connections
        assert mgr.active_connection_count == 1
        ws.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_with_user(self):
        """Test connection with user info."""
        mgr = ConnectionManager()
        ws = AsyncMock()
        user_id = uuid4()
        org_id = uuid4()

        conn = await mgr.connect(ws, user_id=user_id, org_id=org_id)

        assert conn.user_id == user_id
        assert conn.org_id == org_id
        assert conn.connection_id in mgr._org_connections.get(org_id, set())

    @pytest.mark.asyncio
    async def test_connect_multiple(self):
        """Test multiple connections."""
        mgr = ConnectionManager()

        for _ in range(3):
            ws = AsyncMock()
            await mgr.connect(ws)

        assert mgr.active_connection_count == 3


class TestConnectionManagerDisconnect:
    """Test disconnection handling."""

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnecting a connection."""
        mgr = ConnectionManager()
        ws = AsyncMock()

        conn = await mgr.connect(ws)
        await mgr.disconnect(conn)

        assert conn.connection_id not in mgr._connections
        assert mgr.active_connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_with_session(self):
        """Test disconnecting removes session mapping."""
        mgr = ConnectionManager()
        ws = AsyncMock()
        session_id = uuid4()

        conn = await mgr.connect(ws)
        await mgr.register_session(conn, session_id, SessionConfig())

        await mgr.disconnect(conn)

        assert session_id not in mgr._session_connections

    @pytest.mark.asyncio
    async def test_disconnect_with_org(self):
        """Test disconnecting removes org mapping."""
        mgr = ConnectionManager()
        ws = AsyncMock()
        org_id = uuid4()

        conn = await mgr.connect(ws, org_id=org_id)
        await mgr.disconnect(conn)

        assert conn.connection_id not in mgr._org_connections.get(org_id, set())


class TestConnectionManagerRegisterSession:
    """Test session registration."""

    @pytest.mark.asyncio
    async def test_register_session(self):
        """Test registering a session."""
        mgr = ConnectionManager()
        ws = AsyncMock()
        session_id = uuid4()

        conn = await mgr.connect(ws)
        config = SessionConfig(
            asr_backend="whisperx",
            enable_transcription=True,
        )
        await mgr.register_session(conn, session_id, config)

        assert conn.session_id == session_id
        assert conn.config == config
        assert mgr.active_session_count == 1

    @pytest.mark.asyncio
    async def test_register_duplicate_session_fails(self):
        """Test registering duplicate session raises error."""
        mgr = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        session_id = uuid4()

        conn1 = await mgr.connect(ws1)
        conn2 = await mgr.connect(ws2)

        await mgr.register_session(conn1, session_id, SessionConfig())

        with pytest.raises(ValueError, match="already active"):
            await mgr.register_session(conn2, session_id, SessionConfig())


class TestConnectionManagerGetConnection:
    """Test connection retrieval."""

    @pytest.mark.asyncio
    async def test_get_connection(self):
        """Test getting connection by ID."""
        mgr = ConnectionManager()
        ws = AsyncMock()

        conn = await mgr.connect(ws)
        retrieved = await mgr.get_connection(conn.connection_id)

        assert retrieved is conn

    @pytest.mark.asyncio
    async def test_get_connection_not_found(self):
        """Test getting non-existent connection."""
        mgr = ConnectionManager()

        retrieved = await mgr.get_connection(uuid4())

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_session_connection(self):
        """Test getting connection by session ID."""
        mgr = ConnectionManager()
        ws = AsyncMock()
        session_id = uuid4()

        conn = await mgr.connect(ws)
        await mgr.register_session(conn, session_id, SessionConfig())

        retrieved = await mgr.get_session_connection(session_id)

        assert retrieved is conn

    @pytest.mark.asyncio
    async def test_get_session_connection_not_found(self):
        """Test getting non-existent session connection."""
        mgr = ConnectionManager()

        retrieved = await mgr.get_session_connection(uuid4())

        assert retrieved is None


class TestConnectionManagerESPSubscription:
    """Test ESP subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_to_esp_public(self):
        """Test subscribing to public ESP."""
        mgr = ConnectionManager()

        # Create target session with public consent
        target_ws = AsyncMock()
        target_conn = await mgr.connect(target_ws)
        target_session_id = uuid4()
        target_config = SessionConfig(esp_consent_level="public")
        await mgr.register_session(target_conn, target_session_id, target_config)

        # Create subscriber
        sub_ws = AsyncMock()
        sub_conn = await mgr.connect(sub_ws)

        result = await mgr.subscribe_to_esp(sub_conn, target_session_id)

        assert result is True
        assert sub_conn.connection_id in target_conn.esp_subscribers

    @pytest.mark.asyncio
    async def test_subscribe_to_esp_self_denied(self):
        """Test subscribing to self-only ESP is denied."""
        mgr = ConnectionManager()

        # Create target session with self consent
        target_ws = AsyncMock()
        target_user = uuid4()
        target_conn = await mgr.connect(target_ws, user_id=target_user)
        target_session_id = uuid4()
        target_config = SessionConfig(esp_consent_level="self")
        await mgr.register_session(target_conn, target_session_id, target_config)

        # Create different user subscriber
        sub_ws = AsyncMock()
        sub_conn = await mgr.connect(sub_ws, user_id=uuid4())

        result = await mgr.subscribe_to_esp(sub_conn, target_session_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_subscribe_to_nonexistent_session(self):
        """Test subscribing to non-existent session."""
        mgr = ConnectionManager()
        ws = AsyncMock()

        conn = await mgr.connect(ws)
        result = await mgr.subscribe_to_esp(conn, uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_unsubscribe_from_esp(self):
        """Test unsubscribing from ESP."""
        mgr = ConnectionManager()

        # Setup subscription
        target_ws = AsyncMock()
        target_conn = await mgr.connect(target_ws)
        target_session_id = uuid4()
        target_config = SessionConfig(esp_consent_level="public")
        await mgr.register_session(target_conn, target_session_id, target_config)

        sub_ws = AsyncMock()
        sub_conn = await mgr.connect(sub_ws)
        await mgr.subscribe_to_esp(sub_conn, target_session_id)

        # Unsubscribe
        await mgr.unsubscribe_from_esp(sub_conn, target_session_id)

        assert sub_conn.connection_id not in target_conn.esp_subscribers


class TestConnectionManagerBroadcast:
    """Test message broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_to_session(self):
        """Test broadcasting to a session."""
        mgr = ConnectionManager()
        ws = AsyncMock()
        session_id = uuid4()

        conn = await mgr.connect(ws)
        await mgr.register_session(conn, session_id, SessionConfig())

        message = RealtimeMessage(
            type=RealtimeMessageType.SESSION_CREATED,
            session_id=session_id,
        )

        result = await mgr.broadcast_to_session(session_id, message)

        assert result is True
        ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_to_nonexistent_session(self):
        """Test broadcasting to non-existent session."""
        mgr = ConnectionManager()

        message = RealtimeMessage(
            type=RealtimeMessageType.SESSION_CREATED,
            session_id=uuid4(),
        )

        result = await mgr.broadcast_to_session(uuid4(), message)

        assert result is False

    @pytest.mark.asyncio
    async def test_broadcast_to_org(self):
        """Test broadcasting to organization."""
        mgr = ConnectionManager()
        org_id = uuid4()

        # Create multiple connections in org
        connections = []
        for _ in range(3):
            ws = AsyncMock()
            conn = await mgr.connect(ws, org_id=org_id)
            connections.append(conn)

        message = RealtimeMessage(
            type=RealtimeMessageType.SESSION_CREATED,
            session_id=uuid4(),
        )

        notified = await mgr.broadcast_to_org(org_id, message)

        assert notified == 3

    @pytest.mark.asyncio
    async def test_broadcast_to_org_with_exclusion(self):
        """Test broadcasting to org with exclusion."""
        mgr = ConnectionManager()
        org_id = uuid4()

        # Create connections
        ws1 = AsyncMock()
        conn1 = await mgr.connect(ws1, org_id=org_id)
        ws2 = AsyncMock()
        conn2 = await mgr.connect(ws2, org_id=org_id)

        message = RealtimeMessage(
            type=RealtimeMessageType.SESSION_CREATED,
            session_id=uuid4(),
        )

        # Exclude first connection
        notified = await mgr.broadcast_to_org(
            org_id, message, exclude_connection=conn1.connection_id
        )

        assert notified == 1
        ws2.send_json.assert_called_once()
        ws1.send_json.assert_not_called()


class TestConnectionManagerStats:
    """Test connection statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        """Test stats with no connections."""
        mgr = ConnectionManager()

        stats = mgr.get_stats()

        assert stats["active_connections"] == 0
        assert stats["active_sessions"] == 0
        assert stats["organizations_connected"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_connections(self):
        """Test stats with active connections."""
        mgr = ConnectionManager()

        org_id = uuid4()
        for _ in range(2):
            ws = AsyncMock()
            await mgr.connect(ws, org_id=org_id)

        ws = AsyncMock()
        conn = await mgr.connect(ws)
        await mgr.register_session(conn, uuid4(), SessionConfig())

        stats = mgr.get_stats()

        assert stats["active_connections"] == 3
        assert stats["active_sessions"] == 1
        assert stats["organizations_connected"] == 1


# ============================================================
# Global Instance Tests
# ============================================================


class TestGlobalManager:
    """Test global manager instance."""

    def test_manager_is_connection_manager(self):
        """Test global manager is a ConnectionManager."""
        assert isinstance(manager, ConnectionManager)

    @pytest.mark.asyncio
    async def test_get_connection_manager(self):
        """Test dependency getter returns manager."""
        result = await get_connection_manager()

        assert result is manager


# ============================================================
# ESP Broadcast Tests
# ============================================================


class TestBroadcastESP:
    """Test ESP broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_esp_disabled(self):
        """Test broadcasting when ESP is disabled."""
        mgr = ConnectionManager()
        ws = AsyncMock()

        conn = await mgr.connect(ws)
        session_id = uuid4()
        config = SessionConfig(esp_enabled=False)
        await mgr.register_session(conn, session_id, config)

        esp = ESPMessage(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            engagement_score=0.5,
            stress_index=0.3,
        )

        result = await mgr.broadcast_esp(conn, esp)

        assert result == 0

    @pytest.mark.asyncio
    async def test_broadcast_esp_enabled(self):
        """Test broadcasting when ESP is enabled."""
        mgr = ConnectionManager()
        ws = AsyncMock()

        conn = await mgr.connect(ws)
        session_id = uuid4()
        config = SessionConfig(esp_enabled=True)
        await mgr.register_session(conn, session_id, config)

        esp = ESPMessage(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            engagement_score=0.5,
            stress_index=0.3,
        )

        result = await mgr.broadcast_esp(conn, esp)

        assert result == 1  # Self notification
        ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_esp_to_subscribers(self):
        """Test broadcasting ESP to subscribers."""
        mgr = ConnectionManager()

        # Create source connection
        source_ws = AsyncMock()
        source_conn = await mgr.connect(source_ws)
        source_session_id = uuid4()
        source_config = SessionConfig(
            esp_enabled=True,
            esp_consent_level="public",
        )
        await mgr.register_session(source_conn, source_session_id, source_config)

        # Create subscriber
        sub_ws = AsyncMock()
        sub_conn = await mgr.connect(sub_ws)
        await mgr.subscribe_to_esp(sub_conn, source_session_id)

        esp = ESPMessage(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            engagement_score=0.5,
            stress_index=0.3,
        )

        result = await mgr.broadcast_esp(source_conn, esp)

        assert result == 2  # Self + subscriber
        source_ws.send_json.assert_called_once()
        sub_ws.send_json.assert_called_once()
