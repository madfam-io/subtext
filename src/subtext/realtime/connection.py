"""
WebSocket Connection Manager

Manages real-time WebSocket connections for audio streaming and ESP broadcasting.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from subtext.core.models import ESPMessage
from .protocol import (
    RealtimeMessage,
    RealtimeMessageType,
    SessionConfig,
    AudioConfig,
    ErrorPayload,
)

logger = structlog.get_logger()


@dataclass
class RealtimeConnection:
    """Represents an active real-time WebSocket connection."""

    connection_id: UUID
    websocket: WebSocket
    session_id: UUID | None = None
    user_id: UUID | None = None
    org_id: UUID | None = None

    # Configuration
    config: SessionConfig = field(default_factory=SessionConfig)

    # State
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    is_streaming: bool = False
    sequence_counter: int = 0

    # Audio buffering
    audio_buffer: bytes = field(default=b"", repr=False)
    samples_received: int = 0
    total_duration_ms: int = 0

    # Subscribers for ESP updates
    esp_subscribers: set[UUID] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.connection_id)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RealtimeConnection):
            return self.connection_id == other.connection_id
        return False

    async def send_message(self, message: RealtimeMessage) -> None:
        """Send a message to the client."""
        try:
            message.sequence = self.sequence_counter
            self.sequence_counter += 1
            await self.websocket.send_json(message.model_dump(mode="json"))
        except Exception as e:
            logger.error(
                "Failed to send WebSocket message",
                connection_id=str(self.connection_id),
                error=str(e),
            )
            raise

    async def send_error(
        self,
        code: str,
        message: str,
        details: dict | None = None,
        recoverable: bool = True,
    ) -> None:
        """Send an error message to the client."""
        error_payload = ErrorPayload(
            code=code,
            message=message,
            details=details or {},
            recoverable=recoverable,
        )
        await self.send_message(
            RealtimeMessage(
                type=RealtimeMessageType.ERROR,
                session_id=self.session_id,
                payload=error_payload.model_dump(),
            )
        )

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


class ConnectionManager:
    """
    Manages all active WebSocket connections.

    Handles connection lifecycle, message routing, and ESP broadcasting.
    """

    def __init__(self) -> None:
        # Active connections by connection_id
        self._connections: dict[UUID, RealtimeConnection] = {}

        # Session to connection mapping (one session = one connection)
        self._session_connections: dict[UUID, UUID] = {}

        # Organization to connections mapping (for org-wide ESP)
        self._org_connections: dict[UUID, set[UUID]] = {}

        # ESP subscription groups
        self._esp_groups: dict[str, set[UUID]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info("ConnectionManager initialized")

    @property
    def active_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connections)

    @property
    def active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._session_connections)

    async def connect(
        self,
        websocket: WebSocket,
        user_id: UUID | None = None,
        org_id: UUID | None = None,
    ) -> RealtimeConnection:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The FastAPI WebSocket instance
            user_id: Authenticated user ID (optional)
            org_id: User's organization ID (optional)

        Returns:
            RealtimeConnection instance
        """
        await websocket.accept()

        connection_id = uuid4()
        connection = RealtimeConnection(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            org_id=org_id,
        )

        async with self._lock:
            self._connections[connection_id] = connection

            # Track org connections for org-wide ESP
            if org_id:
                if org_id not in self._org_connections:
                    self._org_connections[org_id] = set()
                self._org_connections[org_id].add(connection_id)

        logger.info(
            "WebSocket connected",
            connection_id=str(connection_id),
            user_id=str(user_id) if user_id else None,
            org_id=str(org_id) if org_id else None,
        )

        return connection

    async def disconnect(self, connection: RealtimeConnection) -> None:
        """
        Handle connection disconnect.

        Cleans up all references and notifies relevant parties.
        """
        async with self._lock:
            # Remove from connections
            self._connections.pop(connection.connection_id, None)

            # Remove from session mapping
            if connection.session_id:
                self._session_connections.pop(connection.session_id, None)

            # Remove from org connections
            if connection.org_id:
                org_conns = self._org_connections.get(connection.org_id)
                if org_conns:
                    org_conns.discard(connection.connection_id)
                    if not org_conns:
                        del self._org_connections[connection.org_id]

            # Remove from ESP groups
            for group_connections in self._esp_groups.values():
                group_connections.discard(connection.connection_id)

        logger.info(
            "WebSocket disconnected",
            connection_id=str(connection.connection_id),
            session_id=str(connection.session_id) if connection.session_id else None,
            duration_ms=connection.total_duration_ms,
        )

    async def register_session(
        self,
        connection: RealtimeConnection,
        session_id: UUID,
        config: SessionConfig,
    ) -> None:
        """
        Register a session for a connection.

        Args:
            connection: The connection to register
            session_id: The session ID
            config: Session configuration
        """
        async with self._lock:
            # Check if session already exists
            if session_id in self._session_connections:
                raise ValueError(f"Session {session_id} already active")

            connection.session_id = session_id
            connection.config = config
            self._session_connections[session_id] = connection.connection_id

        logger.info(
            "Session registered",
            connection_id=str(connection.connection_id),
            session_id=str(session_id),
            config={
                "asr_backend": config.asr_backend,
                "enable_transcription": config.enable_transcription,
                "enable_emotion": config.enable_emotion,
            },
        )

    async def get_connection(self, connection_id: UUID) -> RealtimeConnection | None:
        """Get a connection by ID."""
        return self._connections.get(connection_id)

    async def get_session_connection(
        self, session_id: UUID
    ) -> RealtimeConnection | None:
        """Get the connection for a session."""
        connection_id = self._session_connections.get(session_id)
        if connection_id:
            return self._connections.get(connection_id)
        return None

    async def subscribe_to_esp(
        self,
        connection: RealtimeConnection,
        target_session_id: UUID,
    ) -> bool:
        """
        Subscribe to ESP updates from another session.

        Args:
            connection: The subscribing connection
            target_session_id: The session to subscribe to

        Returns:
            True if subscription successful
        """
        # Verify target session exists and allows subscription
        target_connection_id = self._session_connections.get(target_session_id)
        if not target_connection_id:
            return False

        target_conn = self._connections.get(target_connection_id)
        if not target_conn:
            return False

        # Check consent level
        consent = target_conn.config.esp_consent_level
        if consent == "self":
            # Only the session owner can see ESP
            if connection.user_id != target_conn.user_id:
                return False
        elif consent == "team":
            # Only same org can see ESP
            if connection.org_id != target_conn.org_id:
                return False
        # "org" and "public" allow broader access

        # Add subscriber
        target_conn.esp_subscribers.add(connection.connection_id)

        logger.info(
            "ESP subscription added",
            subscriber=str(connection.connection_id),
            target_session=str(target_session_id),
        )

        return True

    async def unsubscribe_from_esp(
        self,
        connection: RealtimeConnection,
        target_session_id: UUID,
    ) -> None:
        """Unsubscribe from ESP updates."""
        target_connection_id = self._session_connections.get(target_session_id)
        if target_connection_id:
            target_conn = self._connections.get(target_connection_id)
            if target_conn:
                target_conn.esp_subscribers.discard(connection.connection_id)

    async def broadcast_esp(
        self,
        source_connection: RealtimeConnection,
        esp_message: ESPMessage,
    ) -> int:
        """
        Broadcast ESP update to subscribers.

        Args:
            source_connection: The connection generating the ESP
            esp_message: The ESP message to broadcast

        Returns:
            Number of subscribers notified
        """
        if not source_connection.config.esp_enabled:
            return 0

        message = RealtimeMessage(
            type=RealtimeMessageType.ESP_UPDATE,
            session_id=source_connection.session_id,
            payload=esp_message.model_dump(mode="json"),
        )

        notified = 0

        # Always send to self
        try:
            await source_connection.send_message(message)
            notified += 1
        except Exception as e:
            logger.error("Failed to send ESP to self", error=str(e))

        # Send to subscribers
        for subscriber_id in source_connection.esp_subscribers:
            subscriber = self._connections.get(subscriber_id)
            if subscriber:
                try:
                    await subscriber.send_message(message)
                    notified += 1
                except Exception as e:
                    logger.error(
                        "Failed to send ESP to subscriber",
                        subscriber_id=str(subscriber_id),
                        error=str(e),
                    )

        return notified

    async def broadcast_to_session(
        self,
        session_id: UUID,
        message: RealtimeMessage,
    ) -> bool:
        """
        Send a message to a specific session.

        Returns True if message was sent successfully.
        """
        connection = await self.get_session_connection(session_id)
        if connection:
            try:
                await connection.send_message(message)
                return True
            except Exception as e:
                logger.error(
                    "Failed to broadcast to session",
                    session_id=str(session_id),
                    error=str(e),
                )
        return False

    async def broadcast_to_org(
        self,
        org_id: UUID,
        message: RealtimeMessage,
        exclude_connection: UUID | None = None,
    ) -> int:
        """
        Broadcast a message to all connections in an organization.

        Returns number of connections notified.
        """
        connection_ids = self._org_connections.get(org_id, set())
        notified = 0

        for conn_id in connection_ids:
            if conn_id == exclude_connection:
                continue

            connection = self._connections.get(conn_id)
            if connection:
                try:
                    await connection.send_message(message)
                    notified += 1
                except Exception:
                    pass

        return notified

    def get_stats(self) -> dict[str, Any]:
        """Get connection manager statistics."""
        return {
            "active_connections": len(self._connections),
            "active_sessions": len(self._session_connections),
            "organizations_connected": len(self._org_connections),
            "esp_groups": len(self._esp_groups),
        }


# Global connection manager instance
manager = ConnectionManager()


async def get_connection_manager() -> ConnectionManager:
    """Dependency to get the connection manager."""
    return manager
