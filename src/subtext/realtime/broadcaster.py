"""
ESP (Emotional State Protocol) Broadcaster

Handles broadcasting of emotional state updates to subscribers
with privacy controls and rate limiting.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import UUID

import structlog

from subtext.core.models import ESPMessage
from .protocol import RealtimeMessage, RealtimeMessageType, ESPPayload

logger = structlog.get_logger()


@dataclass
class ESPSubscription:
    """Represents a subscription to ESP updates."""

    subscriber_id: UUID
    source_session_id: UUID
    consent_level: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime | None = None
    updates_received: int = 0


@dataclass
class ESPChannel:
    """A channel for ESP broadcasting."""

    session_id: UUID
    owner_id: UUID | None
    org_id: UUID | None

    # Privacy settings
    consent_level: str = "self"  # self, team, org, public
    rate_limit_ms: int = 500

    # State
    is_active: bool = True
    last_broadcast: datetime | None = None
    broadcast_count: int = 0

    # Subscribers
    subscribers: dict[UUID, ESPSubscription] = field(default_factory=dict)

    # Latest state for new subscribers
    latest_esp: ESPMessage | None = None


class ESPBroadcaster:
    """
    Manages ESP broadcasting with privacy-aware subscription model.

    Features:
    - Privacy-controlled subscriptions (self, team, org, public)
    - Rate-limited broadcasting
    - Subscription management
    - Aggregate ESP generation for groups
    """

    def __init__(self) -> None:
        # Active ESP channels by session_id
        self._channels: dict[UUID, ESPChannel] = {}

        # Index: subscriber_id -> list of subscribed session_ids
        self._subscriber_sessions: dict[UUID, set[UUID]] = defaultdict(set)

        # Callback registry for sending messages
        self._send_callbacks: dict[UUID, Callable] = {}

        # Lock for thread-safety
        self._lock = asyncio.Lock()

        # Background task for cleanup
        self._cleanup_task: asyncio.Task | None = None

        logger.info("ESPBroadcaster initialized")

    async def start(self) -> None:
        """Start the broadcaster background tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("ESPBroadcaster started")

    async def stop(self) -> None:
        """Stop the broadcaster."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("ESPBroadcaster stopped")

    async def create_channel(
        self,
        session_id: UUID,
        owner_id: UUID | None = None,
        org_id: UUID | None = None,
        consent_level: str = "self",
        rate_limit_ms: int = 500,
    ) -> ESPChannel:
        """
        Create a new ESP broadcast channel for a session.

        Args:
            session_id: The realtime session ID
            owner_id: User ID of the session owner
            org_id: Organization ID
            consent_level: Privacy level (self, team, org, public)
            rate_limit_ms: Minimum interval between broadcasts

        Returns:
            The created ESPChannel
        """
        async with self._lock:
            if session_id in self._channels:
                raise ValueError(f"Channel already exists for session {session_id}")

            channel = ESPChannel(
                session_id=session_id,
                owner_id=owner_id,
                org_id=org_id,
                consent_level=consent_level,
                rate_limit_ms=rate_limit_ms,
            )

            self._channels[session_id] = channel

            logger.info(
                "ESP channel created",
                session_id=str(session_id),
                consent_level=consent_level,
            )

            return channel

    async def close_channel(self, session_id: UUID) -> None:
        """Close an ESP channel and notify subscribers."""
        async with self._lock:
            channel = self._channels.pop(session_id, None)

            if channel:
                # Notify subscribers
                for subscriber_id in channel.subscribers:
                    self._subscriber_sessions[subscriber_id].discard(session_id)

                logger.info(
                    "ESP channel closed",
                    session_id=str(session_id),
                    broadcast_count=channel.broadcast_count,
                )

    async def subscribe(
        self,
        subscriber_id: UUID,
        session_id: UUID,
        subscriber_org_id: UUID | None = None,
        subscriber_user_id: UUID | None = None,
    ) -> bool:
        """
        Subscribe to ESP updates from a session.

        Returns True if subscription was successful.
        """
        async with self._lock:
            channel = self._channels.get(session_id)
            if not channel or not channel.is_active:
                return False

            # Check consent level
            if not self._check_consent(
                channel,
                subscriber_org_id,
                subscriber_user_id,
            ):
                logger.warning(
                    "ESP subscription denied - insufficient consent",
                    subscriber_id=str(subscriber_id),
                    session_id=str(session_id),
                    consent_level=channel.consent_level,
                )
                return False

            # Create subscription
            subscription = ESPSubscription(
                subscriber_id=subscriber_id,
                source_session_id=session_id,
                consent_level=channel.consent_level,
            )

            channel.subscribers[subscriber_id] = subscription
            self._subscriber_sessions[subscriber_id].add(session_id)

            logger.info(
                "ESP subscription created",
                subscriber_id=str(subscriber_id),
                session_id=str(session_id),
            )

            # Send latest state if available
            if channel.latest_esp:
                await self._send_to_subscriber(
                    subscriber_id,
                    session_id,
                    channel.latest_esp,
                )

            return True

    async def unsubscribe(
        self,
        subscriber_id: UUID,
        session_id: UUID,
    ) -> None:
        """Unsubscribe from ESP updates."""
        async with self._lock:
            channel = self._channels.get(session_id)
            if channel:
                channel.subscribers.pop(subscriber_id, None)

            self._subscriber_sessions[subscriber_id].discard(session_id)

            logger.debug(
                "ESP subscription removed",
                subscriber_id=str(subscriber_id),
                session_id=str(session_id),
            )

    async def unsubscribe_all(self, subscriber_id: UUID) -> None:
        """Unsubscribe from all ESP channels."""
        async with self._lock:
            session_ids = list(self._subscriber_sessions.get(subscriber_id, set()))

            for session_id in session_ids:
                channel = self._channels.get(session_id)
                if channel:
                    channel.subscribers.pop(subscriber_id, None)

            self._subscriber_sessions.pop(subscriber_id, None)

    def _check_consent(
        self,
        channel: ESPChannel,
        subscriber_org_id: UUID | None,
        subscriber_user_id: UUID | None,
    ) -> bool:
        """Check if subscriber has consent to receive ESP."""
        consent = channel.consent_level

        if consent == "public":
            return True

        if consent == "org":
            return subscriber_org_id == channel.org_id

        if consent == "team":
            # Team consent requires same org
            return subscriber_org_id == channel.org_id

        if consent == "self":
            return subscriber_user_id == channel.owner_id

        return False

    async def broadcast(
        self,
        session_id: UUID,
        esp: ESPMessage,
        force: bool = False,
    ) -> int:
        """
        Broadcast ESP update to all subscribers.

        Args:
            session_id: The source session ID
            esp: The ESP message to broadcast
            force: Bypass rate limiting

        Returns:
            Number of subscribers notified
        """
        channel = self._channels.get(session_id)
        if not channel or not channel.is_active:
            return 0

        # Rate limiting
        if not force and channel.last_broadcast:
            elapsed = (datetime.utcnow() - channel.last_broadcast).total_seconds() * 1000
            if elapsed < channel.rate_limit_ms:
                return 0

        # Update channel state
        channel.latest_esp = esp
        channel.last_broadcast = datetime.utcnow()
        channel.broadcast_count += 1

        # Send to all subscribers
        notified = 0
        for subscriber_id, subscription in channel.subscribers.items():
            try:
                await self._send_to_subscriber(subscriber_id, session_id, esp)
                subscription.last_update = datetime.utcnow()
                subscription.updates_received += 1
                notified += 1
            except Exception as e:
                logger.error(
                    "Failed to send ESP to subscriber",
                    subscriber_id=str(subscriber_id),
                    error=str(e),
                )

        return notified

    async def _send_to_subscriber(
        self,
        subscriber_id: UUID,
        session_id: UUID,
        esp: ESPMessage,
    ) -> None:
        """Send ESP message to a subscriber."""
        callback = self._send_callbacks.get(subscriber_id)
        if callback:
            message = RealtimeMessage(
                type=RealtimeMessageType.ESP_UPDATE,
                session_id=session_id,
                payload=esp.model_dump(mode="json"),
            )
            await callback(message)

    def register_send_callback(
        self,
        subscriber_id: UUID,
        callback: Callable,
    ) -> None:
        """Register a callback for sending messages to a subscriber."""
        self._send_callbacks[subscriber_id] = callback

    def unregister_send_callback(self, subscriber_id: UUID) -> None:
        """Unregister a subscriber's send callback."""
        self._send_callbacks.pop(subscriber_id, None)

    async def get_aggregate_esp(
        self,
        session_ids: list[UUID],
    ) -> ESPMessage | None:
        """
        Generate aggregate ESP from multiple sessions.

        Useful for team/meeting-level emotional state.
        """
        if not session_ids:
            return None

        esp_messages = []
        for session_id in session_ids:
            channel = self._channels.get(session_id)
            if channel and channel.latest_esp:
                esp_messages.append(channel.latest_esp)

        if not esp_messages:
            return None

        # Aggregate by averaging
        n = len(esp_messages)
        valence = sum(e.valence for e in esp_messages) / n
        arousal = sum(e.arousal for e in esp_messages) / n
        dominance = sum(e.dominance for e in esp_messages) / n
        engagement = sum(e.engagement_score for e in esp_messages) / n
        stress = sum(e.stress_index for e in esp_messages) / n

        # Aggregate signals
        all_signals = []
        for esp in esp_messages:
            all_signals.extend(esp.signals)

        return ESPMessage(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            engagement_score=engagement,
            stress_index=stress,
            signals=all_signals[:10],  # Limit signals
            consent_level="aggregate",
        )

    async def _cleanup_loop(self) -> None:
        """Background task to clean up stale channels."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                async with self._lock:
                    stale_threshold = datetime.utcnow()

                    # Find stale channels (no broadcast in 5 minutes)
                    stale_sessions = []
                    for session_id, channel in self._channels.items():
                        if channel.last_broadcast:
                            elapsed = (stale_threshold - channel.last_broadcast).total_seconds()
                            if elapsed > 300:  # 5 minutes
                                stale_sessions.append(session_id)

                    # Remove stale channels
                    for session_id in stale_sessions:
                        await self.close_channel(session_id)

                    if stale_sessions:
                        logger.info(
                            "Cleaned up stale ESP channels",
                            count=len(stale_sessions),
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ESP cleanup error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get broadcaster statistics."""
        total_subscribers = sum(
            len(ch.subscribers) for ch in self._channels.values()
        )
        total_broadcasts = sum(
            ch.broadcast_count for ch in self._channels.values()
        )

        return {
            "active_channels": len(self._channels),
            "total_subscribers": total_subscribers,
            "total_broadcasts": total_broadcasts,
            "channels_by_consent": {
                "self": sum(1 for ch in self._channels.values() if ch.consent_level == "self"),
                "team": sum(1 for ch in self._channels.values() if ch.consent_level == "team"),
                "org": sum(1 for ch in self._channels.values() if ch.consent_level == "org"),
                "public": sum(1 for ch in self._channels.values() if ch.consent_level == "public"),
            },
        }


# Global broadcaster instance
broadcaster = ESPBroadcaster()


async def get_esp_broadcaster() -> ESPBroadcaster:
    """Dependency to get the ESP broadcaster."""
    return broadcaster
