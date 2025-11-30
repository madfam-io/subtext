"""
Unit Tests for ESP Broadcaster Module

Tests the ESP (Emotional State Protocol) broadcasting system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from subtext.core.models import ESPMessage
from subtext.realtime.broadcaster import (
    ESPSubscription,
    ESPChannel,
    ESPBroadcaster,
    broadcaster,
    get_esp_broadcaster,
)


# ============================================================
# ESPSubscription Tests
# ============================================================


class TestESPSubscription:
    """Test ESPSubscription dataclass."""

    def test_subscription_creation(self):
        """Test creating an ESP subscription."""
        subscriber_id = uuid4()
        session_id = uuid4()

        subscription = ESPSubscription(
            subscriber_id=subscriber_id,
            source_session_id=session_id,
            consent_level="org",
        )

        assert subscription.subscriber_id == subscriber_id
        assert subscription.source_session_id == session_id
        assert subscription.consent_level == "org"
        assert subscription.updates_received == 0
        assert subscription.last_update is None
        assert isinstance(subscription.created_at, datetime)

    def test_subscription_with_all_fields(self):
        """Test subscription with all fields set."""
        subscriber_id = uuid4()
        session_id = uuid4()
        created_at = datetime(2024, 1, 1, 12, 0, 0)
        last_update = datetime(2024, 1, 1, 12, 5, 0)

        subscription = ESPSubscription(
            subscriber_id=subscriber_id,
            source_session_id=session_id,
            consent_level="team",
            created_at=created_at,
            last_update=last_update,
            updates_received=5,
        )

        assert subscription.created_at == created_at
        assert subscription.last_update == last_update
        assert subscription.updates_received == 5


# ============================================================
# ESPChannel Tests
# ============================================================


class TestESPChannel:
    """Test ESPChannel dataclass."""

    def test_channel_creation_minimal(self):
        """Test creating a channel with minimal fields."""
        session_id = uuid4()

        channel = ESPChannel(
            session_id=session_id,
            owner_id=None,
            org_id=None,
        )

        assert channel.session_id == session_id
        assert channel.consent_level == "self"
        assert channel.rate_limit_ms == 500
        assert channel.is_active is True
        assert channel.broadcast_count == 0
        assert isinstance(channel.subscribers, dict)
        assert len(channel.subscribers) == 0

    def test_channel_creation_full(self):
        """Test creating a channel with all fields."""
        session_id = uuid4()
        owner_id = uuid4()
        org_id = uuid4()

        channel = ESPChannel(
            session_id=session_id,
            owner_id=owner_id,
            org_id=org_id,
            consent_level="public",
            rate_limit_ms=250,
            is_active=True,
        )

        assert channel.owner_id == owner_id
        assert channel.org_id == org_id
        assert channel.consent_level == "public"
        assert channel.rate_limit_ms == 250

    def test_channel_with_subscribers(self):
        """Test channel with pre-populated subscribers."""
        session_id = uuid4()
        subscriber_id = uuid4()

        subscription = ESPSubscription(
            subscriber_id=subscriber_id,
            source_session_id=session_id,
            consent_level="org",
        )

        channel = ESPChannel(
            session_id=session_id,
            owner_id=None,
            org_id=None,
            subscribers={subscriber_id: subscription},
        )

        assert len(channel.subscribers) == 1
        assert subscriber_id in channel.subscribers


# ============================================================
# ESPBroadcaster Tests
# ============================================================


class TestESPBroadcasterInit:
    """Test ESPBroadcaster initialization."""

    def test_broadcaster_init(self):
        """Test broadcaster initializes correctly."""
        bc = ESPBroadcaster()

        assert isinstance(bc._channels, dict)
        assert len(bc._channels) == 0
        assert isinstance(bc._send_callbacks, dict)

    @pytest.mark.asyncio
    async def test_broadcaster_start_stop(self):
        """Test broadcaster start and stop."""
        bc = ESPBroadcaster()

        await bc.start()
        assert bc._cleanup_task is not None

        await bc.stop()
        # Cleanup task should be cancelled


class TestESPBroadcasterChannels:
    """Test ESP channel management."""

    @pytest.mark.asyncio
    async def test_create_channel(self):
        """Test creating a new channel."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        owner_id = uuid4()
        org_id = uuid4()

        channel = await bc.create_channel(
            session_id=session_id,
            owner_id=owner_id,
            org_id=org_id,
            consent_level="org",
            rate_limit_ms=1000,
        )

        assert channel.session_id == session_id
        assert channel.owner_id == owner_id
        assert channel.org_id == org_id
        assert channel.consent_level == "org"
        assert session_id in bc._channels

    @pytest.mark.asyncio
    async def test_create_channel_duplicate(self):
        """Test creating a duplicate channel raises error."""
        bc = ESPBroadcaster()
        session_id = uuid4()

        await bc.create_channel(session_id=session_id)

        with pytest.raises(ValueError, match="already exists"):
            await bc.create_channel(session_id=session_id)

    @pytest.mark.asyncio
    async def test_close_channel(self):
        """Test closing a channel."""
        bc = ESPBroadcaster()
        session_id = uuid4()

        await bc.create_channel(session_id=session_id)
        assert session_id in bc._channels

        await bc.close_channel(session_id)
        assert session_id not in bc._channels

    @pytest.mark.asyncio
    async def test_close_channel_with_subscribers(self):
        """Test closing a channel removes subscriber references."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()

        channel = await bc.create_channel(
            session_id=session_id,
            consent_level="public",
        )

        # Add subscriber directly
        subscription = ESPSubscription(
            subscriber_id=subscriber_id,
            source_session_id=session_id,
            consent_level="public",
        )
        channel.subscribers[subscriber_id] = subscription
        bc._subscriber_sessions[subscriber_id].add(session_id)

        await bc.close_channel(session_id)

        assert session_id not in bc._subscriber_sessions.get(subscriber_id, set())

    @pytest.mark.asyncio
    async def test_close_nonexistent_channel(self):
        """Test closing a nonexistent channel doesn't error."""
        bc = ESPBroadcaster()
        session_id = uuid4()

        # Should not raise
        await bc.close_channel(session_id)


class TestESPBroadcasterSubscriptions:
    """Test ESP subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_public(self):
        """Test subscribing to a public channel."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()

        await bc.create_channel(
            session_id=session_id,
            consent_level="public",
        )

        result = await bc.subscribe(subscriber_id, session_id)

        assert result is True
        assert subscriber_id in bc._channels[session_id].subscribers
        assert session_id in bc._subscriber_sessions[subscriber_id]

    @pytest.mark.asyncio
    async def test_subscribe_org_same_org(self):
        """Test subscribing to org channel from same org."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()
        org_id = uuid4()

        await bc.create_channel(
            session_id=session_id,
            org_id=org_id,
            consent_level="org",
        )

        result = await bc.subscribe(
            subscriber_id,
            session_id,
            subscriber_org_id=org_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_subscribe_org_different_org(self):
        """Test subscribing to org channel from different org."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()
        channel_org_id = uuid4()
        subscriber_org_id = uuid4()

        await bc.create_channel(
            session_id=session_id,
            org_id=channel_org_id,
            consent_level="org",
        )

        result = await bc.subscribe(
            subscriber_id,
            session_id,
            subscriber_org_id=subscriber_org_id,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_subscribe_self_consent(self):
        """Test subscribing to self-only channel."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        owner_id = uuid4()
        subscriber_id = uuid4()

        await bc.create_channel(
            session_id=session_id,
            owner_id=owner_id,
            consent_level="self",
        )

        # Different user should be denied
        result = await bc.subscribe(
            subscriber_id,
            session_id,
            subscriber_user_id=subscriber_id,
        )
        assert result is False

        # Same user (owner) should be allowed
        result = await bc.subscribe(
            owner_id,
            session_id,
            subscriber_user_id=owner_id,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_subscribe_nonexistent_channel(self):
        """Test subscribing to nonexistent channel."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()

        result = await bc.subscribe(subscriber_id, session_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_subscribe_inactive_channel(self):
        """Test subscribing to inactive channel."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()

        channel = await bc.create_channel(
            session_id=session_id,
            consent_level="public",
        )
        channel.is_active = False

        result = await bc.subscribe(subscriber_id, session_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from a channel."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()

        await bc.create_channel(
            session_id=session_id,
            consent_level="public",
        )
        await bc.subscribe(subscriber_id, session_id)

        await bc.unsubscribe(subscriber_id, session_id)

        assert subscriber_id not in bc._channels[session_id].subscribers
        assert session_id not in bc._subscriber_sessions.get(subscriber_id, set())

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self):
        """Test unsubscribing from all channels."""
        bc = ESPBroadcaster()
        session_id_1 = uuid4()
        session_id_2 = uuid4()
        subscriber_id = uuid4()

        await bc.create_channel(session_id=session_id_1, consent_level="public")
        await bc.create_channel(session_id=session_id_2, consent_level="public")

        await bc.subscribe(subscriber_id, session_id_1)
        await bc.subscribe(subscriber_id, session_id_2)

        await bc.unsubscribe_all(subscriber_id)

        assert subscriber_id not in bc._channels[session_id_1].subscribers
        assert subscriber_id not in bc._channels[session_id_2].subscribers
        assert subscriber_id not in bc._subscriber_sessions


class TestESPBroadcasterConsentCheck:
    """Test consent checking logic."""

    def test_check_consent_public(self):
        """Test public consent allows anyone."""
        bc = ESPBroadcaster()
        channel = ESPChannel(
            session_id=uuid4(),
            owner_id=uuid4(),
            org_id=uuid4(),
            consent_level="public",
        )

        result = bc._check_consent(channel, None, None)
        assert result is True

    def test_check_consent_org_match(self):
        """Test org consent with matching org."""
        bc = ESPBroadcaster()
        org_id = uuid4()
        channel = ESPChannel(
            session_id=uuid4(),
            owner_id=uuid4(),
            org_id=org_id,
            consent_level="org",
        )

        result = bc._check_consent(channel, org_id, None)
        assert result is True

    def test_check_consent_org_mismatch(self):
        """Test org consent with different org."""
        bc = ESPBroadcaster()
        channel = ESPChannel(
            session_id=uuid4(),
            owner_id=uuid4(),
            org_id=uuid4(),
            consent_level="org",
        )

        result = bc._check_consent(channel, uuid4(), None)
        assert result is False

    def test_check_consent_team_match(self):
        """Test team consent with matching org."""
        bc = ESPBroadcaster()
        org_id = uuid4()
        channel = ESPChannel(
            session_id=uuid4(),
            owner_id=uuid4(),
            org_id=org_id,
            consent_level="team",
        )

        result = bc._check_consent(channel, org_id, None)
        assert result is True

    def test_check_consent_self_match(self):
        """Test self consent with matching user."""
        bc = ESPBroadcaster()
        owner_id = uuid4()
        channel = ESPChannel(
            session_id=uuid4(),
            owner_id=owner_id,
            org_id=uuid4(),
            consent_level="self",
        )

        result = bc._check_consent(channel, None, owner_id)
        assert result is True

    def test_check_consent_self_mismatch(self):
        """Test self consent with different user."""
        bc = ESPBroadcaster()
        channel = ESPChannel(
            session_id=uuid4(),
            owner_id=uuid4(),
            org_id=uuid4(),
            consent_level="self",
        )

        result = bc._check_consent(channel, None, uuid4())
        assert result is False

    def test_check_consent_unknown_level(self):
        """Test unknown consent level defaults to deny."""
        bc = ESPBroadcaster()
        channel = ESPChannel(
            session_id=uuid4(),
            owner_id=uuid4(),
            org_id=uuid4(),
            consent_level="unknown",
        )

        result = bc._check_consent(channel, uuid4(), uuid4())
        assert result is False


class TestESPBroadcasterBroadcast:
    """Test ESP broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_no_channel(self):
        """Test broadcasting to nonexistent channel."""
        bc = ESPBroadcaster()

        esp = ESPMessage(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            engagement_score=0.5,
            stress_index=0.3,
        )

        result = await bc.broadcast(uuid4(), esp)
        assert result == 0

    @pytest.mark.asyncio
    async def test_broadcast_inactive_channel(self):
        """Test broadcasting to inactive channel."""
        bc = ESPBroadcaster()
        session_id = uuid4()

        channel = await bc.create_channel(session_id=session_id)
        channel.is_active = False

        esp = ESPMessage(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            engagement_score=0.5,
            stress_index=0.3,
        )

        result = await bc.broadcast(session_id, esp)
        assert result == 0

    @pytest.mark.asyncio
    async def test_broadcast_rate_limited(self):
        """Test broadcast rate limiting."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()

        channel = await bc.create_channel(
            session_id=session_id,
            consent_level="public",
            rate_limit_ms=1000,
        )
        await bc.subscribe(subscriber_id, session_id)

        esp = ESPMessage(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            engagement_score=0.5,
            stress_index=0.3,
        )

        # First broadcast succeeds
        callback = AsyncMock()
        bc.register_send_callback(subscriber_id, callback)

        result1 = await bc.broadcast(session_id, esp)
        assert result1 == 1

        # Second broadcast within rate limit should be skipped
        result2 = await bc.broadcast(session_id, esp)
        assert result2 == 0

    @pytest.mark.asyncio
    async def test_broadcast_force_bypasses_rate_limit(self):
        """Test force broadcast bypasses rate limiting."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()

        channel = await bc.create_channel(
            session_id=session_id,
            consent_level="public",
            rate_limit_ms=1000,
        )
        await bc.subscribe(subscriber_id, session_id)

        esp = ESPMessage(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            engagement_score=0.5,
            stress_index=0.3,
        )

        callback = AsyncMock()
        bc.register_send_callback(subscriber_id, callback)

        await bc.broadcast(session_id, esp)
        result = await bc.broadcast(session_id, esp, force=True)

        assert result == 1
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_broadcast_updates_channel_state(self):
        """Test broadcast updates channel state."""
        bc = ESPBroadcaster()
        session_id = uuid4()

        channel = await bc.create_channel(
            session_id=session_id,
            consent_level="public",
        )

        esp = ESPMessage(
            valence=0.7,
            arousal=0.6,
            dominance=0.5,
            engagement_score=0.8,
            stress_index=0.2,
        )

        await bc.broadcast(session_id, esp, force=True)

        assert channel.latest_esp == esp
        assert channel.last_broadcast is not None
        assert channel.broadcast_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_subscribers(self):
        """Test broadcasting to multiple subscribers."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_ids = [uuid4() for _ in range(3)]

        await bc.create_channel(
            session_id=session_id,
            consent_level="public",
        )

        callbacks = []
        for sub_id in subscriber_ids:
            await bc.subscribe(sub_id, session_id)
            callback = AsyncMock()
            bc.register_send_callback(sub_id, callback)
            callbacks.append(callback)

        esp = ESPMessage(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            engagement_score=0.5,
            stress_index=0.3,
        )

        result = await bc.broadcast(session_id, esp)

        assert result == 3
        for callback in callbacks:
            callback.assert_called_once()


class TestESPBroadcasterCallbacks:
    """Test send callback management."""

    def test_register_send_callback(self):
        """Test registering a send callback."""
        bc = ESPBroadcaster()
        subscriber_id = uuid4()
        callback = AsyncMock()

        bc.register_send_callback(subscriber_id, callback)

        assert subscriber_id in bc._send_callbacks
        assert bc._send_callbacks[subscriber_id] == callback

    def test_unregister_send_callback(self):
        """Test unregistering a send callback."""
        bc = ESPBroadcaster()
        subscriber_id = uuid4()
        callback = AsyncMock()

        bc.register_send_callback(subscriber_id, callback)
        bc.unregister_send_callback(subscriber_id)

        assert subscriber_id not in bc._send_callbacks

    def test_unregister_nonexistent_callback(self):
        """Test unregistering nonexistent callback doesn't error."""
        bc = ESPBroadcaster()
        subscriber_id = uuid4()

        # Should not raise
        bc.unregister_send_callback(subscriber_id)


class TestESPBroadcasterAggregate:
    """Test aggregate ESP generation."""

    @pytest.mark.asyncio
    async def test_aggregate_empty_list(self):
        """Test aggregate with empty session list."""
        bc = ESPBroadcaster()

        result = await bc.get_aggregate_esp([])

        assert result is None

    @pytest.mark.asyncio
    async def test_aggregate_no_esp_data(self):
        """Test aggregate when no channels have ESP data."""
        bc = ESPBroadcaster()
        session_id = uuid4()

        await bc.create_channel(session_id=session_id)

        result = await bc.get_aggregate_esp([session_id])

        assert result is None

    @pytest.mark.asyncio
    async def test_aggregate_single_session(self):
        """Test aggregate with single session."""
        bc = ESPBroadcaster()
        session_id = uuid4()

        channel = await bc.create_channel(session_id=session_id)

        esp = ESPMessage(
            valence=0.8,
            arousal=0.6,
            dominance=0.5,
            engagement_score=0.9,
            stress_index=0.2,
        )
        channel.latest_esp = esp

        result = await bc.get_aggregate_esp([session_id])

        assert result is not None
        assert result.valence == 0.8
        assert result.arousal == 0.6
        assert result.engagement_score == 0.9

    @pytest.mark.asyncio
    async def test_aggregate_multiple_sessions(self):
        """Test aggregate with multiple sessions."""
        bc = ESPBroadcaster()
        session_ids = [uuid4(), uuid4(), uuid4()]

        for i, session_id in enumerate(session_ids):
            channel = await bc.create_channel(session_id=session_id)
            channel.latest_esp = ESPMessage(
                valence=0.3 * (i + 1),  # 0.3, 0.6, 0.9
                arousal=0.2 * (i + 1),  # 0.2, 0.4, 0.6
                dominance=0.5,
                engagement_score=0.5,
                stress_index=0.2,
            )

        result = await bc.get_aggregate_esp(session_ids)

        assert result is not None
        assert result.consent_level == "aggregate"
        # Average of 0.3, 0.6, 0.9 = 0.6
        assert abs(result.valence - 0.6) < 0.01


class TestESPBroadcasterStats:
    """Test broadcaster statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        """Test stats with no channels."""
        bc = ESPBroadcaster()

        stats = bc.get_stats()

        assert stats["active_channels"] == 0
        assert stats["total_subscribers"] == 0
        assert stats["total_broadcasts"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_channels(self):
        """Test stats with active channels."""
        bc = ESPBroadcaster()

        await bc.create_channel(uuid4(), consent_level="self")
        await bc.create_channel(uuid4(), consent_level="public")
        await bc.create_channel(uuid4(), consent_level="org")

        stats = bc.get_stats()

        assert stats["active_channels"] == 3
        assert stats["channels_by_consent"]["self"] == 1
        assert stats["channels_by_consent"]["public"] == 1
        assert stats["channels_by_consent"]["org"] == 1


# ============================================================
# Global Instance Tests
# ============================================================


class TestGlobalBroadcaster:
    """Test global broadcaster instance."""

    def test_broadcaster_is_esp_broadcaster(self):
        """Test global broadcaster is an ESPBroadcaster."""
        assert isinstance(broadcaster, ESPBroadcaster)

    @pytest.mark.asyncio
    async def test_get_esp_broadcaster(self):
        """Test dependency getter returns broadcaster."""
        result = await get_esp_broadcaster()

        assert result is broadcaster


# ============================================================
# Subscribe with Latest ESP Tests
# ============================================================


class TestSubscribeWithLatestESP:
    """Test subscribing sends latest ESP."""

    @pytest.mark.asyncio
    async def test_subscribe_sends_latest_esp(self):
        """Test new subscriber receives latest ESP."""
        bc = ESPBroadcaster()
        session_id = uuid4()
        subscriber_id = uuid4()

        channel = await bc.create_channel(
            session_id=session_id,
            consent_level="public",
        )

        # Set latest ESP
        esp = ESPMessage(
            valence=0.7,
            arousal=0.5,
            dominance=0.5,
            engagement_score=0.6,
            stress_index=0.3,
        )
        channel.latest_esp = esp

        # Register callback before subscribing
        callback = AsyncMock()
        bc.register_send_callback(subscriber_id, callback)

        await bc.subscribe(subscriber_id, session_id)

        # Should have sent latest ESP
        callback.assert_called_once()
