"""
Unit Tests for Redis Module

Tests Redis connection management.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================
# init_redis Tests
# ============================================================


class TestInitRedis:
    """Test Redis initialization."""

    @pytest.mark.asyncio
    async def test_init_redis_success(self):
        """Test successful Redis initialization."""
        import subtext.db.redis as redis_module

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_client) as mock_from_url:
            with patch.object(redis_module, "_redis_client", None):
                await redis_module.init_redis()

            mock_from_url.assert_called_once()
            mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_redis_configures_client(self):
        """Test init_redis configures client correctly."""
        import subtext.db.redis as redis_module

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_client) as mock_from_url:
            with patch.object(redis_module, "_redis_client", None):
                await redis_module.init_redis()

            # Check that encoding options are set
            call_kwargs = mock_from_url.call_args
            assert call_kwargs.kwargs.get("encoding") == "utf-8"
            assert call_kwargs.kwargs.get("decode_responses") is True


# ============================================================
# close_redis Tests
# ============================================================


class TestCloseRedis:
    """Test Redis connection closing."""

    @pytest.mark.asyncio
    async def test_close_redis_with_client(self):
        """Test closing Redis with an active client."""
        import subtext.db.redis as redis_module

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        # Store original value
        original_client = redis_module._redis_client

        try:
            redis_module._redis_client = mock_client
            await redis_module.close_redis()

            mock_client.close.assert_called_once()
        finally:
            # Restore original
            redis_module._redis_client = original_client

    @pytest.mark.asyncio
    async def test_close_redis_without_client(self):
        """Test closing Redis when no client exists."""
        import subtext.db.redis as redis_module

        # Store original
        original_client = redis_module._redis_client

        try:
            redis_module._redis_client = None
            # Should not raise
            await redis_module.close_redis()
        finally:
            redis_module._redis_client = original_client


# ============================================================
# get_redis Tests
# ============================================================


class TestGetRedis:
    """Test getting Redis client."""

    @pytest.mark.asyncio
    async def test_get_redis_with_client(self):
        """Test getting Redis when client exists."""
        import subtext.db.redis as redis_module

        mock_client = AsyncMock()

        # Store original
        original_client = redis_module._redis_client

        try:
            redis_module._redis_client = mock_client
            result = await redis_module.get_redis()

            assert result is mock_client
        finally:
            redis_module._redis_client = original_client

    @pytest.mark.asyncio
    async def test_get_redis_without_client(self):
        """Test getting Redis when client not initialized."""
        import subtext.db.redis as redis_module

        # Store original
        original_client = redis_module._redis_client

        try:
            redis_module._redis_client = None

            with pytest.raises(RuntimeError, match="not initialized"):
                await redis_module.get_redis()
        finally:
            redis_module._redis_client = original_client


# ============================================================
# Module Exports Tests
# ============================================================


class TestModuleExports:
    """Test module exports."""

    def test_all_exports(self):
        """Test __all__ exports."""
        import subtext.db.redis as redis_module

        assert "init_redis" in redis_module.__all__
        assert "close_redis" in redis_module.__all__
        assert "get_redis" in redis_module.__all__

    def test_init_redis_exported(self):
        """Test init_redis is exported."""
        from subtext.db.redis import init_redis

        assert callable(init_redis)

    def test_close_redis_exported(self):
        """Test close_redis is exported."""
        from subtext.db.redis import close_redis

        assert callable(close_redis)

    def test_get_redis_exported(self):
        """Test get_redis is exported."""
        from subtext.db.redis import get_redis

        assert callable(get_redis)
