"""
Redis Module

Redis connection for caching, job queues, and real-time features.
"""

import redis.asyncio as redis
import structlog

from subtext.config import settings

logger = structlog.get_logger()

_redis_client: redis.Redis | None = None


async def init_redis() -> None:
    """Initialize Redis connection."""
    global _redis_client

    _redis_client = redis.from_url(
        str(settings.redis_url),
        encoding="utf-8",
        decode_responses=True,
    )

    # Test connection
    await _redis_client.ping()
    logger.info("Redis connection initialized")


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client

    if _redis_client:
        await _redis_client.close()
        logger.info("Redis connection closed")


async def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    if not _redis_client:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis_client


__all__ = ["init_redis", "close_redis", "get_redis"]
