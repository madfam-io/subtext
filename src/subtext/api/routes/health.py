"""Health check endpoints."""

from fastapi import APIRouter, Response

from subtext import __version__

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic health check."""
    return {
        "status": "healthy",
        "version": __version__,
    }


@router.get("/health/ready")
async def readiness_check() -> dict:
    """Kubernetes readiness probe."""
    # Check database connection
    from subtext.db import get_session

    try:
        async with get_session() as session:
            await session.execute("SELECT 1")
    except Exception:
        return Response(status_code=503, content="Database not ready")

    # Check Redis connection
    from subtext.db.redis import get_redis

    try:
        redis = await get_redis()
        await redis.ping()
    except Exception:
        return Response(status_code=503, content="Redis not ready")

    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check() -> dict:
    """Kubernetes liveness probe."""
    return {"status": "alive"}
