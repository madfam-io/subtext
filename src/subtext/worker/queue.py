"""
ARQ Queue Configuration

Sets up the async Redis queue for background job processing.
"""

from datetime import timedelta
from enum import Enum
from typing import Any, Callable

import structlog
from arq import create_pool
from arq.connections import RedisSettings, ArqRedis

from subtext.config import settings

logger = structlog.get_logger()


class JobPriority(str, Enum):
    """Job priority levels."""

    HIGH = "high"      # Real-time processing, urgent notifications
    NORMAL = "normal"  # Standard audio processing
    LOW = "low"        # Background tasks, cleanup, reports


# Queue names by priority
QUEUE_NAMES = {
    JobPriority.HIGH: "subtext:high",
    JobPriority.NORMAL: "subtext:default",
    JobPriority.LOW: "subtext:low",
}

# Global connection pool
_pool: ArqRedis | None = None


async def get_redis_pool() -> ArqRedis:
    """Get or create ARQ Redis connection pool."""
    global _pool

    if _pool is None:
        redis_settings = RedisSettings(
            host=settings.redis_url.host or "localhost",
            port=settings.redis_url.port or 6379,
            password=settings.redis_url.password,
            database=int(settings.redis_url.path.lstrip("/") or "0") if settings.redis_url.path else 0,
        )
        _pool = await create_pool(redis_settings)
        logger.info("ARQ Redis pool created")

    return _pool


async def close_redis_pool() -> None:
    """Close the ARQ Redis pool."""
    global _pool

    if _pool:
        await _pool.close()
        _pool = None
        logger.info("ARQ Redis pool closed")


async def enqueue_job(
    func: str | Callable,
    *args: Any,
    priority: JobPriority = JobPriority.NORMAL,
    defer_by: timedelta | None = None,
    job_id: str | None = None,
    **kwargs: Any,
) -> str | None:
    """
    Enqueue a job for background processing.

    Args:
        func: Function name or callable to execute
        *args: Positional arguments for the function
        priority: Job priority level
        defer_by: Delay before job execution
        job_id: Optional custom job ID
        **kwargs: Keyword arguments for the function

    Returns:
        Job ID if enqueued successfully, None otherwise
    """
    pool = await get_redis_pool()

    func_name = func if isinstance(func, str) else func.__name__

    try:
        job = await pool.enqueue_job(
            func_name,
            *args,
            _queue_name=QUEUE_NAMES[priority],
            _defer_by=defer_by,
            _job_id=job_id,
            **kwargs,
        )

        if job:
            logger.info(
                "Job enqueued",
                job_id=job.job_id,
                func=func_name,
                priority=priority.value,
                defer_by=str(defer_by) if defer_by else None,
            )
            return job.job_id

        return None

    except Exception as e:
        logger.error(
            "Failed to enqueue job",
            func=func_name,
            error=str(e),
        )
        return None


async def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Get the status of a job."""
    pool = await get_redis_pool()

    try:
        job = await pool.job(job_id)
        if job:
            info = await job.info()
            return {
                "job_id": job_id,
                "status": info.status if info else "unknown",
                "result": info.result if info else None,
                "start_time": info.start_time if info else None,
                "finish_time": info.finish_time if info else None,
            }
        return None

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return None


async def cancel_job(job_id: str) -> bool:
    """Cancel a pending job."""
    pool = await get_redis_pool()

    try:
        job = await pool.job(job_id)
        if job:
            await job.abort()
            logger.info(f"Job cancelled: {job_id}")
            return True
        return False

    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        return False


def get_worker_settings() -> dict[str, Any]:
    """
    Get ARQ worker settings.

    This is used by the CLI to start workers.
    """
    from .tasks import (
        process_audio_file,
        process_realtime_session,
        generate_session_insights,
        export_session_data,
        cleanup_expired_sessions,
        startup,
        shutdown,
    )

    redis_settings = RedisSettings(
        host=settings.redis_url.host or "localhost",
        port=settings.redis_url.port or 6379,
        password=settings.redis_url.password,
        database=int(settings.redis_url.path.lstrip("/") or "0") if settings.redis_url.path else 0,
    )

    return {
        "functions": [
            process_audio_file,
            process_realtime_session,
            generate_session_insights,
            export_session_data,
            cleanup_expired_sessions,
        ],
        "on_startup": startup,
        "on_shutdown": shutdown,
        "redis_settings": redis_settings,
        "queue_name": QUEUE_NAMES[JobPriority.NORMAL],
        "max_jobs": settings.worker_max_jobs,
        "job_timeout": settings.worker_job_timeout,
        "keep_result": 3600,  # Keep results for 1 hour
        "health_check_interval": 30,
        "retry_jobs": True,
        "max_tries": 3,
    }


# ARQ worker entry point (used by arq CLI)
class WorkerSettings:
    """ARQ Worker Settings class for CLI."""

    @classmethod
    def get_settings(cls) -> dict[str, Any]:
        return get_worker_settings()
