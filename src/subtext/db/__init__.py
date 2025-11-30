"""
Database Module

SQLAlchemy async database configuration with PostgreSQL + TimescaleDB.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
import structlog

from subtext.config import settings

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# Base Model
# ══════════════════════════════════════════════════════════════


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# ══════════════════════════════════════════════════════════════
# Engine & Session Factory
# ══════════════════════════════════════════════════════════════

_engine = None
_session_factory = None


async def init_db() -> None:
    """Initialize database connection pool."""
    global _engine, _session_factory

    _engine = create_async_engine(
        settings.async_database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        echo=settings.database_echo,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    logger.info("Database connection pool initialized")


async def close_db() -> None:
    """Close database connection pool."""
    global _engine

    if _engine:
        await _engine.dispose()
        logger.info("Database connection pool closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session context manager."""
    if not _session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with get_session() as session:
        yield session


# Import models to ensure they're registered with Base
from subtext.db.models import (
    OrganizationModel,
    UserModel,
    SessionModel,
    SpeakerModel,
    TranscriptSegmentModel,
    SignalModel,
    ProsodicsModel,
    InsightModel,
    TimelineModel,
    VoiceFingerprintModel,
    APIKeyModel,
    UsageRecordModel,
)

__all__ = [
    "Base",
    "init_db",
    "close_db",
    "get_session",
    "get_db",
    # Models
    "OrganizationModel",
    "UserModel",
    "SessionModel",
    "SpeakerModel",
    "TranscriptSegmentModel",
    "SignalModel",
    "ProsodicsModel",
    "InsightModel",
    "TimelineModel",
    "VoiceFingerprintModel",
    "APIKeyModel",
    "UsageRecordModel",
]
