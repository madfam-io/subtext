"""
Unit Tests for Database Module

Tests database connection and session management.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from subtext.db import Base


# ============================================================
# Base Model Tests
# ============================================================


class TestBaseModel:
    """Test Base model class."""

    def test_base_is_declarative(self):
        """Test Base is a declarative base."""
        from sqlalchemy.orm import DeclarativeBase

        assert issubclass(Base, DeclarativeBase)


# ============================================================
# init_db Tests
# ============================================================


class TestInitDb:
    """Test database initialization."""

    @pytest.mark.asyncio
    async def test_init_db_creates_engine(self):
        """Test init_db creates engine."""
        import subtext.db as db_module

        # Store originals
        original_engine = db_module._engine
        original_factory = db_module._session_factory

        try:
            db_module._engine = None
            db_module._session_factory = None

            mock_engine = MagicMock()
            mock_sessionmaker = MagicMock()

            with patch("subtext.db.create_async_engine", return_value=mock_engine) as mock_create:
                with patch("subtext.db.async_sessionmaker", return_value=mock_sessionmaker):
                    await db_module.init_db()

            mock_create.assert_called_once()
            assert db_module._engine is mock_engine
            assert db_module._session_factory is mock_sessionmaker
        finally:
            db_module._engine = original_engine
            db_module._session_factory = original_factory


# ============================================================
# close_db Tests
# ============================================================


class TestCloseDb:
    """Test database closing."""

    @pytest.mark.asyncio
    async def test_close_db_disposes_engine(self):
        """Test close_db disposes engine."""
        import subtext.db as db_module

        # Store original
        original_engine = db_module._engine

        try:
            mock_engine = AsyncMock()
            mock_engine.dispose = AsyncMock()
            db_module._engine = mock_engine

            await db_module.close_db()

            mock_engine.dispose.assert_called_once()
        finally:
            db_module._engine = original_engine

    @pytest.mark.asyncio
    async def test_close_db_no_engine(self):
        """Test close_db with no engine does nothing."""
        import subtext.db as db_module

        # Store original
        original_engine = db_module._engine

        try:
            db_module._engine = None

            # Should not raise
            await db_module.close_db()
        finally:
            db_module._engine = original_engine


# ============================================================
# get_session Tests
# ============================================================


class TestGetSession:
    """Test session management."""

    @pytest.mark.asyncio
    async def test_get_session_raises_if_not_initialized(self):
        """Test get_session raises if db not initialized."""
        import subtext.db as db_module

        # Store original
        original_factory = db_module._session_factory

        try:
            db_module._session_factory = None

            with pytest.raises(RuntimeError, match="not initialized"):
                async with db_module.get_session():
                    pass
        finally:
            db_module._session_factory = original_factory

    @pytest.mark.asyncio
    async def test_get_session_commits_on_success(self):
        """Test get_session commits on success."""
        import subtext.db as db_module

        # Store original
        original_factory = db_module._session_factory

        try:
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.rollback = AsyncMock()

            # Mock session factory as async context manager
            mock_factory = MagicMock()
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value = mock_cm

            db_module._session_factory = mock_factory

            async with db_module.get_session() as session:
                assert session is mock_session

            mock_session.commit.assert_called_once()
            mock_session.rollback.assert_not_called()
        finally:
            db_module._session_factory = original_factory

    @pytest.mark.asyncio
    async def test_get_session_rollbacks_on_exception(self):
        """Test get_session rolls back on exception."""
        import subtext.db as db_module

        # Store original
        original_factory = db_module._session_factory

        try:
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock(side_effect=Exception("Commit failed"))
            mock_session.rollback = AsyncMock()

            mock_factory = MagicMock()
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value = mock_cm

            db_module._session_factory = mock_factory

            with pytest.raises(Exception, match="Commit failed"):
                async with db_module.get_session():
                    pass

            mock_session.rollback.assert_called_once()
        finally:
            db_module._session_factory = original_factory


# ============================================================
# get_db Tests
# ============================================================


class TestGetDb:
    """Test FastAPI dependency."""

    @pytest.mark.asyncio
    async def test_get_db_yields_session(self):
        """Test get_db yields a session."""
        import subtext.db as db_module

        # Store original
        original_factory = db_module._session_factory

        try:
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()

            mock_factory = MagicMock()
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value = mock_cm

            db_module._session_factory = mock_factory

            async for session in db_module.get_db():
                assert session is mock_session
        finally:
            db_module._session_factory = original_factory


# ============================================================
# Module Exports Tests
# ============================================================


class TestModuleExports:
    """Test module exports."""

    def test_all_exports(self):
        """Test __all__ exports."""
        import subtext.db as db_module

        expected_exports = [
            "Base",
            "init_db",
            "close_db",
            "get_session",
            "get_db",
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

        for export in expected_exports:
            assert export in db_module.__all__

    def test_init_db_exported(self):
        """Test init_db is exported."""
        from subtext.db import init_db

        assert callable(init_db)

    def test_close_db_exported(self):
        """Test close_db is exported."""
        from subtext.db import close_db

        assert callable(close_db)

    def test_get_session_exported(self):
        """Test get_session is exported."""
        from subtext.db import get_session

        # Should be a context manager/function
        assert callable(get_session)

    def test_models_exported(self):
        """Test models are exported."""
        from subtext.db import SessionModel, SignalModel, InsightModel

        assert SessionModel is not None
        assert SignalModel is not None
        assert InsightModel is not None
