"""
Unit Tests for API Application Module

Tests the FastAPI application factory and configuration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


# ============================================================
# Application Factory Tests
# ============================================================


class TestCreateApp:
    """Test create_app factory function."""

    def test_create_app_returns_fastapi(self):
        """Test create_app returns a FastAPI instance."""
        with patch("subtext.api.app.settings") as mock_settings:
            mock_settings.debug = True
            mock_settings.cors_origins = ["*"]
            mock_settings.api_version = "v1"
            mock_settings.app_env = "test"

            from subtext.api.app import create_app

            app = create_app()

            assert isinstance(app, FastAPI)

    def test_create_app_title(self):
        """Test app has correct title."""
        with patch("subtext.api.app.settings") as mock_settings:
            mock_settings.debug = True
            mock_settings.cors_origins = ["*"]
            mock_settings.api_version = "v1"
            mock_settings.app_env = "test"

            from subtext.api.app import create_app

            app = create_app()

            assert app.title == "Subtext API"

    def test_create_app_debug_docs_enabled(self):
        """Test docs URLs are enabled in debug mode."""
        with patch("subtext.api.app.settings") as mock_settings:
            mock_settings.debug = True
            mock_settings.cors_origins = ["*"]
            mock_settings.api_version = "v1"
            mock_settings.app_env = "test"

            from subtext.api.app import create_app

            app = create_app()

            assert app.docs_url == "/docs"
            assert app.redoc_url == "/redoc"
            assert app.openapi_url == "/openapi.json"

    def test_create_app_prod_docs_disabled(self):
        """Test docs URLs are disabled in production."""
        with patch("subtext.api.app.settings") as mock_settings:
            mock_settings.debug = False
            mock_settings.cors_origins = ["*"]
            mock_settings.api_version = "v1"
            mock_settings.app_env = "production"

            from subtext.api.app import create_app

            app = create_app()

            assert app.docs_url is None
            assert app.redoc_url is None
            assert app.openapi_url is None


class TestAppRoutes:
    """Test app routes are registered."""

    def test_health_routes_registered(self):
        """Test health routes are registered."""
        with patch("subtext.api.app.settings") as mock_settings:
            mock_settings.debug = True
            mock_settings.cors_origins = ["*"]
            mock_settings.api_version = "v1"
            mock_settings.app_env = "test"

            from subtext.api.app import create_app

            app = create_app()

            routes = [route.path for route in app.routes]

            assert "/health" in routes or any("/health" in r for r in routes)

    def test_api_routes_registered(self):
        """Test API v1 routes are registered."""
        with patch("subtext.api.app.settings") as mock_settings:
            mock_settings.debug = True
            mock_settings.cors_origins = ["*"]
            mock_settings.api_version = "v1"
            mock_settings.app_env = "test"

            from subtext.api.app import create_app

            app = create_app()

            routes = [route.path for route in app.routes]

            # Check that API routes exist
            api_routes = [r for r in routes if "/api/v1" in r]
            assert len(api_routes) > 0

    def test_websocket_routes_registered(self):
        """Test WebSocket routes are registered."""
        with patch("subtext.api.app.settings") as mock_settings:
            mock_settings.debug = True
            mock_settings.cors_origins = ["*"]
            mock_settings.api_version = "v1"
            mock_settings.app_env = "test"

            from subtext.api.app import create_app

            app = create_app()

            routes = [route.path for route in app.routes]

            ws_routes = [r for r in routes if "/ws" in r]
            assert len(ws_routes) > 0


class TestAppMiddleware:
    """Test app middleware configuration."""

    def test_cors_middleware_added(self):
        """Test CORS middleware is configured."""
        with patch("subtext.api.app.settings") as mock_settings:
            mock_settings.debug = True
            mock_settings.cors_origins = ["http://localhost:3000"]
            mock_settings.api_version = "v1"
            mock_settings.app_env = "test"

            from subtext.api.app import create_app

            app = create_app()

            # Check middleware stack contains CORS
            middleware_names = [
                m.cls.__name__ if hasattr(m, "cls") else str(m)
                for m in app.user_middleware
            ]

            assert "CORSMiddleware" in middleware_names


# ============================================================
# Default App Instance Tests
# ============================================================


class TestDefaultAppInstance:
    """Test the default app instance."""

    def test_app_exists(self):
        """Test default app instance exists."""
        from subtext.api.app import app

        assert app is not None
        assert isinstance(app, FastAPI)

    def test_app_has_lifespan(self):
        """Test app has lifespan configured."""
        from subtext.api.app import app

        assert app.router.lifespan_context is not None


# ============================================================
# Lifespan Tests
# ============================================================


class TestLifespan:
    """Test application lifespan management."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_and_shutdown(self):
        """Test lifespan startup and shutdown."""
        from subtext.api.app import lifespan

        mock_app = MagicMock(spec=FastAPI)

        with patch("subtext.db.init_db", new_callable=AsyncMock) as mock_init_db:
            with patch("subtext.db.close_db", new_callable=AsyncMock) as mock_close_db:
                with patch("subtext.db.redis.init_redis", new_callable=AsyncMock) as mock_init_redis:
                    with patch("subtext.db.redis.close_redis", new_callable=AsyncMock) as mock_close_redis:
                        with patch("subtext.realtime.broadcaster.broadcaster") as mock_broadcaster:
                            mock_broadcaster.start = AsyncMock()
                            mock_broadcaster.stop = AsyncMock()

                            async with lifespan(mock_app):
                                # During lifespan, services should be initialized
                                pass

                            # After lifespan, cleanup should happen
                            mock_broadcaster.stop.assert_called_once()


# ============================================================
# Version Tests
# ============================================================


class TestVersion:
    """Test version information."""

    def test_app_version(self):
        """Test app has version set."""
        from subtext.api.app import app
        from subtext import __version__

        assert app.version == __version__


# ============================================================
# Module Exports Tests
# ============================================================


class TestModuleExports:
    """Test module exports."""

    def test_app_exported(self):
        """Test app is exported."""
        from subtext.api.app import app

        assert app is not None

    def test_create_app_exported(self):
        """Test create_app is exported."""
        from subtext.api.app import create_app

        assert callable(create_app)

    def test_lifespan_exported(self):
        """Test lifespan is exported."""
        from subtext.api.app import lifespan

        assert lifespan is not None
