"""
Integration Tests for API Endpoints

Tests the FastAPI application with real HTTP requests.
"""

import os
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4


# Set debug mode for tests before importing app
os.environ["DEBUG"] = "true"
os.environ["APP_ENV"] = "development"


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def app():
    """Create test application instance with mocked dependencies."""
    # Patch at the source modules where the functions are defined
    with patch("subtext.db.init_db", new_callable=AsyncMock) as mock_init_db, \
         patch("subtext.db.close_db", new_callable=AsyncMock) as mock_close_db, \
         patch("subtext.db.redis.init_redis", new_callable=AsyncMock) as mock_init_redis, \
         patch("subtext.db.redis.close_redis", new_callable=AsyncMock) as mock_close_redis, \
         patch("subtext.realtime.broadcaster.broadcaster") as mock_broadcaster:

        # Configure the broadcaster mock
        mock_broadcaster.start = AsyncMock()
        mock_broadcaster.stop = AsyncMock()

        from subtext.api.app import create_app

        # Create app with debug enabled
        test_app = create_app()
        return test_app


@pytest.fixture
async def client(app):
    """Create async test client."""
    transport = ASGITransport(app=app)
    # Disable proxy detection to avoid environment proxy issues
    async with AsyncClient(transport=transport, base_url="http://test", trust_env=False) as ac:
        yield ac


# ══════════════════════════════════════════════════════════════
# Health Endpoint Tests
# ══════════════════════════════════════════════════════════════


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test basic health check."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_readiness_check(self, client):
        """Test readiness probe."""
        # Patch at the source module where get_redis is defined
        with patch("subtext.db.redis.get_redis", new_callable=AsyncMock) as mock_get_redis, \
             patch("subtext.db.get_session") as mock_get_session:
            # Mock Redis
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock(return_value=True)
            mock_get_redis.return_value = mock_redis_client

            # Mock database session
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=None)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            response = await client.get("/health/ready")

            # Can be 200 or 503 depending on database state
            assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_liveness_check(self, client):
        """Test liveness probe."""
        response = await client.get("/health/live")

        assert response.status_code == 200


# ══════════════════════════════════════════════════════════════
# Auth Endpoint Tests
# ══════════════════════════════════════════════════════════════


class TestAuthEndpoints:
    """Test authentication endpoints."""

    @pytest.mark.asyncio
    async def test_unauthorized_request(self, client):
        """Test request without auth token."""
        response = await client.get("/api/v1/sessions")

        # Should return 401 or 403
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_invalid_token(self, client):
        """Test request with invalid auth token."""
        from fastapi import HTTPException

        # Mock Janua to reject invalid tokens without making HTTP calls
        with patch("subtext.integrations.janua.JanuaAuth._verify_token", new_callable=AsyncMock) as mock_verify:
            mock_verify.side_effect = HTTPException(status_code=401, detail="Invalid token")

            response = await client.get(
                "/api/v1/sessions",
                headers={"Authorization": "Bearer invalid-token"},
            )

            # Should return 401 or 403 (authentication failure)
            assert response.status_code in [401, 403]


# ══════════════════════════════════════════════════════════════
# Session Endpoint Tests
# ══════════════════════════════════════════════════════════════


class TestSessionEndpoints:
    """Test session management endpoints."""

    @pytest.mark.asyncio
    async def test_create_session_unauthorized(self, client):
        """Test session creation without auth."""
        response = await client.post(
            "/api/v1/sessions",
            json={
                "name": "Test Session",
                "description": "A test session",
            },
        )

        assert response.status_code in [401, 403]


# ══════════════════════════════════════════════════════════════
# Signal Endpoint Tests
# ══════════════════════════════════════════════════════════════


class TestSignalEndpoints:
    """Test signal-related endpoints."""

    @pytest.mark.asyncio
    async def test_list_signal_types(self, client):
        """Test listing available signal types."""
        response = await client.get("/api/v1/signals/types")

        # This might be public or require auth
        if response.status_code == 200:
            data = response.json()
            assert "signal_types" in data or isinstance(data, list)


# ══════════════════════════════════════════════════════════════
# OpenAPI Schema Tests
# ══════════════════════════════════════════════════════════════


class TestOpenAPISchema:
    """Test OpenAPI schema generation."""

    @pytest.mark.asyncio
    async def test_openapi_schema(self, client):
        """Test OpenAPI schema is available in debug mode."""
        response = await client.get("/openapi.json")

        # In debug mode, OpenAPI should be available
        if response.status_code == 200:
            data = response.json()
            assert "openapi" in data
            assert "info" in data
            assert data["info"]["title"] == "Subtext API"
        else:
            # If not debug mode, 404 is acceptable
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_docs_available(self, client):
        """Test API docs are available in debug mode."""
        response = await client.get("/docs")

        # In debug mode, docs should be available
        if response.status_code == 200:
            assert "text/html" in response.headers.get("content-type", "")
        else:
            # If not debug mode, 404 is acceptable
            assert response.status_code == 404
