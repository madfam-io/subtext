"""
Integration Tests for API Endpoints

Tests the FastAPI application with real HTTP requests.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from uuid import uuid4

from subtext.api.app import create_app


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def app():
    """Create test application instance."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
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
        response = await client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data

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
        response = await client.get(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer invalid-token"},
        )

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
        """Test OpenAPI schema is available."""
        response = await client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Subtext API"

    @pytest.mark.asyncio
    async def test_docs_available(self, client):
        """Test API docs are available."""
        response = await client.get("/docs")

        # Should return HTML
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
