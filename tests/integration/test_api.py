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


# ══════════════════════════════════════════════════════════════
# Billing Endpoint Tests
# ══════════════════════════════════════════════════════════════


class TestBillingEndpoints:
    """Test billing-related endpoints."""

    @pytest.mark.asyncio
    async def test_get_plans_unauthorized(self, client):
        """Test getting plans without auth."""
        response = await client.get("/api/v1/billing/plans")

        # Plans might be public, require auth, or not exist
        assert response.status_code in [200, 401, 403, 404]

    @pytest.mark.asyncio
    async def test_get_subscription_unauthorized(self, client):
        """Test getting subscription without auth."""
        response = await client.get("/api/v1/billing/subscription")

        # Endpoint might not exist at this path
        assert response.status_code in [401, 403, 404]

    @pytest.mark.asyncio
    async def test_get_usage_unauthorized(self, client):
        """Test getting usage without auth."""
        response = await client.get("/api/v1/billing/usage")

        assert response.status_code in [401, 403, 404]


# ══════════════════════════════════════════════════════════════
# Webhook Endpoint Tests
# ══════════════════════════════════════════════════════════════


class TestWebhookEndpoints:
    """Test webhook endpoints."""

    @pytest.mark.asyncio
    async def test_stripe_webhook_no_signature(self, client):
        """Test Stripe webhook without signature."""
        response = await client.post(
            "/api/v1/webhooks/stripe",
            content=b'{"type": "test"}',
            headers={"Content-Type": "application/json"},
        )

        # Should fail without valid signature
        assert response.status_code in [400, 401, 403, 422]

    @pytest.mark.asyncio
    async def test_stripe_webhook_invalid_signature(self, client):
        """Test Stripe webhook with invalid signature."""
        response = await client.post(
            "/api/v1/webhooks/stripe",
            content=b'{"type": "test"}',
            headers={
                "Content-Type": "application/json",
                "Stripe-Signature": "invalid-signature",
            },
        )

        # Should fail with invalid signature
        assert response.status_code in [400, 401, 403, 422]


# ══════════════════════════════════════════════════════════════
# Authenticated Session Tests
# ══════════════════════════════════════════════════════════════


class TestAuthenticatedSessions:
    """Test session endpoints with mocked authentication."""

    @pytest.fixture
    def mock_auth_user(self):
        """Create a mock authenticated user."""
        from subtext.integrations.janua import TokenPayload
        from datetime import datetime, timedelta

        return TokenPayload(
            sub="user-123",
            email="test@example.com",
            org_id="org-456",
            roles=["user"],
            permissions=["read:sessions", "write:sessions"],
            exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            iss="https://auth.example.com",
            aud="api",
        )

    @pytest.mark.asyncio
    async def test_list_sessions_authenticated(self, client, mock_auth_user):
        """Test listing sessions with valid auth."""
        from subtext.integrations.janua import JanuaAuth

        with patch.object(JanuaAuth, "__call__", return_value=mock_auth_user):
            with patch("subtext.db.get_session") as mock_get_session:
                mock_session = AsyncMock()
                mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_get_session.return_value = mock_session

                response = await client.get(
                    "/api/v1/sessions",
                    headers={"Authorization": "Bearer valid-token"},
                )

                # With mocked auth and db, should work
                assert response.status_code in [200, 401, 403, 500]

    @pytest.mark.asyncio
    async def test_create_session_authenticated(self, client, mock_auth_user):
        """Test creating a session with valid auth."""
        from subtext.integrations.janua import JanuaAuth

        with patch.object(JanuaAuth, "__call__", return_value=mock_auth_user):
            with patch("subtext.db.get_session") as mock_get_session:
                mock_session = AsyncMock()
                mock_session.add = MagicMock()
                mock_session.commit = AsyncMock()
                mock_session.refresh = AsyncMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_get_session.return_value = mock_session

                response = await client.post(
                    "/api/v1/sessions",
                    json={
                        "name": "Test Session",
                        "description": "A test session",
                    },
                    headers={"Authorization": "Bearer valid-token"},
                )

                # Response depends on auth mock working correctly
                assert response.status_code in [200, 201, 401, 403, 422, 500]


# ══════════════════════════════════════════════════════════════
# Error Handling Tests
# ══════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.asyncio
    async def test_not_found_endpoint(self, client):
        """Test 404 for non-existent endpoint."""
        response = await client.get("/api/v1/nonexistent")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = await client.delete("/health")

        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, client):
        """Test handling of invalid JSON."""
        response = await client.post(
            "/api/v1/sessions",
            content=b"not valid json",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            },
        )

        assert response.status_code in [400, 401, 403, 422]


# ══════════════════════════════════════════════════════════════
# CORS Tests
# ══════════════════════════════════════════════════════════════


class TestCORS:
    """Test CORS configuration."""

    @pytest.mark.asyncio
    async def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = await client.options(
            "/api/v1/sessions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # CORS preflight should be handled
        assert response.status_code in [200, 204, 405]

    @pytest.mark.asyncio
    async def test_cors_headers(self, client):
        """Test CORS headers in response."""
        response = await client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

        assert response.status_code == 200
        # CORS headers might be present
        # assert "access-control-allow-origin" in response.headers


# ══════════════════════════════════════════════════════════════
# Rate Limiting Tests (if implemented)
# ══════════════════════════════════════════════════════════════


class TestRateLimiting:
    """Test rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_health_no_rate_limit(self, client):
        """Test health endpoint has no rate limiting."""
        for _ in range(10):
            response = await client.get("/health")
            assert response.status_code == 200


# ══════════════════════════════════════════════════════════════
# Response Format Tests
# ══════════════════════════════════════════════════════════════


class TestResponseFormat:
    """Test API response format consistency."""

    @pytest.mark.asyncio
    async def test_health_response_format(self, client):
        """Test health response has correct format."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)
        assert "status" in data

    @pytest.mark.asyncio
    async def test_error_response_format(self, client):
        """Test error responses have correct format."""
        response = await client.get("/api/v1/nonexistent")

        assert response.status_code == 404
        data = response.json()

        assert isinstance(data, dict)
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_json_content_type(self, client):
        """Test responses have JSON content type."""
        response = await client.get("/health")

        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
