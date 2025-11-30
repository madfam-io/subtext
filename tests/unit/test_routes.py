"""
Unit Tests for API Routes

Tests the FastAPI route handlers.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from subtext.core.models import SignalType


# ══════════════════════════════════════════════════════════════
# Health Routes Tests
# ══════════════════════════════════════════════════════════════


class TestHealthRoutes:
    """Test health check endpoints."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app with health routes."""
        from subtext.api.routes.health import router

        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_liveness_check(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_check_endpoint_exists(self, client):
        """Test readiness probe endpoint exists."""
        # Just test that the endpoint exists and returns something
        # The actual DB/Redis checks will fail without real connections
        response = client.get("/health/ready")

        # Either succeeds with ready or fails with 503
        assert response.status_code in [200, 503]


# ══════════════════════════════════════════════════════════════
# Signals Routes Tests
# ══════════════════════════════════════════════════════════════


class TestSignalsRoutes:
    """Test signals API endpoints."""

    @pytest.fixture
    def mock_user(self):
        """Create mock authenticated user."""
        from subtext.integrations.janua import TokenPayload

        return TokenPayload(
            sub="user-123",
            email="test@example.com",
            org_id="org-123",
            roles=["user"],
            permissions=["read"],
            exp=9999999999,
            iat=1700000000,
            iss="https://auth.example.com",
            aud="subtext-api",
        )

    @pytest.fixture
    def app(self, mock_user):
        """Create test FastAPI app with signals routes."""
        from subtext.api.routes.signals import router
        from subtext.integrations.janua import get_current_user

        app = FastAPI()
        app.include_router(router, prefix="/signals")

        # Override auth dependency
        app.dependency_overrides[get_current_user] = lambda: mock_user

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_list_signal_types(self, client):
        """Test listing all signal types."""
        response = client.get("/signals/types")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Check structure of first item
        first = data[0]
        assert "type" in first
        assert "name" in first
        assert "description" in first
        assert "psychological_interpretation" in first

    def test_get_signal_type(self, client):
        """Test getting a specific signal type."""
        response = client.get("/signals/types/truth_gap")

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "truth_gap"
        assert data["name"] == "Truth Gap"
        assert "thresholds" in data
        assert "required_features" in data

    def test_get_signal_type_invalid(self, client):
        """Test getting an invalid signal type."""
        response = client.get("/signals/types/invalid_type")

        assert response.status_code == 422  # Validation error

    def test_list_signal_categories(self, client):
        """Test listing signal categories."""
        response = client.get("/signals/categories")

        assert response.status_code == 200
        data = response.json()
        assert "temporal" in data
        assert "spectral" in data
        assert "composite" in data
        assert "truth_gap" in data["temporal"]
        assert "micro_tremor" in data["spectral"]
        assert "stress_spike" in data["composite"]


# ══════════════════════════════════════════════════════════════
# Auth Routes Tests
# ══════════════════════════════════════════════════════════════


class TestAuthRoutes:
    """Test authentication endpoints."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app with auth routes."""
        from subtext.api.routes.auth import router

        app = FastAPI()
        app.include_router(router, prefix="/auth")
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_auth_status_no_token(self, client):
        """Test auth status without token."""
        response = client.get("/auth/me")

        # Should fail without valid auth
        assert response.status_code in [401, 403, 422]


# ══════════════════════════════════════════════════════════════
# Sessions Routes Tests
# ══════════════════════════════════════════════════════════════


class TestSessionsRoutes:
    """Test session API endpoints."""

    @pytest.fixture
    def mock_user(self):
        """Create mock authenticated user."""
        from subtext.integrations.janua import TokenPayload

        return TokenPayload(
            sub="user-123",
            email="test@example.com",
            org_id="org-123",
            roles=["user"],
            permissions=["sessions:read", "sessions:write"],
            exp=9999999999,
            iat=1700000000,
            iss="https://auth.example.com",
            aud="subtext-api",
        )

    @pytest.fixture
    def app(self, mock_user):
        """Create test FastAPI app with sessions routes."""
        from subtext.api.routes.sessions import router
        from subtext.integrations.janua import get_current_user

        app = FastAPI()
        app.include_router(router, prefix="/sessions")

        # Override auth dependency
        app.dependency_overrides[get_current_user] = lambda: mock_user

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_sessions_routes_exist(self, app):
        """Test sessions routes are registered."""
        # Check routes are registered
        routes = [route.path for route in app.routes]
        assert any("/sessions" in r for r in routes)


# ══════════════════════════════════════════════════════════════
# Billing Routes Tests
# ══════════════════════════════════════════════════════════════


class TestBillingRoutes:
    """Test billing API endpoints."""

    @pytest.fixture
    def mock_user(self):
        """Create mock authenticated user."""
        from subtext.integrations.janua import TokenPayload

        return TokenPayload(
            sub="user-123",
            email="test@example.com",
            org_id="org-123",
            roles=["admin"],
            permissions=["billing:read", "billing:write"],
            exp=9999999999,
            iat=1700000000,
            iss="https://auth.example.com",
            aud="subtext-api",
        )

    @pytest.fixture
    def app(self, mock_user):
        """Create test FastAPI app with billing routes."""
        from subtext.api.routes.billing import router
        from subtext.integrations.janua import get_current_user

        app = FastAPI()
        app.include_router(router, prefix="/billing")

        # Override auth dependency
        app.dependency_overrides[get_current_user] = lambda: mock_user

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_get_subscription_tiers(self, client):
        """Test getting subscription tiers."""
        response = client.get("/billing/tiers")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


# ══════════════════════════════════════════════════════════════
# Webhooks Routes Tests
# ══════════════════════════════════════════════════════════════


class TestWebhooksRoutes:
    """Test webhook endpoints."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app with webhooks routes."""
        from subtext.api.routes.webhooks import router

        app = FastAPI()
        app.include_router(router, prefix="/webhooks")
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_webhooks_routes_exist(self, app):
        """Test webhooks routes are registered."""
        routes = [route.path for route in app.routes]
        assert any("/webhooks" in r for r in routes)


# ══════════════════════════════════════════════════════════════
# API App Tests
# ══════════════════════════════════════════════════════════════


class TestAPIApp:
    """Test the main API app configuration."""

    def test_app_creation(self):
        """Test API app can be created."""
        from subtext.api.app import create_app

        app = create_app()

        assert app is not None
        assert app.title == "Subtext API"

    def test_app_routes_included(self):
        """Test all routes are included in the app."""
        from subtext.api.app import create_app

        app = create_app()

        # Get all route paths
        routes = [route.path for route in app.routes]

        # Check key routes exist
        assert "/health" in routes or any("/health" in r for r in routes)


# ══════════════════════════════════════════════════════════════
# Route Response Model Tests
# ══════════════════════════════════════════════════════════════


class TestRouteResponseModels:
    """Test response models used by routes."""

    def test_signal_type_values(self):
        """Test all signal types have string values."""
        for signal_type in SignalType:
            assert isinstance(signal_type.value, str)
            assert len(signal_type.value) > 0

    def test_signal_atlas_accessible(self):
        """Test SignalAtlas is accessible for routes."""
        from subtext.pipeline.signals import SignalAtlas

        atlas = SignalAtlas()
        definitions = atlas.get_all_definitions()

        assert len(definitions) > 0
        for sig_type, definition in definitions.items():
            assert definition.name
            assert definition.description
