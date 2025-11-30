"""
Unit Tests for Auth Routes

Tests authentication and organization management endpoints.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from subtext.api.routes.auth import (
    router,
    UserResponse,
    OrganizationResponse,
    CreateOrganizationRequest,
    get_current_user_info,
    list_user_organizations,
    create_organization,
    get_organization,
)
from subtext.integrations.janua import TokenPayload, JanuaOrganization


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return TokenPayload(
        sub="user-123",
        email="test@example.com",
        org_id="org-456",
        roles=["user", "admin"],
        permissions=["read:sessions", "write:sessions"],
        exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        iat=int(datetime.utcnow().timestamp()),
        iss="https://auth.example.com",
        aud="subtext-api",
    )


@pytest.fixture
def mock_org():
    """Create a mock organization."""
    return JanuaOrganization(
        id="org-456",
        name="Test Organization",
        slug="test-org",
        owner_id="user-123",
    )


# ══════════════════════════════════════════════════════════════
# Response Model Tests
# ══════════════════════════════════════════════════════════════


class TestResponseModels:
    """Test response model schemas."""

    def test_user_response(self):
        """Test UserResponse model."""
        response = UserResponse(
            id="user-123",
            email="test@example.com",
            name="Test User",
            org_id="org-456",
            roles=["admin"],
        )
        assert response.id == "user-123"
        assert response.email == "test@example.com"
        assert response.name == "Test User"

    def test_user_response_minimal(self):
        """Test UserResponse with minimal fields."""
        response = UserResponse(
            id="user-123",
            email="test@example.com",
            name=None,
            org_id=None,
            roles=[],
        )
        assert response.name is None
        assert response.org_id is None

    def test_organization_response(self):
        """Test OrganizationResponse model."""
        response = OrganizationResponse(
            id="org-123",
            name="My Org",
            slug="my-org",
        )
        assert response.id == "org-123"
        assert response.name == "My Org"
        assert response.slug == "my-org"

    def test_create_organization_request(self):
        """Test CreateOrganizationRequest model."""
        request = CreateOrganizationRequest(
            name="New Org",
            slug="new-org",
        )
        assert request.name == "New Org"
        assert request.slug == "new-org"

    def test_create_organization_request_no_slug(self):
        """Test CreateOrganizationRequest without slug."""
        request = CreateOrganizationRequest(name="New Org")
        assert request.name == "New Org"
        assert request.slug is None


# ══════════════════════════════════════════════════════════════
# Get Current User Tests
# ══════════════════════════════════════════════════════════════


class TestGetCurrentUser:
    """Test get current user endpoint."""

    @pytest.mark.asyncio
    async def test_get_current_user_success(self, mock_user):
        """Test getting current user info."""
        result = await get_current_user_info(mock_user)

        assert result.id == "user-123"
        assert result.email == "test@example.com"
        assert result.org_id == "org-456"
        assert "admin" in result.roles

    @pytest.mark.asyncio
    async def test_get_current_user_no_org(self):
        """Test getting user without org."""
        user = TokenPayload(
            sub="user-456",
            email="noorg@example.com",
            org_id=None,
            roles=["user"],
            permissions=[],
            exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            iss="https://auth.example.com",
            aud="subtext-api",
        )

        result = await get_current_user_info(user)

        assert result.id == "user-456"
        assert result.org_id is None


# ══════════════════════════════════════════════════════════════
# List Organizations Tests
# ══════════════════════════════════════════════════════════════


class TestListOrganizations:
    """Test list organizations endpoint."""

    @pytest.mark.asyncio
    async def test_list_organizations_success(self, mock_user):
        """Test listing user organizations."""
        mock_orgs = [
            JanuaOrganization(id="org-1", name="Org 1", slug="org-1", owner_id="user-123"),
            JanuaOrganization(id="org-2", name="Org 2", slug="org-2", owner_id="user-456"),
        ]

        mock_client = MagicMock()
        mock_client.get_user_organizations = AsyncMock(return_value=mock_orgs)

        with patch("subtext.api.routes.auth.get_janua_client", return_value=mock_client):
            result = await list_user_organizations(mock_user)

        assert len(result) == 2
        assert result[0].name == "Org 1"
        assert result[1].name == "Org 2"

    @pytest.mark.asyncio
    async def test_list_organizations_empty(self, mock_user):
        """Test listing when user has no organizations."""
        mock_client = MagicMock()
        mock_client.get_user_organizations = AsyncMock(return_value=[])

        with patch("subtext.api.routes.auth.get_janua_client", return_value=mock_client):
            result = await list_user_organizations(mock_user)

        assert len(result) == 0


# ══════════════════════════════════════════════════════════════
# Create Organization Tests
# ══════════════════════════════════════════════════════════════


class TestCreateOrganization:
    """Test create organization endpoint."""

    @pytest.mark.asyncio
    async def test_create_organization_success(self, mock_user, mock_org):
        """Test successful organization creation."""
        request = CreateOrganizationRequest(
            name="Test Organization",
            slug="test-org",
        )

        mock_janua = MagicMock()
        mock_janua.create_organization = AsyncMock(return_value=mock_org)

        mock_billing = MagicMock()
        mock_billing.setup_organization_billing = AsyncMock()

        mock_email = MagicMock()
        mock_email.send_welcome = AsyncMock()

        with patch("subtext.api.routes.auth.get_janua_client", return_value=mock_janua):
            with patch("subtext.integrations.stripe.get_billing_service", return_value=mock_billing):
                with patch("subtext.integrations.resend.get_email_service", return_value=mock_email):
                    result = await create_organization(request, mock_user)

        assert result.id == "org-456"
        assert result.name == "Test Organization"
        assert result.slug == "test-org"

        mock_janua.create_organization.assert_called_once_with(
            name="Test Organization",
            owner_id="user-123",
            slug="test-org",
        )
        mock_billing.setup_organization_billing.assert_called_once()
        mock_email.send_welcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_organization_no_slug(self, mock_user, mock_org):
        """Test creating organization without slug."""
        request = CreateOrganizationRequest(name="New Org")

        mock_janua = MagicMock()
        mock_janua.create_organization = AsyncMock(return_value=mock_org)

        mock_billing = MagicMock()
        mock_billing.setup_organization_billing = AsyncMock()

        mock_email = MagicMock()
        mock_email.send_welcome = AsyncMock()

        with patch("subtext.api.routes.auth.get_janua_client", return_value=mock_janua):
            with patch("subtext.integrations.stripe.get_billing_service", return_value=mock_billing):
                with patch("subtext.integrations.resend.get_email_service", return_value=mock_email):
                    result = await create_organization(request, mock_user)

        assert result is not None
        mock_janua.create_organization.assert_called_once_with(
            name="New Org",
            owner_id="user-123",
            slug=None,
        )


# ══════════════════════════════════════════════════════════════
# Get Organization Tests
# ══════════════════════════════════════════════════════════════


class TestGetOrganization:
    """Test get organization endpoint."""

    @pytest.mark.asyncio
    async def test_get_organization_success(self, mock_user, mock_org):
        """Test getting organization details."""
        mock_client = MagicMock()
        mock_client.get_organization = AsyncMock(return_value=mock_org)

        with patch("subtext.api.routes.auth.get_janua_client", return_value=mock_client):
            result = await get_organization("org-456", mock_user)

        assert result.id == "org-456"
        assert result.name == "Test Organization"

    @pytest.mark.asyncio
    async def test_get_organization_forbidden(self, mock_user):
        """Test accessing unauthorized organization."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await get_organization("other-org", mock_user)

        assert exc_info.value.status_code == 403
        assert "Not authorized" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_organization_not_found(self, mock_user):
        """Test getting non-existent organization."""
        from fastapi import HTTPException

        mock_client = MagicMock()
        mock_client.get_organization = AsyncMock(return_value=None)

        with patch("subtext.api.routes.auth.get_janua_client", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await get_organization("org-456", mock_user)

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail


# ══════════════════════════════════════════════════════════════
# Router Tests
# ══════════════════════════════════════════════════════════════


class TestRouterConfiguration:
    """Test router configuration."""

    def test_router_exists(self):
        """Test router is configured."""
        assert router is not None

    def test_router_has_routes(self):
        """Test router has expected routes."""
        routes = [r.path for r in router.routes]
        assert "/me" in routes
        assert "/organizations" in routes
        assert "/organizations/{org_id}" in routes
