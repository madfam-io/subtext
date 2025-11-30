"""
Unit Tests for Janua Authentication Integration

Tests the Janua auth platform integration.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import HTTPException

from subtext.integrations.janua import (
    JanuaUser,
    JanuaOrganization,
    JanuaToken,
    TokenPayload,
    JanuaClient,
    JanuaAuth,
    auth_required,
    admin_required,
    get_current_user,
    get_optional_user,
    get_janua_client,
)


# ============================================================
# Model Tests
# ============================================================


class TestJanuaUser:
    """Test JanuaUser model."""

    def test_user_minimal(self):
        """Test user with minimal fields."""
        user = JanuaUser(id="user-123", email="test@example.com")

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.name is None
        assert user.email_verified is False
        assert user.metadata == {}

    def test_user_full(self):
        """Test user with all fields."""
        created = datetime(2024, 1, 1)
        user = JanuaUser(
            id="user-123",
            email="test@example.com",
            name="Test User",
            email_verified=True,
            avatar_url="https://example.com/avatar.png",
            metadata={"plan": "pro"},
            created_at=created,
        )

        assert user.name == "Test User"
        assert user.email_verified is True
        assert user.avatar_url == "https://example.com/avatar.png"
        assert user.metadata == {"plan": "pro"}
        assert user.created_at == created


class TestJanuaOrganization:
    """Test JanuaOrganization model."""

    def test_organization_creation(self):
        """Test organization model."""
        org = JanuaOrganization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            owner_id="user-123",
        )

        assert org.id == "org-123"
        assert org.name == "Test Org"
        assert org.slug == "test-org"
        assert org.owner_id == "user-123"
        assert org.metadata == {}

    def test_organization_with_metadata(self):
        """Test organization with metadata."""
        org = JanuaOrganization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            owner_id="user-123",
            metadata={"tier": "enterprise"},
        )

        assert org.metadata == {"tier": "enterprise"}


class TestJanuaToken:
    """Test JanuaToken model."""

    def test_token_minimal(self):
        """Test token with minimal fields."""
        token = JanuaToken(
            access_token="access123",
            expires_in=3600,
        )

        assert token.access_token == "access123"
        assert token.token_type == "Bearer"
        assert token.expires_in == 3600
        assert token.refresh_token is None
        assert token.id_token is None

    def test_token_full(self):
        """Test token with all fields."""
        token = JanuaToken(
            access_token="access123",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="refresh456",
            id_token="id789",
        )

        assert token.refresh_token == "refresh456"
        assert token.id_token == "id789"


class TestTokenPayload:
    """Test TokenPayload model."""

    def test_payload_creation(self):
        """Test token payload."""
        now = int(datetime.utcnow().timestamp())
        payload = TokenPayload(
            sub="user-123",
            email="test@example.com",
            org_id="org-456",
            roles=["user", "admin"],
            permissions=["read", "write"],
            exp=now + 3600,
            iat=now,
            iss="https://auth.example.com",
            aud="api.example.com",
        )

        assert payload.sub == "user-123"
        assert payload.email == "test@example.com"
        assert payload.org_id == "org-456"
        assert "admin" in payload.roles
        assert "write" in payload.permissions


# ============================================================
# JanuaClient Tests
# ============================================================


class TestJanuaClientInit:
    """Test JanuaClient initialization."""

    def test_client_init_defaults(self):
        """Test client uses settings defaults."""
        with patch("subtext.integrations.janua.settings") as mock_settings:
            mock_settings.janua_base_url = "https://auth.example.com"
            mock_settings.janua_client_id = "client-123"
            mock_settings.janua_client_secret = "secret-456"

            client = JanuaClient()

            assert client.base_url == "https://auth.example.com"
            assert client.client_id == "client-123"
            assert client.client_secret == "secret-456"

    def test_client_init_custom(self):
        """Test client with custom values."""
        client = JanuaClient(
            base_url="https://custom.auth.com/",
            client_id="custom-client",
            client_secret="custom-secret",
        )

        assert client.base_url == "https://custom.auth.com"  # Trailing slash removed
        assert client.client_id == "custom-client"


class TestJanuaClientHttpClient:
    """Test JanuaClient HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Test _get_client creates HTTP client."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        http_client = await client._get_client()

        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)

        await client.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        """Test _get_client reuses existing client."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        http_client_1 = await client._get_client()
        http_client_2 = await client._get_client()

        assert http_client_1 is http_client_2

        await client.close()

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing HTTP client."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        # Create client
        await client._get_client()
        assert client._client is not None

        await client.close()
        assert client._client is None


class TestJanuaClientServiceToken:
    """Test service token management."""

    @pytest.mark.asyncio
    async def test_get_service_token(self):
        """Test getting service token."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "service-token-123",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_client", return_value=mock_http):
            token = await client._get_service_token()

        assert token == "service-token-123"
        assert client._service_token == "service-token-123"

    @pytest.mark.asyncio
    async def test_service_token_caching(self):
        """Test service token is cached."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        # Set cached token
        client._service_token = "cached-token"
        client._token_expires_at = datetime.utcnow() + timedelta(hours=1)

        token = await client._get_service_token()

        assert token == "cached-token"


# ============================================================
# JanuaAuth Tests
# ============================================================


class TestJanuaAuthInit:
    """Test JanuaAuth initialization."""

    def test_auth_init_defaults(self):
        """Test auth with defaults."""
        auth = JanuaAuth()

        assert auth.required_roles == []
        assert auth.required_permissions == []

    def test_auth_init_with_roles(self):
        """Test auth with required roles."""
        auth = JanuaAuth(required_roles=["admin", "superuser"])

        assert auth.required_roles == ["admin", "superuser"]

    def test_auth_init_with_permissions(self):
        """Test auth with required permissions."""
        auth = JanuaAuth(required_permissions=["read:users", "write:users"])

        assert auth.required_permissions == ["read:users", "write:users"]


class TestJanuaAuthRoleCheck:
    """Test role checking."""

    def test_check_roles_passes(self):
        """Test role check passes when user has role."""
        auth = JanuaAuth(required_roles=["admin"])

        payload = TokenPayload(
            sub="user-123",
            email="test@example.com",
            roles=["user", "admin"],
            permissions=[],
            exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            iss="https://auth.example.com",
            aud="api",
        )

        # Should not raise
        auth._check_roles(payload)

    def test_check_roles_fails(self):
        """Test role check fails when user lacks role."""
        auth = JanuaAuth(required_roles=["admin"])

        payload = TokenPayload(
            sub="user-123",
            email="test@example.com",
            roles=["user"],
            permissions=[],
            exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            iss="https://auth.example.com",
            aud="api",
        )

        with pytest.raises(HTTPException) as exc_info:
            auth._check_roles(payload)

        assert exc_info.value.status_code == 403
        assert "role" in exc_info.value.detail.lower()

    def test_check_roles_no_requirements(self):
        """Test role check passes when no roles required."""
        auth = JanuaAuth()

        payload = TokenPayload(
            sub="user-123",
            email="test@example.com",
            roles=[],
            permissions=[],
            exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            iss="https://auth.example.com",
            aud="api",
        )

        # Should not raise
        auth._check_roles(payload)


class TestJanuaAuthPermissionCheck:
    """Test permission checking."""

    def test_check_permissions_passes(self):
        """Test permission check passes when user has all permissions."""
        auth = JanuaAuth(required_permissions=["read:users", "write:users"])

        payload = TokenPayload(
            sub="user-123",
            email="test@example.com",
            roles=[],
            permissions=["read:users", "write:users", "delete:users"],
            exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            iss="https://auth.example.com",
            aud="api",
        )

        # Should not raise
        auth._check_permissions(payload)

    def test_check_permissions_fails(self):
        """Test permission check fails when user lacks permission."""
        auth = JanuaAuth(required_permissions=["read:users", "write:users"])

        payload = TokenPayload(
            sub="user-123",
            email="test@example.com",
            roles=[],
            permissions=["read:users"],  # Missing write:users
            exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            iss="https://auth.example.com",
            aud="api",
        )

        with pytest.raises(HTTPException) as exc_info:
            auth._check_permissions(payload)

        assert exc_info.value.status_code == 403
        assert "permission" in exc_info.value.detail.lower()


# ============================================================
# Convenience Dependency Tests
# ============================================================


class TestConvenienceDependencies:
    """Test convenience dependency instances."""

    def test_auth_required_exists(self):
        """Test auth_required is a JanuaAuth instance."""
        assert isinstance(auth_required, JanuaAuth)
        assert auth_required.required_roles == []

    def test_admin_required_exists(self):
        """Test admin_required is a JanuaAuth instance."""
        assert isinstance(admin_required, JanuaAuth)
        assert "admin" in admin_required.required_roles
        assert "owner" in admin_required.required_roles


# ============================================================
# Singleton Client Tests
# ============================================================


class TestSingletonClient:
    """Test singleton Janua client."""

    def test_get_janua_client_returns_client(self):
        """Test get_janua_client returns a client."""
        import subtext.integrations.janua as janua_module

        # Reset singleton
        original = janua_module._janua_client
        janua_module._janua_client = None

        try:
            client = get_janua_client()
            assert isinstance(client, JanuaClient)
        finally:
            janua_module._janua_client = original

    def test_get_janua_client_returns_same_instance(self):
        """Test get_janua_client returns same instance."""
        import subtext.integrations.janua as janua_module

        # Reset singleton
        original = janua_module._janua_client
        janua_module._janua_client = None

        try:
            client1 = get_janua_client()
            client2 = get_janua_client()

            assert client1 is client2
        finally:
            janua_module._janua_client = original


# ============================================================
# get_current_user Tests
# ============================================================


class TestGetCurrentUser:
    """Test get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user_returns_payload(self):
        """Test get_current_user returns the payload."""
        mock_request = MagicMock()

        payload = TokenPayload(
            sub="user-123",
            email="test@example.com",
            roles=[],
            permissions=[],
            exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            iss="https://auth.example.com",
            aud="api",
        )

        result = await get_current_user(mock_request, payload)

        assert result is payload
        assert result.sub == "user-123"


# ============================================================
# get_optional_user Tests
# ============================================================


class TestGetOptionalUser:
    """Test get_optional_user dependency."""

    @pytest.mark.asyncio
    async def test_optional_user_no_credentials(self):
        """Test optional user with no credentials returns None."""
        mock_request = MagicMock()

        result = await get_optional_user(mock_request, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_optional_user_invalid_token(self):
        """Test optional user with invalid token returns None."""
        mock_request = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.credentials = "invalid-token"

        # Mock the _verify_token to raise HTTPException
        with patch.object(JanuaAuth, "_verify_token", side_effect=HTTPException(status_code=401)):
            result = await get_optional_user(mock_request, mock_credentials)

        assert result is None


# ============================================================
# JanuaClient API Methods Tests
# ============================================================


class TestJanuaClientGetUser:
    """Test JanuaClient get_user method."""

    @pytest.mark.asyncio
    async def test_get_user_success(self):
        """Test getting user successfully."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "user-123",
            "email": "test@example.com",
            "name": "Test User",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_authed_request", AsyncMock(return_value=mock_response)):
            user = await client.get_user("user-123")

        assert user is not None
        assert user.id == "user-123"
        assert user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_user_not_found(self):
        """Test getting non-existent user returns None."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(client, "_authed_request", AsyncMock(return_value=mock_response)):
            user = await client.get_user("nonexistent")

        assert user is None


class TestJanuaClientGetOrganization:
    """Test JanuaClient get_organization method."""

    @pytest.mark.asyncio
    async def test_get_organization_success(self):
        """Test getting organization successfully."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "org-123",
            "name": "Test Org",
            "slug": "test-org",
            "owner_id": "user-123",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_authed_request", AsyncMock(return_value=mock_response)):
            org = await client.get_organization("org-123")

        assert org is not None
        assert org.id == "org-123"
        assert org.name == "Test Org"

    @pytest.mark.asyncio
    async def test_get_organization_not_found(self):
        """Test getting non-existent org returns None."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(client, "_authed_request", AsyncMock(return_value=mock_response)):
            org = await client.get_organization("nonexistent")

        assert org is None


class TestJanuaClientUserOrganizations:
    """Test JanuaClient get_user_organizations method."""

    @pytest.mark.asyncio
    async def test_get_user_organizations_success(self):
        """Test getting user organizations."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organizations": [
                {"id": "org-1", "name": "Org 1", "slug": "org-1", "owner_id": "user-123"},
                {"id": "org-2", "name": "Org 2", "slug": "org-2", "owner_id": "user-456"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_authed_request", AsyncMock(return_value=mock_response)):
            orgs = await client.get_user_organizations("user-123")

        assert len(orgs) == 2
        assert orgs[0].name == "Org 1"
        assert orgs[1].name == "Org 2"


class TestJanuaClientExchangeCode:
    """Test JanuaClient exchange_code method."""

    @pytest.mark.asyncio
    async def test_exchange_code_success(self):
        """Test exchanging authorization code."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "access-123",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "refresh-456",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_client", return_value=mock_http):
            token = await client.exchange_code("auth-code", "https://callback.example.com")

        assert token.access_token == "access-123"
        assert token.refresh_token == "refresh-456"


class TestJanuaClientRefreshToken:
    """Test JanuaClient refresh_token method."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(self):
        """Test refreshing access token."""
        client = JanuaClient(
            base_url="https://auth.example.com",
            client_id="test",
            client_secret="test",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new-access-123",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_client", return_value=mock_http):
            token = await client.refresh_token("old-refresh-token")

        assert token.access_token == "new-access-123"
