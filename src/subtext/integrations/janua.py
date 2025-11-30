"""
Janua Authentication Integration

Integrates with the Janua auth platform (https://github.com/madfam-io/janua)
for user authentication, session management, and access control.
"""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import httpx
import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from subtext.config import settings

logger = structlog.get_logger()

# ══════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════


class JanuaUser(BaseModel):
    """User data from Janua."""

    id: str
    email: str
    name: str | None = None
    email_verified: bool = False
    avatar_url: str | None = None
    metadata: dict[str, Any] = {}
    created_at: datetime | None = None


class JanuaOrganization(BaseModel):
    """Organization data from Janua."""

    id: str
    name: str
    slug: str
    owner_id: str
    metadata: dict[str, Any] = {}


class JanuaToken(BaseModel):
    """Token response from Janua."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: str | None = None
    id_token: str | None = None


class TokenPayload(BaseModel):
    """Decoded JWT token payload."""

    sub: str  # User ID
    email: str
    org_id: str | None = None
    roles: list[str] = []
    permissions: list[str] = []
    exp: int
    iat: int
    iss: str
    aud: str | list[str]


# ══════════════════════════════════════════════════════════════
# Janua Client
# ══════════════════════════════════════════════════════════════


class JanuaClient:
    """
    HTTP client for Janua API interactions.

    Usage:
        client = JanuaClient()
        user = await client.get_user(user_id)
        orgs = await client.get_user_organizations(user_id)
    """

    def __init__(
        self,
        base_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ):
        self.base_url = (base_url or settings.janua_base_url).rstrip("/")
        self.client_id = client_id or settings.janua_client_id
        self.client_secret = client_secret or settings.janua_client_secret
        self._client: httpx.AsyncClient | None = None
        self._service_token: str | None = None
        self._token_expires_at: datetime | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get_service_token(self) -> str:
        """Get service-to-service authentication token."""
        if (
            self._service_token
            and self._token_expires_at
            and datetime.utcnow() < self._token_expires_at
        ):
            return self._service_token

        client = await self._get_client()
        response = await client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "audience": settings.janua_audience,
            },
        )
        response.raise_for_status()
        data = response.json()

        self._service_token = data["access_token"]
        self._token_expires_at = datetime.utcnow() + timedelta(
            seconds=data.get("expires_in", 3600) - 60
        )
        return self._service_token

    async def _authed_request(
        self, method: str, path: str, **kwargs
    ) -> httpx.Response:
        """Make authenticated request to Janua API."""
        token = await self._get_service_token()
        client = await self._get_client()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        return await client.request(method, path, headers=headers, **kwargs)

    # ──────────────────────────────────────────────────────────
    # User Operations
    # ──────────────────────────────────────────────────────────

    async def get_user(self, user_id: str) -> JanuaUser | None:
        """Get user by ID."""
        try:
            response = await self._authed_request("GET", f"/api/v1/users/{user_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return JanuaUser(**response.json())
        except httpx.HTTPError as e:
            logger.error("Failed to get user from Janua", user_id=user_id, error=str(e))
            raise

    async def get_user_by_email(self, email: str) -> JanuaUser | None:
        """Get user by email."""
        try:
            response = await self._authed_request(
                "GET", "/api/v1/users", params={"email": email}
            )
            response.raise_for_status()
            data = response.json()
            users = data.get("users", [])
            return JanuaUser(**users[0]) if users else None
        except httpx.HTTPError as e:
            logger.error("Failed to get user by email", email=email, error=str(e))
            raise

    async def create_user(
        self,
        email: str,
        password: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JanuaUser:
        """Create a new user in Janua."""
        try:
            response = await self._authed_request(
                "POST",
                "/api/v1/users",
                json={
                    "email": email,
                    "password": password,
                    "name": name,
                    "metadata": metadata or {},
                },
            )
            response.raise_for_status()
            return JanuaUser(**response.json())
        except httpx.HTTPError as e:
            logger.error("Failed to create user in Janua", email=email, error=str(e))
            raise

    async def update_user(
        self, user_id: str, **updates: Any
    ) -> JanuaUser:
        """Update user attributes."""
        try:
            response = await self._authed_request(
                "PATCH", f"/api/v1/users/{user_id}", json=updates
            )
            response.raise_for_status()
            return JanuaUser(**response.json())
        except httpx.HTTPError as e:
            logger.error("Failed to update user", user_id=user_id, error=str(e))
            raise

    # ──────────────────────────────────────────────────────────
    # Organization Operations
    # ──────────────────────────────────────────────────────────

    async def get_organization(self, org_id: str) -> JanuaOrganization | None:
        """Get organization by ID."""
        try:
            response = await self._authed_request(
                "GET", f"/api/v1/organizations/{org_id}"
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return JanuaOrganization(**response.json())
        except httpx.HTTPError as e:
            logger.error("Failed to get organization", org_id=org_id, error=str(e))
            raise

    async def get_user_organizations(self, user_id: str) -> list[JanuaOrganization]:
        """Get organizations for a user."""
        try:
            response = await self._authed_request(
                "GET", f"/api/v1/users/{user_id}/organizations"
            )
            response.raise_for_status()
            data = response.json()
            return [JanuaOrganization(**org) for org in data.get("organizations", [])]
        except httpx.HTTPError as e:
            logger.error(
                "Failed to get user organizations", user_id=user_id, error=str(e)
            )
            raise

    async def create_organization(
        self,
        name: str,
        owner_id: str,
        slug: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JanuaOrganization:
        """Create a new organization."""
        try:
            response = await self._authed_request(
                "POST",
                "/api/v1/organizations",
                json={
                    "name": name,
                    "owner_id": owner_id,
                    "slug": slug,
                    "metadata": metadata or {},
                },
            )
            response.raise_for_status()
            return JanuaOrganization(**response.json())
        except httpx.HTTPError as e:
            logger.error("Failed to create organization", name=name, error=str(e))
            raise

    # ──────────────────────────────────────────────────────────
    # Token Operations
    # ──────────────────────────────────────────────────────────

    async def exchange_code(self, code: str, redirect_uri: str) -> JanuaToken:
        """Exchange authorization code for tokens (OAuth flow)."""
        client = await self._get_client()
        response = await client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
        response.raise_for_status()
        return JanuaToken(**response.json())

    async def refresh_token(self, refresh_token: str) -> JanuaToken:
        """Refresh access token."""
        client = await self._get_client()
        response = await client.post(
            "/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": refresh_token,
            },
        )
        response.raise_for_status()
        return JanuaToken(**response.json())

    async def get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set for token verification."""
        client = await self._get_client()
        response = await client.get("/.well-known/jwks.json")
        response.raise_for_status()
        return response.json()


# ══════════════════════════════════════════════════════════════
# FastAPI Auth Dependency
# ══════════════════════════════════════════════════════════════


class JanuaAuth:
    """
    FastAPI dependency for JWT token validation with Janua.

    Usage:
        @app.get("/protected")
        async def protected_route(user: TokenPayload = Depends(JanuaAuth())):
            return {"user_id": user.sub}
    """

    def __init__(
        self,
        required_roles: list[str] | None = None,
        required_permissions: list[str] | None = None,
    ):
        self.required_roles = required_roles or []
        self.required_permissions = required_permissions or []
        self._jwks: dict[str, Any] | None = None
        self._jwks_fetched_at: datetime | None = None

    async def _get_jwks(self) -> dict[str, Any]:
        """Get JWKS with caching."""
        if (
            self._jwks
            and self._jwks_fetched_at
            and datetime.utcnow() - self._jwks_fetched_at < timedelta(hours=1)
        ):
            return self._jwks

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.janua_base_url}/.well-known/jwks.json"
            )
            response.raise_for_status()
            self._jwks = response.json()
            self._jwks_fetched_at = datetime.utcnow()
            return self._jwks

    async def _verify_token(self, token: str) -> TokenPayload:
        """Verify JWT token and extract payload."""
        try:
            # Get JWKS for verification
            jwks = await self._get_jwks()

            # Decode without verification first to get header
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            # Find matching key
            key = None
            for jwk in jwks.get("keys", []):
                if jwk.get("kid") == kid:
                    key = jwk
                    break

            if not key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: key not found",
                )

            # Verify and decode token
            payload = jwt.decode(
                token,
                key,
                algorithms=settings.janua_algorithms,
                audience=settings.janua_audience,
                issuer=settings.janua_base_url,
            )

            return TokenPayload(**payload)

        except JWTError as e:
            logger.warning("JWT verification failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
            )

    def _check_roles(self, payload: TokenPayload) -> None:
        """Check if user has required roles."""
        if self.required_roles:
            if not any(role in payload.roles for role in self.required_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient role permissions",
                )

    def _check_permissions(self, payload: TokenPayload) -> None:
        """Check if user has required permissions."""
        if self.required_permissions:
            if not all(perm in payload.permissions for perm in self.required_permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions",
                )

    async def __call__(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    ) -> TokenPayload:
        """Validate token and return payload."""
        token = credentials.credentials
        payload = await self._verify_token(token)

        # Check roles and permissions
        self._check_roles(payload)
        self._check_permissions(payload)

        # Store in request state for downstream use
        request.state.user = payload
        request.state.user_id = payload.sub
        request.state.org_id = payload.org_id

        return payload


# Convenience dependency instances
auth_required = JanuaAuth()
admin_required = JanuaAuth(required_roles=["admin", "owner"])


async def get_current_user(
    request: Request,
    payload: TokenPayload = Depends(auth_required),
) -> TokenPayload:
    """Get the current authenticated user."""
    return payload


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> TokenPayload | None:
    """Get current user if authenticated, None otherwise."""
    if not credentials:
        return None

    auth = JanuaAuth()
    try:
        return await auth._verify_token(credentials.credentials)
    except HTTPException:
        return None


# ══════════════════════════════════════════════════════════════
# Singleton Client
# ══════════════════════════════════════════════════════════════

_janua_client: JanuaClient | None = None


def get_janua_client() -> JanuaClient:
    """Get singleton Janua client instance."""
    global _janua_client
    if _janua_client is None:
        _janua_client = JanuaClient()
    return _janua_client
