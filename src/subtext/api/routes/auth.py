"""
Authentication Routes

Integrates with Janua for user authentication and organization management.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from subtext.integrations.janua import (
    JanuaAuth,
    TokenPayload,
    get_current_user,
    get_janua_client,
)

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request/Response Models
# ══════════════════════════════════════════════════════════════


class UserResponse(BaseModel):
    """User information response."""

    id: str
    email: str
    name: str | None
    org_id: str | None
    roles: list[str]


class OrganizationResponse(BaseModel):
    """Organization information response."""

    id: str
    name: str
    slug: str


class CreateOrganizationRequest(BaseModel):
    """Request to create a new organization."""

    name: str
    slug: str | None = None


# ══════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    user: TokenPayload = Depends(get_current_user),
) -> UserResponse:
    """Get current authenticated user information."""
    return UserResponse(
        id=user.sub,
        email=user.email,
        name=None,  # Would fetch from Janua
        org_id=user.org_id,
        roles=user.roles,
    )


@router.get("/organizations", response_model=list[OrganizationResponse])
async def list_user_organizations(
    user: TokenPayload = Depends(get_current_user),
) -> list[OrganizationResponse]:
    """List organizations the user belongs to."""
    janua = get_janua_client()
    orgs = await janua.get_user_organizations(user.sub)

    return [
        OrganizationResponse(
            id=org.id,
            name=org.name,
            slug=org.slug,
        )
        for org in orgs
    ]


@router.post("/organizations", response_model=OrganizationResponse, status_code=201)
async def create_organization(
    request: CreateOrganizationRequest,
    user: TokenPayload = Depends(get_current_user),
) -> OrganizationResponse:
    """Create a new organization."""
    janua = get_janua_client()

    # Create org in Janua
    org = await janua.create_organization(
        name=request.name,
        owner_id=user.sub,
        slug=request.slug,
    )

    # Set up billing for the new org
    from subtext.integrations.stripe import get_billing_service

    billing = get_billing_service()
    await billing.setup_organization_billing(
        org_id=org.id,
        email=user.email,
        name=request.name,
    )

    # Send welcome email
    from subtext.integrations.resend import get_email_service

    email_service = get_email_service()
    await email_service.send_welcome(
        email=user.email,
        name=None,
    )

    return OrganizationResponse(
        id=org.id,
        name=org.name,
        slug=org.slug,
    )


@router.get("/organizations/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> OrganizationResponse:
    """Get organization details."""
    # Verify user has access to this org
    if user.org_id != org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this organization",
        )

    janua = get_janua_client()
    org = await janua.get_organization(org_id)

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    return OrganizationResponse(
        id=org.id,
        name=org.name,
        slug=org.slug,
    )
