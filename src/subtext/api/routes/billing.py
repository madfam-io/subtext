"""
Billing Routes

Stripe integration for subscription management and usage tracking.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from subtext.core.models import SubscriptionTier
from subtext.integrations.janua import TokenPayload, get_current_user
from subtext.integrations.stripe import get_billing_service

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════


class UsageResponse(BaseModel):
    """Usage information response."""

    tier: SubscriptionTier
    minutes_used: int
    minutes_limit: int
    minutes_remaining: int
    api_calls_used: int
    api_calls_limit: int
    api_calls_remaining: int
    usage_percent: float


class CheckoutRequest(BaseModel):
    """Request to create checkout session."""

    tier: SubscriptionTier
    success_url: str
    cancel_url: str


class CheckoutResponse(BaseModel):
    """Checkout session response."""

    checkout_url: str
    session_id: str


class PortalResponse(BaseModel):
    """Billing portal response."""

    portal_url: str


class TierLimitsResponse(BaseModel):
    """Tier limits information."""

    tier: SubscriptionTier
    monthly_minutes: int
    api_calls: int
    max_file_size_mb: int
    max_duration_minutes: int
    realtime: bool
    voice_fingerprinting: bool
    integrations: int


# ══════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    user: TokenPayload = Depends(get_current_user),
) -> UsageResponse:
    """Get current usage for the organization."""
    # In production, fetch from database
    # For now, return mock data
    tier = SubscriptionTier.FREE
    minutes_used = 45
    api_calls_used = 150

    billing = get_billing_service()
    limits = billing.check_usage_limits(tier, minutes_used, api_calls_used)
    tier_limits = billing.get_tier_limits(tier)

    return UsageResponse(
        tier=tier,
        minutes_used=minutes_used,
        minutes_limit=tier_limits["monthly_minutes"],
        minutes_remaining=limits["minutes_remaining"],
        api_calls_used=api_calls_used,
        api_calls_limit=tier_limits["api_calls"],
        api_calls_remaining=limits["api_calls_remaining"],
        usage_percent=limits["minutes_percent"],
    )


@router.get("/tiers", response_model=list[TierLimitsResponse])
async def list_tiers() -> list[TierLimitsResponse]:
    """List available subscription tiers and their limits."""
    billing = get_billing_service()

    tiers = []
    for tier in [
        SubscriptionTier.FREE,
        SubscriptionTier.PERSONAL,
        SubscriptionTier.TEAMS,
        SubscriptionTier.ENTERPRISE,
    ]:
        limits = billing.get_tier_limits(tier)
        tiers.append(
            TierLimitsResponse(
                tier=tier,
                monthly_minutes=limits["monthly_minutes"],
                api_calls=limits["api_calls"],
                max_file_size_mb=limits["max_file_size_mb"],
                max_duration_minutes=limits["max_duration_minutes"],
                realtime=limits["realtime"],
                voice_fingerprinting=limits["voice_fingerprinting"],
                integrations=limits["integrations"],
            )
        )

    return tiers


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    request: CheckoutRequest,
    user: TokenPayload = Depends(get_current_user),
) -> CheckoutResponse:
    """Create a Stripe checkout session for subscription upgrade."""
    billing = get_billing_service()

    # Get or create Stripe customer ID
    # In production, fetch from database
    customer_id = f"cus_{user.org_id}"  # Mock

    try:
        session = await billing.create_upgrade_checkout(
            customer_id=customer_id,
            tier=request.tier,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            trial_days=14 if request.tier == SubscriptionTier.PERSONAL else None,
        )

        return CheckoutResponse(
            checkout_url=session.url,
            session_id=session.id,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/portal", response_model=PortalResponse)
async def get_billing_portal(
    return_url: str,
    user: TokenPayload = Depends(get_current_user),
) -> PortalResponse:
    """Get Stripe billing portal URL for self-service management."""
    billing = get_billing_service()

    # Get Stripe customer ID
    customer_id = f"cus_{user.org_id}"  # Mock

    try:
        portal_url = await billing.create_billing_portal(
            customer_id=customer_id,
            return_url=return_url,
        )

        return PortalResponse(portal_url=portal_url)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/invoices")
async def list_invoices(
    user: TokenPayload = Depends(get_current_user),
    limit: int = 10,
) -> list[dict]:
    """List invoices for the organization."""
    from subtext.integrations.stripe import StripeClient

    customer_id = f"cus_{user.org_id}"  # Mock

    try:
        invoices = await StripeClient.list_invoices(customer_id, limit)
        return [inv.model_dump() for inv in invoices]
    except Exception:
        return []  # Return empty if no Stripe customer
