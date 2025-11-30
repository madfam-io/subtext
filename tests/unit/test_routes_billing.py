"""
Unit Tests for Billing Routes

Tests Stripe integration endpoints for subscriptions and usage.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from subtext.api.routes.billing import (
    router,
    UsageResponse,
    CheckoutRequest,
    CheckoutResponse,
    PortalResponse,
    TierLimitsResponse,
    get_usage,
    list_tiers,
    create_checkout,
    get_billing_portal,
    list_invoices,
)
from subtext.core.models import SubscriptionTier
from subtext.integrations.janua import TokenPayload


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
        roles=["user"],
        permissions=["billing:read", "billing:write"],
        exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        iat=int(datetime.utcnow().timestamp()),
        iss="https://auth.example.com",
        aud="subtext-api",
    )


@pytest.fixture
def mock_billing_service():
    """Create a mock billing service."""
    service = MagicMock()
    service.get_tier_limits.return_value = {
        "monthly_minutes": 60,
        "api_calls": 500,
        "max_file_size_mb": 25,
        "max_duration_minutes": 30,
        "realtime": False,
        "voice_fingerprinting": False,
        "integrations": 1,
    }
    service.check_usage_limits.return_value = {
        "minutes_remaining": 15,
        "api_calls_remaining": 350,
        "minutes_percent": 75.0,
        "api_calls_percent": 30.0,
        "minutes_exceeded": False,
        "api_calls_exceeded": False,
    }
    return service


# ══════════════════════════════════════════════════════════════
# Response Model Tests
# ══════════════════════════════════════════════════════════════


class TestResponseModels:
    """Test response model schemas."""

    def test_usage_response(self):
        """Test UsageResponse model."""
        response = UsageResponse(
            tier=SubscriptionTier.FREE,
            minutes_used=45,
            minutes_limit=60,
            minutes_remaining=15,
            api_calls_used=150,
            api_calls_limit=500,
            api_calls_remaining=350,
            usage_percent=75.0,
        )
        assert response.tier == SubscriptionTier.FREE
        assert response.minutes_used == 45
        assert response.usage_percent == 75.0

    def test_checkout_request(self):
        """Test CheckoutRequest model."""
        request = CheckoutRequest(
            tier=SubscriptionTier.PERSONAL,
            success_url="https://example.com/success",
            cancel_url="https://example.com/cancel",
        )
        assert request.tier == SubscriptionTier.PERSONAL
        assert "success" in request.success_url

    def test_checkout_response(self):
        """Test CheckoutResponse model."""
        response = CheckoutResponse(
            checkout_url="https://checkout.stripe.com/pay/xyz",
            session_id="cs_test_123",
        )
        assert "stripe.com" in response.checkout_url
        assert response.session_id.startswith("cs_")

    def test_portal_response(self):
        """Test PortalResponse model."""
        response = PortalResponse(
            portal_url="https://billing.stripe.com/session/xyz",
        )
        assert "stripe.com" in response.portal_url

    def test_tier_limits_response(self):
        """Test TierLimitsResponse model."""
        response = TierLimitsResponse(
            tier=SubscriptionTier.TEAMS,
            monthly_minutes=500,
            api_calls=10000,
            max_file_size_mb=100,
            max_duration_minutes=120,
            realtime=True,
            voice_fingerprinting=True,
            integrations=10,
        )
        assert response.tier == SubscriptionTier.TEAMS
        assert response.realtime is True


# ══════════════════════════════════════════════════════════════
# Get Usage Tests
# ══════════════════════════════════════════════════════════════


class TestGetUsage:
    """Test get usage endpoint."""

    @pytest.mark.asyncio
    async def test_get_usage_success(self, mock_user, mock_billing_service):
        """Test getting usage information."""
        with patch("subtext.api.routes.billing.get_billing_service", return_value=mock_billing_service):
            result = await get_usage(mock_user)

        assert result.tier == SubscriptionTier.FREE
        assert result.minutes_used == 45
        assert result.minutes_remaining == 15
        assert result.usage_percent == 75.0


# ══════════════════════════════════════════════════════════════
# List Tiers Tests
# ══════════════════════════════════════════════════════════════


class TestListTiers:
    """Test list tiers endpoint."""

    @pytest.mark.asyncio
    async def test_list_tiers_success(self, mock_billing_service):
        """Test listing available tiers."""
        with patch("subtext.api.routes.billing.get_billing_service", return_value=mock_billing_service):
            result = await list_tiers()

        assert len(result) == 4
        tier_names = [t.tier for t in result]
        assert SubscriptionTier.FREE in tier_names
        assert SubscriptionTier.PERSONAL in tier_names
        assert SubscriptionTier.TEAMS in tier_names
        assert SubscriptionTier.ENTERPRISE in tier_names


# ══════════════════════════════════════════════════════════════
# Create Checkout Tests
# ══════════════════════════════════════════════════════════════


class TestCreateCheckout:
    """Test create checkout endpoint."""

    @pytest.mark.asyncio
    async def test_create_checkout_success(self, mock_user):
        """Test creating checkout session."""
        request = CheckoutRequest(
            tier=SubscriptionTier.PERSONAL,
            success_url="https://app.example.com/success",
            cancel_url="https://app.example.com/cancel",
        )

        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/pay/cs_test_123"
        mock_session.id = "cs_test_123"

        mock_billing = MagicMock()
        mock_billing.create_upgrade_checkout = AsyncMock(return_value=mock_session)

        with patch("subtext.api.routes.billing.get_billing_service", return_value=mock_billing):
            result = await create_checkout(request, mock_user)

        assert result.checkout_url == "https://checkout.stripe.com/pay/cs_test_123"
        assert result.session_id == "cs_test_123"

        mock_billing.create_upgrade_checkout.assert_called_once_with(
            customer_id="cus_org-456",
            tier=SubscriptionTier.PERSONAL,
            success_url="https://app.example.com/success",
            cancel_url="https://app.example.com/cancel",
            trial_days=14,  # Personal tier gets trial
        )

    @pytest.mark.asyncio
    async def test_create_checkout_teams_no_trial(self, mock_user):
        """Test creating checkout for Teams tier (no trial)."""
        request = CheckoutRequest(
            tier=SubscriptionTier.TEAMS,
            success_url="https://app.example.com/success",
            cancel_url="https://app.example.com/cancel",
        )

        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/pay/cs_test_456"
        mock_session.id = "cs_test_456"

        mock_billing = MagicMock()
        mock_billing.create_upgrade_checkout = AsyncMock(return_value=mock_session)

        with patch("subtext.api.routes.billing.get_billing_service", return_value=mock_billing):
            result = await create_checkout(request, mock_user)

        mock_billing.create_upgrade_checkout.assert_called_once_with(
            customer_id="cus_org-456",
            tier=SubscriptionTier.TEAMS,
            success_url="https://app.example.com/success",
            cancel_url="https://app.example.com/cancel",
            trial_days=None,  # No trial for Teams
        )

    @pytest.mark.asyncio
    async def test_create_checkout_failure(self, mock_user):
        """Test checkout creation failure."""
        from fastapi import HTTPException

        request = CheckoutRequest(
            tier=SubscriptionTier.PERSONAL,
            success_url="https://app.example.com/success",
            cancel_url="https://app.example.com/cancel",
        )

        mock_billing = MagicMock()
        mock_billing.create_upgrade_checkout = AsyncMock(
            side_effect=Exception("Stripe API error")
        )

        with patch("subtext.api.routes.billing.get_billing_service", return_value=mock_billing):
            with pytest.raises(HTTPException) as exc_info:
                await create_checkout(request, mock_user)

        assert exc_info.value.status_code == 400
        assert "Stripe API error" in exc_info.value.detail


# ══════════════════════════════════════════════════════════════
# Get Billing Portal Tests
# ══════════════════════════════════════════════════════════════


class TestGetBillingPortal:
    """Test get billing portal endpoint."""

    @pytest.mark.asyncio
    async def test_get_portal_success(self, mock_user):
        """Test getting billing portal URL."""
        mock_billing = MagicMock()
        mock_billing.create_billing_portal = AsyncMock(
            return_value="https://billing.stripe.com/session/bps_123"
        )

        with patch("subtext.api.routes.billing.get_billing_service", return_value=mock_billing):
            result = await get_billing_portal(
                return_url="https://app.example.com/settings",
                user=mock_user,
            )

        assert result.portal_url == "https://billing.stripe.com/session/bps_123"

        mock_billing.create_billing_portal.assert_called_once_with(
            customer_id="cus_org-456",
            return_url="https://app.example.com/settings",
        )

    @pytest.mark.asyncio
    async def test_get_portal_failure(self, mock_user):
        """Test billing portal failure."""
        from fastapi import HTTPException

        mock_billing = MagicMock()
        mock_billing.create_billing_portal = AsyncMock(
            side_effect=Exception("Customer not found")
        )

        with patch("subtext.api.routes.billing.get_billing_service", return_value=mock_billing):
            with pytest.raises(HTTPException) as exc_info:
                await get_billing_portal(
                    return_url="https://app.example.com/settings",
                    user=mock_user,
                )

        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════
# List Invoices Tests
# ══════════════════════════════════════════════════════════════


class TestListInvoices:
    """Test list invoices endpoint."""

    @pytest.mark.asyncio
    async def test_list_invoices_success(self, mock_user):
        """Test listing invoices."""
        mock_invoice = MagicMock()
        mock_invoice.model_dump.return_value = {
            "id": "inv_123",
            "amount_due": 2999,
            "status": "paid",
        }

        with patch("subtext.integrations.stripe.StripeClient.list_invoices", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [mock_invoice]
            result = await list_invoices(mock_user, limit=10)

        assert len(result) == 1
        assert result[0]["id"] == "inv_123"
        assert result[0]["status"] == "paid"

    @pytest.mark.asyncio
    async def test_list_invoices_empty(self, mock_user):
        """Test listing invoices when none exist."""
        with patch("subtext.integrations.stripe.StripeClient.list_invoices", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []
            result = await list_invoices(mock_user, limit=10)

        assert result == []

    @pytest.mark.asyncio
    async def test_list_invoices_error(self, mock_user):
        """Test listing invoices returns empty on error."""
        with patch("subtext.integrations.stripe.StripeClient.list_invoices", new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = Exception("Stripe error")
            result = await list_invoices(mock_user, limit=10)

        assert result == []


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
        assert "/usage" in routes
        assert "/tiers" in routes
        assert "/checkout" in routes
        assert "/portal" in routes
        assert "/invoices" in routes
