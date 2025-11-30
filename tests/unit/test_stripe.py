"""
Unit Tests for Stripe Billing Integration

Tests the Stripe billing integration models and utilities.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from subtext.core.models import SubscriptionTier
from subtext.integrations.stripe import (
    BillingPeriod,
    SubscriptionStatus,
    CheckoutSession,
    Subscription,
    UsageRecord,
    Invoice,
    PaymentMethod,
    TIER_LIMITS,
    StripeClient,
)


# ============================================================
# Enum Tests
# ============================================================


class TestBillingPeriod:
    """Test BillingPeriod enum."""

    def test_billing_period_values(self):
        """Test billing period enum values."""
        assert BillingPeriod.MONTHLY.value == "monthly"
        assert BillingPeriod.YEARLY.value == "yearly"


class TestSubscriptionStatus:
    """Test SubscriptionStatus enum."""

    def test_subscription_status_values(self):
        """Test subscription status enum values."""
        assert SubscriptionStatus.ACTIVE.value == "active"
        assert SubscriptionStatus.PAST_DUE.value == "past_due"
        assert SubscriptionStatus.CANCELED.value == "canceled"
        assert SubscriptionStatus.INCOMPLETE.value == "incomplete"
        assert SubscriptionStatus.TRIALING.value == "trialing"
        assert SubscriptionStatus.UNPAID.value == "unpaid"


# ============================================================
# Model Tests
# ============================================================


class TestCheckoutSession:
    """Test CheckoutSession model."""

    def test_checkout_session_minimal(self):
        """Test checkout session with minimal fields."""
        session = CheckoutSession(
            id="cs_test123",
            url="https://checkout.stripe.com/test",
            status="open",
        )

        assert session.id == "cs_test123"
        assert session.url == "https://checkout.stripe.com/test"
        assert session.status == "open"
        assert session.customer_id is None
        assert session.subscription_id is None

    def test_checkout_session_full(self):
        """Test checkout session with all fields."""
        session = CheckoutSession(
            id="cs_test123",
            url="https://checkout.stripe.com/test",
            status="complete",
            customer_id="cus_123",
            subscription_id="sub_456",
        )

        assert session.customer_id == "cus_123"
        assert session.subscription_id == "sub_456"


class TestSubscription:
    """Test Subscription model."""

    def test_subscription_creation(self):
        """Test subscription model creation."""
        now = datetime.utcnow()
        sub = Subscription(
            id="sub_123",
            customer_id="cus_456",
            status=SubscriptionStatus.ACTIVE,
            tier=SubscriptionTier.PERSONAL,
            current_period_start=now,
            current_period_end=now,
        )

        assert sub.id == "sub_123"
        assert sub.customer_id == "cus_456"
        assert sub.status == SubscriptionStatus.ACTIVE
        assert sub.tier == SubscriptionTier.PERSONAL
        assert sub.cancel_at_period_end is False
        assert sub.trial_end is None

    def test_subscription_with_trial(self):
        """Test subscription with trial period."""
        now = datetime.utcnow()
        trial_end = datetime(2024, 12, 31)

        sub = Subscription(
            id="sub_123",
            customer_id="cus_456",
            status=SubscriptionStatus.TRIALING,
            tier=SubscriptionTier.TEAMS,
            current_period_start=now,
            current_period_end=now,
            cancel_at_period_end=True,
            trial_end=trial_end,
        )

        assert sub.status == SubscriptionStatus.TRIALING
        assert sub.trial_end == trial_end
        assert sub.cancel_at_period_end is True


class TestUsageRecord:
    """Test UsageRecord model."""

    def test_usage_record_creation(self):
        """Test usage record creation."""
        now = datetime.utcnow()
        record = UsageRecord(
            subscription_item_id="si_123",
            quantity=100,
            timestamp=now,
        )

        assert record.subscription_item_id == "si_123"
        assert record.quantity == 100
        assert record.timestamp == now
        assert record.action == "increment"

    def test_usage_record_custom_action(self):
        """Test usage record with custom action."""
        now = datetime.utcnow()
        record = UsageRecord(
            subscription_item_id="si_123",
            quantity=50,
            timestamp=now,
            action="set",
        )

        assert record.action == "set"


class TestInvoice:
    """Test Invoice model."""

    def test_invoice_creation(self):
        """Test invoice model creation."""
        now = datetime.utcnow()
        invoice = Invoice(
            id="inv_123",
            customer_id="cus_456",
            status="paid",
            amount_due=5000,
            amount_paid=5000,
            currency="usd",
            created_at=now,
        )

        assert invoice.id == "inv_123"
        assert invoice.amount_due == 5000
        assert invoice.amount_paid == 5000
        assert invoice.currency == "usd"
        assert invoice.hosted_invoice_url is None
        assert invoice.pdf_url is None

    def test_invoice_with_urls(self):
        """Test invoice with URLs."""
        now = datetime.utcnow()
        invoice = Invoice(
            id="inv_123",
            customer_id="cus_456",
            status="paid",
            amount_due=5000,
            amount_paid=5000,
            currency="usd",
            created_at=now,
            hosted_invoice_url="https://invoice.stripe.com/i/test",
            pdf_url="https://invoice.stripe.com/i/test/pdf",
        )

        assert invoice.hosted_invoice_url == "https://invoice.stripe.com/i/test"
        assert invoice.pdf_url == "https://invoice.stripe.com/i/test/pdf"


class TestPaymentMethod:
    """Test PaymentMethod model."""

    def test_payment_method_minimal(self):
        """Test payment method with minimal fields."""
        pm = PaymentMethod(
            id="pm_123",
            type="card",
        )

        assert pm.id == "pm_123"
        assert pm.type == "card"
        assert pm.card_brand is None
        assert pm.is_default is False

    def test_payment_method_card(self):
        """Test payment method with card details."""
        pm = PaymentMethod(
            id="pm_123",
            type="card",
            card_brand="visa",
            card_last4="4242",
            card_exp_month=12,
            card_exp_year=2025,
            is_default=True,
        )

        assert pm.card_brand == "visa"
        assert pm.card_last4 == "4242"
        assert pm.card_exp_month == 12
        assert pm.card_exp_year == 2025
        assert pm.is_default is True


# ============================================================
# Tier Limits Tests
# ============================================================


class TestTierLimits:
    """Test tier limits configuration."""

    def test_tier_limits_exist(self):
        """Test all tiers have limits defined."""
        for tier in SubscriptionTier:
            assert tier in TIER_LIMITS
            limits = TIER_LIMITS[tier]
            assert "monthly_minutes" in limits
            assert "api_calls" in limits
            assert "max_file_size_mb" in limits

    def test_free_tier_limits(self):
        """Test free tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.FREE]

        assert limits["monthly_minutes"] == 300
        assert limits["api_calls"] == 1000
        assert limits["max_file_size_mb"] == 100
        assert limits["realtime"] is False
        assert limits["voice_fingerprinting"] is False

    def test_personal_tier_limits(self):
        """Test personal tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.PERSONAL]

        assert limits["monthly_minutes"] == 3000
        assert limits["realtime"] is True
        assert limits["voice_fingerprinting"] is False

    def test_teams_tier_limits(self):
        """Test teams tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.TEAMS]

        assert limits["monthly_minutes"] == -1  # Unlimited
        assert limits["realtime"] is True
        assert limits["voice_fingerprinting"] is True

    def test_enterprise_tier_limits(self):
        """Test enterprise tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.ENTERPRISE]

        assert limits["monthly_minutes"] == -1  # Unlimited
        assert limits["api_calls"] == -1  # Unlimited
        assert "all" in limits["signals_enabled"]

    def test_api_tier_limits(self):
        """Test API tier has expected limits."""
        limits = TIER_LIMITS[SubscriptionTier.API]

        assert limits["monthly_minutes"] == -1  # Pay per use
        assert limits["api_calls"] == -1


# ============================================================
# StripeClient Tests
# ============================================================


class TestStripeClientCustomer:
    """Test StripeClient customer operations."""

    @pytest.mark.asyncio
    async def test_create_customer(self):
        """Test creating a customer."""
        mock_customer = MagicMock()
        mock_customer.id = "cus_test123"

        with patch("stripe.Customer.create", return_value=mock_customer):
            customer = await StripeClient.create_customer(
                email="test@example.com",
                name="Test User",
                metadata={"user_id": "123"},
            )

        assert customer.id == "cus_test123"

    @pytest.mark.asyncio
    async def test_create_customer_error(self):
        """Test customer creation error handling."""
        import stripe

        with patch("stripe.Customer.create", side_effect=stripe.StripeError("API error")):
            with pytest.raises(stripe.StripeError):
                await StripeClient.create_customer(email="test@example.com")

    @pytest.mark.asyncio
    async def test_get_customer(self):
        """Test retrieving a customer."""
        mock_customer = MagicMock()
        mock_customer.id = "cus_test123"

        with patch("stripe.Customer.retrieve", return_value=mock_customer):
            customer = await StripeClient.get_customer("cus_test123")

        assert customer.id == "cus_test123"

    @pytest.mark.asyncio
    async def test_get_customer_not_found(self):
        """Test retrieving non-existent customer."""
        import stripe

        with patch("stripe.Customer.retrieve", side_effect=stripe.InvalidRequestError("not found", None)):
            customer = await StripeClient.get_customer("cus_invalid")

        assert customer is None


class TestStripeClientSubscription:
    """Test StripeClient subscription operations."""

    @pytest.mark.asyncio
    async def test_get_subscription(self):
        """Test retrieving a subscription."""
        mock_sub = MagicMock()
        mock_sub.id = "sub_test123"

        with patch("stripe.Subscription.retrieve", return_value=mock_sub):
            sub = await StripeClient.get_subscription("sub_test123")

        assert sub is not None

    @pytest.mark.asyncio
    async def test_get_subscription_not_found(self):
        """Test retrieving non-existent subscription."""
        import stripe

        with patch("stripe.Subscription.retrieve", side_effect=stripe.InvalidRequestError("not found", None)):
            sub = await StripeClient.get_subscription("sub_invalid")

        assert sub is None

    @pytest.mark.asyncio
    async def test_cancel_subscription(self):
        """Test cancelling a subscription."""
        mock_sub = MagicMock()
        mock_sub.id = "sub_test123"

        with patch("stripe.Subscription.modify", return_value=mock_sub):
            sub = await StripeClient.cancel_subscription("sub_test123")

        assert sub.id == "sub_test123"


class TestStripeClientInvoice:
    """Test StripeClient invoice operations."""

    @pytest.mark.asyncio
    async def test_list_invoices(self):
        """Test listing customer invoices."""
        mock_invoice = MagicMock()
        mock_invoice.id = "inv_test123"
        mock_invoice.status = "paid"
        mock_invoice.customer = "cus_123"
        mock_invoice.amount_due = 5000
        mock_invoice.amount_paid = 5000
        mock_invoice.currency = "usd"
        mock_invoice.hosted_invoice_url = None
        mock_invoice.invoice_pdf = None
        mock_invoice.created = 1704067200

        mock_list = MagicMock()
        mock_list.data = [mock_invoice]

        with patch("stripe.Invoice.list", return_value=mock_list):
            invoices = await StripeClient.list_invoices("cus_123", limit=10)

        assert len(invoices) == 1


class TestStripeClientPaymentMethod:
    """Test StripeClient payment method operations."""

    @pytest.mark.asyncio
    async def test_list_payment_methods(self):
        """Test listing payment methods."""
        mock_pm = MagicMock()
        mock_pm.id = "pm_test123"
        mock_pm.type = "card"
        mock_pm.card = MagicMock()
        mock_pm.card.brand = "visa"
        mock_pm.card.last4 = "4242"
        mock_pm.card.exp_month = 12
        mock_pm.card.exp_year = 2025

        mock_list = MagicMock()
        mock_list.data = [mock_pm]

        with patch("stripe.PaymentMethod.list", return_value=mock_list):
            with patch("stripe.Customer.retrieve") as mock_cust:
                mock_cust.return_value = MagicMock()
                mock_cust.return_value.invoice_settings = MagicMock()
                mock_cust.return_value.invoice_settings.default_payment_method = None
                methods = await StripeClient.list_payment_methods("cus_123")

        assert len(methods) == 1
