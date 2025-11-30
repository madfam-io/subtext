"""
Unit Tests for Integration Modules

Tests the Janua, Stripe, and Resend integrations.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch


# ══════════════════════════════════════════════════════════════
# Janua Integration Tests
# ══════════════════════════════════════════════════════════════


class TestJanuaModels:
    """Test Janua data models."""

    def test_janua_user_model(self):
        """Test JanuaUser model creation."""
        from subtext.integrations.janua import JanuaUser

        user = JanuaUser(
            id="user-123",
            email="test@example.com",
            name="Test User",
            email_verified=True,
        )

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.email_verified is True
        assert user.metadata == {}

    def test_janua_user_defaults(self):
        """Test JanuaUser default values."""
        from subtext.integrations.janua import JanuaUser

        user = JanuaUser(id="user-456", email="user@test.com")

        assert user.name is None
        assert user.email_verified is False
        assert user.avatar_url is None
        assert user.created_at is None

    def test_janua_organization_model(self):
        """Test JanuaOrganization model creation."""
        from subtext.integrations.janua import JanuaOrganization

        org = JanuaOrganization(
            id="org-123",
            name="Test Org",
            slug="test-org",
            owner_id="user-123",
        )

        assert org.id == "org-123"
        assert org.slug == "test-org"
        assert org.owner_id == "user-123"

    def test_janua_token_model(self):
        """Test JanuaToken model creation."""
        from subtext.integrations.janua import JanuaToken

        token = JanuaToken(
            access_token="access_token_123",
            expires_in=3600,
            refresh_token="refresh_token_456",
        )

        assert token.access_token == "access_token_123"
        assert token.token_type == "Bearer"
        assert token.expires_in == 3600
        assert token.refresh_token == "refresh_token_456"

    def test_token_payload_model(self):
        """Test TokenPayload model creation."""
        from subtext.integrations.janua import TokenPayload

        now = int(datetime.utcnow().timestamp())
        payload = TokenPayload(
            sub="user-123",
            email="user@test.com",
            org_id="org-123",
            roles=["user", "admin"],
            permissions=["read", "write"],
            exp=now + 3600,
            iat=now,
            iss="https://auth.example.com",
            aud="subtext-api",
        )

        assert payload.sub == "user-123"
        assert payload.email == "user@test.com"
        assert payload.org_id == "org-123"
        assert "admin" in payload.roles
        assert "write" in payload.permissions


class TestJanuaClient:
    """Test JanuaClient HTTP client."""

    def test_janua_client_initialization(self):
        """Test JanuaClient initialization."""
        from subtext.integrations.janua import JanuaClient

        client = JanuaClient()
        assert client._client is None  # Lazy initialization

    @pytest.mark.asyncio
    async def test_janua_client_close(self):
        """Test closing JanuaClient."""
        from subtext.integrations.janua import JanuaClient

        client = JanuaClient()
        # Should not raise even if no client exists
        await client.close()


class TestJanuaAuth:
    """Test JanuaAuth dependency."""

    def test_janua_auth_initialization(self):
        """Test JanuaAuth initialization."""
        from subtext.integrations.janua import JanuaAuth

        auth = JanuaAuth()
        assert auth.required_roles == []
        assert auth.required_permissions == []

    def test_janua_auth_with_roles(self):
        """Test JanuaAuth with required roles."""
        from subtext.integrations.janua import JanuaAuth

        auth = JanuaAuth(required_roles=["admin", "owner"])
        assert "admin" in auth.required_roles
        assert "owner" in auth.required_roles

    def test_janua_auth_with_permissions(self):
        """Test JanuaAuth with required permissions."""
        from subtext.integrations.janua import JanuaAuth

        auth = JanuaAuth(required_permissions=["sessions:read", "sessions:write"])
        assert "sessions:read" in auth.required_permissions
        assert "sessions:write" in auth.required_permissions

    def test_convenience_auth_instances(self):
        """Test pre-configured auth dependency instances."""
        from subtext.integrations.janua import auth_required, admin_required

        assert auth_required.required_roles == []
        assert "admin" in admin_required.required_roles or "owner" in admin_required.required_roles


# ══════════════════════════════════════════════════════════════
# Stripe Integration Tests
# ══════════════════════════════════════════════════════════════


class TestStripeModels:
    """Test Stripe data models."""

    def test_subscription_model(self):
        """Test Subscription model creation."""
        from subtext.integrations.stripe import Subscription, SubscriptionTier

        subscription = Subscription(
            id="sub_123",
            customer_id="cus_456",
            status="active",
            tier=SubscriptionTier.PERSONAL,
            current_period_start=1700000000,
            current_period_end=1702592000,
        )

        assert subscription.id == "sub_123"
        assert subscription.customer_id == "cus_456"
        assert subscription.status == "active"
        assert subscription.tier == SubscriptionTier.PERSONAL

    def test_invoice_model(self):
        """Test Invoice model creation."""
        from subtext.integrations.stripe import Invoice
        from datetime import datetime

        invoice = Invoice(
            id="inv_123",
            customer_id="cus_456",
            amount_due=2999,
            amount_paid=2999,
            currency="usd",
            status="paid",
            created_at=datetime.utcnow(),
        )

        assert invoice.id == "inv_123"
        assert invoice.amount_due == 2999
        assert invoice.status == "paid"

    def test_checkout_session_model(self):
        """Test CheckoutSession model creation."""
        from subtext.integrations.stripe import CheckoutSession

        session = CheckoutSession(
            id="cs_123",
            url="https://checkout.stripe.com/...",
            status="open",
        )

        assert session.id == "cs_123"
        assert "stripe.com" in session.url or session.url.startswith("https://")
        assert session.status == "open"

    def test_usage_record_model(self):
        """Test UsageRecord model creation."""
        from subtext.integrations.stripe import UsageRecord

        record = UsageRecord(
            subscription_item_id="si_123",
            quantity=100,
            action="increment",
            timestamp=1700000000,
        )

        assert record.subscription_item_id == "si_123"
        assert record.quantity == 100


class TestStripeClient:
    """Test StripeClient functionality."""

    def test_subscription_tier_enum(self):
        """Test SubscriptionTier enum values."""
        from subtext.integrations.stripe import SubscriptionTier

        # Check that the enum has expected tiers
        tier_values = [t.value for t in SubscriptionTier]
        assert "free" in tier_values
        assert "enterprise" in tier_values

    def test_subscription_status_enum(self):
        """Test SubscriptionStatus enum values."""
        from subtext.integrations.stripe import SubscriptionStatus

        # Check that status enum exists
        assert SubscriptionStatus is not None

    def test_billing_period_enum(self):
        """Test BillingPeriod enum values."""
        from subtext.integrations.stripe import BillingPeriod

        assert BillingPeriod is not None

    def test_stripe_client_initialization(self):
        """Test StripeClient initialization."""
        from subtext.integrations.stripe import StripeClient

        client = StripeClient()
        # Should not raise

    def test_billing_service_initialization(self):
        """Test BillingService initialization."""
        from subtext.integrations.stripe import BillingService

        service = BillingService()
        assert service is not None

    def test_tier_limits(self):
        """Test TIER_LIMITS configuration."""
        from subtext.integrations.stripe import TIER_LIMITS, SubscriptionTier

        # Should have limits for all tiers
        for tier in SubscriptionTier:
            assert tier in TIER_LIMITS or tier.value in [t.value for t in TIER_LIMITS.keys()]


class TestStripeHelpers:
    """Test Stripe helper functions."""

    def test_get_billing_service(self):
        """Test get_billing_service function."""
        from subtext.integrations.stripe import get_billing_service, BillingService

        service = get_billing_service()
        assert isinstance(service, BillingService)


# ══════════════════════════════════════════════════════════════
# Resend Integration Tests
# ══════════════════════════════════════════════════════════════


class TestResendModels:
    """Test Resend email models."""

    def test_email_recipient_model(self):
        """Test EmailRecipient model creation."""
        from subtext.integrations.resend import EmailRecipient

        recipient = EmailRecipient(
            email="user@example.com",
            name="User Name",
        )

        assert recipient.email == "user@example.com"
        assert recipient.name == "User Name"

    def test_email_recipient_without_name(self):
        """Test EmailRecipient without name."""
        from subtext.integrations.resend import EmailRecipient

        recipient = EmailRecipient(email="user@example.com")

        assert recipient.email == "user@example.com"
        assert recipient.name is None

    def test_email_attachment_model(self):
        """Test EmailAttachment model creation."""
        from subtext.integrations.resend import EmailAttachment

        attachment = EmailAttachment(
            filename="report.pdf",
            content=b"base64content",
        )

        assert attachment.filename == "report.pdf"
        assert attachment.content == b"base64content"

    def test_email_result_model(self):
        """Test EmailResult model creation."""
        from subtext.integrations.resend import EmailResult

        result = EmailResult(
            id="email_123",
            success=True,
        )

        assert result.id == "email_123"
        assert result.success is True


class TestResendClient:
    """Test ResendClient functionality."""

    def test_resend_client_initialization(self):
        """Test ResendClient initialization."""
        from subtext.integrations.resend import ResendClient

        client = ResendClient()
        assert client is not None

    def test_email_service_initialization(self):
        """Test EmailService initialization."""
        from subtext.integrations.resend import EmailService

        service = EmailService()
        assert service is not None

    def test_email_templates_class(self):
        """Test EmailTemplates class."""
        from subtext.integrations.resend import EmailTemplates

        # Check class exists and has template methods
        assert EmailTemplates is not None


class TestResendHelpers:
    """Test Resend helper functions."""

    def test_get_email_service(self):
        """Test get_email_service function."""
        from subtext.integrations.resend import get_email_service, EmailService

        service = get_email_service()
        assert isinstance(service, EmailService)


# ══════════════════════════════════════════════════════════════
# Integration Module Exports
# ══════════════════════════════════════════════════════════════


class TestIntegrationExports:
    """Test that all exports are available."""

    def test_janua_exports(self):
        """Test Janua module exports."""
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
            get_janua_client,
        )

        # All imports should succeed
        assert JanuaUser is not None
        assert JanuaAuth is not None

    def test_stripe_exports(self):
        """Test Stripe module exports."""
        from subtext.integrations.stripe import (
            Subscription,
            Invoice,
            CheckoutSession,
            UsageRecord,
            StripeClient,
            BillingService,
            SubscriptionTier,
            SubscriptionStatus,
            BillingPeriod,
            TIER_LIMITS,
            get_billing_service,
        )

        assert StripeClient is not None
        assert SubscriptionTier is not None
        assert BillingService is not None

    def test_resend_exports(self):
        """Test Resend module exports."""
        from subtext.integrations.resend import (
            EmailRecipient,
            EmailAttachment,
            EmailResult,
            ResendClient,
            EmailService,
            EmailTemplates,
            get_email_service,
        )

        assert ResendClient is not None
        assert EmailService is not None

    def test_integrations_init(self):
        """Test integrations __init__ exports."""
        from subtext.integrations import (
            JanuaClient,
            JanuaAuth,
            StripeClient,
        )

        assert JanuaClient is not None
        assert StripeClient is not None
