"""
Stripe Billing Integration

Handles subscription management, usage-based billing, and payment processing
for Subtext's tiered pricing model.
"""

from datetime import datetime
from enum import Enum
from typing import Any

import stripe
import structlog
from pydantic import BaseModel

from subtext.config import settings
from subtext.core.models import SubscriptionTier

logger = structlog.get_logger()

# Initialize Stripe
stripe.api_key = settings.stripe_secret_key


# ══════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════


class BillingPeriod(str, Enum):
    MONTHLY = "monthly"
    YEARLY = "yearly"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"
    TRIALING = "trialing"
    UNPAID = "unpaid"


class CheckoutSession(BaseModel):
    """Stripe checkout session data."""

    id: str
    url: str
    status: str
    customer_id: str | None = None
    subscription_id: str | None = None


class Subscription(BaseModel):
    """Subscription details."""

    id: str
    customer_id: str
    status: SubscriptionStatus
    tier: SubscriptionTier
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False
    trial_end: datetime | None = None


class UsageRecord(BaseModel):
    """Usage record for metered billing."""

    subscription_item_id: str
    quantity: int
    timestamp: datetime
    action: str = "increment"


class Invoice(BaseModel):
    """Invoice details."""

    id: str
    customer_id: str
    status: str
    amount_due: int
    amount_paid: int
    currency: str
    hosted_invoice_url: str | None = None
    pdf_url: str | None = None
    created_at: datetime


class PaymentMethod(BaseModel):
    """Payment method details."""

    id: str
    type: str
    card_brand: str | None = None
    card_last4: str | None = None
    card_exp_month: int | None = None
    card_exp_year: int | None = None
    is_default: bool = False


# ══════════════════════════════════════════════════════════════
# Tier Configuration
# ══════════════════════════════════════════════════════════════

TIER_LIMITS = {
    SubscriptionTier.FREE: {
        "monthly_minutes": 300,  # 5 hours
        "api_calls": 1000,
        "max_file_size_mb": 100,
        "max_duration_minutes": 30,
        "signals_enabled": ["basic"],
        "realtime": False,
        "voice_fingerprinting": False,
        "integrations": 2,
    },
    SubscriptionTier.PERSONAL: {
        "monthly_minutes": 3000,  # 50 hours
        "api_calls": 10000,
        "max_file_size_mb": 500,
        "max_duration_minutes": 120,
        "signals_enabled": ["basic", "advanced"],
        "realtime": True,
        "voice_fingerprinting": False,
        "integrations": 5,
    },
    SubscriptionTier.TEAMS: {
        "monthly_minutes": -1,  # Unlimited
        "api_calls": 100000,
        "max_file_size_mb": 1000,
        "max_duration_minutes": 240,
        "signals_enabled": ["basic", "advanced", "enterprise"],
        "realtime": True,
        "voice_fingerprinting": True,
        "integrations": 10,
    },
    SubscriptionTier.ENTERPRISE: {
        "monthly_minutes": -1,
        "api_calls": -1,
        "max_file_size_mb": 2000,
        "max_duration_minutes": 480,
        "signals_enabled": ["all"],
        "realtime": True,
        "voice_fingerprinting": True,
        "integrations": -1,
    },
    SubscriptionTier.API: {
        "monthly_minutes": -1,  # Pay per use
        "api_calls": -1,
        "max_file_size_mb": 1000,
        "max_duration_minutes": 240,
        "signals_enabled": ["all"],
        "realtime": True,
        "voice_fingerprinting": True,
        "integrations": -1,
    },
}


# ══════════════════════════════════════════════════════════════
# Stripe Client
# ══════════════════════════════════════════════════════════════


class StripeClient:
    """
    Low-level Stripe API client.

    Handles direct Stripe API interactions.
    """

    # ──────────────────────────────────────────────────────────
    # Customer Operations
    # ──────────────────────────────────────────────────────────

    @staticmethod
    async def create_customer(
        email: str,
        name: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> stripe.Customer:
        """Create a new Stripe customer."""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata=metadata or {},
            )
            logger.info("Created Stripe customer", customer_id=customer.id, email=email)
            return customer
        except stripe.StripeError as e:
            logger.error("Failed to create Stripe customer", email=email, error=str(e))
            raise

    @staticmethod
    async def get_customer(customer_id: str) -> stripe.Customer | None:
        """Get customer by ID."""
        try:
            return stripe.Customer.retrieve(customer_id)
        except stripe.InvalidRequestError:
            return None

    @staticmethod
    async def update_customer(
        customer_id: str, **updates: Any
    ) -> stripe.Customer:
        """Update customer attributes."""
        return stripe.Customer.modify(customer_id, **updates)

    # ──────────────────────────────────────────────────────────
    # Checkout Operations
    # ──────────────────────────────────────────────────────────

    @staticmethod
    async def create_checkout_session(
        customer_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        trial_days: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> CheckoutSession:
        """Create a checkout session for subscription."""
        try:
            params: dict[str, Any] = {
                "customer": customer_id,
                "payment_method_types": ["card"],
                "line_items": [{"price": price_id, "quantity": 1}],
                "mode": "subscription",
                "success_url": success_url,
                "cancel_url": cancel_url,
                "metadata": metadata or {},
            }

            if trial_days:
                params["subscription_data"] = {"trial_period_days": trial_days}

            session = stripe.checkout.Session.create(**params)

            logger.info(
                "Created checkout session",
                session_id=session.id,
                customer_id=customer_id,
            )

            return CheckoutSession(
                id=session.id,
                url=session.url,
                status=session.status,
                customer_id=customer_id,
            )
        except stripe.StripeError as e:
            logger.error("Failed to create checkout session", error=str(e))
            raise

    @staticmethod
    async def create_portal_session(
        customer_id: str,
        return_url: str,
    ) -> str:
        """Create a billing portal session for self-service management."""
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return session.url

    # ──────────────────────────────────────────────────────────
    # Subscription Operations
    # ──────────────────────────────────────────────────────────

    @staticmethod
    async def get_subscription(subscription_id: str) -> stripe.Subscription | None:
        """Get subscription by ID."""
        try:
            return stripe.Subscription.retrieve(subscription_id)
        except stripe.InvalidRequestError:
            return None

    @staticmethod
    async def cancel_subscription(
        subscription_id: str,
        immediately: bool = False,
    ) -> stripe.Subscription:
        """Cancel a subscription."""
        if immediately:
            return stripe.Subscription.delete(subscription_id)
        else:
            return stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True,
            )

    @staticmethod
    async def update_subscription(
        subscription_id: str,
        price_id: str,
    ) -> stripe.Subscription:
        """Update subscription to a new price/tier."""
        subscription = stripe.Subscription.retrieve(subscription_id)
        return stripe.Subscription.modify(
            subscription_id,
            items=[
                {
                    "id": subscription["items"]["data"][0]["id"],
                    "price": price_id,
                }
            ],
            proration_behavior="create_prorations",
        )

    # ──────────────────────────────────────────────────────────
    # Usage-Based Billing
    # ──────────────────────────────────────────────────────────

    @staticmethod
    async def report_usage(
        subscription_item_id: str,
        quantity: int,
        timestamp: int | None = None,
        action: str = "increment",
    ) -> stripe.UsageRecord:
        """Report usage for metered billing (API tier)."""
        params: dict[str, Any] = {
            "quantity": quantity,
            "action": action,
        }
        if timestamp:
            params["timestamp"] = timestamp

        return stripe.SubscriptionItem.create_usage_record(
            subscription_item_id,
            **params,
        )

    @staticmethod
    async def get_usage_summary(
        subscription_item_id: str,
    ) -> list[stripe.UsageRecordSummary]:
        """Get usage summary for a subscription item."""
        summaries = stripe.SubscriptionItem.list_usage_record_summaries(
            subscription_item_id,
            limit=10,
        )
        return summaries.data

    # ──────────────────────────────────────────────────────────
    # Invoice Operations
    # ──────────────────────────────────────────────────────────

    @staticmethod
    async def list_invoices(
        customer_id: str,
        limit: int = 10,
    ) -> list[Invoice]:
        """List invoices for a customer."""
        invoices = stripe.Invoice.list(customer=customer_id, limit=limit)
        return [
            Invoice(
                id=inv.id,
                customer_id=inv.customer,
                status=inv.status,
                amount_due=inv.amount_due,
                amount_paid=inv.amount_paid,
                currency=inv.currency,
                hosted_invoice_url=inv.hosted_invoice_url,
                pdf_url=inv.invoice_pdf,
                created_at=datetime.fromtimestamp(inv.created),
            )
            for inv in invoices.data
        ]

    # ──────────────────────────────────────────────────────────
    # Payment Method Operations
    # ──────────────────────────────────────────────────────────

    @staticmethod
    async def list_payment_methods(
        customer_id: str,
    ) -> list[PaymentMethod]:
        """List payment methods for a customer."""
        methods = stripe.PaymentMethod.list(customer=customer_id, type="card")
        customer = stripe.Customer.retrieve(customer_id)
        default_pm = customer.invoice_settings.default_payment_method

        return [
            PaymentMethod(
                id=pm.id,
                type=pm.type,
                card_brand=pm.card.brand if pm.card else None,
                card_last4=pm.card.last4 if pm.card else None,
                card_exp_month=pm.card.exp_month if pm.card else None,
                card_exp_year=pm.card.exp_year if pm.card else None,
                is_default=pm.id == default_pm,
            )
            for pm in methods.data
        ]


# ══════════════════════════════════════════════════════════════
# Billing Service
# ══════════════════════════════════════════════════════════════


class BillingService:
    """
    High-level billing service for Subtext.

    Handles business logic around subscriptions, usage tracking, and tier management.
    """

    def __init__(self):
        self.client = StripeClient()

    # ──────────────────────────────────────────────────────────
    # Organization Billing
    # ──────────────────────────────────────────────────────────

    async def setup_organization_billing(
        self,
        org_id: str,
        email: str,
        name: str,
    ) -> str:
        """Set up billing for a new organization."""
        customer = await self.client.create_customer(
            email=email,
            name=name,
            metadata={"org_id": org_id},
        )
        return customer.id

    async def get_subscription_tier(
        self,
        stripe_subscription_id: str | None,
    ) -> SubscriptionTier:
        """Get the subscription tier from a Stripe subscription."""
        if not stripe_subscription_id:
            return SubscriptionTier.FREE

        subscription = await self.client.get_subscription(stripe_subscription_id)
        if not subscription:
            return SubscriptionTier.FREE

        # Map price ID to tier
        price_id = subscription["items"]["data"][0]["price"]["id"]
        return self._price_to_tier(price_id)

    def _price_to_tier(self, price_id: str) -> SubscriptionTier:
        """Map Stripe price ID to subscription tier."""
        mapping = {
            settings.stripe_price_personal_monthly: SubscriptionTier.PERSONAL,
            settings.stripe_price_teams_monthly: SubscriptionTier.TEAMS,
            settings.stripe_price_api_per_minute: SubscriptionTier.API,
        }
        return mapping.get(price_id, SubscriptionTier.FREE)

    def _tier_to_price(self, tier: SubscriptionTier) -> str:
        """Map subscription tier to Stripe price ID."""
        mapping = {
            SubscriptionTier.PERSONAL: settings.stripe_price_personal_monthly,
            SubscriptionTier.TEAMS: settings.stripe_price_teams_monthly,
            SubscriptionTier.API: settings.stripe_price_api_per_minute,
        }
        return mapping.get(tier, settings.stripe_price_personal_monthly)

    # ──────────────────────────────────────────────────────────
    # Checkout & Portal
    # ──────────────────────────────────────────────────────────

    async def create_upgrade_checkout(
        self,
        customer_id: str,
        tier: SubscriptionTier,
        success_url: str,
        cancel_url: str,
        trial_days: int | None = None,
    ) -> CheckoutSession:
        """Create checkout session to upgrade to a tier."""
        price_id = self._tier_to_price(tier)
        return await self.client.create_checkout_session(
            customer_id=customer_id,
            price_id=price_id,
            success_url=success_url,
            cancel_url=cancel_url,
            trial_days=trial_days,
        )

    async def create_billing_portal(
        self,
        customer_id: str,
        return_url: str,
    ) -> str:
        """Create billing portal URL for self-service."""
        return await self.client.create_portal_session(
            customer_id=customer_id,
            return_url=return_url,
        )

    # ──────────────────────────────────────────────────────────
    # Usage Tracking
    # ──────────────────────────────────────────────────────────

    async def track_audio_usage(
        self,
        org_id: str,
        subscription_item_id: str,
        minutes: float,
    ) -> None:
        """Track audio processing usage for API billing."""
        # Round up to nearest minute
        quantity = max(1, int(minutes + 0.5))

        await self.client.report_usage(
            subscription_item_id=subscription_item_id,
            quantity=quantity,
            action="increment",
        )

        logger.info(
            "Tracked audio usage",
            org_id=org_id,
            minutes=minutes,
            quantity=quantity,
        )

    def check_usage_limits(
        self,
        tier: SubscriptionTier,
        monthly_minutes_used: int,
        api_calls_used: int,
    ) -> dict[str, Any]:
        """Check if organization is within usage limits."""
        limits = TIER_LIMITS[tier]

        minutes_limit = limits["monthly_minutes"]
        api_limit = limits["api_calls"]

        return {
            "within_limits": (
                (minutes_limit == -1 or monthly_minutes_used < minutes_limit)
                and (api_limit == -1 or api_calls_used < api_limit)
            ),
            "minutes_remaining": (
                -1 if minutes_limit == -1 else max(0, minutes_limit - monthly_minutes_used)
            ),
            "api_calls_remaining": (
                -1 if api_limit == -1 else max(0, api_limit - api_calls_used)
            ),
            "minutes_percent": (
                0 if minutes_limit == -1 else (monthly_minutes_used / minutes_limit * 100)
            ),
            "api_percent": (
                0 if api_limit == -1 else (api_calls_used / api_limit * 100)
            ),
        }

    def get_tier_limits(self, tier: SubscriptionTier) -> dict[str, Any]:
        """Get limits for a subscription tier."""
        return TIER_LIMITS.get(tier, TIER_LIMITS[SubscriptionTier.FREE])

    # ──────────────────────────────────────────────────────────
    # Webhook Processing
    # ──────────────────────────────────────────────────────────

    async def handle_webhook_event(
        self,
        payload: bytes,
        signature: str,
    ) -> dict[str, Any]:
        """Handle Stripe webhook event."""
        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                settings.stripe_webhook_secret,
            )
        except stripe.SignatureVerificationError:
            logger.warning("Invalid webhook signature")
            raise ValueError("Invalid signature")

        logger.info("Processing webhook event", event_type=event.type)

        handlers = {
            "checkout.session.completed": self._handle_checkout_completed,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "invoice.paid": self._handle_invoice_paid,
            "invoice.payment_failed": self._handle_payment_failed,
        }

        handler = handlers.get(event.type)
        if handler:
            return await handler(event.data.object)

        return {"status": "ignored", "event_type": event.type}

    async def _handle_checkout_completed(
        self, session: stripe.checkout.Session
    ) -> dict[str, Any]:
        """Handle successful checkout."""
        return {
            "status": "processed",
            "action": "subscription_created",
            "customer_id": session.customer,
            "subscription_id": session.subscription,
        }

    async def _handle_subscription_updated(
        self, subscription: stripe.Subscription
    ) -> dict[str, Any]:
        """Handle subscription update."""
        return {
            "status": "processed",
            "action": "subscription_updated",
            "subscription_id": subscription.id,
            "new_status": subscription.status,
        }

    async def _handle_subscription_deleted(
        self, subscription: stripe.Subscription
    ) -> dict[str, Any]:
        """Handle subscription cancellation."""
        return {
            "status": "processed",
            "action": "subscription_canceled",
            "subscription_id": subscription.id,
        }

    async def _handle_invoice_paid(
        self, invoice: stripe.Invoice
    ) -> dict[str, Any]:
        """Handle successful payment."""
        return {
            "status": "processed",
            "action": "payment_succeeded",
            "invoice_id": invoice.id,
            "amount": invoice.amount_paid,
        }

    async def _handle_payment_failed(
        self, invoice: stripe.Invoice
    ) -> dict[str, Any]:
        """Handle failed payment."""
        return {
            "status": "processed",
            "action": "payment_failed",
            "invoice_id": invoice.id,
            "customer_id": invoice.customer,
        }


# Singleton instance
_billing_service: BillingService | None = None


def get_billing_service() -> BillingService:
    """Get singleton billing service instance."""
    global _billing_service
    if _billing_service is None:
        _billing_service = BillingService()
    return _billing_service
