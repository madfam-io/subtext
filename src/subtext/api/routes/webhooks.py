"""
Webhook Routes

Handle incoming webhooks from Stripe and other services.
"""

from fastapi import APIRouter, Header, HTTPException, Request
import structlog

from subtext.integrations.stripe import get_billing_service

router = APIRouter()
logger = structlog.get_logger()


@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
) -> dict:
    """Handle Stripe webhook events."""
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")

    payload = await request.body()

    billing = get_billing_service()

    try:
        result = await billing.handle_webhook_event(payload, stripe_signature)
        logger.info("Stripe webhook processed", result=result)
        return result

    except ValueError as e:
        logger.warning("Invalid webhook signature", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid signature")

    except Exception as e:
        logger.error("Webhook processing failed", error=str(e))
        raise HTTPException(status_code=500, detail="Webhook processing failed")
