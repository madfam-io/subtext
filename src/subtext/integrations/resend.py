"""
Resend Email Integration

Handles transactional emails for Subtext including:
- Welcome emails
- Session analysis completion notifications
- Usage alerts
- Billing notifications
"""

from datetime import datetime
from typing import Any

import resend
import structlog
from pydantic import BaseModel, EmailStr

from subtext.config import settings

logger = structlog.get_logger()

# Initialize Resend
resend.api_key = settings.resend_api_key


# ══════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════


class EmailRecipient(BaseModel):
    """Email recipient."""

    email: EmailStr
    name: str | None = None


class EmailAttachment(BaseModel):
    """Email attachment."""

    filename: str
    content: bytes
    content_type: str = "application/octet-stream"


class EmailResult(BaseModel):
    """Result of sending an email."""

    id: str
    success: bool
    error: str | None = None


# ══════════════════════════════════════════════════════════════
# Email Templates
# ══════════════════════════════════════════════════════════════


class EmailTemplates:
    """HTML email templates for Subtext."""

    BASE_STYLE = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 12px 12px 0 0; }
        .header h1 { color: white; margin: 0; font-size: 24px; }
        .content { background: #ffffff; padding: 30px; border: 1px solid #e5e7eb; }
        .footer { background: #f9fafb; padding: 20px; border-radius: 0 0 12px 12px; text-align: center; color: #6b7280; font-size: 12px; }
        .button { display: inline-block; background: #667eea; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; }
        .stat { background: #f3f4f6; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .stat-value { font-size: 24px; font-weight: bold; color: #111827; }
        .stat-label { color: #6b7280; font-size: 14px; }
        .signal-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
        .signal-high { background: #fee2e2; color: #991b1b; }
        .signal-medium { background: #fef3c7; color: #92400e; }
        .signal-low { background: #d1fae5; color: #065f46; }
    </style>
    """

    @classmethod
    def welcome(cls, user_name: str) -> str:
        """Welcome email for new users."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>{cls.BASE_STYLE}</head>
        <body>
        <div class="container">
            <div class="header">
                <h1>Welcome to Subtext</h1>
            </div>
            <div class="content">
                <p>Hi {user_name or 'there'},</p>
                <p>Welcome to Subtext - where we help you read the room, not just the transcript.</p>
                <p>With Subtext, you can:</p>
                <ul>
                    <li><strong>Analyze conversations</strong> - Upload recordings to detect emotional signals</li>
                    <li><strong>Understand dynamics</strong> - See who dominates, who hesitates, who's stressed</li>
                    <li><strong>Get real-time insights</strong> - Stream live audio for instant feedback</li>
                </ul>
                <p style="margin-top: 30px;">
                    <a href="https://subtext.live/dashboard" class="button">Go to Dashboard</a>
                </p>
                <p style="margin-top: 30px; color: #6b7280;">
                    Need help? Check out our <a href="https://docs.subtext.live">documentation</a> or reply to this email.
                </p>
            </div>
            <div class="footer">
                <p>Subtext - Conversational Intelligence Infrastructure</p>
                <p>MADFAM SAS de CV</p>
            </div>
        </div>
        </body>
        </html>
        """

    @classmethod
    def analysis_complete(
        cls,
        user_name: str,
        session_name: str,
        session_id: str,
        duration_minutes: float,
        speaker_count: int,
        signal_count: int,
        top_signals: list[dict[str, Any]],
    ) -> str:
        """Analysis completion notification."""
        signal_badges = ""
        for signal in top_signals[:5]:
            intensity = signal.get("intensity", 0.5)
            css_class = "signal-high" if intensity > 0.7 else "signal-medium" if intensity > 0.4 else "signal-low"
            signal_badges += f'<span class="signal-badge {css_class}">{signal["type"].replace("_", " ").title()}</span>'

        return f"""
        <!DOCTYPE html>
        <html>
        <head>{cls.BASE_STYLE}</head>
        <body>
        <div class="container">
            <div class="header">
                <h1>Analysis Complete</h1>
            </div>
            <div class="content">
                <p>Hi {user_name or 'there'},</p>
                <p>Your session <strong>"{session_name}"</strong> has been analyzed.</p>

                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;">
                    <div class="stat">
                        <div class="stat-value">{duration_minutes:.1f}</div>
                        <div class="stat-label">Minutes</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{speaker_count}</div>
                        <div class="stat-label">Speakers</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{signal_count}</div>
                        <div class="stat-label">Signals</div>
                    </div>
                </div>

                <p><strong>Top Signals Detected:</strong></p>
                <p>{signal_badges or '<em>No significant signals detected</em>'}</p>

                <p style="margin-top: 30px;">
                    <a href="https://subtext.live/sessions/{session_id}" class="button">View Full Analysis</a>
                </p>
            </div>
            <div class="footer">
                <p>Subtext - Conversational Intelligence Infrastructure</p>
            </div>
        </div>
        </body>
        </html>
        """

    @classmethod
    def usage_alert(
        cls,
        user_name: str,
        org_name: str,
        usage_percent: float,
        minutes_used: int,
        minutes_limit: int,
    ) -> str:
        """Usage limit warning email."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>{cls.BASE_STYLE}</head>
        <body>
        <div class="container">
            <div class="header" style="background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);">
                <h1>Usage Alert</h1>
            </div>
            <div class="content">
                <p>Hi {user_name or 'there'},</p>
                <p>Your organization <strong>{org_name}</strong> has used <strong>{usage_percent:.0f}%</strong> of its monthly audio processing quota.</p>

                <div class="stat">
                    <div class="stat-value">{minutes_used} / {minutes_limit}</div>
                    <div class="stat-label">Minutes Used This Month</div>
                </div>

                <p>To continue analyzing conversations without interruption, consider upgrading your plan.</p>

                <p style="margin-top: 30px;">
                    <a href="https://subtext.live/settings/billing" class="button">Upgrade Plan</a>
                </p>
            </div>
            <div class="footer">
                <p>Subtext - Conversational Intelligence Infrastructure</p>
            </div>
        </div>
        </body>
        </html>
        """

    @classmethod
    def payment_failed(
        cls,
        user_name: str,
        invoice_amount: float,
        currency: str,
        retry_url: str,
    ) -> str:
        """Payment failure notification."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>{cls.BASE_STYLE}</head>
        <body>
        <div class="container">
            <div class="header" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                <h1>Payment Failed</h1>
            </div>
            <div class="content">
                <p>Hi {user_name or 'there'},</p>
                <p>We were unable to process your payment of <strong>{currency.upper()} {invoice_amount/100:.2f}</strong>.</p>
                <p>Please update your payment method to continue using Subtext without interruption.</p>

                <p style="margin-top: 30px;">
                    <a href="{retry_url}" class="button">Update Payment Method</a>
                </p>

                <p style="margin-top: 20px; color: #6b7280;">
                    If you believe this is an error or need assistance, please contact us at support@subtext.live.
                </p>
            </div>
            <div class="footer">
                <p>Subtext - Conversational Intelligence Infrastructure</p>
            </div>
        </div>
        </body>
        </html>
        """

    @classmethod
    def subscription_canceled(cls, user_name: str, end_date: datetime) -> str:
        """Subscription cancellation confirmation."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>{cls.BASE_STYLE}</head>
        <body>
        <div class="container">
            <div class="header">
                <h1>Subscription Canceled</h1>
            </div>
            <div class="content">
                <p>Hi {user_name or 'there'},</p>
                <p>Your Subtext subscription has been canceled.</p>
                <p>You'll continue to have access to your current plan until <strong>{end_date.strftime('%B %d, %Y')}</strong>.</p>
                <p>After that date, your account will revert to the free tier with limited features.</p>

                <p style="margin-top: 20px;">We're sorry to see you go. If you change your mind, you can resubscribe anytime.</p>

                <p style="margin-top: 30px;">
                    <a href="https://subtext.live/settings/billing" class="button">Resubscribe</a>
                </p>
            </div>
            <div class="footer">
                <p>Subtext - Conversational Intelligence Infrastructure</p>
            </div>
        </div>
        </body>
        </html>
        """


# ══════════════════════════════════════════════════════════════
# Resend Client
# ══════════════════════════════════════════════════════════════


class ResendClient:
    """Low-level Resend API client."""

    @staticmethod
    async def send(
        to: str | list[str],
        subject: str,
        html: str,
        text: str | None = None,
        from_email: str | None = None,
        reply_to: str | None = None,
        attachments: list[EmailAttachment] | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> EmailResult:
        """Send an email via Resend."""
        try:
            params: dict[str, Any] = {
                "from": from_email or settings.resend_from_email,
                "to": [to] if isinstance(to, str) else to,
                "subject": subject,
                "html": html,
            }

            if text:
                params["text"] = text
            if reply_to:
                params["reply_to"] = reply_to
            if attachments:
                params["attachments"] = [
                    {
                        "filename": a.filename,
                        "content": a.content,
                        "type": a.content_type,
                    }
                    for a in attachments
                ]
            if tags:
                params["tags"] = tags

            response = resend.Emails.send(params)

            logger.info(
                "Email sent",
                email_id=response.get("id"),
                to=to,
                subject=subject,
            )

            return EmailResult(
                id=response.get("id", ""),
                success=True,
            )

        except Exception as e:
            logger.error("Failed to send email", to=to, error=str(e))
            return EmailResult(
                id="",
                success=False,
                error=str(e),
            )


# ══════════════════════════════════════════════════════════════
# Email Service
# ══════════════════════════════════════════════════════════════


class EmailService:
    """
    High-level email service for Subtext.

    Provides typed methods for common email scenarios.
    """

    def __init__(self):
        self.client = ResendClient()
        self.templates = EmailTemplates()

    async def send_welcome(
        self,
        email: str,
        name: str | None = None,
    ) -> EmailResult:
        """Send welcome email to new user."""
        return await self.client.send(
            to=email,
            subject="Welcome to Subtext",
            html=self.templates.welcome(name or "there"),
            tags=[{"name": "type", "value": "welcome"}],
        )

    async def send_analysis_complete(
        self,
        email: str,
        name: str | None,
        session_name: str,
        session_id: str,
        duration_minutes: float,
        speaker_count: int,
        signal_count: int,
        top_signals: list[dict[str, Any]],
    ) -> EmailResult:
        """Send analysis completion notification."""
        return await self.client.send(
            to=email,
            subject=f'Analysis Complete: "{session_name}"',
            html=self.templates.analysis_complete(
                user_name=name or "there",
                session_name=session_name,
                session_id=session_id,
                duration_minutes=duration_minutes,
                speaker_count=speaker_count,
                signal_count=signal_count,
                top_signals=top_signals,
            ),
            tags=[
                {"name": "type", "value": "analysis_complete"},
                {"name": "session_id", "value": session_id},
            ],
        )

    async def send_usage_alert(
        self,
        email: str,
        name: str | None,
        org_name: str,
        usage_percent: float,
        minutes_used: int,
        minutes_limit: int,
    ) -> EmailResult:
        """Send usage limit warning."""
        return await self.client.send(
            to=email,
            subject=f"Usage Alert: {usage_percent:.0f}% of monthly quota used",
            html=self.templates.usage_alert(
                user_name=name or "there",
                org_name=org_name,
                usage_percent=usage_percent,
                minutes_used=minutes_used,
                minutes_limit=minutes_limit,
            ),
            tags=[{"name": "type", "value": "usage_alert"}],
        )

    async def send_payment_failed(
        self,
        email: str,
        name: str | None,
        invoice_amount: float,
        currency: str,
        retry_url: str,
    ) -> EmailResult:
        """Send payment failure notification."""
        return await self.client.send(
            to=email,
            subject="Action Required: Payment Failed",
            html=self.templates.payment_failed(
                user_name=name or "there",
                invoice_amount=invoice_amount,
                currency=currency,
                retry_url=retry_url,
            ),
            tags=[{"name": "type", "value": "payment_failed"}],
        )

    async def send_subscription_canceled(
        self,
        email: str,
        name: str | None,
        end_date: datetime,
    ) -> EmailResult:
        """Send subscription cancellation confirmation."""
        return await self.client.send(
            to=email,
            subject="Your Subtext Subscription Has Been Canceled",
            html=self.templates.subscription_canceled(
                user_name=name or "there",
                end_date=end_date,
            ),
            tags=[{"name": "type", "value": "subscription_canceled"}],
        )


# Singleton instance
_email_service: EmailService | None = None


def get_email_service() -> EmailService:
    """Get singleton email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
