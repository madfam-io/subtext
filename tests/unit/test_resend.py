"""
Unit Tests for Resend Email Integration

Tests the Resend email service models and templates.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from subtext.integrations.resend import (
    EmailRecipient,
    EmailAttachment,
    EmailResult,
    EmailTemplates,
    ResendClient,
    EmailService,
    get_email_service,
)


# ============================================================
# Model Tests
# ============================================================


class TestEmailRecipient:
    """Test EmailRecipient model."""

    def test_recipient_with_email_only(self):
        """Test recipient with just email."""
        recipient = EmailRecipient(email="test@example.com")

        assert recipient.email == "test@example.com"
        assert recipient.name is None

    def test_recipient_with_name(self):
        """Test recipient with email and name."""
        recipient = EmailRecipient(
            email="test@example.com",
            name="Test User",
        )

        assert recipient.email == "test@example.com"
        assert recipient.name == "Test User"


class TestEmailAttachment:
    """Test EmailAttachment model."""

    def test_attachment_minimal(self):
        """Test attachment with minimal fields."""
        attachment = EmailAttachment(
            filename="report.pdf",
            content=b"PDF content",
        )

        assert attachment.filename == "report.pdf"
        assert attachment.content == b"PDF content"
        assert attachment.content_type == "application/octet-stream"

    def test_attachment_with_content_type(self):
        """Test attachment with custom content type."""
        attachment = EmailAttachment(
            filename="data.json",
            content=b'{"key": "value"}',
            content_type="application/json",
        )

        assert attachment.content_type == "application/json"


class TestEmailResult:
    """Test EmailResult model."""

    def test_result_success(self):
        """Test successful email result."""
        result = EmailResult(
            id="email_123",
            success=True,
        )

        assert result.id == "email_123"
        assert result.success is True
        assert result.error is None

    def test_result_failure(self):
        """Test failed email result."""
        result = EmailResult(
            id="",
            success=False,
            error="API error",
        )

        assert result.success is False
        assert result.error == "API error"


# ============================================================
# Email Templates Tests
# ============================================================


class TestEmailTemplates:
    """Test EmailTemplates class."""

    def test_base_style_exists(self):
        """Test BASE_STYLE is defined."""
        assert EmailTemplates.BASE_STYLE is not None
        assert "font-family" in EmailTemplates.BASE_STYLE
        assert ".container" in EmailTemplates.BASE_STYLE

    def test_welcome_template(self):
        """Test welcome email template."""
        html = EmailTemplates.welcome("Test User")

        assert "Test User" in html
        assert "Welcome to Subtext" in html
        assert "subtext.live/dashboard" in html
        assert "<!DOCTYPE html>" in html

    def test_welcome_template_no_name(self):
        """Test welcome template with empty name."""
        html = EmailTemplates.welcome("")

        assert "there" in html  # Falls back to "there"

    def test_analysis_complete_template(self):
        """Test analysis complete email template."""
        signals = [
            {"type": "hesitation", "intensity": 0.8},
            {"type": "enthusiasm_surge", "intensity": 0.6},
        ]

        html = EmailTemplates.analysis_complete(
            user_name="Test User",
            session_name="Team Meeting",
            session_id="session-123",
            duration_minutes=45.5,
            speaker_count=4,
            signal_count=12,
            top_signals=signals,
        )

        assert "Test User" in html
        assert "Team Meeting" in html
        assert "45.5" in html
        assert "4" in html
        assert "12" in html
        assert "Analysis Complete" in html
        assert "session-123" in html
        assert "Hesitation" in html
        assert "Enthusiasm Surge" in html

    def test_analysis_complete_no_signals(self):
        """Test analysis complete with no signals."""
        html = EmailTemplates.analysis_complete(
            user_name="User",
            session_name="Meeting",
            session_id="123",
            duration_minutes=10,
            speaker_count=2,
            signal_count=0,
            top_signals=[],
        )

        assert "No significant signals detected" in html

    def test_usage_alert_template(self):
        """Test usage alert email template."""
        html = EmailTemplates.usage_alert(
            user_name="Test User",
            org_name="Acme Corp",
            usage_percent=85.0,
            minutes_used=850,
            minutes_limit=1000,
        )

        assert "Test User" in html
        assert "Acme Corp" in html
        assert "85%" in html  # Formatted as integer
        assert "850" in html
        assert "1000" in html
        assert "Usage Alert" in html
        assert "Upgrade Plan" in html

    def test_payment_failed_template(self):
        """Test payment failed email template."""
        html = EmailTemplates.payment_failed(
            user_name="Test User",
            invoice_amount=4999,  # In cents
            currency="usd",
            retry_url="https://pay.stripe.com/retry",
        )

        assert "Test User" in html
        assert "USD 49.99" in html
        assert "Payment Failed" in html
        assert "https://pay.stripe.com/retry" in html
        assert "Update Payment Method" in html

    def test_subscription_canceled_template(self):
        """Test subscription canceled email template."""
        end_date = datetime(2024, 12, 31)

        html = EmailTemplates.subscription_canceled(
            user_name="Test User",
            end_date=end_date,
        )

        assert "Test User" in html
        assert "Subscription Canceled" in html
        assert "December 31, 2024" in html
        assert "Resubscribe" in html


# ============================================================
# ResendClient Tests
# ============================================================


class TestResendClient:
    """Test ResendClient class."""

    @pytest.mark.asyncio
    async def test_send_success(self):
        """Test successful email sending."""
        mock_response = {"id": "email_123"}

        with patch("resend.Emails.send", return_value=mock_response):
            result = await ResendClient.send(
                to="test@example.com",
                subject="Test Subject",
                html="<p>Test content</p>",
            )

        assert result.success is True
        assert result.id == "email_123"

    @pytest.mark.asyncio
    async def test_send_with_list_recipients(self):
        """Test sending to multiple recipients."""
        mock_response = {"id": "email_456"}

        with patch("resend.Emails.send", return_value=mock_response) as mock_send:
            result = await ResendClient.send(
                to=["user1@example.com", "user2@example.com"],
                subject="Test Subject",
                html="<p>Test content</p>",
            )

        assert result.success is True
        # Verify list was passed correctly
        call_args = mock_send.call_args[0][0]
        assert call_args["to"] == ["user1@example.com", "user2@example.com"]

    @pytest.mark.asyncio
    async def test_send_with_options(self):
        """Test sending with all options."""
        mock_response = {"id": "email_789"}

        with patch("resend.Emails.send", return_value=mock_response) as mock_send:
            result = await ResendClient.send(
                to="test@example.com",
                subject="Test Subject",
                html="<p>Test content</p>",
                text="Test content",
                reply_to="reply@example.com",
                tags=[{"name": "type", "value": "test"}],
            )

        assert result.success is True
        call_args = mock_send.call_args[0][0]
        assert call_args["text"] == "Test content"
        assert call_args["reply_to"] == "reply@example.com"
        assert call_args["tags"] == [{"name": "type", "value": "test"}]

    @pytest.mark.asyncio
    async def test_send_with_attachments(self):
        """Test sending with attachments."""
        mock_response = {"id": "email_attach"}
        attachments = [
            EmailAttachment(
                filename="report.pdf",
                content=b"PDF data",
                content_type="application/pdf",
            )
        ]

        with patch("resend.Emails.send", return_value=mock_response) as mock_send:
            result = await ResendClient.send(
                to="test@example.com",
                subject="With Attachment",
                html="<p>See attached</p>",
                attachments=attachments,
            )

        assert result.success is True
        call_args = mock_send.call_args[0][0]
        assert "attachments" in call_args
        assert call_args["attachments"][0]["filename"] == "report.pdf"

    @pytest.mark.asyncio
    async def test_send_failure(self):
        """Test email sending failure."""
        with patch("resend.Emails.send", side_effect=Exception("API Error")):
            result = await ResendClient.send(
                to="test@example.com",
                subject="Test Subject",
                html="<p>Test content</p>",
            )

        assert result.success is False
        assert "API Error" in result.error


# ============================================================
# EmailService Tests
# ============================================================


class TestEmailService:
    """Test EmailService class."""

    def test_service_init(self):
        """Test EmailService initialization."""
        service = EmailService()

        assert service.client is not None
        assert service.templates is not None

    @pytest.mark.asyncio
    async def test_send_welcome(self):
        """Test sending welcome email."""
        service = EmailService()

        mock_result = EmailResult(id="email_123", success=True)

        with patch.object(service.client, "send", return_value=mock_result) as mock_send:
            result = await service.send_welcome(
                email="user@example.com",
                name="Test User",
            )

        assert result.success is True
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["to"] == "user@example.com"
        assert call_kwargs["subject"] == "Welcome to Subtext"

    @pytest.mark.asyncio
    async def test_send_analysis_complete(self):
        """Test sending analysis complete email."""
        service = EmailService()

        mock_result = EmailResult(id="email_456", success=True)

        with patch.object(service.client, "send", return_value=mock_result) as mock_send:
            result = await service.send_analysis_complete(
                email="user@example.com",
                name="Test User",
                session_name="Team Standup",
                session_id="session-123",
                duration_minutes=15.0,
                speaker_count=3,
                signal_count=5,
                top_signals=[{"type": "hesitation", "intensity": 0.7}],
            )

        assert result.success is True
        call_kwargs = mock_send.call_args[1]
        assert 'Team Standup' in call_kwargs["subject"]

    @pytest.mark.asyncio
    async def test_send_usage_alert(self):
        """Test sending usage alert email."""
        service = EmailService()

        mock_result = EmailResult(id="email_789", success=True)

        with patch.object(service.client, "send", return_value=mock_result) as mock_send:
            result = await service.send_usage_alert(
                email="user@example.com",
                name="Test User",
                org_name="Acme Corp",
                usage_percent=90.0,
                minutes_used=900,
                minutes_limit=1000,
            )

        assert result.success is True
        call_kwargs = mock_send.call_args[1]
        assert "90%" in call_kwargs["subject"]

    @pytest.mark.asyncio
    async def test_send_payment_failed(self):
        """Test sending payment failed email."""
        service = EmailService()

        mock_result = EmailResult(id="email_pay", success=True)

        with patch.object(service.client, "send", return_value=mock_result) as mock_send:
            result = await service.send_payment_failed(
                email="user@example.com",
                name="Test User",
                invoice_amount=5000,
                currency="usd",
                retry_url="https://pay.example.com",
            )

        assert result.success is True
        call_kwargs = mock_send.call_args[1]
        assert "Payment Failed" in call_kwargs["subject"]

    @pytest.mark.asyncio
    async def test_send_subscription_canceled(self):
        """Test sending subscription canceled email."""
        service = EmailService()

        mock_result = EmailResult(id="email_cancel", success=True)

        with patch.object(service.client, "send", return_value=mock_result) as mock_send:
            result = await service.send_subscription_canceled(
                email="user@example.com",
                name="Test User",
                end_date=datetime(2024, 12, 31),
            )

        assert result.success is True
        call_kwargs = mock_send.call_args[1]
        assert "Canceled" in call_kwargs["subject"]


# ============================================================
# Singleton Tests
# ============================================================


class TestEmailServiceSingleton:
    """Test email service singleton."""

    def test_get_email_service(self):
        """Test get_email_service returns service."""
        import subtext.integrations.resend as resend_module

        # Reset singleton
        original = resend_module._email_service
        resend_module._email_service = None

        try:
            service = get_email_service()
            assert isinstance(service, EmailService)
        finally:
            resend_module._email_service = original

    def test_get_email_service_returns_same_instance(self):
        """Test get_email_service returns same instance."""
        import subtext.integrations.resend as resend_module

        # Reset singleton
        original = resend_module._email_service
        resend_module._email_service = None

        try:
            service1 = get_email_service()
            service2 = get_email_service()

            assert service1 is service2
        finally:
            resend_module._email_service = original
