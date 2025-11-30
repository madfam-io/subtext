"""
Unit Tests for Core Models

Tests Pydantic model validation and serialization.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from subtext.core.models import (
    # Enums
    SignalType,
    SessionStatus,
    SubscriptionTier,
    UserRole,
    # Models
    Organization,
    User,
    Session,
    Speaker,
    Signal,
    TranscriptSegment,
    ProsodicsFeatures,
    TimelinePoint,
    SessionInsight,
    KeyMoment,
    RiskFlag,
    ESPMessage,
    # API models
    SessionCreate,
    SessionResponse,
    AnalysisResult,
)


# ══════════════════════════════════════════════════════════════
# Enum Tests
# ══════════════════════════════════════════════════════════════


class TestEnums:
    """Test enum definitions."""

    def test_signal_types(self):
        """Test all signal types are defined."""
        expected_signals = [
            "truth_gap", "steamroll", "dead_air",
            "micro_tremor", "monotone", "uptick",
            "echo_chamber", "coffee_shop",
            "stress_spike", "disengagement", "deception_marker",
            "enthusiasm_surge", "agreement_signal", "disagreement_signal",
        ]

        for signal in expected_signals:
            assert SignalType(signal) is not None

    def test_session_status_values(self):
        """Test session status enum values."""
        assert SessionStatus.PENDING.value == "pending"
        assert SessionStatus.PROCESSING.value == "processing"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.FAILED.value == "failed"

    def test_subscription_tiers(self):
        """Test subscription tier enum values."""
        tiers = [SubscriptionTier.FREE, SubscriptionTier.PERSONAL,
                 SubscriptionTier.TEAMS, SubscriptionTier.ENTERPRISE,
                 SubscriptionTier.API]
        assert len(tiers) == 5

    def test_user_roles(self):
        """Test user role enum values."""
        roles = [UserRole.OWNER, UserRole.ADMIN,
                 UserRole.MEMBER, UserRole.VIEWER]
        assert len(roles) == 4


# ══════════════════════════════════════════════════════════════
# Organization Model Tests
# ══════════════════════════════════════════════════════════════


class TestOrganizationModel:
    """Test Organization model."""

    def test_organization_creation(self):
        """Test basic organization creation."""
        org = Organization(
            id=uuid4(),
            name="Test Company",
            slug="test-company",
        )

        assert org.name == "Test Company"
        assert org.slug == "test-company"
        assert org.subscription_tier == SubscriptionTier.FREE
        assert org.monthly_minutes_limit == 300

    def test_organization_with_subscription(self):
        """Test organization with enterprise subscription."""
        org = Organization(
            id=uuid4(),
            name="Enterprise Corp",
            slug="enterprise",
            subscription_tier=SubscriptionTier.ENTERPRISE,
            monthly_minutes_limit=10000,
            stripe_customer_id="cus_123",
        )

        assert org.subscription_tier == SubscriptionTier.ENTERPRISE
        assert org.monthly_minutes_limit == 10000
        assert org.stripe_customer_id == "cus_123"


# ══════════════════════════════════════════════════════════════
# User Model Tests
# ══════════════════════════════════════════════════════════════


class TestUserModel:
    """Test User model."""

    def test_user_creation(self):
        """Test basic user creation."""
        user = User(
            id=uuid4(),
            org_id=uuid4(),
            email="test@example.com",
            name="Test User",
        )

        assert user.email == "test@example.com"
        assert user.role == UserRole.MEMBER
        assert user.voice_fingerprint_enabled is False

    def test_user_with_admin_role(self):
        """Test user with admin role."""
        user = User(
            id=uuid4(),
            org_id=uuid4(),
            email="admin@example.com",
            role=UserRole.ADMIN,
        )

        assert user.role == UserRole.ADMIN


# ══════════════════════════════════════════════════════════════
# Session Model Tests
# ══════════════════════════════════════════════════════════════


class TestSessionModel:
    """Test Session model."""

    def test_session_creation(self):
        """Test basic session creation."""
        session = Session(
            id=uuid4(),
            org_id=uuid4(),
            created_by=uuid4(),
            name="Test Meeting",
        )

        assert session.name == "Test Meeting"
        assert session.status == SessionStatus.PENDING
        assert session.source_type == "upload"

    def test_session_with_results(self):
        """Test session with processing results."""
        session = Session(
            id=uuid4(),
            org_id=uuid4(),
            created_by=uuid4(),
            name="Completed Meeting",
            status=SessionStatus.COMPLETED,
            duration_ms=3600000,
            speaker_count=3,
            signal_count=15,
            processing_time_ms=45000,
            language="en",
        )

        assert session.status == SessionStatus.COMPLETED
        assert session.duration_ms == 3600000
        assert session.speaker_count == 3


# ══════════════════════════════════════════════════════════════
# Signal Model Tests
# ══════════════════════════════════════════════════════════════


class TestSignalModel:
    """Test Signal model."""

    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = Signal(
            id=uuid4(),
            session_id=uuid4(),
            signal_type=SignalType.TRUTH_GAP,
            timestamp_ms=5000,
            confidence=0.85,
            intensity=0.7,
        )

        assert signal.signal_type == SignalType.TRUTH_GAP
        assert signal.confidence == 0.85
        assert signal.intensity == 0.7

    def test_signal_with_metrics(self):
        """Test signal with raw metrics."""
        signal = Signal(
            id=uuid4(),
            session_id=uuid4(),
            signal_type=SignalType.MICRO_TREMOR,
            timestamp_ms=10000,
            confidence=0.9,
            intensity=0.8,
            metrics={
                "jitter": 0.05,
                "shimmer": 0.08,
                "hnr": 12.5,
            },
        )

        assert "jitter" in signal.metrics
        assert signal.metrics["jitter"] == 0.05

    def test_signal_confidence_bounds(self):
        """Test signal confidence value bounds."""
        # Valid confidence
        signal = Signal(
            id=uuid4(),
            session_id=uuid4(),
            signal_type=SignalType.STRESS_SPIKE,
            timestamp_ms=0,
            confidence=0.5,
            intensity=0.5,
        )
        assert 0 <= signal.confidence <= 1

        # Invalid confidence should raise
        with pytest.raises(ValueError):
            Signal(
                id=uuid4(),
                session_id=uuid4(),
                signal_type=SignalType.STRESS_SPIKE,
                timestamp_ms=0,
                confidence=1.5,  # Invalid
                intensity=0.5,
            )


# ══════════════════════════════════════════════════════════════
# Prosodics Model Tests
# ══════════════════════════════════════════════════════════════


class TestProsodicsModel:
    """Test ProsodicsFeatures model."""

    def test_prosodics_creation(self):
        """Test prosodics feature creation."""
        prosodics = ProsodicsFeatures(
            session_id=uuid4(),
            timestamp_ms=5000,
            pitch_mean=150.0,
            pitch_std=20.0,
            energy_mean=0.5,
            valence=0.3,
            arousal=0.7,
            dominance=0.5,
        )

        assert prosodics.pitch_mean == 150.0
        assert prosodics.valence == 0.3

    def test_prosodics_vad_bounds(self):
        """Test VAD value bounds."""
        # Valid VAD values
        prosodics = ProsodicsFeatures(
            session_id=uuid4(),
            timestamp_ms=0,
            valence=-0.5,
            arousal=0.8,
            dominance=0.3,
        )

        assert -1.0 <= prosodics.valence <= 1.0
        assert 0.0 <= prosodics.arousal <= 1.0
        assert 0.0 <= prosodics.dominance <= 1.0


# ══════════════════════════════════════════════════════════════
# ESP Message Tests
# ══════════════════════════════════════════════════════════════


class TestESPMessage:
    """Test ESP (Emotional State Protocol) message model."""

    def test_esp_message_creation(self):
        """Test ESP message creation."""
        message = ESPMessage(
            valence=0.5,
            arousal=0.7,
            dominance=0.6,
            engagement_score=0.8,
            stress_index=0.3,
        )

        assert message.version == "1.0"
        assert message.valence == 0.5
        assert message.consent_level == "self"

    def test_esp_message_with_signals(self):
        """Test ESP message with active signals."""
        message = ESPMessage(
            valence=0.0,
            arousal=0.9,
            dominance=0.4,
            engagement_score=0.6,
            stress_index=0.8,
            signals=[
                {"type": "stress_spike", "confidence": 0.85},
                {"type": "micro_tremor", "confidence": 0.7},
            ],
        )

        assert len(message.signals) == 2
        assert message.signals[0]["type"] == "stress_spike"

    def test_esp_message_signal_validation(self):
        """Test ESP message signal validation."""
        # Invalid signal (missing required fields)
        with pytest.raises(ValueError):
            ESPMessage(
                valence=0.0,
                arousal=0.5,
                dominance=0.5,
                engagement_score=0.5,
                stress_index=0.5,
                signals=[
                    {"type": "stress_spike"},  # Missing confidence
                ],
            )


# ══════════════════════════════════════════════════════════════
# API Model Tests
# ══════════════════════════════════════════════════════════════


class TestAPIModels:
    """Test API request/response models."""

    def test_session_create(self):
        """Test SessionCreate model."""
        create = SessionCreate(
            name="New Meeting",
            description="Weekly sync",
            language="en",
        )

        assert create.name == "New Meeting"
        assert create.language == "en"

    def test_session_create_validation(self):
        """Test SessionCreate validation."""
        # Name too short
        with pytest.raises(ValueError):
            SessionCreate(name="")

    def test_session_response(self):
        """Test SessionResponse model."""
        response = SessionResponse(
            id=uuid4(),
            name="Test Session",
            status=SessionStatus.COMPLETED,
            duration_ms=60000,
            speaker_count=2,
            signal_count=5,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )

        assert response.status == SessionStatus.COMPLETED
        assert response.speaker_count == 2
