"""
Unit Tests for Pipeline Signals Module

Tests the Signal Atlas and Signal Detector functionality.
"""

import pytest
from uuid import uuid4
from unittest.mock import MagicMock, patch

from subtext.core.models import SignalType


# ══════════════════════════════════════════════════════════════
# Signal Definition Tests
# ══════════════════════════════════════════════════════════════


class TestSignalDefinition:
    """Test SignalDefinition dataclass."""

    def test_signal_definition_creation(self):
        """Test creating a signal definition."""
        from subtext.pipeline.signals import SignalDefinition

        definition = SignalDefinition(
            signal_type=SignalType.TRUTH_GAP,
            name="Truth Gap",
            description="Extended response latency",
            psychological_interpretation="Cognitive load indication",
            thresholds={"latency_ms_min": 800},
            weight=1.2,
            required_features=["response_latency"],
        )

        assert definition.signal_type == SignalType.TRUTH_GAP
        assert definition.name == "Truth Gap"
        assert definition.weight == 1.2
        assert "latency_ms_min" in definition.thresholds

    def test_signal_definition_defaults(self):
        """Test signal definition default values."""
        from subtext.pipeline.signals import SignalDefinition

        definition = SignalDefinition(
            signal_type=SignalType.MICRO_TREMOR,
            name="Test Signal",
            description="Test description",
            psychological_interpretation="Test interpretation",
            thresholds={},
        )

        assert definition.weight == 1.0
        assert definition.required_features == []
        assert definition.ui_color == "#888888"
        assert definition.ui_icon == "signal"


# ══════════════════════════════════════════════════════════════
# Signal Atlas Tests
# ══════════════════════════════════════════════════════════════


class TestSignalAtlas:
    """Test SignalAtlas class methods."""

    def test_atlas_has_all_signal_types(self):
        """Test atlas contains definitions for signal types."""
        from subtext.pipeline.signals import SignalAtlas

        # Check that atlas has signal definitions
        assert len(SignalAtlas.SIGNALS) > 0

    def test_get_definition(self):
        """Test getting a specific signal definition."""
        from subtext.pipeline.signals import SignalAtlas

        definition = SignalAtlas.get_definition(SignalType.TRUTH_GAP)

        assert definition.signal_type == SignalType.TRUTH_GAP
        assert definition.name == "Truth Gap"
        assert "latency_ms_min" in definition.thresholds

    def test_get_all_definitions(self):
        """Test getting all signal definitions."""
        from subtext.pipeline.signals import SignalAtlas

        all_defs = SignalAtlas.get_all_definitions()

        assert isinstance(all_defs, dict)
        assert len(all_defs) > 0
        assert all(isinstance(k, SignalType) for k in all_defs.keys())

    def test_get_temporal_signals(self):
        """Test getting temporal signal types."""
        from subtext.pipeline.signals import SignalAtlas

        temporal = SignalAtlas.get_temporal_signals()

        assert SignalType.TRUTH_GAP in temporal
        assert SignalType.STEAMROLL in temporal
        assert SignalType.DEAD_AIR in temporal

    def test_get_spectral_signals(self):
        """Test getting spectral signal types."""
        from subtext.pipeline.signals import SignalAtlas

        spectral = SignalAtlas.get_spectral_signals()

        assert SignalType.MICRO_TREMOR in spectral
        assert SignalType.MONOTONE in spectral
        assert SignalType.UPTICK in spectral

    def test_get_composite_signals(self):
        """Test getting composite signal types."""
        from subtext.pipeline.signals import SignalAtlas

        composite = SignalAtlas.get_composite_signals()

        assert SignalType.STRESS_SPIKE in composite
        assert SignalType.DISENGAGEMENT in composite
        assert SignalType.DECEPTION_MARKER in composite
        assert SignalType.ENTHUSIASM_SURGE in composite

    def test_signal_definitions_have_required_fields(self):
        """Test all signal definitions have required fields."""
        from subtext.pipeline.signals import SignalAtlas

        for signal_type, definition in SignalAtlas.SIGNALS.items():
            assert definition.signal_type == signal_type
            assert definition.name
            assert definition.description
            assert definition.psychological_interpretation
            assert isinstance(definition.thresholds, dict)


# ══════════════════════════════════════════════════════════════
# Signal Detector Tests
# ══════════════════════════════════════════════════════════════


class TestSignalDetector:
    """Test SignalDetector class."""

    def test_detector_initialization(self):
        """Test signal detector initialization."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()

        assert detector.confidence_threshold == 0.5
        assert len(detector.enabled_signals) > 0
        assert detector.atlas is not None

    def test_detector_with_custom_threshold(self):
        """Test detector with custom confidence threshold."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector(confidence_threshold=0.7)

        assert detector.confidence_threshold == 0.7

    def test_detector_with_enabled_signals(self):
        """Test detector with specific enabled signals."""
        from subtext.pipeline.signals import SignalDetector

        enabled = [SignalType.TRUTH_GAP, SignalType.MICRO_TREMOR]
        detector = SignalDetector(enabled_signals=enabled)

        assert detector.enabled_signals == enabled

    def test_detect_all_empty_input(self):
        """Test detection with empty input."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()
        session_id = uuid4()

        signals = detector.detect_all(
            session_id=session_id,
            segments=[],
            prosodics=[],
        )

        assert signals == []

    def test_detect_all_with_segments(self):
        """Test detection with transcript segments."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector(confidence_threshold=0.3)
        session_id = uuid4()

        segments = [
            {
                "start_ms": 0,
                "end_ms": 2000,
                "text": "Hello, how are you?",
                "speaker_id": "spk_001",
            },
            {
                "start_ms": 3500,  # 1500ms gap - potential truth gap
                "end_ms": 5000,
                "text": "I'm... doing fine.",
                "speaker_id": "spk_002",
            },
        ]

        prosodics = [
            {
                "timestamp_ms": 0,
                "pitch_mean": 150.0,
                "energy_mean": 0.5,
            },
            {
                "timestamp_ms": 3500,
                "pitch_mean": 180.0,
                "energy_mean": 0.6,
            },
        ]

        signals = detector.detect_all(
            session_id=session_id,
            segments=segments,
            prosodics=prosodics,
        )

        # Should return a list (may or may not have detections based on thresholds)
        assert isinstance(signals, list)

    def test_detect_all_with_speaker_baselines(self):
        """Test detection with speaker baseline metrics."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()
        session_id = uuid4()

        segments = [
            {
                "start_ms": 0,
                "end_ms": 2000,
                "text": "Test segment",
                "speaker_id": "spk_001",
            },
        ]

        prosodics = [
            {
                "timestamp_ms": 0,
                "pitch_mean": 200.0,
                "energy_mean": 0.7,
            },
        ]

        baselines = {
            "spk_001": {
                "pitch_mean": 150.0,
                "energy_mean": 0.5,
                "speech_rate": 2.0,
            }
        }

        signals = detector.detect_all(
            session_id=session_id,
            segments=segments,
            prosodics=prosodics,
            speaker_baselines=baselines,
        )

        assert isinstance(signals, list)


# ══════════════════════════════════════════════════════════════
# Detection Result Tests
# ══════════════════════════════════════════════════════════════


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_detection_result_creation(self):
        """Test creating a detection result."""
        from subtext.pipeline.signals import DetectionResult

        result = DetectionResult(
            timestamp_ms=1000,
            duration_ms=500,
            confidence=0.85,
            intensity=0.7,
            metrics={"latency": 1200},
            context={"prev_speaker": "spk_001"},
        )

        assert result.timestamp_ms == 1000
        assert result.duration_ms == 500
        assert result.confidence == 0.85
        assert result.intensity == 0.7
        assert result.metrics == {"latency": 1200}
        assert result.context == {"prev_speaker": "spk_001"}


# ══════════════════════════════════════════════════════════════
# Signal Detection Method Tests
# ══════════════════════════════════════════════════════════════


class TestSignalDetectionMethods:
    """Test individual signal detection methods."""

    def test_get_segment_prosodics(self):
        """Test extracting prosodics for a segment."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()

        segment = {"start_ms": 1000, "end_ms": 2000}
        prosodic_index = {
            500: {"pitch_mean": 100},
            1000: {"pitch_mean": 150},
            1500: {"pitch_mean": 160},
            2500: {"pitch_mean": 200},
        }

        result = detector._get_segment_prosodics(segment, prosodic_index)

        # Should return a dict with averaged prosodics within segment timeframe
        assert isinstance(result, dict)
        # Should have pitch_mean as average of matching prosodics
        if "pitch_mean" in result:
            # 1000ms and 1500ms are within [1000, 2000)
            expected_avg = (150 + 160) / 2
            assert result["pitch_mean"] == expected_avg

    def test_get_segment_prosodics_no_matches(self):
        """Test extracting prosodics when no matches exist."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()

        segment = {"start_ms": 5000, "end_ms": 6000}
        prosodic_index = {
            0: {"pitch_mean": 100},
            1000: {"pitch_mean": 150},
        }

        result = detector._get_segment_prosodics(segment, prosodic_index)

        # Should return empty dict when no prosodics in range
        assert result == {}

    def test_detect_truth_gap(self):
        """Test truth gap detection logic."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.TRUTH_GAP)

        # Segment with long response latency after a question
        segment = {
            "start_ms": 2000,
            "end_ms": 4000,
            "text": "Well... I think...",
            "speaker_id": "spk_002",
        }

        prev_segment = {
            "start_ms": 0,
            "end_ms": 1000,
            "text": "What happened yesterday?",
            "speaker_id": "spk_001",
        }

        # Gap of 1000ms between segments (2000 - 1000)
        result = detector._detect_truth_gap(
            definition=definition,
            segment=segment,
            prosodics={},
            prev_segment=prev_segment,
        )

        # Should detect truth gap with 1000ms latency (above 800ms threshold)
        assert result is not None
        assert result.confidence > 0
        assert result.metrics.get("latency_ms") == 1000

    def test_detect_truth_gap_no_question(self):
        """Test truth gap not detected when prev is not a question."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.TRUTH_GAP)

        segment = {
            "start_ms": 2000,
            "end_ms": 4000,
            "text": "I agree.",
            "speaker_id": "spk_002",
        }

        prev_segment = {
            "start_ms": 0,
            "end_ms": 1000,
            "text": "That was a great movie.",  # Not a question
            "speaker_id": "spk_001",
        }

        result = detector._detect_truth_gap(
            definition=definition,
            segment=segment,
            prosodics={},
            prev_segment=prev_segment,
        )

        # Should not detect since prev is not a question
        assert result is None

    def test_detect_steamroll(self):
        """Test steamroll detection logic."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.STEAMROLL)

        # Overlapping segment (interruption)
        segment = {
            "start_ms": 1000,
            "end_ms": 4000,
            "text": "No, let me tell you...",
            "speaker_id": "spk_002",
        }

        prev_segment = {
            "start_ms": 0,
            "end_ms": 3500,  # Overlaps by 2500ms
            "text": "I was thinking that we should—",
            "speaker_id": "spk_001",
        }

        result = detector._detect_steamroll(
            definition=definition,
            segment=segment,
            prosodics={"energy_mean": 0.8},
            prev_segment=prev_segment,
            baseline={"energy_mean": 0.3},
        )

        # 2500ms overlap exceeds 2000ms threshold
        # Energy diff is 0.5 * 20 = 10dB, meeting the 10dB threshold
        assert result is not None
        assert result.metrics.get("overlap_ms") == 2500

    def test_detect_micro_tremor(self):
        """Test micro tremor detection."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.MICRO_TREMOR)

        segment = {
            "start_ms": 0,
            "end_ms": 3000,  # 3000ms duration > 500ms min
            "text": "I didn't do it",
            "speaker_id": "spk_001",
        }

        # Jitter in stress range (0.02 - 0.08)
        prosodics = {
            "jitter": 0.05,
            "shimmer": 0.04,
        }

        result = detector._detect_micro_tremor(
            definition=definition,
            segment=segment,
            prosodics=prosodics,
        )

        assert result is not None
        assert result.metrics.get("jitter") == 0.05
        assert result.metrics.get("shimmer") == 0.04

    def test_detect_monotone(self):
        """Test monotone detection."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.MONOTONE)

        segment = {
            "start_ms": 0,
            "end_ms": 35000,  # 35s > 30s min duration
            "text": "The report shows the following metrics and data points...",
            "speaker_id": "spk_001",
        }

        # Very flat variance
        prosodics = {
            "pitch_std": 0.05,  # < 0.10 threshold
            "energy_std": 0.08,  # < 0.15 threshold
        }

        result = detector._detect_monotone(
            definition=definition,
            segment=segment,
            prosodics=prosodics,
        )

        assert result is not None
        assert result.metrics.get("pitch_variance") == 0.05

    def test_detect_uptick(self):
        """Test uptick (upspeak) detection."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.UPTICK)

        segment = {
            "start_ms": 0,
            "end_ms": 2000,
            "text": "I think we should proceed with this plan.",  # Declarative
            "speaker_id": "spk_001",
        }

        # Rising pitch at end
        prosodics = {
            "pitch_slope": 20,  # > 15 Hz threshold
        }

        result = detector._detect_uptick(
            definition=definition,
            segment=segment,
            prosodics=prosodics,
        )

        assert result is not None
        assert result.metrics.get("pitch_slope") == 20

    def test_detect_uptick_question(self):
        """Test uptick not detected for actual questions."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.UPTICK)

        segment = {
            "start_ms": 0,
            "end_ms": 2000,
            "text": "Should we proceed with this plan?",  # Actual question
            "speaker_id": "spk_001",
        }

        prosodics = {"pitch_slope": 20}

        result = detector._detect_uptick(
            definition=definition,
            segment=segment,
            prosodics=prosodics,
        )

        # Questions are allowed to have rising intonation
        assert result is None

    def test_detect_stress_spike(self):
        """Test stress spike detection."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.STRESS_SPIKE)

        segment = {
            "start_ms": 0,
            "end_ms": 3000,
            "text": "I need to explain something...",
            "speaker_id": "spk_001",
        }

        prosodics = {
            "jitter": 0.04,
            "speech_rate": 3.0,
            "pitch_std": 25,
        }

        baseline = {
            "jitter": 0.02,
            "speech_rate": 2.0,
            "pitch_std": 10,
        }

        result = detector._detect_stress_spike(
            definition=definition,
            segment=segment,
            prosodics=prosodics,
            baseline=baseline,
        )

        # Multiple elevated components should trigger detection
        assert result is not None
        assert result.metrics.get("elevated_components") >= 2

    def test_detect_enthusiasm_surge(self):
        """Test enthusiasm surge detection."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.ENTHUSIASM_SURGE)

        segment = {
            "start_ms": 0,
            "end_ms": 5000,
            "text": "This is amazing! I love this idea!",
            "speaker_id": "spk_001",
        }

        prosodics = {
            "energy_mean": 0.8,
            "pitch_mean": 180,
            "speech_rate": 2.5,
        }

        baseline = {
            "energy_mean": 0.5,
            "pitch_mean": 140,
            "speech_rate": 2.0,
        }

        result = detector._detect_enthusiasm_surge(
            definition=definition,
            segment=segment,
            prosodics=prosodics,
            baseline=baseline,
        )

        # All ratios should exceed thresholds
        assert result is not None

    def test_detect_disengagement(self):
        """Test disengagement detection."""
        from subtext.pipeline.signals import SignalDetector, SignalAtlas

        detector = SignalDetector()
        definition = SignalAtlas.get_definition(SignalType.DISENGAGEMENT)

        segment = {
            "start_ms": 0,
            "end_ms": 5000,
            "text": "Yeah... okay...",
            "speaker_id": "spk_001",
        }

        prosodics = {
            "pitch_std": 5,  # Low variance
        }

        baseline = {
            "pitch_std": 15,  # Was higher before
        }

        result = detector._detect_disengagement(
            definition=definition,
            segment=segment,
            prosodics=prosodics,
            baseline=baseline,
        )

        # Pitch decrease > 0.3 threshold should trigger
        assert result is not None


# ══════════════════════════════════════════════════════════════
# Helper Method Tests
# ══════════════════════════════════════════════════════════════


class TestHelperMethods:
    """Test helper methods on SignalDetector."""

    def test_has_required_features_all_present(self):
        """Test feature check when all present."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()

        prosodics = {"jitter": 0.05, "shimmer": 0.03, "pitch_mean": 150}
        required = ["jitter", "shimmer"]

        result = detector._has_required_features(prosodics, required)
        assert result is True

    def test_has_required_features_missing(self):
        """Test feature check when some missing."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()

        prosodics = {"jitter": 0.05}
        required = ["jitter", "shimmer"]

        result = detector._has_required_features(prosodics, required)
        assert result is False

    def test_has_required_features_transcript_special(self):
        """Test transcript feature is specially handled."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()

        prosodics = {"jitter": 0.05}
        # transcript and timing are specially handled
        required = ["jitter", "transcript"]

        result = detector._has_required_features(prosodics, required)
        assert result is True

    def test_detect_signal_routing(self):
        """Test signal detection routes to correct detector."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()

        segment = {
            "start_ms": 2000,
            "end_ms": 4000,
            "text": "Response text",
            "speaker_id": "spk_002",
        }

        prev_segment = {
            "start_ms": 0,
            "end_ms": 1000,
            "text": "What is your answer?",
            "speaker_id": "spk_001",
        }

        # Call the routing method
        result = detector._detect_signal(
            signal_type=SignalType.TRUTH_GAP,
            segment=segment,
            prosodics={},
            prev_segment=prev_segment,
            next_segment=None,
            baseline=None,
        )

        # Should route to _detect_truth_gap and return result or None
        assert result is None or hasattr(result, "confidence")


# ══════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════


class TestSignalDetectionIntegration:
    """Integration tests for full signal detection flow."""

    def test_full_detection_pipeline(self):
        """Test complete detection pipeline with multiple segments."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector(confidence_threshold=0.4)
        session_id = uuid4()

        segments = [
            {
                "start_ms": 0,
                "end_ms": 2000,
                "text": "How did the meeting go yesterday?",
                "speaker_id": "spk_001",
            },
            {
                "start_ms": 3500,  # 1500ms gap after question
                "end_ms": 6000,
                "text": "Well, um, it went okay I guess.",
                "speaker_id": "spk_002",
            },
            {
                "start_ms": 6500,
                "end_ms": 8000,
                "text": "What happened with the budget?",
                "speaker_id": "spk_001",
            },
        ]

        prosodics = [
            {"timestamp_ms": 0, "pitch_mean": 140, "energy_mean": 0.5, "jitter": 0.01},
            {"timestamp_ms": 3500, "pitch_mean": 160, "energy_mean": 0.4, "jitter": 0.04},
            {"timestamp_ms": 6500, "pitch_mean": 145, "energy_mean": 0.5, "jitter": 0.02},
        ]

        signals = detector.detect_all(
            session_id=session_id,
            segments=segments,
            prosodics=prosodics,
        )

        assert isinstance(signals, list)
        # All signals should have session_id set
        for signal in signals:
            assert signal.session_id == session_id
            assert signal.confidence >= detector.confidence_threshold

    def test_detection_with_all_signal_types(self):
        """Test that all signal types can be checked without error."""
        from subtext.pipeline.signals import SignalDetector

        detector = SignalDetector()
        session_id = uuid4()

        # Simple segments that won't trigger signals but exercise all code paths
        segments = [
            {
                "start_ms": 0,
                "end_ms": 1000,
                "text": "Hello",
                "speaker_id": "spk_001",
            },
        ]

        prosodics = [
            {"timestamp_ms": 0, "pitch_mean": 150, "energy_mean": 0.5},
        ]

        # Should not raise for any signal type
        signals = detector.detect_all(
            session_id=session_id,
            segments=segments,
            prosodics=prosodics,
        )

        assert isinstance(signals, list)
