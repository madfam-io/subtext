"""
Signal Atlas & Detection Engine

The Signal Atlas maps bio-acoustic markers to psychological signals.
This is the core IP of Subtext - the "Rosetta Stone" that translates
engineering metrics (Hz, dB, ms) into psychological states.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID, uuid4

import numpy as np
import structlog

from subtext.core.models import Signal, SignalType, ProsodicsFeatures

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# Signal Definitions
# ══════════════════════════════════════════════════════════════


@dataclass
class SignalDefinition:
    """
    Definition of a signal from the Signal Atlas.

    Each signal maps specific acoustic/temporal patterns to psychological states.
    """

    signal_type: SignalType
    name: str
    description: str
    psychological_interpretation: str

    # Detection thresholds
    thresholds: dict[str, float]

    # Weight for importance scoring
    weight: float = 1.0

    # Required features for detection
    required_features: list[str] = field(default_factory=list)

    # Visual/UI representation
    ui_color: str = "#888888"
    ui_icon: str = "signal"


# ══════════════════════════════════════════════════════════════
# The Signal Atlas
# ══════════════════════════════════════════════════════════════


class SignalAtlas:
    """
    The Signal Atlas - the complete mapping of bio-acoustic signals.

    This class contains all signal definitions and their detection parameters.
    It serves as the "rulebook" for the AI analysis layer.
    """

    SIGNALS: dict[SignalType, SignalDefinition] = {
        # ──────────────────────────────────────────────────────────
        # TEMPORAL SIGNALS (Time & Silence)
        # ──────────────────────────────────────────────────────────
        SignalType.TRUTH_GAP: SignalDefinition(
            signal_type=SignalType.TRUTH_GAP,
            name="Truth Gap",
            description="Extended response latency after a question",
            psychological_interpretation=(
                "Cognitive load indicating calculation rather than recall. "
                "High probability of fabrication, hesitation, or careful word choice."
            ),
            thresholds={
                "latency_ms_min": 800,
                "latency_ms_high": 1500,
                "confidence_boost_per_100ms": 0.05,
            },
            weight=1.2,
            required_features=["response_latency"],
            ui_color="#FCD34D",
            ui_icon="pause",
        ),
        SignalType.STEAMROLL: SignalDefinition(
            signal_type=SignalType.STEAMROLL,
            name="Steamroll",
            description="Aggressive interruption with volume dominance",
            psychological_interpretation=(
                "Dominance/aggression signal. The speaker is not listening; "
                "they are conquering the airtime."
            ),
            thresholds={
                "overlap_ms_min": 2000,
                "volume_diff_db_min": 10,
                "interrupt_velocity_min": 0.8,
            },
            weight=1.1,
            required_features=["overlap_duration", "energy_mean"],
            ui_color="#EF4444",
            ui_icon="volume-high",
        ),
        SignalType.DEAD_AIR: SignalDefinition(
            signal_type=SignalType.DEAD_AIR,
            name="Dead Air",
            description="Extended silence in group conversation",
            psychological_interpretation=(
                "Disengagement or friction. The 'awkward silence' indicating "
                "group cohesion failure or topic avoidance."
            ),
            thresholds={
                "silence_ms_min": 5000,
                "participant_count_min": 3,
                "previous_activity_threshold": 0.5,
            },
            weight=1.0,
            required_features=["silence_duration"],
            ui_color="#6B7280",
            ui_icon="silence",
        ),
        # ──────────────────────────────────────────────────────────
        # SPECTRAL SIGNALS (Pitch & Tone)
        # ──────────────────────────────────────────────────────────
        SignalType.MICRO_TREMOR: SignalDefinition(
            signal_type=SignalType.MICRO_TREMOR,
            name="Micro-Tremor",
            description="High jitter indicating vocal cord tension",
            psychological_interpretation=(
                "Stress or deception marker. The vocal cords are tightening "
                "due to fight-or-flight response activation."
            ),
            thresholds={
                "jitter_min": 0.02,
                "jitter_max": 0.08,
                "duration_ms_min": 500,
                "shimmer_boost_threshold": 0.03,
            },
            weight=1.3,
            required_features=["jitter", "shimmer"],
            ui_color="#F97316",
            ui_icon="wave",
        ),
        SignalType.MONOTONE: SignalDefinition(
            signal_type=SignalType.MONOTONE,
            name="Monotone",
            description="Flat pitch variance over extended period",
            psychological_interpretation=(
                "Burnout or apathy. The speaker has 'checked out' emotionally "
                "or is deliberately suppressing affect."
            ),
            thresholds={
                "pitch_variance_max": 0.10,
                "duration_ms_min": 30000,
                "energy_variance_max": 0.15,
            },
            weight=0.9,
            required_features=["pitch_std", "energy_std"],
            ui_color="#9CA3AF",
            ui_icon="minus",
        ),
        SignalType.UPTICK: SignalDefinition(
            signal_type=SignalType.UPTICK,
            name="Uptick",
            description="Rising intonation on declarative statements",
            psychological_interpretation=(
                "Submissiveness or insecurity. Seeking validation rather than "
                "stating facts with confidence (upspeak)."
            ),
            thresholds={
                "pitch_slope_min": 15,  # Hz rise in final 500ms
                "sentence_type": "declarative",
                "frequency_threshold": 0.3,
            },
            weight=0.8,
            required_features=["pitch_slope"],
            ui_color="#A78BFA",
            ui_icon="trending-up",
        ),
        # ──────────────────────────────────────────────────────────
        # CONTEXTUAL SIGNALS (Environment)
        # ──────────────────────────────────────────────────────────
        SignalType.ECHO_CHAMBER: SignalDefinition(
            signal_type=SignalType.ECHO_CHAMBER,
            name="Echo Chamber",
            description="High reverb and distant mic signature",
            psychological_interpretation=(
                "Low professional presence. Implies a large, empty room or "
                "lack of dedicated communication equipment."
            ),
            thresholds={
                "reverb_ratio_min": 0.3,
                "signal_clarity_max": 0.6,
            },
            weight=0.5,
            required_features=["reverb_estimate", "signal_clarity"],
            ui_color="#D1D5DB",
            ui_icon="building",
        ),
        SignalType.COFFEE_SHOP: SignalDefinition(
            signal_type=SignalType.COFFEE_SHOP,
            name="Coffee Shop",
            description="Background noise classified as public chatter",
            psychological_interpretation=(
                "Distracted or exposed environment. The user is likely "
                "multitasking or feels exposed in public."
            ),
            thresholds={
                "background_noise_class": "chatter",
                "noise_level_db_min": -30,
            },
            weight=0.5,
            required_features=["noise_classification"],
            ui_color="#92400E",
            ui_icon="coffee",
        ),
        # ──────────────────────────────────────────────────────────
        # COMPOSITE SIGNALS
        # ──────────────────────────────────────────────────────────
        SignalType.STRESS_SPIKE: SignalDefinition(
            signal_type=SignalType.STRESS_SPIKE,
            name="Stress Spike",
            description="Sudden multi-indicator stress increase",
            psychological_interpretation=(
                "Acute stress response. Multiple physiological stress "
                "indicators activated simultaneously."
            ),
            thresholds={
                "stress_delta_min": 0.3,
                "window_ms": 5000,
                "required_components": 2,
            },
            weight=1.4,
            required_features=["jitter", "speech_rate", "pitch_std"],
            ui_color="#DC2626",
            ui_icon="alert",
        ),
        SignalType.DISENGAGEMENT: SignalDefinition(
            signal_type=SignalType.DISENGAGEMENT,
            name="Disengagement",
            description="Pattern indicating loss of interest",
            psychological_interpretation=(
                "Cognitive or emotional withdrawal from the conversation. "
                "May indicate boredom, disagreement, or burnout."
            ),
            thresholds={
                "talk_ratio_decrease": 0.5,
                "response_latency_increase": 1.5,
                "pitch_variance_decrease": 0.3,
            },
            weight=1.1,
            required_features=["talk_ratio", "response_latency", "pitch_std"],
            ui_color="#4B5563",
            ui_icon="user-minus",
        ),
        SignalType.DECEPTION_MARKER: SignalDefinition(
            signal_type=SignalType.DECEPTION_MARKER,
            name="Deception Marker",
            description="Combination suggesting untruthfulness",
            psychological_interpretation=(
                "Multiple deception indicators present. Note: This is a "
                "probabilistic signal, not definitive proof of deception."
            ),
            thresholds={
                "truth_gap_present": True,
                "micro_tremor_present": True,
                "hedge_word_density": 0.1,
                "min_component_confidence": 0.6,
            },
            weight=1.5,
            required_features=["response_latency", "jitter", "transcript"],
            ui_color="#7C2D12",
            ui_icon="eye-off",
        ),
        SignalType.ENTHUSIASM_SURGE: SignalDefinition(
            signal_type=SignalType.ENTHUSIASM_SURGE,
            name="Enthusiasm Surge",
            description="Positive engagement spike",
            psychological_interpretation=(
                "Genuine positive engagement. Energy, pitch, and speech "
                "rate all elevated in a harmonious pattern."
            ),
            thresholds={
                "energy_increase": 1.3,
                "pitch_increase": 1.2,
                "speech_rate_increase": 1.15,
                "window_ms": 10000,
            },
            weight=1.0,
            required_features=["energy_mean", "pitch_mean", "speech_rate"],
            ui_color="#10B981",
            ui_icon="smile",
        ),
        SignalType.AGREEMENT_SIGNAL: SignalDefinition(
            signal_type=SignalType.AGREEMENT_SIGNAL,
            name="Agreement Signal",
            description="Vocal pattern indicating agreement",
            psychological_interpretation="Genuine agreement and alignment with the speaker.",
            thresholds={
                "pitch_mirror_correlation": 0.6,
                "timing_sync_threshold": 0.3,
            },
            weight=0.8,
            required_features=["pitch_mean", "timing"],
            ui_color="#059669",
            ui_icon="check",
        ),
        SignalType.DISAGREEMENT_SIGNAL: SignalDefinition(
            signal_type=SignalType.DISAGREEMENT_SIGNAL,
            name="Disagreement Signal",
            description="Vocal pattern indicating disagreement",
            psychological_interpretation=(
                "Suppressed or emerging disagreement. The speaker may not "
                "verbalize it but vocal patterns indicate misalignment."
            ),
            thresholds={
                "pitch_divergence": 0.4,
                "micro_pause_frequency": 0.5,
                "breath_hold_indicator": True,
            },
            weight=1.0,
            required_features=["pitch_mean", "pause_frequency"],
            ui_color="#DC2626",
            ui_icon="x",
        ),
    }

    @classmethod
    def get_definition(cls, signal_type: SignalType) -> SignalDefinition:
        """Get signal definition by type."""
        return cls.SIGNALS[signal_type]

    @classmethod
    def get_all_definitions(cls) -> dict[SignalType, SignalDefinition]:
        """Get all signal definitions."""
        return cls.SIGNALS

    @classmethod
    def get_temporal_signals(cls) -> list[SignalType]:
        """Get temporal signal types."""
        return [SignalType.TRUTH_GAP, SignalType.STEAMROLL, SignalType.DEAD_AIR]

    @classmethod
    def get_spectral_signals(cls) -> list[SignalType]:
        """Get spectral signal types."""
        return [SignalType.MICRO_TREMOR, SignalType.MONOTONE, SignalType.UPTICK]

    @classmethod
    def get_composite_signals(cls) -> list[SignalType]:
        """Get composite signal types."""
        return [
            SignalType.STRESS_SPIKE,
            SignalType.DISENGAGEMENT,
            SignalType.DECEPTION_MARKER,
            SignalType.ENTHUSIASM_SURGE,
        ]


# ══════════════════════════════════════════════════════════════
# Signal Detector
# ══════════════════════════════════════════════════════════════


class SignalDetector:
    """
    Signal detection engine.

    Analyzes prosodic features, transcript segments, and context to
    detect signals defined in the Signal Atlas.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        enabled_signals: list[SignalType] | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.enabled_signals = enabled_signals or list(SignalType)
        self.atlas = SignalAtlas()

    def detect_all(
        self,
        session_id: UUID,
        segments: list[dict[str, Any]],
        prosodics: list[dict[str, Any]],
        speaker_baselines: dict[str, dict[str, float]] | None = None,
    ) -> list[Signal]:
        """
        Detect all signals across the session.

        Args:
            session_id: UUID of the analysis session
            segments: Transcript segments with speaker attribution
            prosodics: Prosodic features indexed by timestamp
            speaker_baselines: Optional baseline metrics per speaker

        Returns:
            List of detected Signal objects
        """
        signals: list[Signal] = []
        prosodic_index = {p["timestamp_ms"]: p for p in prosodics}

        for i, segment in enumerate(segments):
            prev_segment = segments[i - 1] if i > 0 else None
            next_segment = segments[i + 1] if i < len(segments) - 1 else None

            segment_prosodics = self._get_segment_prosodics(segment, prosodic_index)
            speaker_baseline = (
                speaker_baselines.get(segment.get("speaker_id", ""))
                if speaker_baselines
                else None
            )

            # Check each enabled signal type
            for signal_type in self.enabled_signals:
                detected = self._detect_signal(
                    signal_type=signal_type,
                    segment=segment,
                    prosodics=segment_prosodics,
                    prev_segment=prev_segment,
                    next_segment=next_segment,
                    baseline=speaker_baseline,
                )

                if detected and detected.confidence >= self.confidence_threshold:
                    detected_signal = Signal(
                        id=uuid4(),
                        session_id=session_id,
                        speaker_id=segment.get("speaker_id"),
                        signal_type=signal_type,
                        timestamp_ms=detected.timestamp_ms,
                        duration_ms=detected.duration_ms,
                        confidence=detected.confidence,
                        intensity=detected.intensity,
                        metrics=detected.metrics,
                        context=detected.context,
                    )
                    signals.append(detected_signal)

        # Sort by timestamp
        signals.sort(key=lambda s: s.timestamp_ms)

        logger.info(
            "Signal detection complete",
            session_id=str(session_id),
            signal_count=len(signals),
        )

        return signals

    def _detect_signal(
        self,
        signal_type: SignalType,
        segment: dict[str, Any],
        prosodics: dict[str, float],
        prev_segment: dict[str, Any] | None,
        next_segment: dict[str, Any] | None,
        baseline: dict[str, float] | None,
    ) -> "DetectionResult | None":
        """Detect a specific signal type in a segment."""
        definition = self.atlas.get_definition(signal_type)

        # Check required features are present
        if not self._has_required_features(prosodics, definition.required_features):
            return None

        # Route to specific detector
        detectors: dict[SignalType, Callable] = {
            SignalType.TRUTH_GAP: self._detect_truth_gap,
            SignalType.STEAMROLL: self._detect_steamroll,
            SignalType.MICRO_TREMOR: self._detect_micro_tremor,
            SignalType.MONOTONE: self._detect_monotone,
            SignalType.UPTICK: self._detect_uptick,
            SignalType.STRESS_SPIKE: self._detect_stress_spike,
            SignalType.DISENGAGEMENT: self._detect_disengagement,
            SignalType.ENTHUSIASM_SURGE: self._detect_enthusiasm_surge,
        }

        detector = detectors.get(signal_type)
        if detector:
            return detector(
                definition=definition,
                segment=segment,
                prosodics=prosodics,
                prev_segment=prev_segment,
                next_segment=next_segment,
                baseline=baseline,
            )

        return None

    def _detect_truth_gap(
        self,
        definition: SignalDefinition,
        segment: dict[str, Any],
        prosodics: dict[str, float],
        prev_segment: dict[str, Any] | None,
        **kwargs,
    ) -> "DetectionResult | None":
        """Detect Truth Gap signal."""
        if not prev_segment:
            return None

        # Calculate response latency
        latency = segment["start_ms"] - prev_segment["end_ms"]

        # Check if this looks like a Q&A pattern
        is_qa = (
            prev_segment.get("is_question", False)
            or prev_segment.get("text", "").strip().endswith("?")
        )
        different_speakers = segment.get("speaker_id") != prev_segment.get("speaker_id")

        if not (is_qa and different_speakers):
            return None

        thresholds = definition.thresholds
        if latency < thresholds["latency_ms_min"]:
            return None

        # Calculate confidence
        base_confidence = 0.5
        extra_ms = latency - thresholds["latency_ms_min"]
        confidence = min(
            1.0, base_confidence + (extra_ms / 100) * thresholds["confidence_boost_per_100ms"]
        )

        # Calculate intensity
        intensity = min(1.0, latency / thresholds["latency_ms_high"])

        return DetectionResult(
            timestamp_ms=prev_segment["end_ms"],
            duration_ms=int(latency),
            confidence=confidence,
            intensity=intensity,
            metrics={"latency_ms": latency},
            context={
                "question": prev_segment.get("text", "")[:100],
                "answer_start": segment.get("text", "")[:50],
            },
        )

    def _detect_steamroll(
        self,
        definition: SignalDefinition,
        segment: dict[str, Any],
        prosodics: dict[str, float],
        prev_segment: dict[str, Any] | None,
        **kwargs,
    ) -> "DetectionResult | None":
        """Detect Steamroll signal."""
        if not prev_segment:
            return None

        # Check for overlap
        overlap = max(0, prev_segment["end_ms"] - segment["start_ms"])

        thresholds = definition.thresholds
        if overlap < thresholds["overlap_ms_min"]:
            return None

        # Check energy differential (simplified - would use actual dB in production)
        energy_current = prosodics.get("energy_mean", 0.5)
        energy_prev = kwargs.get("baseline", {}).get("energy_mean", 0.5)
        volume_diff = (energy_current - energy_prev) * 20  # Rough dB estimate

        if volume_diff < thresholds["volume_diff_db_min"]:
            return None

        confidence = min(1.0, 0.6 + (overlap / 5000) * 0.2 + (volume_diff / 20) * 0.2)
        intensity = min(1.0, overlap / 4000)

        return DetectionResult(
            timestamp_ms=segment["start_ms"],
            duration_ms=overlap,
            confidence=confidence,
            intensity=intensity,
            metrics={
                "overlap_ms": overlap,
                "volume_diff_db": volume_diff,
            },
            context={
                "interrupted_text": prev_segment.get("text", "")[:100],
                "interrupting_text": segment.get("text", "")[:50],
            },
        )

    def _detect_micro_tremor(
        self,
        definition: SignalDefinition,
        segment: dict[str, Any],
        prosodics: dict[str, float],
        **kwargs,
    ) -> "DetectionResult | None":
        """Detect Micro-Tremor signal."""
        jitter = prosodics.get("jitter", 0)
        shimmer = prosodics.get("shimmer", 0)

        thresholds = definition.thresholds

        # Check jitter is in stress range
        if not (thresholds["jitter_min"] <= jitter <= thresholds["jitter_max"]):
            return None

        duration = segment["end_ms"] - segment["start_ms"]
        if duration < thresholds["duration_ms_min"]:
            return None

        # Calculate confidence
        jitter_range = thresholds["jitter_max"] - thresholds["jitter_min"]
        jitter_normalized = (jitter - thresholds["jitter_min"]) / jitter_range

        shimmer_boost = 0.1 if shimmer > thresholds["shimmer_boost_threshold"] else 0
        confidence = min(1.0, 0.5 + jitter_normalized * 0.3 + shimmer_boost)
        intensity = jitter_normalized

        return DetectionResult(
            timestamp_ms=segment["start_ms"],
            duration_ms=duration,
            confidence=confidence,
            intensity=intensity,
            metrics={
                "jitter": jitter,
                "shimmer": shimmer,
            },
            context={"segment_text": segment.get("text", "")[:100]},
        )

    def _detect_monotone(
        self,
        definition: SignalDefinition,
        segment: dict[str, Any],
        prosodics: dict[str, float],
        **kwargs,
    ) -> "DetectionResult | None":
        """Detect Monotone signal."""
        pitch_std = prosodics.get("pitch_std", 1.0)
        energy_std = prosodics.get("energy_std", 1.0)

        thresholds = definition.thresholds

        duration = segment["end_ms"] - segment["start_ms"]
        if duration < thresholds["duration_ms_min"]:
            return None

        # Check for flat variance
        pitch_flat = pitch_std < thresholds["pitch_variance_max"]
        energy_flat = energy_std < thresholds["energy_variance_max"]

        if not (pitch_flat and energy_flat):
            return None

        confidence = min(
            1.0,
            0.6
            + (1 - pitch_std / thresholds["pitch_variance_max"]) * 0.2
            + (1 - energy_std / thresholds["energy_variance_max"]) * 0.2,
        )
        intensity = 1 - (pitch_std + energy_std) / 2

        return DetectionResult(
            timestamp_ms=segment["start_ms"],
            duration_ms=duration,
            confidence=confidence,
            intensity=intensity,
            metrics={
                "pitch_variance": pitch_std,
                "energy_variance": energy_std,
            },
            context={"segment_text": segment.get("text", "")[:100]},
        )

    def _detect_uptick(
        self,
        definition: SignalDefinition,
        segment: dict[str, Any],
        prosodics: dict[str, float],
        **kwargs,
    ) -> "DetectionResult | None":
        """Detect Uptick (upspeak) signal."""
        pitch_slope = prosodics.get("pitch_slope", 0)

        thresholds = definition.thresholds

        # Check for rising intonation at end
        if pitch_slope < thresholds["pitch_slope_min"]:
            return None

        # Check it's a declarative sentence (simplified check)
        text = segment.get("text", "").strip()
        if text.endswith("?"):
            return None  # Actual questions are fine to have rising intonation

        confidence = min(1.0, 0.5 + (pitch_slope / 30) * 0.5)
        intensity = min(1.0, pitch_slope / 25)

        return DetectionResult(
            timestamp_ms=segment["end_ms"] - 500,  # Focus on end of segment
            duration_ms=500,
            confidence=confidence,
            intensity=intensity,
            metrics={"pitch_slope": pitch_slope},
            context={"sentence": text[-100:]},
        )

    def _detect_stress_spike(
        self,
        definition: SignalDefinition,
        segment: dict[str, Any],
        prosodics: dict[str, float],
        baseline: dict[str, float] | None,
        **kwargs,
    ) -> "DetectionResult | None":
        """Detect Stress Spike (composite signal)."""
        if not baseline:
            return None

        thresholds = definition.thresholds

        # Calculate deltas from baseline
        jitter_delta = prosodics.get("jitter", 0) - baseline.get("jitter", 0)
        rate_delta = prosodics.get("speech_rate", 0) / max(
            baseline.get("speech_rate", 1), 0.1
        )
        pitch_delta = prosodics.get("pitch_std", 0) - baseline.get("pitch_std", 0)

        # Count elevated components
        elevated = 0
        if jitter_delta > 0.01:
            elevated += 1
        if rate_delta > 1.2:
            elevated += 1
        if pitch_delta > 5:
            elevated += 1

        if elevated < thresholds["required_components"]:
            return None

        avg_delta = (jitter_delta * 10 + (rate_delta - 1) + pitch_delta / 20) / 3
        if avg_delta < thresholds["stress_delta_min"]:
            return None

        confidence = min(1.0, 0.5 + elevated * 0.15 + avg_delta * 0.2)
        intensity = min(1.0, avg_delta)

        return DetectionResult(
            timestamp_ms=segment["start_ms"],
            duration_ms=segment["end_ms"] - segment["start_ms"],
            confidence=confidence,
            intensity=intensity,
            metrics={
                "jitter_delta": jitter_delta,
                "rate_delta": rate_delta,
                "pitch_delta": pitch_delta,
                "elevated_components": elevated,
            },
            context={"segment_text": segment.get("text", "")[:100]},
        )

    def _detect_disengagement(
        self,
        definition: SignalDefinition,
        segment: dict[str, Any],
        prosodics: dict[str, float],
        baseline: dict[str, float] | None,
        **kwargs,
    ) -> "DetectionResult | None":
        """Detect Disengagement signal."""
        if not baseline:
            return None

        # Check for decreased engagement indicators
        pitch_std = prosodics.get("pitch_std", 0)
        baseline_pitch = baseline.get("pitch_std", 1)

        if baseline_pitch > 0:
            pitch_decrease = 1 - (pitch_std / baseline_pitch)
        else:
            pitch_decrease = 0

        thresholds = definition.thresholds
        if pitch_decrease < thresholds["pitch_variance_decrease"]:
            return None

        confidence = min(1.0, 0.5 + pitch_decrease * 0.5)
        intensity = pitch_decrease

        return DetectionResult(
            timestamp_ms=segment["start_ms"],
            duration_ms=segment["end_ms"] - segment["start_ms"],
            confidence=confidence,
            intensity=intensity,
            metrics={
                "pitch_decrease": pitch_decrease,
                "current_pitch_std": pitch_std,
                "baseline_pitch_std": baseline_pitch,
            },
            context={"segment_text": segment.get("text", "")[:100]},
        )

    def _detect_enthusiasm_surge(
        self,
        definition: SignalDefinition,
        segment: dict[str, Any],
        prosodics: dict[str, float],
        baseline: dict[str, float] | None,
        **kwargs,
    ) -> "DetectionResult | None":
        """Detect Enthusiasm Surge signal."""
        if not baseline:
            return None

        thresholds = definition.thresholds

        # Calculate ratios vs baseline
        energy_ratio = prosodics.get("energy_mean", 0) / max(
            baseline.get("energy_mean", 0.1), 0.1
        )
        pitch_ratio = prosodics.get("pitch_mean", 0) / max(
            baseline.get("pitch_mean", 100), 100
        )
        rate_ratio = prosodics.get("speech_rate", 0) / max(
            baseline.get("speech_rate", 1), 0.1
        )

        # Check all indicators elevated
        if (
            energy_ratio < thresholds["energy_increase"]
            or pitch_ratio < thresholds["pitch_increase"]
            or rate_ratio < thresholds["speech_rate_increase"]
        ):
            return None

        avg_increase = (energy_ratio + pitch_ratio + rate_ratio) / 3
        confidence = min(1.0, 0.5 + (avg_increase - 1) * 0.5)
        intensity = min(1.0, avg_increase - 1)

        return DetectionResult(
            timestamp_ms=segment["start_ms"],
            duration_ms=segment["end_ms"] - segment["start_ms"],
            confidence=confidence,
            intensity=intensity,
            metrics={
                "energy_ratio": energy_ratio,
                "pitch_ratio": pitch_ratio,
                "rate_ratio": rate_ratio,
            },
            context={"segment_text": segment.get("text", "")[:100]},
        )

    @staticmethod
    def _get_segment_prosodics(
        segment: dict[str, Any], prosodic_index: dict[int, dict]
    ) -> dict[str, float]:
        """Get aggregated prosodics for a segment."""
        start = segment["start_ms"]
        end = segment["end_ms"]

        matching = [
            p for ts, p in prosodic_index.items() if start <= ts < end
        ]

        if not matching:
            return {}

        # Average numeric fields
        result: dict[str, float] = {}
        for key in matching[0].keys():
            if key != "timestamp_ms" and isinstance(matching[0][key], (int, float)):
                values = [m[key] for m in matching if m.get(key) is not None]
                if values:
                    result[key] = sum(values) / len(values)

        return result

    @staticmethod
    def _has_required_features(
        prosodics: dict[str, float], required: list[str]
    ) -> bool:
        """Check if prosodics has required features."""
        for feature in required:
            if feature not in prosodics and feature not in ["transcript", "timing"]:
                return False
        return True


@dataclass
class DetectionResult:
    """Result of signal detection attempt."""

    timestamp_ms: int
    duration_ms: int
    confidence: float
    intensity: float
    metrics: dict[str, Any]
    context: dict[str, Any]
