"""
Pipeline Orchestrator

Coordinates the execution of all pipeline stages for audio analysis.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog

from subtext.config import settings
from subtext.core.models import Signal

from .signals import SignalDetector
from .stages import (
    CleanseStage,
    DiarizeStage,
    ProsodicsStage,
    StageResult,
    SynthesizeStage,
    TranscribeStage,
)

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    # Model configuration
    whisper_model: str = "large-v3"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    llm_model: str = "gpt-4-turbo-preview"
    llm_provider: str = "openai"

    # Processing options
    language: str | None = None  # Auto-detect if None
    min_speakers: int | None = None
    max_speakers: int | None = None

    # Signal detection
    signal_confidence_threshold: float = 0.5
    enabled_signals: list[str] | None = None

    # Feature extraction
    prosodics_window_ms: int = 1000
    prosodics_hop_ms: int = 500

    # Resource limits
    max_duration_seconds: int = 7200  # 2 hours
    target_sample_rate: int = 16000


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    session_id: UUID
    success: bool

    # Core results
    speakers: list[dict[str, Any]] = field(default_factory=list)
    transcript_segments: list[dict[str, Any]] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    insights: dict[str, Any] = field(default_factory=dict)
    speaker_metrics: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    duration_ms: int = 0
    language: str | None = None
    processing_time_ms: float = 0
    stage_results: list[StageResult] = field(default_factory=list)

    # Error info
    error: str | None = None


# ══════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════


class PipelineOrchestrator:
    """
    Orchestrates the Subtext audio analysis pipeline.

    Pipeline stages:
    1. Cleanse   - Noise suppression (DeepFilterNet)
    2. Diarize   - Speaker identification (Pyannote)
    3. Transcribe - Speech-to-text (WhisperX)
    4. Prosodics - Acoustic feature extraction
    5. Detect    - Signal detection from Signal Atlas
    6. Synthesize - Final analysis and insights

    Usage:
        config = PipelineConfig(language="en")
        orchestrator = PipelineOrchestrator(config)
        result = await orchestrator.process_file(session_id, audio_path)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._stages_initialized = False

        # Initialize stages
        self.cleanse = CleanseStage(
            preserve_prosody=True,
            target_sample_rate=self.config.target_sample_rate,
        )
        self.diarize = DiarizeStage(
            model_name=self.config.pyannote_model,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )
        self.transcribe = TranscribeStage(
            model_name=self.config.whisper_model,
            language=self.config.language,
            word_timestamps=True,
        )
        self.prosodics = ProsodicsStage(
            window_size_ms=self.config.prosodics_window_ms,
            hop_size_ms=self.config.prosodics_hop_ms,
        )
        self.synthesize = SynthesizeStage(
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
            signal_confidence_threshold=self.config.signal_confidence_threshold,
        )

        # Signal detector
        self.detector = SignalDetector(
            confidence_threshold=self.config.signal_confidence_threshold,
        )

    async def initialize(self) -> None:
        """Initialize all pipeline stages."""
        if self._stages_initialized:
            return

        logger.info("Initializing pipeline stages")

        await asyncio.gather(
            self.cleanse.initialize(),
            self.diarize.initialize(),
            self.transcribe.initialize(),
            self.prosodics.initialize(),
            self.synthesize.initialize(),
        )

        self._stages_initialized = True
        logger.info("Pipeline initialization complete")

    async def process_file(
        self,
        session_id: UUID,
        audio_path: str,
    ) -> PipelineResult:
        """
        Process a complete audio file through the pipeline.

        Args:
            session_id: UUID of the analysis session
            audio_path: Path to the audio file

        Returns:
            PipelineResult with all analysis data
        """
        log = logger.bind(session_id=str(session_id))
        log.info("Starting file pipeline", audio_path=audio_path)

        start_time = time.perf_counter()
        stage_results: list[StageResult] = []
        context: dict[str, Any] = {
            "session_id": session_id,
            "audio_path": audio_path,
        }

        try:
            await self.initialize()

            # ──────────────────────────────────────────────────────
            # Stage 1: Cleanse (Noise Suppression)
            # ──────────────────────────────────────────────────────
            cleanse_result = await self._run_stage(
                self.cleanse,
                audio_path=audio_path,
            )
            stage_results.append(cleanse_result)

            if not cleanse_result.success:
                raise RuntimeError(f"Cleanse failed: {cleanse_result.error}")

            context["audio"] = cleanse_result.data["audio"]
            context["sample_rate"] = cleanse_result.data["sample_rate"]
            context["noise_profile"] = cleanse_result.data["noise_profile"]

            # Calculate duration
            duration_ms = int(
                len(context["audio"]) / context["sample_rate"] * 1000
            )

            # ──────────────────────────────────────────────────────
            # Stage 2 & 3: Diarize and Transcribe (parallel)
            # ──────────────────────────────────────────────────────
            diarize_task = self._run_stage(
                self.diarize,
                audio=context["audio"],
                sample_rate=context["sample_rate"],
            )
            transcribe_task = self._run_stage(
                self.transcribe,
                audio=context["audio"],
                sample_rate=context["sample_rate"],
            )

            diarize_result, transcribe_result = await asyncio.gather(
                diarize_task, transcribe_task
            )
            stage_results.extend([diarize_result, transcribe_result])

            if not diarize_result.success:
                raise RuntimeError(f"Diarize failed: {diarize_result.error}")
            if not transcribe_result.success:
                raise RuntimeError(f"Transcribe failed: {transcribe_result.error}")

            context["speakers"] = diarize_result.data["speakers"]
            context["diarization_segments"] = diarize_result.data["segments"]
            context["transcript"] = transcribe_result.data["transcript"]
            context["words"] = transcribe_result.data["words"]
            context["language"] = transcribe_result.data["language"]

            # Align transcript with speaker segments
            context["aligned_segments"] = self._align_transcript_speakers(
                context["words"],
                context["diarization_segments"],
            )

            # ──────────────────────────────────────────────────────
            # Stage 4: Prosodics (Acoustic Features)
            # ──────────────────────────────────────────────────────
            prosodics_result = await self._run_stage(
                self.prosodics,
                audio=context["audio"],
                sample_rate=context["sample_rate"],
                segments=context["aligned_segments"],
            )
            stage_results.append(prosodics_result)

            if not prosodics_result.success:
                raise RuntimeError(f"Prosodics failed: {prosodics_result.error}")

            context["prosodics"] = prosodics_result.data["features"]

            # ──────────────────────────────────────────────────────
            # Stage 5: Signal Detection
            # ──────────────────────────────────────────────────────
            signals = self.detector.detect_all(
                session_id=session_id,
                segments=context["aligned_segments"],
                prosodics=context["prosodics"],
            )
            context["signals"] = signals

            # ──────────────────────────────────────────────────────
            # Stage 6: Synthesize (Timeline & Insights)
            # ──────────────────────────────────────────────────────
            synthesize_result = await self._run_stage(
                self.synthesize,
                session_id=session_id,
                transcript_segments=context["aligned_segments"],
                speakers=context["speakers"],
                prosodics=context["prosodics"],
                signals=[s.model_dump() for s in signals],
            )
            stage_results.append(synthesize_result)

            if not synthesize_result.success:
                raise RuntimeError(f"Synthesize failed: {synthesize_result.error}")

            # ──────────────────────────────────────────────────────
            # Build Final Result
            # ──────────────────────────────────────────────────────
            processing_time = (time.perf_counter() - start_time) * 1000

            log.info(
                "Pipeline complete",
                duration_ms=duration_ms,
                processing_time_ms=processing_time,
                speaker_count=len(context["speakers"]),
                signal_count=len(signals),
            )

            return PipelineResult(
                session_id=session_id,
                success=True,
                speakers=context["speakers"],
                transcript_segments=context["aligned_segments"],
                signals=signals,
                timeline=synthesize_result.data["timeline"],
                insights=synthesize_result.data["insights"],
                speaker_metrics=synthesize_result.data["speaker_metrics"],
                duration_ms=duration_ms,
                language=context["language"],
                processing_time_ms=processing_time,
                stage_results=stage_results,
            )

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            log.error("Pipeline failed", error=str(e))

            return PipelineResult(
                session_id=session_id,
                success=False,
                error=str(e),
                processing_time_ms=processing_time,
                stage_results=stage_results,
            )

    async def _run_stage(self, stage: Any, **kwargs) -> StageResult:
        """Execute a pipeline stage with timing and error handling."""
        log = logger.bind(stage=stage.name)
        start = time.perf_counter()

        try:
            log.debug("Stage starting")
            result = await stage.process(**kwargs)
            duration = (time.perf_counter() - start) * 1000

            log.debug("Stage completed", duration_ms=duration)

            return StageResult(
                success=True,
                data=result,
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            log.error("Stage failed", error=str(e), duration_ms=duration)

            return StageResult(
                success=False,
                data={},
                duration_ms=duration,
                error=str(e),
            )

    def _align_transcript_speakers(
        self,
        words: list[dict[str, Any]],
        diarization_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Align transcribed words with speaker diarization segments.

        Creates transcript segments with speaker attribution.
        """
        aligned = []

        for segment in diarization_segments:
            start_ms = segment["start_ms"]
            end_ms = segment["end_ms"]
            speaker_id = segment["speaker_id"]

            # Find words in this segment
            segment_words = [
                w for w in words if start_ms <= w.get("start_ms", 0) < end_ms
            ]

            if segment_words:
                text = " ".join(w.get("text", "") for w in segment_words)
                avg_confidence = (
                    sum(w.get("confidence", 0) for w in segment_words)
                    / len(segment_words)
                )

                # Check if it's a question
                is_question = text.strip().endswith("?")

                aligned.append(
                    {
                        "speaker_id": speaker_id,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": text,
                        "words": segment_words,
                        "confidence": avg_confidence,
                        "is_question": is_question,
                    }
                )

        return aligned


# ══════════════════════════════════════════════════════════════
# Factory Function
# ══════════════════════════════════════════════════════════════


def create_pipeline(
    language: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> PipelineOrchestrator:
    """Create a configured pipeline orchestrator."""
    config = PipelineConfig(
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        whisper_model=settings.whisper_model,
        pyannote_model=settings.pyannote_model,
        llm_model=settings.llm_model,
        llm_provider=settings.llm_provider,
        signal_confidence_threshold=settings.signal_confidence_threshold,
    )
    return PipelineOrchestrator(config)
