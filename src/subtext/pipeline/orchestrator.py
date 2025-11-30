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
    VADStage,
    DiarizeStage,
    TranscribeStage,
    EmotionStage,
    ProsodicsStage,
    StageResult,
    SynthesizeStage,
)

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    # ASR Configuration (supports multiple backends)
    asr_backend: str = "whisperx"  # 'whisperx', 'canary', 'parakeet'
    whisper_model: str = "large-v3"

    # Diarization Configuration
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    extract_speaker_embeddings: bool = True

    # Emotion Configuration
    emotion_model: str = "iic/emotion2vec_plus_large"
    enable_emotion_detection: bool = True

    # LLM Configuration
    llm_model: str = "gpt-4-turbo-preview"
    llm_provider: str = "openai"

    # Processing options
    language: str | None = None  # Auto-detect if None
    min_speakers: int | None = None
    max_speakers: int | None = None

    # VAD Configuration
    vad_threshold: float = 0.5
    enable_vad: bool = True

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

    # VAD results
    speech_segments: list[dict[str, Any]] = field(default_factory=list)
    speech_ratio: float = 0.0

    # Emotion results
    emotions: list[dict[str, Any]] = field(default_factory=list)
    dominant_emotion: str | None = None
    vad_scores: dict[str, float] = field(default_factory=dict)  # Valence-Arousal-Dominance

    # Speaker embeddings (for voice fingerprinting)
    speaker_embeddings: dict[str, list[float]] = field(default_factory=dict)

    # Metadata
    duration_ms: int = 0
    language: str | None = None
    asr_backend: str | None = None
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
    1. Cleanse    - Noise suppression (DeepFilterNet)
    2. VAD        - Voice activity detection (Silero VAD)
    3. Diarize    - Speaker identification (Pyannote + ECAPA-TDNN)
    4. Transcribe - Speech-to-text (WhisperX/Canary/Parakeet)
    5. Emotion    - Speech emotion recognition (Emotion2Vec)
    6. Prosodics  - Acoustic feature extraction
    7. Detect     - Signal detection from Signal Atlas
    8. Synthesize - Final analysis and insights

    Usage:
        config = PipelineConfig(language="en", asr_backend="whisperx")
        orchestrator = PipelineOrchestrator(config)
        result = await orchestrator.process_file(session_id, audio_path)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._stages_initialized = False

        # Initialize stages - Best-in-class open source models
        self.cleanse = CleanseStage(
            preserve_prosody=True,
            target_sample_rate=self.config.target_sample_rate,
        )

        # Silero VAD (87.7% TPR vs WebRTC's 50%)
        self.vad = VADStage(
            threshold=self.config.vad_threshold,
        ) if self.config.enable_vad else None

        # Pyannote + ECAPA-TDNN (1.71% EER)
        self.diarize = DiarizeStage(
            model_name=self.config.pyannote_model,
            embedding_model=self.config.speaker_embedding_model,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
            extract_embeddings=self.config.extract_speaker_embeddings,
        )

        # Multi-ASR support: WhisperX (default), Canary (accuracy), Parakeet (speed)
        self.transcribe = TranscribeStage(
            backend=self.config.asr_backend,
            model_name=self.config.whisper_model if self.config.asr_backend == "whisperx" else None,
            language=self.config.language,
            word_timestamps=True,
        )

        # Emotion2Vec (SOTA on 9 multilingual datasets)
        self.emotion = EmotionStage(
            model_name=self.config.emotion_model,
        ) if self.config.enable_emotion_detection else None

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

        init_tasks = [
            self.cleanse.initialize(),
            self.diarize.initialize(),
            self.transcribe.initialize(),
            self.prosodics.initialize(),
            self.synthesize.initialize(),
        ]

        # Add optional stages
        if self.vad:
            init_tasks.append(self.vad.initialize())
        if self.emotion:
            init_tasks.append(self.emotion.initialize())

        await asyncio.gather(*init_tasks)

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
            # Stage 1: Cleanse (Noise Suppression - DeepFilterNet)
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
            # Stage 2: VAD (Voice Activity Detection - Silero VAD)
            # ──────────────────────────────────────────────────────
            speech_segments = []
            speech_ratio = 1.0

            if self.vad:
                vad_result = await self._run_stage(
                    self.vad,
                    audio=context["audio"],
                    sample_rate=context["sample_rate"],
                )
                stage_results.append(vad_result)

                if vad_result.success:
                    speech_segments = vad_result.data.get("speech_segments", [])
                    speech_ratio = vad_result.data.get("speech_ratio", 1.0)
                    context["speech_segments"] = speech_segments
                    log.debug("VAD complete", speech_ratio=f"{speech_ratio:.2%}")

            # ──────────────────────────────────────────────────────
            # Stage 3, 4, 5: Diarize, Transcribe, Emotion (parallel)
            # ──────────────────────────────────────────────────────
            parallel_tasks = [
                self._run_stage(
                    self.diarize,
                    audio=context["audio"],
                    sample_rate=context["sample_rate"],
                ),
                self._run_stage(
                    self.transcribe,
                    audio=context["audio"],
                    sample_rate=context["sample_rate"],
                ),
            ]

            # Add emotion detection if enabled
            if self.emotion:
                parallel_tasks.append(
                    self._run_stage(
                        self.emotion,
                        audio=context["audio"],
                        sample_rate=context["sample_rate"],
                    )
                )

            parallel_results = await asyncio.gather(*parallel_tasks)
            diarize_result = parallel_results[0]
            transcribe_result = parallel_results[1]
            emotion_result = parallel_results[2] if self.emotion else None

            stage_results.extend([diarize_result, transcribe_result])
            if emotion_result:
                stage_results.append(emotion_result)

            if not diarize_result.success:
                raise RuntimeError(f"Diarize failed: {diarize_result.error}")
            if not transcribe_result.success:
                raise RuntimeError(f"Transcribe failed: {transcribe_result.error}")

            context["speakers"] = diarize_result.data["speakers"]
            context["diarization_segments"] = diarize_result.data["segments"]
            context["speaker_embeddings"] = diarize_result.data.get("embeddings", {})
            context["transcript"] = transcribe_result.data["transcript"]
            context["words"] = transcribe_result.data["words"]
            context["language"] = transcribe_result.data["language"]
            context["asr_backend"] = transcribe_result.data.get("backend", "unknown")

            # Emotion results
            emotions = []
            dominant_emotion = None
            vad_scores = {}
            if emotion_result and emotion_result.success:
                emotions = emotion_result.data.get("emotions", [])
                dominant_emotion = emotion_result.data.get("dominant_emotion")
                vad_scores = emotion_result.data.get("vad", {})
                log.debug("Emotion detection complete", dominant=dominant_emotion)

            # Align transcript with speaker segments
            context["aligned_segments"] = self._align_transcript_speakers(
                context["words"],
                context["diarization_segments"],
            )

            # ──────────────────────────────────────────────────────
            # Stage 6: Prosodics (Acoustic Features)
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
            # Stage 7: Signal Detection
            # ──────────────────────────────────────────────────────
            signals = self.detector.detect_all(
                session_id=session_id,
                segments=context["aligned_segments"],
                prosodics=context["prosodics"],
            )
            context["signals"] = signals

            # ──────────────────────────────────────────────────────
            # Stage 8: Synthesize (Timeline & Insights)
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
                asr_backend=context["asr_backend"],
            )

            return PipelineResult(
                session_id=session_id,
                success=True,
                # Core results
                speakers=context["speakers"],
                transcript_segments=context["aligned_segments"],
                signals=signals,
                timeline=synthesize_result.data["timeline"],
                insights=synthesize_result.data["insights"],
                speaker_metrics=synthesize_result.data["speaker_metrics"],
                # VAD results
                speech_segments=speech_segments,
                speech_ratio=speech_ratio,
                # Emotion results
                emotions=emotions,
                dominant_emotion=dominant_emotion,
                vad_scores=vad_scores,
                # Speaker embeddings
                speaker_embeddings=context["speaker_embeddings"],
                # Metadata
                duration_ms=duration_ms,
                language=context["language"],
                asr_backend=context["asr_backend"],
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
    asr_backend: str = "whisperx",
    enable_vad: bool = True,
    enable_emotion: bool = True,
    extract_embeddings: bool = True,
) -> PipelineOrchestrator:
    """
    Create a configured pipeline orchestrator.

    Args:
        language: Language code for ASR (None for auto-detect)
        min_speakers: Minimum expected speakers for diarization
        max_speakers: Maximum expected speakers for diarization
        asr_backend: ASR backend - 'whisperx' (default), 'canary' (accuracy), 'parakeet' (speed)
        enable_vad: Enable Silero VAD for voice activity detection
        enable_emotion: Enable Emotion2Vec for speech emotion recognition
        extract_embeddings: Enable ECAPA-TDNN speaker embeddings

    Returns:
        Configured PipelineOrchestrator instance
    """
    config = PipelineConfig(
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        # ASR
        asr_backend=asr_backend,
        whisper_model=settings.whisper_model,
        # Diarization
        pyannote_model=settings.pyannote_model,
        speaker_embedding_model=settings.speaker_embedding_model,
        extract_speaker_embeddings=extract_embeddings,
        # Emotion
        emotion_model=settings.emotion_model,
        enable_emotion_detection=enable_emotion,
        # VAD
        enable_vad=enable_vad,
        # LLM
        llm_model=settings.llm_model,
        llm_provider=settings.llm_provider,
        # Signal detection
        signal_confidence_threshold=settings.signal_confidence_threshold,
    )
    return PipelineOrchestrator(config)
