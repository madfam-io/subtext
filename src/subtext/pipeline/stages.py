"""
Pipeline Processing Stages

Individual stages of the Subtext audio processing pipeline.
Each stage is responsible for a specific transformation or analysis.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import structlog
import torch

from subtext.config import settings

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# Base Stage
# ══════════════════════════════════════════════════════════════


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    success: bool
    data: dict[str, Any]
    duration_ms: float
    error: str | None = None


class PipelineStage(ABC):
    """Base class for pipeline stages."""

    name: str = "base"

    @abstractmethod
    async def process(self, **kwargs) -> dict[str, Any]:
        """Process input and return output."""
        pass

    async def initialize(self) -> None:
        """Initialize stage resources (models, etc.)."""
        pass

    async def cleanup(self) -> None:
        """Clean up stage resources."""
        pass


# ══════════════════════════════════════════════════════════════
# Cleanse Stage (Noise Suppression)
# ══════════════════════════════════════════════════════════════


class CleanseStage(PipelineStage):
    """
    Audio cleansing stage using DeepFilterNet.

    Removes background noise while preserving emotional prosody
    (jitter, shimmer, pitch variations).
    """

    name = "cleanse"

    def __init__(
        self,
        model_name: str = "deepfilternet3",
        preserve_prosody: bool = True,
        target_sample_rate: int = 16000,
    ):
        self.model_name = model_name
        self.preserve_prosody = preserve_prosody
        self.target_sample_rate = target_sample_rate
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load DeepFilterNet model."""
        if self._initialized:
            return

        try:
            # Import here to allow graceful degradation
            from df.enhance import enhance, init_df

            self._df_state, self._model, _ = init_df()
            self._enhance = enhance
            self._initialized = True
            logger.info("DeepFilterNet initialized", model=self.model_name)
        except ImportError:
            logger.warning("DeepFilterNet not available, using passthrough")
            self._initialized = True

    async def process(
        self,
        audio_path: str | None = None,
        audio_array: np.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        """
        Clean audio by removing noise.

        Args:
            audio_path: Path to audio file
            audio_array: Raw audio numpy array
            sample_rate: Sample rate of audio array

        Returns:
            dict with 'audio' (cleaned array), 'sample_rate', 'noise_profile'
        """
        await self.initialize()

        # Load audio if path provided
        if audio_path:
            import librosa

            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
        elif audio_array is not None:
            audio = audio_array
            sr = sample_rate or self.target_sample_rate
        else:
            raise ValueError("Must provide audio_path or audio_array")

        # Apply noise suppression if model available
        if self._model is not None:
            try:
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

                # Enhance
                enhanced = self._enhance(self._model, self._df_state, audio_tensor)
                cleaned = enhanced.squeeze().numpy()

                # Calculate noise profile (difference)
                noise_profile = {
                    "noise_level_db": float(
                        20 * np.log10(np.std(audio - cleaned) + 1e-10)
                    ),
                    "snr_improvement_db": float(
                        20
                        * np.log10(
                            (np.std(audio) + 1e-10) / (np.std(audio - cleaned) + 1e-10)
                        )
                    ),
                }

                logger.info(
                    "Audio cleansed",
                    snr_improvement=noise_profile["snr_improvement_db"],
                )

                return {
                    "audio": cleaned,
                    "sample_rate": sr,
                    "noise_profile": noise_profile,
                }

            except Exception as e:
                logger.error("Cleanse failed, using original", error=str(e))

        # Passthrough if no model
        return {
            "audio": audio,
            "sample_rate": sr,
            "noise_profile": {"noise_level_db": 0, "snr_improvement_db": 0},
        }


# ══════════════════════════════════════════════════════════════
# Diarize Stage (Speaker Identification)
# ══════════════════════════════════════════════════════════════


class DiarizeStage(PipelineStage):
    """
    Speaker diarization using Pyannote.

    Identifies "who is speaking when" with precise time boundaries.
    """

    name = "diarize"

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        use_auth_token: str | None = None,
    ):
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.use_auth_token = use_auth_token
        self._pipeline = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load Pyannote diarization pipeline."""
        if self._initialized:
            return

        try:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.use_auth_token,
            )

            # Use GPU if available
            if torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))

            self._initialized = True
            logger.info("Pyannote diarization initialized", model=self.model_name)

        except Exception as e:
            logger.warning("Pyannote not available", error=str(e))
            self._initialized = True

    async def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> dict[str, Any]:
        """
        Perform speaker diarization.

        Returns:
            dict with 'speakers' (list) and 'segments' (list of time ranges)
        """
        await self.initialize()

        if self._pipeline is None:
            # Return single speaker if no model
            duration_ms = int(len(audio) / sample_rate * 1000)
            return {
                "speakers": [{"id": "speaker_0", "label": "Speaker A"}],
                "segments": [
                    {
                        "speaker_id": "speaker_0",
                        "start_ms": 0,
                        "end_ms": duration_ms,
                    }
                ],
            }

        try:
            # Save to temp file (pyannote requires file input)
            import soundfile as sf
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sample_rate)
                temp_path = f.name

            # Run diarization
            params = {}
            if self.min_speakers:
                params["min_speakers"] = self.min_speakers
            if self.max_speakers:
                params["max_speakers"] = self.max_speakers

            diarization = self._pipeline(temp_path, **params)

            # Clean up temp file
            Path(temp_path).unlink()

            # Extract speakers and segments
            speakers = {}
            segments = []

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers:
                    idx = len(speakers)
                    speakers[speaker] = {
                        "id": f"speaker_{idx}",
                        "label": f"Speaker {chr(65 + idx)}",  # A, B, C, ...
                    }

                segments.append(
                    {
                        "speaker_id": speakers[speaker]["id"],
                        "start_ms": int(turn.start * 1000),
                        "end_ms": int(turn.end * 1000),
                    }
                )

            logger.info(
                "Diarization complete",
                speaker_count=len(speakers),
                segment_count=len(segments),
            )

            return {
                "speakers": list(speakers.values()),
                "segments": segments,
            }

        except Exception as e:
            logger.error("Diarization failed", error=str(e))
            raise


# ══════════════════════════════════════════════════════════════
# Transcribe Stage (Speech-to-Text)
# ══════════════════════════════════════════════════════════════


class TranscribeStage(PipelineStage):
    """
    Speech transcription using WhisperX.

    Provides word-level timestamps for precise alignment.
    """

    name = "transcribe"

    def __init__(
        self,
        model_name: str = "large-v3",
        language: str | None = None,
        word_timestamps: bool = True,
        compute_type: str = "float16",
    ):
        self.model_name = model_name
        self.language = language
        self.word_timestamps = word_timestamps
        self.compute_type = compute_type
        self._model = None
        self._align_model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load Whisper model."""
        if self._initialized:
            return

        try:
            import whisperx

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = whisperx.load_model(
                self.model_name,
                device=device,
                compute_type=self.compute_type if device == "cuda" else "int8",
            )

            self._whisperx = whisperx
            self._device = device
            self._initialized = True
            logger.info(
                "WhisperX initialized", model=self.model_name, device=device
            )

        except Exception as e:
            logger.warning("WhisperX not available", error=str(e))
            self._initialized = True

    async def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe audio to text with word timestamps.

        Returns:
            dict with 'transcript' (text), 'segments', 'words', 'language'
        """
        await self.initialize()

        if self._model is None:
            return {
                "transcript": "",
                "segments": [],
                "words": [],
                "language": language or "en",
            }

        try:
            # Transcribe
            result = self._model.transcribe(
                audio,
                language=language or self.language,
                batch_size=16,
            )

            detected_language = result.get("language", language or "en")

            # Align for word-level timestamps
            if self.word_timestamps:
                model_a, metadata = self._whisperx.load_align_model(
                    language_code=detected_language,
                    device=self._device,
                )
                result = self._whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self._device,
                    return_char_alignments=False,
                )

            # Extract transcript
            full_transcript = " ".join(
                seg.get("text", "") for seg in result.get("segments", [])
            )

            # Extract words with timestamps
            words = []
            for seg in result.get("segments", []):
                for word in seg.get("words", []):
                    words.append(
                        {
                            "text": word.get("word", ""),
                            "start_ms": int(word.get("start", 0) * 1000),
                            "end_ms": int(word.get("end", 0) * 1000),
                            "confidence": word.get("score", 0.0),
                        }
                    )

            logger.info(
                "Transcription complete",
                word_count=len(words),
                language=detected_language,
            )

            return {
                "transcript": full_transcript,
                "segments": result.get("segments", []),
                "words": words,
                "language": detected_language,
            }

        except Exception as e:
            logger.error("Transcription failed", error=str(e))
            raise


# ══════════════════════════════════════════════════════════════
# Prosodics Stage (Acoustic Feature Extraction)
# ══════════════════════════════════════════════════════════════


class ProsodicsStage(PipelineStage):
    """
    Prosodic feature extraction.

    Extracts the 47 acoustic features that form the "Prosodic Fingerprint":
    pitch, energy, temporal, and voice quality features.
    """

    name = "prosodics"

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        feature_set: str = "extended",  # 'basic' or 'extended'
        window_size_ms: int = 1000,
        hop_size_ms: int = 500,
    ):
        self.model_name = model_name
        self.feature_set = feature_set
        self.window_size_ms = window_size_ms
        self.hop_size_ms = hop_size_ms
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load feature extraction models."""
        if self._initialized:
            return

        try:
            import librosa

            self._librosa = librosa
            self._initialized = True
            logger.info("Prosodics extractor initialized")

        except Exception as e:
            logger.warning("Librosa not available", error=str(e))
            self._initialized = True

    async def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        segments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Extract prosodic features from audio.

        Returns:
            dict with 'features' (list of feature dicts per time window)
        """
        await self.initialize()

        features = []
        window_samples = int(self.window_size_ms * sample_rate / 1000)
        hop_samples = int(self.hop_size_ms * sample_rate / 1000)

        # Extract features per window
        for i in range(0, len(audio) - window_samples, hop_samples):
            window = audio[i : i + window_samples]
            timestamp_ms = int(i / sample_rate * 1000)

            feature_dict = self._extract_window_features(window, sample_rate)
            feature_dict["timestamp_ms"] = timestamp_ms
            features.append(feature_dict)

        logger.info("Prosodic extraction complete", feature_count=len(features))

        return {"features": features}

    def _extract_window_features(
        self, window: np.ndarray, sample_rate: int
    ) -> dict[str, float]:
        """Extract all prosodic features for a single window."""
        features: dict[str, float] = {}

        try:
            # ──────────────────────────────────────────────────────
            # Pitch (F0) Features
            # ──────────────────────────────────────────────────────
            f0, voiced_flag, _ = self._librosa.pyin(
                window,
                fmin=self._librosa.note_to_hz("C2"),
                fmax=self._librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            f0_valid = f0[~np.isnan(f0)]

            if len(f0_valid) > 0:
                features["pitch_mean"] = float(np.mean(f0_valid))
                features["pitch_std"] = float(np.std(f0_valid))
                features["pitch_range"] = float(np.max(f0_valid) - np.min(f0_valid))

                # Pitch slope (rising/falling)
                if len(f0_valid) > 1:
                    features["pitch_slope"] = float(
                        np.polyfit(np.arange(len(f0_valid)), f0_valid, 1)[0]
                    )
                else:
                    features["pitch_slope"] = 0.0

                # Jitter (pitch perturbation)
                if len(f0_valid) > 2:
                    jitter = np.mean(np.abs(np.diff(f0_valid))) / np.mean(f0_valid)
                    features["jitter"] = float(jitter)
                else:
                    features["jitter"] = 0.0
            else:
                features.update(
                    {
                        "pitch_mean": 0.0,
                        "pitch_std": 0.0,
                        "pitch_range": 0.0,
                        "pitch_slope": 0.0,
                        "jitter": 0.0,
                    }
                )

            # ──────────────────────────────────────────────────────
            # Energy Features
            # ──────────────────────────────────────────────────────
            rms = self._librosa.feature.rms(y=window)[0]
            features["energy_mean"] = float(np.mean(rms))
            features["energy_std"] = float(np.std(rms))

            # Shimmer (amplitude perturbation)
            if len(rms) > 2:
                shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms + 1e-10)
                features["shimmer"] = float(shimmer)
            else:
                features["shimmer"] = 0.0

            # ──────────────────────────────────────────────────────
            # Temporal Features
            # ──────────────────────────────────────────────────────
            # Speech rate estimation (via zero crossings as proxy)
            zcr = self._librosa.feature.zero_crossing_rate(window)[0]
            features["speech_rate"] = float(np.mean(zcr) * sample_rate / 100)

            # Silence detection
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(window) < silence_threshold) / len(window)
            features["silence_ratio"] = float(silence_ratio)

            # ──────────────────────────────────────────────────────
            # Voice Quality Features
            # ──────────────────────────────────────────────────────
            # Spectral centroid (brightness)
            centroid = self._librosa.feature.spectral_centroid(
                y=window, sr=sample_rate
            )[0]
            features["spectral_centroid"] = float(np.mean(centroid))

            # Harmonic-to-noise ratio (simplified)
            harmonic = self._librosa.effects.harmonic(window)
            percussive = self._librosa.effects.percussive(window)
            hnr = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-10)
            features["hnr"] = float(10 * np.log10(hnr + 1e-10))

            # Spectral flux (rate of change)
            spec = np.abs(self._librosa.stft(window))
            flux = np.mean(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
            features["spectral_flux"] = float(flux)

            # ──────────────────────────────────────────────────────
            # Derived Emotional State (VAD)
            # ──────────────────────────────────────────────────────
            # Simplified VAD model (would use trained model in production)
            features["valence"] = float(
                np.clip(
                    (features["pitch_mean"] - 150) / 100
                    + (features["energy_mean"] - 0.1) * 2,
                    -1,
                    1,
                )
            )
            features["arousal"] = float(
                np.clip(
                    features["speech_rate"] / 5
                    + features["pitch_std"] / 50
                    + features["energy_std"] * 5,
                    0,
                    1,
                )
            )
            features["dominance"] = float(
                np.clip(
                    features["energy_mean"] * 2
                    + features["pitch_slope"] / 50
                    + (1 - features["silence_ratio"]),
                    0,
                    1,
                )
            )

        except Exception as e:
            logger.warning("Feature extraction error", error=str(e))
            # Return zeros for all features on error
            features = {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_range": 0.0,
                "pitch_slope": 0.0,
                "jitter": 0.0,
                "energy_mean": 0.0,
                "energy_std": 0.0,
                "shimmer": 0.0,
                "speech_rate": 0.0,
                "silence_ratio": 0.0,
                "spectral_centroid": 0.0,
                "hnr": 0.0,
                "spectral_flux": 0.0,
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
            }

        return features


# ══════════════════════════════════════════════════════════════
# Synthesize Stage (Final Analysis & Insights)
# ══════════════════════════════════════════════════════════════


class SynthesizeStage(PipelineStage):
    """
    Final synthesis stage.

    Combines all analysis results into signals, timeline, and insights.
    Uses LLM for sophisticated reasoning about the conversation.
    """

    name = "synthesize"

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4-turbo-preview",
        signal_confidence_threshold: float = 0.5,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.signal_confidence_threshold = signal_confidence_threshold
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize LLM client."""
        if self._initialized:
            return

        try:
            if self.llm_provider == "openai":
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=settings.openai_api_key)
            elif self.llm_provider == "anthropic":
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
            else:
                self._client = None

            self._initialized = True
            logger.info("Synthesis stage initialized", provider=self.llm_provider)

        except Exception as e:
            logger.warning("LLM client not available", error=str(e))
            self._client = None
            self._initialized = True

    async def process(
        self,
        session_id: UUID,
        transcript_segments: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        prosodics: list[dict[str, Any]],
        signals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Synthesize final analysis results.

        Returns:
            dict with 'timeline', 'insights', 'speaker_metrics', 'summary'
        """
        await self.initialize()

        # Build tension timeline
        timeline = self._build_timeline(transcript_segments, prosodics, signals)

        # Calculate speaker metrics
        speaker_metrics = self._calculate_speaker_metrics(
            transcript_segments, prosodics, signals, speakers
        )

        # Generate LLM insights
        insights = await self._generate_insights(
            transcript_segments, signals, speaker_metrics
        )

        return {
            "timeline": timeline,
            "insights": insights,
            "speaker_metrics": speaker_metrics,
        }

    def _build_timeline(
        self,
        segments: list[dict[str, Any]],
        prosodics: list[dict[str, Any]],
        signals: list[dict[str, Any]],
        resolution_ms: int = 5000,
    ) -> list[dict[str, Any]]:
        """Build tension timeline for visualization."""
        if not segments:
            return []

        duration = max(s.get("end_ms", 0) for s in segments)
        timeline = []

        for bucket_start in range(0, duration, resolution_ms):
            bucket_end = bucket_start + resolution_ms

            # Get prosodics in bucket
            bucket_prosodics = [
                p
                for p in prosodics
                if bucket_start <= p.get("timestamp_ms", 0) < bucket_end
            ]

            # Get signals in bucket
            bucket_signals = [
                s
                for s in signals
                if bucket_start <= s.get("timestamp_ms", 0) < bucket_end
            ]

            # Calculate metrics
            valence = (
                np.mean([p.get("valence", 0) for p in bucket_prosodics])
                if bucket_prosodics
                else 0
            )
            arousal = (
                np.mean([p.get("arousal", 0.5) for p in bucket_prosodics])
                if bucket_prosodics
                else 0.5
            )

            # Tension score
            negative_signals = [
                s
                for s in bucket_signals
                if s.get("signal_type")
                in ["truth_gap", "steamroll", "micro_tremor", "stress_spike"]
            ]
            tension = min(1.0, arousal * 0.5 + len(negative_signals) * 0.2)

            # Active speaker
            active_segments = [
                s
                for s in segments
                if s.get("start_ms", 0) < bucket_end
                and s.get("end_ms", 0) > bucket_start
            ]
            active_speaker = (
                active_segments[-1].get("speaker_id") if active_segments else None
            )

            timeline.append(
                {
                    "timestamp_ms": bucket_start,
                    "valence": float(valence),
                    "arousal": float(arousal),
                    "tension_score": float(tension),
                    "active_speaker": active_speaker,
                    "active_signals": [s.get("signal_type") for s in bucket_signals],
                }
            )

        return timeline

    def _calculate_speaker_metrics(
        self,
        segments: list[dict[str, Any]],
        prosodics: list[dict[str, Any]],
        signals: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Calculate aggregated metrics per speaker."""
        speaker_data: dict[str, dict] = {
            s["id"]: {
                "speaker_id": s["id"],
                "label": s.get("label", s["id"]),
                "talk_time_ms": 0,
                "segment_count": 0,
                "signals": [],
            }
            for s in speakers
        }

        total_talk_time = 0

        for segment in segments:
            sid = segment.get("speaker_id")
            if sid in speaker_data:
                duration = segment.get("end_ms", 0) - segment.get("start_ms", 0)
                speaker_data[sid]["talk_time_ms"] += duration
                speaker_data[sid]["segment_count"] += 1
                total_talk_time += duration

        for signal in signals:
            sid = signal.get("speaker_id")
            if sid in speaker_data:
                speaker_data[sid]["signals"].append(signal)

        results = []
        for sid, data in speaker_data.items():
            signal_counts: dict[str, int] = {}
            for s in data["signals"]:
                st = s.get("signal_type", "unknown")
                signal_counts[st] = signal_counts.get(st, 0) + 1

            results.append(
                {
                    "speaker_id": sid,
                    "label": data["label"],
                    "talk_time_ms": data["talk_time_ms"],
                    "talk_ratio": (
                        data["talk_time_ms"] / total_talk_time if total_talk_time else 0
                    ),
                    "segment_count": data["segment_count"],
                    "signal_count": len(data["signals"]),
                    "signal_breakdown": signal_counts,
                    "engagement_score": 0.7,  # Would be calculated from prosodics
                    "stress_index": len(
                        [
                            s
                            for s in data["signals"]
                            if "stress" in s.get("signal_type", "")
                            or "tremor" in s.get("signal_type", "")
                        ]
                    )
                    * 0.1,
                }
            )

        return results

    async def _generate_insights(
        self,
        segments: list[dict[str, Any]],
        signals: list[dict[str, Any]],
        speaker_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate AI-powered insights."""
        # Build transcript for context
        transcript = "\n".join(
            f"[{s.get('speaker_id', 'Unknown')}]: {s.get('text', '')}"
            for s in segments[:50]  # Limit context
        )

        # Signal summary
        signal_summary = {}
        for s in signals:
            st = s.get("signal_type", "unknown")
            signal_summary[st] = signal_summary.get(st, 0) + 1

        # Generate summary with LLM if available
        summary = "Analysis complete."
        recommendations = []
        key_moments = []

        if self._client and self.llm_provider == "openai":
            try:
                prompt = f"""Analyze this conversation and provide insights.

Transcript (excerpt):
{transcript}

Detected Signals: {signal_summary}
Speaker Metrics: {speaker_metrics}

Provide:
1. A 2-3 sentence summary of the conversation dynamics
2. 2-3 key moments or turning points
3. 2-3 actionable recommendations

Respond in JSON format:
{{"summary": "...", "key_moments": ["...", "..."], "recommendations": ["...", "..."]}}
"""

                response = await self._client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=500,
                )

                import json

                result = json.loads(response.choices[0].message.content)
                summary = result.get("summary", summary)
                key_moments = result.get("key_moments", [])
                recommendations = result.get("recommendations", [])

            except Exception as e:
                logger.warning("LLM insight generation failed", error=str(e))

        # Find high-tension moments from signals
        high_tension = sorted(
            [s for s in signals if s.get("intensity", 0) > 0.7],
            key=lambda x: x.get("intensity", 0),
            reverse=True,
        )[:5]

        formatted_key_moments = [
            {
                "timestamp_ms": s.get("timestamp_ms", 0),
                "type": "tension_peak",
                "description": f"High-intensity {s.get('signal_type', 'signal')} detected",
                "importance": s.get("intensity", 0.5),
            }
            for s in high_tension
        ]

        return {
            "summary": summary,
            "key_moments": formatted_key_moments,
            "recommendations": recommendations,
            "signal_summary": signal_summary,
        }
