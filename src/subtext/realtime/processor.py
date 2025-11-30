"""
Realtime Pipeline Processor

Handles streaming audio processing with incremental analysis results.
"""

import asyncio
import base64
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import numpy as np
import structlog

from subtext.core.models import SignalType, ESPMessage
from .protocol import (
    SessionConfig,
    AudioConfig,
    RealtimeMessage,
    RealtimeMessageType,
    TranscriptPayload,
    SignalPayload,
    SpeakerPayload,
    ProsodicsPayload,
    TimelinePayload,
)

logger = structlog.get_logger()


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""

    data: np.ndarray
    timestamp_ms: int
    sample_rate: int
    duration_ms: int
    is_speech: bool | None = None

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        timestamp_ms: int,
        config: AudioConfig,
    ) -> "AudioChunk":
        """Create AudioChunk from raw bytes."""
        if config.encoding == "pcm_s16le":
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        elif config.encoding == "pcm_f32le":
            audio = np.frombuffer(data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported encoding: {config.encoding}")

        # Handle stereo -> mono conversion
        if config.channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        duration_ms = int(len(audio) / config.sample_rate * 1000)

        return cls(
            data=audio,
            timestamp_ms=timestamp_ms,
            sample_rate=config.sample_rate,
            duration_ms=duration_ms,
        )

    @classmethod
    def from_base64(
        cls,
        data: str,
        timestamp_ms: int,
        config: AudioConfig,
    ) -> "AudioChunk":
        """Create AudioChunk from base64-encoded bytes."""
        raw_bytes = base64.b64decode(data)
        return cls.from_bytes(raw_bytes, timestamp_ms, config)


@dataclass
class SpeakerState:
    """Tracks state for an identified speaker."""

    speaker_id: str
    label: str
    embedding: np.ndarray | None = None
    talk_time_ms: int = 0
    segment_count: int = 0
    last_active_ms: int = 0

    # Running emotional state
    valence_sum: float = 0.0
    arousal_sum: float = 0.0
    dominance_sum: float = 0.0
    sample_count: int = 0

    @property
    def avg_valence(self) -> float:
        return self.valence_sum / max(self.sample_count, 1)

    @property
    def avg_arousal(self) -> float:
        return self.arousal_sum / max(self.sample_count, 1)

    @property
    def avg_dominance(self) -> float:
        return self.dominance_sum / max(self.sample_count, 1)


@dataclass
class ProcessorState:
    """Holds the state of the realtime processor."""

    session_id: UUID
    config: SessionConfig

    # Audio buffer for processing
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    buffer_timestamp_ms: int = 0
    total_duration_ms: int = 0

    # Speech segments buffer for transcription
    speech_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    speech_start_ms: int = 0
    is_speaking: bool = False

    # Speakers
    speakers: dict[str, SpeakerState] = field(default_factory=dict)
    current_speaker: str | None = None

    # Partial transcription state
    partial_transcript: str = ""
    last_transcript_ms: int = 0

    # Prosodics window
    prosodics_buffer: deque = field(default_factory=lambda: deque(maxlen=100))

    # Signal history
    recent_signals: deque = field(default_factory=lambda: deque(maxlen=50))

    # ESP state
    last_esp_broadcast_ms: int = 0
    current_valence: float = 0.0
    current_arousal: float = 0.5
    current_dominance: float = 0.5
    current_engagement: float = 0.5
    current_stress: float = 0.0


class RealtimeProcessor:
    """
    Processes streaming audio in real-time.

    Handles:
    - Voice Activity Detection (VAD)
    - Incremental Speaker Diarization
    - Streaming ASR with partial results
    - Continuous Emotion Recognition
    - Real-time Signal Detection
    - ESP Message Generation
    """

    # Processing window sizes
    VAD_WINDOW_MS = 100
    PROSODICS_WINDOW_MS = 1000
    TRANSCRIBE_BUFFER_MS = 3000
    ESP_BROADCAST_INTERVAL_MS = 500

    def __init__(
        self,
        session_id: UUID,
        config: SessionConfig,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.state = ProcessorState(session_id=session_id, config=config)

        # ML models (lazy loaded)
        self._vad_model = None
        self._diarization_model = None
        self._asr_model = None
        self._emotion_model = None
        self._embedding_model = None

        # Processing locks
        self._process_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()

        logger.info(
            "RealtimeProcessor initialized",
            session_id=str(session_id),
            asr_backend=config.asr_backend,
        )

    async def initialize(self) -> None:
        """Initialize ML models for processing."""
        async with self._model_lock:
            try:
                # Load VAD model (Silero)
                if self.config.enable_transcription or self.config.enable_signals:
                    await self._load_vad_model()

                # Load embedding model for speaker identification
                if self.config.enable_diarization:
                    await self._load_embedding_model()

                # Note: ASR and emotion models are loaded on-demand
                # to reduce memory footprint

                logger.info(
                    "RealtimeProcessor models loaded",
                    session_id=str(self.session_id),
                )
            except Exception as e:
                logger.error(
                    "Failed to load models",
                    session_id=str(self.session_id),
                    error=str(e),
                )

    async def _load_vad_model(self) -> None:
        """Load Silero VAD model."""
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._vad_model = model
            self._vad_utils = utils
            logger.debug("Silero VAD model loaded")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}")
            self._vad_model = None

    async def _load_embedding_model(self) -> None:
        """Load speaker embedding model."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self._embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/tmp/speechbrain_models",
                run_opts={"device": "cpu"},
            )
            logger.debug("Speaker embedding model loaded")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self._embedding_model = None

    async def process_chunk(
        self,
        chunk: AudioChunk,
    ) -> AsyncIterator[RealtimeMessage]:
        """
        Process an audio chunk and yield analysis messages.

        This is the main entry point for streaming audio processing.
        """
        async with self._process_lock:
            # Update state
            self.state.audio_buffer = np.concatenate(
                [self.state.audio_buffer, chunk.data]
            )
            self.state.total_duration_ms += chunk.duration_ms

            # Run VAD
            is_speech = await self._run_vad(chunk)
            chunk.is_speech = is_speech

            # Handle speech/silence transitions
            async for msg in self._handle_speech_state(chunk, is_speech):
                yield msg

            # Run prosodics extraction
            if self.config.enable_prosodics:
                async for msg in self._extract_prosodics(chunk):
                    yield msg

            # Check for signals
            if self.config.enable_signals:
                async for msg in self._detect_signals(chunk):
                    yield msg

            # Generate ESP update if interval elapsed
            if self.config.esp_enabled:
                async for msg in self._generate_esp():
                    yield msg

            # Generate timeline update
            async for msg in self._generate_timeline():
                yield msg

            # Trim buffers to prevent memory growth
            self._trim_buffers()

    async def _run_vad(self, chunk: AudioChunk) -> bool:
        """Run Voice Activity Detection on chunk."""
        if self._vad_model is None:
            # Fallback: assume all audio is speech
            return True

        try:
            import torch

            # Resample if needed (Silero expects 16kHz)
            audio = chunk.data
            if chunk.sample_rate != 16000:
                # Simple resampling - in production use proper resampler
                ratio = 16000 / chunk.sample_rate
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio), new_length),
                    np.arange(len(audio)),
                    audio,
                )

            # Run VAD
            tensor = torch.from_numpy(audio).float()
            speech_prob = self._vad_model(tensor, 16000).item()

            return speech_prob > 0.5

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True  # Assume speech on error

    async def _handle_speech_state(
        self,
        chunk: AudioChunk,
        is_speech: bool,
    ) -> AsyncIterator[RealtimeMessage]:
        """Handle speech/silence state transitions."""
        if is_speech and not self.state.is_speaking:
            # Speech started
            self.state.is_speaking = True
            self.state.speech_start_ms = chunk.timestamp_ms
            self.state.speech_buffer = chunk.data.copy()

        elif is_speech and self.state.is_speaking:
            # Continued speech - accumulate buffer
            self.state.speech_buffer = np.concatenate(
                [self.state.speech_buffer, chunk.data]
            )

            # Check if we should emit partial transcript
            buffer_duration_ms = int(
                len(self.state.speech_buffer)
                / self.config.audio.sample_rate
                * 1000
            )

            if (
                self.config.enable_transcription
                and self.config.asr_partial_results
                and buffer_duration_ms >= self.TRANSCRIBE_BUFFER_MS
            ):
                async for msg in self._transcribe_partial():
                    yield msg

        elif not is_speech and self.state.is_speaking:
            # Speech ended - finalize transcript
            self.state.is_speaking = False

            if self.config.enable_transcription:
                async for msg in self._transcribe_final():
                    yield msg

            # Identify speaker from the speech segment
            if self.config.enable_diarization:
                async for msg in self._identify_speaker():
                    yield msg

            # Clear speech buffer
            self.state.speech_buffer = np.array([], dtype=np.float32)

    async def _transcribe_partial(self) -> AsyncIterator[RealtimeMessage]:
        """Generate partial transcription result."""
        if len(self.state.speech_buffer) == 0:
            return

        try:
            # For now, yield a placeholder - in production, use streaming ASR
            text = await self._run_asr(self.state.speech_buffer, partial=True)

            if text and text != self.state.partial_transcript:
                self.state.partial_transcript = text

                yield RealtimeMessage(
                    type=RealtimeMessageType.TRANSCRIPT_PARTIAL,
                    session_id=self.session_id,
                    payload=TranscriptPayload(
                        text=text,
                        speaker_id=self.state.current_speaker,
                        speaker_label=self._get_speaker_label(self.state.current_speaker),
                        start_ms=self.state.speech_start_ms,
                        end_ms=self.state.total_duration_ms,
                        is_final=False,
                    ).model_dump(),
                )

        except Exception as e:
            logger.error(f"Partial transcription error: {e}")

    async def _transcribe_final(self) -> AsyncIterator[RealtimeMessage]:
        """Generate final transcription result."""
        if len(self.state.speech_buffer) == 0:
            return

        try:
            text = await self._run_asr(self.state.speech_buffer, partial=False)

            if text:
                end_ms = self.state.total_duration_ms

                yield RealtimeMessage(
                    type=RealtimeMessageType.TRANSCRIPT_FINAL,
                    session_id=self.session_id,
                    payload=TranscriptPayload(
                        text=text,
                        speaker_id=self.state.current_speaker,
                        speaker_label=self._get_speaker_label(self.state.current_speaker),
                        start_ms=self.state.speech_start_ms,
                        end_ms=end_ms,
                        confidence=0.9,  # Would come from ASR
                        is_final=True,
                    ).model_dump(),
                )

                self.state.partial_transcript = ""
                self.state.last_transcript_ms = end_ms

        except Exception as e:
            logger.error(f"Final transcription error: {e}")

    async def _run_asr(
        self,
        audio: np.ndarray,
        partial: bool = False,
    ) -> str | None:
        """Run ASR on audio segment."""
        # Placeholder - in production, integrate with ASR backend
        # For streaming, would use models with streaming support
        return None

    async def _identify_speaker(self) -> AsyncIterator[RealtimeMessage]:
        """Identify or create speaker from speech segment."""
        if self._embedding_model is None or len(self.state.speech_buffer) == 0:
            return

        try:
            import torch

            # Extract embedding
            tensor = torch.from_numpy(self.state.speech_buffer).unsqueeze(0)
            embedding = self._embedding_model.encode_batch(tensor).squeeze().numpy()

            # Find closest existing speaker or create new one
            speaker_id, is_new = self._match_or_create_speaker(embedding)
            self.state.current_speaker = speaker_id

            if is_new:
                yield RealtimeMessage(
                    type=RealtimeMessageType.SPEAKER_IDENTIFIED,
                    session_id=self.session_id,
                    payload=SpeakerPayload(
                        speaker_id=speaker_id,
                        speaker_label=self._get_speaker_label(speaker_id),
                        is_new=True,
                    ).model_dump(),
                )

        except Exception as e:
            logger.error(f"Speaker identification error: {e}")

    def _match_or_create_speaker(
        self,
        embedding: np.ndarray,
    ) -> tuple[str, bool]:
        """Match embedding to existing speaker or create new one."""
        SIMILARITY_THRESHOLD = 0.7

        best_match = None
        best_similarity = 0.0

        for speaker_id, speaker in self.state.speakers.items():
            if speaker.embedding is not None:
                # Cosine similarity
                similarity = np.dot(embedding, speaker.embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(speaker.embedding)
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_id

        if best_match and best_similarity >= SIMILARITY_THRESHOLD:
            return best_match, False

        # Create new speaker
        speaker_num = len(self.state.speakers)
        speaker_id = f"speaker_{speaker_num}"
        label = chr(ord("A") + speaker_num) if speaker_num < 26 else f"Speaker {speaker_num + 1}"

        self.state.speakers[speaker_id] = SpeakerState(
            speaker_id=speaker_id,
            label=f"Speaker {label}",
            embedding=embedding,
        )

        return speaker_id, True

    def _get_speaker_label(self, speaker_id: str | None) -> str | None:
        """Get display label for speaker."""
        if speaker_id and speaker_id in self.state.speakers:
            return self.state.speakers[speaker_id].label
        return None

    async def _extract_prosodics(
        self,
        chunk: AudioChunk,
    ) -> AsyncIterator[RealtimeMessage]:
        """Extract prosodic features from audio chunk."""
        try:
            # Basic prosodic feature extraction
            # In production, use more sophisticated analysis

            audio = chunk.data
            if len(audio) == 0:
                return

            # Energy
            energy = np.sqrt(np.mean(audio ** 2))
            energy_db = 20 * np.log10(max(energy, 1e-10))

            # Zero crossing rate (rough pitch proxy)
            zcr = np.mean(np.abs(np.diff(np.signbit(audio))))

            # Simple pitch estimation using autocorrelation
            pitch = self._estimate_pitch(audio, chunk.sample_rate)

            # Map to emotional state (simplified)
            valence = (pitch - 150) / 100 if pitch else 0  # Higher pitch -> positive
            arousal = min(1.0, (energy_db + 40) / 40)  # Louder -> aroused
            dominance = 0.5  # Would need more context

            # Update running state
            alpha = 0.3  # Smoothing factor
            self.state.current_valence = (
                alpha * valence + (1 - alpha) * self.state.current_valence
            )
            self.state.current_arousal = (
                alpha * arousal + (1 - alpha) * self.state.current_arousal
            )

            # Store in buffer
            prosodics = {
                "timestamp_ms": chunk.timestamp_ms,
                "pitch_mean": pitch,
                "energy_mean": energy_db,
                "zcr": zcr,
                "valence": self.state.current_valence,
                "arousal": self.state.current_arousal,
                "dominance": self.state.current_dominance,
            }
            self.state.prosodics_buffer.append(prosodics)

            # Yield prosodics update
            yield RealtimeMessage(
                type=RealtimeMessageType.PROSODICS_UPDATE,
                session_id=self.session_id,
                payload=ProsodicsPayload(
                    timestamp_ms=chunk.timestamp_ms,
                    speaker_id=self.state.current_speaker,
                    pitch_mean=pitch,
                    energy_mean=energy_db,
                    valence=self.state.current_valence,
                    arousal=self.state.current_arousal,
                    dominance=self.state.current_dominance,
                ).model_dump(),
            )

        except Exception as e:
            logger.error(f"Prosodics extraction error: {e}")

    def _estimate_pitch(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float | None:
        """Estimate pitch using autocorrelation."""
        if len(audio) < sample_rate // 50:  # Need at least 20ms
            return None

        try:
            # Autocorrelation
            corr = np.correlate(audio, audio, mode="full")
            corr = corr[len(corr) // 2:]

            # Find peaks
            min_lag = sample_rate // 500  # Max 500 Hz
            max_lag = sample_rate // 50   # Min 50 Hz

            if max_lag >= len(corr):
                return None

            segment = corr[min_lag:max_lag]
            if len(segment) == 0:
                return None

            peak_idx = np.argmax(segment) + min_lag
            if peak_idx == 0:
                return None

            pitch = sample_rate / peak_idx
            return float(pitch) if 50 <= pitch <= 500 else None

        except Exception:
            return None

    async def _detect_signals(
        self,
        chunk: AudioChunk,
    ) -> AsyncIterator[RealtimeMessage]:
        """Detect signals from the Signal Atlas."""
        try:
            signals_detected = []

            # Dead air detection
            if not chunk.is_speech:
                silence_duration = self._calculate_silence_duration()
                if silence_duration >= 3000:  # 3 seconds
                    signals_detected.append({
                        "type": SignalType.DEAD_AIR,
                        "confidence": min(1.0, silence_duration / 5000),
                        "intensity": min(1.0, silence_duration / 10000),
                    })

            # Stress spike detection (high arousal + negative valence)
            if (
                self.state.current_arousal > 0.7
                and self.state.current_valence < -0.3
            ):
                signals_detected.append({
                    "type": SignalType.STRESS_SPIKE,
                    "confidence": self.state.current_arousal,
                    "intensity": abs(self.state.current_valence),
                })

            # Enthusiasm surge (high arousal + positive valence)
            if (
                self.state.current_arousal > 0.7
                and self.state.current_valence > 0.3
            ):
                signals_detected.append({
                    "type": SignalType.ENTHUSIASM_SURGE,
                    "confidence": self.state.current_arousal,
                    "intensity": self.state.current_valence,
                })

            # Monotone detection (low pitch variance over time)
            pitch_variance = self._calculate_pitch_variance()
            if pitch_variance is not None and pitch_variance < 10:
                signals_detected.append({
                    "type": SignalType.MONOTONE,
                    "confidence": max(0.5, 1 - pitch_variance / 20),
                    "intensity": max(0.3, 1 - pitch_variance / 30),
                })

            # Yield detected signals
            for signal_data in signals_detected:
                if signal_data["confidence"] >= self.config.signal_confidence_threshold:
                    signal = SignalPayload(
                        signal_type=signal_data["type"].value,
                        timestamp_ms=chunk.timestamp_ms,
                        confidence=signal_data["confidence"],
                        intensity=signal_data["intensity"],
                        speaker_id=self.state.current_speaker,
                        speaker_label=self._get_speaker_label(self.state.current_speaker),
                    )

                    # Track signal
                    self.state.recent_signals.append({
                        "type": signal_data["type"],
                        "timestamp_ms": chunk.timestamp_ms,
                        "confidence": signal_data["confidence"],
                    })

                    yield RealtimeMessage(
                        type=RealtimeMessageType.SIGNAL_DETECTED,
                        session_id=self.session_id,
                        payload=signal.model_dump(),
                    )

        except Exception as e:
            logger.error(f"Signal detection error: {e}")

    def _calculate_silence_duration(self) -> int:
        """Calculate current silence duration in ms."""
        if self.state.is_speaking:
            return 0

        # Find last speech timestamp
        for prosodic in reversed(list(self.state.prosodics_buffer)):
            if prosodic.get("energy_mean", -60) > -30:  # Active speech threshold
                return self.state.total_duration_ms - prosodic["timestamp_ms"]

        return self.state.total_duration_ms

    def _calculate_pitch_variance(self) -> float | None:
        """Calculate pitch variance over recent prosodics window."""
        pitches = [
            p["pitch_mean"]
            for p in self.state.prosodics_buffer
            if p.get("pitch_mean") is not None
        ]

        if len(pitches) < 5:
            return None

        return float(np.std(pitches))

    async def _generate_esp(self) -> AsyncIterator[RealtimeMessage]:
        """Generate ESP (Emotional State Protocol) update."""
        # Check if enough time has passed
        elapsed = self.state.total_duration_ms - self.state.last_esp_broadcast_ms
        if elapsed < self.config.esp_interval_ms:
            return

        try:
            # Gather active signals
            active_signals = [
                {"type": s["type"].value, "confidence": s["confidence"]}
                for s in self.state.recent_signals
                if self.state.total_duration_ms - s["timestamp_ms"] < 5000
            ]

            # Calculate engagement (based on speech activity and arousal)
            speech_ratio = self._calculate_speech_ratio()
            engagement = (speech_ratio + self.state.current_arousal) / 2

            # Calculate stress (based on negative valence and high arousal)
            stress = max(
                0,
                self.state.current_arousal - self.state.current_valence
            ) / 2

            # Update state
            self.state.current_engagement = engagement
            self.state.current_stress = stress
            self.state.last_esp_broadcast_ms = self.state.total_duration_ms

            yield RealtimeMessage(
                type=RealtimeMessageType.ESP_UPDATE,
                session_id=self.session_id,
                payload={
                    "version": "1.0",
                    "speaker_id": self.state.current_speaker,
                    "speaker_label": self._get_speaker_label(self.state.current_speaker),
                    "valence": round(self.state.current_valence, 3),
                    "arousal": round(self.state.current_arousal, 3),
                    "dominance": round(self.state.current_dominance, 3),
                    "signals": active_signals,
                    "engagement_score": round(engagement, 3),
                    "stress_index": round(stress, 3),
                    "consent_level": self.config.esp_consent_level,
                },
            )

        except Exception as e:
            logger.error(f"ESP generation error: {e}")

    def _calculate_speech_ratio(self) -> float:
        """Calculate ratio of speech time in recent window."""
        if not self.state.prosodics_buffer:
            return 0.5

        speech_samples = sum(
            1 for p in self.state.prosodics_buffer
            if p.get("energy_mean", -60) > -30
        )

        return speech_samples / len(self.state.prosodics_buffer)

    async def _generate_timeline(self) -> AsyncIterator[RealtimeMessage]:
        """Generate timeline update for visualization."""
        # Emit timeline point every second
        if self.state.total_duration_ms % 1000 != 0:
            return

        try:
            # Tension score: combination of arousal and negative valence
            tension = (
                self.state.current_arousal * 0.5
                + max(0, -self.state.current_valence) * 0.5
            )

            # Get active signals
            active_signal_types = list(set(
                s["type"].value
                for s in self.state.recent_signals
                if self.state.total_duration_ms - s["timestamp_ms"] < 2000
            ))

            yield RealtimeMessage(
                type=RealtimeMessageType.TIMELINE_UPDATE,
                session_id=self.session_id,
                payload=TimelinePayload(
                    timestamp_ms=self.state.total_duration_ms,
                    valence=round(self.state.current_valence, 3),
                    arousal=round(self.state.current_arousal, 3),
                    tension_score=round(tension, 3),
                    active_speaker=self._get_speaker_label(self.state.current_speaker),
                    active_signals=active_signal_types,
                ).model_dump(),
            )

        except Exception as e:
            logger.error(f"Timeline generation error: {e}")

    def _trim_buffers(self) -> None:
        """Trim audio buffers to prevent memory growth."""
        MAX_BUFFER_SAMPLES = self.config.audio.sample_rate * 30  # 30 seconds

        if len(self.state.audio_buffer) > MAX_BUFFER_SAMPLES:
            trim_samples = len(self.state.audio_buffer) - MAX_BUFFER_SAMPLES
            self.state.audio_buffer = self.state.audio_buffer[trim_samples:]
            self.state.buffer_timestamp_ms += int(
                trim_samples / self.config.audio.sample_rate * 1000
            )

    async def finalize(self) -> dict[str, Any]:
        """
        Finalize processing and return session summary.

        Called when the realtime session ends.
        """
        # Flush any remaining speech buffer
        if self.state.is_speaking and len(self.state.speech_buffer) > 0:
            self.state.is_speaking = False
            # Would process final transcript here

        # Generate summary
        summary = {
            "session_id": str(self.session_id),
            "duration_ms": self.state.total_duration_ms,
            "speaker_count": len(self.state.speakers),
            "speakers": [
                {
                    "speaker_id": s.speaker_id,
                    "label": s.label,
                    "talk_time_ms": s.talk_time_ms,
                    "segment_count": s.segment_count,
                    "avg_valence": s.avg_valence,
                    "avg_arousal": s.avg_arousal,
                    "avg_dominance": s.avg_dominance,
                }
                for s in self.state.speakers.values()
            ],
            "signal_count": len(self.state.recent_signals),
            "signals": list(self.state.recent_signals),
        }

        logger.info(
            "RealtimeProcessor finalized",
            session_id=str(self.session_id),
            duration_ms=self.state.total_duration_ms,
            speaker_count=len(self.state.speakers),
        )

        return summary
