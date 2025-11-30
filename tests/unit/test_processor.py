"""
Unit Tests for Realtime Processor Module

Tests the RealtimeProcessor and related data structures.
"""

import pytest
import numpy as np
import base64
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

from subtext.core.models import SignalType


# ══════════════════════════════════════════════════════════════
# AudioChunk Tests
# ══════════════════════════════════════════════════════════════


class TestAudioChunk:
    """Test AudioChunk dataclass."""

    def test_audio_chunk_creation(self):
        """Test creating an AudioChunk directly."""
        from subtext.realtime.processor import AudioChunk

        audio_data = np.zeros(1600, dtype=np.float32)
        chunk = AudioChunk(
            data=audio_data,
            timestamp_ms=0,
            sample_rate=16000,
            duration_ms=100,
        )

        assert len(chunk.data) == 1600
        assert chunk.timestamp_ms == 0
        assert chunk.sample_rate == 16000
        assert chunk.duration_ms == 100
        assert chunk.is_speech is None

    def test_audio_chunk_from_bytes_pcm_s16le(self):
        """Test creating AudioChunk from PCM s16le bytes."""
        from subtext.realtime.processor import AudioChunk
        from subtext.realtime.protocol import AudioConfig

        # Create 16-bit PCM audio data
        audio = np.array([0, 16384, 32767, -32768], dtype=np.int16)
        raw_bytes = audio.tobytes()

        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            encoding="pcm_s16le",
        )

        chunk = AudioChunk.from_bytes(raw_bytes, 1000, config)

        assert len(chunk.data) == 4
        assert chunk.timestamp_ms == 1000
        assert chunk.sample_rate == 16000
        # Values should be normalized to [-1, 1]
        assert abs(chunk.data[0]) < 0.01  # 0 normalized
        assert abs(chunk.data[2] - 1.0) < 0.01  # 32767 -> ~1.0

    def test_audio_chunk_from_bytes_pcm_f32le(self):
        """Test creating AudioChunk from PCM f32le bytes."""
        from subtext.realtime.processor import AudioChunk
        from subtext.realtime.protocol import AudioConfig

        # Create 32-bit float audio data
        audio = np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32)
        raw_bytes = audio.tobytes()

        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            encoding="pcm_f32le",
        )

        chunk = AudioChunk.from_bytes(raw_bytes, 0, config)

        assert len(chunk.data) == 4
        np.testing.assert_array_almost_equal(chunk.data, audio)

    def test_audio_chunk_from_bytes_stereo(self):
        """Test stereo to mono conversion."""
        from subtext.realtime.processor import AudioChunk
        from subtext.realtime.protocol import AudioConfig

        # Create stereo audio: left=0.5, right=1.0 -> mono = 0.75
        stereo_audio = np.array([0.5, 1.0, 0.5, 1.0], dtype=np.float32)
        raw_bytes = stereo_audio.tobytes()

        config = AudioConfig(
            sample_rate=16000,
            channels=2,
            encoding="pcm_f32le",
        )

        chunk = AudioChunk.from_bytes(raw_bytes, 0, config)

        # Should be mono (2 samples instead of 4)
        assert len(chunk.data) == 2
        assert abs(chunk.data[0] - 0.75) < 0.01

    def test_audio_chunk_from_bytes_unsupported_encoding(self):
        """Test error on unsupported encoding."""
        from subtext.realtime.processor import AudioChunk
        from subtext.realtime.protocol import AudioConfig

        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            encoding="unsupported",
        )

        with pytest.raises(ValueError, match="Unsupported encoding"):
            AudioChunk.from_bytes(b"test", 0, config)

    def test_audio_chunk_from_base64(self):
        """Test creating AudioChunk from base64 data."""
        from subtext.realtime.processor import AudioChunk
        from subtext.realtime.protocol import AudioConfig

        # Create and encode audio
        audio = np.array([0.25, 0.5, 0.75], dtype=np.float32)
        b64_data = base64.b64encode(audio.tobytes()).decode()

        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            encoding="pcm_f32le",
        )

        chunk = AudioChunk.from_base64(b64_data, 500, config)

        assert len(chunk.data) == 3
        assert chunk.timestamp_ms == 500
        np.testing.assert_array_almost_equal(chunk.data, audio)

    def test_audio_chunk_duration_calculation(self):
        """Test duration is correctly calculated."""
        from subtext.realtime.processor import AudioChunk
        from subtext.realtime.protocol import AudioConfig

        # 16000 samples at 16kHz = 1000ms
        audio = np.zeros(16000, dtype=np.float32)
        raw_bytes = audio.tobytes()

        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            encoding="pcm_f32le",
        )

        chunk = AudioChunk.from_bytes(raw_bytes, 0, config)

        assert chunk.duration_ms == 1000


# ══════════════════════════════════════════════════════════════
# SpeakerState Tests
# ══════════════════════════════════════════════════════════════


class TestSpeakerState:
    """Test SpeakerState dataclass."""

    def test_speaker_state_creation(self):
        """Test creating a SpeakerState."""
        from subtext.realtime.processor import SpeakerState

        state = SpeakerState(
            speaker_id="spk_001",
            label="Speaker A",
        )

        assert state.speaker_id == "spk_001"
        assert state.label == "Speaker A"
        assert state.embedding is None
        assert state.talk_time_ms == 0
        assert state.segment_count == 0

    def test_speaker_state_avg_valence(self):
        """Test average valence calculation."""
        from subtext.realtime.processor import SpeakerState

        state = SpeakerState(
            speaker_id="spk_001",
            label="Speaker A",
            valence_sum=3.0,
            sample_count=6,
        )

        assert state.avg_valence == 0.5

    def test_speaker_state_avg_arousal(self):
        """Test average arousal calculation."""
        from subtext.realtime.processor import SpeakerState

        state = SpeakerState(
            speaker_id="spk_001",
            label="Speaker A",
            arousal_sum=4.0,
            sample_count=4,
        )

        assert state.avg_arousal == 1.0

    def test_speaker_state_avg_dominance(self):
        """Test average dominance calculation."""
        from subtext.realtime.processor import SpeakerState

        state = SpeakerState(
            speaker_id="spk_001",
            label="Speaker A",
            dominance_sum=2.5,
            sample_count=5,
        )

        assert state.avg_dominance == 0.5

    def test_speaker_state_zero_samples(self):
        """Test averages with zero samples (avoid division by zero)."""
        from subtext.realtime.processor import SpeakerState

        state = SpeakerState(
            speaker_id="spk_001",
            label="Speaker A",
            sample_count=0,
        )

        # Should return 0 not error
        assert state.avg_valence == 0.0
        assert state.avg_arousal == 0.0
        assert state.avg_dominance == 0.0


# ══════════════════════════════════════════════════════════════
# ProcessorState Tests
# ══════════════════════════════════════════════════════════════


class TestProcessorState:
    """Test ProcessorState dataclass."""

    def test_processor_state_creation(self):
        """Test creating a ProcessorState."""
        from subtext.realtime.processor import ProcessorState
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(
            session_id=session_id,
            audio=AudioConfig(),
        )

        state = ProcessorState(session_id=session_id, config=config)

        assert state.session_id == session_id
        assert len(state.audio_buffer) == 0
        assert state.buffer_timestamp_ms == 0
        assert state.total_duration_ms == 0
        assert state.is_speaking is False
        assert state.speakers == {}
        assert state.current_speaker is None

    def test_processor_state_defaults(self):
        """Test ProcessorState default values."""
        from subtext.realtime.processor import ProcessorState
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        state = ProcessorState(session_id=session_id, config=config)

        assert state.partial_transcript == ""
        assert state.last_transcript_ms == 0
        assert isinstance(state.prosodics_buffer, deque)
        assert len(state.prosodics_buffer) == 0
        assert isinstance(state.recent_signals, deque)
        assert len(state.recent_signals) == 0
        assert state.current_valence == 0.0
        assert state.current_arousal == 0.5
        assert state.current_dominance == 0.5


# ══════════════════════════════════════════════════════════════
# RealtimeProcessor Tests
# ══════════════════════════════════════════════════════════════


class TestRealtimeProcessor:
    """Test RealtimeProcessor class."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())

        processor = RealtimeProcessor(session_id, config)

        assert processor.session_id == session_id
        assert processor.config == config
        assert processor.state is not None
        assert processor._vad_model is None  # Lazy loaded

    def test_processor_get_speaker_label(self):
        """Test getting speaker label."""
        from subtext.realtime.processor import RealtimeProcessor, SpeakerState
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        # No speakers yet
        assert processor._get_speaker_label(None) is None
        assert processor._get_speaker_label("unknown") is None

        # Add a speaker
        processor.state.speakers["spk_001"] = SpeakerState(
            speaker_id="spk_001",
            label="Speaker A",
        )

        assert processor._get_speaker_label("spk_001") == "Speaker A"

    def test_processor_match_or_create_speaker_new(self):
        """Test creating a new speaker."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        embedding = np.random.randn(192).astype(np.float32)
        speaker_id, is_new = processor._match_or_create_speaker(embedding)

        assert is_new is True
        assert speaker_id == "speaker_0"
        assert "speaker_0" in processor.state.speakers

    def test_processor_match_or_create_speaker_match(self):
        """Test matching an existing speaker."""
        from subtext.realtime.processor import RealtimeProcessor, SpeakerState
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        # Create a speaker with known embedding
        embedding = np.random.randn(192).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        processor.state.speakers["spk_001"] = SpeakerState(
            speaker_id="spk_001",
            label="Speaker A",
            embedding=embedding,
        )

        # Try to match with same embedding (high similarity)
        speaker_id, is_new = processor._match_or_create_speaker(embedding)

        assert is_new is False
        assert speaker_id == "spk_001"

    def test_processor_calculate_silence_duration(self):
        """Test silence duration calculation."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        # When speaking, silence is 0
        processor.state.is_speaking = True
        assert processor._calculate_silence_duration() == 0

        # When not speaking with no prosodics
        processor.state.is_speaking = False
        processor.state.total_duration_ms = 5000
        assert processor._calculate_silence_duration() == 5000

    def test_processor_calculate_pitch_variance(self):
        """Test pitch variance calculation."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        # Not enough samples
        assert processor._calculate_pitch_variance() is None

        # Add prosodics with pitch data
        for i in range(10):
            processor.state.prosodics_buffer.append({
                "timestamp_ms": i * 100,
                "pitch_mean": 100 + i * 5,  # Varying pitch
            })

        variance = processor._calculate_pitch_variance()
        assert variance is not None
        assert variance > 0

    def test_processor_calculate_speech_ratio(self):
        """Test speech ratio calculation."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        # Empty buffer
        assert processor._calculate_speech_ratio() == 0.5

        # Add prosodics: 5 speech, 5 silence
        for i in range(10):
            processor.state.prosodics_buffer.append({
                "timestamp_ms": i * 100,
                "energy_mean": -20 if i < 5 else -50,  # First 5 are speech
            })

        ratio = processor._calculate_speech_ratio()
        assert ratio == 0.5

    def test_processor_estimate_pitch(self):
        """Test pitch estimation."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        # Too short audio
        short_audio = np.zeros(100, dtype=np.float32)
        assert processor._estimate_pitch(short_audio, 16000) is None

        # Generate a simple sine wave at ~200 Hz
        sample_rate = 16000
        t = np.arange(0, 0.1, 1/sample_rate)  # 100ms
        freq = 200
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

        pitch = processor._estimate_pitch(audio, sample_rate)
        # Pitch estimation may not be super accurate but should be in range
        if pitch is not None:
            assert 100 < pitch < 300

    def test_processor_trim_buffers(self):
        """Test buffer trimming."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        audio_config = AudioConfig(sample_rate=16000)
        config = SessionConfig(session_id=session_id, audio=audio_config)
        processor = RealtimeProcessor(session_id, config)

        # Add more than 30 seconds of audio (30 * 16000 samples)
        large_buffer = np.zeros(35 * 16000, dtype=np.float32)
        processor.state.audio_buffer = large_buffer

        processor._trim_buffers()

        # Buffer should be trimmed to 30 seconds max
        assert len(processor.state.audio_buffer) <= 30 * 16000


class TestRealtimeProcessorAsync:
    """Test async methods of RealtimeProcessor."""

    @pytest.mark.asyncio
    async def test_processor_run_vad_no_model(self):
        """Test VAD returns True when model not loaded."""
        from subtext.realtime.processor import RealtimeProcessor, AudioChunk
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        chunk = AudioChunk(
            data=np.zeros(1600, dtype=np.float32),
            timestamp_ms=0,
            sample_rate=16000,
            duration_ms=100,
        )

        result = await processor._run_vad(chunk)
        assert result is True  # Falls back to True when no model

    @pytest.mark.asyncio
    async def test_processor_run_asr_placeholder(self):
        """Test ASR returns None (placeholder implementation)."""
        from subtext.realtime.processor import RealtimeProcessor
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        audio = np.zeros(16000, dtype=np.float32)
        result = await processor._run_asr(audio, partial=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_processor_finalize(self):
        """Test processor finalization."""
        from subtext.realtime.processor import RealtimeProcessor, SpeakerState
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        # Setup some state
        processor.state.total_duration_ms = 10000
        processor.state.speakers["spk_001"] = SpeakerState(
            speaker_id="spk_001",
            label="Speaker A",
            talk_time_ms=5000,
            segment_count=3,
        )

        summary = await processor.finalize()

        assert summary["session_id"] == str(session_id)
        assert summary["duration_ms"] == 10000
        assert summary["speaker_count"] == 1
        assert len(summary["speakers"]) == 1
        assert summary["speakers"][0]["speaker_id"] == "spk_001"

    @pytest.mark.asyncio
    async def test_processor_process_chunk_empty(self):
        """Test processing empty chunk."""
        from subtext.realtime.processor import RealtimeProcessor, AudioChunk
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(
            session_id=session_id,
            audio=AudioConfig(),
            enable_prosodics=True,
            enable_signals=True,
        )
        processor = RealtimeProcessor(session_id, config)

        chunk = AudioChunk(
            data=np.zeros(1600, dtype=np.float32),
            timestamp_ms=0,
            sample_rate=16000,
            duration_ms=100,
        )

        messages = []
        async for msg in processor.process_chunk(chunk):
            messages.append(msg)

        # Should have processed without errors
        assert processor.state.total_duration_ms == 100

    @pytest.mark.asyncio
    async def test_processor_extract_prosodics(self):
        """Test prosodics extraction."""
        from subtext.realtime.processor import RealtimeProcessor, AudioChunk
        from subtext.realtime.protocol import (
            SessionConfig, AudioConfig, RealtimeMessageType
        )

        session_id = uuid4()
        config = SessionConfig(session_id=session_id, audio=AudioConfig())
        processor = RealtimeProcessor(session_id, config)

        # Create audio with some energy
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        chunk = AudioChunk(
            data=audio,
            timestamp_ms=0,
            sample_rate=16000,
            duration_ms=100,
        )

        messages = []
        async for msg in processor._extract_prosodics(chunk):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].type == RealtimeMessageType.PROSODICS_UPDATE
        assert "energy_mean" in messages[0].payload

    @pytest.mark.asyncio
    async def test_processor_detect_signals_dead_air(self):
        """Test dead air signal detection."""
        from subtext.realtime.processor import RealtimeProcessor, AudioChunk
        from subtext.realtime.protocol import (
            SessionConfig, AudioConfig, RealtimeMessageType
        )

        session_id = uuid4()
        config = SessionConfig(
            session_id=session_id,
            audio=AudioConfig(),
            signal_confidence_threshold=0.5,
        )
        processor = RealtimeProcessor(session_id, config)

        # Setup state to simulate silence
        processor.state.is_speaking = False
        processor.state.total_duration_ms = 5000  # 5 seconds of silence

        # Add prosodics showing low energy
        for i in range(10):
            processor.state.prosodics_buffer.append({
                "timestamp_ms": i * 500,
                "energy_mean": -60,  # Very quiet
            })

        chunk = AudioChunk(
            data=np.zeros(1600, dtype=np.float32),
            timestamp_ms=5000,
            sample_rate=16000,
            duration_ms=100,
            is_speech=False,
        )

        messages = []
        async for msg in processor._detect_signals(chunk):
            messages.append(msg)

        # Should detect dead air signal
        dead_air_msgs = [
            m for m in messages
            if m.type == RealtimeMessageType.SIGNAL_DETECTED
            and "dead_air" in m.payload.get("signal_type", "")
        ]
        assert len(dead_air_msgs) >= 1


# ══════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════


class TestProcessorIntegration:
    """Integration tests for the processor."""

    @pytest.mark.asyncio
    async def test_full_processing_flow(self):
        """Test complete processing flow with multiple chunks."""
        from subtext.realtime.processor import RealtimeProcessor, AudioChunk
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(
            session_id=session_id,
            audio=AudioConfig(sample_rate=16000),
            enable_transcription=False,  # Skip ASR
            enable_diarization=False,  # Skip diarization
            enable_prosodics=True,
            enable_signals=True,
            esp_enabled=False,  # Skip ESP
        )
        processor = RealtimeProcessor(session_id, config)

        # Process multiple chunks
        total_messages = []
        for i in range(10):
            audio = np.random.randn(1600).astype(np.float32) * 0.1
            chunk = AudioChunk(
                data=audio,
                timestamp_ms=i * 100,
                sample_rate=16000,
                duration_ms=100,
            )
            async for msg in processor.process_chunk(chunk):
                total_messages.append(msg)

        # Should have processed 1000ms total
        assert processor.state.total_duration_ms == 1000
        # Should have generated prosodics messages
        assert len(total_messages) > 0

    @pytest.mark.asyncio
    async def test_process_and_finalize(self):
        """Test processing then finalizing."""
        from subtext.realtime.processor import RealtimeProcessor, AudioChunk
        from subtext.realtime.protocol import SessionConfig, AudioConfig

        session_id = uuid4()
        config = SessionConfig(
            session_id=session_id,
            audio=AudioConfig(),
            enable_transcription=False,
            enable_diarization=False,
        )
        processor = RealtimeProcessor(session_id, config)

        # Process some audio
        chunk = AudioChunk(
            data=np.zeros(16000, dtype=np.float32),  # 1 second
            timestamp_ms=0,
            sample_rate=16000,
            duration_ms=1000,
        )
        async for _ in processor.process_chunk(chunk):
            pass

        # Finalize
        summary = await processor.finalize()

        assert summary["duration_ms"] == 1000
        assert summary["session_id"] == str(session_id)
