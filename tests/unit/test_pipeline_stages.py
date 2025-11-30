"""
Unit Tests for Pipeline Stages

Tests individual pipeline stages with mock/synthetic audio data.
"""

import pytest
import numpy as np
from uuid import uuid4

from subtext.pipeline.stages import (
    CleanseStage,
    VADStage,
    DiarizeStage,
    TranscribeStage,
    EmotionStage,
    ProsodicsStage,
    SynthesizeStage,
)


# ══════════════════════════════════════════════════════════════
# Cleanse Stage Tests
# ══════════════════════════════════════════════════════════════


class TestCleanseStage:
    """Tests for the CleanseStage (noise suppression)."""

    @pytest.mark.asyncio
    async def test_cleanse_stage_passthrough(self, sample_audio):
        """Test cleanse stage passes through audio when model unavailable."""
        audio, sample_rate = sample_audio

        stage = CleanseStage(preserve_prosody=True)
        # Don't initialize model - test passthrough behavior

        result = await stage.process(audio_array=audio, sample_rate=sample_rate)

        assert "audio" in result
        assert "sample_rate" in result
        assert "noise_profile" in result
        assert result["sample_rate"] == sample_rate
        assert len(result["audio"]) == len(audio)

    @pytest.mark.asyncio
    async def test_cleanse_stage_from_path(self, tmp_path, sample_audio):
        """Test cleanse stage can load audio from file path."""
        import soundfile as sf

        audio, sample_rate = sample_audio
        audio_path = tmp_path / "test.wav"
        sf.write(str(audio_path), audio, sample_rate)

        stage = CleanseStage()
        result = await stage.process(audio_path=str(audio_path))

        assert "audio" in result
        assert len(result["audio"]) > 0


# ══════════════════════════════════════════════════════════════
# VAD Stage Tests
# ══════════════════════════════════════════════════════════════


class TestVADStage:
    """Tests for the VADStage (voice activity detection)."""

    @pytest.mark.asyncio
    async def test_vad_stage_fallback(self, sample_speech_audio):
        """Test VAD stage returns full audio when model unavailable."""
        audio, sample_rate = sample_speech_audio

        stage = VADStage(threshold=0.5)

        result = await stage.process(audio=audio, sample_rate=sample_rate)

        assert "speech_segments" in result
        assert "speech_ratio" in result
        assert "total_speech_ms" in result

        # Without model, should return full audio as speech
        if stage._model is None:
            assert result["speech_ratio"] == 1.0
            assert len(result["speech_segments"]) == 1

    @pytest.mark.asyncio
    async def test_vad_stage_parameters(self):
        """Test VAD stage accepts various parameters."""
        stage = VADStage(
            threshold=0.7,
            min_speech_duration_ms=500,
            min_silence_duration_ms=200,
            speech_pad_ms=50,
        )

        assert stage.threshold == 0.7
        assert stage.min_speech_duration_ms == 500
        assert stage.min_silence_duration_ms == 200
        assert stage.speech_pad_ms == 50


# ══════════════════════════════════════════════════════════════
# Diarize Stage Tests
# ══════════════════════════════════════════════════════════════


class TestDiarizeStage:
    """Tests for the DiarizeStage (speaker diarization)."""

    @pytest.mark.asyncio
    async def test_diarize_stage_fallback(self, sample_multi_speaker_audio):
        """Test diarize stage returns single speaker when model unavailable."""
        audio, sample_rate = sample_multi_speaker_audio

        stage = DiarizeStage(extract_embeddings=False)

        result = await stage.process(audio=audio, sample_rate=sample_rate)

        assert "speakers" in result
        assert "segments" in result
        assert "embeddings" in result

        # Without model, returns single speaker
        if stage._pipeline is None:
            assert len(result["speakers"]) == 1
            assert result["speakers"][0]["label"] == "Speaker A"

    @pytest.mark.asyncio
    async def test_diarize_stage_with_embeddings(self):
        """Test diarize stage embedding extraction config."""
        stage = DiarizeStage(
            embedding_model="speechbrain/spkrec-ecapa-voxceleb",
            extract_embeddings=True,
        )

        assert stage.extract_embeddings is True
        assert stage.embedding_model == "speechbrain/spkrec-ecapa-voxceleb"


# ══════════════════════════════════════════════════════════════
# Transcribe Stage Tests
# ══════════════════════════════════════════════════════════════


class TestTranscribeStage:
    """Tests for the TranscribeStage (ASR)."""

    @pytest.mark.asyncio
    async def test_transcribe_stage_fallback(self, sample_speech_audio):
        """Test transcribe stage returns empty when model unavailable."""
        audio, sample_rate = sample_speech_audio

        stage = TranscribeStage(backend="whisperx")

        result = await stage.process(audio=audio, sample_rate=sample_rate)

        assert "transcript" in result
        assert "segments" in result
        assert "words" in result
        assert "language" in result
        assert "backend" in result

    def test_transcribe_stage_backend_selection(self):
        """Test transcribe stage backend selection."""
        # WhisperX backend
        stage_whisperx = TranscribeStage(backend="whisperx")
        assert stage_whisperx.backend == "whisperx"

        # Canary backend
        stage_canary = TranscribeStage(backend="canary")
        assert stage_canary.backend == "canary"

        # Parakeet backend
        stage_parakeet = TranscribeStage(backend="parakeet")
        assert stage_parakeet.backend == "parakeet"

    def test_transcribe_stage_model_name_override(self):
        """Test custom model name override."""
        stage = TranscribeStage(
            backend="whisperx",
            model_name="medium",
        )
        assert stage.model_name == "medium"


# ══════════════════════════════════════════════════════════════
# Emotion Stage Tests
# ══════════════════════════════════════════════════════════════


class TestEmotionStage:
    """Tests for the EmotionStage (speech emotion recognition)."""

    @pytest.mark.asyncio
    async def test_emotion_stage_fallback(self, sample_speech_audio):
        """Test emotion stage returns neutral when model unavailable."""
        audio, sample_rate = sample_speech_audio

        stage = EmotionStage()

        result = await stage.process(audio=audio, sample_rate=sample_rate)

        assert "emotions" in result
        assert "dominant_emotion" in result
        assert "emotion_timeline" in result
        assert "vad" in result

        # Without model, returns neutral
        if stage._model is None:
            assert result["dominant_emotion"] == "neutral"

    def test_emotion_stage_vad_estimation(self):
        """Test VAD estimation from emotions."""
        stage = EmotionStage()

        emotions = [
            {"label": "happy", "confidence": 0.8},
            {"label": "neutral", "confidence": 0.6},
        ]

        vad = stage._estimate_vad_from_emotions(emotions)

        assert "valence" in vad
        assert "arousal" in vad
        assert "dominance" in vad
        assert -1.0 <= vad["valence"] <= 1.0
        assert 0.0 <= vad["arousal"] <= 1.0
        assert 0.0 <= vad["dominance"] <= 1.0

    def test_emotion_labels(self):
        """Test all emotion labels are defined."""
        expected_labels = [
            "angry", "disgusted", "fearful", "happy",
            "neutral", "other", "sad", "surprised"
        ]
        assert EmotionStage.EMOTION_LABELS == expected_labels


# ══════════════════════════════════════════════════════════════
# Prosodics Stage Tests
# ══════════════════════════════════════════════════════════════


class TestProsodicsStage:
    """Tests for the ProsodicsStage (acoustic features)."""

    @pytest.mark.asyncio
    async def test_prosodics_stage_extraction(self, sample_speech_audio):
        """Test prosodic feature extraction."""
        audio, sample_rate = sample_speech_audio

        stage = ProsodicsStage(
            window_size_ms=1000,
            hop_size_ms=500,
        )

        result = await stage.process(audio=audio, sample_rate=sample_rate)

        assert "features" in result
        features = result["features"]

        # Should have multiple feature windows
        assert len(features) > 0

        # Check feature structure
        first_feature = features[0]
        assert "timestamp_ms" in first_feature
        assert "pitch_mean" in first_feature
        assert "energy_mean" in first_feature
        assert "valence" in first_feature
        assert "arousal" in first_feature
        assert "dominance" in first_feature

    def test_prosodics_stage_window_calculation(self):
        """Test window size calculation."""
        stage = ProsodicsStage(
            window_size_ms=2000,
            hop_size_ms=1000,
        )

        assert stage.window_size_ms == 2000
        assert stage.hop_size_ms == 1000


# ══════════════════════════════════════════════════════════════
# Synthesize Stage Tests
# ══════════════════════════════════════════════════════════════


class TestSynthesizeStage:
    """Tests for the SynthesizeStage (final analysis)."""

    @pytest.mark.asyncio
    async def test_synthesize_stage_basic(self, session_id):
        """Test synthesize stage with minimal input."""
        stage = SynthesizeStage(
            llm_provider="openai",
            signal_confidence_threshold=0.5,
        )

        result = await stage.process(
            session_id=session_id,
            transcript_segments=[],
            speakers=[],
            prosodics=[],
            signals=[],
        )

        assert "timeline" in result
        assert "insights" in result
        assert "speaker_metrics" in result

    @pytest.mark.asyncio
    async def test_synthesize_stage_with_data(self, session_id):
        """Test synthesize stage with sample data."""
        stage = SynthesizeStage()

        speakers = [
            {"id": "speaker_0", "label": "Speaker A"},
            {"id": "speaker_1", "label": "Speaker B"},
        ]

        segments = [
            {
                "speaker_id": "speaker_0",
                "start_ms": 0,
                "end_ms": 5000,
                "text": "Hello, how are you?",
            },
            {
                "speaker_id": "speaker_1",
                "start_ms": 5000,
                "end_ms": 10000,
                "text": "I'm doing well, thank you.",
            },
        ]

        prosodics = [
            {
                "timestamp_ms": 0,
                "valence": 0.5,
                "arousal": 0.6,
                "pitch_mean": 150,
            },
            {
                "timestamp_ms": 5000,
                "valence": 0.7,
                "arousal": 0.5,
                "pitch_mean": 200,
            },
        ]

        signals = [
            {
                "signal_type": "enthusiasm_surge",
                "timestamp_ms": 5000,
                "intensity": 0.8,
            },
        ]

        result = await stage.process(
            session_id=session_id,
            transcript_segments=segments,
            speakers=speakers,
            prosodics=prosodics,
            signals=signals,
        )

        assert len(result["speaker_metrics"]) == 2
        assert len(result["timeline"]) > 0

    def test_synthesize_stage_timeline_building(self):
        """Test timeline construction."""
        stage = SynthesizeStage()

        segments = [
            {"start_ms": 0, "end_ms": 5000, "speaker_id": "speaker_0"},
            {"start_ms": 5000, "end_ms": 10000, "speaker_id": "speaker_1"},
        ]

        prosodics = [
            {"timestamp_ms": 0, "valence": 0.5, "arousal": 0.6},
            {"timestamp_ms": 5000, "valence": 0.7, "arousal": 0.5},
        ]

        signals = []

        timeline = stage._build_timeline(segments, prosodics, signals)

        assert len(timeline) > 0
        for point in timeline:
            assert "timestamp_ms" in point
            assert "valence" in point
            assert "arousal" in point
            assert "tension_score" in point
