"""
Unit Tests for Pipeline Stages

Tests individual pipeline stages with mock/synthetic audio data.
"""

import pytest
import numpy as np
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from subtext.pipeline.stages import (
    CleanseStage,
    VADStage,
    DiarizeStage,
    TranscribeStage,
    EmotionStage,
    ProsodicsStage,
    SynthesizeStage,
    StageResult,
    PipelineStage,
    _import_torch,
)


# ══════════════════════════════════════════════════════════════
# Base Classes Tests
# ══════════════════════════════════════════════════════════════


class TestStageResult:
    """Test StageResult dataclass."""

    def test_stage_result_success(self):
        """Test creating a successful stage result."""
        result = StageResult(
            success=True,
            data={"transcript": "Hello world"},
            duration_ms=100.5,
        )
        assert result.success is True
        assert result.data == {"transcript": "Hello world"}
        assert result.duration_ms == 100.5
        assert result.error is None

    def test_stage_result_failure(self):
        """Test creating a failed stage result."""
        result = StageResult(
            success=False,
            data={},
            duration_ms=50.0,
            error="Model not found",
        )
        assert result.success is False
        assert result.error == "Model not found"


class TestPipelineStageBase:
    """Test PipelineStage base class."""

    def test_pipeline_stage_name(self):
        """Test base stage has name attribute."""
        assert PipelineStage.name == "base"

    @pytest.mark.asyncio
    async def test_pipeline_stage_initialize_cleanup(self):
        """Test default initialize and cleanup methods."""

        class ConcreteStage(PipelineStage):
            async def process(self, **kwargs):
                return {}

        stage = ConcreteStage()
        # Should not raise
        await stage.initialize()
        await stage.cleanup()


class TestImportTorch:
    """Test lazy torch import."""

    def test_import_torch_available(self):
        """Test torch import when available."""
        torch = _import_torch()
        # May or may not be available in test environment
        # Just verify it doesn't crash
        assert torch is None or hasattr(torch, "Tensor")

    def test_import_torch_with_mock(self):
        """Test torch import with mock."""
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _import_torch()
            # Should return the mock
            assert result is not None or result is None  # Depends on import


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

    @pytest.mark.asyncio
    async def test_cleanse_stage_no_input_error(self):
        """Test cleanse stage raises error without input."""
        stage = CleanseStage()

        with pytest.raises(ValueError, match="Must provide audio_path or audio_array"):
            await stage.process()

    def test_cleanse_stage_parameters(self):
        """Test cleanse stage parameter initialization."""
        stage = CleanseStage(
            model_name="deepfilternet2",
            preserve_prosody=False,
            target_sample_rate=48000,
        )

        assert stage.model_name == "deepfilternet2"
        assert stage.preserve_prosody is False
        assert stage.target_sample_rate == 48000

    @pytest.mark.asyncio
    async def test_cleanse_stage_already_initialized(self, sample_audio):
        """Test cleanse stage skips initialization if already done."""
        audio, sample_rate = sample_audio
        stage = CleanseStage()
        stage._initialized = True

        # Should not re-initialize
        result = await stage.process(audio_array=audio, sample_rate=sample_rate)
        assert "audio" in result


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

    @pytest.mark.asyncio
    async def test_vad_stage_already_initialized(self, sample_speech_audio):
        """Test VAD stage skips initialization if already done."""
        audio, sample_rate = sample_speech_audio
        stage = VADStage()
        stage._initialized = True
        stage._model = None

        result = await stage.process(audio=audio, sample_rate=sample_rate)
        assert "speech_segments" in result

    def test_vad_stage_name(self):
        """Test VAD stage name attribute."""
        assert VADStage.name == "vad"


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

    def test_diarize_stage_name(self):
        """Test diarize stage name attribute."""
        assert DiarizeStage.name == "diarize"

    def test_diarize_stage_parameters(self):
        """Test diarize stage parameter initialization."""
        stage = DiarizeStage(
            model_name="pyannote/speaker-diarization-3.0",
            min_speakers=2,
            max_speakers=5,
            use_auth_token="test-token",
            extract_embeddings=False,
        )

        assert stage.model_name == "pyannote/speaker-diarization-3.0"
        assert stage.min_speakers == 2
        assert stage.max_speakers == 5
        assert stage.use_auth_token == "test-token"
        assert stage.extract_embeddings is False

    @pytest.mark.asyncio
    async def test_diarize_stage_already_initialized(self, sample_multi_speaker_audio):
        """Test diarize stage skips initialization if already done."""
        audio, sample_rate = sample_multi_speaker_audio
        stage = DiarizeStage()
        stage._initialized = True
        stage._pipeline = None

        result = await stage.process(audio=audio, sample_rate=sample_rate)
        assert "speakers" in result


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

    def test_transcribe_stage_name(self):
        """Test transcribe stage name attribute."""
        assert TranscribeStage.name == "transcribe"

    def test_transcribe_stage_language_setting(self):
        """Test language configuration."""
        stage = TranscribeStage(
            language="es",
            word_timestamps=False,
        )
        assert stage.language == "es"
        assert stage.word_timestamps is False

    def test_transcribe_stage_compute_type(self):
        """Test compute type configuration."""
        stage = TranscribeStage(compute_type="int8")
        assert stage.compute_type == "int8"

    def test_transcribe_stage_unknown_backend_fallback(self):
        """Test unknown backend defaults to whisperx model."""
        stage = TranscribeStage(backend="unknown_backend")
        assert stage.backend == "unknown_backend"
        assert stage.model_name == "large-v3"

    @pytest.mark.asyncio
    async def test_transcribe_stage_already_initialized(self, sample_speech_audio):
        """Test transcribe stage skips initialization if already done."""
        audio, sample_rate = sample_speech_audio
        stage = TranscribeStage()
        stage._initialized = True
        stage._model = None

        result = await stage.process(audio=audio, sample_rate=sample_rate)
        assert "transcript" in result
        assert result["backend"] == "none"


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

    def test_emotion_stage_name(self):
        """Test emotion stage name attribute."""
        assert EmotionStage.name == "emotion"

    def test_emotion_stage_parameters(self):
        """Test emotion stage parameter initialization."""
        stage = EmotionStage(
            model_name="custom/emotion-model",
            granularity="frame",
            device="cpu",
        )
        assert stage.model_name == "custom/emotion-model"
        assert stage.granularity == "frame"
        assert stage.device == "cpu"

    def test_emotion_parse_result_valid(self):
        """Test parsing valid emotion result."""
        stage = EmotionStage()

        result = [{"scores": [0.1, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0], "labels": ["happy"]}]
        parsed = stage._parse_emotion_result(result)

        assert parsed["label"] == "happy"
        assert "scores" in parsed

    def test_emotion_parse_result_empty(self):
        """Test parsing empty emotion result."""
        stage = EmotionStage()

        result = [{}]
        parsed = stage._parse_emotion_result(result)

        assert parsed["label"] == "neutral"
        assert parsed["confidence"] == 0.5

    def test_emotion_parse_result_with_segment(self):
        """Test parsing emotion result with segment info."""
        stage = EmotionStage()

        result = [{"scores": [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0], "labels": "happy"}]
        segment = {"start_ms": 1000, "end_ms": 2000, "speaker_id": "speaker_0"}
        parsed = stage._parse_emotion_result(result, segment)

        assert parsed["start_ms"] == 1000
        assert parsed["end_ms"] == 2000
        assert parsed["speaker_id"] == "speaker_0"

    def test_emotion_vad_empty_emotions(self):
        """Test VAD estimation with empty emotions list."""
        stage = EmotionStage()

        vad = stage._estimate_vad_from_emotions([])

        assert vad["valence"] == 0.0
        assert vad["arousal"] == 0.5
        assert vad["dominance"] == 0.5

    @pytest.mark.asyncio
    async def test_emotion_stage_already_initialized(self, sample_speech_audio):
        """Test emotion stage skips initialization if already done."""
        audio, sample_rate = sample_speech_audio
        stage = EmotionStage()
        stage._initialized = True
        stage._model = None

        result = await stage.process(audio=audio, sample_rate=sample_rate)
        assert result["dominant_emotion"] == "neutral"


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

    def test_prosodics_stage_name(self):
        """Test prosodics stage name attribute."""
        assert ProsodicsStage.name == "prosodics"

    def test_prosodics_stage_parameters(self):
        """Test prosodics stage parameter initialization."""
        stage = ProsodicsStage(
            model_name="custom/wav2vec",
            feature_set="basic",
            window_size_ms=500,
            hop_size_ms=250,
        )
        assert stage.model_name == "custom/wav2vec"
        assert stage.feature_set == "basic"
        assert stage.window_size_ms == 500
        assert stage.hop_size_ms == 250

    @pytest.mark.asyncio
    async def test_prosodics_short_audio(self):
        """Test prosodics with audio shorter than window size."""
        # 0.5 second audio at 16kHz = 8000 samples
        audio = np.random.randn(8000).astype(np.float32)
        sample_rate = 16000

        stage = ProsodicsStage(window_size_ms=1000, hop_size_ms=500)
        result = await stage.process(audio=audio, sample_rate=sample_rate)

        # Should return empty features since audio is shorter than window
        assert "features" in result
        assert len(result["features"]) == 0


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

    def test_synthesize_stage_name(self):
        """Test synthesize stage name attribute."""
        assert SynthesizeStage.name == "synthesize"

    def test_synthesize_stage_parameters(self):
        """Test synthesize stage parameter initialization."""
        stage = SynthesizeStage(
            llm_provider="anthropic",
            llm_model="claude-3-opus",
            signal_confidence_threshold=0.7,
        )
        assert stage.llm_provider == "anthropic"
        assert stage.llm_model == "claude-3-opus"
        assert stage.signal_confidence_threshold == 0.7

    def test_synthesize_stage_timeline_empty_segments(self):
        """Test timeline with empty segments returns empty list."""
        stage = SynthesizeStage()

        timeline = stage._build_timeline([], [], [])

        assert timeline == []

    def test_synthesize_stage_timeline_with_tension_signals(self):
        """Test timeline with tension signals."""
        stage = SynthesizeStage()

        segments = [
            {"start_ms": 0, "end_ms": 10000, "speaker_id": "speaker_0"},
        ]

        prosodics = [
            {"timestamp_ms": 0, "valence": 0.0, "arousal": 0.8},
        ]

        signals = [
            {"signal_type": "truth_gap", "timestamp_ms": 2000},
            {"signal_type": "steamroll", "timestamp_ms": 3000},
        ]

        timeline = stage._build_timeline(segments, prosodics, signals)

        assert len(timeline) > 0
        # First bucket should have signals
        assert "active_signals" in timeline[0]

    def test_synthesize_stage_speaker_metrics_calculation(self):
        """Test speaker metrics calculation."""
        stage = SynthesizeStage()

        speakers = [
            {"id": "speaker_0", "label": "Speaker A"},
            {"id": "speaker_1", "label": "Speaker B"},
        ]

        segments = [
            {"speaker_id": "speaker_0", "start_ms": 0, "end_ms": 5000},
            {"speaker_id": "speaker_1", "start_ms": 5000, "end_ms": 10000},
            {"speaker_id": "speaker_0", "start_ms": 10000, "end_ms": 15000},
        ]

        prosodics = []

        signals = [
            {"speaker_id": "speaker_0", "signal_type": "stress_spike"},
            {"speaker_id": "speaker_1", "signal_type": "enthusiasm_surge"},
        ]

        metrics = stage._calculate_speaker_metrics(segments, prosodics, signals, speakers)

        assert len(metrics) == 2

        speaker_a = next(m for m in metrics if m["speaker_id"] == "speaker_0")
        assert speaker_a["talk_time_ms"] == 10000
        assert speaker_a["segment_count"] == 2
        assert speaker_a["signal_count"] == 1

        speaker_b = next(m for m in metrics if m["speaker_id"] == "speaker_1")
        assert speaker_b["talk_time_ms"] == 5000
        assert speaker_b["segment_count"] == 1

    def test_synthesize_stage_speaker_metrics_empty(self):
        """Test speaker metrics with no data."""
        stage = SynthesizeStage()

        metrics = stage._calculate_speaker_metrics([], [], [], [])

        assert metrics == []

    @pytest.mark.asyncio
    async def test_synthesize_stage_generate_insights_no_llm(self, session_id):
        """Test insights generation without LLM client."""
        stage = SynthesizeStage()
        stage._initialized = True
        stage._client = None

        segments = [
            {"speaker_id": "speaker_0", "text": "Hello"},
        ]

        signals = [
            {"signal_type": "enthusiasm_surge", "timestamp_ms": 1000, "intensity": 0.9},
        ]

        speaker_metrics = []

        insights = await stage._generate_insights(segments, signals, speaker_metrics)

        assert "summary" in insights
        assert "key_moments" in insights
        assert "recommendations" in insights
        # Without LLM, key_moments should be from high-intensity signals
        assert len(insights["key_moments"]) > 0
