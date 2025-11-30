"""
Unit Tests for Pipeline Orchestrator Module

Tests the PipelineOrchestrator and related configurations.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from subtext.core.models import Signal, SignalType


# ══════════════════════════════════════════════════════════════
# PipelineConfig Tests
# ══════════════════════════════════════════════════════════════


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_config(self):
        """Test default PipelineConfig values."""
        from subtext.pipeline.orchestrator import PipelineConfig

        config = PipelineConfig()

        assert config.asr_backend == "whisperx"
        assert config.whisper_model == "large-v3"
        assert config.pyannote_model == "pyannote/speaker-diarization-3.1"
        assert config.enable_emotion_detection is True
        assert config.enable_vad is True
        assert config.signal_confidence_threshold == 0.5
        assert config.max_duration_seconds == 7200
        assert config.target_sample_rate == 16000

    def test_custom_config(self):
        """Test custom PipelineConfig values."""
        from subtext.pipeline.orchestrator import PipelineConfig

        config = PipelineConfig(
            asr_backend="canary",
            language="en",
            min_speakers=2,
            max_speakers=4,
            enable_vad=False,
            signal_confidence_threshold=0.7,
        )

        assert config.asr_backend == "canary"
        assert config.language == "en"
        assert config.min_speakers == 2
        assert config.max_speakers == 4
        assert config.enable_vad is False
        assert config.signal_confidence_threshold == 0.7

    def test_config_vad_settings(self):
        """Test VAD configuration options."""
        from subtext.pipeline.orchestrator import PipelineConfig

        config = PipelineConfig(vad_threshold=0.8)

        assert config.vad_threshold == 0.8

    def test_config_prosodics_settings(self):
        """Test prosodics configuration options."""
        from subtext.pipeline.orchestrator import PipelineConfig

        config = PipelineConfig(
            prosodics_window_ms=2000,
            prosodics_hop_ms=1000,
        )

        assert config.prosodics_window_ms == 2000
        assert config.prosodics_hop_ms == 1000

    def test_config_llm_settings(self):
        """Test LLM configuration options."""
        from subtext.pipeline.orchestrator import PipelineConfig

        config = PipelineConfig(
            llm_model="gpt-4",
            llm_provider="openai",
        )

        assert config.llm_model == "gpt-4"
        assert config.llm_provider == "openai"


# ══════════════════════════════════════════════════════════════
# PipelineResult Tests
# ══════════════════════════════════════════════════════════════


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_successful_result(self):
        """Test successful pipeline result."""
        from subtext.pipeline.orchestrator import PipelineResult

        session_id = uuid4()
        result = PipelineResult(
            session_id=session_id,
            success=True,
            duration_ms=60000,
            language="en",
        )

        assert result.session_id == session_id
        assert result.success is True
        assert result.duration_ms == 60000
        assert result.language == "en"
        assert result.error is None

    def test_failed_result(self):
        """Test failed pipeline result."""
        from subtext.pipeline.orchestrator import PipelineResult

        session_id = uuid4()
        result = PipelineResult(
            session_id=session_id,
            success=False,
            error="Audio file not found",
        )

        assert result.session_id == session_id
        assert result.success is False
        assert result.error == "Audio file not found"

    def test_result_default_values(self):
        """Test default values for PipelineResult."""
        from subtext.pipeline.orchestrator import PipelineResult

        session_id = uuid4()
        result = PipelineResult(session_id=session_id, success=True)

        assert result.speakers == []
        assert result.transcript_segments == []
        assert result.signals == []
        assert result.timeline == []
        assert result.insights == {}
        assert result.speaker_metrics == []
        assert result.speech_segments == []
        assert result.speech_ratio == 0.0
        assert result.emotions == []
        assert result.dominant_emotion is None
        assert result.vad_scores == {}
        assert result.speaker_embeddings == {}
        assert result.processing_time_ms == 0
        assert result.stage_results == []

    def test_result_with_speakers(self):
        """Test result with speaker data."""
        from subtext.pipeline.orchestrator import PipelineResult

        session_id = uuid4()
        speakers = [
            {"speaker_id": "spk_001", "name": "Speaker 1"},
            {"speaker_id": "spk_002", "name": "Speaker 2"},
        ]

        result = PipelineResult(
            session_id=session_id,
            success=True,
            speakers=speakers,
        )

        assert len(result.speakers) == 2
        assert result.speakers[0]["speaker_id"] == "spk_001"

    def test_result_with_signals(self):
        """Test result with detected signals."""
        from subtext.pipeline.orchestrator import PipelineResult

        session_id = uuid4()
        signal = Signal(
            id=uuid4(),
            session_id=session_id,
            signal_type=SignalType.TRUTH_GAP,
            timestamp_ms=5000,
            confidence=0.85,
            intensity=0.7,
        )

        result = PipelineResult(
            session_id=session_id,
            success=True,
            signals=[signal],
        )

        assert len(result.signals) == 1
        assert result.signals[0].signal_type == SignalType.TRUTH_GAP

    def test_result_with_emotions(self):
        """Test result with emotion data."""
        from subtext.pipeline.orchestrator import PipelineResult

        session_id = uuid4()
        result = PipelineResult(
            session_id=session_id,
            success=True,
            emotions=[{"emotion": "happy", "confidence": 0.8}],
            dominant_emotion="happy",
            vad_scores={"valence": 0.7, "arousal": 0.5, "dominance": 0.6},
        )

        assert len(result.emotions) == 1
        assert result.dominant_emotion == "happy"
        assert result.vad_scores["valence"] == 0.7


# ══════════════════════════════════════════════════════════════
# PipelineOrchestrator Tests
# ══════════════════════════════════════════════════════════════


class TestPipelineOrchestrator:
    """Test PipelineOrchestrator class."""

    def test_orchestrator_initialization_default(self):
        """Test orchestrator with default config."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        assert orchestrator.config is not None
        assert orchestrator.config.asr_backend == "whisperx"
        assert orchestrator._stages_initialized is False

    def test_orchestrator_initialization_custom_config(self):
        """Test orchestrator with custom config."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig

        config = PipelineConfig(
            asr_backend="canary",
            language="es",
            enable_vad=False,
        )
        orchestrator = PipelineOrchestrator(config)

        assert orchestrator.config.asr_backend == "canary"
        assert orchestrator.config.language == "es"
        assert orchestrator.vad is None  # VAD disabled

    def test_orchestrator_stages_created(self):
        """Test that all stages are created on initialization."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        assert orchestrator.cleanse is not None
        assert orchestrator.vad is not None
        assert orchestrator.diarize is not None
        assert orchestrator.transcribe is not None
        assert orchestrator.emotion is not None
        assert orchestrator.prosodics is not None
        assert orchestrator.synthesize is not None
        assert orchestrator.detector is not None

    def test_orchestrator_vad_disabled(self):
        """Test orchestrator with VAD disabled."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig

        config = PipelineConfig(enable_vad=False)
        orchestrator = PipelineOrchestrator(config)

        assert orchestrator.vad is None

    def test_orchestrator_emotion_disabled(self):
        """Test orchestrator with emotion detection disabled."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig

        config = PipelineConfig(enable_emotion_detection=False)
        orchestrator = PipelineOrchestrator(config)

        assert orchestrator.emotion is None


class TestOrchestratorAlignTranscript:
    """Test transcript alignment helper method."""

    def test_align_empty_inputs(self):
        """Test alignment with empty inputs."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        result = orchestrator._align_transcript_speakers([], [])

        assert result == []

    def test_align_basic_alignment(self):
        """Test basic transcript-speaker alignment."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        words = [
            {"text": "Hello", "start_ms": 0, "end_ms": 500, "confidence": 0.9},
            {"text": "there", "start_ms": 500, "end_ms": 1000, "confidence": 0.85},
            {"text": "how", "start_ms": 2000, "end_ms": 2500, "confidence": 0.95},
            {"text": "are", "start_ms": 2500, "end_ms": 3000, "confidence": 0.9},
        ]

        diarization = [
            {"speaker_id": "spk_001", "start_ms": 0, "end_ms": 1500},
            {"speaker_id": "spk_002", "start_ms": 2000, "end_ms": 3500},
        ]

        result = orchestrator._align_transcript_speakers(words, diarization)

        assert len(result) == 2
        assert result[0]["speaker_id"] == "spk_001"
        assert result[0]["text"] == "Hello there"
        assert result[1]["speaker_id"] == "spk_002"
        assert result[1]["text"] == "how are"

    def test_align_with_question(self):
        """Test alignment identifies questions."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        words = [
            {"text": "How", "start_ms": 0, "end_ms": 200, "confidence": 0.9},
            {"text": "are", "start_ms": 200, "end_ms": 400, "confidence": 0.9},
            {"text": "you?", "start_ms": 400, "end_ms": 700, "confidence": 0.9},
        ]

        diarization = [
            {"speaker_id": "spk_001", "start_ms": 0, "end_ms": 1000},
        ]

        result = orchestrator._align_transcript_speakers(words, diarization)

        assert len(result) == 1
        assert result[0]["is_question"] is True
        assert result[0]["text"] == "How are you?"

    def test_align_preserves_word_data(self):
        """Test alignment preserves word-level data."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        words = [
            {"text": "Test", "start_ms": 0, "end_ms": 500, "confidence": 0.95},
        ]

        diarization = [
            {"speaker_id": "spk_001", "start_ms": 0, "end_ms": 1000},
        ]

        result = orchestrator._align_transcript_speakers(words, diarization)

        assert len(result) == 1
        assert "words" in result[0]
        assert len(result[0]["words"]) == 1
        assert result[0]["confidence"] == 0.95

    def test_align_empty_segment(self):
        """Test alignment handles segments with no words."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        words = [
            {"text": "Hello", "start_ms": 0, "end_ms": 500, "confidence": 0.9},
        ]

        diarization = [
            {"speaker_id": "spk_001", "start_ms": 0, "end_ms": 1000},
            {"speaker_id": "spk_002", "start_ms": 2000, "end_ms": 3000},  # No words here
        ]

        result = orchestrator._align_transcript_speakers(words, diarization)

        # Only segment with words should be returned
        assert len(result) == 1
        assert result[0]["speaker_id"] == "spk_001"


class TestOrchestratorRunStage:
    """Test stage runner helper."""

    @pytest.mark.asyncio
    async def test_run_stage_success(self):
        """Test running a stage successfully."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Create mock stage
        mock_stage = MagicMock()
        mock_stage.name = "test_stage"
        mock_stage.process = AsyncMock(return_value={"result": "success"})

        result = await orchestrator._run_stage(mock_stage, param1="value1")

        assert result.success is True
        assert result.data == {"result": "success"}
        assert result.duration_ms > 0
        mock_stage.process.assert_called_once_with(param1="value1")

    @pytest.mark.asyncio
    async def test_run_stage_failure(self):
        """Test running a stage that fails."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Create mock stage that fails
        mock_stage = MagicMock()
        mock_stage.name = "failing_stage"
        mock_stage.process = AsyncMock(side_effect=ValueError("Test error"))

        result = await orchestrator._run_stage(mock_stage)

        assert result.success is False
        assert result.error == "Test error"
        assert result.data == {}
        assert result.duration_ms > 0


class TestOrchestratorInitialize:
    """Test orchestrator initialization."""

    @pytest.mark.asyncio
    async def test_initialize_all_stages(self):
        """Test initializing all pipeline stages."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Mock all stage initialize methods
        orchestrator.cleanse.initialize = AsyncMock()
        orchestrator.vad.initialize = AsyncMock()
        orchestrator.diarize.initialize = AsyncMock()
        orchestrator.transcribe.initialize = AsyncMock()
        orchestrator.emotion.initialize = AsyncMock()
        orchestrator.prosodics.initialize = AsyncMock()
        orchestrator.synthesize.initialize = AsyncMock()

        await orchestrator.initialize()

        assert orchestrator._stages_initialized is True
        orchestrator.cleanse.initialize.assert_called_once()
        orchestrator.diarize.initialize.assert_called_once()
        orchestrator.transcribe.initialize.assert_called_once()
        orchestrator.prosodics.initialize.assert_called_once()
        orchestrator.synthesize.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize only runs once."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Mock all stage initialize methods
        orchestrator.cleanse.initialize = AsyncMock()
        orchestrator.vad.initialize = AsyncMock()
        orchestrator.diarize.initialize = AsyncMock()
        orchestrator.transcribe.initialize = AsyncMock()
        orchestrator.emotion.initialize = AsyncMock()
        orchestrator.prosodics.initialize = AsyncMock()
        orchestrator.synthesize.initialize = AsyncMock()

        await orchestrator.initialize()
        await orchestrator.initialize()  # Second call

        # Should only be called once
        assert orchestrator.cleanse.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_initialize_without_optional_stages(self):
        """Test initialize skips disabled optional stages."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig

        config = PipelineConfig(enable_vad=False, enable_emotion_detection=False)
        orchestrator = PipelineOrchestrator(config)

        # Mock required stage initialize methods
        orchestrator.cleanse.initialize = AsyncMock()
        orchestrator.diarize.initialize = AsyncMock()
        orchestrator.transcribe.initialize = AsyncMock()
        orchestrator.prosodics.initialize = AsyncMock()
        orchestrator.synthesize.initialize = AsyncMock()

        await orchestrator.initialize()

        assert orchestrator._stages_initialized is True
        # VAD and emotion should not have been initialized (they're None)
        assert orchestrator.vad is None
        assert orchestrator.emotion is None


# ══════════════════════════════════════════════════════════════
# Factory Function Tests
# ══════════════════════════════════════════════════════════════


class TestCreatePipeline:
    """Test create_pipeline factory function."""

    def test_create_default_pipeline(self):
        """Test creating pipeline with defaults."""
        from subtext.pipeline.orchestrator import create_pipeline

        pipeline = create_pipeline()

        assert pipeline is not None
        assert pipeline.config.asr_backend == "whisperx"
        assert pipeline.config.enable_vad is True
        assert pipeline.config.enable_emotion_detection is True

    def test_create_pipeline_with_language(self):
        """Test creating pipeline with specific language."""
        from subtext.pipeline.orchestrator import create_pipeline

        pipeline = create_pipeline(language="es")

        assert pipeline.config.language == "es"

    def test_create_pipeline_with_speaker_limits(self):
        """Test creating pipeline with speaker constraints."""
        from subtext.pipeline.orchestrator import create_pipeline

        pipeline = create_pipeline(min_speakers=2, max_speakers=5)

        assert pipeline.config.min_speakers == 2
        assert pipeline.config.max_speakers == 5

    def test_create_pipeline_with_asr_backend(self):
        """Test creating pipeline with different ASR backends."""
        from subtext.pipeline.orchestrator import create_pipeline

        # Test each backend
        for backend in ["whisperx", "canary", "parakeet"]:
            pipeline = create_pipeline(asr_backend=backend)
            assert pipeline.config.asr_backend == backend

    def test_create_pipeline_disabled_features(self):
        """Test creating pipeline with disabled features."""
        from subtext.pipeline.orchestrator import create_pipeline

        pipeline = create_pipeline(
            enable_vad=False,
            enable_emotion=False,
            extract_embeddings=False,
        )

        assert pipeline.config.enable_vad is False
        assert pipeline.config.enable_emotion_detection is False
        assert pipeline.config.extract_speaker_embeddings is False


# ══════════════════════════════════════════════════════════════
# Integration-style Tests
# ══════════════════════════════════════════════════════════════


class TestOrchestratorProcessFile:
    """Test process_file method with mocked stages."""

    @pytest.mark.asyncio
    async def test_process_file_success(self):
        """Test successful file processing with mocked stages."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator
        import numpy as np

        orchestrator = PipelineOrchestrator()

        # Create mock audio data
        mock_audio = np.zeros(16000 * 10, dtype=np.float32)  # 10 seconds

        # Mock all stages
        orchestrator.cleanse.process = AsyncMock(return_value={
            "audio": mock_audio,
            "sample_rate": 16000,
            "noise_profile": {"snr": 20},
        })
        orchestrator.cleanse.initialize = AsyncMock()

        orchestrator.vad.process = AsyncMock(return_value={
            "speech_segments": [{"start_ms": 0, "end_ms": 5000}],
            "speech_ratio": 0.5,
        })
        orchestrator.vad.initialize = AsyncMock()

        orchestrator.diarize.process = AsyncMock(return_value={
            "speakers": [{"speaker_id": "spk_001"}],
            "segments": [{"speaker_id": "spk_001", "start_ms": 0, "end_ms": 5000}],
            "embeddings": {},
        })
        orchestrator.diarize.initialize = AsyncMock()

        orchestrator.transcribe.process = AsyncMock(return_value={
            "transcript": "Hello world",
            "words": [
                {"text": "Hello", "start_ms": 0, "end_ms": 500, "confidence": 0.9},
                {"text": "world", "start_ms": 500, "end_ms": 1000, "confidence": 0.95},
            ],
            "language": "en",
            "backend": "whisperx",
        })
        orchestrator.transcribe.initialize = AsyncMock()

        orchestrator.emotion.process = AsyncMock(return_value={
            "emotions": [{"emotion": "neutral", "confidence": 0.8}],
            "dominant_emotion": "neutral",
            "vad": {"valence": 0.5, "arousal": 0.4, "dominance": 0.5},
        })
        orchestrator.emotion.initialize = AsyncMock()

        orchestrator.prosodics.process = AsyncMock(return_value={
            "features": [{"timestamp_ms": 0, "pitch_mean": 150}],
        })
        orchestrator.prosodics.initialize = AsyncMock()

        orchestrator.synthesize.process = AsyncMock(return_value={
            "timeline": [{"time": 0, "event": "speech"}],
            "insights": {"summary": "A greeting"},
            "speaker_metrics": [{"speaker_id": "spk_001", "talk_time": 5000}],
        })
        orchestrator.synthesize.initialize = AsyncMock()

        # Mock signal detector
        orchestrator.detector.detect_all = MagicMock(return_value=[])

        session_id = uuid4()
        result = await orchestrator.process_file(session_id, "/fake/audio.wav")

        assert result.success is True
        assert result.session_id == session_id
        assert len(result.speakers) == 1
        assert result.language == "en"
        assert result.speech_ratio == 0.5
        assert result.dominant_emotion == "neutral"

    @pytest.mark.asyncio
    async def test_process_file_cleanse_failure(self):
        """Test processing failure when cleanse stage fails."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Mock cleanse to fail
        orchestrator.cleanse.process = AsyncMock(side_effect=Exception("Audio decode failed"))
        orchestrator.cleanse.initialize = AsyncMock()

        # Mock other initializations
        orchestrator.vad.initialize = AsyncMock()
        orchestrator.diarize.initialize = AsyncMock()
        orchestrator.transcribe.initialize = AsyncMock()
        orchestrator.emotion.initialize = AsyncMock()
        orchestrator.prosodics.initialize = AsyncMock()
        orchestrator.synthesize.initialize = AsyncMock()

        session_id = uuid4()
        result = await orchestrator.process_file(session_id, "/fake/audio.wav")

        assert result.success is False
        assert "Audio decode failed" in result.error

    @pytest.mark.asyncio
    async def test_process_file_without_vad(self):
        """Test processing without VAD stage."""
        from subtext.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig
        import numpy as np

        config = PipelineConfig(enable_vad=False)
        orchestrator = PipelineOrchestrator(config)

        # Create mock audio data
        mock_audio = np.zeros(16000 * 5, dtype=np.float32)

        # Mock stages
        orchestrator.cleanse.process = AsyncMock(return_value={
            "audio": mock_audio,
            "sample_rate": 16000,
            "noise_profile": {},
        })
        orchestrator.cleanse.initialize = AsyncMock()

        orchestrator.diarize.process = AsyncMock(return_value={
            "speakers": [],
            "segments": [],
            "embeddings": {},
        })
        orchestrator.diarize.initialize = AsyncMock()

        orchestrator.transcribe.process = AsyncMock(return_value={
            "transcript": "",
            "words": [],
            "language": "en",
        })
        orchestrator.transcribe.initialize = AsyncMock()

        orchestrator.prosodics.process = AsyncMock(return_value={
            "features": [],
        })
        orchestrator.prosodics.initialize = AsyncMock()

        orchestrator.synthesize.process = AsyncMock(return_value={
            "timeline": [],
            "insights": {},
            "speaker_metrics": [],
        })
        orchestrator.synthesize.initialize = AsyncMock()

        orchestrator.detector.detect_all = MagicMock(return_value=[])

        session_id = uuid4()
        result = await orchestrator.process_file(session_id, "/fake/audio.wav")

        assert result.success is True
        # Speech ratio should be 1.0 when VAD is disabled
        assert result.speech_ratio == 1.0


# ══════════════════════════════════════════════════════════════
# Stage Result Tests
# ══════════════════════════════════════════════════════════════


class TestStageResult:
    """Test StageResult dataclass."""

    def test_stage_result_success(self):
        """Test successful stage result."""
        from subtext.pipeline.stages import StageResult

        result = StageResult(
            success=True,
            data={"output": "value"},
            duration_ms=150.5,
        )

        assert result.success is True
        assert result.data == {"output": "value"}
        assert result.duration_ms == 150.5
        assert result.error is None

    def test_stage_result_failure(self):
        """Test failed stage result."""
        from subtext.pipeline.stages import StageResult

        result = StageResult(
            success=False,
            data={},
            duration_ms=50.0,
            error="Stage failed",
        )

        assert result.success is False
        assert result.error == "Stage failed"
        assert result.data == {}
