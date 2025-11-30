"""
Subtext Audio Processing Pipeline

The core ML pipeline for audio analysis, signal detection, and insight generation.

Best-in-class open source models (2025):
- VAD: Silero VAD (87.7% TPR)
- Noise Suppression: DeepFilterNet3
- Diarization: Pyannote 3.1+ with ECAPA-TDNN embeddings
- ASR: WhisperX (default), NVIDIA Canary (accuracy), Parakeet (speed)
- Emotion: Emotion2Vec (SOTA on 9 datasets)
- Speaker Embeddings: ECAPA-TDNN (1.71% EER)
"""

from .orchestrator import PipelineOrchestrator, PipelineConfig
from .signals import SignalAtlas, SignalDetector
from .stages import (
    CleanseStage,
    VADStage,
    DiarizeStage,
    TranscribeStage,
    EmotionStage,
    ProsodicsStage,
    SynthesizeStage,
)

__all__ = [
    "PipelineOrchestrator",
    "PipelineConfig",
    "SignalAtlas",
    "SignalDetector",
    # Pipeline stages
    "CleanseStage",
    "VADStage",
    "DiarizeStage",
    "TranscribeStage",
    "EmotionStage",
    "ProsodicsStage",
    "SynthesizeStage",
]
