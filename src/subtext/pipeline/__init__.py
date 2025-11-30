"""
Subtext Audio Processing Pipeline

The core ML pipeline for audio analysis, signal detection, and insight generation.
"""

from .orchestrator import PipelineOrchestrator, PipelineConfig
from .signals import SignalAtlas, SignalDetector
from .stages import (
    CleanseStage,
    DiarizeStage,
    TranscribeStage,
    ProsodicsStage,
    SynthesizeStage,
)

__all__ = [
    "PipelineOrchestrator",
    "PipelineConfig",
    "SignalAtlas",
    "SignalDetector",
    "CleanseStage",
    "DiarizeStage",
    "TranscribeStage",
    "ProsodicsStage",
    "SynthesizeStage",
]
