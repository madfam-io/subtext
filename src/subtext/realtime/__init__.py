"""
Subtext Realtime Module

WebSocket-based real-time audio streaming, processing, and ESP broadcasting.
"""

from .connection import ConnectionManager, RealtimeConnection
from .processor import RealtimeProcessor, AudioChunk
from .broadcaster import ESPBroadcaster
from .protocol import (
    RealtimeMessage,
    RealtimeMessageType,
    AudioConfig,
    SessionConfig,
)

__all__ = [
    # Connection management
    "ConnectionManager",
    "RealtimeConnection",
    # Processing
    "RealtimeProcessor",
    "AudioChunk",
    # Broadcasting
    "ESPBroadcaster",
    # Protocol
    "RealtimeMessage",
    "RealtimeMessageType",
    "AudioConfig",
    "SessionConfig",
]
