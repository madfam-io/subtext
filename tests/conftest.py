"""
Pytest Configuration and Fixtures

Shared fixtures for unit and integration tests.
"""

import asyncio
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio
import numpy as np

from subtext.config import Settings


# ══════════════════════════════════════════════════════════════
# Async Event Loop
# ══════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ══════════════════════════════════════════════════════════════
# Settings Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def test_settings() -> Settings:
    """Test settings with in-memory/mock configurations."""
    return Settings(
        app_env="development",
        debug=True,
        database_url="postgresql+asyncpg://test:test@localhost:5432/subtext_test",
        redis_url="redis://localhost:6379/15",
        janua_base_url="http://localhost:8080",
        model_cache_dir="/tmp/subtext_models",
    )


# ══════════════════════════════════════════════════════════════
# Audio Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, int]:
    """Generate a sample audio signal for testing.

    Returns a 5-second stereo sine wave at 440Hz.
    """
    sample_rate = 16000
    duration = 5.0  # seconds
    frequency = 440  # Hz

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio, sample_rate


@pytest.fixture
def sample_speech_audio() -> tuple[np.ndarray, int]:
    """Generate a sample speech-like audio signal.

    Simulates speech with varying amplitude and frequency modulation.
    """
    sample_rate = 16000
    duration = 10.0  # seconds

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Base frequency with formant-like variations
    f0 = 150  # Fundamental frequency
    audio = np.zeros_like(t)

    # Add harmonics
    for harmonic in range(1, 6):
        audio += (0.5 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)

    # Add amplitude modulation (speech-like envelope)
    envelope = 0.5 * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))  # 3 Hz modulation
    audio = audio * envelope

    # Add some noise
    audio += 0.01 * np.random.randn(len(audio)).astype(np.float32)

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32), sample_rate


@pytest.fixture
def sample_multi_speaker_audio() -> tuple[np.ndarray, int]:
    """Generate audio simulating multiple speakers.

    Creates alternating segments with different pitch characteristics.
    """
    sample_rate = 16000
    duration = 20.0  # seconds

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.zeros_like(t)

    # Speaker A: Lower pitch (100 Hz fundamental)
    # Speaker B: Higher pitch (200 Hz fundamental)
    segment_duration = 2.0  # seconds per speaker segment

    for i in range(int(duration / segment_duration)):
        start_idx = int(i * segment_duration * sample_rate)
        end_idx = int((i + 1) * segment_duration * sample_rate)
        segment_t = t[start_idx:end_idx] - t[start_idx]

        if i % 2 == 0:
            # Speaker A
            f0 = 100
        else:
            # Speaker B
            f0 = 200

        segment = np.zeros(end_idx - start_idx, dtype=np.float32)
        for harmonic in range(1, 4):
            segment += (0.5 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * segment_t)

        # Add envelope
        envelope = 0.5 * (1 + 0.3 * np.sin(2 * np.pi * 4 * segment_t))
        segment = segment * envelope

        audio[start_idx:end_idx] = segment

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32), sample_rate


# ══════════════════════════════════════════════════════════════
# Model Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def session_id():
    """Generate a random session ID."""
    return uuid4()


@pytest.fixture
def org_id():
    """Generate a random organization ID."""
    return uuid4()


@pytest.fixture
def user_id():
    """Generate a random user ID."""
    return uuid4()


# ══════════════════════════════════════════════════════════════
# Pipeline Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def pipeline_config():
    """Default pipeline configuration for tests."""
    from subtext.pipeline import PipelineConfig

    return PipelineConfig(
        asr_backend="whisperx",
        enable_vad=True,
        enable_emotion_detection=True,
        extract_speaker_embeddings=False,  # Disable for faster tests
        signal_confidence_threshold=0.5,
    )


# ══════════════════════════════════════════════════════════════
# Mock Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_janua_token() -> str:
    """Generate a mock Janua JWT token."""
    import jwt
    from datetime import datetime, timedelta

    payload = {
        "sub": str(uuid4()),
        "org_id": str(uuid4()),
        "email": "test@example.com",
        "roles": ["member"],
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "iss": "janua-test",
    }

    return jwt.encode(payload, "test-secret", algorithm="HS256")
