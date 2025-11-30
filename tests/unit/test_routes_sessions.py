"""
Unit tests for sessions API routes.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
from io import BytesIO

from subtext.api.routes.sessions import (
    SessionListResponse,
    UploadResponse,
    TranscriptResponse,
    SignalListResponse,
    TimelineResponse,
    InsightsResponse,
    AnalysisResponse,
    _sessions,
    _process_session,
)
from subtext.core.models import SessionCreate, SessionResponse, SessionStatus
from subtext.integrations.janua import TokenPayload


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return TokenPayload(
        sub="user-123",
        email="test@example.com",
        org_id="org-456",
        roles=["user"],
        permissions=["read:sessions", "write:sessions"],
        exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        iat=int(datetime.utcnow().timestamp()),
        iss="https://auth.example.com",
        aud="subtext-api",
    )


@pytest.fixture
def mock_other_user():
    """Create a different user for access control tests."""
    return TokenPayload(
        sub="user-999",
        email="other@example.com",
        org_id="org-999",
        roles=["user"],
        permissions=["read:sessions"],
        exp=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
        iat=int(datetime.utcnow().timestamp()),
        iss="https://auth.example.com",
        aud="subtext-api",
    )


@pytest.fixture
def sample_session(mock_user):
    """Create a sample session in memory."""
    session_id = uuid4()
    session = {
        "id": session_id,
        "org_id": mock_user.org_id,
        "created_by": mock_user.sub,
        "name": "Test Session",
        "description": "Test description",
        "status": SessionStatus.PENDING,
        "language": "en",
        "settings": {},
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "duration_ms": None,
        "speaker_count": None,
        "signal_count": None,
    }
    _sessions[session_id] = session
    yield session
    # Cleanup
    _sessions.pop(session_id, None)


@pytest.fixture
def completed_session(mock_user):
    """Create a completed session with results."""
    session_id = uuid4()
    session = {
        "id": session_id,
        "org_id": mock_user.org_id,
        "created_by": mock_user.sub,
        "name": "Completed Session",
        "description": "A completed session",
        "status": SessionStatus.COMPLETED,
        "language": "en",
        "settings": {},
        "created_at": datetime.utcnow() - timedelta(hours=1),
        "completed_at": datetime.utcnow(),
        "duration_ms": 120000,
        "speaker_count": 2,
        "signal_count": 5,
        "result": {
            "transcript_segments": [
                {"speaker": "A", "text": "Hello there", "start_ms": 0, "end_ms": 1000},
                {"speaker": "B", "text": "Hi, how are you?", "start_ms": 1000, "end_ms": 2000},
            ],
            "speakers": [
                {"id": "A", "name": "Speaker A"},
                {"id": "B", "name": "Speaker B"},
            ],
            "signals": [
                {"signal_type": "hesitation", "confidence": 0.8, "intensity": 0.6},
                {"signal_type": "emphasis", "confidence": 0.9, "intensity": 0.7},
                {"signal_type": "hesitation", "confidence": 0.6, "intensity": 0.4},
            ],
            "timeline": [
                {"timestamp_ms": 0, "tension": 0.3},
                {"timestamp_ms": 5000, "tension": 0.5},
            ],
            "insights": {
                "summary": "A productive conversation",
                "key_moments": [{"timestamp_ms": 1000, "description": "Key moment"}],
                "recommendations": ["Follow up on topic A"],
            },
            "speaker_metrics": [
                {"speaker_id": "A", "talk_time_ms": 60000},
            ],
        },
    }
    _sessions[session_id] = session
    yield session
    _sessions.pop(session_id, None)


@pytest.fixture
def clear_sessions():
    """Clear sessions before and after a test."""
    _sessions.clear()
    yield
    _sessions.clear()


# ══════════════════════════════════════════════════════════════
# Response Model Tests
# ══════════════════════════════════════════════════════════════


class TestResponseModels:
    """Test response model validation."""

    def test_session_list_response(self):
        """Test SessionListResponse model."""
        response = SessionListResponse(
            sessions=[],
            total=0,
            cursor=None,
        )
        assert response.sessions == []
        assert response.total == 0
        assert response.cursor is None

    def test_upload_response(self):
        """Test UploadResponse model."""
        session_id = uuid4()
        response = UploadResponse(
            session_id=session_id,
            status=SessionStatus.UPLOADING,
            message="Upload complete",
        )
        assert response.session_id == session_id
        assert response.status == SessionStatus.UPLOADING

    def test_transcript_response(self):
        """Test TranscriptResponse model."""
        response = TranscriptResponse(
            segments=[{"text": "hello"}],
            speakers=[{"id": "A"}],
            full_text="hello",
        )
        assert len(response.segments) == 1
        assert response.full_text == "hello"

    def test_signal_list_response(self):
        """Test SignalListResponse model."""
        response = SignalListResponse(
            signals=[{"type": "hesitation"}],
            total=1,
            summary={"hesitation": 1},
        )
        assert response.total == 1
        assert response.summary["hesitation"] == 1

    def test_timeline_response(self):
        """Test TimelineResponse model."""
        response = TimelineResponse(
            duration_ms=60000,
            resolution_ms=5000,
            data_points=[{"timestamp_ms": 0, "tension": 0.5}],
        )
        assert response.duration_ms == 60000
        assert response.resolution_ms == 5000

    def test_insights_response(self):
        """Test InsightsResponse model."""
        response = InsightsResponse(
            summary="Test summary",
            key_moments=[],
            recommendations=["Do this"],
            speaker_metrics=[],
            risk_flags=[],
        )
        assert response.summary == "Test summary"
        assert len(response.recommendations) == 1

    def test_analysis_response(self):
        """Test AnalysisResponse model."""
        session = SessionResponse(
            id=uuid4(),
            name="Test",
            status=SessionStatus.COMPLETED,
            duration_ms=60000,
            speaker_count=2,
            signal_count=10,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        transcript = TranscriptResponse(segments=[], speakers=[], full_text="")
        signals = SignalListResponse(signals=[], total=0, summary={})
        timeline = TimelineResponse(duration_ms=60000, resolution_ms=5000, data_points=[])
        insights = InsightsResponse(
            summary="", key_moments=[], recommendations=[], speaker_metrics=[], risk_flags=[]
        )

        response = AnalysisResponse(
            session=session,
            transcript=transcript,
            signals=signals,
            timeline=timeline,
            insights=insights,
        )
        assert response.session.name == "Test"


# ══════════════════════════════════════════════════════════════
# Create Session Tests
# ══════════════════════════════════════════════════════════════


class TestCreateSession:
    """Test session creation endpoint."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, mock_user):
        """Test successful session creation."""
        from subtext.api.routes.sessions import create_session

        request = SessionCreate(
            name="My Session",
            description="Test description",
            language="en",
        )

        response = await create_session(request, mock_user)

        assert response.name == "My Session"
        assert response.status == SessionStatus.PENDING
        assert response.id in _sessions
        assert _sessions[response.id]["org_id"] == mock_user.org_id

    @pytest.mark.asyncio
    async def test_create_session_with_settings(self, mock_user):
        """Test session creation with custom settings."""
        from subtext.api.routes.sessions import create_session

        request = SessionCreate(
            name="Custom Session",
            settings={"custom_key": "value"},
        )

        response = await create_session(request, mock_user)

        assert response.name == "Custom Session"
        assert _sessions[response.id]["settings"] == {"custom_key": "value"}

    @pytest.mark.asyncio
    async def test_create_session_defaults(self, mock_user):
        """Test session creation with defaults."""
        from subtext.api.routes.sessions import create_session

        request = SessionCreate(name="Default Session")

        response = await create_session(request, mock_user)

        assert response.duration_ms is None
        assert response.speaker_count is None
        assert response.completed_at is None


# ══════════════════════════════════════════════════════════════
# List Sessions Tests
# ══════════════════════════════════════════════════════════════


class TestListSessions:
    """Test session listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, mock_user, clear_sessions):
        """Test listing sessions when none exist."""
        from subtext.api.routes.sessions import list_sessions

        # Pass all positional args to avoid Query default issues
        response = await list_sessions(mock_user, None, 20, None)

        assert response.sessions == []
        assert response.total == 0

    @pytest.mark.asyncio
    async def test_list_sessions_filters_by_org(self, mock_user, mock_other_user, clear_sessions):
        """Test that listing filters by organization."""
        from subtext.api.routes.sessions import create_session, list_sessions

        # Create sessions for both users
        await create_session(SessionCreate(name="User Session"), mock_user)
        await create_session(SessionCreate(name="Other Session"), mock_other_user)

        # List for first user
        response = await list_sessions(mock_user, None, 20, None)
        assert response.total == 1
        assert response.sessions[0].name == "User Session"

        # List for second user
        response = await list_sessions(mock_other_user, None, 20, None)
        assert response.total == 1
        assert response.sessions[0].name == "Other Session"

    @pytest.mark.asyncio
    async def test_list_sessions_with_status_filter(self, mock_user, sample_session, completed_session):
        """Test filtering by status."""
        from subtext.api.routes.sessions import list_sessions

        # Filter for pending
        response = await list_sessions(mock_user, SessionStatus.PENDING, 20, None)
        assert response.total == 1
        assert response.sessions[0].status == SessionStatus.PENDING

        # Filter for completed
        response = await list_sessions(mock_user, SessionStatus.COMPLETED, 20, None)
        assert response.total == 1
        assert response.sessions[0].status == SessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_list_sessions_with_limit(self, mock_user, clear_sessions):
        """Test pagination limit."""
        from subtext.api.routes.sessions import create_session, list_sessions

        # Create multiple sessions
        for i in range(5):
            await create_session(SessionCreate(name=f"Session {i}"), mock_user)

        response = await list_sessions(mock_user, None, 3, None)
        assert len(response.sessions) == 3
        assert response.total == 5

    @pytest.mark.asyncio
    async def test_list_sessions_sorted_by_created_at(self, mock_user, clear_sessions):
        """Test sessions are sorted by creation time."""
        from subtext.api.routes.sessions import list_sessions

        # Create sessions with different timestamps
        for i in range(3):
            session_id = uuid4()
            _sessions[session_id] = {
                "id": session_id,
                "org_id": mock_user.org_id,
                "name": f"Session {i}",
                "status": SessionStatus.PENDING,
                "created_at": datetime.utcnow() - timedelta(hours=i),
            }

        response = await list_sessions(mock_user, None, 20, None)

        # Most recent first
        assert response.sessions[0].name == "Session 0"


# ══════════════════════════════════════════════════════════════
# Get Session Tests
# ══════════════════════════════════════════════════════════════


class TestGetSession:
    """Test session retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_get_session_success(self, mock_user, sample_session):
        """Test successful session retrieval."""
        from subtext.api.routes.sessions import get_session

        response = await get_session(sample_session["id"], mock_user)

        assert response.id == sample_session["id"]
        assert response.name == "Test Session"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, mock_user):
        """Test getting nonexistent session."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_session

        with pytest.raises(HTTPException) as exc:
            await get_session(uuid4(), mock_user)

        assert exc.value.status_code == 404
        assert "not found" in exc.value.detail

    @pytest.mark.asyncio
    async def test_get_session_wrong_org(self, mock_other_user, sample_session):
        """Test accessing session from different org."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_session

        with pytest.raises(HTTPException) as exc:
            await get_session(sample_session["id"], mock_other_user)

        assert exc.value.status_code == 403
        assert "Not authorized" in exc.value.detail


# ══════════════════════════════════════════════════════════════
# Upload Audio Tests
# ══════════════════════════════════════════════════════════════


class TestUploadAudio:
    """Test audio upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_audio_success(self, mock_user, sample_session):
        """Test successful audio upload."""
        from fastapi import BackgroundTasks
        from subtext.api.routes.sessions import upload_audio

        background_tasks = BackgroundTasks()

        # Create mock UploadFile
        mock_file = MagicMock()
        mock_file.content_type = "audio/wav"
        mock_file.read = AsyncMock(return_value=b"RIFF" + b"\x00" * 100)

        response = await upload_audio(
            sample_session["id"],
            background_tasks,
            mock_file,
            mock_user,
        )

        assert response.session_id == sample_session["id"]
        assert response.status == SessionStatus.UPLOADING

    @pytest.mark.asyncio
    async def test_upload_audio_session_not_found(self, mock_user):
        """Test upload to nonexistent session."""
        from fastapi import BackgroundTasks, HTTPException
        from subtext.api.routes.sessions import upload_audio

        background_tasks = BackgroundTasks()
        mock_file = MagicMock()
        mock_file.content_type = "audio/wav"

        with pytest.raises(HTTPException) as exc:
            await upload_audio(uuid4(), background_tasks, mock_file, mock_user)

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_audio_wrong_org(self, mock_other_user, sample_session):
        """Test upload to session from different org."""
        from fastapi import BackgroundTasks, HTTPException
        from subtext.api.routes.sessions import upload_audio

        background_tasks = BackgroundTasks()
        mock_file = MagicMock()
        mock_file.content_type = "audio/wav"

        with pytest.raises(HTTPException) as exc:
            await upload_audio(
                sample_session["id"],
                background_tasks,
                mock_file,
                mock_other_user,
            )

        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_upload_audio_invalid_format(self, mock_user, sample_session):
        """Test upload with invalid audio format."""
        from fastapi import BackgroundTasks, HTTPException
        from subtext.api.routes.sessions import upload_audio

        background_tasks = BackgroundTasks()
        mock_file = MagicMock()
        mock_file.content_type = "text/plain"

        with pytest.raises(HTTPException) as exc:
            await upload_audio(
                sample_session["id"],
                background_tasks,
                mock_file,
                mock_user,
            )

        assert exc.value.status_code == 400
        assert "Unsupported audio format" in exc.value.detail

    @pytest.mark.asyncio
    async def test_upload_audio_already_uploaded(self, mock_user, sample_session):
        """Test upload when session already has audio."""
        from fastapi import BackgroundTasks, HTTPException
        from subtext.api.routes.sessions import upload_audio

        # Change session status
        sample_session["status"] = SessionStatus.PROCESSING

        background_tasks = BackgroundTasks()
        mock_file = MagicMock()
        mock_file.content_type = "audio/wav"

        with pytest.raises(HTTPException) as exc:
            await upload_audio(
                sample_session["id"],
                background_tasks,
                mock_file,
                mock_user,
            )

        assert exc.value.status_code == 400
        assert "already has audio" in exc.value.detail

    @pytest.mark.asyncio
    async def test_upload_audio_supported_formats(self, mock_user, sample_session):
        """Test various supported audio formats."""
        from fastapi import BackgroundTasks
        from subtext.api.routes.sessions import upload_audio

        supported = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/webm", "audio/ogg"]

        for content_type in supported:
            # Reset session status
            sample_session["status"] = SessionStatus.PENDING

            background_tasks = BackgroundTasks()
            mock_file = MagicMock()
            mock_file.content_type = content_type
            mock_file.read = AsyncMock(return_value=b"audio data")

            response = await upload_audio(
                sample_session["id"],
                background_tasks,
                mock_file,
                mock_user,
            )

            assert response.status == SessionStatus.UPLOADING


# ══════════════════════════════════════════════════════════════
# Get Transcript Tests
# ══════════════════════════════════════════════════════════════


class TestGetTranscript:
    """Test transcript retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_get_transcript_success(self, mock_user, completed_session):
        """Test successful transcript retrieval."""
        from subtext.api.routes.sessions import get_transcript

        response = await get_transcript(completed_session["id"], mock_user)

        assert len(response.segments) == 2
        assert len(response.speakers) == 2
        assert "Hello there" in response.full_text

    @pytest.mark.asyncio
    async def test_get_transcript_not_found(self, mock_user):
        """Test getting transcript for nonexistent session."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_transcript

        with pytest.raises(HTTPException) as exc:
            await get_transcript(uuid4(), mock_user)

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_transcript_wrong_org(self, mock_other_user, completed_session):
        """Test getting transcript from different org."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_transcript

        with pytest.raises(HTTPException) as exc:
            await get_transcript(completed_session["id"], mock_other_user)

        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_transcript_not_complete(self, mock_user, sample_session):
        """Test getting transcript before analysis complete."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_transcript

        with pytest.raises(HTTPException) as exc:
            await get_transcript(sample_session["id"], mock_user)

        assert exc.value.status_code == 400
        assert "not complete" in exc.value.detail


# ══════════════════════════════════════════════════════════════
# Get Signals Tests
# ══════════════════════════════════════════════════════════════


class TestGetSignals:
    """Test signals retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_get_signals_success(self, mock_user, completed_session):
        """Test successful signals retrieval."""
        from subtext.api.routes.sessions import get_signals

        # Pass positional args: session_id, user, signal_type, min_confidence
        response = await get_signals(completed_session["id"], mock_user, None, 0.5)

        assert response.total == 3
        assert "hesitation" in response.summary
        assert response.summary["hesitation"] == 2

    @pytest.mark.asyncio
    async def test_get_signals_filter_by_type(self, mock_user, completed_session):
        """Test filtering signals by type."""
        from subtext.api.routes.sessions import get_signals

        response = await get_signals(completed_session["id"], mock_user, "hesitation", 0.5)

        assert response.total == 2
        assert all(s["signal_type"] == "hesitation" for s in response.signals)

    @pytest.mark.asyncio
    async def test_get_signals_filter_by_confidence(self, mock_user, completed_session):
        """Test filtering signals by minimum confidence."""
        from subtext.api.routes.sessions import get_signals

        response = await get_signals(completed_session["id"], mock_user, None, 0.7)

        assert response.total == 2
        assert all(s["confidence"] >= 0.7 for s in response.signals)

    @pytest.mark.asyncio
    async def test_get_signals_not_found(self, mock_user):
        """Test getting signals for nonexistent session."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_signals

        with pytest.raises(HTTPException) as exc:
            await get_signals(uuid4(), mock_user, None, 0.5)

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_signals_not_complete(self, mock_user, sample_session):
        """Test getting signals before analysis complete."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_signals

        with pytest.raises(HTTPException) as exc:
            await get_signals(sample_session["id"], mock_user, None, 0.5)

        assert exc.value.status_code == 400


# ══════════════════════════════════════════════════════════════
# Get Timeline Tests
# ══════════════════════════════════════════════════════════════


class TestGetTimeline:
    """Test timeline retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_get_timeline_success(self, mock_user, completed_session):
        """Test successful timeline retrieval."""
        from subtext.api.routes.sessions import get_timeline

        response = await get_timeline(completed_session["id"], mock_user)

        assert response.duration_ms == 120000
        assert response.resolution_ms == 5000
        assert len(response.data_points) == 2

    @pytest.mark.asyncio
    async def test_get_timeline_not_found(self, mock_user):
        """Test getting timeline for nonexistent session."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_timeline

        with pytest.raises(HTTPException) as exc:
            await get_timeline(uuid4(), mock_user)

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_timeline_wrong_org(self, mock_other_user, completed_session):
        """Test getting timeline from different org."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_timeline

        with pytest.raises(HTTPException) as exc:
            await get_timeline(completed_session["id"], mock_other_user)

        assert exc.value.status_code == 403


# ══════════════════════════════════════════════════════════════
# Get Insights Tests
# ══════════════════════════════════════════════════════════════


class TestGetInsights:
    """Test insights retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_get_insights_success(self, mock_user, completed_session):
        """Test successful insights retrieval."""
        from subtext.api.routes.sessions import get_insights

        response = await get_insights(completed_session["id"], mock_user)

        assert response.summary == "A productive conversation"
        assert len(response.key_moments) == 1
        assert "Follow up on topic A" in response.recommendations

    @pytest.mark.asyncio
    async def test_get_insights_not_found(self, mock_user):
        """Test getting insights for nonexistent session."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_insights

        with pytest.raises(HTTPException) as exc:
            await get_insights(uuid4(), mock_user)

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_insights_not_complete(self, mock_user, sample_session):
        """Test getting insights before analysis complete."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import get_insights

        with pytest.raises(HTTPException) as exc:
            await get_insights(sample_session["id"], mock_user)

        assert exc.value.status_code == 400


# ══════════════════════════════════════════════════════════════
# Delete Session Tests
# ══════════════════════════════════════════════════════════════


class TestDeleteSession:
    """Test session deletion endpoint."""

    @pytest.mark.asyncio
    async def test_delete_session_success(self, mock_user, sample_session):
        """Test successful session deletion."""
        from subtext.api.routes.sessions import delete_session

        session_id = sample_session["id"]
        assert session_id in _sessions

        await delete_session(session_id, mock_user)

        assert session_id not in _sessions

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, mock_user):
        """Test deleting nonexistent session."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import delete_session

        with pytest.raises(HTTPException) as exc:
            await delete_session(uuid4(), mock_user)

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session_wrong_org(self, mock_other_user, sample_session):
        """Test deleting session from different org."""
        from fastapi import HTTPException
        from subtext.api.routes.sessions import delete_session

        with pytest.raises(HTTPException) as exc:
            await delete_session(sample_session["id"], mock_other_user)

        assert exc.value.status_code == 403


# ══════════════════════════════════════════════════════════════
# Process Session Tests
# ══════════════════════════════════════════════════════════════


class TestProcessSession:
    """Test background session processing."""

    @pytest.mark.asyncio
    async def test_process_session_not_found(self, mock_user):
        """Test processing nonexistent session."""
        # Should return without error
        await _process_session(uuid4(), "/tmp/test.wav", "test@example.com")

    @pytest.mark.asyncio
    async def test_process_session_success(self, mock_user, sample_session):
        """Test successful session processing."""
        import tempfile
        from pathlib import Path

        # Create temp audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"RIFF" + b"\x00" * 100)
            audio_path = f.name

        # Mock the pipeline and email service
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.duration_ms = 60000
        mock_result.speakers = [{"id": "A"}]
        mock_result.transcript_segments = [{"text": "Hello"}]
        mock_result.signals = []
        mock_result.timeline = []
        mock_result.insights = {}
        mock_result.speaker_metrics = []
        mock_result.language = "en"

        with patch("subtext.pipeline.PipelineOrchestrator") as mock_pipeline_cls:
            mock_pipeline = AsyncMock()
            mock_pipeline.process_file = AsyncMock(return_value=mock_result)
            mock_pipeline_cls.return_value = mock_pipeline

            with patch("subtext.integrations.resend.get_email_service") as mock_email:
                mock_email_service = AsyncMock()
                mock_email.return_value = mock_email_service

                await _process_session(
                    sample_session["id"],
                    audio_path,
                    mock_user.email,
                )

        # Verify session was updated
        assert sample_session["status"] == SessionStatus.COMPLETED
        assert sample_session["duration_ms"] == 60000

        # Cleanup
        Path(audio_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_process_session_failure(self, mock_user, sample_session):
        """Test session processing failure."""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"data")
            audio_path = f.name

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Processing failed"

        with patch("subtext.pipeline.PipelineOrchestrator") as mock_pipeline_cls:
            mock_pipeline = AsyncMock()
            mock_pipeline.process_file = AsyncMock(return_value=mock_result)
            mock_pipeline_cls.return_value = mock_pipeline

            await _process_session(
                sample_session["id"],
                audio_path,
                mock_user.email,
            )

        assert sample_session["status"] == SessionStatus.FAILED
        assert sample_session["error"] == "Processing failed"

        Path(audio_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_process_session_exception(self, mock_user, sample_session):
        """Test session processing with exception."""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"data")
            audio_path = f.name

        with patch("subtext.pipeline.PipelineOrchestrator") as mock_pipeline_cls:
            mock_pipeline_cls.side_effect = Exception("Pipeline error")

            await _process_session(
                sample_session["id"],
                audio_path,
                mock_user.email,
            )

        assert sample_session["status"] == SessionStatus.FAILED
        assert "Pipeline error" in sample_session["error"]

        Path(audio_path).unlink(missing_ok=True)
