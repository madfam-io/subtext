"""
Session Routes

Handle audio analysis sessions - create, upload, process, and retrieve results.
"""

import asyncio
from typing import Any
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel

from subtext.core.models import SessionCreate, SessionResponse, SessionStatus
from subtext.integrations.janua import TokenPayload, get_current_user

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request/Response Models
# ══════════════════════════════════════════════════════════════


class SessionListResponse(BaseModel):
    """Paginated session list response."""

    sessions: list[SessionResponse]
    total: int
    cursor: str | None = None


class UploadResponse(BaseModel):
    """Response after uploading audio for analysis."""

    session_id: UUID
    status: SessionStatus
    message: str


class TranscriptResponse(BaseModel):
    """Transcript response."""

    segments: list[dict[str, Any]]
    speakers: list[dict[str, Any]]
    full_text: str


class SignalListResponse(BaseModel):
    """Signals detected in a session."""

    signals: list[dict[str, Any]]
    total: int
    summary: dict[str, int]


class TimelineResponse(BaseModel):
    """Tension timeline response."""

    duration_ms: int
    resolution_ms: int
    data_points: list[dict[str, Any]]


class InsightsResponse(BaseModel):
    """AI-generated insights response."""

    summary: str
    key_moments: list[dict[str, Any]]
    recommendations: list[str]
    speaker_metrics: list[dict[str, Any]]
    risk_flags: list[dict[str, Any]]


class AnalysisResponse(BaseModel):
    """Complete analysis response."""

    session: SessionResponse
    transcript: TranscriptResponse
    signals: SignalListResponse
    timeline: TimelineResponse
    insights: InsightsResponse


# ══════════════════════════════════════════════════════════════
# In-Memory Store (would be database in production)
# ══════════════════════════════════════════════════════════════

# Temporary in-memory storage for demo
_sessions: dict[UUID, dict[str, Any]] = {}
_processing_tasks: dict[UUID, asyncio.Task] = {}


# ══════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    request: SessionCreate,
    user: TokenPayload = Depends(get_current_user),
) -> SessionResponse:
    """Create a new analysis session."""
    from datetime import datetime

    session_id = uuid4()

    session = {
        "id": session_id,
        "org_id": user.org_id,
        "created_by": user.sub,
        "name": request.name,
        "description": request.description,
        "status": SessionStatus.PENDING,
        "language": request.language,
        "settings": request.settings,
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "duration_ms": None,
        "speaker_count": None,
        "signal_count": None,
    }

    _sessions[session_id] = session

    return SessionResponse(
        id=session_id,
        name=request.name,
        status=SessionStatus.PENDING,
        duration_ms=None,
        speaker_count=None,
        signal_count=None,
        created_at=session["created_at"],
        completed_at=None,
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    user: TokenPayload = Depends(get_current_user),
    status_filter: SessionStatus | None = Query(None, alias="status"),
    limit: int = Query(20, le=100),
    cursor: str | None = None,
) -> SessionListResponse:
    """List sessions for the current organization."""
    # Filter sessions by org
    org_sessions = [
        s for s in _sessions.values() if s.get("org_id") == user.org_id
    ]

    # Apply status filter
    if status_filter:
        org_sessions = [s for s in org_sessions if s["status"] == status_filter]

    # Sort by created_at descending
    org_sessions.sort(key=lambda s: s["created_at"], reverse=True)

    # Apply pagination (simplified)
    total = len(org_sessions)
    org_sessions = org_sessions[:limit]

    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=s["id"],
                name=s["name"],
                status=s["status"],
                duration_ms=s.get("duration_ms"),
                speaker_count=s.get("speaker_count"),
                signal_count=s.get("signal_count"),
                created_at=s["created_at"],
                completed_at=s.get("completed_at"),
            )
            for s in org_sessions
        ],
        total=total,
        cursor=None,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    user: TokenPayload = Depends(get_current_user),
) -> SessionResponse:
    """Get session details."""
    session = _sessions.get(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.get("org_id") != user.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session",
        )

    return SessionResponse(
        id=session["id"],
        name=session["name"],
        status=session["status"],
        duration_ms=session.get("duration_ms"),
        speaker_count=session.get("speaker_count"),
        signal_count=session.get("signal_count"),
        created_at=session["created_at"],
        completed_at=session.get("completed_at"),
    )


@router.post("/{session_id}/upload", response_model=UploadResponse)
async def upload_audio(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: TokenPayload = Depends(get_current_user),
) -> UploadResponse:
    """Upload audio file for analysis."""
    session = _sessions.get(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.get("org_id") != user.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized",
        )

    if session["status"] != SessionStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session already has audio uploaded",
        )

    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/webm", "audio/ogg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format. Allowed: {allowed_types}",
        )

    # Save file temporarily
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Update session status
    session["status"] = SessionStatus.UPLOADING
    session["audio_path"] = tmp_path

    # Start background processing
    background_tasks.add_task(
        _process_session,
        session_id=session_id,
        audio_path=tmp_path,
        user_email=user.email,
    )

    return UploadResponse(
        session_id=session_id,
        status=SessionStatus.UPLOADING,
        message="Audio uploaded. Processing started.",
    )


async def _process_session(
    session_id: UUID,
    audio_path: str,
    user_email: str,
) -> None:
    """Background task to process uploaded audio."""
    from datetime import datetime
    from pathlib import Path

    import structlog

    logger = structlog.get_logger()

    session = _sessions.get(session_id)
    if not session:
        return

    try:
        session["status"] = SessionStatus.PROCESSING
        logger.info("Processing started", session_id=str(session_id))

        # Run pipeline
        from subtext.pipeline import PipelineOrchestrator, PipelineConfig

        config = PipelineConfig(
            language=session.get("language"),
        )
        pipeline = PipelineOrchestrator(config)
        result = await pipeline.process_file(session_id, audio_path)

        if result.success:
            # Store results
            session["status"] = SessionStatus.COMPLETED
            session["completed_at"] = datetime.utcnow()
            session["duration_ms"] = result.duration_ms
            session["speaker_count"] = len(result.speakers)
            session["signal_count"] = len(result.signals)
            session["language"] = result.language
            session["result"] = {
                "speakers": result.speakers,
                "transcript_segments": result.transcript_segments,
                "signals": [s.model_dump() for s in result.signals],
                "timeline": result.timeline,
                "insights": result.insights,
                "speaker_metrics": result.speaker_metrics,
            }

            logger.info(
                "Processing complete",
                session_id=str(session_id),
                duration_ms=result.duration_ms,
                signal_count=len(result.signals),
            )

            # Send completion email
            from subtext.integrations.resend import get_email_service

            email_service = get_email_service()
            top_signals = sorted(
                [s.model_dump() for s in result.signals],
                key=lambda x: x.get("intensity", 0),
                reverse=True,
            )[:5]

            await email_service.send_analysis_complete(
                email=user_email,
                name=None,
                session_name=session["name"],
                session_id=str(session_id),
                duration_minutes=result.duration_ms / 60000,
                speaker_count=len(result.speakers),
                signal_count=len(result.signals),
                top_signals=top_signals,
            )

        else:
            session["status"] = SessionStatus.FAILED
            session["error"] = result.error
            logger.error("Processing failed", session_id=str(session_id), error=result.error)

    except Exception as e:
        session["status"] = SessionStatus.FAILED
        session["error"] = str(e)
        logger.error("Processing exception", session_id=str(session_id), error=str(e))

    finally:
        # Clean up temp file
        Path(audio_path).unlink(missing_ok=True)


@router.get("/{session_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(
    session_id: UUID,
    user: TokenPayload = Depends(get_current_user),
) -> TranscriptResponse:
    """Get session transcript."""
    session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("org_id") != user.org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if session["status"] != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not complete")

    result = session.get("result", {})

    return TranscriptResponse(
        segments=result.get("transcript_segments", []),
        speakers=result.get("speakers", []),
        full_text=" ".join(
            s.get("text", "") for s in result.get("transcript_segments", [])
        ),
    )


@router.get("/{session_id}/signals", response_model=SignalListResponse)
async def get_signals(
    session_id: UUID,
    user: TokenPayload = Depends(get_current_user),
    signal_type: str | None = Query(None, alias="type"),
    min_confidence: float = Query(0.5, ge=0, le=1),
) -> SignalListResponse:
    """Get detected signals for a session."""
    session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("org_id") != user.org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if session["status"] != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not complete")

    result = session.get("result", {})
    signals = result.get("signals", [])

    # Filter by type
    if signal_type:
        signals = [s for s in signals if s.get("signal_type") == signal_type]

    # Filter by confidence
    signals = [s for s in signals if s.get("confidence", 0) >= min_confidence]

    # Build summary
    summary: dict[str, int] = {}
    for s in signals:
        st = s.get("signal_type", "unknown")
        summary[st] = summary.get(st, 0) + 1

    return SignalListResponse(
        signals=signals,
        total=len(signals),
        summary=summary,
    )


@router.get("/{session_id}/timeline", response_model=TimelineResponse)
async def get_timeline(
    session_id: UUID,
    user: TokenPayload = Depends(get_current_user),
) -> TimelineResponse:
    """Get tension timeline for a session."""
    session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("org_id") != user.org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if session["status"] != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not complete")

    result = session.get("result", {})
    timeline = result.get("timeline", [])

    return TimelineResponse(
        duration_ms=session.get("duration_ms", 0),
        resolution_ms=5000,
        data_points=timeline,
    )


@router.get("/{session_id}/insights", response_model=InsightsResponse)
async def get_insights(
    session_id: UUID,
    user: TokenPayload = Depends(get_current_user),
) -> InsightsResponse:
    """Get AI-generated insights for a session."""
    session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("org_id") != user.org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if session["status"] != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not complete")

    result = session.get("result", {})
    insights = result.get("insights", {})

    return InsightsResponse(
        summary=insights.get("summary", ""),
        key_moments=insights.get("key_moments", []),
        recommendations=insights.get("recommendations", []),
        speaker_metrics=result.get("speaker_metrics", []),
        risk_flags=[],  # Would extract from insights
    )


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    user: TokenPayload = Depends(get_current_user),
) -> None:
    """Delete a session and its data."""
    session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("org_id") != user.org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    del _sessions[session_id]
