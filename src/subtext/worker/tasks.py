"""
Background Tasks

Defines all background jobs that can be executed by workers.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from arq import ArqRedis
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from subtext.config import settings
from subtext.core.models import SessionStatus, SignalType
from subtext.db import get_session, SessionModel, SignalModel, InsightModel, SpeakerModel, TranscriptSegmentModel, ProsodicsModel

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# Worker Lifecycle
# ══════════════════════════════════════════════════════════════


async def startup(ctx: dict[str, Any]) -> None:
    """Initialize worker resources on startup."""
    logger.info("Worker starting up")

    # Initialize database connection
    from subtext.db import init_db

    await init_db()

    # Store in context
    ctx["db_initialized"] = True

    # Pre-load ML models if configured
    if settings.worker_preload_models:
        await _preload_models(ctx)

    logger.info("Worker startup complete")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Cleanup worker resources on shutdown."""
    logger.info("Worker shutting down")

    # Close database
    from subtext.db import close_db

    await close_db()

    logger.info("Worker shutdown complete")


async def _preload_models(ctx: dict[str, Any]) -> None:
    """Pre-load ML models for faster job execution."""
    try:
        # Load VAD model
        import torch

        vad_model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        ctx["vad_model"] = vad_model
        ctx["vad_utils"] = utils
        logger.info("VAD model preloaded")

    except Exception as e:
        logger.warning(f"Failed to preload VAD model: {e}")

    try:
        # Load speaker embedding model
        from speechbrain.inference.speaker import EncoderClassifier

        embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_models",
            run_opts={"device": "cpu"},
        )
        ctx["embedding_model"] = embedding_model
        logger.info("Speaker embedding model preloaded")

    except Exception as e:
        logger.warning(f"Failed to preload embedding model: {e}")


# ══════════════════════════════════════════════════════════════
# Audio Processing Tasks
# ══════════════════════════════════════════════════════════════


async def process_audio_file(
    ctx: dict[str, Any],
    session_id: str,
    file_path: str,
    asr_backend: str = "whisperx",
    enable_diarization: bool = True,
    enable_emotion: bool = True,
    enable_signals: bool = True,
) -> dict[str, Any]:
    """
    Process an uploaded audio file through the full pipeline.

    This is the main batch processing task.

    Args:
        ctx: ARQ context
        session_id: UUID of the session
        file_path: Path to the audio file
        asr_backend: ASR backend to use
        enable_diarization: Whether to run diarization
        enable_emotion: Whether to run emotion detection
        enable_signals: Whether to detect signals

    Returns:
        Processing result summary
    """
    session_uuid = UUID(session_id)
    start_time = datetime.utcnow()

    logger.info(
        "Starting audio processing",
        session_id=session_id,
        file_path=file_path,
        asr_backend=asr_backend,
    )

    try:
        # Update session status
        await _update_session_status(session_uuid, SessionStatus.PROCESSING)

        # Create pipeline
        from subtext.pipeline.orchestrator import create_pipeline

        pipeline = create_pipeline(
            asr_backend=asr_backend,
            enable_diarization=enable_diarization,
            enable_emotion=enable_emotion,
            enable_signals=enable_signals,
        )

        # Process file
        result = await pipeline.process_file(session_uuid, file_path)

        # Save results to database
        await _save_processing_results(session_uuid, result)

        # Update session status
        processing_time_ms = int(
            (datetime.utcnow() - start_time).total_seconds() * 1000
        )

        await _update_session_completed(
            session_uuid,
            duration_ms=result.get("duration_ms", 0),
            speaker_count=result.get("speaker_count", 0),
            signal_count=result.get("signal_count", 0),
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "Audio processing complete",
            session_id=session_id,
            duration_ms=result.get("duration_ms", 0),
            processing_time_ms=processing_time_ms,
        )

        return {
            "success": True,
            "session_id": session_id,
            "duration_ms": result.get("duration_ms", 0),
            "speaker_count": result.get("speaker_count", 0),
            "signal_count": result.get("signal_count", 0),
            "processing_time_ms": processing_time_ms,
        }

    except Exception as e:
        logger.error(
            "Audio processing failed",
            session_id=session_id,
            error=str(e),
        )

        await _update_session_status(
            session_uuid,
            SessionStatus.FAILED,
            error_message=str(e),
        )

        return {
            "success": False,
            "session_id": session_id,
            "error": str(e),
        }


async def process_realtime_session(
    ctx: dict[str, Any],
    session_id: str,
    audio_path: str | None = None,
) -> dict[str, Any]:
    """
    Post-process a completed realtime session.

    Runs additional analysis and generates insights after
    the realtime stream has ended.

    Args:
        ctx: ARQ context
        session_id: UUID of the session
        audio_path: Path to saved audio (if retention enabled)

    Returns:
        Processing result
    """
    session_uuid = UUID(session_id)

    logger.info(
        "Post-processing realtime session",
        session_id=session_id,
    )

    try:
        # Get session data from database
        session_data = await _get_session_data(session_uuid)
        if not session_data:
            raise ValueError(f"Session not found: {session_id}")

        # Generate insights
        insights = await _generate_insights(session_data)

        # Save insights
        await _save_insights(session_uuid, insights)

        # If audio was saved, run additional analysis
        if audio_path and Path(audio_path).exists():
            # Could run more thorough analysis here
            pass

        logger.info(
            "Realtime session post-processing complete",
            session_id=session_id,
            insight_count=len(insights),
        )

        return {
            "success": True,
            "session_id": session_id,
            "insight_count": len(insights),
        }

    except Exception as e:
        logger.error(
            "Realtime session post-processing failed",
            session_id=session_id,
            error=str(e),
        )

        return {
            "success": False,
            "session_id": session_id,
            "error": str(e),
        }


# ══════════════════════════════════════════════════════════════
# Insight Generation Tasks
# ══════════════════════════════════════════════════════════════


async def generate_session_insights(
    ctx: dict[str, Any],
    session_id: str,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Generate AI-powered insights for a session.

    Args:
        ctx: ARQ context
        session_id: UUID of the session
        use_llm: Whether to use LLM for insight generation

    Returns:
        Generated insights
    """
    session_uuid = UUID(session_id)

    logger.info(
        "Generating session insights",
        session_id=session_id,
        use_llm=use_llm,
    )

    try:
        # Get session data
        session_data = await _get_session_data(session_uuid)
        if not session_data:
            raise ValueError(f"Session not found: {session_id}")

        insights = []

        # Generate summary insight
        summary = await _generate_summary(session_data, use_llm)
        insights.append({
            "type": "summary",
            "content": summary,
            "importance": 1.0,
        })

        # Find key moments
        key_moments = await _find_key_moments(session_data)
        for moment in key_moments:
            insights.append({
                "type": "key_moment",
                "content": moment,
                "importance": moment.get("importance", 0.7),
            })

        # Identify risk flags
        risk_flags = await _identify_risk_flags(session_data)
        for flag in risk_flags:
            insights.append({
                "type": "risk_flag",
                "content": flag,
                "importance": 0.9 if flag.get("severity") == "high" else 0.6,
            })

        # Generate recommendations
        recommendations = await _generate_recommendations(session_data, use_llm)
        for rec in recommendations:
            insights.append({
                "type": "recommendation",
                "content": rec,
                "importance": 0.5,
            })

        # Save insights
        await _save_insights(session_uuid, insights)

        logger.info(
            "Session insights generated",
            session_id=session_id,
            insight_count=len(insights),
        )

        return {
            "success": True,
            "session_id": session_id,
            "insights": insights,
        }

    except Exception as e:
        logger.error(
            "Insight generation failed",
            session_id=session_id,
            error=str(e),
        )

        return {
            "success": False,
            "session_id": session_id,
            "error": str(e),
        }


# ══════════════════════════════════════════════════════════════
# Export Tasks
# ══════════════════════════════════════════════════════════════


async def export_session_data(
    ctx: dict[str, Any],
    session_id: str,
    format: str = "json",
    include_audio: bool = False,
) -> dict[str, Any]:
    """
    Export session data in various formats.

    Args:
        ctx: ARQ context
        session_id: UUID of the session
        format: Export format (json, csv, pdf)
        include_audio: Whether to include audio file

    Returns:
        Export result with download URL
    """
    session_uuid = UUID(session_id)

    logger.info(
        "Exporting session data",
        session_id=session_id,
        format=format,
    )

    try:
        # Get full session data
        session_data = await _get_full_session_export(session_uuid)

        # Generate export based on format
        if format == "json":
            export_path = await _export_json(session_uuid, session_data)
        elif format == "csv":
            export_path = await _export_csv(session_uuid, session_data)
        elif format == "pdf":
            export_path = await _export_pdf(session_uuid, session_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        # Upload to storage
        download_url = await _upload_to_storage(export_path, session_uuid, format)

        logger.info(
            "Session export complete",
            session_id=session_id,
            format=format,
        )

        return {
            "success": True,
            "session_id": session_id,
            "format": format,
            "download_url": download_url,
        }

    except Exception as e:
        logger.error(
            "Session export failed",
            session_id=session_id,
            error=str(e),
        )

        return {
            "success": False,
            "session_id": session_id,
            "error": str(e),
        }


# ══════════════════════════════════════════════════════════════
# Cleanup Tasks
# ══════════════════════════════════════════════════════════════


async def cleanup_expired_sessions(
    ctx: dict[str, Any],
    retention_days: int = 30,
) -> dict[str, Any]:
    """
    Clean up expired sessions and associated data.

    Args:
        ctx: ARQ context
        retention_days: Number of days to retain data

    Returns:
        Cleanup result
    """
    logger.info(
        "Starting session cleanup",
        retention_days=retention_days,
    )

    try:
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        # Find expired sessions
        expired_sessions = await _find_expired_sessions(cutoff_date)

        deleted_count = 0
        for session in expired_sessions:
            try:
                await _delete_session_data(session["id"])
                deleted_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to delete session {session['id']}: {e}"
                )

        logger.info(
            "Session cleanup complete",
            deleted_count=deleted_count,
            total_expired=len(expired_sessions),
        )

        return {
            "success": True,
            "deleted_count": deleted_count,
            "total_expired": len(expired_sessions),
        }

    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")

        return {
            "success": False,
            "error": str(e),
        }


# ══════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════


async def _update_session_status(
    session_id: UUID,
    status: SessionStatus,
    error_message: str | None = None,
) -> None:
    """Update session status in database."""
    async with get_session() as session:
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(status=status, error_message=error_message)
        )
        await session.execute(stmt)
        logger.info("Session status updated", session_id=str(session_id), status=status.value)


async def _update_session_completed(
    session_id: UUID,
    duration_ms: int,
    speaker_count: int,
    signal_count: int,
    processing_time_ms: int,
) -> None:
    """Update session with completion data."""
    async with get_session() as session:
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                status=SessionStatus.COMPLETED,
                duration_ms=duration_ms,
                speaker_count=speaker_count,
                signal_count=signal_count,
                processing_time_ms=processing_time_ms,
                completed_at=datetime.utcnow(),
            )
        )
        await session.execute(stmt)
        logger.info(
            "Session completed",
            session_id=str(session_id),
            duration_ms=duration_ms,
            speaker_count=speaker_count,
            signal_count=signal_count,
        )


async def _save_processing_results(
    session_id: UUID,
    result: dict[str, Any],
) -> None:
    """Save processing results to database."""
    async with get_session() as session:
        # Save signals
        if result.get("signals"):
            for signal_data in result["signals"]:
                signal = SignalModel(
                    session_id=session_id,
                    signal_type=signal_data.get("type"),
                    timestamp_ms=signal_data.get("timestamp_ms", 0),
                    duration_ms=signal_data.get("duration_ms"),
                    confidence=signal_data.get("confidence", 0.0),
                    intensity=signal_data.get("intensity", 0.0),
                    speaker_id=signal_data.get("speaker_id"),
                    context=signal_data.get("context", {}),
                )
                session.add(signal)

        # Update session metadata with results summary
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                session_metadata={
                    "processed_at": datetime.utcnow().isoformat(),
                    "signal_count": len(result.get("signals", [])),
                }
            )
        )
        await session.execute(stmt)
        logger.info("Processing results saved", session_id=str(session_id))


async def _get_session_data(session_id: UUID) -> dict[str, Any] | None:
    """Get session data from database."""
    async with get_session() as session:
        stmt = (
            select(SessionModel)
            .options(
                selectinload(SessionModel.signals),
                selectinload(SessionModel.speakers),
                selectinload(SessionModel.segments),
            )
            .where(SessionModel.id == session_id)
        )
        result = await session.execute(stmt)
        session_model = result.scalar_one_or_none()

        if not session_model:
            return None

        return {
            "id": str(session_model.id),
            "name": session_model.name,
            "status": session_model.status.value if session_model.status else None,
            "duration_ms": session_model.duration_ms,
            "speaker_count": session_model.speaker_count,
            "signals": [
                {
                    "type": s.signal_type,
                    "timestamp_ms": s.timestamp_ms,
                    "confidence": s.confidence,
                    "intensity": s.intensity,
                }
                for s in session_model.signals
            ],
            "speakers": [
                {"id": str(sp.id), "label": sp.label}
                for sp in session_model.speakers
            ],
            "transcript": [
                {
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "text": seg.text,
                    "speaker_label": seg.speaker.label if seg.speaker else None,
                }
                for seg in session_model.segments
            ],
        }


async def _generate_insights(session_data: dict) -> list[dict]:
    """Generate basic insights from session data."""
    insights = []

    # Placeholder implementation
    if session_data.get("signals"):
        signal_types = set(s["type"] for s in session_data["signals"])

        if SignalType.STRESS_SPIKE.value in signal_types:
            insights.append({
                "type": "observation",
                "content": "Elevated stress levels detected",
                "importance": 0.8,
            })

    return insights


async def _save_insights(session_id: UUID, insights: list[dict]) -> None:
    """Save insights to database."""
    async with get_session() as session:
        for insight_data in insights:
            insight = InsightModel(
                session_id=session_id,
                insight_type=insight_data.get("type", "observation"),
                content=insight_data.get("content", ""),
                importance=insight_data.get("importance", 0.5),
                context=insight_data.get("context", {}),
            )
            session.add(insight)
        logger.info("Insights saved", session_id=str(session_id), count=len(insights))


async def _generate_summary(
    session_data: dict,
    use_llm: bool,
) -> dict[str, Any]:
    """Generate session summary."""
    # Placeholder - would use LLM in production
    return {
        "duration_ms": session_data.get("duration_ms", 0),
        "speaker_count": session_data.get("speaker_count", 0),
        "signal_count": len(session_data.get("signals", [])),
        "text": "Session analysis complete.",
    }


async def _find_key_moments(session_data: dict) -> list[dict]:
    """Find key moments in session."""
    # Placeholder implementation
    return []


async def _identify_risk_flags(session_data: dict) -> list[dict]:
    """Identify risk flags in session."""
    # Placeholder implementation
    return []


async def _generate_recommendations(
    session_data: dict,
    use_llm: bool,
) -> list[dict]:
    """Generate recommendations based on session."""
    # Placeholder implementation
    return []


async def _get_full_session_export(session_id: UUID) -> dict[str, Any]:
    """Get full session data for export."""
    async with get_session() as session:
        stmt = (
            select(SessionModel)
            .options(
                selectinload(SessionModel.signals),
                selectinload(SessionModel.speakers),
                selectinload(SessionModel.segments),
                selectinload(SessionModel.prosodics),
                selectinload(SessionModel.insights),
            )
            .where(SessionModel.id == session_id)
        )
        result = await session.execute(stmt)
        session_model = result.scalar_one_or_none()

        if not session_model:
            return {}

        return {
            "session": {
                "id": str(session_model.id),
                "name": session_model.name,
                "status": session_model.status.value if session_model.status else None,
                "duration_ms": session_model.duration_ms,
                "speaker_count": session_model.speaker_count,
                "signal_count": session_model.signal_count,
                "created_at": session_model.created_at.isoformat() if session_model.created_at else None,
                "completed_at": session_model.completed_at.isoformat() if session_model.completed_at else None,
            },
            "transcript": [
                {
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "text": seg.text,
                    "speaker_label": seg.speaker.label if seg.speaker else None,
                    "confidence": seg.confidence,
                }
                for seg in session_model.segments
            ],
            "signals": [
                {
                    "type": s.signal_type,
                    "timestamp_ms": s.timestamp_ms,
                    "duration_ms": s.duration_ms,
                    "confidence": s.confidence,
                    "intensity": s.intensity,
                }
                for s in session_model.signals
            ],
            "speakers": [
                {"id": str(sp.id), "label": sp.label}
                for sp in session_model.speakers
            ],
            "insights": [
                {
                    "type": ins.insight_type,
                    "content": ins.content,
                    "importance": ins.importance,
                }
                for ins in session_model.insights
            ],
        }


async def _export_json(session_id: UUID, data: dict) -> str:
    """Export session as JSON."""
    import json
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        json.dump(data, f, indent=2, default=str)
        return f.name


async def _export_csv(session_id: UUID, data: dict) -> str:
    """Export session as CSV."""
    import csv
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
    ) as f:
        # Write transcript segments as CSV
        if data.get("transcript"):
            writer = csv.DictWriter(
                f,
                fieldnames=["start_ms", "end_ms", "speaker", "text"],
            )
            writer.writeheader()
            for segment in data["transcript"]:
                writer.writerow({
                    "start_ms": segment.get("start_ms", 0),
                    "end_ms": segment.get("end_ms", 0),
                    "speaker": segment.get("speaker_label", ""),
                    "text": segment.get("text", ""),
                })
        return f.name


async def _export_pdf(session_id: UUID, data: dict) -> str:
    """Export session as PDF."""
    # Placeholder - would use reportlab or similar
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(
        mode="w",
        suffix=".txt",  # Would be .pdf in production
        delete=False,
    ) as f:
        f.write(f"Session Report: {session_id}\n")
        f.write("=" * 50 + "\n")
        # Would generate proper PDF
        return f.name


async def _upload_to_storage(
    file_path: str,
    session_id: UUID,
    format: str,
) -> str:
    """Upload file to storage and return download URL."""
    # Placeholder - would upload to S3/R2
    return f"/downloads/{session_id}.{format}"


async def _find_expired_sessions(cutoff_date: datetime) -> list[dict]:
    """Find sessions older than cutoff date."""
    async with get_session() as session:
        stmt = (
            select(SessionModel)
            .where(SessionModel.created_at < cutoff_date)
            .where(SessionModel.status == SessionStatus.COMPLETED)
        )
        result = await session.execute(stmt)
        sessions = result.scalars().all()

        return [
            {
                "id": str(s.id),
                "name": s.name,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "org_id": str(s.org_id),
            }
            for s in sessions
        ]


async def _delete_session_data(session_id: UUID) -> None:
    """Delete session and all associated data."""
    async with get_session() as session:
        # Delete the session (cascades to related data)
        stmt = delete(SessionModel).where(SessionModel.id == session_id)
        await session.execute(stmt)
        logger.info("Session deleted", session_id=str(session_id))
