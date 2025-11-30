"""
Subtext CLI

Command-line interface for the Subtext platform.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

import click
import structlog

from subtext.config import settings

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# CLI Group
# ══════════════════════════════════════════════════════════════


@click.group()
@click.version_option(version="0.1.0", prog_name="subtext")
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
def cli(debug: bool) -> None:
    """Subtext - Metacognitive Engine for Conversational Intelligence.

    Read the room, not just the transcript.
    """
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        )


# ══════════════════════════════════════════════════════════════
# Server Commands
# ══════════════════════════════════════════════════════════════


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload")
@click.option("--workers", default=1, help="Number of worker processes")
def serve(host: str, port: int, reload: bool, workers: int) -> None:
    """Start the Subtext API server."""
    import uvicorn

    click.echo(f"Starting Subtext API on {host}:{port}")

    uvicorn.run(
        "subtext.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        factory=True,
    )


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8001, help="Port to bind to")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload")
def realtime(host: str, port: int, reload: bool) -> None:
    """Start the realtime WebSocket server for live audio processing.

    The realtime server handles WebSocket connections for:
    - Live audio streaming and processing
    - ESP (Emotional State Protocol) broadcasting
    - Real-time transcription and signal detection
    """
    import uvicorn

    click.echo(f"Starting Subtext Realtime WebSocket server on ws://{host}:{port}")
    click.echo("Endpoints:")
    click.echo(f"  - ws://{host}:{port}/ws/realtime")
    click.echo(f"  - ws://{host}:{port}/ws/esp/subscribe/{{session_id}}")

    uvicorn.run(
        "subtext.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


# ══════════════════════════════════════════════════════════════
# Pipeline Commands
# ══════════════════════════════════════════════════════════════


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--language", "-l", default=None, help="Language code (auto-detect if not set)")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output JSON file")
@click.option("--backend", "-b", default="whisperx", type=click.Choice(["whisperx", "canary", "parakeet"]), help="ASR backend")
@click.option("--no-emotion/--emotion", default=False, help="Disable emotion detection")
@click.option("--no-vad/--vad", default=False, help="Disable VAD")
def analyze(
    audio_file: Path,
    language: Optional[str],
    output: Optional[Path],
    backend: str,
    no_emotion: bool,
    no_vad: bool,
) -> None:
    """Analyze an audio file and extract conversational intelligence.

    AUDIO_FILE: Path to the audio file to analyze (WAV, MP3, etc.)
    """
    click.echo(f"Analyzing: {audio_file}")
    click.echo(f"Backend: {backend}")

    async def run_analysis():
        from subtext.pipeline import create_pipeline, PipelineResult

        pipeline = create_pipeline(
            language=language,
            asr_backend=backend,
            enable_vad=not no_vad,
            enable_emotion=not no_emotion,
        )

        session_id = uuid4()
        result: PipelineResult = await pipeline.process_file(session_id, str(audio_file))

        if result.success:
            click.echo(f"\n✓ Analysis complete!")
            click.echo(f"  Duration: {result.duration_ms / 1000:.1f}s")
            click.echo(f"  Speakers: {len(result.speakers)}")
            click.echo(f"  Signals: {len(result.signals)}")
            click.echo(f"  Processing time: {result.processing_time_ms / 1000:.1f}s")

            if result.dominant_emotion:
                click.echo(f"  Dominant emotion: {result.dominant_emotion}")

            if output:
                import json
                output_data = {
                    "session_id": str(session_id),
                    "duration_ms": result.duration_ms,
                    "language": result.language,
                    "asr_backend": result.asr_backend,
                    "speakers": result.speakers,
                    "transcript_segments": result.transcript_segments,
                    "signals": [s.model_dump() for s in result.signals],
                    "timeline": result.timeline,
                    "insights": result.insights,
                    "speaker_metrics": result.speaker_metrics,
                    "speech_ratio": result.speech_ratio,
                    "emotions": result.emotions,
                    "dominant_emotion": result.dominant_emotion,
                    "vad_scores": result.vad_scores,
                }
                output.write_text(json.dumps(output_data, indent=2, default=str))
                click.echo(f"  Output: {output}")
        else:
            click.echo(f"\n✗ Analysis failed: {result.error}", err=True)
            sys.exit(1)

    asyncio.run(run_analysis())


@cli.command()
def models() -> None:
    """List available ML models and their status."""
    click.echo("Subtext ML Model Stack (2025)\n")

    model_info = [
        ("VAD", "Silero VAD", settings.silero_vad_model, "87.7% TPR"),
        ("Noise Suppression", "DeepFilterNet", settings.deepfilternet_model, "SOTA real-time"),
        ("Diarization", "Pyannote", settings.pyannote_model, "10% DER"),
        ("Speaker Embedding", "ECAPA-TDNN", settings.speaker_embedding_model, "1.71% EER"),
        ("ASR (default)", "WhisperX", settings.whisper_model, "Multilingual"),
        ("ASR (accuracy)", "NVIDIA Canary", settings.asr_model_accuracy, "5.63% WER"),
        ("ASR (speed)", "NVIDIA Parakeet", settings.asr_model_speed, "2000+ RTFx"),
        ("Emotion", "Emotion2Vec", settings.emotion_model, "SOTA 9 datasets"),
        ("LLM (cloud)", "GPT-4", settings.llm_model, "Insights"),
        ("LLM (local)", "Llama 3.1", settings.llm_model_local, "70B params"),
    ]

    for stage, name, model_id, metric in model_info:
        click.echo(f"  {stage:20} {name:15} {metric:15}")
        click.echo(f"                       {model_id}")
        click.echo()


# ══════════════════════════════════════════════════════════════
# Database Commands
# ══════════════════════════════════════════════════════════════


@cli.group()
def db() -> None:
    """Database management commands."""
    pass


@db.command("init")
def db_init() -> None:
    """Initialize the database schema."""
    click.echo("Initializing database...")

    async def init():
        from subtext.db import init_db, Base, close_db
        from sqlalchemy import text

        await init_db()

        # Import all models
        from subtext.db.models import (
            OrganizationModel, UserModel, SessionModel,
            SpeakerModel, TranscriptSegmentModel, SignalModel,
            ProsodicsModel, InsightModel, TimelineModel,
            VoiceFingerprintModel, APIKeyModel, UsageRecordModel,
        )

        click.echo("Database initialized. Run migrations with: alembic upgrade head")
        await close_db()

    asyncio.run(init())


@db.command("migrate")
@click.option("--message", "-m", required=True, help="Migration message")
def db_migrate(message: str) -> None:
    """Create a new migration."""
    import subprocess
    result = subprocess.run(
        ["alembic", "revision", "--autogenerate", "-m", message],
        capture_output=True,
        text=True,
    )
    click.echo(result.stdout)
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        sys.exit(1)


@db.command("upgrade")
@click.argument("revision", default="head")
def db_upgrade(revision: str) -> None:
    """Upgrade database to a revision."""
    import subprocess
    result = subprocess.run(
        ["alembic", "upgrade", revision],
        capture_output=True,
        text=True,
    )
    click.echo(result.stdout)
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        sys.exit(1)


@db.command("downgrade")
@click.argument("revision", default="-1")
def db_downgrade(revision: str) -> None:
    """Downgrade database to a revision."""
    import subprocess
    result = subprocess.run(
        ["alembic", "downgrade", revision],
        capture_output=True,
        text=True,
    )
    click.echo(result.stdout)
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# Worker Commands
# ══════════════════════════════════════════════════════════════


@cli.group()
def worker() -> None:
    """Background worker commands."""
    pass


@worker.command("start")
@click.option("--concurrency", "-c", default=None, type=int, help="Max concurrent jobs")
@click.option(
    "--queue", "-q",
    type=click.Choice(["high", "default", "low"]),
    default="default",
    help="Queue to process",
)
@click.option("--burst/--no-burst", default=False, help="Exit when queue is empty")
def worker_start(concurrency: int | None, queue: str, burst: bool) -> None:
    """Start an ARQ background worker for pipeline processing."""
    from arq import run_worker as arq_run_worker
    from subtext.worker.queue import get_worker_settings, QUEUE_NAMES, JobPriority

    click.echo(f"Starting Subtext ARQ worker (queue={queue})")

    settings_dict = get_worker_settings()

    # Override queue if specified
    priority = {"high": JobPriority.HIGH, "default": JobPriority.NORMAL, "low": JobPriority.LOW}
    settings_dict["queue_name"] = QUEUE_NAMES[priority[queue]]

    # Override concurrency if specified
    if concurrency:
        settings_dict["max_jobs"] = concurrency

    # Burst mode for CI/testing
    if burst:
        settings_dict["burst"] = True

    click.echo(f"Worker starting with settings: queue={settings_dict['queue_name']}, max_jobs={settings_dict['max_jobs']}")

    arq_run_worker(settings_dict)


@worker.command("enqueue")
@click.argument("task_name")
@click.option("--session-id", "-s", required=True, help="Session ID to process")
@click.option("--priority", "-p", type=click.Choice(["high", "normal", "low"]), default="normal")
def worker_enqueue(task_name: str, session_id: str, priority: str) -> None:
    """Manually enqueue a task for processing."""
    from subtext.worker.queue import enqueue_job, JobPriority

    priority_map = {
        "high": JobPriority.HIGH,
        "normal": JobPriority.NORMAL,
        "low": JobPriority.LOW,
    }

    async def enqueue():
        job_id = await enqueue_job(
            task_name,
            session_id=session_id,
            priority=priority_map[priority],
        )
        if job_id:
            click.echo(f"Task enqueued: {job_id}")
        else:
            click.echo("Failed to enqueue task", err=True)
            sys.exit(1)

    asyncio.run(enqueue())


@worker.command("status")
@click.argument("job_id", required=False)
def worker_status(job_id: str | None) -> None:
    """Check worker/job status."""
    from subtext.worker.queue import get_job_status, get_redis_pool

    async def check_status():
        if job_id:
            status = await get_job_status(job_id)
            if status:
                click.echo(f"Job: {status['job_id']}")
                click.echo(f"  Status: {status['status']}")
                if status['result']:
                    click.echo(f"  Result: {status['result']}")
            else:
                click.echo(f"Job not found: {job_id}")
        else:
            # Show queue stats
            pool = await get_redis_pool()
            info = await pool.info()
            click.echo("ARQ Worker Status")
            click.echo(f"  Connected: {info is not None}")

    asyncio.run(check_status())


# ══════════════════════════════════════════════════════════════
# Config Commands
# ══════════════════════════════════════════════════════════════


@cli.command()
def config() -> None:
    """Show current configuration."""
    click.echo("Subtext Configuration\n")

    config_items = [
        ("Environment", settings.app_env),
        ("Debug", str(settings.debug)),
        ("Database", settings.database_url.unicode_string() if settings.database_url else "Not set"),
        ("Redis", str(settings.redis_url)),
        ("Janua URL", settings.janua_base_url),
        ("Model Cache", settings.model_cache_dir),
        ("ASR Model", settings.asr_model),
        ("Emotion Model", settings.emotion_model),
        ("LLM Provider", settings.llm_provider),
    ]

    for key, value in config_items:
        # Mask sensitive values
        if "key" in key.lower() or "secret" in key.lower():
            value = "***" if value else "Not set"
        click.echo(f"  {key:20} {value}")


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
