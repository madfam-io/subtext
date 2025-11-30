"""
Unit Tests for Worker Tasks Module

Tests the background task functions.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
import json

from subtext.core.models import SessionStatus, SignalType


# ══════════════════════════════════════════════════════════════
# Worker Lifecycle Tests
# ══════════════════════════════════════════════════════════════


class TestWorkerLifecycle:
    """Test worker startup and shutdown functions."""

    @pytest.mark.asyncio
    async def test_startup_initializes_db(self):
        """Test startup initializes database."""
        from subtext.worker.tasks import startup

        with patch("subtext.db.init_db", new_callable=AsyncMock) as mock_init_db:
            with patch("subtext.worker.tasks.settings") as mock_settings:
                mock_settings.worker_preload_models = False
                ctx = {}
                await startup(ctx)

        mock_init_db.assert_called_once()
        assert ctx["db_initialized"] is True

    @pytest.mark.asyncio
    async def test_shutdown_closes_db(self):
        """Test shutdown closes database."""
        from subtext.worker.tasks import shutdown

        with patch("subtext.db.close_db", new_callable=AsyncMock) as mock_close_db:
            ctx = {"db_initialized": True}
            await shutdown(ctx)

        mock_close_db.assert_called_once()


# ══════════════════════════════════════════════════════════════
# Process Audio File Tests
# ══════════════════════════════════════════════════════════════


class TestProcessAudioFile:
    """Test audio file processing task."""

    @pytest.mark.asyncio
    async def test_process_audio_file_success(self):
        """Test successful audio file processing."""
        from subtext.worker.tasks import process_audio_file
        import subtext.worker.tasks as tasks_module

        # Setup mocks
        mock_result = MagicMock()
        mock_result.duration_ms = 60000
        mock_result.speaker_count = 2
        mock_result.signal_count = 5
        mock_result.get = lambda k, d=None: getattr(mock_result, k, d)

        mock_pipeline = AsyncMock()
        mock_pipeline.process_file.return_value = mock_result

        with patch.object(tasks_module, "_update_session_status", new_callable=AsyncMock):
            with patch.object(tasks_module, "_save_processing_results", new_callable=AsyncMock):
                with patch.object(tasks_module, "_update_session_completed", new_callable=AsyncMock):
                    with patch("subtext.pipeline.orchestrator.create_pipeline", return_value=mock_pipeline):
                        ctx = {}
                        session_id = str(uuid4())

                        result = await process_audio_file(
                            ctx,
                            session_id=session_id,
                            file_path="/tmp/test.wav",
                        )

        assert result["success"] is True
        assert result["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_process_audio_file_failure(self):
        """Test audio file processing failure."""
        from subtext.worker.tasks import process_audio_file
        import subtext.worker.tasks as tasks_module

        # Setup mock to fail
        mock_pipeline = AsyncMock()
        mock_pipeline.process_file.side_effect = Exception("Processing failed")

        with patch.object(tasks_module, "_update_session_status", new_callable=AsyncMock):
            with patch("subtext.pipeline.orchestrator.create_pipeline", return_value=mock_pipeline):
                ctx = {}
                session_id = str(uuid4())

                result = await process_audio_file(
                    ctx,
                    session_id=session_id,
                    file_path="/tmp/test.wav",
                )

        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Processing failed"


# ══════════════════════════════════════════════════════════════
# Process Realtime Session Tests
# ══════════════════════════════════════════════════════════════


class TestProcessRealtimeSession:
    """Test realtime session processing task."""

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._get_session_data")
    @patch("subtext.worker.tasks._generate_insights")
    @patch("subtext.worker.tasks._save_insights")
    async def test_process_realtime_session_success(
        self, mock_save, mock_generate, mock_get_session
    ):
        """Test successful realtime session processing."""
        from subtext.worker.tasks import process_realtime_session

        session_id = str(uuid4())
        mock_get_session.return_value = {
            "id": session_id,
            "signals": [],
            "speakers": [],
        }
        mock_generate.return_value = [{"type": "summary", "content": "Test"}]

        ctx = {}
        result = await process_realtime_session(ctx, session_id=session_id)

        assert result["success"] is True
        assert result["insight_count"] == 1
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._get_session_data")
    async def test_process_realtime_session_not_found(self, mock_get_session):
        """Test realtime session processing when session not found."""
        from subtext.worker.tasks import process_realtime_session

        mock_get_session.return_value = None

        ctx = {}
        session_id = str(uuid4())

        result = await process_realtime_session(ctx, session_id=session_id)

        assert result["success"] is False
        assert "not found" in result["error"]


# ══════════════════════════════════════════════════════════════
# Generate Session Insights Tests
# ══════════════════════════════════════════════════════════════


class TestGenerateSessionInsights:
    """Test session insight generation task."""

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._get_session_data")
    @patch("subtext.worker.tasks._generate_summary")
    @patch("subtext.worker.tasks._find_key_moments")
    @patch("subtext.worker.tasks._identify_risk_flags")
    @patch("subtext.worker.tasks._generate_recommendations")
    @patch("subtext.worker.tasks._save_insights")
    async def test_generate_insights_success(
        self, mock_save, mock_recommendations, mock_risk,
        mock_moments, mock_summary, mock_get_session
    ):
        """Test successful insight generation."""
        from subtext.worker.tasks import generate_session_insights

        session_id = str(uuid4())
        mock_get_session.return_value = {
            "id": session_id,
            "duration_ms": 60000,
            "signals": [],
        }
        mock_summary.return_value = {"text": "Summary"}
        mock_moments.return_value = []
        mock_risk.return_value = []
        mock_recommendations.return_value = []

        ctx = {}
        result = await generate_session_insights(ctx, session_id=session_id)

        assert result["success"] is True
        assert "insights" in result
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._get_session_data")
    async def test_generate_insights_session_not_found(self, mock_get_session):
        """Test insight generation when session not found."""
        from subtext.worker.tasks import generate_session_insights

        mock_get_session.return_value = None

        ctx = {}
        session_id = str(uuid4())

        result = await generate_session_insights(ctx, session_id=session_id)

        assert result["success"] is False


# ══════════════════════════════════════════════════════════════
# Export Session Data Tests
# ══════════════════════════════════════════════════════════════


class TestExportSessionData:
    """Test session data export task."""

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._get_full_session_export")
    @patch("subtext.worker.tasks._export_json")
    @patch("subtext.worker.tasks._upload_to_storage")
    async def test_export_json_success(
        self, mock_upload, mock_export_json, mock_get_export
    ):
        """Test successful JSON export."""
        from subtext.worker.tasks import export_session_data

        session_id = str(uuid4())
        mock_get_export.return_value = {"session": {"id": session_id}}
        mock_export_json.return_value = "/tmp/export.json"
        mock_upload.return_value = f"/downloads/{session_id}.json"

        ctx = {}
        result = await export_session_data(
            ctx, session_id=session_id, format="json"
        )

        assert result["success"] is True
        assert result["format"] == "json"
        assert "download_url" in result
        mock_export_json.assert_called_once()

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._get_full_session_export")
    @patch("subtext.worker.tasks._export_csv")
    @patch("subtext.worker.tasks._upload_to_storage")
    async def test_export_csv_success(
        self, mock_upload, mock_export_csv, mock_get_export
    ):
        """Test successful CSV export."""
        from subtext.worker.tasks import export_session_data

        session_id = str(uuid4())
        mock_get_export.return_value = {
            "session": {"id": session_id},
            "transcript": [],
        }
        mock_export_csv.return_value = "/tmp/export.csv"
        mock_upload.return_value = f"/downloads/{session_id}.csv"

        ctx = {}
        result = await export_session_data(
            ctx, session_id=session_id, format="csv"
        )

        assert result["success"] is True
        assert result["format"] == "csv"
        mock_export_csv.assert_called_once()

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._get_full_session_export")
    async def test_export_unsupported_format(self, mock_get_export):
        """Test export with unsupported format."""
        from subtext.worker.tasks import export_session_data

        session_id = str(uuid4())
        mock_get_export.return_value = {"session": {"id": session_id}}

        ctx = {}
        result = await export_session_data(
            ctx, session_id=session_id, format="xml"
        )

        assert result["success"] is False
        assert "Unsupported" in result["error"]


# ══════════════════════════════════════════════════════════════
# Cleanup Expired Sessions Tests
# ══════════════════════════════════════════════════════════════


class TestCleanupExpiredSessions:
    """Test session cleanup task."""

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._find_expired_sessions")
    @patch("subtext.worker.tasks._delete_session_data")
    async def test_cleanup_success(self, mock_delete, mock_find):
        """Test successful session cleanup."""
        from subtext.worker.tasks import cleanup_expired_sessions

        session_id = str(uuid4())
        mock_find.return_value = [{"id": session_id, "name": "Old Session"}]
        mock_delete.return_value = None

        ctx = {}
        result = await cleanup_expired_sessions(ctx, retention_days=30)

        assert result["success"] is True
        assert result["deleted_count"] == 1
        assert result["total_expired"] == 1
        mock_delete.assert_called_once()

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._find_expired_sessions")
    async def test_cleanup_no_expired(self, mock_find):
        """Test cleanup when no expired sessions."""
        from subtext.worker.tasks import cleanup_expired_sessions

        mock_find.return_value = []

        ctx = {}
        result = await cleanup_expired_sessions(ctx, retention_days=30)

        assert result["success"] is True
        assert result["deleted_count"] == 0
        assert result["total_expired"] == 0

    @pytest.mark.asyncio
    @patch("subtext.worker.tasks._find_expired_sessions")
    @patch("subtext.worker.tasks._delete_session_data")
    async def test_cleanup_partial_failure(self, mock_delete, mock_find):
        """Test cleanup with partial delete failure."""
        from subtext.worker.tasks import cleanup_expired_sessions

        mock_find.return_value = [
            {"id": str(uuid4()), "name": "Session 1"},
            {"id": str(uuid4()), "name": "Session 2"},
        ]
        # First succeeds, second fails
        mock_delete.side_effect = [None, Exception("Delete failed")]

        ctx = {}
        result = await cleanup_expired_sessions(ctx, retention_days=30)

        assert result["success"] is True
        assert result["deleted_count"] == 1
        assert result["total_expired"] == 2


# ══════════════════════════════════════════════════════════════
# Helper Function Tests
# ══════════════════════════════════════════════════════════════


class TestHelperFunctions:
    """Test helper functions."""

    @pytest.mark.asyncio
    async def test_generate_insights_with_stress_signal(self):
        """Test insight generation detects stress signals."""
        from subtext.worker.tasks import _generate_insights

        session_data = {
            "signals": [
                {"type": SignalType.STRESS_SPIKE.value, "confidence": 0.8}
            ]
        }

        insights = await _generate_insights(session_data)

        assert len(insights) > 0
        assert any("stress" in i.get("content", "").lower() for i in insights)

    @pytest.mark.asyncio
    async def test_generate_insights_no_signals(self):
        """Test insight generation with no signals."""
        from subtext.worker.tasks import _generate_insights

        session_data = {"signals": []}

        insights = await _generate_insights(session_data)

        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_generate_summary(self):
        """Test summary generation."""
        from subtext.worker.tasks import _generate_summary

        session_data = {
            "duration_ms": 60000,
            "speaker_count": 2,
            "signals": [{"type": "test"}],
        }

        summary = await _generate_summary(session_data, use_llm=False)

        assert summary["duration_ms"] == 60000
        assert summary["speaker_count"] == 2
        assert summary["signal_count"] == 1

    @pytest.mark.asyncio
    async def test_find_key_moments_placeholder(self):
        """Test key moments finder placeholder."""
        from subtext.worker.tasks import _find_key_moments

        result = await _find_key_moments({})
        assert result == []

    @pytest.mark.asyncio
    async def test_identify_risk_flags_placeholder(self):
        """Test risk flags identifier placeholder."""
        from subtext.worker.tasks import _identify_risk_flags

        result = await _identify_risk_flags({})
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_recommendations_placeholder(self):
        """Test recommendations generator placeholder."""
        from subtext.worker.tasks import _generate_recommendations

        result = await _generate_recommendations({}, use_llm=False)
        assert result == []

    @pytest.mark.asyncio
    async def test_upload_to_storage_placeholder(self):
        """Test storage upload placeholder."""
        from subtext.worker.tasks import _upload_to_storage

        session_id = uuid4()
        result = await _upload_to_storage("/tmp/file.json", session_id, "json")

        assert f"/downloads/{session_id}.json" == result


# ══════════════════════════════════════════════════════════════
# Export Function Tests
# ══════════════════════════════════════════════════════════════


class TestExportFunctions:
    """Test export helper functions."""

    @pytest.mark.asyncio
    async def test_export_json_creates_file(self):
        """Test JSON export creates a file."""
        from subtext.worker.tasks import _export_json
        import os

        session_id = uuid4()
        data = {"test": "data", "number": 42}

        path = await _export_json(session_id, data)

        assert os.path.exists(path)
        with open(path) as f:
            content = json.load(f)
        assert content == data

        # Cleanup
        os.unlink(path)

    @pytest.mark.asyncio
    async def test_export_csv_creates_file(self):
        """Test CSV export creates a file."""
        from subtext.worker.tasks import _export_csv
        import os
        import csv

        session_id = uuid4()
        data = {
            "transcript": [
                {"start_ms": 0, "end_ms": 1000, "speaker_label": "A", "text": "Hello"},
                {"start_ms": 1000, "end_ms": 2000, "speaker_label": "B", "text": "Hi"},
            ]
        }

        path = await _export_csv(session_id, data)

        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["text"] == "Hello"

        # Cleanup
        os.unlink(path)

    @pytest.mark.asyncio
    async def test_export_pdf_creates_file(self):
        """Test PDF export creates a file (placeholder)."""
        from subtext.worker.tasks import _export_pdf
        import os

        session_id = uuid4()
        data = {"session": {"id": str(session_id)}}

        path = await _export_pdf(session_id, data)

        assert os.path.exists(path)

        # Cleanup
        os.unlink(path)


# ══════════════════════════════════════════════════════════════
# Integration-style Tests
# ══════════════════════════════════════════════════════════════


class TestTaskIntegration:
    """Integration-style tests for tasks."""

    @pytest.mark.asyncio
    async def test_full_processing_flow(self):
        """Test complete audio processing flow."""
        from subtext.worker.tasks import process_audio_file
        import subtext.worker.tasks as tasks_module

        # Setup comprehensive mock
        mock_result = MagicMock()
        mock_result.duration_ms = 120000
        mock_result.speaker_count = 3
        mock_result.signal_count = 10
        mock_result.get = lambda k, d=None: getattr(mock_result, k, d)

        mock_pipeline = AsyncMock()
        mock_pipeline.process_file.return_value = mock_result

        with patch.object(tasks_module, "_update_session_status", new_callable=AsyncMock):
            with patch.object(tasks_module, "_save_processing_results", new_callable=AsyncMock):
                with patch.object(tasks_module, "_update_session_completed", new_callable=AsyncMock):
                    with patch("subtext.pipeline.orchestrator.create_pipeline", return_value=mock_pipeline) as mock_create:
                        ctx = {}
                        session_id = str(uuid4())

                        result = await process_audio_file(
                            ctx,
                            session_id=session_id,
                            file_path="/tmp/audio.wav",
                            asr_backend="canary",
                            enable_diarization=True,
                            enable_emotion=True,
                            enable_signals=True,
                        )

        assert result["success"] is True
        assert result["session_id"] == session_id

        # Verify pipeline was created with correct options
        mock_create.assert_called_once_with(
            asr_backend="canary",
            enable_diarization=True,
            enable_emotion=True,
            enable_signals=True,
        )
