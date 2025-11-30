"""
Unit Tests for Worker Module

Tests the ARQ queue configuration and background task definitions.
"""

import pytest
from datetime import timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from subtext.worker.queue import (
    JobPriority,
    QUEUE_NAMES,
    get_worker_settings,
)


# ══════════════════════════════════════════════════════════════
# Job Priority Tests
# ══════════════════════════════════════════════════════════════


class TestJobPriority:
    """Test job priority enums and queue names."""

    def test_job_priority_values(self):
        """Test job priority enum values."""
        assert JobPriority.HIGH.value == "high"
        assert JobPriority.NORMAL.value == "normal"
        assert JobPriority.LOW.value == "low"

    def test_queue_names_mapping(self):
        """Test queue names for each priority."""
        assert QUEUE_NAMES[JobPriority.HIGH] == "subtext:high"
        assert QUEUE_NAMES[JobPriority.NORMAL] == "subtext:default"
        assert QUEUE_NAMES[JobPriority.LOW] == "subtext:low"

    def test_all_priorities_have_queues(self):
        """Test all priority levels have queue names."""
        for priority in JobPriority:
            assert priority in QUEUE_NAMES
            assert QUEUE_NAMES[priority].startswith("subtext:")


# ══════════════════════════════════════════════════════════════
# Worker Settings Tests
# ══════════════════════════════════════════════════════════════


class TestWorkerSettings:
    """Test ARQ worker settings configuration."""

    def test_worker_settings_structure(self):
        """Test worker settings contain required keys."""
        settings = get_worker_settings()

        assert "functions" in settings
        assert "on_startup" in settings
        assert "on_shutdown" in settings
        assert "redis_settings" in settings
        assert "queue_name" in settings
        assert "max_jobs" in settings
        assert "job_timeout" in settings

    def test_worker_settings_functions(self):
        """Test worker has correct task functions registered."""
        settings = get_worker_settings()

        function_names = [f.__name__ for f in settings["functions"]]

        assert "process_audio_file" in function_names
        assert "process_realtime_session" in function_names
        assert "generate_session_insights" in function_names
        assert "export_session_data" in function_names
        assert "cleanup_expired_sessions" in function_names

    def test_worker_settings_lifecycle_callbacks(self):
        """Test worker has startup/shutdown callbacks."""
        settings = get_worker_settings()

        assert callable(settings["on_startup"])
        assert callable(settings["on_shutdown"])
        assert settings["on_startup"].__name__ == "startup"
        assert settings["on_shutdown"].__name__ == "shutdown"

    def test_worker_settings_defaults(self):
        """Test worker settings have reasonable defaults."""
        settings = get_worker_settings()

        assert settings["queue_name"] == QUEUE_NAMES[JobPriority.NORMAL]
        assert settings["keep_result"] == 3600  # 1 hour
        assert settings["health_check_interval"] == 30
        assert settings["retry_jobs"] is True
        assert settings["max_tries"] == 3


# ══════════════════════════════════════════════════════════════
# Queue Functions Tests
# ══════════════════════════════════════════════════════════════


class TestQueueFunctions:
    """Test queue management functions."""

    @pytest.mark.asyncio
    async def test_get_redis_pool(self):
        """Test Redis pool creation."""
        with patch("subtext.worker.queue.create_pool", new_callable=AsyncMock) as mock_pool:
            from subtext.worker.queue import get_redis_pool, _pool
            import subtext.worker.queue as queue_module

            # Reset global pool
            queue_module._pool = None

            mock_pool.return_value = AsyncMock()
            pool = await get_redis_pool()

            mock_pool.assert_called_once()
            assert pool is not None

            # Reset after test
            queue_module._pool = None

    @pytest.mark.asyncio
    async def test_enqueue_job(self):
        """Test job enqueueing."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import enqueue_job, JobPriority

            mock_pool = AsyncMock()
            mock_job = AsyncMock()
            mock_job.job_id = "test-job-123"
            mock_pool.enqueue_job = AsyncMock(return_value=mock_job)
            mock_get_pool.return_value = mock_pool

            job_id = await enqueue_job(
                "process_audio_file",
                session_id="sess-123",
                priority=JobPriority.HIGH,
            )

            assert job_id == "test-job-123"
            mock_pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_enqueue_job_with_defer(self):
        """Test job enqueueing with defer_by."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import enqueue_job, JobPriority

            mock_pool = AsyncMock()
            mock_job = AsyncMock()
            mock_job.job_id = "test-job-456"
            mock_pool.enqueue_job = AsyncMock(return_value=mock_job)
            mock_get_pool.return_value = mock_pool

            job_id = await enqueue_job(
                "cleanup_expired_sessions",
                defer_by=timedelta(hours=1),
                priority=JobPriority.LOW,
            )

            assert job_id == "test-job-456"
            # Verify defer_by was passed
            call_kwargs = mock_pool.enqueue_job.call_args
            assert call_kwargs.kwargs.get("_defer_by") == timedelta(hours=1)

    @pytest.mark.asyncio
    async def test_enqueue_job_failure(self):
        """Test job enqueueing failure handling."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import enqueue_job

            mock_pool = AsyncMock()
            mock_pool.enqueue_job = AsyncMock(side_effect=Exception("Redis error"))
            mock_get_pool.return_value = mock_pool

            job_id = await enqueue_job("process_audio_file", session_id="test")

            assert job_id is None

    @pytest.mark.asyncio
    async def test_get_job_status(self):
        """Test getting job status."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import get_job_status

            mock_pool = AsyncMock()
            mock_job = AsyncMock()
            mock_info = MagicMock()
            mock_info.status = "complete"
            mock_info.result = {"success": True}
            mock_info.start_time = None
            mock_info.finish_time = None
            mock_job.info = AsyncMock(return_value=mock_info)
            mock_pool.job = AsyncMock(return_value=mock_job)
            mock_get_pool.return_value = mock_pool

            status = await get_job_status("job-123")

            assert status is not None
            assert status["job_id"] == "job-123"
            assert status["status"] == "complete"
            assert status["result"] == {"success": True}

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self):
        """Test getting status of non-existent job."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import get_job_status

            mock_pool = AsyncMock()
            mock_pool.job = AsyncMock(return_value=None)
            mock_get_pool.return_value = mock_pool

            status = await get_job_status("nonexistent-job")

            assert status is None

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        """Test cancelling a job."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import cancel_job

            mock_pool = AsyncMock()
            mock_job = AsyncMock()
            mock_job.abort = AsyncMock()
            mock_pool.job = AsyncMock(return_value=mock_job)
            mock_get_pool.return_value = mock_pool

            result = await cancel_job("job-123")

            assert result is True
            mock_job.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self):
        """Test cancelling non-existent job."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import cancel_job

            mock_pool = AsyncMock()
            mock_pool.job = AsyncMock(return_value=None)
            mock_get_pool.return_value = mock_pool

            result = await cancel_job("nonexistent-job")

            assert result is False


# ══════════════════════════════════════════════════════════════
# Worker Settings Class Tests
# ══════════════════════════════════════════════════════════════


class TestCloseRedisPool:
    """Test Redis pool closing."""

    @pytest.mark.asyncio
    async def test_close_redis_pool_with_pool(self):
        """Test closing Redis pool when pool exists."""
        import subtext.worker.queue as queue_module
        from subtext.worker.queue import close_redis_pool

        # Set up mock pool
        mock_pool = AsyncMock()
        mock_pool.close = AsyncMock()
        original_pool = queue_module._pool
        queue_module._pool = mock_pool

        try:
            await close_redis_pool()

            mock_pool.close.assert_called_once()
            assert queue_module._pool is None
        finally:
            queue_module._pool = original_pool

    @pytest.mark.asyncio
    async def test_close_redis_pool_without_pool(self):
        """Test closing Redis pool when no pool exists."""
        import subtext.worker.queue as queue_module
        from subtext.worker.queue import close_redis_pool

        original_pool = queue_module._pool
        queue_module._pool = None

        try:
            # Should not raise
            await close_redis_pool()
            assert queue_module._pool is None
        finally:
            queue_module._pool = original_pool


class TestEnqueueJobEdgeCases:
    """Test edge cases for job enqueueing."""

    @pytest.mark.asyncio
    async def test_enqueue_job_returns_none_when_no_job(self):
        """Test enqueue returns None when pool returns None."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import enqueue_job

            mock_pool = AsyncMock()
            mock_pool.enqueue_job = AsyncMock(return_value=None)
            mock_get_pool.return_value = mock_pool

            job_id = await enqueue_job("process_audio_file", session_id="test")

            assert job_id is None


class TestGetJobStatusEdgeCases:
    """Test edge cases for job status retrieval."""

    @pytest.mark.asyncio
    async def test_get_job_status_exception(self):
        """Test get_job_status handles exceptions."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import get_job_status

            mock_pool = AsyncMock()
            mock_pool.job = AsyncMock(side_effect=Exception("Redis error"))
            mock_get_pool.return_value = mock_pool

            status = await get_job_status("job-123")

            assert status is None


class TestCancelJobEdgeCases:
    """Test edge cases for job cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_job_exception(self):
        """Test cancel_job handles exceptions."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_get_pool:
            from subtext.worker.queue import cancel_job

            mock_pool = AsyncMock()
            mock_pool.job = AsyncMock(side_effect=Exception("Redis error"))
            mock_get_pool.return_value = mock_pool

            result = await cancel_job("job-123")

            assert result is False


class TestWorkerSettingsClass:
    """Test the WorkerSettings class for ARQ CLI."""

    def test_worker_settings_class_exists(self):
        """Test WorkerSettings class is properly defined."""
        from subtext.worker.queue import WorkerSettings

        assert hasattr(WorkerSettings, "get_settings")

    def test_worker_settings_class_method(self):
        """Test WorkerSettings.get_settings returns valid settings."""
        from subtext.worker.queue import WorkerSettings

        settings = WorkerSettings.get_settings()

        assert isinstance(settings, dict)
        assert "functions" in settings
        assert "redis_settings" in settings


# ══════════════════════════════════════════════════════════════
# Task Function Signatures Tests
# ══════════════════════════════════════════════════════════════


class TestTaskSignatures:
    """Test task function signatures and metadata."""

    def test_process_audio_file_signature(self):
        """Test process_audio_file has correct signature."""
        from subtext.worker.tasks import process_audio_file
        import inspect

        sig = inspect.signature(process_audio_file)
        params = list(sig.parameters.keys())

        # Should have ctx and session_id at minimum
        assert "ctx" in params
        assert "session_id" in params

    def test_process_realtime_session_signature(self):
        """Test process_realtime_session has correct signature."""
        from subtext.worker.tasks import process_realtime_session
        import inspect

        sig = inspect.signature(process_realtime_session)
        params = list(sig.parameters.keys())

        assert "ctx" in params
        assert "session_id" in params

    def test_generate_session_insights_signature(self):
        """Test generate_session_insights has correct signature."""
        from subtext.worker.tasks import generate_session_insights
        import inspect

        sig = inspect.signature(generate_session_insights)
        params = list(sig.parameters.keys())

        assert "ctx" in params
        assert "session_id" in params

    def test_export_session_data_signature(self):
        """Test export_session_data has correct signature."""
        from subtext.worker.tasks import export_session_data
        import inspect

        sig = inspect.signature(export_session_data)
        params = list(sig.parameters.keys())

        assert "ctx" in params
        assert "session_id" in params

    def test_cleanup_expired_sessions_signature(self):
        """Test cleanup_expired_sessions has correct signature."""
        from subtext.worker.tasks import cleanup_expired_sessions
        import inspect

        sig = inspect.signature(cleanup_expired_sessions)
        params = list(sig.parameters.keys())

        assert "ctx" in params

    def test_startup_signature(self):
        """Test startup function has correct signature."""
        from subtext.worker.tasks import startup
        import inspect

        sig = inspect.signature(startup)
        params = list(sig.parameters.keys())

        assert "ctx" in params

    def test_shutdown_signature(self):
        """Test shutdown function has correct signature."""
        from subtext.worker.tasks import shutdown
        import inspect

        sig = inspect.signature(shutdown)
        params = list(sig.parameters.keys())

        assert "ctx" in params
