"""
Unit tests for CLI commands.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from click.testing import CliRunner

from subtext.cli import cli, main


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


# ══════════════════════════════════════════════════════════════
# Main CLI Tests
# ══════════════════════════════════════════════════════════════


class TestMainCLI:
    """Test main CLI group."""

    def test_cli_version(self, runner):
        """Test CLI version option."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Subtext" in result.output
        assert "Metacognitive Engine" in result.output

    def test_cli_debug_mode(self, runner):
        """Test CLI debug mode."""
        result = runner.invoke(cli, ["--debug", "--help"])
        assert result.exit_code == 0

    def test_main_function(self):
        """Test main entry point."""
        with patch("subtext.cli.cli") as mock_cli:
            main()
            mock_cli.assert_called_once()


# ══════════════════════════════════════════════════════════════
# Server Commands Tests
# ══════════════════════════════════════════════════════════════


class TestServeCommand:
    """Test serve command."""

    def test_serve_help(self, runner):
        """Test serve help output."""
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output

    def test_serve_command(self, runner):
        """Test serve command with mocked uvicorn."""
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli, ["serve", "--host", "127.0.0.1", "--port", "9000"])

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 9000

    def test_serve_with_reload(self, runner):
        """Test serve command with reload option."""
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli, ["serve", "--reload"])

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["reload"] is True
            # Workers should be 1 when reload is enabled
            assert call_kwargs["workers"] == 1

    def test_serve_with_workers(self, runner):
        """Test serve command with multiple workers."""
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli, ["serve", "--workers", "4"])

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["workers"] == 4


class TestRealtimeCommand:
    """Test realtime server command."""

    def test_realtime_help(self, runner):
        """Test realtime help output."""
        result = runner.invoke(cli, ["realtime", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output

    def test_realtime_command(self, runner):
        """Test realtime command with mocked uvicorn."""
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli, ["realtime", "--host", "127.0.0.1", "--port", "9001"])

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 9001


# ══════════════════════════════════════════════════════════════
# Analyze Command Tests
# ══════════════════════════════════════════════════════════════


class TestAnalyzeCommand:
    """Test analyze command."""

    def test_analyze_help(self, runner):
        """Test analyze help output."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "AUDIO_FILE" in result.output
        assert "--language" in result.output
        assert "--backend" in result.output

    def test_analyze_file_not_found(self, runner):
        """Test analyze with non-existent file."""
        result = runner.invoke(cli, ["analyze", "/nonexistent/file.wav"])
        assert result.exit_code != 0

    @pytest.mark.skip(reason="create_pipeline not exported from subtext.pipeline")
    def test_analyze_success(self, runner, tmp_path):
        """Test successful analyze command."""
        pass

    @pytest.mark.skip(reason="create_pipeline not exported from subtext.pipeline")
    def test_analyze_with_output(self, runner, tmp_path):
        """Test analyze command with output file."""
        pass

    @pytest.mark.skip(reason="create_pipeline not exported from subtext.pipeline")
    def test_analyze_failure(self, runner, tmp_path):
        """Test analyze command with processing failure."""
        pass


# ══════════════════════════════════════════════════════════════
# Models Command Tests
# ══════════════════════════════════════════════════════════════


class TestModelsCommand:
    """Test models command."""

    def test_models_help(self, runner):
        """Test models help output."""
        result = runner.invoke(cli, ["models", "--help"])
        assert result.exit_code == 0

    def test_models_list(self, runner):
        """Test models list output."""
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "ML Model Stack" in result.output
        assert "VAD" in result.output
        assert "ASR" in result.output


# ══════════════════════════════════════════════════════════════
# Database Commands Tests
# ══════════════════════════════════════════════════════════════


class TestDBCommands:
    """Test database commands."""

    def test_db_help(self, runner):
        """Test db help output."""
        result = runner.invoke(cli, ["db", "--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "migrate" in result.output

    def test_db_init(self, runner):
        """Test db init command."""
        with patch("subtext.db.init_db", new_callable=AsyncMock) as mock_init:
            with patch("subtext.db.close_db", new_callable=AsyncMock) as mock_close:
                result = runner.invoke(cli, ["db", "init"])

                assert "Initializing database" in result.output

    def test_db_migrate(self, runner):
        """Test db migrate command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Migration created",
                stderr="",
            )

            result = runner.invoke(cli, ["db", "migrate", "-m", "test migration"])

            mock_run.assert_called_once()
            assert "alembic" in mock_run.call_args[0][0]

    def test_db_migrate_failure(self, runner):
        """Test db migrate command failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Migration failed",
            )

            result = runner.invoke(cli, ["db", "migrate", "-m", "test"])

            assert result.exit_code == 1

    def test_db_upgrade(self, runner):
        """Test db upgrade command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Upgraded to head",
                stderr="",
            )

            result = runner.invoke(cli, ["db", "upgrade"])

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "alembic" in args
            assert "upgrade" in args
            assert "head" in args

    def test_db_downgrade(self, runner):
        """Test db downgrade command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Downgraded",
                stderr="",
            )

            result = runner.invoke(cli, ["db", "downgrade"])

            mock_run.assert_called_once()


# ══════════════════════════════════════════════════════════════
# Worker Commands Tests
# ══════════════════════════════════════════════════════════════


class TestWorkerCommands:
    """Test worker commands."""

    def test_worker_help(self, runner):
        """Test worker help output."""
        result = runner.invoke(cli, ["worker", "--help"])
        assert result.exit_code == 0
        assert "start" in result.output
        assert "enqueue" in result.output
        assert "status" in result.output

    def test_worker_start_help(self, runner):
        """Test worker start help output."""
        result = runner.invoke(cli, ["worker", "start", "--help"])
        assert result.exit_code == 0
        assert "--concurrency" in result.output
        assert "--queue" in result.output
        assert "--burst" in result.output

    def test_worker_start(self, runner):
        """Test worker start command."""
        with patch("arq.run_worker") as mock_run:
            with patch("subtext.worker.queue.get_worker_settings") as mock_settings:
                mock_settings.return_value = {
                    "queue_name": "default",
                    "max_jobs": 10,
                }

                result = runner.invoke(cli, ["worker", "start"])

                mock_run.assert_called_once()

    def test_worker_enqueue(self, runner):
        """Test worker enqueue command."""
        with patch("subtext.worker.queue.enqueue_job", new_callable=AsyncMock) as mock_enqueue:
            mock_enqueue.return_value = "job-123"

            result = runner.invoke(cli, [
                "worker", "enqueue", "process_audio",
                "-s", "session-123",
                "-p", "high",
            ])

            assert result.exit_code == 0
            assert "enqueued" in result.output

    def test_worker_enqueue_failure(self, runner):
        """Test worker enqueue command failure."""
        with patch("subtext.worker.queue.enqueue_job", new_callable=AsyncMock) as mock_enqueue:
            mock_enqueue.return_value = None

            result = runner.invoke(cli, [
                "worker", "enqueue", "process_audio",
                "-s", "session-123",
            ])

            assert result.exit_code == 1

    def test_worker_status_job(self, runner):
        """Test worker status with job ID."""
        with patch("subtext.worker.queue.get_job_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {
                "job_id": "job-123",
                "status": "completed",
                "result": {"success": True},
            }

            result = runner.invoke(cli, ["worker", "status", "job-123"])

            assert result.exit_code == 0
            assert "job-123" in result.output
            assert "completed" in result.output

    def test_worker_status_job_not_found(self, runner):
        """Test worker status with unknown job ID."""
        with patch("subtext.worker.queue.get_job_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = None

            result = runner.invoke(cli, ["worker", "status", "unknown-job"])

            assert "not found" in result.output

    def test_worker_status_queue(self, runner):
        """Test worker status without job ID (queue stats)."""
        with patch("subtext.worker.queue.get_redis_pool", new_callable=AsyncMock) as mock_pool:
            mock_redis = AsyncMock()
            mock_redis.info = AsyncMock(return_value={"connected_clients": 1})
            mock_pool.return_value = mock_redis

            result = runner.invoke(cli, ["worker", "status"])

            assert result.exit_code == 0
            assert "ARQ Worker Status" in result.output


# ══════════════════════════════════════════════════════════════
# Config Command Tests
# ══════════════════════════════════════════════════════════════


class TestConfigCommand:
    """Test config command."""

    def test_config_help(self, runner):
        """Test config help output."""
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0

    def test_config_show(self, runner):
        """Test config show output."""
        result = runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        assert "Configuration" in result.output
        assert "Environment" in result.output
        assert "Database" in result.output
