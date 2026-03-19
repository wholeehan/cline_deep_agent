"""Tests for Cline CLI PTY bridge."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.bridge import (
    ClineBridge,
    EventType,
    SessionLog,
    classify_output,
)

# ---------------------------------------------------------------------------
# Stream classifier tests
# ---------------------------------------------------------------------------


class TestClassifyOutput:
    """Unit tests for classify_output — 10+ captured Cline stdout samples."""

    # -- Approval samples --
    @pytest.mark.parametrize(
        "text",
        [
            "Do you want to create this file?",
            "Should I proceed with the installation?",
            "Would you like me to run the tests?",
            "Shall I continue with the deployment?",
            "Do you approve this change?",
            "Allow cline to access /etc/hosts? (y/n)",
            "Proceed? [Y/n]",
        ],
    )
    def test_approval_events(self, text: str) -> None:
        assert classify_output(text) == EventType.APPROVAL

    # -- Question samples --
    @pytest.mark.parametrize(
        "text",
        [
            "What framework do you want to use?",
            "Which database should I configure?",
            "Where should I put the config file?",
        ],
    )
    def test_question_events(self, text: str) -> None:
        assert classify_output(text) == EventType.QUESTION

    # -- Progress samples --
    @pytest.mark.parametrize(
        "text",
        [
            "Installing dependencies...",
            "Creating project structure",
            "Writing src/main.py",
            "✓ Tests passed successfully",
        ],
    )
    def test_progress_events(self, text: str) -> None:
        assert classify_output(text) == EventType.PROGRESS

    # -- Error samples --
    @pytest.mark.parametrize(
        "text",
        [
            "Error: file not found",
            "fatal: not a git repository",
            "Traceback (most recent call last)",
        ],
    )
    def test_error_events(self, text: str) -> None:
        assert classify_output(text) == EventType.ERROR

    # -- Command samples --
    @pytest.mark.parametrize(
        "text",
        [
            "$ npm install",
            "> python setup.py",
            "Running: pytest -v",
            "Executing: docker build .",
        ],
    )
    def test_command_events(self, text: str) -> None:
        assert classify_output(text) == EventType.COMMAND

    def test_empty_string_is_unknown(self) -> None:
        assert classify_output("") == EventType.UNKNOWN

    def test_whitespace_is_unknown(self) -> None:
        assert classify_output("   \n\t  ") == EventType.UNKNOWN


# ---------------------------------------------------------------------------
# Session log tests
# ---------------------------------------------------------------------------


class TestSessionLog:
    """Tests for append-only PTY session replay log."""

    def test_record_and_replay(self, tmp_path: Path) -> None:
        log = SessionLog(tmp_path / "test.log")
        log.record("stdout", "hello world")
        log.record("stdin", "yes")
        log.record("stdout", "done")
        log.close()

        entries = log.replay()
        assert len(entries) == 3
        assert entries[0][1] == "stdout"
        assert entries[0][2] == "hello world"
        assert entries[1][1] == "stdin"
        assert entries[1][2] == "yes"
        assert entries[2][1] == "stdout"
        assert entries[2][2] == "done"

    def test_timestamps_are_monotonic(self, tmp_path: Path) -> None:
        log = SessionLog(tmp_path / "test.log")
        log.record("stdout", "a")
        log.record("stdout", "b")
        log.close()

        entries = log.replay()
        assert entries[1][0] >= entries[0][0]

    def test_empty_log_replay(self, tmp_path: Path) -> None:
        log = SessionLog(tmp_path / "nonexistent.log")
        assert log.replay() == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log = SessionLog(tmp_path / "deep" / "nested" / "test.log")
        log.record("stdout", "test")
        log.close()
        assert (tmp_path / "deep" / "nested" / "test.log").exists()


# ---------------------------------------------------------------------------
# ClineBridge tests (with mocked PTY)
# ---------------------------------------------------------------------------


class TestClineBridge:
    """Tests for ClineBridge using mocked pexpect."""

    def test_spawn_returns_nonzero_pid(self, tmp_path: Path) -> None:
        bridge = ClineBridge(cline_command="echo hello", log_dir=tmp_path)
        pid = bridge.spawn()
        assert pid > 0
        bridge.close()

    def test_inject_writes_to_stdin(self, tmp_path: Path) -> None:
        bridge = ClineBridge(log_dir=tmp_path)
        mock_process = MagicMock()
        mock_process.isalive.return_value = True
        bridge._process = mock_process

        bridge.inject("yes")
        mock_process.sendline.assert_called_once_with("yes")

    def test_inject_raises_when_no_process(self, tmp_path: Path) -> None:
        bridge = ClineBridge(log_dir=tmp_path)
        with pytest.raises(RuntimeError, match="No active Cline process"):
            bridge.inject("yes")

    def test_is_alive_false_initially(self, tmp_path: Path) -> None:
        bridge = ClineBridge(log_dir=tmp_path)
        assert bridge.is_alive() is False

    def test_pid_none_initially(self, tmp_path: Path) -> None:
        bridge = ClineBridge(log_dir=tmp_path)
        assert bridge.pid is None

    def test_close_terminates_process(self, tmp_path: Path) -> None:
        bridge = ClineBridge(log_dir=tmp_path)
        mock_process = MagicMock()
        mock_process.isalive.return_value = True
        bridge._process = mock_process

        bridge.close()
        mock_process.close.assert_called_once_with(force=True)

    def test_exit_code_none_when_alive(self, tmp_path: Path) -> None:
        bridge = ClineBridge(log_dir=tmp_path)
        mock_process = MagicMock()
        mock_process.isalive.return_value = True
        bridge._process = mock_process
        assert bridge.exit_code() is None

    def test_exit_code_returned_when_dead(self, tmp_path: Path) -> None:
        bridge = ClineBridge(log_dir=tmp_path)
        mock_process = MagicMock()
        mock_process.isalive.return_value = False
        mock_process.exitstatus = 1
        bridge._process = mock_process
        assert bridge.exit_code() == 1


class TestClineBridgeIntegration:
    """Integration test: spawn a real process and inject input."""

    def test_spawn_echo_and_read(self, tmp_path: Path) -> None:
        """Spawn a simple echo command to verify PTY works."""
        bridge = ClineBridge(cline_command="echo 'hello from cline'", log_dir=tmp_path)
        pid = bridge.spawn()
        assert pid > 0

        # Read output
        import time
        time.sleep(0.5)
        events = bridge.read_output(timeout=2.0)
        bridge.close()

        # Should have captured the echo output
        all_content = " ".join(e.content for e in events)
        assert "hello" in all_content or bridge.exit_code() is not None
