"""Cline CLI PTY bridge — spawn, stream-parse, and inject responses."""

from __future__ import annotations

import enum
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pexpect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(enum.Enum):
    """Classification of Cline stdout chunks."""

    PROGRESS = "progress"
    QUESTION = "question"
    APPROVAL = "approval"
    COMMAND = "command"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class StreamEvent:
    """A parsed event from Cline stdout."""

    event_type: EventType
    content: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Stream classifier
# ---------------------------------------------------------------------------

# Patterns for approval prompts
_APPROVAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"Do you want to (?:proceed|continue|create|delete|overwrite"
        r"|save|apply|run|install|execute)\b",
        re.IGNORECASE,
    ),
    re.compile(r"Should I proceed\b", re.IGNORECASE),
    re.compile(r"Would you like me to\b", re.IGNORECASE),
    re.compile(r"Shall I\b", re.IGNORECASE),
    re.compile(r"Do you approve\b", re.IGNORECASE),
    re.compile(r"Allow .+ to\b", re.IGNORECASE),
    re.compile(r"Proceed\?", re.IGNORECASE),
    re.compile(r"\(y/n\)", re.IGNORECASE),
    re.compile(r"\[Y/n\]", re.IGNORECASE),
]

# Patterns for questions (lines ending with ?)
_QUESTION_PATTERN = re.compile(r"\?\s*$")

# Patterns for error output
_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:error|Error|ERROR)[:\s]"),
    re.compile(r"(?:fatal|Fatal|FATAL)[:\s]"),
    re.compile(r"Traceback \(most recent call last\)"),
]

# Patterns for command output
_COMMAND_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\$ "),
    re.compile(r"^> "),
    re.compile(r"Running:"),
    re.compile(r"Executing:"),
]


def classify_output(text: str) -> EventType:
    """Classify a chunk of Cline stdout into an event type."""
    stripped = text.strip()
    if not stripped:
        return EventType.UNKNOWN

    # Check approval first (more specific than question)
    for pat in _APPROVAL_PATTERNS:
        if pat.search(stripped):
            return EventType.APPROVAL

    # Check for questions
    if _QUESTION_PATTERN.search(stripped):
        return EventType.QUESTION

    # Check for errors
    for pat in _ERROR_PATTERNS:
        if pat.search(stripped):
            return EventType.ERROR

    # Check for commands
    for pat in _COMMAND_PATTERNS:
        if pat.search(stripped):
            return EventType.COMMAND

    return EventType.PROGRESS


# ---------------------------------------------------------------------------
# PTY session replay log
# ---------------------------------------------------------------------------


class SessionLog:
    """Append-only timestamped log of all PTY stdin/stdout chunks."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] | None = None

    def open(self) -> None:
        self._file = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def record(self, direction: str, data: str) -> None:
        """Record a chunk. direction is 'stdin' or 'stdout'."""
        if self._file is None:
            self.open()
        ts = time.time()
        assert self._file is not None
        self._file.write(f"[{ts:.6f}] [{direction}] {data}\n")
        self._file.flush()

    def replay(self) -> list[tuple[float, str, str]]:
        """Read back all recorded entries."""
        entries: list[tuple[float, str, str]] = []
        if not self.log_path.exists():
            return entries
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                match = re.match(r"\[(\d+\.\d+)\] \[(stdin|stdout)\] (.*)", line)
                if match:
                    entries.append((float(match.group(1)), match.group(2), match.group(3)))
        return entries


# ---------------------------------------------------------------------------
# Cline PTY Bridge
# ---------------------------------------------------------------------------


class ClineBridge:
    """Manages a Cline CLI subprocess via PTY."""

    def __init__(
        self,
        cline_command: str = "cline",
        log_dir: str | Path = "logs",
        timeout: int = 300,
    ) -> None:
        self.cline_command = cline_command
        self.timeout = timeout
        self._process: pexpect.spawn[str] | None = None
        self._session_log = SessionLog(Path(log_dir) / f"session_{int(time.time())}.log")

    @property
    def pid(self) -> int | None:
        """Return the PID of the spawned process, or None."""
        if self._process and self._process.isalive() and self._process.pid is not None:
            return self._process.pid
        return None

    def spawn(self, task: str | None = None) -> int:
        """Start a new Cline PTY session.

        Returns the PID of the spawned process.
        """
        cmd = self.cline_command
        if task:
            cmd = f'{self.cline_command} "{task}"'

        self._process = pexpect.spawn(
            "/bin/bash",
            ["-c", cmd],
            timeout=self.timeout,
            encoding="utf-8",
            env={**os.environ},
        )
        self._session_log.open()
        pid = self._process.pid or 0
        logger.info("Spawned Cline process PID=%d", pid)
        return pid

    def send_task(self, task: str, read_timeout: float = 10.0) -> str:
        """Send a task string to a running Cline session and return output summary.

        Parameters
        ----------
        read_timeout:
            Maximum seconds to wait for each line of output.  When a line
            takes longer than this the read loop ends and collected output
            is returned.  This prevents hanging when Cline is waiting for
            input or producing output slowly.
        """
        if self._process is None or not self._process.isalive():
            self.spawn()

        assert self._process is not None
        self.inject(task)

        # Use a short per-line timeout so we don't hang forever.
        # Save and restore the original timeout.
        original_timeout = self._process.timeout
        self._process.timeout = int(read_timeout)

        output_chunks: list[str] = []
        try:
            while True:
                line = self._process.readline()
                if not line:
                    break
                self._session_log.record("stdout", line.strip())
                output_chunks.append(line.strip())
        except pexpect.TIMEOUT:
            logger.debug(
                "send_task: readline timed out after %ds (%d lines collected)",
                read_timeout, len(output_chunks),
            )
        except pexpect.EOF:
            logger.debug("send_task: process EOF (%d lines collected)", len(output_chunks))
        finally:
            self._process.timeout = original_timeout

        return "\n".join(output_chunks)

    def inject(self, text: str) -> None:
        """Write a string to the PTY stdin."""
        if self._process is None or not self._process.isalive():
            raise RuntimeError("No active Cline process to inject into")

        self._session_log.record("stdin", text.strip())
        self._process.sendline(text)

    def read_output(self, timeout: float = 5.0) -> list[StreamEvent]:
        """Read and classify available output from the PTY."""
        if self._process is None:
            return []

        events: list[StreamEvent] = []
        self._process.timeout = int(timeout)

        try:
            while True:
                line = self._process.readline()
                if not line:
                    break
                content = line.strip()
                if content:
                    self._session_log.record("stdout", content)
                    event_type = classify_output(content)
                    events.append(StreamEvent(event_type=event_type, content=content))
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass

        return events

    def is_alive(self) -> bool:
        """Check if the Cline process is still running."""
        return self._process is not None and self._process.isalive()

    def exit_code(self) -> int | None:
        """Return exit code if process has terminated, else None."""
        if self._process is None:
            return None
        if self._process.isalive():
            return None
        status = self._process.exitstatus
        return int(status) if status is not None else None

    def close(self) -> None:
        """Terminate the Cline process and close the session log."""
        if self._process and self._process.isalive():
            self._process.close(force=True)
        self._session_log.close()
