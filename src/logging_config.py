"""Structured JSON logging for the agent manager."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured log events."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "llm_provider": os.getenv("LLM_PROVIDER", "anthropic"),
        }

        # Add extra fields if present
        for key in ("event_type", "subtask_id", "tool_name", "decision", "status"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
            log_entry["traceback"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(log_entry)


class PostgresLogHandler(logging.Handler):
    """Logging handler that writes structured events to the agent_events table.

    Buffers records and flushes in batches to reduce database round-trips.
    """

    FLUSH_SIZE = 50
    FLUSH_INTERVAL = 5.0  # seconds

    def __init__(self, pool: Any, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self._pool = pool
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._timer: threading.Timer | None = None
        self._start_timer()

    def _start_timer(self) -> None:
        self._timer = threading.Timer(self.FLUSH_INTERVAL, self._timed_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timed_flush(self) -> None:
        self.flush()
        self._start_timer()

    def emit(self, record: logging.LogRecord) -> None:
        row: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "llm_provider": os.getenv("LLM_PROVIDER", "anthropic"),
            "event_type": getattr(record, "event_type", None),
            "subtask_id": getattr(record, "subtask_id", None),
            "tool_name": getattr(record, "tool_name", None),
            "decision": getattr(record, "decision", None),
            "status": getattr(record, "status", None),
            "exception": None,
            "traceback": None,
        }
        if record.exc_info and record.exc_info[1]:
            row["exception"] = str(record.exc_info[1])
            row["traceback"] = "".join(traceback.format_exception(*record.exc_info))

        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self.FLUSH_SIZE:
                self._flush_buffer()

    def flush(self) -> None:
        with self._lock:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        self._last_flush = time.monotonic()
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    for row in batch:
                        cur.execute(
                            """INSERT INTO agent_events
                               (level, logger, message, llm_provider,
                                event_type, subtask_id, tool_name, decision, status,
                                exception, traceback)
                               VALUES (%(level)s, %(logger)s, %(message)s, %(llm_provider)s,
                                       %(event_type)s, %(subtask_id)s, %(tool_name)s,
                                       %(decision)s, %(status)s,
                                       %(exception)s, %(traceback)s)""",
                            row,
                        )
                conn.commit()
        except Exception:
            # Avoid recursive logging — silently drop on DB failure
            pass

    def close(self) -> None:
        if self._timer:
            self._timer.cancel()
        self.flush()
        super().close()


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure structured logging for the application.

    Console output uses a clean human-readable format. JSON structured logs
    are written only to the LOG_FILE (JSONL format) and PostgreSQL handler.
    Noisy third-party loggers are silenced on the console.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler — always human-readable, never JSON
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    # Only show WARNING+ from the app on console (conversation is shown via Rich)
    console_handler.setLevel(logging.WARNING)

    root.handlers.clear()
    root.addHandler(console_handler)

    # Silence noisy third-party loggers
    for name in ("httpx", "urllib3", "langchain", "langchain_core",
                 "langchain_ollama", "langchain_anthropic", "langchain_openai",
                 "openai", "anthropic", "ollama"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Optional file handler for persistent structured JSON logs
    log_file = os.getenv("LOG_FILE")
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    # Optional PostgreSQL handler for queryable telemetry (dual-write with file)
    if os.getenv("DATABASE_URL"):
        try:
            from src.db import get_connection_pool

            pool = get_connection_pool()
            if pool is not None:
                pg_handler = PostgresLogHandler(pool)
                root.addHandler(pg_handler)
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to set up PostgreSQL log handler; continuing with file logging only",
                exc_info=True,
            )


def log_event(
    logger: logging.Logger,
    message: str,
    event_type: str | None = None,
    subtask_id: str | None = None,
    tool_name: str | None = None,
    decision: str | None = None,
    status: str | None = None,
) -> None:
    """Log a structured event with optional metadata fields."""
    extra: dict[str, Any] = {}
    if event_type:
        extra["event_type"] = event_type
    if subtask_id:
        extra["subtask_id"] = subtask_id
    if tool_name:
        extra["tool_name"] = tool_name
    if decision:
        extra["decision"] = decision
    if status:
        extra["status"] = status

    logger.info(message, extra=extra)
