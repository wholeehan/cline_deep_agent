"""Structured JSON logging for the agent manager."""

from __future__ import annotations

import json
import logging
import os
import sys
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

        return json.dumps(log_entry)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure structured logging for the application."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))

    root.handlers.clear()
    root.addHandler(handler)


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
