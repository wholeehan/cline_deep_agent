"""Human-in-the-loop flow helpers — interrupt detection, resume commands."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

logger = logging.getLogger(__name__)


@dataclass
class ActionRequest:
    """A pending decision from an interrupt payload."""

    tool_name: str
    action: dict[str, Any]
    description: str = ""


def extract_interrupts(result: dict[str, Any]) -> list[ActionRequest]:
    """Extract pending action requests from an agent invocation result.

    Looks for ``__interrupt__`` in the result and parses it into
    a list of ActionRequest objects.

    Returns an empty list if no interrupts are present.
    """
    interrupt_data = result.get("__interrupt__")
    if not interrupt_data:
        return []

    requests: list[ActionRequest] = []

    items = interrupt_data if isinstance(interrupt_data, list) else [interrupt_data]

    for item in items:
        if isinstance(item, dict):
            requests.append(ActionRequest(
                tool_name=item.get("tool_name", "unknown"),
                action=item.get("args", item.get("action", {})) or {},
                description=item.get("description", ""),
            ))
        else:
            # Handle interrupt objects with .value attribute (LangGraph Interrupt)
            val = getattr(item, "value", None)
            if isinstance(val, dict):
                requests.append(ActionRequest(
                    tool_name=val.get("tool_name", "unknown"),
                    action=val.get("args", val.get("action", {})) or {},
                    description=val.get("description", ""),
                ))
            else:
                requests.append(ActionRequest(
                    tool_name="unknown",
                    action={"raw": str(item)},
                    description=str(item),
                ))

    return requests


def build_approve_command(thread_id: str | None = None) -> Command:  # type: ignore[type-arg]
    """Build a Command to approve all pending actions."""
    return Command(resume={"decisions": [{"type": "approve"}]})


def build_reject_command(thread_id: str | None = None) -> Command:  # type: ignore[type-arg]
    """Build a Command to reject all pending actions."""
    return Command(resume={"decisions": [{"type": "reject"}]})


def build_edit_command(
    edited_action: dict[str, Any],
    thread_id: str | None = None,
) -> Command:  # type: ignore[type-arg]
    """Build a Command to approve with edited arguments."""
    return Command(resume={"decisions": [{"type": "edit", "edited_action": edited_action}]})


def build_batch_command(decisions: list[dict[str, Any]]) -> Command:  # type: ignore[type-arg]
    """Build a Command resolving multiple interrupts at once.

    Each decision should have a 'type' key ('approve', 'reject', or 'edit')
    and optionally 'edited_action' for edit decisions.
    """
    return Command(resume={"decisions": decisions})
