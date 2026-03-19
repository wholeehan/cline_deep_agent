"""Cline executor subagent — tools and definition."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from src.bridge import ClineBridge
from src.llm import get_llm

logger = logging.getLogger(__name__)

# Module-level bridge instance (lazily initialized)
_bridge: ClineBridge | None = None


def _get_bridge() -> ClineBridge:
    global _bridge
    if _bridge is None:
        _bridge = ClineBridge()
    return _bridge


def set_bridge(bridge: ClineBridge) -> None:
    """Override the bridge instance (useful for testing)."""
    global _bridge
    _bridge = bridge


# ---------------------------------------------------------------------------
# Subagent tools
# ---------------------------------------------------------------------------


@tool
def dispatch_subtask(subtask: str) -> str:
    """Send a subtask to the Cline CLI for execution.

    Args:
        subtask: The task description to send to Cline.

    Returns:
        A summary of the Cline PTY session output.
    """
    bridge = _get_bridge()
    if not bridge.is_alive():
        bridge.spawn()

    output = bridge.send_task(subtask)
    logger.info("dispatch_subtask completed: %d chars output", len(output))
    return output if output else "(no output from Cline)"


@tool
def answer_question(question: str, context: str) -> str:
    """Answer a question from the Cline CLI using shared context.

    Formulates an answer using the LLM and injects it into the PTY stdin.

    Args:
        question: The question asked by Cline.
        context: Relevant project context to inform the answer.

    Returns:
        The answer that was injected.
    """
    llm = get_llm(temperature=0.0)

    prompt = (
        f"Answer this question from the Cline CLI concisely and accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (one line, no explanation):"
    )

    response = llm.invoke(prompt)
    answer = str(response.content).strip()

    bridge = _get_bridge()
    if bridge.is_alive():
        bridge.inject(answer)
        logger.info("Injected answer: %s", answer)

    return answer


@tool
def approve_cline_action(action: str, action_type: str) -> str:
    """Evaluate and approve/reject a Cline action based on decision policy.

    This tool triggers an interrupt for human review when the action
    requires escalation.

    Args:
        action: Description of the action Cline wants to perform.
        action_type: Category of the action (e.g., 'file_write', 'http_request', 'delete').

    Returns:
        The approval decision.
    """
    # Auto-approve safe actions
    auto_approve_types = {
        "file_read", "file_write", "file_create", "file_edit",
        "local_command", "git_local", "test_run",
    }

    if action_type in auto_approve_types:
        bridge = _get_bridge()
        if bridge.is_alive():
            bridge.inject("yes")
        return f"auto-approved: {action_type} — {action}"

    # Everything else needs escalation — return info for interrupt
    return f"escalate: {action_type} — {action}"


# ---------------------------------------------------------------------------
# Subagent definition
# ---------------------------------------------------------------------------


def get_cline_executor_subagent() -> dict[str, Any]:
    """Return the cline-executor subagent definition dict.

    This dict is passed to create_deep_agent(subagents=[...]).
    """
    return {
        "name": "cline-executor",
        "description": (
            "Executes subtasks by delegating to the Cline CLI. "
            "Handles Cline questions, approval prompts, and streams output. "
            "Use this subagent whenever a subtask needs to be executed via Cline."
        ),
        "system_prompt": (
            "You are the Cline Executor. Your job is to:\n"
            "1. Receive subtasks from the agent manager\n"
            "2. Dispatch them to the Cline CLI via dispatch_subtask\n"
            "3. Answer any questions Cline asks using answer_question\n"
            "4. Handle approval prompts using approve_cline_action\n"
            "5. Return a summary of what was accomplished\n\n"
            "Always check the output for errors. If Cline crashes or fails, "
            "report the failure clearly."
        ),
        "tools": [dispatch_subtask, answer_question, approve_cline_action],
        "skills": ["/skills/cline_qa/"],
        "interrupt_on": {
            "approve_cline_action": {
                "allowed_decisions": ["approve", "edit", "reject"],
                "description": "Cline wants to perform an action that requires user approval",
            },
            "dispatch_subtask": {
                "allowed_decisions": ["approve", "reject"],
                "description": "A subtask is about to be dispatched to Cline",
            },
            "answer_question": False,
        },
    }
