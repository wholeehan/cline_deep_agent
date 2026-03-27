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


def get_cline_executor_subagent(
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    """Return the cline-executor subagent definition dict.

    This dict is passed to create_deep_agent(subagents=[...]).

    The subagent uses its own LLM model (defaulting to qwen3-coder:latest
    via ``OLLAMA_SUBAGENT_MODEL``), while the supervisor uses the main
    ``OLLAMA_MODEL`` (gpt-oss:20b by default).

    Parameters
    ----------
    callbacks:
        Optional list of LangChain callbacks to bind directly to the
        subagent's LLM instance. This ensures telemetry is captured for
        subagent LLM calls even when the framework doesn't forward
        parent config callbacks.
    """
    import os

    subagent_model = os.getenv("OLLAMA_SUBAGENT_MODEL", "qwen3-coder-tools:latest")
    # skip_tool_check=False: verify the subagent model supports tool calling,
    # since the deepagents framework requires it via bind_tools()
    subagent_llm = get_llm(
        temperature=0.0, model_override=subagent_model, skip_tool_check=False,
        callbacks=callbacks or [],
    )
    logger.info(
        "cline-executor subagent LLM created: model=%s callbacks=%s",
        subagent_llm.model,
        [type(cb).__name__ for cb in (subagent_llm.callbacks or [])],
    )

    return {
        "name": "cline-executor",
        "model": subagent_llm,
        "description": (
            "Executes subtasks by delegating to the Cline CLI. "
            "Handles Cline questions, approval prompts, and streams output. "
            "Use this subagent whenever a subtask needs to be executed via Cline."
        ),
        "system_prompt": (
            "You are the Cline Executor. You run tasks via the Cline CLI.\n\n"
            "## Workflow:\n"
            "1. Call dispatch_subtask with the task description.\n"
            "2. If Cline asks a question, answer it using answer_question.\n"
            "3. If Cline requests approval for a risky action, use approve_cline_action.\n\n"
            "## Rules:\n"
            "- Call dispatch_subtask exactly ONCE for the given task.\n"
            "- Do NOT retry or re-dispatch if the first attempt produces output.\n"
            "- Include the full output (code, file contents, results) in your response,\n"
            "  not just a summary.\n"
            "- If Cline fails or crashes, report the error clearly and stop.\n"
            "- When done, return the result immediately. Do not call additional tools."
        ),
        "tools": [dispatch_subtask, answer_question, approve_cline_action],
        "skills": ["/skills/cline_qa/"],
        "interrupt_on": {
            "approve_cline_action": {
                "allowed_decisions": ["approve", "edit", "reject"],
                "description": "Cline wants to perform an action that requires user approval",
            },
            "dispatch_subtask": False,
            "answer_question": False,
        },
    }
