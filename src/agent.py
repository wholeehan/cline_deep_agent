"""DeepAgents core agent setup — manager agent + composite backend."""

from __future__ import annotations

import logging
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from src.llm import get_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------


@tool
def report_progress(message: str) -> str:
    """Append a progress message to /workspace/progress.log and return it.

    Use this to report status updates to the user during task execution.
    """
    return f"[progress] {message}"


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def create_composite_backend(runtime: Any) -> CompositeBackend:
    """Build a CompositeBackend routing paths to appropriate backends.

    - /workspace/ → StateBackend (ephemeral, per-thread)
    - /memories/  → StoreBackend (persistent, cross-thread)
    - /project/   → FilesystemBackend(virtual_mode=True)
    """
    state = StateBackend(runtime)
    store = StoreBackend(runtime)
    filesystem = FilesystemBackend(virtual_mode=True)

    return CompositeBackend(
        default=state,
        routes={
            "/workspace/": state,
            "/memories/": store,
            "/project/": filesystem,
        },
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agent_manager(
    subagents: list[dict[str, Any]] | None = None,
    skills: list[str] | None = None,
    interrupt_on: dict[str, Any] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Create the main agent manager using deepagents.

    Returns a compiled LangGraph agent.
    """
    llm = get_llm()

    if checkpointer is None:
        checkpointer = MemorySaver()

    agent = create_deep_agent(
        model=llm,
        tools=[report_progress],
        system_prompt=(
            "You are the Cline Agent Manager. You orchestrate software engineering tasks "
            "by decomposing them into subtasks, dispatching them to the cline-executor "
            "subagent, verifying results, and reporting progress to the user. "
            "Use write_todos to plan, task() to delegate, and report_progress to update the user."
        ),
        subagents=subagents or [],  # type: ignore[arg-type]
        skills=skills or [],
        checkpointer=checkpointer,
        backend=create_composite_backend,
        interrupt_on=interrupt_on or {},
        name="agent-manager",
    )

    return agent
