"""DeepAgents core agent setup — manager agent + composite backend."""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from src.llm import get_llm

logger = logging.getLogger(__name__)

# Try to import SqliteSaver; fall back to MemorySaver if unavailable
try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False

# Try to import PostgresSaver for PostgreSQL-backed checkpointing
try:
    from langgraph.checkpoint.postgres import PostgresSaver

    _POSTGRES_AVAILABLE = True
except ImportError:
    _POSTGRES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------


@tool
def report_progress(message: str) -> str:
    """Append a progress message to /workspace/progress.log and return it.

    Use this to report status updates to the user during task execution.
    """
    return f"[progress] {message}"


@tool
def save_output(file_path: str, content: str) -> str:
    """Save agent-generated code or output to the persistent output directory.

    Files saved here survive restarts and are available at data/output/ on disk.

    Args:
        file_path: Relative path within the output directory (e.g. "plan.md", "src/main.py").
        content: The text content to write.
    """
    output_dir = os.getenv("OUTPUT_DIR", "./data/output")
    # Prevent path traversal
    clean = os.path.normpath(file_path).lstrip(os.sep)
    if clean.startswith(".."):
        return "Error: path traversal not allowed"
    full_path = os.path.join(output_dir, clean)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved to {full_path}"


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def create_composite_backend(runtime: Any) -> CompositeBackend:
    """Build a CompositeBackend routing paths to appropriate backends.

    - /workspace/ → StateBackend (ephemeral, per-thread)
    - /memories/  → StoreBackend (persistent, cross-thread)
    - /project/   → FilesystemBackend(virtual_mode=True)
    - /output/    → FilesystemBackend rooted at OUTPUT_DIR (persistent to disk)
    """
    state = StateBackend(runtime)
    store = StoreBackend(runtime)
    filesystem = FilesystemBackend(virtual_mode=True)

    output_dir = os.getenv("OUTPUT_DIR", "./data/output")
    os.makedirs(output_dir, exist_ok=True)
    output_backend = FilesystemBackend(root_dir=output_dir, virtual_mode=True)

    return CompositeBackend(
        default=state,
        routes={
            "/workspace/": state,
            "/memories/": store,
            "/project/": filesystem,
            "/output/": output_backend,
        },
    )


# ---------------------------------------------------------------------------
# Checkpointer factory
# ---------------------------------------------------------------------------


def get_checkpointer(db_path: str | None = None) -> Any:
    """Return a persistent checkpointer: PostgreSQL > SQLite > MemorySaver.

    Priority:
    1. PostgresSaver when DATABASE_URL is set and langgraph-checkpoint-postgres installed
    2. SqliteSaver when CHECKPOINT_DB is set and langgraph-checkpoint-sqlite installed
    3. MemorySaver as final fallback (state will not persist across restarts)
    """
    # Prefer PostgreSQL when DATABASE_URL is configured
    db_url = os.getenv("DATABASE_URL")
    if db_url and _POSTGRES_AVAILABLE:
        import psycopg

        logger.info("Using PostgresSaver checkpointer with DATABASE_URL")
        # Setup requires autocommit for CREATE INDEX CONCURRENTLY
        setup_conn = psycopg.connect(db_url, autocommit=True)
        PostgresSaver(setup_conn).setup()
        setup_conn.close()
        conn = psycopg.connect(db_url)
        return PostgresSaver(conn)
    if db_url and not _POSTGRES_AVAILABLE:
        logger.warning(
            "DATABASE_URL set but langgraph-checkpoint-postgres not installed; "
            "falling back to SQLite or MemorySaver"
        )

    # Fall back to SQLite
    path = db_path or os.getenv("CHECKPOINT_DB")
    if path and _SQLITE_AVAILABLE:
        db_dir = os.path.dirname(path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        logger.info("Using SqliteSaver checkpointer at %s", path)
        conn = sqlite3.connect(path, check_same_thread=False)
        return SqliteSaver(conn)
    if path and not _SQLITE_AVAILABLE:
        logger.warning(
            "CHECKPOINT_DB set but langgraph-checkpoint-sqlite not installed; "
            "falling back to MemorySaver (state will not persist across restarts)"
        )
    return MemorySaver()


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
        checkpointer = get_checkpointer()

    agent = create_deep_agent(
        model=llm,
        tools=[report_progress, save_output],
        system_prompt=(
            "You are the Cline Agent Manager. You help users with software engineering tasks.\n\n"
            "## Simple tasks — do them yourself:\n"
            "For writing code, scripts, functions, configs, or fixing known bugs:\n"
            "- Write the code yourself and save it with save_output.\n"
            "- Use read_file / write_file / edit_file for file operations.\n"
            "- Do NOT delegate simple tasks to the cline-executor.\n\n"
            "## Complex tasks — delegate ONCE:\n"
            "Only use task(subagent_type='cline-executor') for tasks that genuinely\n"
            "require the Cline CLI (interactive multi-file refactors, autonomous coding\n"
            "sessions, tasks requiring shell interaction).\n"
            "- Call task() exactly ONCE per subtask.\n"
            "- When the subagent returns a result, ACCEPT it. Do NOT re-call task()\n"
            "  with the same or similar description.\n"
            "- If the result is unsatisfactory, tell the user — don't retry silently.\n\n"
            "## File paths:\n"
            "- /workspace/ — temporary scratch (ephemeral, lost on restart)\n"
            "- /output/ — permanent storage (persisted to data/output/)\n"
            "- Always save final deliverables to /output/ using save_output.\n\n"
            "## Completion:\n"
            "Once the task is done, present the result to the user and STOP.\n"
            "Do not re-run, re-verify, or re-delegate completed work."
        ),
        subagents=subagents or [],  # type: ignore[arg-type]
        skills=skills or [],
        checkpointer=checkpointer,
        store=InMemoryStore(),
        backend=create_composite_backend,
        interrupt_on=interrupt_on or {},
        name="agent-manager",
    )

    return agent
