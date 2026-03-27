"""CLI entry point for the Cline Agent Manager."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import ExitStack
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console

from openai import APIConnectionError as OpenAIConnectionError

from src.agent import create_agent_manager
from src.hitl import build_approve_command, build_reject_command, extract_interrupts
from src.llm import OllamaUnavailableError
from src.logging_config import configure_logging
from src.subagent import get_cline_executor_subagent

load_dotenv()

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation display
# ---------------------------------------------------------------------------


def _display_messages(messages: list[Any], seen_count: int) -> int:
    """Display new messages since the last turn using Rich formatting.

    Shows agent text, tool calls, and tool results in a clean conversation
    format instead of raw JSON logs.

    Returns the new seen_count.
    """
    for msg in messages[seen_count:]:
        if isinstance(msg, HumanMessage):
            # User messages are already shown via the input prompt
            pass
        elif isinstance(msg, AIMessage):
            # Show agent text
            if msg.content:
                console.print(f"\n[bold cyan]Agent:[/] {msg.content}")
            # Show tool calls
            for tc in getattr(msg, "tool_calls", None) or []:
                name = tc.get("name", "?")
                args = tc.get("args", {})
                args_str = json.dumps(args, ensure_ascii=False)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                console.print(
                    f"  [dim]\u2192 calling [yellow]{name}[/yellow]({args_str})[/dim]"
                )
        elif isinstance(msg, ToolMessage):
            # Show tool result (truncated for readability)
            name = getattr(msg, "name", None) or "tool"
            content = str(msg.content)
            if len(content) > 300:
                content = content[:300] + "..."
            console.print(
                f"  [dim]\u2190 [green]{name}[/green]: {content}[/dim]"
            )

    return len(messages)

_BANNER = r"""
   _____ _ _
  / ____| (_)
 | |    | |_ _ __   ___
 | |    | | | '_ \ / _ \
 | |____| | | | | |  __/
  \_____|_|_|_| |_|\___|
"""


# ---------------------------------------------------------------------------
# Data directory setup
# ---------------------------------------------------------------------------

_DATA_DIRS = [
    "data/output",
    "data/logs",
    "data/traces",
]


def _ensure_data_dirs() -> None:
    """Create data directories for persistence on startup."""
    for d in _DATA_DIRS:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Telemetry callback
# ---------------------------------------------------------------------------


def _open_telemetry_handler(
    stack: ExitStack, session_id: str | None = None
) -> list:  # type: ignore[type-arg]
    """Build LangChain callbacks for LLM telemetry logging.

    Uses ExitStack so the FileCallbackHandler is properly closed on exit.
    """
    callbacks: list = []  # type: ignore[type-arg]

    telemetry_path = os.getenv("TELEMETRY_FILE", "./data/logs/telemetry.log")
    telemetry_dir = os.path.dirname(telemetry_path)
    if telemetry_dir:
        os.makedirs(telemetry_dir, exist_ok=True)

    try:
        from langchain_core.callbacks.file import FileCallbackHandler

        handler = stack.enter_context(FileCallbackHandler(telemetry_path))
        callbacks.append(handler)
        logger.info("Telemetry logging to %s", telemetry_path)
    except ImportError:
        logger.warning("FileCallbackHandler not available; telemetry logging disabled")

    # Optional PostgreSQL telemetry handler (dual-write with file)
    if os.getenv("DATABASE_URL"):
        try:
            from src.db import get_connection_pool
            from src.telemetry import PostgresCallbackHandler, register_global_callback

            pool = get_connection_pool()
            if pool is not None:
                pg_cb = PostgresCallbackHandler(pool, session_id=session_id)
                callbacks.append(pg_cb)
                # Register globally so ALL LLM instances (subagents, standalone
                # tool calls) automatically get telemetry capture via get_llm()
                register_global_callback(pg_cb)
                logger.info("Telemetry also logging to PostgreSQL llm_traces table")
        except Exception:
            logger.warning(
                "Failed to set up PostgreSQL telemetry handler", exc_info=True
            )

    return callbacks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Cline Agent Manager interactively."""
    _ensure_data_dirs()

    # Configure structured logging (stdout + optional file via LOG_FILE env var)
    configure_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        json_output=True,
    )

    console.print(f"[bold cyan]{_BANNER}[/]")
    console.print(f"[dim]LLM Provider: {os.getenv('LLM_PROVIDER', 'anthropic')}[/]")
    console.print("[dim]" + "=" * 50 + "[/]")

    with ExitStack() as stack:
        thread_id = str(uuid.uuid4())
        callbacks = _open_telemetry_handler(stack, session_id=thread_id)

        try:
            subagent = get_cline_executor_subagent(callbacks=callbacks)
            agent = create_agent_manager(
                subagents=[subagent],
                skills=["/skills/"],
            )
        except OllamaUnavailableError as exc:
            logger.error("Ollama startup check failed: %s", exc)
            console.print(f"[bold red]{exc}[/]\n")
            return
        config = {"configurable": {"thread_id": thread_id}, "callbacks": callbacks}

        console.print(f"\n[green]Session: {thread_id}[/]")
        console.print("Type your task (or 'quit' to exit):\n")

        seen_count = 0  # Track how many messages we've already displayed

        while True:
            try:
                user_input = console.input("[bold]> [/]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/]")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/]")
                break

            if not user_input:
                continue

            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )

                # Display new messages (tool calls, results, agent text)
                messages = result.get("messages", [])
                seen_count = _display_messages(messages, seen_count)

                # Check for interrupts
                interrupts = extract_interrupts(result)
                while interrupts:
                    console.print("\n[bold yellow]--- Approval Required ---[/]")
                    for i, req in enumerate(interrupts):
                        desc = req.description or json.dumps(req.action)
                        console.print(
                            f"  \\[{i}] [yellow]{req.tool_name}[/]: {desc}"
                        )

                    decision = console.input("  [yellow]\\[a]pprove / \\[r]eject?[/] ").strip().lower()
                    if decision.startswith("a"):
                        cmd = build_approve_command()
                    else:
                        cmd = build_reject_command()

                    result = agent.invoke(cmd, config=config)

                    # Display messages from the resumed invocation
                    messages = result.get("messages", [])
                    seen_count = _display_messages(messages, seen_count)
                    interrupts = extract_interrupts(result)

                console.print()  # Blank line before next prompt

            except OpenAIConnectionError:
                if os.getenv("LLM_PROVIDER") != "vllm":
                    raise
                base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
                model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-Coder-Next")
                logger.exception("Cannot reach vLLM server")
                console.print(
                    f"[bold red]Cannot reach vLLM server at {base_url}.[/]\n"
                    f"Ensure vLLM is running with the expected model:\n\n"
                    f"  [cyan]vllm serve {model} "
                    f"--tensor-parallel-size 1 "
                    f"--gpu-memory-utilization 0.85 "
                    f"--max-model-len 16384 "
                    f"--enable-auto-tool-choice "
                    f"--tool-call-parser qwen3_coder[/]\n"
                )
            except Exception:
                logger.exception("Error during agent invocation")
                console.print("[bold red]An error occurred. See logs for details.[/]\n")


if __name__ == "__main__":
    main()
