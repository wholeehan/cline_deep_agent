"""CLI entry point for the Cline Agent Manager."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import ExitStack

from dotenv import load_dotenv
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
            from src.telemetry import PostgresCallbackHandler

            pool = get_connection_pool()
            if pool is not None:
                pg_cb = PostgresCallbackHandler(pool, session_id=session_id)
                callbacks.append(pg_cb)
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

    try:
        subagent = get_cline_executor_subagent()
        agent = create_agent_manager(
            subagents=[subagent],
            skills=["/skills/"],
        )
    except OllamaUnavailableError as exc:
        logger.error("Ollama startup check failed: %s", exc)
        console.print(f"[bold red]{exc}[/]\n")
        return

    with ExitStack() as stack:
        thread_id = str(uuid.uuid4())
        callbacks = _open_telemetry_handler(stack, session_id=thread_id)
        config = {"configurable": {"thread_id": thread_id}, "callbacks": callbacks}

        console.print(f"\n[green]Session: {thread_id}[/]")
        console.print("Type your task (or 'quit' to exit):\n")

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
                    interrupts = extract_interrupts(result)

                # Print final response
                messages = result.get("messages", [])
                if messages:
                    last = messages[-1]
                    content = getattr(last, "content", str(last))
                    console.print(f"\n{content}\n")

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
