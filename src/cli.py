"""CLI entry point for the Cline Agent Manager."""

from __future__ import annotations

import json
import logging
import os
import uuid

from dotenv import load_dotenv

from src.agent import create_agent_manager
from src.hitl import build_approve_command, build_reject_command, extract_interrupts
from src.subagent import get_cline_executor_subagent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the Cline Agent Manager interactively."""
    print("Cline Agent Manager")
    print(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'anthropic')}")
    print("=" * 50)

    subagent = get_cline_executor_subagent()
    agent = create_agent_manager(
        subagents=[subagent],
        skills=["/skills/"],
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nSession: {thread_id}")
    print("Type your task (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
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
                print("\n--- Approval Required ---")
                for i, req in enumerate(interrupts):
                    print(f"  [{i}] {req.tool_name}: {req.description or json.dumps(req.action)}")

                decision = input("  [a]pprove / [r]eject? ").strip().lower()
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
                print(f"\n{content}\n")

        except Exception:
            logger.exception("Error during agent invocation")
            print("An error occurred. See logs for details.\n")


if __name__ == "__main__":
    main()
