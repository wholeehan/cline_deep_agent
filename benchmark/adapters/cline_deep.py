"""Adapter that invokes this project's agent programmatically."""

from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

from benchmark.adapters.base import AgentAdapter, RunResult
from benchmark.config import TokenUsage

logger = logging.getLogger(__name__)


class ClineDeepAdapter(AgentAdapter):
    """Runs the cline-deep agent on a benchmark task."""

    name = "cline-deep"

    def run(self, instruction: str, workspace: Path, timeout: int) -> RunResult:
        """Execute the agent against the workspace and return metrics."""
        from src.agent import create_agent_manager
        from src.hitl import build_approve_command, extract_interrupts
        from src.subagent import get_cline_executor_subagent

        # Save and override environment for this run
        saved_env = self._set_provider_env()

        # Point output to the workspace directory
        original_output_dir = os.environ.get("OUTPUT_DIR")
        os.environ["OUTPUT_DIR"] = str(workspace)

        start = time.perf_counter()
        token_usage = TokenUsage()
        error: str | None = None
        files_before = self._snapshot_files(workspace)

        try:
            subagent = get_cline_executor_subagent()
            agent = create_agent_manager(subagents=[subagent], skills=["/skills/"])

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            result = agent.invoke(
                {"messages": [{"role": "user", "content": instruction}]},
                config=config,
            )

            # Auto-approve all HITL interrupts (same pattern as cli.py)
            interrupts = extract_interrupts(result)
            approve_rounds = 0
            max_rounds = 50
            while interrupts and approve_rounds < max_rounds:
                cmd = build_approve_command()
                result = agent.invoke(cmd, config=config)
                interrupts = extract_interrupts(result)
                approve_rounds += 1

            # Try to extract token usage from callback metadata
            token_usage = self._extract_token_usage(result)

        except Exception as exc:
            logger.exception("Agent invocation failed for task in %s", workspace)
            error = str(exc)
        finally:
            elapsed = time.perf_counter() - start
            # Restore OUTPUT_DIR
            if original_output_dir is not None:
                os.environ["OUTPUT_DIR"] = original_output_dir
            else:
                os.environ.pop("OUTPUT_DIR", None)
            # Restore provider env vars
            self._restore_env(saved_env)

        files_after = self._snapshot_files(workspace)
        changed = sorted(files_after - files_before)

        return RunResult(
            wall_clock_seconds=elapsed,
            token_usage=token_usage,
            files_changed=changed,
            error=error,
        )

    # Provider-to-env mapping
    _PROVIDER_ENV: dict[str, tuple[str, str]] = {
        "ollama": ("OLLAMA_MODEL", "OLLAMA_BASE_URL"),
        "vllm": ("VLLM_MODEL", "VLLM_BASE_URL"),
        "anthropic": ("ANTHROPIC_MODEL", ""),
    }

    def _set_provider_env(self) -> dict[str, str | None]:
        """Set LLM_PROVIDER and model env vars from RunConfig. Returns saved values."""
        saved: dict[str, str | None] = {}
        keys_to_set = {"LLM_PROVIDER": self.config.provider}

        model_key = self._PROVIDER_ENV.get(self.config.provider, ("", ""))[0]
        if model_key:
            keys_to_set[model_key] = self.config.model

        for key, value in keys_to_set.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = value

        return saved

    @staticmethod
    def _restore_env(saved: dict[str, str | None]) -> None:
        """Restore environment variables from saved dict."""
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    @staticmethod
    def _snapshot_files(workspace: Path) -> set[str]:
        """Return set of relative file paths in the workspace."""
        result: set[str] = set()
        for p in workspace.rglob("*"):
            if p.is_file():
                result.add(str(p.relative_to(workspace)))
        return result

    @staticmethod
    def _extract_token_usage(result: dict) -> TokenUsage:  # type: ignore[type-arg]
        """Best-effort extraction of token usage from agent result metadata."""
        try:
            messages = result.get("messages", [])
            prompt_total = 0
            completion_total = 0
            for msg in messages:
                usage = getattr(msg, "usage_metadata", None)
                if usage and isinstance(usage, dict):
                    prompt_total += usage.get("input_tokens", 0)
                    completion_total += usage.get("output_tokens", 0)
            return TokenUsage(
                prompt_tokens=prompt_total,
                completion_tokens=completion_total,
                total_tokens=prompt_total + completion_total,
            )
        except Exception:
            return TokenUsage()

    @staticmethod
    def _get_git_changed_files(workspace: Path) -> list[str]:
        """Use git diff to detect changed files if workspace is a git repo."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return [f for f in result.stdout.strip().split("\n") if f]
        except Exception:
            pass
        return []
