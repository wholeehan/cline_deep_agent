"""Abstract base class for agent adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from benchmark.config import RunConfig, TokenUsage


class RunResult(BaseModel):
    """Outcome of an adapter invocation on a single task."""

    wall_clock_seconds: float
    token_usage: TokenUsage | None = None
    cost_usd: float | None = None
    files_changed: list[str] = Field(default_factory=list)
    error: str | None = None


class AgentAdapter(ABC):
    """Protocol that every benchmark adapter must implement."""

    name: str

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config or RunConfig()

    @abstractmethod
    def run(self, instruction: str, workspace: Path, timeout: int) -> RunResult:
        """Execute the agent on a task workspace and return metrics."""
        ...

    def setup(self) -> None:
        """Optional one-time setup before benchmark runs."""

    def teardown(self) -> None:
        """Optional cleanup after all benchmark runs complete."""
