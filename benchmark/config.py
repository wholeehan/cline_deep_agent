"""Pydantic models for benchmark configuration and results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    """Parsed from task.toml."""

    id: str
    title: str
    category: Literal["bugfix", "feature", "refactor", "multifile"]
    difficulty: Literal["easy", "medium", "hard"]
    language: str = "python"
    expected_files_changed: int = 1
    token_budget: int = 50_000
    timeout_seconds: int = 600
    tags: list[str] = Field(default_factory=list)


class RunConfig(BaseModel):
    """CLI run parameters."""

    agent: str = "cline-deep"
    provider: str = "ollama"
    model: str = "gpt-oss:20b"
    tasks: list[str] | Literal["all"] = "all"
    repetitions: int = 1
    parallel: int = 1


class TokenUsage(BaseModel):
    """Token consumption for a single run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TaskResult(BaseModel):
    """Result of running a single task once."""

    task_id: str
    repetition: int
    passed: bool
    wall_clock_seconds: float
    token_usage: TokenUsage | None = None
    cost_usd: float | None = None
    files_changed: list[str] = Field(default_factory=list)
    test_output: str = ""
    error: str | None = None


class BenchmarkReport(BaseModel):
    """Full report from a benchmark run."""

    run_id: str
    timestamp: str
    agent: str
    provider: str
    model: str
    tasks: list[TaskResult]
    summary: dict[str, Any] = Field(default_factory=dict)
