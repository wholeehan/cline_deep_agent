"""Tests for benchmark configuration models."""

import pytest
from pydantic import ValidationError

from benchmark.config import (
    BenchmarkReport,
    RunConfig,
    TaskConfig,
    TaskResult,
    TokenUsage,
)


class TestTaskConfig:
    def test_valid_config(self):
        cfg = TaskConfig(
            id="bugfix-001",
            title="Fix a bug",
            category="bugfix",
            difficulty="easy",
        )
        assert cfg.id == "bugfix-001"
        assert cfg.language == "python"
        assert cfg.token_budget == 50_000
        assert cfg.timeout_seconds == 600
        assert cfg.tags == []

    def test_all_categories(self):
        for cat in ("bugfix", "feature", "refactor", "multifile"):
            cfg = TaskConfig(id="t", title="t", category=cat, difficulty="easy")
            assert cfg.category == cat

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            TaskConfig(id="t", title="t", category="unknown", difficulty="easy")

    def test_all_difficulties(self):
        for diff in ("easy", "medium", "hard"):
            cfg = TaskConfig(id="t", title="t", category="bugfix", difficulty=diff)
            assert cfg.difficulty == diff

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            TaskConfig(id="t", title="t", category="bugfix", difficulty="extreme")

    def test_custom_fields(self):
        cfg = TaskConfig(
            id="t",
            title="t",
            category="feature",
            difficulty="hard",
            language="javascript",
            expected_files_changed=5,
            token_budget=10_000,
            timeout_seconds=120,
            tags=["api", "auth"],
        )
        assert cfg.language == "javascript"
        assert cfg.expected_files_changed == 5
        assert cfg.tags == ["api", "auth"]


class TestRunConfig:
    def test_defaults(self):
        cfg = RunConfig()
        assert cfg.agent == "cline-deep"
        assert cfg.tasks == "all"
        assert cfg.repetitions == 1
        assert cfg.parallel == 1

    def test_task_list(self):
        cfg = RunConfig(tasks=["bugfix-001", "feature-001"])
        assert cfg.tasks == ["bugfix-001", "feature-001"]


class TestTokenUsage:
    def test_defaults(self):
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.total_tokens == 0

    def test_values(self):
        t = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert t.total_tokens == 150


class TestTaskResult:
    def test_minimal(self):
        r = TaskResult(task_id="t", repetition=1, passed=True, wall_clock_seconds=1.5)
        assert r.passed is True
        assert r.files_changed == []
        assert r.error is None

    def test_with_error(self):
        r = TaskResult(
            task_id="t", repetition=1, passed=False,
            wall_clock_seconds=0.1, error="timeout",
        )
        assert r.error == "timeout"


class TestBenchmarkReport:
    def test_roundtrip_json(self):
        report = BenchmarkReport(
            run_id="test",
            timestamp="2026-01-01T00:00:00Z",
            agent="cline-deep",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            tasks=[
                TaskResult(task_id="t", repetition=1, passed=True, wall_clock_seconds=5.0),
            ],
            summary={"pass_rate": 1.0},
        )
        data = report.model_dump_json()
        restored = BenchmarkReport.model_validate_json(data)
        assert restored.run_id == "test"
        assert restored.tasks[0].passed is True
        assert restored.summary["pass_rate"] == 1.0
