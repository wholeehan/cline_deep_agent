"""Tests for benchmark leaderboard generation."""

from benchmark.config import BenchmarkReport, TaskResult, TokenUsage
from benchmark.leaderboard import generate_markdown


class TestGenerateMarkdown:
    def _make_report(self, agent="test-agent", model="test-model", pass_rate=0.8):
        return BenchmarkReport(
            run_id="test",
            timestamp="2026-01-01T00:00:00Z",
            agent=agent,
            provider="anthropic",
            model=model,
            tasks=[],
            summary={
                "pass_rate": pass_rate,
                "pass_at_k": {"pass@1": pass_rate},
                "avg_tokens": 5000.0,
                "avg_cost_usd": 0.05,
            },
        )

    def test_empty(self):
        result = generate_markdown([])
        assert "No results" in result

    def test_single_report(self):
        md = generate_markdown([self._make_report()])
        assert "test-agent" in md
        assert "test-model" in md
        assert "80%" in md

    def test_multiple_reports(self):
        reports = [
            self._make_report(agent="agent-a", pass_rate=0.9),
            self._make_report(agent="agent-b", pass_rate=0.6),
        ]
        md = generate_markdown(reports)
        assert "agent-a" in md
        assert "agent-b" in md
        assert "90%" in md
        assert "60%" in md

    def test_contains_table_headers(self):
        md = generate_markdown([self._make_report()])
        assert "Agent" in md
        assert "Model" in md
        assert "Pass Rate" in md
        assert "Avg Tokens" in md
