"""Tests for benchmark scorer."""

from benchmark.config import BenchmarkReport, TaskResult, TokenUsage
from benchmark.scorer import compute_summary, estimate_cost, pass_at_k


class TestPassAtK:
    def test_all_pass(self):
        assert pass_at_k(5, 5, 1) == 1.0
        assert pass_at_k(5, 5, 3) == 1.0

    def test_none_pass(self):
        assert pass_at_k(5, 0, 1) == 0.0
        assert pass_at_k(5, 0, 3) == 0.0

    def test_partial(self):
        # 4 out of 5 correct, pass@1
        result = pass_at_k(5, 4, 1)
        assert 0.7 < result < 0.9

    def test_pass_at_3_higher_than_1(self):
        p1 = pass_at_k(5, 3, 1)
        p3 = pass_at_k(5, 3, 3)
        assert p3 > p1

    def test_n_less_than_k(self):
        assert pass_at_k(2, 1, 3) == 1.0  # c > 0
        assert pass_at_k(2, 0, 3) == 0.0  # c == 0

    def test_single_attempt(self):
        assert pass_at_k(1, 1, 1) == 1.0
        assert pass_at_k(1, 0, 1) == 0.0


class TestEstimateCost:
    def test_known_model(self):
        r = TaskResult(
            task_id="t", repetition=1, passed=True, wall_clock_seconds=1.0,
            token_usage=TokenUsage(
                prompt_tokens=1_000_000, completion_tokens=100_000, total_tokens=1_100_000,
            ),
        )
        cost = estimate_cost(r, "claude-3-5-sonnet-20241022")
        assert cost is not None
        # 1M input * $3/1M + 100k output * $15/1M = $3 + $1.5 = $4.5
        assert abs(cost - 4.5) < 0.01

    def test_unknown_model(self):
        r = TaskResult(
            task_id="t", repetition=1, passed=True, wall_clock_seconds=1.0,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        assert estimate_cost(r, "unknown-model") is None

    def test_no_token_usage(self):
        r = TaskResult(task_id="t", repetition=1, passed=True, wall_clock_seconds=1.0)
        assert estimate_cost(r, "claude-3-5-sonnet-20241022") is None


class TestComputeSummary:
    def _make_report(self, results):
        return BenchmarkReport(
            run_id="test", timestamp="t", agent="a", provider="p",
            model="claude-3-5-sonnet-20241022", tasks=results,
        )

    def test_empty(self):
        s = compute_summary(self._make_report([]))
        assert s["total_tasks"] == 0
        assert s["passed"] == 0
        assert s["pass_rate"] == 0.0
        assert s["avg_tokens"] == 0.0

    def test_all_pass(self):
        results = [
            TaskResult(
                task_id="bugfix-001", repetition=i, passed=True,
                wall_clock_seconds=10.0,
                token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
            )
            for i in range(1, 4)
        ]
        s = compute_summary(self._make_report(results))
        assert s["total_tasks"] == 3
        assert s["passed"] == 3
        assert s["pass_rate"] == 1.0
        assert s["avg_tokens"] == 1500.0

    def test_mixed_results(self):
        results = [
            TaskResult(task_id="bugfix-001", repetition=1, passed=True, wall_clock_seconds=5.0),
            TaskResult(task_id="bugfix-001", repetition=2, passed=False, wall_clock_seconds=8.0),
            TaskResult(task_id="feature-001", repetition=1, passed=True, wall_clock_seconds=12.0),
        ]
        s = compute_summary(self._make_report(results))
        assert s["passed"] == 2
        assert 0.66 < s["pass_rate"] < 0.67
        assert "bugfix" in s["by_category"]
        assert "feature" in s["by_category"]

    def test_by_category(self):
        results = [
            TaskResult(task_id="bugfix-001", repetition=1, passed=True, wall_clock_seconds=1.0),
            TaskResult(task_id="bugfix-002", repetition=1, passed=False, wall_clock_seconds=1.0),
            TaskResult(task_id="feature-001", repetition=1, passed=True, wall_clock_seconds=1.0),
        ]
        s = compute_summary(self._make_report(results))
        assert s["by_category"]["bugfix"]["pass_rate"] == 0.5
        assert s["by_category"]["feature"]["pass_rate"] == 1.0
