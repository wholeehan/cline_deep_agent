"""Metrics computation for benchmark results."""

from __future__ import annotations

from math import comb
from typing import Any

from benchmark.config import BenchmarkReport, TaskResult

# Price table: $/1M tokens (input, output) — extend as needed
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute unbiased pass@k estimator.

    n = total attempts, c = number of correct, k = draws.
    Formula: 1 - C(n-c, k) / C(n, k)
    """
    if n < k:
        return float(c > 0)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def estimate_cost(result: TaskResult, model: str) -> float | None:
    """Estimate USD cost for a single task result."""
    if result.token_usage is None:
        return None
    prices = PRICE_TABLE.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    cost = (
        result.token_usage.prompt_tokens * input_price / 1_000_000
        + result.token_usage.completion_tokens * output_price / 1_000_000
    )
    return round(cost, 6)


def compute_summary(report: BenchmarkReport) -> dict[str, Any]:
    """Compute aggregate metrics from a benchmark report."""
    results = report.tasks
    if not results:
        return {
            "total_tasks": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "pass_at_k": {},
            "pass_at_k_by_task": {},
            "avg_tokens": 0.0,
            "avg_cost_usd": 0.0,
            "avg_wall_clock_seconds": 0.0,
            "by_category": {},
        }

    # Fill in cost estimates
    for r in results:
        if r.cost_usd is None:
            r.cost_usd = estimate_cost(r, report.model)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / total if total > 0 else 0.0

    # pass@k per task_id
    task_ids = sorted({r.task_id for r in results})
    pass_at_k_scores: dict[str, dict[str, float]] = {}
    for tid in task_ids:
        task_results = [r for r in results if r.task_id == tid]
        n = len(task_results)
        c = sum(1 for r in task_results if r.passed)
        scores: dict[str, float] = {"pass@1": pass_at_k(n, c, 1)}
        if n >= 3:
            scores["pass@3"] = pass_at_k(n, c, 3)
        pass_at_k_scores[tid] = scores

    # Averages
    tokens = [r.token_usage.total_tokens for r in results if r.token_usage]
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    times = [r.wall_clock_seconds for r in results]

    avg_tokens = sum(tokens) / len(tokens) if tokens else 0.0
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    avg_wall_clock = sum(times) / len(times) if times else 0.0

    # By category
    categories: dict[str, list[TaskResult]] = {}
    for r in results:
        cat = r.task_id.split("-")[0]  # e.g. "bugfix" from "bugfix-001-..."
        categories.setdefault(cat, []).append(r)

    by_category = {}
    for cat, cat_results in sorted(categories.items()):
        cat_passed = sum(1 for r in cat_results if r.passed)
        by_category[cat] = {
            "total": len(cat_results),
            "passed": cat_passed,
            "pass_rate": cat_passed / len(cat_results) if cat_results else 0.0,
        }

    # Global pass@k
    global_pass_at_k: dict[str, float] = {"pass@1": pass_at_k(total, passed, 1)}
    if total >= 3:
        global_pass_at_k["pass@3"] = pass_at_k(total, passed, 3)

    return {
        "total_tasks": total,
        "passed": passed,
        "pass_rate": round(pass_rate, 4),
        "pass_at_k": global_pass_at_k,
        "pass_at_k_by_task": pass_at_k_scores,
        "avg_tokens": round(avg_tokens, 1),
        "avg_cost_usd": round(avg_cost, 6),
        "avg_wall_clock_seconds": round(avg_wall_clock, 2),
        "by_category": by_category,
    }
