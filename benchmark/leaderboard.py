"""Generate comparison leaderboard from benchmark result files."""

from __future__ import annotations

import json
import os
from pathlib import Path

from benchmark.config import BenchmarkReport, TaskResult, TokenUsage
from benchmark.scorer import compute_summary


def load_reports(results_dir: Path) -> list[BenchmarkReport]:
    """Load benchmark reports from PostgreSQL if available, otherwise from JSON files."""
    if os.getenv("DATABASE_URL"):
        reports = _load_reports_from_postgres()
        if reports is not None:
            return reports

    return _load_reports_from_files(results_dir)


def _load_reports_from_files(results_dir: Path) -> list[BenchmarkReport]:
    """Load all JSON report files from a directory."""
    reports: list[BenchmarkReport] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            report = BenchmarkReport(**data)
            if not report.summary:
                report.summary = compute_summary(report)
            reports.append(report)
        except Exception as exc:
            print(f"Warning: skipping {path.name}: {exc}")
    return reports


def _load_reports_from_postgres() -> list[BenchmarkReport] | None:
    """Load benchmark reports from PostgreSQL. Returns None on failure."""
    try:
        from src.db import get_connection_pool

        pool = get_connection_pool()
        if pool is None:
            return None

        reports: list[BenchmarkReport] = []
        with pool.connection() as conn:
            runs = conn.execute(
                "SELECT run_id, timestamp, agent, provider, model, summary "
                "FROM benchmark_runs ORDER BY timestamp"
            ).fetchall()

            for run_id, ts, agent, provider, model, summary in runs:
                results = conn.execute(
                    """SELECT task_id, repetition, passed, wall_clock_seconds,
                              prompt_tokens, completion_tokens, total_tokens,
                              cost_usd, files_changed, test_output, error
                       FROM benchmark_results WHERE run_id = %s""",
                    (run_id,),
                ).fetchall()

                tasks = []
                for row in results:
                    token_usage = None
                    if row[4] is not None:
                        token_usage = TokenUsage(
                            prompt_tokens=row[4],
                            completion_tokens=row[5] or 0,
                            total_tokens=row[6] or 0,
                        )
                    tasks.append(
                        TaskResult(
                            task_id=row[0],
                            repetition=row[1],
                            passed=row[2],
                            wall_clock_seconds=row[3],
                            token_usage=token_usage,
                            cost_usd=row[7],
                            files_changed=row[8] or [],
                            test_output=row[9] or "",
                            error=row[10],
                        )
                    )

                report = BenchmarkReport(
                    run_id=run_id,
                    timestamp=ts if isinstance(ts, str) else ts.isoformat(),
                    agent=agent,
                    provider=provider,
                    model=model,
                    tasks=tasks,
                    summary=summary if isinstance(summary, dict) else json.loads(summary or "{}"),
                )
                if not report.summary:
                    report.summary = compute_summary(report)
                reports.append(report)

        return reports
    except Exception as exc:
        print(f"Warning: PostgreSQL load failed, falling back to files: {exc}")
        return None


def generate_markdown(reports: list[BenchmarkReport]) -> str:
    """Generate a markdown comparison table from reports."""
    if not reports:
        return "No results found."

    lines = [
        "# Benchmark Leaderboard",
        "",
        "| Agent | Model | Pass Rate | Pass@1 | Pass@3 | Avg Tokens | Avg Cost |",
        "|-------|-------|-----------|--------|--------|------------|----------|",
    ]

    for report in reports:
        summary = report.summary
        pass_rate = f"{summary.get('pass_rate', 0) * 100:.0f}%"
        pak = summary.get("pass_at_k", {})
        pass_1 = f"{pak.get('pass@1', 0) * 100:.0f}%" if "pass@1" in pak else "—"
        pass_3 = f"{pak.get('pass@3', 0) * 100:.0f}%" if "pass@3" in pak else "—"
        avg_tokens = f"{summary.get('avg_tokens', 0):,.0f}"
        avg_cost = f"${summary.get('avg_cost_usd', 0):.4f}" if summary.get("avg_cost_usd") else "—"

        lines.append(
            f"| {report.agent} | {report.model} | {pass_rate} | {pass_1} "
            f"| {pass_3} | {avg_tokens} | {avg_cost} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_json(reports: list[BenchmarkReport]) -> str:
    """Generate a JSON summary from reports."""
    entries = []
    for report in reports:
        entries.append({
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "agent": report.agent,
            "model": report.model,
            "summary": report.summary,
        })
    return json.dumps(entries, indent=2)
