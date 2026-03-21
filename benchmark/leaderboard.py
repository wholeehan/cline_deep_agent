"""Generate comparison leaderboard from benchmark result files."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.config import BenchmarkReport
from benchmark.scorer import compute_summary


def load_reports(results_dir: Path) -> list[BenchmarkReport]:
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
