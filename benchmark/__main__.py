"""CLI entry point: python -m benchmark run|score|list|leaderboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.config import BenchmarkReport, RunConfig
from benchmark.leaderboard import generate_markdown, load_reports
from benchmark.runner import TASKS_DIR, BenchmarkRunner, _parse_task_toml, save_report
from benchmark.scorer import compute_summary


def cmd_list(args: argparse.Namespace) -> None:
    """List available benchmark tasks."""
    if not TASKS_DIR.exists():
        print("No tasks directory found.")
        return

    tasks = []
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        toml_path = task_dir / "task.toml"
        if not toml_path.exists():
            continue
        try:
            config = _parse_task_toml(toml_path)
            tasks.append(config)
        except Exception as exc:
            print(f"Warning: could not parse {toml_path}: {exc}")

    if not tasks:
        print("No tasks found.")
        return

    print(f"{'ID':<35} {'Category':<12} {'Difficulty':<10} Title")
    print("-" * 90)
    for t in tasks:
        print(f"{t.id:<35} {t.category:<12} {t.difficulty:<10} {t.title}")


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmark tasks."""
    tasks_list: list[str] | str = "all"
    if args.tasks and args.tasks != "all":
        tasks_list = [t.strip() for t in args.tasks.split(",")]

    config = RunConfig(
        agent=args.agent,
        provider=args.provider,
        model=args.model,
        tasks=tasks_list,
        repetitions=args.repetitions,
    )

    runner = BenchmarkRunner(config)
    report = runner.run_all()
    report.summary = compute_summary(report)

    out_path = save_report(report)
    print(f"\nResults saved to {out_path}")

    # Print summary
    s = report.summary
    print(f"\nPass rate: {s['passed']}/{s['total_tasks']} ({s['pass_rate'] * 100:.0f}%)")
    if s.get("avg_tokens"):
        print(f"Avg tokens: {s['avg_tokens']:,.0f}")
    if s.get("avg_cost_usd"):
        print(f"Avg cost: ${s['avg_cost_usd']:.4f}")
    print(f"Avg wall clock: {s['avg_wall_clock_seconds']:.1f}s")


def cmd_score(args: argparse.Namespace) -> None:
    """Recompute metrics from existing result files."""
    for path_str in args.files:
        path = Path(path_str)
        if not path.exists():
            print(f"File not found: {path}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            report = BenchmarkReport(**data)
            report.summary = compute_summary(report)
            path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            s = report.summary
            print(f"{path.name}: {s['passed']}/{s['total_tasks']} passed ({s['pass_rate'] * 100:.0f}%)")
            if s.get("avg_tokens"):
                print(f"  Avg tokens: {s['avg_tokens']:,.0f}")
            if s.get("avg_cost_usd"):
                print(f"  Avg cost: ${s['avg_cost_usd']:.4f}")
        except Exception as exc:
            print(f"Error processing {path}: {exc}")


def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Generate comparison table from result files."""
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return

    reports = load_reports(results_dir)
    if not reports:
        print("No result files found.")
        return

    # Ensure summaries are computed
    for report in reports:
        if not report.summary:
            report.summary = compute_summary(report)

    md = generate_markdown(reports)
    print(md)


def main() -> None:
    """Parse arguments and dispatch to subcommand."""
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Benchmark suite for Cline Deep Agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    subparsers.add_parser("list", help="List available benchmark tasks")

    # run
    run_parser = subparsers.add_parser("run", help="Run benchmark tasks")
    run_parser.add_argument("--agent", default="cline-deep", help="Agent adapter to use")
    run_parser.add_argument("--provider", default="ollama", help="LLM provider (ollama, anthropic, vllm)")
    run_parser.add_argument("--model", default="gpt-oss:20b", help="Model name for the provider")
    run_parser.add_argument("--tasks", default="all", help="Comma-separated task IDs or 'all'")
    run_parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per task")

    # score
    score_parser = subparsers.add_parser("score", help="Recompute metrics from result files")
    score_parser.add_argument("files", nargs="+", help="Result JSON files to score")

    # leaderboard
    lb_parser = subparsers.add_parser("leaderboard", help="Generate comparison table")
    lb_parser.add_argument("results_dir", help="Directory containing result JSON files")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "run": cmd_run,
        "score": cmd_score,
        "leaderboard": cmd_leaderboard,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
