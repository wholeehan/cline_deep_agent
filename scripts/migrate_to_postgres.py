#!/usr/bin/env python3
"""One-time migration: backfill existing JSON/JSONL files into PostgreSQL.

Usage:
    DATABASE_URL=postgresql://cline:cline@localhost:5432/cline_telemetry \
        python -m scripts.migrate_to_postgres

Idempotent — safe to run multiple times (duplicates are skipped).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def migrate_benchmark_results(pool) -> int:  # type: ignore[type-arg]
    """Import benchmark/results/*.json into benchmark_runs + benchmark_results."""
    results_dir = PROJECT_ROOT / "benchmark" / "results"
    if not results_dir.exists():
        print("No benchmark/results/ directory found — skipping.")
        return 0

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print("No JSON result files found — skipping benchmark migration.")
        return 0

    imported = 0
    with pool.connection() as conn:
        for path in json_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                run_id = data["run_id"]

                # Check if already imported
                exists = conn.execute(
                    "SELECT 1 FROM benchmark_runs WHERE run_id = %s", (run_id,)
                ).fetchone()
                if exists:
                    print(f"  Skipping {path.name} (already imported)")
                    continue

                conn.execute(
                    """INSERT INTO benchmark_runs (run_id, timestamp, agent, provider, model, summary)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        run_id,
                        data["timestamp"],
                        data["agent"],
                        data["provider"],
                        data["model"],
                        json.dumps(data.get("summary", {})),
                    ),
                )

                for task in data.get("tasks", []):
                    token_usage = task.get("token_usage") or {}
                    conn.execute(
                        """INSERT INTO benchmark_results
                           (run_id, task_id, repetition, passed, wall_clock_seconds,
                            prompt_tokens, completion_tokens, total_tokens,
                            cost_usd, files_changed, test_output, error)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            run_id,
                            task["task_id"],
                            task["repetition"],
                            task["passed"],
                            task.get("wall_clock_seconds", 0),
                            token_usage.get("prompt_tokens"),
                            token_usage.get("completion_tokens"),
                            token_usage.get("total_tokens"),
                            task.get("cost_usd"),
                            task.get("files_changed", []),
                            task.get("test_output", ""),
                            task.get("error"),
                        ),
                    )

                conn.commit()
                imported += 1
                print(f"  Imported {path.name} ({len(data.get('tasks', []))} tasks)")

            except Exception as exc:
                conn.rollback()
                print(f"  Error importing {path.name}: {exc}")

    return imported


def migrate_agent_events(pool) -> int:  # type: ignore[type-arg]
    """Import data/logs/agent.jsonl into agent_events table."""
    jsonl_path = PROJECT_ROOT / "data" / "logs" / "agent.jsonl"
    if not jsonl_path.exists():
        print("No data/logs/agent.jsonl found — skipping.")
        return 0

    imported = 0
    with pool.connection() as conn:
        with open(jsonl_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    conn.execute(
                        """INSERT INTO agent_events
                           (level, logger, message, llm_provider,
                            event_type, subtask_id, tool_name, decision, status,
                            exception, traceback)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            entry.get("level", "INFO"),
                            entry.get("logger"),
                            entry.get("message", ""),
                            entry.get("llm_provider"),
                            entry.get("event_type"),
                            entry.get("subtask_id"),
                            entry.get("tool_name"),
                            entry.get("decision"),
                            entry.get("status"),
                            entry.get("exception"),
                            entry.get("traceback"),
                        ),
                    )
                    imported += 1
                except Exception as exc:
                    print(f"  Warning: skipping line {line_num}: {exc}")

        conn.commit()

    return imported


def main() -> None:
    """Run all migrations."""
    from src.db import get_connection_pool

    pool = get_connection_pool()
    if pool is None:
        print("ERROR: DATABASE_URL not set or PostgreSQL connection failed.")
        print("Set DATABASE_URL and ensure PostgreSQL is running.")
        sys.exit(1)

    print("=== Migrating benchmark results ===")
    bench_count = migrate_benchmark_results(pool)
    print(f"  Total imported: {bench_count} reports\n")

    print("=== Migrating agent event logs ===")
    event_count = migrate_agent_events(pool)
    print(f"  Total imported: {event_count} events\n")

    print("Migration complete.")


if __name__ == "__main__":
    main()
