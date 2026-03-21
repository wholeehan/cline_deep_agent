"""Benchmark runner — orchestrates task execution."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from benchmark.adapters.base import AgentAdapter
from benchmark.adapters.cline_deep import ClineDeepAdapter
from benchmark.config import BenchmarkReport, RunConfig, TaskConfig, TaskResult

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_ADAPTERS: dict[str, type[AgentAdapter]] = {
    "cline-deep": ClineDeepAdapter,
}


def _parse_task_toml(path: Path) -> TaskConfig:
    """Parse a task.toml file into a TaskConfig."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return TaskConfig(**data["task"])


class BenchmarkRunner:
    """Discovers and runs benchmark tasks."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        adapter_cls = _ADAPTERS.get(config.agent)
        if adapter_cls is None:
            raise ValueError(
                f"Unknown agent adapter '{config.agent}'. "
                f"Available: {', '.join(_ADAPTERS)}"
            )
        self.adapter = adapter_cls(config)

    def run_all(self) -> BenchmarkReport:
        """Discover tasks, run each with repetitions, compile report."""
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        tasks = self._discover_tasks()

        if not tasks:
            logger.warning("No benchmark tasks found in %s", TASKS_DIR)

        self.adapter.setup()
        results: list[TaskResult] = []

        try:
            for task_config, task_dir in tasks:
                for rep in range(1, self.config.repetitions + 1):
                    logger.info(
                        "Running task %s (rep %d/%d)",
                        task_config.id,
                        rep,
                        self.config.repetitions,
                    )
                    result = self._run_single(task_config, task_dir, rep)
                    results.append(result)
        finally:
            self.adapter.teardown()

        report = BenchmarkReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=self.config.agent,
            provider=self.config.provider,
            model=self.config.model,
            tasks=results,
        )

        return report

    def _run_single(self, task_config: TaskConfig, task_dir: Path, rep: int) -> TaskResult:
        """Run a single task in an isolated tmp directory."""
        workspace_src = task_dir / "workspace"
        tests_src = task_dir / "tests"
        setup_script = task_dir / "setup.sh"
        instruction_file = task_dir / "instruction.md"

        instruction = instruction_file.read_text(encoding="utf-8") if instruction_file.exists() else ""

        with tempfile.TemporaryDirectory(prefix=f"bench-{task_config.id}-") as tmp:
            tmp_path = Path(tmp)

            # 1. Copy workspace to tmp dir
            if workspace_src.exists():
                shutil.copytree(workspace_src, tmp_path / "workspace")
                work_dir = tmp_path / "workspace"
            else:
                work_dir = tmp_path

            # 2. Run setup.sh if present
            if setup_script.exists():
                self._run_setup(setup_script, work_dir)

            # 3. Run adapter
            run_result = self.adapter.run(
                instruction=instruction,
                workspace=work_dir,
                timeout=task_config.timeout_seconds,
            )

            # 4. Copy tests into tmp dir and run pytest
            passed = False
            test_output = ""
            if tests_src.exists():
                test_dest = work_dir / "tests"
                if test_dest.exists():
                    shutil.rmtree(test_dest)
                shutil.copytree(tests_src, test_dest)
                passed, test_output = self._run_tests(work_dir)
            else:
                test_output = "No tests found for this task."

        return TaskResult(
            task_id=task_config.id,
            repetition=rep,
            passed=passed,
            wall_clock_seconds=run_result.wall_clock_seconds,
            token_usage=run_result.token_usage,
            cost_usd=run_result.cost_usd,
            files_changed=run_result.files_changed,
            test_output=test_output,
            error=run_result.error,
        )

    def _discover_tasks(self) -> list[tuple[TaskConfig, Path]]:
        """Scan benchmark/tasks/*, parse task.toml, skip _template."""
        tasks: list[tuple[TaskConfig, Path]] = []

        if not TASKS_DIR.exists():
            return tasks

        for task_dir in sorted(TASKS_DIR.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue

            toml_path = task_dir / "task.toml"
            if not toml_path.exists():
                logger.warning("Skipping %s — no task.toml found", task_dir.name)
                continue

            try:
                config = _parse_task_toml(toml_path)
            except Exception:
                logger.exception("Failed to parse %s", toml_path)
                continue

            # Filter by requested tasks (support prefix matching, e.g. "bugfix-001")
            if self.config.tasks != "all":
                if not any(
                    config.id == t or config.id.startswith(t)
                    for t in self.config.tasks
                ):
                    continue

            tasks.append((config, task_dir))

        return tasks

    @staticmethod
    def _run_setup(script: Path, work_dir: Path) -> None:
        """Execute setup.sh in the work directory."""
        try:
            subprocess.run(
                ["bash", str(script)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("setup.sh timed out for %s", work_dir)

    @staticmethod
    def _run_tests(work_dir: Path) -> tuple[bool, str]:
        """Run pytest in the work directory and return (passed, output)."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout + "\n" + result.stderr
            return result.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            return False, "Tests timed out."
        except Exception as exc:
            return False, f"Test execution error: {exc}"


def save_report(report: BenchmarkReport) -> Path:
    """Save a benchmark report as JSON to the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{report.run_id}.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Report saved to %s", out_path)
    return out_path
