"""Tests for benchmark runner — task discovery and config parsing."""

from pathlib import Path

from benchmark.config import RunConfig
from benchmark.runner import TASKS_DIR, BenchmarkRunner, _parse_task_toml


class TestParseTaskToml:
    def test_parse_bugfix_001(self):
        toml_path = TASKS_DIR / "bugfix-001-off-by-one" / "task.toml"
        cfg = _parse_task_toml(toml_path)
        assert cfg.id == "bugfix-001-off-by-one"
        assert cfg.category == "bugfix"
        assert cfg.difficulty == "easy"
        assert cfg.language == "python"

    def test_parse_all_tasks(self):
        for task_dir in TASKS_DIR.iterdir():
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            toml_path = task_dir / "task.toml"
            if toml_path.exists():
                cfg = _parse_task_toml(toml_path)
                assert cfg.id
                assert cfg.title


class TestDiscoverTasks:
    def test_discover_all(self):
        config = RunConfig(agent="cline-deep", tasks="all")
        runner = BenchmarkRunner(config)
        tasks = runner._discover_tasks()
        assert len(tasks) == 5
        ids = [t[0].id for t in tasks]
        assert "bugfix-001-off-by-one" in ids
        assert "feature-001-add-endpoint" in ids

    def test_discover_by_exact_id(self):
        config = RunConfig(agent="cline-deep", tasks=["bugfix-001-off-by-one"])
        runner = BenchmarkRunner(config)
        tasks = runner._discover_tasks()
        assert len(tasks) == 1
        assert tasks[0][0].id == "bugfix-001-off-by-one"

    def test_discover_by_prefix(self):
        config = RunConfig(agent="cline-deep", tasks=["bugfix"])
        runner = BenchmarkRunner(config)
        tasks = runner._discover_tasks()
        assert len(tasks) == 2
        ids = [t[0].id for t in tasks]
        assert "bugfix-001-off-by-one" in ids
        assert "bugfix-002-null-check" in ids

    def test_discover_no_match(self):
        config = RunConfig(agent="cline-deep", tasks=["nonexistent"])
        runner = BenchmarkRunner(config)
        tasks = runner._discover_tasks()
        assert len(tasks) == 0

    def test_skips_template(self):
        config = RunConfig(agent="cline-deep", tasks="all")
        runner = BenchmarkRunner(config)
        tasks = runner._discover_tasks()
        ids = [t[0].id for t in tasks]
        assert not any("template" in i for i in ids)


class TestTaskStructure:
    """Verify each task directory has the required files."""

    def test_all_tasks_have_required_files(self):
        required = ["task.toml", "instruction.md", "setup.sh"]
        for task_dir in TASKS_DIR.iterdir():
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            for filename in required:
                assert (task_dir / filename).exists(), (
                    f"{task_dir.name} missing {filename}"
                )

    def test_all_tasks_have_workspace(self):
        for task_dir in TASKS_DIR.iterdir():
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            workspace = task_dir / "workspace"
            assert workspace.exists(), f"{task_dir.name} missing workspace/"
            assert any(workspace.iterdir()), f"{task_dir.name} has empty workspace/"

    def test_all_tasks_have_tests(self):
        for task_dir in TASKS_DIR.iterdir():
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            tests = task_dir / "tests"
            assert tests.exists(), f"{task_dir.name} missing tests/"
            test_files = list(tests.glob("test_*.py"))
            assert test_files, f"{task_dir.name} has no test_*.py files"
