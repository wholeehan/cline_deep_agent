"""Tests for task state models and state machine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    InvalidTransitionError,
    SubTask,
    SubTaskStatus,
    TaskPlan,
)


class TestSubTask:
    """Tests for SubTask Pydantic model."""

    def test_valid_subtask(self) -> None:
        task = SubTask(
            id="1",
            title="Create Flask app",
            context="Use Flask 3.0",
            criterion="app.py exists and runs",
            complexity="simple",
            depends_on=[],
        )
        assert task.status == SubTaskStatus.PENDING
        assert task.escalate_flag is False

    def test_invalid_complexity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SubTask(id="1", title="test", complexity="impossible")

    def test_model_validate(self) -> None:
        data = {
            "id": "2",
            "title": "Add tests",
            "context": "",
            "criterion": "pytest passes",
            "complexity": "moderate",
            "depends_on": ["1"],
        }
        task = SubTask.model_validate(data)
        assert task.id == "2"
        assert task.depends_on == ["1"]

    def test_to_markdown(self) -> None:
        task = SubTask(id="1", title="Setup", criterion="works", complexity="trivial")
        md = task.to_markdown()
        assert "[ ]" in md
        assert "**Setup**" in md
        assert "[trivial]" in md


class TestSubTaskTransitions:
    """Tests for the state machine."""

    def test_pending_to_dispatched(self) -> None:
        task = SubTask(id="1", title="test")
        task.transition_to(SubTaskStatus.DISPATCHED)
        assert task.status == SubTaskStatus.DISPATCHED

    def test_dispatched_to_verified(self) -> None:
        task = SubTask(id="1", title="test", status=SubTaskStatus.DISPATCHED)
        task.transition_to(SubTaskStatus.VERIFIED)
        assert task.status == SubTaskStatus.VERIFIED

    def test_dispatched_to_failed(self) -> None:
        task = SubTask(id="1", title="test", status=SubTaskStatus.DISPATCHED)
        task.transition_to(SubTaskStatus.FAILED)
        assert task.status == SubTaskStatus.FAILED

    def test_failed_to_dispatched_retry(self) -> None:
        task = SubTask(id="1", title="test", status=SubTaskStatus.FAILED)
        task.transition_to(SubTaskStatus.DISPATCHED)
        assert task.status == SubTaskStatus.DISPATCHED

    def test_invalid_pending_to_verified(self) -> None:
        task = SubTask(id="1", title="test")
        with pytest.raises(InvalidTransitionError):
            task.transition_to(SubTaskStatus.VERIFIED)

    def test_invalid_pending_to_failed(self) -> None:
        task = SubTask(id="1", title="test")
        with pytest.raises(InvalidTransitionError):
            task.transition_to(SubTaskStatus.FAILED)

    def test_invalid_verified_to_anything(self) -> None:
        task = SubTask(id="1", title="test", status=SubTaskStatus.VERIFIED)
        with pytest.raises(InvalidTransitionError):
            task.transition_to(SubTaskStatus.DISPATCHED)

    def test_invalid_verified_to_failed(self) -> None:
        task = SubTask(id="1", title="test", status=SubTaskStatus.VERIFIED)
        with pytest.raises(InvalidTransitionError):
            task.transition_to(SubTaskStatus.FAILED)


class TestTaskPlan:
    """Tests for TaskPlan."""

    def _sample_plan(self) -> TaskPlan:
        return TaskPlan(subtasks=[
            SubTask(id="1", title="Setup", criterion="done", complexity="trivial"),
            SubTask(id="2", title="Build", criterion="done", depends_on=["1"]),
            SubTask(id="3", title="Test", criterion="done", depends_on=["2"]),
        ])

    def test_to_markdown(self) -> None:
        plan = self._sample_plan()
        md = plan.to_markdown()
        assert "# Task Plan" in md
        assert "Setup" in md
        assert "Build" in md

    def test_get_ready_tasks_initial(self) -> None:
        plan = self._sample_plan()
        ready = plan.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "1"

    def test_get_ready_tasks_after_first_verified(self) -> None:
        plan = self._sample_plan()
        plan.subtasks[0].status = SubTaskStatus.VERIFIED
        ready = plan.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "2"

    def test_get_ready_tasks_none_when_blocked(self) -> None:
        plan = self._sample_plan()
        plan.subtasks[0].status = SubTaskStatus.DISPATCHED
        ready = plan.get_ready_tasks()
        assert len(ready) == 0

    def test_get_task_by_id(self) -> None:
        plan = self._sample_plan()
        assert plan.get_task("2") is not None
        assert plan.get_task("2").title == "Build"  # type: ignore[union-attr]

    def test_get_task_missing(self) -> None:
        plan = self._sample_plan()
        assert plan.get_task("999") is None

    def test_parallel_tasks_both_ready(self) -> None:
        plan = TaskPlan(subtasks=[
            SubTask(id="1", title="A", criterion="done"),
            SubTask(id="2", title="B", criterion="done"),  # no deps
        ])
        ready = plan.get_ready_tasks()
        assert len(ready) == 2
