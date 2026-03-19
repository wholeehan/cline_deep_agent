"""Pydantic models for task state management."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class SubTaskStatus(enum.StrEnum):
    """Valid states for a subtask."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    VERIFIED = "verified"
    FAILED = "failed"


# Valid state transitions
_VALID_TRANSITIONS: dict[SubTaskStatus, set[SubTaskStatus]] = {
    SubTaskStatus.PENDING: {SubTaskStatus.DISPATCHED},
    SubTaskStatus.DISPATCHED: {SubTaskStatus.VERIFIED, SubTaskStatus.FAILED},
    SubTaskStatus.VERIFIED: set(),  # terminal
    SubTaskStatus.FAILED: {SubTaskStatus.DISPATCHED},  # can retry
}


class InvalidTransitionError(ValueError):
    """Raised when an invalid status transition is attempted."""


class SubTask(BaseModel):
    """A subtask produced by task decomposition."""

    id: str
    title: str
    context: str = ""
    criterion: str = Field(default="", description="Acceptance criterion")
    complexity: str = Field(default="simple", pattern=r"^(trivial|simple|moderate|complex)$")
    depends_on: list[str] = Field(default_factory=list)
    status: SubTaskStatus = SubTaskStatus.PENDING
    escalate_flag: bool = False

    def transition_to(self, new_status: SubTaskStatus) -> None:
        """Transition to a new status, enforcing the state machine."""
        valid = _VALID_TRANSITIONS.get(self.status, set())
        if new_status not in valid:
            raise InvalidTransitionError(
                f"Cannot transition from {self.status.value} to {new_status.value}. "
                f"Valid transitions: {[s.value for s in valid]}"
            )
        self.status = new_status

    def to_markdown(self) -> str:
        """Serialize to a markdown list item."""
        status_icon = {
            SubTaskStatus.PENDING: "[ ]",
            SubTaskStatus.DISPATCHED: "[~]",
            SubTaskStatus.VERIFIED: "[x]",
            SubTaskStatus.FAILED: "[!]",
        }
        icon = status_icon.get(self.status, "[ ]")
        deps = f" (depends: {', '.join(self.depends_on)})" if self.depends_on else ""
        return f"- {icon} **{self.title}** [{self.complexity}]{deps}\n  Criterion: {self.criterion}"


class TaskPlan(BaseModel):
    """A plan consisting of ordered subtasks."""

    subtasks: list[SubTask] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Serialize the full plan to todos.md format."""
        lines = ["# Task Plan\n"]
        for task in self.subtasks:
            lines.append(task.to_markdown())
        return "\n".join(lines)

    def get_ready_tasks(self) -> list[SubTask]:
        """Return tasks that are pending and have all dependencies met."""
        verified_ids = {t.id for t in self.subtasks if t.status == SubTaskStatus.VERIFIED}
        ready: list[SubTask] = []
        for task in self.subtasks:
            if task.status != SubTaskStatus.PENDING:
                continue
            if all(dep in verified_ids for dep in task.depends_on):
                ready.append(task)
        return ready

    def get_task(self, task_id: str) -> SubTask | None:
        """Look up a subtask by ID."""
        for task in self.subtasks:
            if task.id == task_id:
                return task
        return None
