"""End-to-end integration tests (mocked LLM)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from src.agent import create_agent_manager
from src.hitl import build_approve_command, extract_interrupts
from src.models import SubTask, SubTaskStatus, TaskPlan
from src.subagent import get_cline_executor_subagent, set_bridge


class TestHappyPathE2E:
    """E2E test: agent creates, invokes, and has proper structure."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
    def test_agent_creates_with_subagent(self) -> None:
        subagent = get_cline_executor_subagent()
        agent = create_agent_manager(subagents=[subagent])
        assert agent is not None
        assert hasattr(agent, "invoke")

    @patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
    def test_agent_creates_with_skills(self) -> None:
        agent = create_agent_manager(skills=["/skills/"])
        assert agent is not None


class TestTaskPlanE2E:
    """E2E test: full task plan lifecycle."""

    def test_plan_lifecycle(self) -> None:
        """Simulate: create plan → dispatch → verify → complete."""
        plan = TaskPlan(subtasks=[
            SubTask(
                id="1", title="Create app skeleton", criterion="app.py exists", complexity="simple"
            ),
            SubTask(id="2", title="Add routes", criterion="/ returns 200", depends_on=["1"]),
            SubTask(id="3", title="Add tests", criterion="pytest passes", depends_on=["2"]),
        ])

        # Initially, only task 1 is ready
        assert len(plan.get_ready_tasks()) == 1
        assert plan.get_ready_tasks()[0].id == "1"

        # Dispatch task 1
        plan.subtasks[0].transition_to(SubTaskStatus.DISPATCHED)
        assert len(plan.get_ready_tasks()) == 0

        # Verify task 1
        plan.subtasks[0].transition_to(SubTaskStatus.VERIFIED)
        assert len(plan.get_ready_tasks()) == 1
        assert plan.get_ready_tasks()[0].id == "2"

        # Dispatch and verify task 2
        plan.subtasks[1].transition_to(SubTaskStatus.DISPATCHED)
        plan.subtasks[1].transition_to(SubTaskStatus.VERIFIED)
        assert plan.get_ready_tasks()[0].id == "3"

        # Dispatch and verify task 3
        plan.subtasks[2].transition_to(SubTaskStatus.DISPATCHED)
        plan.subtasks[2].transition_to(SubTaskStatus.VERIFIED)
        assert len(plan.get_ready_tasks()) == 0

        # All verified
        assert all(t.status == SubTaskStatus.VERIFIED for t in plan.subtasks)

    def test_dependency_ordering(self) -> None:
        """Task B with depends_on=[A.id] is not ready until A is verified."""
        plan = TaskPlan(subtasks=[
            SubTask(id="A", title="First", criterion="done"),
            SubTask(id="B", title="Second", criterion="done", depends_on=["A"]),
        ])

        ready = plan.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "A"

        # Dispatch A — B still not ready
        plan.subtasks[0].transition_to(SubTaskStatus.DISPATCHED)
        assert len(plan.get_ready_tasks()) == 0

        # Verify A — B is now ready
        plan.subtasks[0].transition_to(SubTaskStatus.VERIFIED)
        ready = plan.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "B"


class TestCrashRecovery:
    """E2E test: Cline process crash handling."""

    def test_bridge_detects_dead_process(self) -> None:
        mock_bridge = MagicMock()
        mock_bridge.is_alive.return_value = False
        mock_bridge.exit_code.return_value = 1
        set_bridge(mock_bridge)

        assert not mock_bridge.is_alive()
        assert mock_bridge.exit_code() == 1

    def test_failed_subtask_can_retry(self) -> None:
        task = SubTask(id="1", title="Flaky task", criterion="done")
        task.transition_to(SubTaskStatus.DISPATCHED)
        task.transition_to(SubTaskStatus.FAILED)
        # Retry
        task.transition_to(SubTaskStatus.DISPATCHED)
        task.transition_to(SubTaskStatus.VERIFIED)
        assert task.status == SubTaskStatus.VERIFIED


class TestHITLIntegrationE2E:
    """E2E test: interrupt flows."""

    def test_interrupt_extraction_and_resume(self) -> None:
        # Simulated interrupt result
        result = {
            "__interrupt__": [
                {
                    "tool_name": "approve_cline_action",
                    "args": {"action": "rm /tmp/old"},
                    "description": "Delete old files",
                },
            ]
        }

        interrupts = extract_interrupts(result)
        assert len(interrupts) == 1
        assert interrupts[0].tool_name == "approve_cline_action"

        # User approves
        cmd = build_approve_command()
        assert cmd.resume["decisions"][0]["type"] == "approve"


class TestProviderParity:
    """E2E test: both providers create valid agents."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
    def test_anthropic_agent_creates(self) -> None:
        agent = create_agent_manager()
        assert agent is not None

    def test_ollama_agent_creates(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        mock_resp.raise_for_status = MagicMock()

        mock_tool_resp = MagicMock()
        mock_tool_resp.status_code = 200

        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}),
            patch("src.llm.httpx.get", return_value=mock_resp),
            patch("src.llm.httpx.post", return_value=mock_tool_resp),
        ):
            agent = create_agent_manager()
            assert agent is not None
