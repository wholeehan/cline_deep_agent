"""Tests for Cline executor subagent definition and tools."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.subagent import (
    answer_question,
    approve_cline_action,
    dispatch_subtask,
    get_cline_executor_subagent,
    set_bridge,
)


class TestClineExecutorSubagentDefinition:
    """Tests for the subagent definition dict."""

    def test_has_required_fields(self) -> None:
        defn = get_cline_executor_subagent()
        assert defn["name"] == "cline-executor"
        assert "description" in defn
        assert "system_prompt" in defn
        assert "tools" in defn
        assert len(defn["tools"]) == 3

    def test_has_skills(self) -> None:
        defn = get_cline_executor_subagent()
        assert "/skills/cline_qa/" in defn["skills"]

    def test_interrupt_on_config(self) -> None:
        defn = get_cline_executor_subagent()
        ion = defn["interrupt_on"]

        # approve_cline_action should interrupt with approve/edit/reject
        assert ion["approve_cline_action"]["allowed_decisions"] == ["approve", "edit", "reject"]

        # dispatch_subtask should interrupt with approve/reject
        assert ion["dispatch_subtask"]["allowed_decisions"] == ["approve", "reject"]

        # answer_question should NOT interrupt
        assert ion["answer_question"] is False

    @patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
    def test_subagent_integrates_with_create_agent(self) -> None:
        from src.agent import create_agent_manager
        defn = get_cline_executor_subagent()
        agent = create_agent_manager(subagents=[defn])
        assert agent is not None


class TestDispatchSubtask:
    """Tests for the dispatch_subtask tool."""

    def test_sends_task_to_bridge(self) -> None:
        mock_bridge = MagicMock()
        mock_bridge.is_alive.return_value = True
        mock_bridge.send_task.return_value = "hello world output"
        set_bridge(mock_bridge)

        result = dispatch_subtask.invoke({"subtask": "echo hello"})
        mock_bridge.send_task.assert_called_once_with("echo hello")
        assert "hello world output" in result

    def test_spawns_bridge_if_not_alive(self) -> None:
        mock_bridge = MagicMock()
        mock_bridge.is_alive.return_value = False
        mock_bridge.send_task.return_value = "output"
        set_bridge(mock_bridge)

        dispatch_subtask.invoke({"subtask": "test"})
        mock_bridge.spawn.assert_called_once()

    def test_returns_fallback_on_empty_output(self) -> None:
        mock_bridge = MagicMock()
        mock_bridge.is_alive.return_value = True
        mock_bridge.send_task.return_value = ""
        set_bridge(mock_bridge)

        result = dispatch_subtask.invoke({"subtask": "test"})
        assert "no output" in result


class TestAnswerQuestion:
    """Tests for the answer_question tool."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
    def test_injects_answer_to_bridge(self) -> None:
        mock_bridge = MagicMock()
        mock_bridge.is_alive.return_value = True
        set_bridge(mock_bridge)

        mock_llm_response = MagicMock()
        mock_llm_response.content = "pytest"

        with patch("src.subagent.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = mock_llm_response
            result = answer_question.invoke({
                "question": "What testing framework?",
                "context": "Project uses pytest",
            })

        assert "pytest" in result
        mock_bridge.inject.assert_called_once_with("pytest")


class TestApproveClineAction:
    """Tests for the approve_cline_action tool."""

    def test_auto_approves_file_write(self) -> None:
        mock_bridge = MagicMock()
        mock_bridge.is_alive.return_value = True
        set_bridge(mock_bridge)

        result = approve_cline_action.invoke({
            "action": "write src/main.py",
            "action_type": "file_write",
        })
        assert "auto-approved" in result
        mock_bridge.inject.assert_called_once_with("yes")

    def test_auto_approves_file_read(self) -> None:
        mock_bridge = MagicMock()
        mock_bridge.is_alive.return_value = True
        set_bridge(mock_bridge)

        result = approve_cline_action.invoke({
            "action": "read config.json",
            "action_type": "file_read",
        })
        assert "auto-approved" in result

    def test_escalates_http_request(self) -> None:
        mock_bridge = MagicMock()
        set_bridge(mock_bridge)

        result = approve_cline_action.invoke({
            "action": "curl https://api.example.com",
            "action_type": "http_request",
        })
        assert "escalate" in result

    def test_escalates_delete(self) -> None:
        mock_bridge = MagicMock()
        set_bridge(mock_bridge)

        result = approve_cline_action.invoke({
            "action": "rm -rf /tmp/project",
            "action_type": "delete",
        })
        assert "escalate" in result

    @pytest.mark.parametrize("action_type", [
        "file_read", "file_write", "file_create", "file_edit",
        "local_command", "git_local", "test_run",
    ])
    def test_all_safe_types_auto_approved(self, action_type: str) -> None:
        mock_bridge = MagicMock()
        mock_bridge.is_alive.return_value = True
        set_bridge(mock_bridge)

        result = approve_cline_action.invoke({
            "action": "test action",
            "action_type": action_type,
        })
        assert "auto-approved" in result

    @pytest.mark.parametrize("action_type", [
        "http_request", "delete", "git_remote", "system_modify",
        "database_write", "package_publish", "unknown",
    ])
    def test_all_unsafe_types_escalated(self, action_type: str) -> None:
        mock_bridge = MagicMock()
        set_bridge(mock_bridge)

        result = approve_cline_action.invoke({
            "action": "test action",
            "action_type": action_type,
        })
        assert "escalate" in result
