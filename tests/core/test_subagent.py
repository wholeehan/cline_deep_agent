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
from src.llm import (
    _active_ollama_models,
    _register_ollama_model,
    _unload_other_ollama_models,
    unload_ollama_model,
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

        # dispatch_subtask should NOT interrupt (auto-approved)
        assert ion["dispatch_subtask"] is False

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


class TestSubagentToolSupport:
    """Tests for subagent model tool-calling support validation."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://localhost:11434"})
    def test_rejects_model_without_tool_support(self) -> None:
        """A model that doesn't support tools should fail at subagent creation."""
        from src.llm import OllamaUnavailableError

        with patch("src.llm._check_ollama_connectivity") as mock_check:
            # Simulate tool check failure
            mock_check.side_effect = OllamaUnavailableError(
                "Model 'no-tools-model' does not support tool calling"
            )
            with pytest.raises(OllamaUnavailableError, match="does not support tool"):
                get_cline_executor_subagent()

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_SUBAGENT_MODEL": "tool-capable-model:latest",
    })
    def test_accepts_model_with_tool_support(self) -> None:
        """A model that supports tools should create the subagent successfully."""
        with patch("src.llm._check_ollama_connectivity"):
            defn = get_cline_executor_subagent()
            assert defn["name"] == "cline-executor"
            assert defn["model"].model == "tool-capable-model:latest"

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_SUBAGENT_MODEL": "qwen3-coder-next:latest",
    })
    def test_subagent_uses_env_var_model(self) -> None:
        """Subagent should use OLLAMA_SUBAGENT_MODEL env var."""
        with patch("src.llm._check_ollama_connectivity"):
            defn = get_cline_executor_subagent()
            assert defn["model"].model == "qwen3-coder-next:latest"


class TestOllamaVramSwap:
    """Tests for VRAM model swapping when running multiple Ollama models."""

    def setup_method(self) -> None:
        """Clear the active models registry before each test."""
        _active_ollama_models.clear()

    def test_register_ollama_model(self) -> None:
        """Registering a model adds it to the active set."""
        _register_ollama_model("model-a")
        assert "model-a" in _active_ollama_models

    def test_register_multiple_models(self) -> None:
        """Multiple models can be registered."""
        _register_ollama_model("model-a")
        _register_ollama_model("model-b")
        assert _active_ollama_models == {"model-a", "model-b"}

    def test_register_same_model_twice_is_idempotent(self) -> None:
        """Registering the same model twice doesn't duplicate it."""
        _register_ollama_model("model-a")
        _register_ollama_model("model-a")
        assert len(_active_ollama_models) == 1

    @patch("src.llm.unload_ollama_model")
    def test_unload_other_models_calls_unload(self, mock_unload: MagicMock) -> None:
        """Swapping to a model should unload all other registered models."""
        _register_ollama_model("gpt-oss:20b")
        _register_ollama_model("qwen3-coder-next:latest")

        _unload_other_ollama_models("qwen3-coder-next:latest")

        mock_unload.assert_called_once_with("gpt-oss:20b", None)

    @patch("src.llm.unload_ollama_model")
    def test_unload_does_not_unload_current_model(self, mock_unload: MagicMock) -> None:
        """The current model should NOT be unloaded."""
        _register_ollama_model("gpt-oss:20b")

        _unload_other_ollama_models("gpt-oss:20b")

        mock_unload.assert_not_called()

    @patch("src.llm.unload_ollama_model")
    def test_unload_with_three_models(self, mock_unload: MagicMock) -> None:
        """All models except the current one should be unloaded."""
        _register_ollama_model("model-a")
        _register_ollama_model("model-b")
        _register_ollama_model("model-c")

        _unload_other_ollama_models("model-b")

        unloaded = {call.args[0] for call in mock_unload.call_args_list}
        assert unloaded == {"model-a", "model-c"}

    @patch("httpx.post")
    def test_unload_ollama_model_sends_keep_alive_zero(self, mock_post: MagicMock) -> None:
        """unload_ollama_model should POST to Ollama with keep_alive=0."""
        mock_post.return_value = MagicMock(status_code=200)

        unload_ollama_model("gpt-oss:20b", "http://localhost:11434")

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={"model": "gpt-oss:20b", "keep_alive": 0},
            timeout=10.0,
        )

    @patch("httpx.post", side_effect=Exception("connection refused"))
    def test_unload_ollama_model_handles_connection_error(self, mock_post: MagicMock) -> None:
        """unload_ollama_model should not raise on connection errors."""
        # Should not raise
        unload_ollama_model("gpt-oss:20b", "http://localhost:11434")
