"""Tests for DeepAgents core agent setup."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from src.agent import create_agent_manager, create_composite_backend, report_progress


class TestReportProgress:
    """Tests for the report_progress tool."""

    def test_returns_formatted_message(self) -> None:
        result = report_progress.invoke({"message": "Step 1 done"})
        assert "[progress]" in result
        assert "Step 1 done" in result


class TestCompositeBackend:
    """Tests for composite backend factory."""

    def test_creates_composite_with_routes(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.store = MagicMock()
        mock_runtime.config = {"configurable": {"thread_id": "test"}}

        with (
            patch("src.agent.StateBackend") as mock_state,
            patch("src.agent.StoreBackend") as mock_store,
            patch("src.agent.FilesystemBackend") as mock_fs,
            patch("src.agent.CompositeBackend") as mock_composite,
        ):
            create_composite_backend(mock_runtime)

            mock_state.assert_called_once_with(mock_runtime)
            mock_store.assert_called_once_with(mock_runtime)
            mock_fs.assert_called_once_with(virtual_mode=True)
            mock_composite.assert_called_once()


class TestCreateAgentManager:
    """Tests for agent manager factory."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
    def test_creates_agent_without_error(self) -> None:
        agent = create_agent_manager()
        assert agent is not None

    @patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
    def test_agent_has_invoke_method(self) -> None:
        agent = create_agent_manager()
        assert hasattr(agent, "invoke")

    @patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"})
    def test_custom_checkpointer(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        cp = MemorySaver()
        agent = create_agent_manager(checkpointer=cp)
        assert agent is not None
