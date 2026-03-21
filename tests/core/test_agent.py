"""Tests for DeepAgents core agent setup."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

from src.agent import (
    _SQLITE_AVAILABLE,
    create_agent_manager,
    create_composite_backend,
    get_checkpointer,
    report_progress,
    save_output,
)


class TestReportProgress:
    """Tests for the report_progress tool."""

    def test_returns_formatted_message(self) -> None:
        result = report_progress.invoke({"message": "Step 1 done"})
        assert "[progress]" in result
        assert "Step 1 done" in result


class TestSaveOutput:
    """Tests for the save_output tool."""

    def test_saves_file_to_output_dir(self, tmp_path: Any) -> None:
        with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
            result = save_output.invoke({"file_path": "hello.txt", "content": "world"})
            assert "Saved to" in result
            assert (tmp_path / "hello.txt").read_text() == "world"

    def test_creates_subdirectories(self, tmp_path: Any) -> None:
        with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
            result = save_output.invoke({"file_path": "src/main.py", "content": "print(1)"})
            assert "Saved to" in result
            assert (tmp_path / "src" / "main.py").read_text() == "print(1)"

    def test_blocks_path_traversal(self, tmp_path: Any) -> None:
        with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
            result = save_output.invoke({"file_path": "../../etc/passwd", "content": "bad"})
            assert "Error" in result


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
            # FilesystemBackend is called twice: virtual project + output
            assert mock_fs.call_count == 2
            mock_composite.assert_called_once()

    def test_output_route_included(self) -> None:
        mock_runtime = MagicMock()
        mock_runtime.store = MagicMock()
        mock_runtime.config = {"configurable": {"thread_id": "test"}}

        with (
            patch("src.agent.StateBackend"),
            patch("src.agent.StoreBackend"),
            patch("src.agent.FilesystemBackend") as mock_fs,
            patch("src.agent.CompositeBackend") as mock_composite,
        ):
            create_composite_backend(mock_runtime)
            call_kwargs = mock_composite.call_args
            routes = call_kwargs.kwargs.get("routes") or call_kwargs[1].get("routes", {})
            assert "/output/" in routes

            # Verify FilesystemBackend for /output/ uses root_dir (not root)
            output_call = mock_fs.call_args_list[1]
            assert "root_dir" in output_call.kwargs
            assert "virtual_mode" in output_call.kwargs


class TestGetCheckpointer:
    """Tests for the checkpointer factory."""

    def test_returns_memory_saver_by_default(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver

        with patch.dict(os.environ, {}, clear=False):
            # Remove CHECKPOINT_DB if present
            os.environ.pop("CHECKPOINT_DB", None)
            cp = get_checkpointer()
            assert isinstance(cp, MemorySaver)

    def test_returns_memory_saver_when_no_path(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver

        cp = get_checkpointer(db_path=None)
        assert isinstance(cp, MemorySaver)

    @patch.dict(os.environ, {"CHECKPOINT_DB": "/tmp/test_cp.db"})
    def test_uses_env_var_path(self) -> None:
        if _SQLITE_AVAILABLE:
            cp = get_checkpointer()
            assert cp is not None
            # Clean up
            if os.path.exists("/tmp/test_cp.db"):
                os.unlink("/tmp/test_cp.db")
        else:
            from langgraph.checkpoint.memory import MemorySaver

            cp = get_checkpointer()
            assert isinstance(cp, MemorySaver)

    def test_explicit_path_overrides_env(self) -> None:
        if _SQLITE_AVAILABLE:
            cp = get_checkpointer(db_path="/tmp/explicit_cp.db")
            assert cp is not None
            if os.path.exists("/tmp/explicit_cp.db"):
                os.unlink("/tmp/explicit_cp.db")
        else:
            from langgraph.checkpoint.memory import MemorySaver

            cp = get_checkpointer(db_path="/tmp/explicit_cp.db")
            assert isinstance(cp, MemorySaver)


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
