"""Tests for Human-in-the-loop flow helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.hitl import (
    ActionRequest,
    build_approve_command,
    build_batch_command,
    build_edit_command,
    build_reject_command,
    extract_interrupts,
)


class TestExtractInterrupts:
    """Tests for interrupt detection helper."""

    def test_no_interrupt_returns_empty(self) -> None:
        result: dict = {"messages": []}
        assert extract_interrupts(result) == []

    def test_single_interrupt_dict(self) -> None:
        result = {
            "__interrupt__": {
                "tool_name": "approve_cline_action",
                "args": {"action": "rm -rf /tmp"},
                "description": "Delete action",
            }
        }
        requests = extract_interrupts(result)
        assert len(requests) == 1
        assert requests[0].tool_name == "approve_cline_action"
        assert requests[0].action == {"action": "rm -rf /tmp"}

    def test_multiple_interrupts_list(self) -> None:
        result = {
            "__interrupt__": [
                {
                    "tool_name": "approve_cline_action",
                    "args": {"action": "delete file"},
                },
                {
                    "tool_name": "dispatch_subtask",
                    "args": {"subtask": "deploy"},
                },
            ]
        }
        requests = extract_interrupts(result)
        assert len(requests) == 2
        assert requests[0].tool_name == "approve_cline_action"
        assert requests[1].tool_name == "dispatch_subtask"

    def test_interrupt_with_value_attribute(self) -> None:
        """LangGraph Interrupt objects have a .value attribute."""
        interrupt_obj = MagicMock()
        interrupt_obj.value = {
            "tool_name": "approve_cline_action",
            "args": {"action": "curl api"},
            "description": "HTTP request",
        }

        result = {"__interrupt__": [interrupt_obj]}
        requests = extract_interrupts(result)
        assert len(requests) == 1
        assert requests[0].tool_name == "approve_cline_action"

    def test_none_interrupt_returns_empty(self) -> None:
        result = {"__interrupt__": None}
        assert extract_interrupts(result) == []


class TestApproveCommand:
    """Tests for approve flow."""

    def test_approve_command_structure(self) -> None:
        cmd = build_approve_command()
        assert cmd.resume == {"decisions": [{"type": "approve"}]}


class TestRejectCommand:
    """Tests for reject flow."""

    def test_reject_command_structure(self) -> None:
        cmd = build_reject_command()
        assert cmd.resume == {"decisions": [{"type": "reject"}]}


class TestEditCommand:
    """Tests for edit flow."""

    def test_edit_command_structure(self) -> None:
        edited = {"action": "rm /tmp/safe_file.txt"}
        cmd = build_edit_command(edited)
        assert cmd.resume == {
            "decisions": [{"type": "edit", "edited_action": edited}]
        }

    def test_edited_args_preserved(self) -> None:
        edited = {"port": 8080, "host": "localhost"}
        cmd = build_edit_command(edited)
        decision = cmd.resume["decisions"][0]
        assert decision["edited_action"]["port"] == 8080


class TestBatchCommand:
    """Tests for multi-interrupt batching."""

    def test_batch_two_decisions(self) -> None:
        decisions = [
            {"type": "approve"},
            {"type": "reject"},
        ]
        cmd = build_batch_command(decisions)
        assert len(cmd.resume["decisions"]) == 2
        assert cmd.resume["decisions"][0]["type"] == "approve"
        assert cmd.resume["decisions"][1]["type"] == "reject"

    def test_batch_with_edit(self) -> None:
        decisions = [
            {"type": "approve"},
            {"type": "edit", "edited_action": {"file": "new.py"}},
        ]
        cmd = build_batch_command(decisions)
        assert cmd.resume["decisions"][1]["edited_action"]["file"] == "new.py"


class TestActionRequest:
    """Tests for ActionRequest dataclass."""

    def test_create_action_request(self) -> None:
        req = ActionRequest(
            tool_name="approve_cline_action",
            action={"action": "delete /tmp"},
            description="Delete action",
        )
        assert req.tool_name == "approve_cline_action"
        assert req.description == "Delete action"

    def test_default_description(self) -> None:
        req = ActionRequest(tool_name="test", action={})
        assert req.description == ""
