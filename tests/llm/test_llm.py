"""Tests for LLM provider abstraction."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.llm import (
    OllamaUnavailableError,
    _check_ollama_connectivity,
    _check_ollama_tool_support,
    get_llm,
    trim_messages_for_context,
)


class TestGetLlmAnthropic:
    """Tests for Anthropic provider path."""

    def test_returns_chat_anthropic_by_default(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"}):
            llm = get_llm()
        assert type(llm).__name__ == "ChatAnthropic"

    def test_custom_model_name(self) -> None:
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_MODEL": "claude-3-opus-20240229",
            "ANTHROPIC_API_KEY": "test-key",
        }):
            llm = get_llm()
        assert llm.model == "claude-3-opus-20240229"


class TestGetLlmOllama:
    """Tests for Ollama provider path."""

    def test_returns_chat_ollama(self) -> None:
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
            llm = get_llm()
        assert type(llm).__name__ == "ChatOllama"

    def test_ollama_unavailable_raises(self) -> None:
        import httpx as _httpx

        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}),
            patch("src.llm.httpx.get", side_effect=_httpx.ConnectError("refused")),
            pytest.raises(OllamaUnavailableError, match="Cannot reach Ollama"),
        ):
            get_llm()

    def test_ollama_model_missing_raises(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_resp.raise_for_status = MagicMock()

        env = {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3-coder:latest"}
        with (
            patch.dict(os.environ, env),
            patch("src.llm.httpx.get", return_value=mock_resp),
            pytest.raises(OllamaUnavailableError, match="not found"),
        ):
            get_llm()


class TestCheckOllamaConnectivity:
    """Unit tests for the connectivity check."""

    def test_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        mock_resp.raise_for_status = MagicMock()

        mock_tool_resp = MagicMock()
        mock_tool_resp.status_code = 200

        with (
            patch("src.llm.httpx.get", return_value=mock_resp),
            patch("src.llm.httpx.post", return_value=mock_tool_resp),
        ):
            _check_ollama_connectivity("http://localhost:11434", "gpt-oss:20b")

    def test_partial_name_match(self) -> None:
        """Model name without tag should match model with tag."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        mock_resp.raise_for_status = MagicMock()

        mock_tool_resp = MagicMock()
        mock_tool_resp.status_code = 200

        with (
            patch("src.llm.httpx.get", return_value=mock_resp),
            patch("src.llm.httpx.post", return_value=mock_tool_resp),
        ):
            _check_ollama_connectivity("http://localhost:11434", "gpt-oss")


class TestCheckOllamaToolSupport:
    """Tests for the tool support validation."""

    def test_raises_when_model_lacks_tool_support(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "registry.ollama.ai/library/qwen3-coder:latest does not support tools"

        with (
            patch("src.llm.httpx.post", return_value=mock_resp),
            pytest.raises(OllamaUnavailableError, match="does not support tool calling"),
        ):
            _check_ollama_tool_support("http://localhost:11434", "qwen3-coder:latest")

    def test_passes_when_model_supports_tools(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("src.llm.httpx.post", return_value=mock_resp):
            _check_ollama_tool_support("http://localhost:11434", "gpt-oss:20b")

    def test_silent_on_other_http_errors(self) -> None:
        """Non-tool-related errors should not crash startup."""
        import httpx as _httpx

        with patch("src.llm.httpx.post", side_effect=_httpx.ConnectError("timeout")):
            # Should not raise — just logs a debug message
            _check_ollama_tool_support("http://localhost:11434", "some-model")


class TestTrimMessagesForContext:
    """Tests for context window guard."""

    def test_no_trimming_for_anthropic(self) -> None:
        msgs = ["x" * 200_000]
        result = trim_messages_for_context(msgs, provider="anthropic")
        assert len(result) == 1

    def test_trimming_for_ollama_when_over_cap(self) -> None:
        # Each message ~40k chars = ~10k tokens, well over 32k cap
        msgs = [MagicMock(type="user", __str__=lambda s: "x" * 40_000) for _ in range(5)]
        result = trim_messages_for_context(msgs, provider="ollama")
        assert len(result) < 5

    def test_no_trimming_when_under_cap(self) -> None:
        msgs = [MagicMock(type="user", __str__=lambda s: "hello")]
        result = trim_messages_for_context(msgs, provider="ollama")
        assert len(result) == 1

    def test_system_messages_preserved(self) -> None:
        sys_msg = MagicMock(type="system", __str__=lambda s: "system prompt")
        user_msgs = [MagicMock(type="user", __str__=lambda s: "x" * 50_000) for _ in range(5)]
        result = trim_messages_for_context([sys_msg, *user_msgs], provider="ollama")
        assert any(getattr(m, "type", None) == "system" for m in result)
