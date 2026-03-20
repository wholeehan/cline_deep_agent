"""LLM provider abstraction — single factory consumed everywhere."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama server is unreachable or requested model is missing."""


def _check_ollama_connectivity(base_url: str, model: str) -> None:
    """Ping Ollama, verify the model exists, and confirm it supports tool calling."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {base_url}. "
            "Ensure Ollama is running (`ollama serve`) and the URL is correct."
        ) from exc

    available: list[str] = [m["name"] for m in resp.json().get("models", [])]
    # Match with or without tag suffix
    matched = any(name == model or name.startswith(f"{model}:") for name in available)
    if not matched:
        raise OllamaUnavailableError(
            f"Model '{model}' not found in Ollama. "
            f"Available models: {available}. "
            f"Pull it with: `ollama pull {model}`"
        )

    # Verify tool calling support with a lightweight probe
    _check_ollama_tool_support(base_url, model)


def _check_ollama_tool_support(base_url: str, model: str) -> None:
    """Send a minimal tool-call request to verify the model supports tools."""
    test_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "_probe",
                    "description": "probe",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
        ],
        "stream": False,
    }
    # Known models that support tool calling with Ollama
    tool_models = [
        "qwen2.5", "qwen3", "qwen3-coder", "llama3.1", "llama3.3",
        "mistral", "mixtral", "command-r", "gpt-oss",
    ]
    try:
        resp = httpx.post(
            f"{base_url}/api/chat", json=test_payload, timeout=30.0
        )
        if resp.status_code == 400 and "does not support tools" in resp.text:
            raise OllamaUnavailableError(
                f"Model '{model}' does not support tool calling, "
                f"which is required by the agent framework.\n"
                f"Switch to a tool-capable model by setting OLLAMA_MODEL "
                f"in your .env file.\n"
                f"Models known to support tools: {', '.join(tool_models)}\n"
                f"Example: OLLAMA_MODEL=gpt-oss:20b"
            )
        if resp.status_code == 500:
            error_text = resp.text or "unknown error"
            raise OllamaUnavailableError(
                f"Model '{model}' failed to load on Ollama server "
                f"(HTTP 500): {error_text}\n"
                f"This is often caused by insufficient RAM/VRAM for the model.\n"
                f"Try a smaller model or free up resources.\n"
                f"Suggested models: {', '.join(tool_models)}\n"
                f"Check Ollama server logs for details: journalctl -u ollama"
            )
    except OllamaUnavailableError:
        raise
    except httpx.HTTPError:
        # Non-critical — if the probe fails for other reasons, let the
        # actual agent invocation surface the error naturally.
        logger.debug("Tool support probe failed for %s, skipping check", model)


def get_llm(temperature: float = 0.0, **kwargs: Any) -> BaseChatModel:
    """Return the configured LLM backend.

    Reads ``LLM_PROVIDER`` env var (default ``"anthropic"``).
    When ``ollama``, pings the server and verifies the model exists.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    if provider == "vllm":
        model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-Coder-Next")
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="not-needed",
            temperature=temperature,
            **kwargs,
        )

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        _check_ollama_connectivity(base_url, model)
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs,
        )

    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),  # type: ignore[call-arg]
        temperature=temperature,
        **kwargs,
    )


# -- Context window guard for Ollama ------------------------------------------

OLLAMA_TOKEN_SOFT_CAP = 32_000


def trim_messages_for_context(
    messages: list[Any],
    provider: str | None = None,
) -> list[Any]:
    """If using Ollama, drop oldest non-system messages when context is large.

    This is a rough heuristic: ~4 chars per token. Real token counting should
    use the model's tokenizer, but this provides a safe guardrail.
    """
    if (provider or os.getenv("LLM_PROVIDER", "anthropic")) not in ("ollama", "vllm"):
        return messages

    estimated_tokens = sum(len(str(m)) for m in messages) // 4
    if estimated_tokens <= OLLAMA_TOKEN_SOFT_CAP:
        return messages

    logger.warning(
        "Context ~%d tokens exceeds Ollama soft cap (%d). Trimming oldest messages.",
        estimated_tokens,
        OLLAMA_TOKEN_SOFT_CAP,
    )
    # Keep system message (index 0) and trim from the front
    system_msgs = [m for m in messages if getattr(m, "type", None) == "system"]
    non_system = [m for m in messages if getattr(m, "type", None) != "system"]

    while (
        non_system
        and sum(len(str(m)) for m in [*system_msgs, *non_system]) // 4 > OLLAMA_TOKEN_SOFT_CAP
    ):
        non_system.pop(0)

    return system_msgs + non_system
