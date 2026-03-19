"""LLM provider abstraction — single factory consumed everywhere."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama server is unreachable or requested model is missing."""


def _check_ollama_connectivity(base_url: str, model: str) -> None:
    """Ping Ollama and verify the requested model is available."""
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


def get_llm(temperature: float = 0.0, **kwargs: Any) -> BaseChatModel:
    """Return the configured LLM backend.

    Reads ``LLM_PROVIDER`` env var (default ``"anthropic"``).
    When ``ollama``, pings the server and verifies the model exists.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen3-coder:latest")
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
    if (provider or os.getenv("LLM_PROVIDER", "anthropic")) != "ollama":
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
