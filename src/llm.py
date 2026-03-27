"""LLM provider abstraction — single factory consumed everywhere."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ollama._types import ResponseError

logger = logging.getLogger(__name__)


_TOOL_CALL_PARSE_RE = re.compile(r"error parsing tool call")


def _extract_json_from_raw(raw: str) -> dict | None:
    """Try to extract a JSON object from a raw string that has reasoning prefix."""
    # Find the last JSON object in the string (tool call args are typically at the end)
    brace_depth = 0
    json_start = None
    for i in range(len(raw) - 1, -1, -1):
        if raw[i] == "}":
            if brace_depth == 0:
                json_end = i + 1
            brace_depth += 1
        elif raw[i] == "{":
            brace_depth -= 1
            if brace_depth == 0:
                json_start = i
                break
    if json_start is not None:
        try:
            return json.loads(raw[json_start:json_end])
        except (json.JSONDecodeError, UnboundLocalError):
            pass
    return None


def _extract_reasoning_text(raw: str) -> str:
    """Extract the reasoning/text portion before any JSON in the raw output."""
    # Find the first '{' that starts a JSON object
    for i, ch in enumerate(raw):
        if ch == "{":
            text = raw[:i].strip()
            return text if text else ""
    return raw.strip()


# ---------------------------------------------------------------------------
# Ollama model memory management
# ---------------------------------------------------------------------------


def unload_ollama_model(model: str, base_url: str | None = None) -> None:
    """Unload a model from Ollama's VRAM by setting keep_alive=0.

    This frees GPU memory so another model can be loaded.
    """
    url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        resp = httpx.post(
            f"{url}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10.0,
        )
        if resp.status_code == 200:
            logger.info("Unloaded model '%s' from Ollama VRAM", model)
        else:
            logger.warning("Failed to unload model '%s': HTTP %d", model, resp.status_code)
    except Exception:
        logger.debug("Could not unload model '%s' (Ollama may not be running)", model)


# Track which Ollama models are managed so we can unload them on swap
_active_ollama_models: set[str] = set()


def _register_ollama_model(model: str) -> None:
    """Register an Ollama model as active (may need unloading before swap)."""
    _active_ollama_models.add(model)


def _unload_other_ollama_models(current_model: str, base_url: str | None = None) -> None:
    """Unload all registered Ollama models except the current one."""
    for m in _active_ollama_models:
        if m != current_model:
            unload_ollama_model(m, base_url)


class SwapAwareChatOllama(ChatOllama):
    """ChatOllama that unloads other Ollama models before generating.

    When running multiple Ollama models on limited VRAM, this ensures only
    one model is loaded at a time by unloading competing models before
    each inference call.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        _unload_other_ollama_models(self.model, self.base_url)
        return super()._generate(messages, stop, run_manager, **kwargs)


class RobustChatOllama(SwapAwareChatOllama):
    """ChatOllama wrapper that handles models mixing reasoning with tool calls.

    Inherits from SwapAwareChatOllama to automatically unload competing models
    before each call, preventing VRAM exhaustion when running multiple models.

    Some Ollama models (e.g. gpt-oss) embed chain-of-thought reasoning before
    the tool call JSON arguments, causing Ollama's server-side parser to fail
    with HTTP 500 "error parsing tool call". This subclass catches that error
    and either:
    1. Extracts the reasoning as text content (since the model chose to reason
       rather than make a clean tool call), or
    2. Retries without tools to get a plain text response.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(messages, stop, run_manager, **kwargs)
        except ResponseError as exc:
            if exc.status_code != 500 or not _TOOL_CALL_PARSE_RE.search(str(exc)):
                raise

            raw = str(exc)
            logger.warning(
                "Ollama tool-call parse failure — model mixed reasoning with "
                "tool JSON. Extracting text response from raw output."
            )

            # Extract the raw= field from the error message
            raw_match = re.search(r"raw='(.*?)', err=", raw, re.DOTALL)
            if raw_match:
                raw_content = raw_match.group(1)
                reasoning = _extract_reasoning_text(raw_content)
                if reasoning:
                    # Return the model's reasoning as a plain text response
                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=AIMessage(content=reasoning),
                            )
                        ]
                    )

            # Fallback: retry without tools to get a text-only response
            logger.info("Retrying Ollama call without tools for text-only response.")
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            # Use bind_tools([]) to clear tools for the retry
            no_tools_llm = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
                num_predict=self.num_predict,
            )
            return no_tools_llm._generate(messages, stop, run_manager, **kwargs)


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama server is unreachable or requested model is missing."""


def _check_ollama_connectivity(
    base_url: str, model: str, *, skip_tool_check: bool = False
) -> None:
    """Ping Ollama, verify the model exists, and optionally confirm tool support."""
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
    if not skip_tool_check:
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


def get_llm(
    temperature: float = 0.0,
    model_override: str | None = None,
    skip_tool_check: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Return the configured LLM backend.

    Reads ``LLM_PROVIDER`` env var (default ``"anthropic"``).
    When ``ollama``, pings the server and verifies the model exists.

    Global telemetry callbacks (registered via ``register_global_callback``)
    are automatically merged into the ``callbacks`` kwarg so every LLM
    instance captures telemetry — including subagent LLMs and standalone
    tool calls like ``answer_question``.

    Parameters
    ----------
    temperature:
        Sampling temperature.
    model_override:
        If given, use this model name instead of the env-var default.
    skip_tool_check:
        If True, skip the Ollama native tool-support probe.  Useful for
        models that handle tool calling through prompt-based methods
        (e.g. qwen3-coder).
    """
    # Merge global telemetry callbacks into kwargs
    try:
        from src.telemetry import get_global_callbacks

        global_cbs = get_global_callbacks()
        if global_cbs:
            existing = list(kwargs.get("callbacks", []) or [])
            # Avoid duplicates
            existing_ids = {id(cb) for cb in existing}
            for cb in global_cbs:
                if id(cb) not in existing_ids:
                    existing.append(cb)
            kwargs["callbacks"] = existing
            logger.info(
                "get_llm: injected %d global telemetry callback(s) for model_override=%s",
                len(global_cbs), model_override,
            )
        else:
            logger.info("get_llm: no global callbacks registered (model_override=%s)", model_override)
    except ImportError:
        logger.warning("get_llm: src.telemetry not available, skipping global callbacks")

    provider = os.getenv("LLM_PROVIDER", "anthropic")

    if provider == "vllm":
        model = model_override or os.getenv("VLLM_MODEL", "Qwen/Qwen3-Coder-Next")
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        logger.info("get_llm: creating ChatOpenAI (vllm) model=%s callbacks=%d", model, len(kwargs.get("callbacks", [])))
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="not-needed",
            temperature=temperature,
            **kwargs,
        )

    if provider == "ollama":
        model = model_override or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        _check_ollama_connectivity(base_url, model, skip_tool_check=skip_tool_check)
        _register_ollama_model(model)
        logger.info("get_llm: creating RobustChatOllama model=%s callbacks=%d", model, len(kwargs.get("callbacks", [])))
        return RobustChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "32768")),
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "8192")),
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
