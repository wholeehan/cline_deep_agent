"""PostgreSQL callback handler for LLM trace telemetry."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global telemetry callbacks registry
# ---------------------------------------------------------------------------
# Any callbacks added here are automatically injected into every LLM instance
# created by get_llm(), ensuring telemetry capture for all LLM calls
# (main agent, subagents, standalone tool calls like answer_question).

_global_callbacks: list[BaseCallbackHandler] = []


def register_global_callback(cb: BaseCallbackHandler) -> None:
    """Register a callback that will be injected into all LLM instances."""
    _global_callbacks.append(cb)


def get_global_callbacks() -> list[BaseCallbackHandler]:
    """Return the list of globally registered telemetry callbacks."""
    return list(_global_callbacks)


# Price table per 1M tokens (input, output) — extend as needed
_PRICES: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-opus-4": (15.0, 75.0),
    "gpt-4o": (2.5, 10.0),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD from token counts."""
    for prefix, (inp_price, out_price) in _PRICES.items():
        if prefix in model.lower():
            return (prompt_tokens * inp_price + completion_tokens * out_price) / 1_000_000
    return 0.0


class PostgresCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that writes LLM call traces to the llm_traces table."""

    def __init__(self, pool: Any, session_id: str | None = None) -> None:
        self._pool = pool
        self._session_id = session_id
        self._pending: dict[UUID, dict[str, Any]] = {}

    def _extract_model_provider(
        self, serialized: dict[str, Any], **kwargs: Any
    ) -> tuple[str, str]:
        """Extract model name and provider from callback kwargs."""
        invocation_params = kwargs.get("invocation_params", {})
        model = (
            invocation_params.get("model")
            or invocation_params.get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("kwargs", {}).get("model_name")
            or kwargs.get("metadata", {}).get("ls_model_name")
            or "unknown"
        )
        provider = (
            invocation_params.get("_type")
            or serialized.get("id", [""])[-1]
            or "unknown"
        )
        return model, provider

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model, provider = self._extract_model_provider(serialized, **kwargs)
        logger.info("on_llm_start: model=%s provider=%s", model, provider)
        self._pending[run_id] = {
            "start_time": time.monotonic(),
            "model": model,
            "provider": provider,
        }

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model, provider = self._extract_model_provider(serialized, **kwargs)
        logger.info("on_chat_model_start: model=%s provider=%s", model, provider)
        self._pending[run_id] = {
            "start_time": time.monotonic(),
            "model": model,
            "provider": provider,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return

        latency_ms = int((time.monotonic() - pending["start_time"]) * 1000)

        # Extract token usage — check multiple locations across providers
        llm_output = response.llm_output or {}
        usage = (
            llm_output.get("token_usage")
            or llm_output.get("usage")
            or {}
        )

        # Also check per-generation usage metadata (langchain-ollama, langchain-openai)
        if not usage or not usage.get("prompt_tokens"):
            for gen_list in (response.generations or []):
                for gen in gen_list:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    gen_usage = gen_info.get("usage", gen_info.get("token_usage", {}))
                    if gen_usage and gen_usage.get("prompt_tokens"):
                        usage = gen_usage
                        break
                    # Ollama puts usage fields at top level of generation_info
                    if gen_info.get("prompt_eval_count"):
                        usage = {
                            "prompt_tokens": gen_info.get("prompt_eval_count", 0),
                            "completion_tokens": gen_info.get("eval_count", 0),
                            "total_tokens": gen_info.get("prompt_eval_count", 0) + gen_info.get("eval_count", 0),
                        }
                        break
                if usage and usage.get("prompt_tokens"):
                    break

        # Also try response_metadata on generations (langchain >=0.2)
        if not usage or not usage.get("prompt_tokens"):
            for gen_list in (response.generations or []):
                for gen in gen_list:
                    resp_meta = getattr(gen, "message", None)
                    if resp_meta:
                        rm_usage = getattr(resp_meta, "usage_metadata", None)
                        if rm_usage:
                            usage = {
                                "prompt_tokens": getattr(rm_usage, "input_tokens", 0) or rm_usage.get("input_tokens", 0),
                                "completion_tokens": getattr(rm_usage, "output_tokens", 0) or rm_usage.get("output_tokens", 0),
                                "total_tokens": getattr(rm_usage, "total_tokens", 0) or rm_usage.get("total_tokens", 0),
                            }
                            break
                if usage and usage.get("prompt_tokens"):
                    break

        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens)
        cost = _estimate_cost(pending["model"], prompt_tokens, completion_tokens)

        logger.info(
            "on_llm_end: model=%s prompt=%d completion=%d total=%d llm_output_keys=%s",
            pending["model"], prompt_tokens, completion_tokens, total_tokens, list(llm_output.keys()),
        )

        self._insert_trace(
            provider=pending["provider"],
            model=pending["model"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            is_retry=False,
            error=None,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return

        latency_ms = int((time.monotonic() - pending["start_time"]) * 1000)
        self._insert_trace(
            provider=pending["provider"],
            model=pending["model"],
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            is_retry=False,
            error=str(error),
        )

    def _insert_trace(self, **row: Any) -> None:
        row["session_id"] = self._session_id
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    """INSERT INTO llm_traces
                       (session_id, provider, model,
                        prompt_tokens, completion_tokens, total_tokens,
                        latency_ms, cost_usd, is_retry, error)
                       VALUES (%(session_id)s, %(provider)s, %(model)s,
                               %(prompt_tokens)s, %(completion_tokens)s, %(total_tokens)s,
                               %(latency_ms)s, %(cost_usd)s, %(is_retry)s, %(error)s)""",
                    row,
                )
                conn.commit()
        except Exception:
            logger.warning("Failed to insert LLM trace row", exc_info=True)
