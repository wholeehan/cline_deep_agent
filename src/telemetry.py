"""PostgreSQL callback handler for LLM trace telemetry."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

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

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("invocation_params", {}).get("model", "")
        if not model:
            model = serialized.get("kwargs", {}).get("model", "unknown")
        self._pending[run_id] = {
            "start_time": time.monotonic(),
            "model": model,
            "provider": kwargs.get("invocation_params", {}).get("_type", "unknown"),
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
        usage = (response.llm_output or {}).get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        cost = _estimate_cost(pending["model"], prompt_tokens, completion_tokens)

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
            logger.debug("Failed to insert LLM trace row", exc_info=True)
