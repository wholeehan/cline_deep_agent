"""PostgreSQL connection pool and schema initialization for telemetry storage."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_pool: Any | None = None
_pool_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agent_events (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT now(),
    level       VARCHAR(10) NOT NULL,
    logger      VARCHAR(100),
    message     TEXT NOT NULL,
    llm_provider VARCHAR(20),
    event_type  VARCHAR(50),
    subtask_id  VARCHAR(100),
    tool_name   VARCHAR(100),
    decision    VARCHAR(50),
    status      VARCHAR(50),
    session_id  UUID,
    exception   TEXT,
    traceback   TEXT,
    extra       JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_agent_events_ts ON agent_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_agent_events_session ON agent_events (session_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_event_type ON agent_events (event_type);
CREATE INDEX IF NOT EXISTS idx_agent_events_extra ON agent_events USING GIN (extra);

CREATE TABLE IF NOT EXISTS llm_traces (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id      UUID,
    provider        VARCHAR(20) NOT NULL,
    model           VARCHAR(100) NOT NULL,
    prompt_tokens   INT,
    completion_tokens INT,
    total_tokens    INT,
    latency_ms      INT,
    cost_usd        NUMERIC(10,6),
    is_retry        BOOLEAN DEFAULT FALSE,
    error           TEXT,
    request_meta    JSONB DEFAULT '{}',
    response_meta   JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_llm_traces_ts ON llm_traces (timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_traces_model ON llm_traces (model);
CREATE INDEX IF NOT EXISTS idx_llm_traces_session ON llm_traces (session_id);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id          SERIAL PRIMARY KEY,
    run_id      VARCHAR(100) UNIQUE NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    agent       VARCHAR(50) NOT NULL,
    provider    VARCHAR(50) NOT NULL,
    model       VARCHAR(100) NOT NULL,
    summary     JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS benchmark_results (
    id                  SERIAL PRIMARY KEY,
    run_id              VARCHAR(100) NOT NULL REFERENCES benchmark_runs(run_id),
    task_id             VARCHAR(100) NOT NULL,
    repetition          INT NOT NULL,
    passed              BOOLEAN NOT NULL,
    wall_clock_seconds  NUMERIC(10,3),
    prompt_tokens       INT,
    completion_tokens   INT,
    total_tokens        INT,
    cost_usd            NUMERIC(10,6),
    files_changed       TEXT[],
    test_output         TEXT,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_bench_results_run ON benchmark_results (run_id);
CREATE INDEX IF NOT EXISTS idx_bench_results_task ON benchmark_results (task_id);
CREATE INDEX IF NOT EXISTS idx_bench_results_passed ON benchmark_results (passed);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_db_url() -> str | None:
    """Return the DATABASE_URL from environment, or None if not configured."""
    return os.getenv("DATABASE_URL")


def get_connection_pool() -> Any | None:
    """Return a shared psycopg ConnectionPool, or None if DATABASE_URL is not set.

    The pool is created once (thread-safe) and the schema is initialized on first call.
    """
    global _pool  # noqa: PLW0603

    url = get_db_url()
    if not url:
        return None

    if _pool is not None:
        return _pool

    with _pool_lock:
        # Double-check after acquiring lock
        if _pool is not None:
            return _pool

        try:
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(url, min_size=1, max_size=5)
            _init_schema(pool)
            _pool = pool
            logger.info("PostgreSQL connection pool created for telemetry storage")
            return _pool
        except Exception:
            logger.exception("Failed to create PostgreSQL connection pool")
            return None


def _init_schema(pool: Any) -> None:
    """Create application tables if they don't exist (idempotent)."""
    with pool.connection() as conn:
        conn.execute(_SCHEMA_SQL)
        conn.commit()
    logger.info("PostgreSQL telemetry schema initialized")
