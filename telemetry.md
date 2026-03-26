# Telemetry Database System

This document describes the PostgreSQL-based telemetry database system used by Cline Deep Agent for storing, querying, and analyzing agent execution data.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Database Schema](#database-schema)
- [How Data Flows Into the Database](#how-data-flows-into-the-database)
- [Querying Telemetry Data](#querying-telemetry-data)
- [Migration from Flat Files](#migration-from-flat-files)
- [Dashboard Integration](#dashboard-integration)
- [Python API Reference](#python-api-reference)
- [Backward Compatibility](#backward-compatibility)
- [Troubleshooting](#troubleshooting)

---

## Overview

The telemetry system captures three categories of operational data:

1. **Agent Events** -- structured log events from agent execution (tool calls, decisions, errors)
2. **LLM Traces** -- per-call telemetry from LLM providers (tokens, latency, cost, errors)
3. **Benchmark Results** -- task pass/fail outcomes, token usage, and cost from benchmark runs

All telemetry is stored in a PostgreSQL database (`cline_telemetry`) and is fully queryable via standard SQL. The system uses a **dual-write** strategy: data is written to both PostgreSQL and the original flat files simultaneously, so nothing is lost if the database is temporarily unavailable.

### Why PostgreSQL

PostgreSQL was chosen over MongoDB and other alternatives for these reasons:

- **Native LangGraph integration** -- `langgraph-checkpoint-postgres` is a first-party, drop-in replacement for the SQLite checkpointer. MongoDB has no equivalent.
- **JSONB for flexibility** -- Semi-structured data (traces, extra metadata) is stored in JSONB columns with GIN indexes, providing the same schema flexibility as a document database.
- **SQL for analytics** -- Relational queries with typed columns, aggregation, window functions, and joins are essential for cross-run analysis and dashboard queries.
- **Dashboard ecosystem** -- PostgreSQL is natively supported by Grafana, Metabase, Superset, and Apache Preset with zero additional connectors.
- **Concurrent writes** -- MVCC with row-level locking handles multiple agents writing simultaneously without contention.

---

## Architecture

```
                           +---------------------+
                           |   Agent Manager     |
                           |   (src/cli.py)      |
                           +----------+----------+
                                      |
              +-----------+-----------+-----------+-----------+
              |           |           |           |           |
              v           v           v           v           v
        +-----------+ +---------+ +---------+ +---------+ +---------+
        | Postgres  | | Postgres| | Postgres| | File    | | File    |
        | Saver     | | Log     | | Callback| | Callback| | Handler |
        | (check-   | | Handler | | Handler | | Handler | | (JSONL) |
        |  points)  | | (events)| | (traces)| | (.log)  | | (.jsonl)|
        +-----+-----+ +----+----+ +----+----+ +----+----+ +----+----+
              |             |           |           |           |
              v             v           v           v           v
        +-----------+ +------------------------------------------+
        | checkpoint| |           PostgreSQL                     |
        | tables    | |  agent_events | llm_traces | benchmark_*|
        | (langgraph| +------------------------------------------+
        |  managed) |
        +-----------+ +------------------------------------------+
                       |           Flat Files                    |
                       |  agent.jsonl | telemetry.log | *.json   |
                       +------------------------------------------+
```

### Data Flow Summary

| Data Type | Source | PostgreSQL Table | Flat File Fallback |
|-----------|--------|------------------|--------------------|
| Agent state checkpoints | LangGraph runtime | `checkpoints` (auto-managed) | `data/checkpoints.db` (SQLite) |
| Structured log events | Python `logging` module | `agent_events` | `data/logs/agent.jsonl` |
| LLM call traces | LangChain callbacks | `llm_traces` | `data/logs/telemetry.log` |
| Benchmark run metadata | Benchmark runner | `benchmark_runs` | `benchmark/results/*.json` |
| Benchmark task results | Benchmark runner | `benchmark_results` | `benchmark/results/*.json` |

---

## Prerequisites

- **PostgreSQL 16+** (runs as a Docker container via `docker-compose.yml`)
- **Python 3.12+**
- Python packages (installed via `pip install -e .`):
  - `psycopg[binary]>=3.1` -- PostgreSQL adapter
  - `psycopg_pool>=3.1` -- Connection pooling
  - `langgraph-checkpoint-postgres>=2.0` -- LangGraph checkpointer

---

## Setup

### Option A: Docker Compose (recommended)

Start PostgreSQL alongside the other services:

```bash
docker compose up -d postgres
```

This starts a `postgres:16-alpine` container with:
- Database: `cline_telemetry`
- User: `cline`
- Password: `cline`
- Port: `5432` (mapped to host)
- Persistent volume: `postgres_data`
- Health check: `pg_isready` every 5 seconds

The full stack (agent + ollama + postgres) can be started with:

```bash
docker compose up -d
```

### Option B: External PostgreSQL

If you have an existing PostgreSQL instance, create the database manually:

```sql
CREATE DATABASE cline_telemetry;
CREATE USER cline WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE cline_telemetry TO cline;
```

Then set the `DATABASE_URL` environment variable (see [Configuration](#configuration)).

### Verify the Connection

```bash
# Using docker compose
docker compose exec postgres psql -U cline -d cline_telemetry -c '\dt'

# Using a local psql client
psql postgresql://cline:cline@localhost:5432/cline_telemetry -c '\dt'
```

On first agent startup, the application tables are created automatically (see [Schema Initialization](#schema-initialization)).

---

## Configuration

The telemetry database is controlled by a single environment variable:

```bash
# .env file
DATABASE_URL=postgresql://cline:cline@localhost:5432/cline_telemetry
```

### Connection String Format

```
postgresql://USER:PASSWORD@HOST:PORT/DATABASE
```

Examples:
```bash
# Local Docker (default)
DATABASE_URL=postgresql://cline:cline@localhost:5432/cline_telemetry

# Docker Compose internal network (used in docker-compose.yml)
DATABASE_URL=postgresql://cline:cline@postgres:5432/cline_telemetry

# Remote PostgreSQL with SSL
DATABASE_URL=postgresql://cline:secret@db.example.com:5432/cline_telemetry?sslmode=require
```

### Behavior When DATABASE_URL Is Not Set

When `DATABASE_URL` is absent or empty, the system operates in **legacy mode**:
- Checkpoints use SQLite (`CHECKPOINT_DB=./data/checkpoints.db`)
- Logs write to JSONL files (`LOG_FILE=./data/logs/agent.jsonl`)
- Telemetry writes to text files (`TELEMETRY_FILE=./data/logs/telemetry.log`)
- Benchmark results write to JSON files (`benchmark/results/*.json`)

No PostgreSQL connection is attempted, no errors are raised.

---

## Database Schema

The schema is defined in `src/db.py` and initialized automatically on first connection. All `CREATE` statements use `IF NOT EXISTS`, making initialization idempotent.

### `agent_events` -- Structured Agent Log Events

Replaces `data/logs/agent.jsonl`. Each row is one structured log event emitted during agent execution.

| Column | Type | Description |
|--------|------|-------------|
| `id` | `BIGSERIAL` | Auto-incrementing primary key |
| `timestamp` | `TIMESTAMPTZ` | Event time (defaults to `now()`) |
| `level` | `VARCHAR(10)` | Log level: `INFO`, `WARNING`, `ERROR`, `DEBUG` |
| `logger` | `VARCHAR(100)` | Python logger name (e.g., `src.agent`, `src.cli`) |
| `message` | `TEXT` | Human-readable log message |
| `llm_provider` | `VARCHAR(20)` | Active LLM provider: `anthropic`, `ollama`, `vllm` |
| `event_type` | `VARCHAR(50)` | Event category: `tool_call`, `decision`, `error`, etc. |
| `subtask_id` | `VARCHAR(100)` | ID of the subtask that generated this event |
| `tool_name` | `VARCHAR(100)` | Name of the tool being called (if applicable) |
| `decision` | `VARCHAR(50)` | Decision outcome: `approve`, `reject`, `escalate` |
| `status` | `VARCHAR(50)` | Execution status: `started`, `completed`, `failed` |
| `session_id` | `UUID` | Agent session (thread) identifier |
| `exception` | `TEXT` | Exception message (if error) |
| `traceback` | `TEXT` | Full Python traceback (if error) |
| `extra` | `JSONB` | Overflow field for arbitrary additional metadata |

**Indexes:**
- `idx_agent_events_ts` -- on `timestamp` (time-range queries)
- `idx_agent_events_session` -- on `session_id` (session replay)
- `idx_agent_events_event_type` -- on `event_type` (filtering by event kind)
- `idx_agent_events_extra` -- GIN index on `extra` (JSONB key lookups)

### `llm_traces` -- LLM Call Telemetry

Replaces `data/logs/telemetry.log`. Each row is one LLM API invocation.

| Column | Type | Description |
|--------|------|-------------|
| `id` | `BIGSERIAL` | Auto-incrementing primary key |
| `timestamp` | `TIMESTAMPTZ` | Call time (defaults to `now()`) |
| `session_id` | `UUID` | Agent session that made this call |
| `provider` | `VARCHAR(20)` | LLM provider: `anthropic`, `ollama`, `vllm`, `openai` |
| `model` | `VARCHAR(100)` | Model identifier (e.g., `claude-3-5-sonnet-20241022`) |
| `prompt_tokens` | `INT` | Number of input tokens |
| `completion_tokens` | `INT` | Number of output tokens |
| `total_tokens` | `INT` | Total tokens (prompt + completion) |
| `latency_ms` | `INT` | Round-trip latency in milliseconds |
| `cost_usd` | `NUMERIC(10,6)` | Estimated cost in USD |
| `is_retry` | `BOOLEAN` | Whether this call was a retry |
| `error` | `TEXT` | Error message (if the call failed) |
| `request_meta` | `JSONB` | Request metadata (prompt summary, tool names) |
| `response_meta` | `JSONB` | Response metadata (finish reason, tool calls made) |

**Indexes:**
- `idx_llm_traces_ts` -- on `timestamp`
- `idx_llm_traces_model` -- on `model`
- `idx_llm_traces_session` -- on `session_id`

### `benchmark_runs` -- Benchmark Run Headers

Replaces the top-level fields of each `benchmark/results/*.json` file. One row per benchmark execution.

| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL` | Auto-incrementing primary key |
| `run_id` | `VARCHAR(100)` | Unique run identifier (e.g., `run-20260321T180103Z-475b1006`) |
| `timestamp` | `TIMESTAMPTZ` | When the benchmark was executed |
| `agent` | `VARCHAR(50)` | Agent adapter used: `cline-deep`, `direct-llm` |
| `provider` | `VARCHAR(50)` | LLM provider used |
| `model` | `VARCHAR(100)` | Model identifier |
| `summary` | `JSONB` | Aggregated metrics: pass rate, avg tokens, by-category breakdown |

**Constraints:** `UNIQUE` on `run_id`

### `benchmark_results` -- Per-Task Benchmark Results

Replaces the `tasks` array inside each JSON report. One row per task per repetition.

| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL` | Auto-incrementing primary key |
| `run_id` | `VARCHAR(100)` | Foreign key to `benchmark_runs.run_id` |
| `task_id` | `VARCHAR(100)` | Task identifier (e.g., `bugfix-001`) |
| `repetition` | `INT` | Repetition number (1-based) |
| `passed` | `BOOLEAN` | Whether all tests passed |
| `wall_clock_seconds` | `NUMERIC(10,3)` | Total execution time |
| `prompt_tokens` | `INT` | Input tokens consumed |
| `completion_tokens` | `INT` | Output tokens generated |
| `total_tokens` | `INT` | Total tokens |
| `cost_usd` | `NUMERIC(10,6)` | Estimated cost in USD |
| `files_changed` | `TEXT[]` | PostgreSQL array of modified file paths |
| `test_output` | `TEXT` | Full pytest output |
| `error` | `TEXT` | Error message (if execution failed) |

**Indexes:**
- `idx_bench_results_run` -- on `run_id`
- `idx_bench_results_task` -- on `task_id`
- `idx_bench_results_passed` -- on `passed`

### LangGraph Checkpoint Tables (Auto-Managed)

These tables are created and managed by `langgraph-checkpoint-postgres`. Do not modify them directly.

- `checkpoints` -- Serialized agent state per thread
- `checkpoint_blobs` -- Large binary state data
- `checkpoint_writes` -- Write-ahead entries for crash recovery

---

## How Data Flows Into the Database

### Schema Initialization

On the first call to `get_connection_pool()` in `src/db.py`, the `_init_schema()` function runs all `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` statements. This is:
- **Automatic** -- no manual migration step required
- **Idempotent** -- safe to run on every startup
- **Non-destructive** -- never drops or alters existing tables

### Agent Events (Logging)

Agent events flow through Python's standard `logging` module:

```
Application code
    |
    v
logging.info("message", extra={"event_type": "tool_call", ...})
    |
    +---> StructuredFormatter --> FileHandler --> data/logs/agent.jsonl
    |
    +---> PostgresLogHandler --> INSERT INTO agent_events
```

The `PostgresLogHandler` (defined in `src/logging_config.py`) buffers log records and flushes them to the database in batches:
- **Batch size:** 50 records
- **Flush interval:** 5 seconds (whichever comes first)
- **Failure mode:** Silently drops records on DB error to avoid recursive logging

The handler is wired into the root logger by `configure_logging()` when `DATABASE_URL` is set.

### LLM Traces (Callbacks)

LLM traces flow through LangChain's callback system:

```
LLM invocation
    |
    v
LangChain callback dispatcher
    |
    +---> FileCallbackHandler --> data/logs/telemetry.log
    |
    +---> PostgresCallbackHandler --> INSERT INTO llm_traces
```

The `PostgresCallbackHandler` (defined in `src/telemetry.py`) implements three LangChain callback methods:
- `on_llm_start` -- Records the start time, model, and provider
- `on_llm_end` -- Computes latency, extracts token usage, estimates cost, and inserts a row
- `on_llm_error` -- Records the error and latency

Cost estimation uses a built-in price table in `src/telemetry.py`:

| Model Prefix | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| `claude-3-5-sonnet` | $3.00 | $15.00 |
| `claude-opus-4` | $15.00 | $75.00 |
| `gpt-4o` | $2.50 | $10.00 |

To add pricing for new models, add entries to the `_PRICES` dictionary in `src/telemetry.py`.

### Benchmark Results

Benchmark results are written by `save_report()` in `benchmark/runner.py`:

```
BenchmarkRunner.run_all()
    |
    v
save_report(report)
    |
    +---> Write JSON file --> benchmark/results/{run_id}.json
    |
    +---> _save_report_to_postgres(report)
              |
              +--> INSERT INTO benchmark_runs (1 row)
              +--> INSERT INTO benchmark_results (N rows, one per task)
```

Duplicate run IDs are handled with `ON CONFLICT (run_id) DO NOTHING`.

### Checkpoints

LangGraph state checkpoints are managed by `PostgresSaver` from `langgraph-checkpoint-postgres`. The checkpointer is selected by `get_checkpointer()` in `src/agent.py` with this priority:

1. `PostgresSaver` -- when `DATABASE_URL` is set
2. `SqliteSaver` -- when `CHECKPOINT_DB` is set (fallback)
3. `MemorySaver` -- final fallback (no persistence)

---

## Querying Telemetry Data

### Connecting to the Database

```bash
# Via Docker Compose
docker compose exec postgres psql -U cline -d cline_telemetry

# Via local psql client
psql postgresql://cline:cline@localhost:5432/cline_telemetry

# Via Python
from src.db import get_connection_pool
pool = get_connection_pool()
with pool.connection() as conn:
    rows = conn.execute("SELECT * FROM llm_traces LIMIT 5").fetchall()
```

### Common Queries

#### Pass rate by model

```sql
SELECT
    br.model,
    COUNT(*) AS total_tasks,
    SUM(CASE WHEN bres.passed THEN 1 ELSE 0 END) AS passed,
    ROUND(AVG(bres.passed::int) * 100, 1) AS pass_rate_pct
FROM benchmark_results bres
JOIN benchmark_runs br USING (run_id)
GROUP BY br.model
ORDER BY pass_rate_pct DESC;
```

#### Average cost per task category

```sql
SELECT
    split_part(task_id, '-', 1) AS category,
    ROUND(AVG(cost_usd)::numeric, 4) AS avg_cost,
    ROUND(AVG(total_tokens)::numeric, 0) AS avg_tokens
FROM benchmark_results
GROUP BY category
ORDER BY avg_cost DESC;
```

#### Token usage trends over time

```sql
SELECT
    date_trunc('day', timestamp) AS day,
    model,
    COUNT(*) AS call_count,
    ROUND(AVG(total_tokens)::numeric, 0) AS avg_tokens,
    ROUND(SUM(cost_usd)::numeric, 4) AS total_cost
FROM llm_traces
GROUP BY day, model
ORDER BY day DESC, total_cost DESC;
```

#### Error rate by LLM provider

```sql
SELECT
    llm_provider,
    COUNT(*) AS total_events,
    COUNT(*) FILTER (WHERE level = 'ERROR') AS errors,
    ROUND(
        COUNT(*) FILTER (WHERE level = 'ERROR') * 100.0 / NULLIF(COUNT(*), 0), 1
    ) AS error_rate_pct
FROM agent_events
GROUP BY llm_provider;
```

#### Most-used tools

```sql
SELECT
    tool_name,
    COUNT(*) AS invocations
FROM agent_events
WHERE event_type = 'tool_call' AND tool_name IS NOT NULL
GROUP BY tool_name
ORDER BY invocations DESC
LIMIT 20;
```

#### LLM latency percentiles

```sql
SELECT
    model,
    COUNT(*) AS calls,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_ms
FROM llm_traces
WHERE error IS NULL
GROUP BY model;
```

#### Session replay -- all events for a specific session

```sql
SELECT timestamp, level, event_type, tool_name, message
FROM agent_events
WHERE session_id = 'your-session-uuid-here'
ORDER BY timestamp;
```

#### Failed benchmark tasks with error details

```sql
SELECT
    br.model,
    bres.task_id,
    bres.error,
    bres.wall_clock_seconds,
    bres.total_tokens
FROM benchmark_results bres
JOIN benchmark_runs br USING (run_id)
WHERE bres.passed = FALSE AND bres.error IS NOT NULL
ORDER BY br.timestamp DESC
LIMIT 20;
```

#### Cost breakdown by session

```sql
SELECT
    session_id,
    model,
    COUNT(*) AS calls,
    SUM(total_tokens) AS total_tokens,
    ROUND(SUM(cost_usd)::numeric, 4) AS total_cost
FROM llm_traces
WHERE session_id IS NOT NULL
GROUP BY session_id, model
ORDER BY total_cost DESC
LIMIT 20;
```

---

## Migration from Flat Files

A one-time migration script imports existing historical data into PostgreSQL.

### Running the Migration

```bash
# Ensure PostgreSQL is running
docker compose up -d postgres

# Run the migration
DATABASE_URL=postgresql://cline:cline@localhost:5432/cline_telemetry \
    python -m scripts.migrate_to_postgres
```

### What Gets Migrated

| Source | Destination | Dedup Strategy |
|--------|-------------|----------------|
| `benchmark/results/*.json` | `benchmark_runs` + `benchmark_results` | Skip if `run_id` already exists |
| `data/logs/agent.jsonl` | `agent_events` | None (appends all rows) |

### Idempotency

The migration script is safe to run multiple times:
- Benchmark results use `run_id` existence checks to skip duplicates
- Agent events are appended (run only once, or truncate `agent_events` before re-running)

### Example Output

```
=== Migrating benchmark results ===
  Imported run-20260321T070000Z-4e983d0c.json (5 tasks)
  Imported run-20260321T180103Z-475b1006.json (5 tasks)
  Skipping run-20260321T070000Z-4e983d0c.json (already imported)
  Total imported: 2 reports

=== Migrating agent event logs ===
  Total imported: 342 events

Migration complete.
```

---

## Dashboard Integration

PostgreSQL is natively supported by all major dashboard tools. No plugins or custom connectors are needed.

### Grafana

1. Add a PostgreSQL data source pointing to `localhost:5432`, database `cline_telemetry`, user `cline`
2. Create panels using the queries from [Common Queries](#common-queries)
3. Recommended panels:
   - **Pass rate over time** -- Time series from `benchmark_results` joined with `benchmark_runs`
   - **Token usage heatmap** -- From `llm_traces` grouped by model and hour
   - **Error rate gauge** -- From `agent_events` filtered by `level = 'ERROR'`
   - **Cost accumulator** -- Running sum of `cost_usd` from `llm_traces`
   - **Latency percentiles** -- Using `PERCENTILE_CONT` on `llm_traces.latency_ms`

### Metabase

1. Connect to the PostgreSQL database in Admin > Databases
2. All four tables are auto-discovered and available for questions/dashboards
3. Use the native question builder or write custom SQL

### Superset / Apache Preset

1. Add a SQLAlchemy connection string: `postgresql://cline:cline@localhost:5432/cline_telemetry`
2. Import tables as datasets
3. Build charts and dashboards

---

## Python API Reference

### `src/db` Module

```python
from src.db import get_db_url, get_connection_pool

# Get the DATABASE_URL (or None if not set)
url = get_db_url()

# Get a shared connection pool (creates tables on first call)
pool = get_connection_pool()  # Returns None if DATABASE_URL is not set

# Use the pool directly
if pool:
    with pool.connection() as conn:
        rows = conn.execute("SELECT COUNT(*) FROM agent_events").fetchone()
        print(f"Total events: {rows[0]}")
```

**Functions:**

| Function | Returns | Description |
|----------|---------|-------------|
| `get_db_url()` | `str \| None` | Returns the `DATABASE_URL` environment variable |
| `get_connection_pool()` | `ConnectionPool \| None` | Returns a shared, thread-safe connection pool. Creates schema on first call. Returns `None` if `DATABASE_URL` is not set or connection fails. |

The connection pool is configured with:
- `min_size=1` -- One connection always kept alive
- `max_size=5` -- Up to 5 concurrent connections

### `src/telemetry` Module

```python
from src.telemetry import PostgresCallbackHandler

# Create a handler for LangChain callbacks
handler = PostgresCallbackHandler(pool, session_id="your-uuid")

# Pass it to LangChain as a callback
agent.invoke(input, config={"callbacks": [handler]})
```

### `src/logging_config` Module

```python
from src.logging_config import PostgresLogHandler, configure_logging

# Option 1: Use configure_logging() — auto-wires PostgreSQL when DATABASE_URL is set
configure_logging(level="INFO", json_output=True)

# Option 2: Manual handler creation
from src.db import get_connection_pool
pool = get_connection_pool()
handler = PostgresLogHandler(pool)
logging.getLogger().addHandler(handler)
```

---

## Backward Compatibility

The telemetry database is fully opt-in. The system behavior depends solely on whether `DATABASE_URL` is set:

| `DATABASE_URL` | Checkpoints | Logs | Telemetry | Benchmarks |
|-----------------|-------------|------|-----------|------------|
| **Set** | PostgreSQL (`PostgresSaver`) | JSONL file + PostgreSQL (dual-write) | `.log` file + PostgreSQL (dual-write) | JSON file + PostgreSQL (dual-write) |
| **Not set** | SQLite or MemorySaver | JSONL file only | `.log` file only | JSON file only |

No code changes are required to run without PostgreSQL. The flat-file paths are always written regardless of database availability.

---

## Troubleshooting

### "Failed to create PostgreSQL connection pool"

**Cause:** The application cannot connect to PostgreSQL.

**Solutions:**
1. Verify PostgreSQL is running: `docker compose ps postgres`
2. Check the connection string: `psql $DATABASE_URL -c 'SELECT 1'`
3. Ensure the database exists: `docker compose exec postgres psql -U cline -l`
4. Check firewall/network if using a remote PostgreSQL instance

### Tables are empty after running the agent

**Cause:** The `PostgresLogHandler` buffers events and flushes every 5 seconds or every 50 records.

**Solutions:**
1. Wait at least 5 seconds after agent activity
2. Check the handler is attached: look for `"Telemetry also logging to PostgreSQL"` in stdout
3. Verify `DATABASE_URL` is set in the agent's environment

### Migration script shows "DATABASE_URL not set"

**Solution:** Pass the variable explicitly:

```bash
DATABASE_URL=postgresql://cline:cline@localhost:5432/cline_telemetry \
    python -m scripts.migrate_to_postgres
```

### Duplicate rows in `agent_events` after re-running migration

**Cause:** The agent events migration appends without deduplication.

**Solution:** Truncate before re-running:

```sql
TRUNCATE agent_events;
```

Then run the migration again.

### PostgreSQL is down but the agent keeps running

**Expected behavior.** The dual-write design means:
- Log events are silently dropped by `PostgresLogHandler` (no crash)
- LLM traces fail silently with a debug-level log
- Benchmark results log a warning but the JSON file is still written
- All flat-file logging continues unaffected

When PostgreSQL recovers, new data flows into it normally. Previously dropped events during the outage are not recovered (they remain in the flat files only).

### Checking table sizes and row counts

```sql
SELECT
    relname AS table,
    n_live_tup AS row_count,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```
