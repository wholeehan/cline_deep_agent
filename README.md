# Cline Deep Agent

An autonomous agent manager that orchestrates the [Cline CLI](https://github.com/cline/cline) through a PTY bridge, built on the [LangChain Deep Agents SDK](https://github.com/langchain-ai/deepagents) and LangGraph.

The manager decomposes user tasks into subtasks, dispatches them to a **cline-executor** subagent, handles Cline's questions and approval prompts autonomously, verifies results against acceptance criteria, and reports progress — all with human-in-the-loop escalation for risky actions.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   User / CLI                         │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              Agent Manager (LangGraph)               │
│                                                      │
│  Skills: task_decomposition, qa_verification,        │
│          decision_policy, cline_qa                   │
│                                                      │
│  Tools:  report_progress, write_todos, task()        │
│                                                      │
│  Backend: CompositeBackend                           │
│    /workspace/ → StateBackend (ephemeral)            │
│    /memories/  → StoreBackend (persistent)           │
│    /project/   → FilesystemBackend                   │
└──────────────┬───────────────────────────────────────┘
               │ task()
               ▼
┌──────────────────────────────────────────────────────┐
│           cline-executor Subagent                    │
│                                                      │
│  Tools:  dispatch_subtask                            │
│          answer_question                             │
│          approve_cline_action (HITL interrupt)        │
└──────────────┬───────────────────────────────────────┘
               │ PTY bridge
               ▼
┌──────────────────────────────────────────────────────┐
│              Cline CLI (pexpect PTY)                 │
│                                                      │
│  Stream parser: progress / question / approval       │
│  Response injector: stdin writes                     │
│  Session log: append-only replay                     │
└──────────────────────────────────────────────────────┘
```

## Features

- **Dual LLM providers** — Claude 3.5 Sonnet (Anthropic API) or qwen3-coder (local via Ollama / vLLM), switchable with a single env var
- **Task decomposition** — breaks vague requests into structured subtasks with acceptance criteria, complexity, and dependency ordering
- **Cline PTY bridge** — spawns Cline in a pseudo-terminal, classifies stdout into event types (progress, question, approval, error, command), injects responses
- **Human-in-the-loop** — auto-approves safe actions (file reads/writes, local commands) and escalates risky ones (HTTP, deletes, git push) via LangGraph interrupts
- **QA verification** — evaluates subtask output against acceptance criteria before marking done
- **State machine** — enforces valid subtask transitions: `pending → dispatched → verified | failed` (with retry from `failed`)
- **Structured logging** — JSON logs with `llm_provider`, event type, tool name, and decision metadata
- **Telemetry database** — Optional PostgreSQL backend for queryable agent events, LLM traces, and benchmark results with dual-write to flat files (see [Telemetry Database](#telemetry-database))
- **Session replay** — timestamped append-only log of all PTY stdin/stdout for debugging
- **Context window guard** — automatic message trimming when Ollama context exceeds 32k tokens

## Quickstart

### Prerequisites

- Python 3.12+
- [Cline CLI](https://github.com/cline/cline) installed and available on `$PATH`

### Path A: Cloud (Claude)

```bash
# Clone and install
git clone https://github.com/<your-user>/cline_deep_agent.git
cd cline_deep_agent
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'

# Configure
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY

# Run
cline-agent
```

### Path B: Local (Ollama)

```bash
# Clone and install
git clone https://github.com/<your-user>/cline_deep_agent.git
cd cline_deep_agent
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'

# Start Ollama and pull the model
ollama serve &
ollama pull qwen3-coder:latest

# Configure
cp .env.example .env
# Edit .env: set LLM_PROVIDER=ollama

# Run
cline-agent
```

### Path C: Docker Compose

```bash
git clone https://github.com/<your-user>/cline_deep_agent.git
cd cline_deep_agent
cp .env.example .env

# Starts agent-manager, ollama, postgres, grafana, and redis
docker compose up
```

## Project Structure

```
cline_deep_agent/
├── src/
│   ├── llm.py              # LLM provider factory (Anthropic / Ollama / vLLM)
│   ├── bridge.py            # Cline PTY bridge — spawn, parse, inject
│   ├── agent.py             # Agent manager setup with CompositeBackend
│   ├── subagent.py          # cline-executor subagent + tools
│   ├── hitl.py              # Human-in-the-loop helpers (interrupt/resume)
│   ├── models.py            # SubTask/TaskPlan Pydantic models + state machine
│   ├── logging_config.py    # Structured JSON logging + PostgreSQL log handler
│   ├── db.py                # PostgreSQL connection pool + schema init
│   ├── telemetry.py         # LLM trace callback handler for PostgreSQL
│   └── cli.py               # Interactive CLI entry point
├── skills/
│   ├── task_decomposition/  # Break tasks into structured subtasks
│   ├── qa_verification/     # Verify outputs against acceptance criteria
│   ├── decision_policy/     # Auto-approve vs. escalate action rules
│   └── cline_qa/            # Answer Cline questions from context
├── benchmark/
│   ├── __main__.py          # CLI: python -m benchmark run|score|list|leaderboard
│   ├── config.py            # Pydantic models for tasks and results
│   ├── runner.py            # Task orchestration (tmp dir isolation, pytest)
│   ├── scorer.py            # pass@k, cost, timing metrics
│   ├── leaderboard.py       # Markdown comparison tables
│   ├── adapters/
│   │   ├── base.py          # AgentAdapter ABC
│   │   └── cline_deep.py    # Adapter for this project's agent
│   ├── tasks/               # 5 benchmark tasks (bugfix, feature, refactor, multifile)
│   └── results/             # Run output (gitignored)
├── tests/
│   ├── llm/                 # LLM provider tests (11 tests)
│   ├── bridge/              # PTY bridge tests (36 tests)
│   ├── core/                # Agent + subagent tests (31 tests)
│   ├── skills/              # SKILL.md validation (22 tests)
│   ├── hitl/                # HITL flow tests (13 tests)
│   ├── memory/              # State model tests (19 tests)
│   ├── e2e/                 # End-to-end integration (9 tests)
│   └── benchmark/           # Benchmark suite tests (40 tests)
├── grafana/
│   ├── provisioning/           # Auto-configures data source + dashboard loader
│   └── dashboards/             # Pre-built 8-panel telemetry dashboard JSON
├── scripts/
│   └── migrate_to_postgres.py  # One-time migration of flat files → PostgreSQL
├── telemetry.md             # Detailed telemetry database documentation
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .pre-commit-config.yaml
└── .github/workflows/ci.yml
```

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `"anthropic"`, `"ollama"`, or `"vllm"` |
| `ANTHROPIC_API_KEY` | — | Required when using Anthropic |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Anthropic model ID |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3-coder:latest` | Ollama model name |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `VLLM_MODEL` | `Qwen/Qwen3-Coder-Next` | vLLM model name |
| `DATABASE_URL` | — | PostgreSQL connection string for telemetry (see [Telemetry Database](#telemetry-database)) |
| `CHECKPOINT_DB` | `./data/checkpoints.db` | SQLite checkpoint path (fallback when `DATABASE_URL` not set) |
| `LOG_FILE` | `./data/logs/agent.jsonl` | Structured log file path |
| `TELEMETRY_FILE` | `./data/logs/telemetry.log` | LangChain telemetry log path |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith trace collection |
| `LANGSMITH_API_KEY` | — | LangSmith API key |
| `LANGSMITH_PROJECT` | `cline-agent-manager` | LangSmith project name |

## Decision Policy

The agent auto-approves safe actions and escalates risky ones:

| Action | Decision |
|---|---|
| File read/write/edit | Auto-approve |
| Local commands (`npm install`, `pytest`, `ruff`) | Auto-approve |
| Git local (`add`, `commit`, `branch`) | Auto-approve |
| Git remote (`push`, `pull`) | Escalate |
| External HTTP requests | Escalate |
| File/directory delete | Always escalate |
| System modification | Always escalate |
| Database destructive ops | Always escalate |

## Telemetry Database

The project includes an optional PostgreSQL-based telemetry system that stores agent events, LLM traces, and benchmark results in a queryable database. When enabled, all telemetry is **dual-written** to both PostgreSQL and flat files, ensuring no data loss if the database is temporarily unavailable.

### Quick Setup

```bash
# Start PostgreSQL via Docker Compose
docker compose up -d postgres

# Set the connection string
echo 'DATABASE_URL=postgresql://cline:cline@localhost:5432/cline_telemetry' >> .env

# Run the agent — tables are created automatically on first startup
cline-agent
```

### What Gets Stored

| Table | Contents | Replaces |
|-------|----------|----------|
| `agent_events` | Structured log events (tool calls, decisions, errors) | `data/logs/agent.jsonl` |
| `llm_traces` | Per-call LLM telemetry (tokens, latency, cost) | `data/logs/telemetry.log` |
| `benchmark_runs` | Benchmark run metadata | `benchmark/results/*.json` |
| `benchmark_results` | Per-task pass/fail, tokens, cost | `benchmark/results/*.json` |
| `checkpoints` | LangGraph agent state (auto-managed) | `data/checkpoints.db` |

### Example Queries

```sql
-- Pass rate by model
SELECT model, ROUND(AVG(passed::int) * 100, 1) AS pass_rate
FROM benchmark_results JOIN benchmark_runs USING (run_id)
GROUP BY model;

-- Token usage trends
SELECT date_trunc('day', timestamp) AS day, model, AVG(total_tokens)::int AS avg_tokens
FROM llm_traces GROUP BY 1, 2 ORDER BY 1 DESC;

-- Most-used tools
SELECT tool_name, COUNT(*) AS calls
FROM agent_events WHERE event_type = 'tool_call'
GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
```

### Migrating Historical Data

```bash
DATABASE_URL=postgresql://cline:cline@localhost:5432/cline_telemetry \
    python -m scripts.migrate_to_postgres
```

This imports existing `benchmark/results/*.json` and `data/logs/agent.jsonl` into PostgreSQL. The script is idempotent — safe to run multiple times.

### Grafana Dashboard

A pre-built 8-panel Grafana dashboard is included and auto-provisioned via Docker Compose:

```bash
docker compose up -d postgres grafana
# Open http://localhost:3000 — login: admin / admin
```

Panels include: LLM call volume, cost accumulation, token usage by model, latency percentiles (p50/p95/p99), error rate, benchmark pass rate, most-used tools, and recent errors.

For full schema details, query examples, Python API reference, and troubleshooting, see [telemetry.md](telemetry.md).

## Development

```bash
# Install with dev dependencies
pip install -e '.[dev]'

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Format
ruff format src/ tests/
```

### Test Suite

196 tests across 8 test modules:

- **tests/llm/** — Provider factory, Ollama connectivity, context trimming
- **tests/bridge/** — Stream classifier (20+ Cline output samples), session log, PTY spawn/inject
- **tests/core/** — Agent creation, subagent definition, tool behavior, interrupt config
- **tests/skills/** — SKILL.md frontmatter validation, policy classification
- **tests/hitl/** — Interrupt extraction, approve/reject/edit/batch commands
- **tests/memory/** — SubTask validation, state transitions, TaskPlan dependency ordering
- **tests/e2e/** — Full lifecycle, crash recovery, provider parity
- **tests/benchmark/** — Config models, scorer, runner discovery, leaderboard generation

## Benchmark Suite

A built-in benchmark harness measures the agent's coding capabilities using isolated task environments, automated test validation, and structured metrics.

### Available Tasks

| ID | Category | Title | Difficulty |
|----|----------|-------|------------|
| bugfix-001-off-by-one | bugfix | Fix off-by-one error in pagination | easy |
| bugfix-002-null-check | bugfix | Fix missing null check causing crash | easy |
| feature-001-add-endpoint | feature | Add REST endpoint with validation | easy |
| refactor-001-extract-class | refactor | Extract class from god-object module | easy |
| multifile-001-api-migration | multifile | Migrate API v1→v2 across 3 files | easy |

### Usage

```bash
# List available tasks
python -m benchmark list

# Run a single task with Ollama
python -m benchmark run --provider ollama --model gpt-oss:20b --tasks bugfix-001

# Run all tasks with 3 repetitions (for pass@3 metrics)
python -m benchmark run --tasks all --repetitions 3

# Run with Anthropic
python -m benchmark run --provider anthropic --model claude-3-5-sonnet-20241022 --tasks all

# Recompute metrics from existing results
python -m benchmark score results/run-*.json

# Generate comparison leaderboard
python -m benchmark leaderboard benchmark/results/
```

### Metrics

- **pass_rate** — fraction of tasks passing all tests
- **pass@k** — unbiased probability of at least one success in k attempts
- **avg_tokens** — mean token consumption per task
- **avg_cost** — estimated USD cost per task
- **by_category** — pass rate breakdown by task category

### Adding Tasks

Copy `benchmark/tasks/_template/` and fill in:
- `task.toml` — task metadata (id, category, difficulty, budget, timeout)
- `instruction.md` — natural language task description for the agent
- `setup.sh` — install task-specific dependencies
- `workspace/` — initial codebase (broken/incomplete code)
- `tests/test_solution.py` — pytest suite that validates the fix

## How It Works

1. **User submits a task** via the CLI (e.g., "create a hello-world Flask app")
2. **Agent Manager** uses the `task_decomposition` skill to break it into subtasks with acceptance criteria and dependencies
3. **Ready subtasks** (those with all dependencies met) are dispatched to the `cline-executor` subagent via `task()`
4. **cline-executor** spawns Cline in a PTY and sends the subtask
5. **Stream parser** classifies Cline output — questions are answered via `answer_question`, approvals are evaluated against the `decision_policy`
6. **Risky actions** trigger a LangGraph `interrupt()` — the user approves, edits, or rejects
7. **QA verification** checks subtask output against its acceptance criterion
8. **State machine** transitions: `pending → dispatched → verified | failed`
9. **Failed subtasks** can be retried; the plan continues until all subtasks are verified

## License

MIT
