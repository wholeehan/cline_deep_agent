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

- **Dual LLM providers** — Claude 3.5 Sonnet (Anthropic API) or qwen3-coder (local via Ollama), switchable with a single env var
- **Task decomposition** — breaks vague requests into structured subtasks with acceptance criteria, complexity, and dependency ordering
- **Cline PTY bridge** — spawns Cline in a pseudo-terminal, classifies stdout into event types (progress, question, approval, error, command), injects responses
- **Human-in-the-loop** — auto-approves safe actions (file reads/writes, local commands) and escalates risky ones (HTTP, deletes, git push) via LangGraph interrupts
- **QA verification** — evaluates subtask output against acceptance criteria before marking done
- **State machine** — enforces valid subtask transitions: `pending → dispatched → verified | failed` (with retry from `failed`)
- **Structured logging** — JSON logs with `llm_provider`, event type, tool name, and decision metadata
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

# Starts agent-manager, ollama (with qwen3-coder), and redis
docker compose up
```

## Project Structure

```
cline_deep_agent/
├── src/
│   ├── llm.py              # LLM provider factory (Anthropic / Ollama)
│   ├── bridge.py            # Cline PTY bridge — spawn, parse, inject
│   ├── agent.py             # Agent manager setup with CompositeBackend
│   ├── subagent.py          # cline-executor subagent + tools
│   ├── hitl.py              # Human-in-the-loop helpers (interrupt/resume)
│   ├── models.py            # SubTask/TaskPlan Pydantic models + state machine
│   ├── logging_config.py    # Structured JSON logging
│   └── cli.py               # Interactive CLI entry point
├── skills/
│   ├── task_decomposition/  # Break tasks into structured subtasks
│   ├── qa_verification/     # Verify outputs against acceptance criteria
│   ├── decision_policy/     # Auto-approve vs. escalate action rules
│   └── cline_qa/            # Answer Cline questions from context
├── tests/
│   ├── llm/                 # LLM provider tests (11 tests)
│   ├── bridge/              # PTY bridge tests (36 tests)
│   ├── core/                # Agent + subagent tests (31 tests)
│   ├── skills/              # SKILL.md validation (22 tests)
│   ├── hitl/                # HITL flow tests (13 tests)
│   ├── memory/              # State model tests (19 tests)
│   └── e2e/                 # End-to-end integration (9 tests)
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
| `LLM_PROVIDER` | `anthropic` | `"anthropic"` or `"ollama"` |
| `ANTHROPIC_API_KEY` | — | Required when using Anthropic |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Anthropic model ID |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3-coder:latest` | Ollama model name |
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

141 tests across 7 test modules:

- **tests/llm/** — Provider factory, Ollama connectivity, context trimming
- **tests/bridge/** — Stream classifier (20+ Cline output samples), session log, PTY spawn/inject
- **tests/core/** — Agent creation, subagent definition, tool behavior, interrupt config
- **tests/skills/** — SKILL.md frontmatter validation, policy classification
- **tests/hitl/** — Interrupt extraction, approve/reject/edit/batch commands
- **tests/memory/** — SubTask validation, state transitions, TaskPlan dependency ordering
- **tests/e2e/** — Full lifecycle, crash recovery, provider parity

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
