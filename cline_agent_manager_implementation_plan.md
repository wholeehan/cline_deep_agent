# Cline CLI Agent Manager — Implementation Plan

> Built on LangChain Deep Agents SDK · `deepagents` + LangGraph

## Overview

| | |
|---|---|
| **Stack** | deepagents, LangGraph, langchain-anthropic, langchain-ollama, pexpect |
| **Agent pattern** | `create_deep_agent()` — single ReAct loop, LangGraph runtime |
| **Subagent** | `cline-executor` — spawned via `task()`, context-isolated |
| **Backend** | `CompositeBackend`: StateBackend / StoreBackend / FilesystemBackend |
| **HITL** | `interrupt_on` + `interrupt()` + `MemorySaver` checkpointer |
| **Skills** | task_decomposition, qa_verification, decision_policy, cline_qa |
| **LLM providers** | `claude-3-5-sonnet-20241022` (Anthropic) · `qwen3-coder:latest` (Ollama local) |
| **Total phases** | 10 · **Total tasks: 51** |

## LLM Provider Strategy

The system supports two interchangeable LLM backends selected via the `LLM_PROVIDER` environment variable. The provider abstraction is implemented once in `src/llm.py` and consumed everywhere — agent manager, Cline subagent, and tools — so switching is a one-line config change.

```python
# src/llm.py
import os
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3-coder:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        temperature=temperature,
    )
```

**Provider characteristics to account for in design:**

| Concern | Claude 3.5 Sonnet | qwen3-coder via Ollama |
|---|---|---|
| Tool calling | Native, reliable | Supported; test all tool schemas |
| Context window | 200k tokens | Model-dependent; cap at 32k to be safe |
| Latency | API round-trip (~1–3s) | Local inference (varies by hardware) |
| Auth | `ANTHROPIC_API_KEY` required | No key; Ollama must be running locally |
| Streaming | Full streaming support | Full streaming support |
| Cost | Per-token billing | Free / local compute |

---

## Tag Legend

| Tag | Meaning |
|---|---|
| `infra` | Project scaffold, tooling, CI |
| `llm` | LLM provider abstraction & switching |
| `core` | DeepAgents agent setup |
| `bridge` | Cline PTY bridge & tools |
| `skill` | SKILL.md files |
| `hitl` | Human-in-the-loop flows |
| `memory` | State, backends, persistence |
| `test` | Verification & integration tests |
| `deploy` | Observability, Docker, CI/CD |

---

## Phase 1 — Project scaffold & environment

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **Initialise repo** — pyproject.toml, .gitignore, pre-commit hooks | `git init && pip install -e '.[dev]'` exits 0 | `infra` |
| 2 | **Pin dependencies** — deepagents, langgraph, langchain-anthropic, langchain-ollama, pytest, ruff, mypy | `pip install -e '.[dev]' && python -c 'import deepagents; import langchain_ollama'` exits 0 | `infra` |
| 3 | **Create .env.example** — `LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` | All required keys documented; dotenv loads without error; missing `ANTHROPIC_API_KEY` does not crash when `LLM_PROVIDER=ollama` | `infra` |
| 4 | **Configure LangSmith tracing** — set `LANGSMITH_TRACING=true` in CI | Smoke-test agent call appears in LangSmith dashboard | `infra` |
| 5 | **Ruff + mypy baseline** — ruff check + mypy --strict pass on empty src/ | Pre-commit hooks pass on first commit | `infra` |

---

## Phase 2 — LLM provider abstraction

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **Implement `src/llm.py`** — `get_llm()` factory returning `ChatAnthropic` or `ChatOllama` based on `LLM_PROVIDER` env var | `get_llm()` with `LLM_PROVIDER=anthropic` returns `ChatAnthropic`; with `LLM_PROVIDER=ollama` returns `ChatOllama` | `llm` |
| 2 | **Ollama connectivity check** — on startup, if `LLM_PROVIDER=ollama`, ping `{OLLAMA_BASE_URL}/api/tags` and verify model is present | Startup raises `OllamaUnavailableError` with actionable message if Ollama not running or model missing | `llm` |
| 3 | **Tool schema compatibility test** — run all custom tool schemas through both providers | All `@tool`-decorated functions invoke correctly with both `ChatAnthropic` and `ChatOllama`; no schema rejection errors | `llm` |
| 4 | **Context window guard** — when `LLM_PROVIDER=ollama`, enforce 32k token soft cap on agent context | Warning logged and oldest messages trimmed when context exceeds 32k tokens with Ollama provider | `llm` |
| 5 | **Provider smoke tests** — `pytest tests/llm/` with both providers | Both providers return non-empty responses to a "say hello" prompt; tests pass with appropriate env vars set | `test` |

---

## Phase 3 — Cline CLI bridge

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **Spawn Cline in PTY** — use pexpect or node-pty subprocess wrapper | `cline_bridge.spawn()` starts process; pid is non-zero | `bridge` |
| 2 | **Stream parser — progress events** — regex/classifier for narration vs. question vs. approval | Unit test: 10 captured Cline stdout samples classified correctly | `bridge` |
| 3 | **Stream parser — question events** — detect lines ending in `?` followed by prompt char | Unit test: question samples return `EventType.QUESTION` | `bridge` |
| 4 | **Stream parser — approval events** — detect "Do you want to…" / "Should I proceed…" patterns | Unit test: approval samples return `EventType.APPROVAL` | `bridge` |
| 5 | **Response injector** — write string to PTY stdin at correct moment | Injected `"yes\n"` unblocks a paused Cline session in integration test | `bridge` |
| 6 | **PTY session replay log** — append-only timestamped log of all stdin/stdout chunks | Log file written; replay reconstructs session byte-for-byte | `bridge` |
| 7 | **Bridge unit test suite** — pytest tests/bridge/ covering all event types + injector | `pytest tests/bridge/` — 100% pass, ≥80% line coverage | `test` |

---

## Phase 4 — DeepAgents core setup

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **`create_deep_agent()` skeleton** — passes `get_llm()` as the `model` argument; no tools yet | `agent.invoke({messages:[...]})` returns without error under both `LLM_PROVIDER` values | `core` |
| 2 | **CompositeBackend wiring** — `/workspace/` → StateBackend, `/memories/` → StoreBackend, `/project/` → FilesystemBackend(virtual_mode=True) | `write_file('/workspace/x.txt')` and `read_file('/workspace/x.txt')` round-trip correctly | `core` |
| 3 | **MemorySaver checkpointer** — required for `interrupt_on` and HITL | Two invocations with same `thread_id` share state | `core` |
| 4 | **`write_todos` smoke test** — agent creates a todo list from a dummy task description | `ls('/workspace/')` shows `todos.md` after invocation | `core` |
| 5 | **`report_progress` custom tool** — `@tool` that appends to a progress log and returns to user | Tool call writes line to `/workspace/progress.log`; log is readable | `core` |

---

## Phase 5 — Skills (SKILL.md files)

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **task_decomposition SKILL.md** — CoT prompt: break task into subtasks with title, context, acceptance criterion, complexity, depends_on | Agent given a vague task produces structured subtask list in `/workspace/todos.md` | `skill` |
| 2 | **qa_verification SKILL.md** — instructions to evaluate subtask output against acceptance criterion; emit pass/fail | Agent invoked with passing criterion returns `"verified"`; failing returns `"failed"` | `skill` |
| 3 | **decision_policy SKILL.md** — rules: file writes = auto-approve, external HTTP = escalate, delete = always escalate | Agent classifies 10 action samples correctly against policy table | `skill` |
| 4 | **cline_qa SKILL.md** — instructions for answering Cline questions from shared context | Subagent given a Cline question and context produces a grounded answer without asking user | `skill` |
| 5 | **Skills loading integration test** — verify progressive disclosure; skills only inject when relevant | Irrelevant skill descriptions do NOT appear in agent context on unrelated prompt | `test` |

---

## Phase 6 — Cline subagent definition

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **Define `cline-executor` subagent dict** — passes `get_llm()` as `model`; tools=[dispatch_subtask, answer_question, approve_cline_action], skills=['/skills/cline_qa/'] | `create_deep_agent(subagents=[cline_subagent])` initialises without error under both providers | `bridge` |
| 2 | **`dispatch_subtask` tool** — calls `cline_bridge.send_task(subtask)`; returns PTY session summary | Integration test: sends a simple `"echo hello"` task; returns stdout summary | `bridge` |
| 3 | **`answer_question` tool** — reads shared context + question; calls `get_llm()`; injects answer to PTY stdin | Unit test with mocked PTY: injected answer unblocks stream parser under both providers | `bridge` |
| 4 | **`approve_cline_action` tool with `interrupt()`** — calls `interrupt({type:'cline_approval', action:...})`; resumes with approved/rejected | Integration test: agent pauses; `Command(resume={'approved':True})` continues execution | `hitl` |
| 5 | **`interrupt_on` subagent config** — approve_cline_action: approve/edit/reject; dispatch_subtask: approve/reject; answer_question: False | `approve_cline_action` call triggers `__interrupt__` in result; `answer_question` does not | `hitl` |
| 6 | **Context isolation test** — main agent context should not contain Cline PTY noise after subagent completes | Main agent message list has ≤3 new messages after a full Cline subtask execution | `test` |

---

## Phase 7 — Human-in-the-loop flows

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **Interrupt detection helper** — `result.get('__interrupt__')` check + `action_requests` extraction | Helper returns structured list of pending decisions from interrupt payload | `hitl` |
| 2 | **Approve flow** — `Command(resume={'decisions':[{'type':'approve'}]})` | Agent resumes and Cline action executes; PTY log shows action ran | `hitl` |
| 3 | **Edit flow** — `Command(resume={'decisions':[{'type':'edit','edited_action':{...}}]})` | Edited args (not original) reach Cline stdin; PTY log confirms | `hitl` |
| 4 | **Reject flow** — `Command(resume={'decisions':[{'type':'reject'}]})` | Tool call skipped; agent continues to next subtask; PTY log shows action did NOT run | `hitl` |
| 5 | **Multi-interrupt batching test** — two `approve_cline_action` calls in same reasoning step | Single `__interrupt__` contains two `action_requests`; both resolved with one `Command` | `hitl` |
| 6 | **`thread_id` persistence test** — interrupt on thread A; resume on thread A; reject on thread B (should fail) | Resume with correct `thread_id` succeeds; wrong `thread_id` raises LangGraph error | `test` |

---

## Phase 8 — Memory & state

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **Task state schema** — Pydantic model: `SubTask(id, title, context, criterion, complexity, depends_on, status, escalate_flag)` | `SubTask.model_validate({...})` parses correctly; invalid schema raises `ValidationError` | `memory` |
| 2 | **`write_todos` → state store mapping** — serialise `SubTask` list to `/workspace/todos.md` on every plan update | `todos.md` present and parseable after each `write_todos` call | `memory` |
| 3 | **Subtask status transitions** — pending → dispatched → verified \| failed | State machine unit test covers all valid and invalid transitions | `memory` |
| 4 | **StoreBackend long-term memory** — agent saves project conventions to `/memories/project.md` across threads | New thread reads `/memories/project.md` and incorporates content into plan | `memory` |
| 5 | **Cline session log backend** — append-only PTY log mapped to FilesystemBackend under `/logs/` | Log survives agent restart; replay produces identical byte sequence | `memory` |

---

## Phase 9 — End-to-end integration

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **Happy-path E2E test** — user submits "create a hello-world Flask app"; agent decomposes, dispatches, verifies, reports done | `tests/e2e/test_happy_path.py` passes; output directory contains working Flask app | `test` |
| 2 | **Clarification escalation E2E** — ambiguous task triggers user question; answer resumes plan | `__interrupt__` raised with `type='clarification'`; resume with answer produces valid plan | `test` |
| 3 | **Cline crash recovery test** — kill Cline process mid-subtask; bridge detects exit code; agent marks subtask failed | `SubTask.status == 'failed'`; agent does not hang; user receives failure report | `test` |
| 4 | **Multi-subtask dependency ordering** — subtask B with `depends_on=[A.id]` not dispatched until `A.status=='verified'` | Execution log shows B starts only after A verified timestamp | `test` |
| 5 | **Context window budget test** — after 5 subtasks, main agent context must stay under provider limit | Claude: token count < 80,000; Ollama/qwen3-coder: token count < 32,000 — verified via LangSmith trace | `test` |
| 6 | **Provider parity E2E test** — run happy-path E2E with both `LLM_PROVIDER=anthropic` and `LLM_PROVIDER=ollama` | Both pass; any provider-specific failure is caught and logged with provider name in error message | `test` |

---

## Phase 10 — Deploy & observability

| # | Task | Acceptance criterion | Tag |
|---|---|---|---|
| 1 | **LangSmith tracing validation** — verify subagent names appear as `lc_agent_name` in metadata; tag traces with `llm_provider` | LangSmith trace shows `"cline-executor"` metadata and correct provider tag on all subagent tool calls | `deploy` |
| 2 | **Structured logging** — JSON logs for every interrupt, approval, rejection, subtask status change; include `llm_provider` field | Log parser extracts events correctly; no raw PTY noise in structured logs | `deploy` |
| 3 | **Docker Compose setup** — services: agent-manager, ollama (with qwen3-coder pre-pulled), redis (StoreBackend), langsmith-proxy | `docker compose up` starts all services; E2E test passes with `LLM_PROVIDER=ollama` against containerised stack | `deploy` |
| 4 | **CI pipeline** — GitHub Actions: ruff, mypy, pytest unit (both providers mocked), pytest e2e (Anthropic only in CI) | All checks green on clean checkout; Ollama tests skipped gracefully if `OLLAMA_BASE_URL` not set | `deploy` |
| 5 | **README with quickstart** — two paths: "Cloud (Claude)" and "Local (Ollama)"; clone → .env → docker compose up → send first task | Colleague with no prior context completes either quickstart path in < 15 minutes | `deploy` |
