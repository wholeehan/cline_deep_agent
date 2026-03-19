# Composite Backend System — Technical Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Backend Protocol](#backend-protocol)
4. [StateBackend — Ephemeral Per-Thread Storage](#statebackend)
5. [StoreBackend — Persistent Cross-Thread Storage](#storebackend)
6. [FilesystemBackend — Direct Filesystem Access](#filesystembackend)
7. [CompositeBackend — Path-Based Routing](#compositebackend)
8. [How Our Project Uses It](#how-our-project-uses-it)
9. [Accessing the System from Your Terminal](#accessing-the-system-from-your-terminal)
10. [Backend Factory Pattern](#backend-factory-pattern)
11. [API Reference](#api-reference)

---

## 1. Overview

The Composite Backend system is the storage layer of the Cline Deep Agent. It provides a unified file-operation interface while routing reads and writes to different storage engines based on path prefixes. This allows the agent to treat all storage — ephemeral state, persistent memory, and the real filesystem — as a single virtual filesystem.

```
Agent sees:
  /workspace/todos.md       →  StateBackend   (ephemeral, in-memory)
  /memories/project.md      →  StoreBackend   (persistent, cross-thread)
  /project/src/main.py      →  FilesystemBackend (actual disk)
  /anything/else            →  StateBackend   (default fallback)
```

The agent doesn't know or care which backend handles a given path. It just calls `read_file("/workspace/todos.md")` or `write_file("/memories/project.md", content)` and the CompositeBackend routes it.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent / Tools                         │
│                                                         │
│   read_file()  write_file()  edit_file()  ls()  grep() │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  CompositeBackend                        │
│                                                         │
│   Routes by longest-prefix match:                       │
│                                                         │
│   /workspace/*  ──→  StateBackend                       │
│   /memories/*   ──→  StoreBackend                       │
│   /project/*    ──→  FilesystemBackend                  │
│   /*  (default) ──→  StateBackend                       │
└──────┬──────────────────┬──────────────────┬────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ StateBackend │  │ StoreBackend │  │ FilesystemBackend│
│              │  │              │  │                  │
│ LangGraph    │  │ LangGraph    │  │ Actual disk      │
│ agent state  │  │ BaseStore    │  │ via os.open()    │
│              │  │              │  │                  │
│ Per-thread   │  │ Cross-thread │  │ Permanent        │
│ Ephemeral    │  │ Persistent   │  │ Persistent       │
└──────────────┘  └──────────────┘  └──────────────────┘
```

---

## 3. Backend Protocol

Every backend implements the `BackendProtocol` interface. This is the contract that the CompositeBackend and the agent tools depend on.

### Methods

| Method | Signature | Description |
|---|---|---|
| `ls_info` | `(path: str) -> list[FileInfo]` | List directory contents non-recursively |
| `read` | `(file_path: str, offset: int = 0, limit: int = 2000) -> str` | Read file with pagination (line numbers included) |
| `write` | `(file_path: str, content: str) -> WriteResult` | Create new file (fails if exists) |
| `edit` | `(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult` | Replace exact strings in file |
| `grep_raw` | `(pattern: str, path: str \| None, glob: str \| None) -> list[GrepMatch] \| str` | Literal text search (not regex) |
| `glob_info` | `(pattern: str, path: str = "/") -> list[FileInfo]` | Find files matching glob pattern |
| `upload_files` | `(files: list[tuple[str, bytes]]) -> list[FileUploadResponse]` | Batch upload |
| `download_files` | `(paths: list[str]) -> list[FileDownloadResponse]` | Batch download |

All methods also have async variants prefixed with `a` (e.g., `aread`, `awrite`).

### File Data Structure

Every backend stores file content in this internal format:

```python
{
    "content": ["line 1", "line 2", "line 3"],   # list of strings (one per line)
    "created_at": "2026-03-19T08:00:00Z",         # ISO timestamp
    "modified_at": "2026-03-19T08:30:00Z",         # ISO timestamp
}
```

### Return Types

All return types are Python objects with attribute access (`.path`, `.error`, etc.):

- **WriteResult**: `.path`, `.error`, `.files_update` — `files_update` is a dict for state backends, `None` for external backends
- **EditResult**: `.path`, `.error`, `.files_update`, `.occurrences` — same pattern, plus count of replacements

Directory listing and search results are returned as **dicts** (accessed via `["key"]`):

- **FileInfo dict**: `{"path", "is_dir", "size", "modified_at"}`
- **GrepMatch dict**: `{"path", "line", "text"}` — `line` is 1-indexed

---

## 4. StateBackend — Ephemeral Per-Thread Storage

### What It Is

StateBackend stores files in the LangGraph agent's in-memory state (`runtime.state["files"]`). Data lives only for the duration of a single conversation thread.

### Characteristics

| Property | Value |
|---|---|
| Persistence | Single thread only — lost when thread ends |
| Storage | `runtime.state["files"]` dict |
| Cross-thread | No |
| Async | Yes (via `asyncio.to_thread`) |
| State sync | Returns `files_update` dict for LangGraph to merge |

### How It Works

```python
from deepagents.backends import StateBackend

backend = StateBackend(runtime)

# Write creates a file in agent state
result = backend.write("/workspace/plan.md", "# My Plan\n- Step 1")
# result.files_update = {"/workspace/plan.md": {"content": [...], ...}}

# Read retrieves from state
content = backend.read("/workspace/plan.md")
# Returns formatted text with line numbers

# Edit performs exact string replacement
result = backend.edit("/workspace/plan.md", "Step 1", "Step 1 (done)")
```

### When To Use

- Temporary files: task plans, progress logs, scratch notes
- Per-conversation working memory
- Any data that should reset between sessions

---

## 5. StoreBackend — Persistent Cross-Thread Storage

### What It Is

StoreBackend uses LangGraph's `BaseStore` for persistent storage that survives across conversation threads. This is the "long-term memory" backend.

### Characteristics

| Property | Value |
|---|---|
| Persistence | Permanent — survives across threads |
| Storage | LangGraph BaseStore (configurable: memory, Redis, etc.) |
| Cross-thread | Yes |
| Async | Native async via `store.aget()` / `store.aput()` |
| State sync | Returns `files_update=None` (already persisted externally) |
| Namespace | Configurable via `NamespaceFactory` for multi-user isolation |

### How It Works

```python
from deepagents.backends import StoreBackend

backend = StoreBackend(runtime)

# Write persists to the external store
result = backend.write("/memories/project.md", "# Project Conventions\n- Use pytest")
# result.files_update = None (already saved to store)

# Read fetches from the store — available in ANY thread
content = backend.read("/memories/project.md")
```

### Namespace Isolation

StoreBackend supports multi-user isolation via namespaces:

```python
from deepagents.backends import StoreBackend, BackendContext

# Namespace factory isolates storage per user
backend = StoreBackend(
    runtime,
    namespace=lambda ctx: ("filesystem", ctx.runtime.context.user_id)
)
```

### When To Use

- Project conventions and preferences
- Cross-session memory (e.g., "this project uses tabs not spaces")
- Shared knowledge between agents

---

## 6. FilesystemBackend — Direct Filesystem Access

### What It Is

FilesystemBackend reads and writes real files on disk. It's the bridge between the agent's virtual filesystem and your actual project directory.

### Characteristics

| Property | Value |
|---|---|
| Persistence | Permanent — actual files on disk |
| Storage | Real filesystem via `os.open()` |
| Cross-thread | N/A (files exist independently) |
| Async | Yes (via `asyncio.to_thread`) |
| State sync | Returns `files_update=None` (already on disk) |
| Security | `O_NOFOLLOW` flag to prevent symlink traversal |

### Virtual Mode

```python
from deepagents.backends import FilesystemBackend

# virtual_mode=True (recommended): paths are sandboxed to root_dir
backend = FilesystemBackend(root_dir="/prj/cline_deep_agent", virtual_mode=True)

# Agent writes to "/project/src/main.py"
# Actual file: /prj/cline_deep_agent/src/main.py
# Blocks "../../../etc/passwd" traversal attempts

# virtual_mode=False (dangerous): absolute paths used as-is
backend = FilesystemBackend(virtual_mode=False)
```

### Search Implementation

- **Primary**: Uses `ripgrep` (`rg --json -F`) if available on the system
- **Fallback**: Python regex search with `wcmatch.glob` for file filtering
- Skips files larger than `max_file_size_mb` (default 10 MB)

### When To Use

- Reading/writing actual project source code
- Accessing config files, test outputs, build artifacts
- CI/CD pipelines where you need real filesystem access

---

## 7. CompositeBackend — Path-Based Routing

### What It Is

CompositeBackend is the top-level backend that routes operations to child backends based on path prefixes. It's the only backend the agent interacts with directly.

### Routing Logic

1. **Longest-prefix match**: Routes sorted by length (longest first)
2. Path is matched against route prefixes
3. The matched prefix is stripped before passing to the child backend
4. If no route matches, the default backend handles the request

```
Agent calls:  read_file("/memories/project.md")
                         │
CompositeBackend checks:  /memories/ → StoreBackend  ✓ match!
                         │
StoreBackend receives:    read("/project.md")   ← prefix stripped
```

### Initialization

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend

composite = CompositeBackend(
    default=StateBackend(runtime),         # fallback for unmatched paths
    routes={
        "/workspace/": StateBackend(runtime),      # ephemeral work area
        "/memories/": StoreBackend(runtime),        # persistent memory
        "/project/": FilesystemBackend(virtual_mode=True),  # real files
    },
)
```

### Special Behaviors

**Root listing (`ls("/")`)**: Aggregates the default backend's listing with virtual directories for each route:

```
$ ls /
workspace/    (from route)
memories/     (from route)
project/      (from route)
todos.md      (from default StateBackend)
```

**Cross-backend grep**: When `path` is `None` or `"/"`, grep searches ALL backends and merges results with full paths:

```
$ grep "TODO" /
/workspace/plan.md:3:  - TODO: implement auth
/project/src/main.py:15:  # TODO: add error handling
/memories/notes.md:2:  TODO: review API design
```

**Batch operations**: `upload_files` and `download_files` group files by target backend for efficiency — each child backend is called only once.

---

## 8. How Our Project Uses It

In `src/agent.py`, we create the CompositeBackend via a factory function:

```python
def create_composite_backend(runtime):
    state = StateBackend(runtime)
    store = StoreBackend(runtime)
    filesystem = FilesystemBackend(virtual_mode=True)

    return CompositeBackend(
        default=state,
        routes={
            "/workspace/": state,      # Task plans, progress logs, todos
            "/memories/": store,       # Project conventions, cross-session state
            "/project/": filesystem,   # Actual project source code
        },
    )
```

This factory is passed to `create_deep_agent(backend=create_composite_backend)`. The framework calls it with the runtime when the agent initializes.

### Path Mapping in Practice

| Agent Path | Backend | Physical Location | Lifetime |
|---|---|---|---|
| `/workspace/todos.md` | StateBackend | In-memory (LangGraph state) | Current thread |
| `/workspace/progress.log` | StateBackend | In-memory | Current thread |
| `/memories/project.md` | StoreBackend | LangGraph BaseStore | Permanent |
| `/project/src/main.py` | FilesystemBackend | `/prj/cline_deep_agent/src/main.py` | Permanent (disk) |
| `/scratch/notes.txt` | StateBackend (default) | In-memory | Current thread |

---

## 9. Accessing the System from Your Terminal

### Option A: Interactive Python Shell

The fastest way to interact with the backends directly:

```bash
cd /prj/cline_deep_agent
source venv/bin/activate

python3 -c "
from deepagents.backends import FilesystemBackend

fs = FilesystemBackend(root_dir='/prj/cline_deep_agent', virtual_mode=True)

# List project root
for f in fs.ls_info('/'):
    kind = 'DIR' if f['is_dir'] else 'FILE'
    print(f'  {kind:>4}  {f[\"path\"]}')

# Read a source file
print(fs.read('/src/llm.py', offset=0, limit=10))

# Search across project
matches = fs.grep_raw('get_llm', path='/')
if isinstance(matches, list):
    for m in matches[:5]:
        print(f'  {m[\"path\"]}:{m[\"line\"]}: {m[\"text\"].strip()}')
"
```

### Option B: Full Composite Backend (requires agent runtime)

To test the full routing system with all three backends:

```bash
cd /prj/cline_deep_agent
source venv/bin/activate

python3 << 'PYEOF'
from unittest.mock import MagicMock
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend

# Create a mock runtime for local testing
runtime = MagicMock()
runtime.state = {"files": {}}
runtime.store = MagicMock()
runtime.config = {"configurable": {"thread_id": "local-test"}}

# Build the composite backend
state = StateBackend(runtime)
filesystem = FilesystemBackend(root_dir="/prj/cline_deep_agent", virtual_mode=True)

composite = CompositeBackend(
    default=state,
    routes={
        "/workspace/": state,
        "/project/": filesystem,
    },
)

# --- Write to ephemeral workspace ---
result = composite.write("/workspace/test.md", "# Test\nHello from the terminal!")
print("Write result:", result.path, "error:", result.error)

# Apply state update (required for StateBackend)
if result.files_update:
    runtime.state["files"].update(result.files_update)

# --- Read it back ---
content = composite.read("/workspace/test.md")
print("\n--- /workspace/test.md ---")
print(content)

# --- Read a real project file via /project/ ---
content = composite.read("/project/pyproject.toml", limit=5)
print("\n--- /project/pyproject.toml (first 5 lines) ---")
print(content)

# --- List root directory ---
print("\n--- ls / ---")
for f in composite.ls_info("/"):
    kind = "DIR" if f["is_dir"] else "FILE"
    print(f"  {kind:>4}  {f['path']}")

# --- Search across all backends ---
print("\n--- grep 'CompositeBackend' across all backends ---")
matches = composite.grep_raw("CompositeBackend", path="/")
if isinstance(matches, list):
    for m in matches[:5]:
        print(f"  {m['path']}:{m['line']}: {m['text'].strip()}")

PYEOF
```

### Option C: Using the Agent CLI

Run the full agent which uses the CompositeBackend internally:

```bash
cd /prj/cline_deep_agent
source venv/bin/activate

# Set your provider
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=gpt-oss:20b

# Run the CLI — the agent uses CompositeBackend for all file operations
python -m src.cli
```

Then interact:
```
> Create a plan to build a REST API
# Agent writes /workspace/todos.md via StateBackend

> What are the project conventions?
# Agent reads /memories/project.md via StoreBackend

> Show me the current source code
# Agent reads /project/src/*.py via FilesystemBackend
```

### Option D: Standalone FilesystemBackend Script

For quick read-only exploration of your project:

```bash
cd /prj/cline_deep_agent
source venv/bin/activate

python3 -c "
from deepagents.backends import FilesystemBackend
fs = FilesystemBackend(root_dir='/prj/cline_deep_agent', virtual_mode=True)

# List all Python files in src/
files = fs.glob_info('src/**/*.py', path='/')
for f in files:
    print(f['path'])

# Search for a function
matches = fs.grep_raw('def get_llm', path='/')
for m in (matches if isinstance(matches, list) else []):
    print(f'{m[\"path\"]}:{m[\"line\"]}: {m[\"text\"]}')
"
```

### Option E: Inspect Backend State at Runtime

Add this to `src/cli.py` or run interactively to inspect what the agent has stored:

```bash
python3 -c "
from deepagents.backends import FilesystemBackend

# Read actual logs
fs = FilesystemBackend(root_dir='/prj/cline_deep_agent', virtual_mode=True)
for f in fs.glob_info('logs/**/*', path='/'):
    print(f'Log: {f[\"path\"]} ({f[\"size\"]} bytes)')
    print(fs.read(f['path'], limit=20))
    print()
"
```

---

## 10. Backend Factory Pattern

The `create_deep_agent()` function accepts backends in two forms:

### Direct Instance

```python
backend = StateBackend(runtime)
create_deep_agent(backend=backend)
```

### Factory Function (recommended)

```python
def my_backend_factory(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={"/project/": FilesystemBackend(virtual_mode=True)},
    )

create_deep_agent(backend=my_backend_factory)
```

The factory pattern is preferred because:
- The `runtime` object isn't available until the agent initializes
- StateBackend and StoreBackend require `runtime` for state/store access
- The factory is called once when the agent graph compiles

Type definition:
```python
BackendFactory = Callable[[ToolRuntime], BackendProtocol]
```

---

## 11. API Reference

### CompositeBackend

```python
CompositeBackend(
    default: BackendProtocol,              # Fallback for unmatched paths
    routes: dict[str, BackendProtocol],    # Path prefix → backend mapping
)
```

### StateBackend

```python
StateBackend(runtime: ToolRuntime)
# Files stored in runtime.state["files"]
# Returns files_update dict for state merging
```

### StoreBackend

```python
StoreBackend(
    runtime: ToolRuntime,
    namespace: NamespaceFactory | None = None,  # Multi-user isolation
)
# Files stored in runtime.store (LangGraph BaseStore)
# Returns files_update = None (externally persisted)
```

### FilesystemBackend

```python
FilesystemBackend(
    root_dir: str | Path | None = None,    # Base directory (default: cwd)
    virtual_mode: bool = False,            # True = sandbox paths to root_dir
    max_file_size_mb: int = 10,            # Skip files larger than this in grep
)
# Reads/writes real filesystem
# Returns files_update = None (already on disk)
```

### Key Error Types

```python
FileOperationError:
    "file_not_found"      # Path does not exist
    "permission_denied"   # OS-level permission error
    "is_directory"        # Expected file, got directory
    "invalid_path"        # Path traversal or invalid characters
```
