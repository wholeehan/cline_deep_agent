---
name: task-decomposition
description: Break vague or complex tasks into structured subtasks with title, context, acceptance criterion, complexity, and dependency ordering.
metadata:
  category: planning
---

# Task Decomposition

## When to Use
- When the user submits a new task that requires multiple steps
- When a task is vague or ambiguous and needs clarification before execution
- When planning the order of subtask execution

## Instructions

1. **Analyze the task** — identify the core objective, constraints, and implicit requirements.

2. **Decompose into subtasks** — each subtask must have:
   - `title`: Short, action-oriented description (e.g., "Create Flask app skeleton")
   - `context`: Relevant background the executor needs (files, APIs, conventions)
   - `acceptance_criterion`: A concrete, testable condition for "done" (e.g., "`pytest tests/` passes")
   - `complexity`: One of `trivial`, `simple`, `moderate`, `complex`
   - `depends_on`: List of subtask IDs that must complete first (empty list if independent)

3. **Order subtasks** — respect dependencies; independent subtasks can run in parallel.

4. **Write the plan** — use `write_todos` to persist the subtask list to `/workspace/todos.md`.

5. **Review** — before dispatching, verify:
   - No circular dependencies
   - Every subtask has a testable acceptance criterion
   - Complexity estimates are reasonable
   - The plan covers the full scope of the original task

## Output Format

```json
{
  "subtasks": [
    {
      "id": "1",
      "title": "...",
      "context": "...",
      "acceptance_criterion": "...",
      "complexity": "simple",
      "depends_on": []
    }
  ]
}
```

## Anti-patterns to Avoid
- Do NOT create subtasks that are too granular (e.g., "create file X", "write line Y")
- Do NOT skip the acceptance criterion — every subtask must be verifiable
- Do NOT create circular dependencies
