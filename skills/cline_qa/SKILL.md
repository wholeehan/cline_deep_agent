---
name: cline-qa
description: Answer questions from the Cline CLI using shared project context without escalating to the user.
metadata:
  category: qa
---

# Cline QA

## When to Use
- When the Cline CLI asks a question during task execution
- When the cline-executor subagent needs to provide an answer to unblock Cline

## Instructions

1. **Read the question** from the Cline PTY stream.

2. **Gather context** — check:
   - The current subtask description and acceptance criterion
   - Project files and structure (via the backend)
   - Previous subtask results and decisions
   - Any project conventions stored in `/memories/project.md`

3. **Formulate an answer** that is:
   - Grounded in the available context (no hallucinations)
   - Specific and actionable (not vague)
   - Consistent with project conventions and the current plan

4. **Inject the answer** into the Cline PTY stdin to unblock execution.

## Rules
- NEVER ask the user unless the question requires information not available in context
- ALWAYS prefer project-specific answers over generic ones
- If the question is about a technology choice, check `/memories/project.md` for existing conventions
- If truly uncertain, escalate to the user via `interrupt()` rather than guessing

## Examples

**Cline asks:** "What testing framework should I use?"
**Context:** Project has `pytest` in dependencies, `tests/` directory exists
**Answer:** "pytest"

**Cline asks:** "Which port should the server run on?"
**Context:** No port specified in subtask or project config
**Action:** Escalate to user (insufficient context to answer confidently)
