---
name: decision-policy
description: Rules for auto-approving, escalating, or rejecting Cline actions based on risk classification.
metadata:
  category: policy
---

# Decision Policy

## When to Use
- When the cline-executor encounters an action that needs approval
- When classifying whether an action should be auto-approved or escalated to the user

## Policy Table

| Action Category | Examples | Decision |
|---|---|---|
| File read | `read_file`, `cat`, `ls`, `grep` | **Auto-approve** |
| File write/create | `write_file`, `edit_file`, `touch` | **Auto-approve** |
| Local command (safe) | `npm install`, `pip install`, `pytest`, `ruff` | **Auto-approve** |
| Git operations (local) | `git add`, `git commit`, `git branch` | **Auto-approve** |
| Git operations (remote) | `git push`, `git pull` | **Escalate** |
| External HTTP requests | `curl`, `wget`, API calls | **Escalate** |
| File/directory delete | `rm`, `rmdir`, `git clean` | **Always escalate** |
| System modification | `chmod`, `chown`, env changes | **Always escalate** |
| Database operations | `DROP`, `DELETE`, `ALTER` | **Always escalate** |
| Package publish | `npm publish`, `pip upload` | **Always escalate** |
| Unknown/unclassified | Any action not in the above categories | **Escalate** |

## Instructions

1. **Classify the action** against the policy table above.
2. **Auto-approve** actions: proceed without interrupting the user.
3. **Escalate** actions: use `interrupt()` to pause and present the action to the user for approval.
4. **Always escalate** actions: NEVER auto-approve these regardless of context.

## Edge Cases
- If an action spans multiple categories (e.g., a script that reads AND deletes), use the most restrictive policy
- If unsure about classification, default to **Escalate**
- Batch multiple escalations into a single interrupt when they occur in the same reasoning step
