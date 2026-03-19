---
name: qa-verification
description: Evaluate subtask output against its acceptance criterion and emit a pass/fail verdict with evidence.
metadata:
  category: verification
---

# QA Verification

## When to Use
- After a subtask has been completed by the cline-executor subagent
- When you need to verify that the output meets the acceptance criterion
- Before marking a subtask as "verified"

## Instructions

1. **Read the acceptance criterion** from the subtask definition.

2. **Examine the output** — read files, check logs, run tests as specified by the criterion.

3. **Evaluate** — does the output satisfy the criterion? Consider:
   - Functional correctness (does it work?)
   - Completeness (does it cover all requirements?)
   - No regressions (did it break anything?)

4. **Emit verdict**:
   - `"verified"` — criterion met; subtask is done
   - `"failed"` — criterion not met; include specific failure reason

5. **If failed** — provide actionable feedback:
   - What specifically failed
   - What the expected vs actual outcome was
   - Suggested fix (if obvious)

## Output Format

```json
{
  "verdict": "verified" | "failed",
  "evidence": "...",
  "failure_reason": "..."
}
```

## Rules
- NEVER mark a subtask as verified without checking the acceptance criterion
- ALWAYS provide evidence for your verdict
- If the criterion is ambiguous, escalate to the user rather than guessing
