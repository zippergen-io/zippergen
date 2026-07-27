# Critically revise the candidate

Act as the primary implementation owner. Re-read the original task and the
independent Claude review. Verify every suggestion against the repository
before acting: accept technically justified findings, reject mistaken or
out-of-scope suggestions, and preserve correct existing behavior.

Update the repository where warranted and run proportionate verification. It is
acceptable to make no change when a review finding is incorrect, but explain
the evidence clearly so the next review can evaluate that decision.

Repository write authority has explicit boundaries:

- Do not modify `examples/codex_claude_review.py` or anything below
  `examples/prompts/codex_claude_review/`; they define the executing workflow.
- Do not deploy, start or restart services, commit, push, publish, or modify
  Git metadata.
- Do not contact external systems or use configured MCP/tool integrations.
- Leave every candidate change visible in the working tree for human review.

Return a concise summary of the resulting candidate and the checks performed.
