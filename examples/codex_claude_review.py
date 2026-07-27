# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportUndefinedVariable=false, reportReturnType=false
"""Codex implements a task and Claude reviews each candidate.

Run this workflow from the repository that the assistants should inspect:

    uv run zippergen run examples/codex_claude_review.py:codex_claude_review \
      --input 'task=Implement the requested repository change.' \
      --input max_review_rounds=3 \
      --store /tmp/zippergen-codex-claude.sqlite \
      --timeout 0

The action declarations select both CLIs explicitly. Codex owns repository
changes; Claude is constrained to read-only review. Completion requires a
Claude approval followed by a read-only Codex final assessment.
"""

from zippergen import Lifeline, assistant, pure, workflow


TaskOwner = Lifeline("TaskOwner")
Codex = Lifeline("Codex")
Claude = Lifeline("Claude")


@assistant(
    instructions_file="examples/prompts/codex_claude_review/implement.md",
    backend="codex",
    access="write",
    external_tools="none",
    workspace=".",
)
def implement_task(task: str) -> str: ...


@assistant(
    instructions_file="examples/prompts/codex_claude_review/review.md",
    backend="claude",
    access="read-only",
    external_tools="none",
    workspace=".",
)
def review_candidate(task: str, implementation_summary: str) -> str: ...


@assistant(
    instructions_file="examples/prompts/codex_claude_review/revise.md",
    backend="codex",
    access="write",
    external_tools="none",
    workspace=".",
)
def revise_candidate(task: str, review: str) -> str: ...


@assistant(
    instructions_file="examples/prompts/codex_claude_review/finalize.md",
    backend="codex",
    access="read-only",
    external_tools="none",
    workspace=".",
)
def finalize_result(task: str, review: str) -> str: ...


@pure
def require_positive_rounds(max_review_rounds: int) -> int:
    if max_review_rounds < 1:
        raise ValueError("max_review_rounds must be at least one")
    return max_review_rounds


@pure
def first_round() -> int:
    return 1


@pure
def next_round(rounds: int) -> int:
    return rounds + 1


@pure
def review_approved(review: str) -> bool:
    first_line = review.strip().splitlines()[0].strip().casefold()
    return first_line == "approve"


@pure
def unresolved_failure(review: str, max_review_rounds: int) -> str:
    return (
        "The implementation was not accepted after "
        f"{max_review_rounds} review round(s). Remaining Claude review:\n"
        f"{review}"
    )


@workflow
def codex_claude_review(
    task: str @ TaskOwner,
    max_review_rounds: int @ TaskOwner,
) -> str:
    TaskOwner: max_review_rounds = require_positive_rounds(max_review_rounds)
    TaskOwner(task, max_review_rounds) >> Codex(task, max_review_rounds)
    TaskOwner(task) >> Claude(task)

    Codex: implementation_summary = implement_task(task)
    Codex(implementation_summary) >> Claude(implementation_summary)
    Claude: review = review_candidate(task, implementation_summary)
    Claude(review) >> Codex(review)
    with Codex:
        approved = review_approved(review)
        rounds = first_round()

    while (not approved and rounds < max_review_rounds) @ Codex:
        Codex: implementation_summary = revise_candidate(task, review)
        Codex: rounds = next_round(rounds)
        Codex(implementation_summary) >> Claude(implementation_summary)
        Claude: review = review_candidate(task, implementation_summary)
        Claude(review) >> Codex(review)
        Codex: approved = review_approved(review)

    if approved @ Codex:
        Codex: result = finalize_result(task, review)
    else:
        Codex: result = unresolved_failure(review, max_review_rounds)

    Codex(result) >> TaskOwner(result)
    return result @ TaskOwner
