# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportUndefinedVariable=false, reportReturnType=false
"""Beginner tutorial: draft, assess, and ask a human to approve a reply.

Start with the built-in mock LLM and an in-terminal approval:

    uv run python examples/tutorial_review.py

Use the durable SQLite runner and approve from another terminal:

    uv run zippergen run examples/tutorial_review.py:tutorial_review \
      --llm mock \
      --input request="Explain why the sky is blue in two sentences." \
      --input max_retries=2 \
      --store /tmp/zippergen-tutorial-review.sqlite \
      --timeout 0
"""

from zippergen import (
    DeploymentField,
    DeploymentSpec,
    Lifeline,
    human,
    llm,
    pure,
    workflow,
)


# Each lifeline is one sequential participant or responsibility boundary.
Requester = Lifeline("Requester")
Writer = Lifeline("Writer")
Reviewer = Lifeline("Reviewer")


@llm(
    system="Write a clear, concise, and factual reply to the request.",
    user="Request: {request}",
    parse="text",
    outputs=(("draft", str),),
)
def draft_reply(request: str) -> None: ...


@llm(
    system=(
        "Act as a careful reviewer. Identify factual, clarity, or safety "
        "concerns in the proposed reply. Be concise."
    ),
    user="Draft to assess: {draft}",
    parse="text",
    outputs=(("concerns", str),),
)
def assess_draft(draft: str) -> None: ...


@human(
    kind="confirm",
    context="Draft:\n{draft}\n\nAutomated reviewer notes:\n{concerns}",
    instruction="Approve this draft?",
    outputs=["approved: bool"],
    submit_label="Approve",
    cancel_label="Revise",
)
def approve_reply(draft: str, concerns: str) -> None: ...


@llm(
    system="Revise the draft after a human rejection. Make it clearer and safer.",
    user="Original request: {request}\nRejected draft: {draft}",
    parse="text",
    outputs=(("draft", str),),
)
def revise_reply(request: str, draft: str) -> None: ...


@pure
def require_nonnegative(max_retries: int) -> int:
    """Reject an invalid retry budget before the workflow starts reviewing."""
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    return max_retries


@pure
def begin_retries() -> int:
    """The initial draft has not consumed a revision attempt."""
    return 0


@pure
def increment_retries(retries: int) -> int:
    return retries + 1


@pure
def review_exhausted(draft: str, max_retries: int) -> str:
    return (
        f"Review stopped after {max_retries} revision attempt(s) without "
        f"approval. Last draft: {draft}"
    )


@workflow
def tutorial_review(
    request: str @ Requester,
    max_retries: int @ Requester,
) -> str:
    Requester: max_retries = require_nonnegative(max_retries)
    Requester(request) >> Writer(request)
    Requester(max_retries) >> Reviewer(max_retries)
    Writer: draft = draft_reply(request)
    Writer(draft) >> Reviewer(draft)
    with Reviewer:
        concerns = assess_draft(draft)
        retries = begin_retries()
        approved = approve_reply(draft, concerns)

    while (not approved and retries < max_retries) @ Reviewer:
        Reviewer: retries = increment_retries(retries)
        Reviewer(draft) >> Writer(draft)
        Writer: draft = revise_reply(request, draft)
        Writer(draft) >> Reviewer(draft)
        with Reviewer:
            concerns = assess_draft(draft)
            approved = approve_reply(draft, concerns)

    if approved @ Reviewer:
        Reviewer(draft) >> Requester(draft)
    else:
        Reviewer: failure = review_exhausted(draft, max_retries)
        Reviewer(failure) >> Requester(draft)
    return draft @ Requester


# This data-only declaration drives `zippergen deploy`. The OpenAI key is
# requested only after the LLM setting is changed from `mock` to `openai:...`.
zippergen_deployment = DeploymentSpec(
    name="tutorial-review",
    description=(
        "Draft a reply with an LLM and wait for a durable human approval. "
        "The default mock backend needs no API key."
    ),
    fields=(
        DeploymentField(
            "llm",
            "LLM provider and model",
            target="llm",
            default="mock",
            required=True,
        ),
        DeploymentField(
            "request",
            "Request to answer",
            target="input",
            default="Explain why the sky is blue in two sentences.",
            required=True,
        ),
        DeploymentField(
            "max_retries",
            "Maximum revisions after the initial draft is rejected",
            target="input",
            default=2,
            required=True,
            help="Use zero or a positive integer.",
        ),
        DeploymentField(
            "openai_api_key",
            "OpenAI API key",
            target="env",
            env="OPENAI_API_KEY",
            secret=True,
            required=True,
            when="llm",
            when_values=("openai*",),
        ),
        DeploymentField(
            "anthropic_api_key",
            "Anthropic API key",
            target="env",
            env="ANTHROPIC_API_KEY",
            secret=True,
            required=True,
            when="llm",
            when_values=("anthropic*", "claude*"),
        ),
    ),
    files=("examples/tutorial_review.py",),
)


if __name__ == "__main__":
    tutorial_review.configure(
        "mock",
        execution="memory",
        ui=False,
        mock_delay=(0.0, 0.0),
    )
    answer = tutorial_review(
        request="Explain why the sky is blue in two sentences.",
        max_retries=2,
    )
    print(f"Result: {answer}")
