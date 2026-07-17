# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""Beginner tutorial: draft a reply and ask a human to approve it.

Start with the built-in mock LLM and an in-terminal approval:

    uv run python examples/tutorial_review.py

Use the durable SQLite runner and approve from another terminal:

    uv run zippergen run examples/tutorial_review.py:tutorial_review \
      --llm mock \
      --input request="Explain why the sky is blue in two sentences." \
      --store /tmp/zippergen-tutorial-review.sqlite \
      --timeout 0
"""

from zippergen import (
    DeploymentField,
    DeploymentSpec,
    Lifeline,
    human,
    llm,
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


@human(
    kind="confirm",
    context="{draft}",
    instruction="Approve this draft?",
    outputs=["approved: bool"],
    submit_label="Approve",
    cancel_label="Revise",
)
def approve_reply(draft: str) -> None: ...


@llm(
    system="Revise the draft after a human rejection. Make it clearer and safer.",
    user="Original request: {request}\nRejected draft: {draft}",
    parse="text",
    outputs=(("draft", str),),
)
def revise_reply(request: str, draft: str) -> None: ...


@workflow
def tutorial_review(request: str @ Requester) -> str:
    Requester(request) >> Writer(request)
    Writer: draft = draft_reply(request)
    Writer(draft) >> Reviewer(draft)
    Reviewer: approved = approve_reply(draft)
    if approved @ Reviewer:
        Reviewer(draft) >> Requester(draft)
    else:
        Reviewer(draft) >> Writer(draft)
        Writer: draft = revise_reply(request, draft)
        Writer(draft) >> Requester(draft)
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
            "openai_api_key",
            "OpenAI API key",
            target="env",
            env="OPENAI_API_KEY",
            secret=True,
            required=True,
            when="llm",
            when_values=("openai*",),
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
        request="Explain why the sky is blue in two sentences."
    )
    print(f"Result: {answer}")
