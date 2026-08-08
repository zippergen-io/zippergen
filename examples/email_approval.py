# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""Draft a reply, ask a person to approve it, then send or discard.

The tutorial workflow. Small enough to read in one sitting, but it already
contains everything that makes ZipperGen worth using: two participants, an LLM
action, an explicit human decision, a branch owned by whoever makes that
decision, and a result that depends on the answer.

The incoming message is an input here. A later tutorial replaces it with a real
mailbox and sends the approved reply for real; the coordination does not change.

    zippergen run examples/email_approval.py:email_approval \\
        --llm mock --execution memory \\
        --input message="Could we move our meeting to Thursday afternoon?"
"""

from zippergen import Lifeline, Var, workflow
from zippergen.actions import human, llm, pure

Writer = Lifeline("Writer")
User = Lifeline("User")

message = Var("message", str)
draft = Var("draft", str)
approved = Var("approved", bool)
result = Var("result", str)


@llm(
    system=(
        "You write short, friendly replies to work email. Two sentences at "
        "most. No greeting, no sign-off."
    ),
    user="Reply to this message:\n\n{message}",
    parse="text",
    outputs=[("draft", str)],
)
def draft_reply(message: str): ...


@human(
    kind="confirm",
    context="Proposed reply:\n\n{draft}",
    instruction="Send this reply?",
    outputs=["approved: bool"],
)
def approve_reply(draft: str): ...


@pure
def sent(draft: str) -> str:
    return f"Sent: {draft}"


@pure
def discarded() -> str:
    return "Discarded. Nothing was sent."


@workflow
def email_approval(message: str @ User) -> str:
    # The User hands the incoming message to the Writer.
    User(message) >> Writer(message)
    Writer: draft = draft_reply(message)
    Writer(draft) >> User(draft)

    # The User decides, so the User owns the branch. ZipperGen works out on its
    # own that the Writer takes no part in it.
    User: approved = approve_reply(draft)
    if approved @ User:
        User: result = sent(draft)
    else:
        User: result = discarded()
    return result @ User


if __name__ == "__main__":
    email_approval.configure(llms="mock", execution="memory")
    print(
        email_approval(
            message="Could we move our meeting to Thursday afternoon?"
        )
    )
