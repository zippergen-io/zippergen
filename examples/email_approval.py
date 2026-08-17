# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""Watch a mailbox, draft a reply, ask a person, then send or discard.

The tutorial workflow. Small enough to read in one sitting, but it already
contains everything that makes ZipperGen worth using: two participants, an LLM
action, an explicit human decision, a branch owned by whoever makes that
decision, and a loop that makes the whole thing a service rather than a script.

It does not stop on its own. `next_unread_message` waits for the next message,
so the workflow keeps running until you stop it — which is the point: this is
something you deploy, not something you launch by hand each time.

The mailbox here is a directory of text files, so the tutorial needs no
credentials. `examples/call_intake.py` is the same shape against real Gmail.

    mkdir -p mailbox
    echo "Can we meet on Thursday" > mailbox/01.txt
    zippergen run examples/email_approval.py:email_approval --llm mock
"""

import time
from pathlib import Path

from zippergen import DeploymentField, DeploymentSpec, Lifeline, Var, workflow
from zippergen.actions import effect, human, llm, pure

Writer = Lifeline("Writer")
Mailbox = Lifeline("Mailbox")

message = Var("message", str)
draft = Var("draft", str)
approved = Var("approved", bool)
handled = Var("handled", int, default=0)

_mailbox = Path("mailbox")
_poll_seconds = 2.0
# Unlimited by default: this is a service. A demo or a test sets a budget so
# the run ends on its own.
_remaining = 2**31 - 1


def zippergen_setup(config) -> None:
    """Hook called by ``zippergen run`` before configuring the workflow."""

    global _mailbox, _poll_seconds, _remaining
    _mailbox = Path(str(config.option("mailbox", "mailbox")))
    _poll_seconds = float(config.option("poll_seconds", 2.0) or 2.0)
    budget = config.option("max_messages", None)
    _remaining = int(budget) if budget is not None else 2**31 - 1


@effect
def next_unread_message() -> str:
    """Return the oldest unread message, waiting until one arrives.

    A message is a ``.txt`` file in the mailbox directory. Taking one renames
    it to ``.done``, so the same message is never handled twice — including
    after a crash and resume.

    This call does not return until there is something to hand back, which is
    why the workflow never finishes on its own. Setting ``max_messages`` gives
    it a budget instead, and it returns ``""`` once that budget is spent; the
    loop then ends and the workflow reports how many it handled.
    """

    global _remaining
    if _remaining <= 0:
        return ""
    while True:
        _mailbox.mkdir(parents=True, exist_ok=True)
        pending = sorted(_mailbox.glob("*.txt"))
        if pending:
            oldest = pending[0]
            text = oldest.read_text(encoding="utf-8").strip()
            oldest.rename(oldest.with_suffix(".done"))
            _remaining -= 1
            return text
        time.sleep(_poll_seconds)


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


@effect
def send_reply(draft: str, handled: int) -> int:
    """Send the approved reply. Printing it is this tutorial's 'send'."""

    # Start on a fresh line because another parallel role may currently own
    # the terminal prompt. Real connectors should use the same compact,
    # participant-labelled convention for user-facing effect notifications.
    print("\n✓ Mailbox · reply sent")
    return handled + 1


@pure
def discard(handled: int) -> int:
    return handled


@workflow
def email_approval() -> int:
    Mailbox: message = next_unread_message()
    while message @ Mailbox:
        Mailbox(message) >> Writer(message)
        Writer: draft = draft_reply(message)
        Writer(draft) >> Mailbox(draft)
        Mailbox: approved = approve_reply(draft)
        if approved @ Mailbox:
            Mailbox: handled = send_reply(draft, handled)
        else:
            Mailbox: handled = discard(handled)
        Mailbox: message = next_unread_message()
    return handled @ Mailbox


# A foreground run defaults to ./mailbox. A deployment runs from an immutable
# source bundle, so its live mailbox is an external site path selected during
# `zg deploy` and stored as an absolute option.
zippergen_deployment = DeploymentSpec(
    description="Watch a mailbox directory and ask before sending a reply.",
    fields=(
        DeploymentField(
            "mailbox",
            "Mailbox directory to watch",
            target="option",
            required=True,
            path_exists=True,
        ),
    ),
)
