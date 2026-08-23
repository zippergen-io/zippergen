# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportUndefinedVariable=false, reportReturnType=false
"""A deployed service: watch a mailbox, classify each message, record it.

Three participants and one loop. `Mailbox` polls, `Classifier` asks a model
what a message is, `Records` writes a row. It runs until stopped, which is what
makes it a service rather than a script.

Setup, once
-----------

    zg provider configure google-work google
    zg provider authorize google-work --scopes gmail,spreadsheets
    zg provider accept google-work
    zg connector configure inbox google-work gmail --account me
    zg connector configure table google-work google-sheets \\
        --spreadsheet-id SHEET_ID --tab Triage
    zg connector assign incoming-mail inbox
    zg connector assign triage-records table
    zg model configure triage openai-main gpt-4o-mini --temperature 0
    zg model assign Classifier triage

Running and deploying
---------------------

    zg run --yes --durable --timeout 0     # in a terminal, Ctrl-C to stop
    zg deploy --yes                        # as a supervised service

`poll_seconds` and `message_limit` are declared below, so `--yes` supplies
them and remembers whatever you change. A limit of 0 means run until stopped.

`examples/email_approval.py` is the same shape with no credentials at all: a
directory of text files instead of Gmail, and a person asked at the terminal.
Read that one first.

What this is meant to show
--------------------------

The coordination is the twenty lines of `inbox_triage` at the bottom. Everything
else is one mailbox operation, one table operation, and a prompt. A workflow
does not grow because it is deployed -- it grows when its domain does.
"""

import json
import time

from zippergen import (
    ConnectorRequirement,
    DeploymentField,
    DeploymentSpec,
    GmailMailbox,
    Json,
    Lifeline,
    effect,
    llm,
    pure,
    upsert_json_row,
    workflow,
    Var,
)


Mailbox = Lifeline("Mailbox")
Classifier = Lifeline("Classifier")
Records = Lifeline("Records")

handled = Var("handled", int, default=0)
working = Var("working", bool, default=True)

RECORD_COLUMNS = ("message_id", "sender", "subject", "kind", "received_at")

zippergen_connectors = (
    ConnectorRequirement(
        name="incoming-mail",
        kind="gmail",
        participant="Mailbox",
        capabilities=("fetch-one-unread", "mark-processed"),
        access="read-write",
        description="The mailbox this service watches.",
    ),
    ConnectorRequirement(
        name="triage-records",
        kind="google-sheets",
        participant="Records",
        capabilities=("upsert-row",),
        access="read-write",
        description="One row per message handled.",
    ),
)

zippergen_deployment = DeploymentSpec(
    description="Watch a mailbox, classify each message, and record it.",
    fields=(
        DeploymentField(
            "poll_seconds",
            "Seconds to wait when the mailbox is empty",
            target="input",
            default=60,
            required=True,
        ),
        DeploymentField(
            "message_limit",
            "Stop after this many messages; 0 runs until stopped",
            target="input",
            default=0,
            required=True,
        ),
    ),
    files=("examples/inbox_triage.py",),
)


# --- the mailbox ----------------------------------------------------------
# `visible=False` keeps an idle poll out of the trace. A poller checking every
# minute would otherwise fill the history with "nothing happened" and push out
# the events worth reading.

@effect(connector="incoming-mail", operation="count-unread", visible=False)
def mailbox_has_mail() -> bool:
    return GmailMailbox.from_requirement("incoming-mail").count_unread() > 0


@effect(connector="incoming-mail", operation="fetch-one-unread")
def take_one_message() -> Json:
    message = GmailMailbox.from_requirement("incoming-mail").fetch_one_unread()
    return message or {}


@effect(connector="incoming-mail", operation="mark-processed")
def finish_message(message: Json) -> str:
    GmailMailbox.from_requirement("incoming-mail").mark_processed(
        str(message.get("gmail_id") or "")
    )
    return "processed"


@effect(visible=False)
def wait_for_mail(poll_seconds: int) -> str:
    time.sleep(max(1, poll_seconds))
    return "waited"


# --- what a message is ----------------------------------------------------

@llm(
    system=(
        "You sort incoming mail. Answer with exactly one word: "
        "request, notice, or other."
    ),
    user="Subject: {subject}\n\n{body}",
    parse="text",
    outputs=(("kind", str),),
    temperature=0,
    retries="forever",
)
def classify(subject: str, body: str): ...


@pure
def message_subject(message: Json) -> str:
    return str(message.get("subject") or "")


@pure
def message_body(message: Json) -> str:
    return str(message.get("body") or "")[:4000]


@pure
def known_kind(kind: str) -> str:
    """A model may answer with more than the one word it was asked for."""
    first = kind.strip().casefold().split()[0] if kind.strip() else ""
    return first if first in {"request", "notice"} else "other"


# --- the record -----------------------------------------------------------

@effect(connector="triage-records", operation="upsert-json-row")
def record_message(message: Json, kind: str) -> str:
    row = {
        "message_id": str(message.get("gmail_id") or ""),
        "sender": str(message.get("from") or ""),
        "subject": str(message.get("subject") or ""),
        "kind": kind,
        "received_at": str(message.get("received_at") or ""),
    }
    return upsert_json_row(
        "triage-records",
        json.dumps(row, sort_keys=True),
        columns=RECORD_COLUMNS,
        key_field="message_id",
    )


# --- when to stop ---------------------------------------------------------
# A guard reads a plain name, so every question the loop asks is answered in an
# action first. `message_limit = 0` is the service case: it never stops itself.

@pure
def counted(handled: int) -> int:
    return handled + 1


@pure
def still_working(handled: int, message_limit: int) -> bool:
    return message_limit <= 0 or handled < message_limit


@pure
def triage_finished(handled: int) -> str:
    return f"Handled {handled} message(s)."


@workflow
def inbox_triage(
    poll_seconds: int @ Mailbox,
    message_limit: int @ Mailbox,
) -> str:
    Mailbox: working = still_working(handled, message_limit)
    while working @ Mailbox:
        Mailbox: has_mail = mailbox_has_mail()
        if has_mail @ Mailbox:
            with Mailbox:
                message = take_one_message()
                subject = message_subject(message)
                body = message_body(message)
            Mailbox(message, subject, body) >> Classifier(message, subject, body)
            with Classifier:
                kind = classify(subject, body)
                kind = known_kind(kind)
            Classifier(message, kind) >> Records(message, kind)
            Records: written = record_message(message, kind)
            Records(message, kind, written) >> Mailbox(message, kind, written)
            with Mailbox:
                done = finish_message(message)
                handled = counted(handled)
        else:
            Mailbox: waited = wait_for_mail(poll_seconds)
        Mailbox: working = still_working(handled, message_limit)

    Mailbox: status = triage_finished(handled)
    return status @ Mailbox
