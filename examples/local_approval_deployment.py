"""Small local deployment example with an out-of-band human approval.

Run the workflow in one terminal:

    uv run zippergen run examples/local_approval_deployment.py:local_approval \
      --store ~/.zippergen/runs/local-approval.sqlite \
      --input request="Create the Friday demo event" \
      --llm mock \
      --timeout 0

Run a notifier in another terminal:

    uv run zippergen notify telegram \
      --store ~/.zippergen/runs/local-approval.sqlite \
      --watch

Set ZIPPERGEN_TELEGRAM_TOKEN and ZIPPERGEN_TELEGRAM_CHAT_ID before starting the
Telegram notifier. The effect writes an idempotent local audit line so a crash
after the write but before journal commit does not duplicate the external side
effect on retry.
"""

import hashlib
import os
from pathlib import Path

from zippergen import Lifeline, effect, pure, workflow
from zippergen.actions import human


Requester = Lifeline("Requester")
Reviewer = Lifeline("Reviewer")
Executor = Lifeline("Executor")


@human(
    kind="confirm",
    context="{request}",
    instruction="Approve this local request?",
    outputs=["approved: bool"],
    submit_label="Approve",
    cancel_label="Decline",
)
def approve_request(request: str): pass


@pure
def format_decision(request: str, approved: bool) -> str:
    return "approved" if approved else "declined"


@effect
def record_decision(request: str, decision: str) -> str:
    key = hashlib.sha1(f"{decision}\n{request}".encode()).hexdigest()[:16]
    path = Path(os.environ.get(
        "ZIPPERGEN_LOCAL_APPROVAL_LOG",
        str(Path.home() / ".zippergen" / "local-approval.log"),
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{key}\t{decision}\t{request}\n"
    existing = path.read_text() if path.exists() else ""
    if key not in {entry.split("\t", 1)[0] for entry in existing.splitlines() if entry}:
        with path.open("a") as f:
            f.write(line)
    return f"{decision}:{key}"


@workflow
def local_approval(request: str @ Requester) -> str:
    Requester(request) >> Reviewer(request)
    Reviewer: approved = approve_request(request)
    Reviewer: decision = format_decision(request, approved)
    Reviewer(request, decision) >> Executor(request, decision)
    Executor: result = record_decision(request, decision)
    return result @ Executor
