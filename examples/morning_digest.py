# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Inbox Triage Assistant — practical multi-agent example.

For each incoming email, two agents analyze it in parallel:
  - Summarizer  : produces a concise summary
  - Analyzer    : classifies the required action and extracts relevant details

The Decider then owns all control flow: based on the classification it
routes to exactly one of four action branches (reply, calendar, todo, archive).

ZipperGen guarantees:
  - Summarizer and Analyzer always run before any action is taken
  - exactly one branch executes per email — no double-booking, no missed steps
  - every action agent receives only what it needs, nothing more
  - the workflow cannot deadlock by construction

Run out of the box with the bundled mock inbox — no credentials needed.
Connect fetch_email to a real IMAP / Gmail client when ready.
"""

from zippergen.syntax import Lifeline, Var
from zippergen.actions import llm, pure
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User         = Lifeline("User")
Decider      = Lifeline("Decider")
Reader       = Lifeline("Reader")
Summarizer   = Lifeline("Summarizer")
Analyzer     = Lifeline("Analyzer")
ReplyAgent   = Lifeline("ReplyAgent")
CalendarTool = Lifeline("CalendarTool")
TodoTool     = Lifeline("TodoTool")
ArchiveTool  = Lifeline("ArchiveTool")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

query   = Var("query",   str)
email   = Var("email",   str)
summary = Var("summary", str)
action  = Var("action",  str)   # "reply" | "calendar" | "todo" | "archive"
details = Var("details", str)   # action-specific content extracted from email
result  = Var("result",  str)

# ---------------------------------------------------------------------------
# Mock inbox — replace with real IMAP / Gmail fetch when ready
# ---------------------------------------------------------------------------

MOCK_EMAIL = """
From: carol@example.com
Subject: URGENT: production outage

Hi,

The payment service has been down for 20 minutes.
I need your go / no-go on the rollback plan before we proceed.
Can you reply ASAP?

Carol
""".strip()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@pure
def fetch_email(query: str) -> str:
    return MOCK_EMAIL


@llm(
    system="You are an assistant that summarizes emails concisely.",
    user="Summarize this email in 2-3 sentences:\n\n{email}",
    parse="text",
    outputs=(("summary", str),),
)
def summarize(email: str) -> None: ...


@llm(
    system=(
        "You are an assistant that classifies emails and extracts action details. "
        "Classify the required action as exactly one of: reply, calendar, todo, archive. "
        "  reply    — the sender is waiting for a response\n"
        "  calendar — the email describes a meeting, event, or deadline to schedule\n"
        "  todo     — the email contains a task or follow-up that needs tracking\n"
        "  archive  — no action needed, safe to archive\n"
        "Also extract the key details needed to carry out that action."
    ),
    user="Classify this email and extract the relevant action details:\n\n{email}",
    parse="json",
    outputs=(("action", str), ("details", str)),
)
def classify_action(email: str) -> None: ...


@llm(
    system="You are an assistant that drafts concise, professional email replies.",
    user="Summary: {summary}\n\nAction context: {details}\n\nDraft a short reply.",
    parse="text",
    outputs=(("result", str),),
)
def draft_reply(summary: str, details: str) -> None: ...


@pure
def create_event(details: str) -> str:
    return f"[Calendar entry created] {details}"


@pure
def add_todo(details: str) -> str:
    return f"[Todo added] {details}"


@pure
def archive(summary: str) -> str:
    return f"[Archived] {summary}"


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow
def inboxTriage(query: str @ User) -> str:
    User(query) >> Decider(query)
    Decider(query) >> Reader(query)
    Reader: email = fetch_email(query)

    # Fan out — Summarizer and Analyzer work in parallel
    Reader(email) >> Summarizer(email)
    Reader(email) >> Analyzer(email)
    Summarizer: summary = summarize(email)
    Analyzer: (action, details) = classify_action(email)

    # Join — Decider waits for both before deciding
    Summarizer(summary) >> Decider(summary)
    Analyzer(action, details) >> Decider(action, details)

    # Decider owns all control flow — exactly one branch executes
    if (action == "reply") @ Decider:
        Decider(summary, details) >> ReplyAgent(summary, details)
        ReplyAgent: result = draft_reply(summary, details)
        ReplyAgent(result) >> Decider(result)
    else:
        if (action == "calendar") @ Decider:
            Decider(details) >> CalendarTool(details)
            CalendarTool: result = create_event(details)
            CalendarTool(result) >> Decider(result)
        else:
            if (action == "todo") @ Decider:
                Decider(details) >> TodoTool(details)
                TodoTool: result = add_todo(details)
                TodoTool(result) >> Decider(result)
            else:
                Decider(summary) >> ArchiveTool(summary)
                ArchiveTool: result = archive(summary)
                ArchiveTool(result) >> Decider(result)

    Decider(result) >> User(result)
    return result @ User


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    USE_UI = True

    inboxTriage.configure(
        llms="mistral",
        # llms="mock",
        ui=USE_UI,
        timeout=120,
    )
    result = inboxTriage(query="latest unread email")
    print(f"\n{'='*60}")
    print("TRIAGE RESULT")
    print('='*60)
    print(result)

    if USE_UI:
        input("\nZipperChat is running at http://localhost:8765 . Press Enter to close. ")
