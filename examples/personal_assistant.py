# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Inbox assistant: triage loop, routing, and a parallel careful-reply branch.

Dispatcher reads the inbox, classifies each email, and routes it:

  spam          → Mailbox marks it as spam
  quick_reply   → Writer drafts; User approves or edits; Mailbox creates draft
  careful_reply → Writer plans the reply while Researcher gathers context (parallel);
                  Writer then composes the final reply from both;
                  User approves or edits; Mailbox creates the draft

The invariant: Mailbox only creates a draft after explicit User approval.

Modes
-----
--mock        Quick smoke-test with a simulated inbox and mock LLM responses.

--live        Connect to a real Gmail inbox.  Requires one-time OAuth2 setup:

                  python examples/gmail_client.py --setup

              Then place credentials.json at ~/.zippergen_gmail_credentials.json
              (or set ZIPPERGEN_GMAIL_CREDENTIALS) and run:

                  python examples/personal_assistant.py --live

(default)     Use a local Ollama server:

                  # On your GPU server:
                  ollama serve && ollama pull qwen2.5:7b

                  # SSH tunnel from your laptop:
                  ssh -L 11434:127.0.0.1:11434 <gpu-server>

                  python examples/personal_assistant.py
"""

from zippergen import Lifeline, Var, branch, llm, parallel, pure, workflow
from zippergen.actions import human
from zippergen.backends import make_openai_backend

# Optional real Gmail backend — imported lazily when --live is passed.
_gmail: object = None

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

Dispatcher = Lifeline("Dispatcher")
Writer     = Lifeline("Writer")
Researcher = Lifeline("Researcher")
User       = Lifeline("User")
Mailbox    = Lifeline("Mailbox")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

email   = Var("email",   str)
route   = Var("route",   str)
draft   = Var("draft",   str)   # quick_reply: Writer's full draft
outline = Var("outline", str)   # careful_reply: Writer's reply plan (parallel step)
context = Var("context", str)
reply   = Var("reply",   str)   # LLM-proposed reply, kept as fallback
edit    = Var("edit",    str)   # User's edit; empty means "approve as-is"
status  = Var("status",  str)   # Mailbox outcome (mark_as_spam / create_draft)
summary = Var("summary", str)

# ---------------------------------------------------------------------------
# Mock inbox (used when --live is NOT passed)
# ---------------------------------------------------------------------------

INBOX = [
    "Hi, can we move our Friday 2pm catch-up to Monday at 10am? Thanks, Sarah",
    "CONGRATULATIONS! You have been selected to receive a $1,000,000 prize. Click here now!",
    (
        "Dear candidate, following up on your application: could you elaborate "
        "on your experience with distributed systems and formal verification? "
        "We are particularly interested in your approach to message-passing correctness."
    ),
]

# Side-channel: formatted email text → Gmail metadata (id, sender, subject).
# Only populated in --live mode; keyed by the text returned from pop_pending().
_email_meta: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Infrastructure  (swap between mock and live via _gmail module-level var)
# ---------------------------------------------------------------------------

# Plain Python predicate — must stay outside @pure; guards are lambdas.
def mail_present() -> bool:
    if _gmail is not None:
        return _gmail.count_unread() > 0  # type: ignore[union-attr]
    return len(INBOX) > 0


@pure
def pop_pending() -> str:
    if _gmail is not None:
        meta = _gmail.fetch_one()  # type: ignore[union-attr]
        if meta is None:
            return ""
        text = f"From: {meta['sender']}\nSubject: {meta['subject']}\n\n{meta['body']}"
        _email_meta[text] = meta
        return text
    return INBOX.pop(0)


@pure
def normalize_route(s: str) -> str:
    return s.strip().lower().split()[0].rstrip(".,;:")


@pure
def inbox_done() -> str:
    return "All emails processed."


# ---------------------------------------------------------------------------
# Mailbox actions
# ---------------------------------------------------------------------------

@pure
def mark_as_spam(email: str) -> str:
    print(f"[Mailbox] Marked as spam: {email[:60]}…")
    if _gmail is not None and email in _email_meta:
        _gmail.mark_as_spam(_email_meta[email]["id"])  # type: ignore[union-attr]
    return "spam"


@pure
def create_draft(email: str, reply: str) -> str:
    print(f"[Mailbox] Draft: {reply[:80]}…")
    if _gmail is not None and email in _email_meta:
        meta = _email_meta[email]
        draft_id = _gmail.create_draft(  # type: ignore[union-attr]
            meta["sender"], meta["subject"], reply
        )
        print(f"[Mailbox] Gmail draft created: {draft_id}")
    return "draft_created"


# ---------------------------------------------------------------------------
# LLM actions
# ---------------------------------------------------------------------------

@llm(
    system=(
        "You are an email triage assistant. "
        "Classify the email as exactly one of three labels:\n"
        "  spam          — unsolicited or promotional\n"
        "  quick_reply   — simple request answerable in one sentence\n"
        "  careful_reply — requires research or careful composition\n"
        "Reply with the single label and nothing else."
    ),
    user="{email}",
    parse="text",
    outputs=(("route", str),),
)
def classify(email: str) -> None: ...


@llm(
    system=(
        "You are a professional email assistant. "
        "Write a concise, polite reply in two sentences or fewer."
    ),
    user="{email}",
    parse="text",
    outputs=(("draft", str),),
)
def write_draft(email: str) -> None: ...


@llm(
    system=(
        "You are a research assistant. Given an email, provide 2–3 sentences "
        "of relevant background context the writer should know before replying."
    ),
    user="{email}",
    parse="text",
    outputs=(("context", str),),
)
def research(email: str) -> None: ...


@llm(
    system=(
        "You are a professional email assistant. "
        "Read the email and plan your reply: identify the key points to address, "
        "the appropriate tone, and the reply structure. "
        "Write a brief outline (3–5 bullet points), not the full reply text."
    ),
    user="{email}",
    parse="text",
    outputs=(("outline", str),),
)
def sketch_reply(email: str) -> None: ...


@llm(
    system=(
        "You are a professional email assistant. "
        "Write a complete, polished reply using the reply outline and the background context. "
        "Keep it professional and under four sentences."
    ),
    user="Email:\n{email}\n\nReply outline:\n{outline}\n\nContext:\n{context}",
    parse="text",
    outputs=(("reply", str),),
)
def write_reply(email: str, outline: str, context: str) -> None: ...


@pure
def accept_edit(edit: str, reply: str) -> str:
    return edit.strip() if edit.strip() else reply


@human(
    kind="edit",
    context="{email}",
    prefill="{reply}",
    instruction="Edit the proposed reply or submit as-is",
    outputs=["edit: str"],
    submit_label="Approve & send →",
    cancel_label="Decline",
)
def approve_or_edit(email: str, reply: str): pass


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow
def inbox_assistant() -> str:
    while mail_present() @ Dispatcher:
        Dispatcher: email = pop_pending()
        Dispatcher: route = classify(email)
        Dispatcher: route = normalize_route(route)

        if (route == "spam") @ Dispatcher:
            Dispatcher(email) >> Mailbox(email)
            Mailbox: status = mark_as_spam(email)

        elif (route == "quick_reply") @ Dispatcher:
            Dispatcher(email) >> Writer(email)
            Writer: draft = write_draft(email)
            Writer(email, draft) >> User(email, draft)
            User: edit = approve_or_edit(email, draft)
            User: reply = accept_edit(edit, draft)
            User(email, reply) >> Mailbox(email, reply)
            Mailbox: status = create_draft(email, reply)

        else:
            # careful_reply: Writer plans the reply while Researcher gathers context.
            # The final reply is only written once both are ready.
            Dispatcher(email) >> Writer(email)
            Dispatcher(email) >> Researcher(email)

            with parallel:
                with branch:
                    Writer: outline = sketch_reply(email)
                with branch:
                    Researcher: context = research(email)

            Researcher(email, context) >> Writer(email, context)
            Writer: reply = write_reply(email, outline, context)
            Writer(email, reply) >> User(email, reply)
            User: edit = approve_or_edit(email, reply)
            User: reply = accept_edit(edit, reply)
            User(email, reply) >> Mailbox(email, reply)
            Mailbox: status = create_draft(email, reply)

    Dispatcher: summary = inbox_done()
    return summary @ Dispatcher


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--mock" in sys.argv:
        inbox_assistant.configure(llms="mock", ui=True, timeout=60)

    elif "--live" in sys.argv:
        import importlib.util
        from pathlib import Path
        _spec = importlib.util.spec_from_file_location(
            "gmail_client", Path(__file__).parent / "gmail_client.py"
        )
        assert _spec and _spec.loader, "Could not load gmail_client.py"
        _gmail_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_gmail_module)  # type: ignore[union-attr]
        _gmail = _gmail_module
        backend = make_openai_backend(
            api_key="ollama",
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434/v1",
            max_tokens=512,
            timeout=120,
        )
        inbox_assistant.configure(backend=backend, ui=True, timeout=600)

    else:
        backend = make_openai_backend(
            api_key="ollama",
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434/v1",
            max_tokens=512,
            timeout=120,
        )
        inbox_assistant.configure(backend=backend, ui=True, timeout=600)

    result = inbox_assistant()
    print(f"\nResult: {result}")
    input("ZipperChat running at http://localhost:8765. Press Enter to exit. ")
