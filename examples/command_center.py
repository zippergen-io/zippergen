# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Personal command center: parallel email and calendar streams with cross-stream context.

Two streams both route decisions through a single shared User lifeline:

  Email branch    — triage incoming emails (spam / quick_reply / scheduling_reply / careful_reply)
  Calendar branch — handle meeting invites (accept / decline)

For scheduling_reply emails, Calendar is an active participant inside the email
branch — not just a passive context store.  It classifies the request, checks or
proposes availability, and only then does Writer compose the reply:

  Dispatcher → Calendar: classify & check/propose
  Calendar   → User:     choose a slot
  User       → Writer:   compose the reply
  User       → Mailbox:  create the draft
  User       → Calendar: store approved scheduling context

The streams proceed independently.  Calendar enriches each invite with whatever
scheduling context has causally reached it so far.  It does not wait for the
email branch — it acts on its local, causally available view.

ZipperGen makes this information flow explicit rather than hidden: each decision
is based on the local state that has actually been communicated, not on a global
shared memory that every agent silently reads.

Modes
-----
--mock   Fake inbox + fake invites, mock LLM responses.
--live   Email branch connects to real Gmail (requires gmail_client setup).
         Calendar branch uses the fake INVITES list until Google Calendar API
         is wired up.
"""

from zippergen import Lifeline, Var, branch, llm, parallel, pure, workflow
from zippergen.actions import human
from zippergen.backends import make_openai_backend

_gmail: object = None
_gcal:  object = None

_invite_meta: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

Dispatcher = Lifeline("Dispatcher")
Writer     = Lifeline("Writer")
Researcher = Lifeline("Researcher")
User       = Lifeline("User")
Mailbox    = Lifeline("Mailbox")
Calendar   = Lifeline("Calendar")
Notifier   = Lifeline("Notifier")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

# Email branch
email        = Var("email",        str)
route        = Var("route",        str)
draft        = Var("draft",        str)
outline      = Var("outline",      str)
context      = Var("context",      str)
reply        = Var("reply",        str)
edit         = Var("edit",         str)
mail_status  = Var("mail_status",  str)

# Scheduling sub-branch (owned by Calendar and User within the email branch)
sched_kind    = Var("sched_kind",    str)
availability  = Var("availability",  str)
confirmed     = Var("confirmed",     bool)   # check_slot: User confirms/declines the proposed slot
choice        = Var("choice",        str)    # unified slot choice going to Writer

# Calendar branch  (distinct names — both branches share User's env)
invite           = Var("invite",           str)
enriched_invite  = Var("enriched_invite",  str)
decision         = Var("decision",         bool)
cal_status       = Var("cal_status",       str)
sched_context    = Var("sched_context",    str)  # Calendar-local; updated by scheduling_reply

# Event creation (scheduling_reply branch, Calendar lifeline)
today            = Var("today",            str)
event_details    = Var("event_details",    str)
sched_event      = Var("sched_event",      str)
event_summary    = Var("event_summary",    str)
ack              = Var("ack",              bool)

# Each branch owns its own summary — no shared variable across branches
email_summary    = Var("email_summary",    str)
calendar_summary = Var("calendar_summary", str)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

INBOX = [
    "Hi, can we move our Friday 2pm catch-up to Monday at 10am? Thanks, Sarah",
    "CONGRATULATIONS! You have been selected for a $1,000,000 prize. Click now!",
    "Could we find some time next week for a quick sync? I'm flexible on timing.",
    (
        "Dear candidate, following up on your application: could you elaborate "
        "on your experience with distributed systems and formal verification? "
        "We are particularly interested in your approach to message-passing correctness."
    ),
]

INVITES = [
    "Meeting with Sarah — Monday 10:00",
    "Project sync — Friday 14:00",
]

_email_meta: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Infrastructure — email
# ---------------------------------------------------------------------------

def mail_present() -> bool:
    if _gmail is not None:
        return _gmail.count_unread() > 0  # type: ignore[union-attr]
    return bool(INBOX)


@pure
def pop_pending_email() -> str:
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


@pure
def email_stream_done() -> str:
    return "Email stream done."


# ---------------------------------------------------------------------------
# Infrastructure — calendar
# ---------------------------------------------------------------------------

def invite_present() -> bool:
    if _gcal is not None:
        return _gcal.count_pending_invites() > 0
    return bool(INVITES)


@pure
def pop_pending_invite() -> str:
    if _gcal is not None:
        meta = _gcal.fetch_one_invite()
        if meta is None:
            return ""
        text = (f"Meeting: {meta['summary']}\n"
                f"When: {meta['start']} – {meta['end']}\n"
                f"Organizer: {meta['organizer']}")
        if meta["description"]:
            text += f"\nDescription: {meta['description']}"
        _invite_meta[text] = meta
        return text
    return INVITES.pop(0)


@pure
def check_requested_slot(email: str) -> str:
    if _gcal is not None:
        return _gcal.check_slot(email)
    print("[Calendar] Checking proposed slot…")
    return "Monday 10:00 is free."


@pure
def propose_available_slots(email: str) -> str:
    if _gcal is not None:
        return _gcal.list_free_slots()
    print("[Calendar] Fetching open slots…")
    return "Monday 10:00, Tuesday 14:00, Thursday 11:00."


@pure
def apply_calendar_decision(invite: str, decision: bool) -> str:
    verb = "Accepted" if decision else "Declined"
    print(f"[Calendar] {verb}: {invite}")
    if _gcal is not None and invite in _invite_meta:
        meta = _invite_meta[invite]
        if decision:
            _gcal.accept_event(meta["id"])
        else:
            _gcal.decline_event(meta["id"])
    return verb.lower()


@pure
def calendar_stream_done() -> str:
    return "Calendar stream done."


@pure
def init_sched_context() -> str:
    return ""


@pure
def remember_scheduling_reply(email: str, reply: str, sched_context: str) -> str:
    snippet = reply.strip().replace("\n", " ")[:100]
    entry = f'• Replied to "{email[:60]}…": "{snippet}"'
    return (sched_context + "\n" + entry).strip() if sched_context else entry


@pure
def enrich_invite(invite: str, sched_context: str) -> str:
    if not sched_context:
        return invite
    return f"{invite}\n\nScheduling context:\n{sched_context}"


@pure
def slot_from_confirm(availability: str, confirmed: bool) -> str:
    return availability if confirmed else "None of these — please suggest another time."


@pure
def todays_date() -> str:
    from datetime import date
    return date.today().strftime("%A %d %B %Y")


@llm(
    system=(
        "You are a calendar assistant. Given an email requesting a meeting and the chosen "
        "time slot, extract the event details and output a JSON object with exactly these keys:\n"
        '  "title":    a short meeting title derived from the email content\n'
        '  "start":    ISO 8601 datetime string (e.g. "2026-05-25T10:00:00")\n'
        '  "end":      ISO 8601 datetime string (default: 1 hour after start)\n'
        '  "attendee": the sender\'s email address extracted from the email headers\n'
        "Output only the JSON object, no other text. "
        "If no clear time can be determined, output an empty JSON object {}."
    ),
    user="Today is {today}.\n\nEmail:\n{email}\n\nChosen slot:\n{choice}",
    parse="text",
    outputs=(("event_details", str),),
)
def extract_event_details(today: str, email: str, choice: str) -> None: ...


@pure
def format_event_confirmation(event_details: str) -> str:
    import json, re
    text = re.sub(r"^```[a-z]*\n?", "", event_details.strip(), flags=re.MULTILINE)
    text = text.replace("```", "").strip()
    try:
        d = json.loads(text)
    except Exception:
        return "Calendar event created."
    title = d.get("title", "Meeting")
    start = d.get("start", "")
    # Format ISO datetime → readable e.g. "Monday 25 May at 10:00"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(start)
        start_fmt = dt.strftime("%A %-d %B at %H:%M")
    except Exception:
        start_fmt = start
    return f"{title} — {start_fmt}" if start_fmt else title


@pure
def create_scheduled_event(event_details: str) -> str:
    import json, re
    # Strip markdown code fences that LLMs sometimes add
    text = re.sub(r"^```[a-z]*\n?", "", event_details.strip(), flags=re.MULTILINE)
    text = text.replace("```", "").strip()
    try:
        d = json.loads(text)
    except Exception:
        print(f"[Calendar] Could not parse event JSON: {text[:120]}")
        return "skipped (parse error)"
    if not d.get("start"):
        print(f"[Calendar] No start time in event details: {text[:120]}")
        return "skipped (no time determined)"
    if _gcal is None:
        print(f"[Calendar] Mock: would create '{d.get('title', 'Meeting')}' at {d['start']}")
        return "skipped (mock mode)"
    try:
        return _gcal.create_event(
            summary=d.get("title", "Meeting"),
            start_iso=d["start"],
            end_iso=d.get("end", d["start"]),
            attendee_email=d.get("attendee", ""),
        )
    except Exception as exc:
        print(f"[Calendar] Event creation failed: {exc}")
        return f"error: {exc}"


# ---------------------------------------------------------------------------
# Shared fallback
# ---------------------------------------------------------------------------

@pure
def accept_edit(edit: str, reply: str) -> str:
    return edit.strip() if edit.strip() else reply


# ---------------------------------------------------------------------------
# LLM actions
# ---------------------------------------------------------------------------

@llm(
    system=(
        "You are an email triage assistant. "
        "Classify the email as exactly one of four labels:\n"
        "  spam              — unsolicited or promotional\n"
        "  quick_reply       — simple request answerable in one sentence\n"
        "  scheduling_reply  — requests a meeting or asks about availability\n"
        "  careful_reply     — requires research or careful composition\n"
        "Reply with the single label and nothing else."
    ),
    user="{email}",
    parse="text",
    outputs=(("route", str),),
)
def classify(email: str) -> None: ...


@llm(
    system=(
        "You are a scheduling assistant. "
        "Classify the scheduling request as exactly one of two labels:\n"
        "  check_slot    — the sender proposes a specific time and asks if it works\n"
        "  propose_slots — the sender asks for available times without proposing one\n"
        "Reply with the single label and nothing else."
    ),
    user="{email}",
    parse="text",
    outputs=(("sched_kind", str),),
)
def classify_scheduling_request(email: str) -> None: ...


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


@llm(
    system=(
        "You are a professional email assistant. "
        "Write a concise, polite scheduling reply confirming the chosen slot or "
        "proposing it to the sender. Keep it under three sentences."
    ),
    user="Email:\n{email}\n\nScheduling decision:\n{choice}",
    parse="text",
    outputs=(("reply", str),),
)
def write_scheduling_reply(email: str, choice: str) -> None: ...


# ---------------------------------------------------------------------------
# Human actions
# ---------------------------------------------------------------------------

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


@human(
    kind="confirm",
    context="{email}\n{availability}",
    instruction="Confirm this slot?",
    outputs=["confirmed: bool"],
    submit_label="Confirm",
    cancel_label="Decline",
)
def confirm_slot(email: str, availability: str): pass


@human(
    kind="select",
    context="{email}",
    prefill="{availability}",
    instruction="Choose a slot",
    outputs=["choice: str"],
    submit_label="Select →",
    cancel_label="Decline",
)
def choose_from_proposed_slots(email: str, availability: str): pass


@human(
    kind="confirm",
    context="{invite_text}",
    instruction="Accept this calendar invite?",
    outputs=["decision: bool"],
    submit_label="Accept",
    cancel_label="Decline",
)
def approve_or_decline(invite_text: str): pass


@human(
    kind="ack",
    context="{event_summary}",
    instruction="Calendar event created.",
    outputs=["ack: bool"],
    submit_label="Noted",
)
def acknowledge_event_created(event_summary: str): pass


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow
def command_center() -> str:
    Calendar: sched_context = init_sched_context()

    with parallel:
        with branch:
            # ── Email stream ──────────────────────────────────────────────
            while mail_present() @ Dispatcher:
                Dispatcher: email = pop_pending_email()
                Dispatcher: route = classify(email)
                Dispatcher: route = normalize_route(route)

                if (route == "spam") @ Dispatcher:
                    Dispatcher(email) >> Mailbox(email)
                    Mailbox: mail_status = mark_as_spam(email)

                elif (route == "quick_reply") @ Dispatcher:
                    Dispatcher(email) >> Writer(email)
                    Writer: draft = write_draft(email)
                    Writer(email, draft) >> User(email, draft)
                    User: edit = approve_or_edit(email, draft)
                    User: reply = accept_edit(edit, draft)
                    User(email, reply) >> Mailbox(email, reply)
                    Mailbox: mail_status = create_draft(email, reply)

                elif (route == "scheduling_reply") @ Dispatcher:
                    # Calendar classifies the request and checks or proposes slots.
                    Dispatcher(email) >> Calendar(email)
                    Calendar: sched_kind = classify_scheduling_request(email)
                    Calendar: sched_kind = normalize_route(sched_kind)

                    if (sched_kind == "check_slot") @ Calendar:
                        Calendar: availability = check_requested_slot(email)
                        Calendar(email, availability) >> User(email, availability)
                        User: confirmed = confirm_slot(email, availability)
                        User: choice = slot_from_confirm(availability, confirmed)
                    else:
                        Calendar: availability = propose_available_slots(email)
                        Calendar(email, availability) >> User(email, availability)
                        User: choice = choose_from_proposed_slots(email, availability)

                    User(email, choice) >> Writer(email, choice)
                    Writer: reply = write_scheduling_reply(email, choice)

                    Writer(email, reply) >> User(email, reply)
                    User: edit = approve_or_edit(email, reply)
                    User: reply = accept_edit(edit, reply)

                    User(email, reply) >> Mailbox(email, reply)
                    Mailbox: mail_status = create_draft(email, reply)

                    # Cross-stream: approved scheduling reply sent to Calendar.
                    # Calendar appends context and creates the calendar event.
                    User(email, choice, reply) >> Calendar(email, choice, reply)
                    Calendar: sched_context = remember_scheduling_reply(email, reply, sched_context)
                    Calendar: today = todays_date()
                    Calendar: event_details = extract_event_details(today, email, choice)
                    Calendar: sched_event = create_scheduled_event(event_details)
                    Calendar: event_summary = format_event_confirmation(event_details)
                    Calendar(event_summary) >> Notifier(event_summary)
                    Notifier: ack = acknowledge_event_created(event_summary)

                else:
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
                    Mailbox: mail_status = create_draft(email, reply)

            Dispatcher: email_summary = email_stream_done()

        with branch:
            # ── Calendar stream ───────────────────────────────────────────
            # Each invite is enriched with the scheduling context that Calendar
            # has received from the email stream so far.  Calendar does not wait
            # for the email branch — it acts on its local, causally available view.
            while invite_present() @ Calendar:
                Calendar: invite = pop_pending_invite()
                Calendar: enriched_invite = enrich_invite(invite, sched_context)
                Calendar(invite, enriched_invite) >> User(invite, enriched_invite)
                User: decision = approve_or_decline(enriched_invite)
                User(invite, decision) >> Calendar(invite, decision)
                Calendar: cal_status = apply_calendar_decision(invite, decision)

            Calendar: calendar_summary = calendar_stream_done()

    return email_summary @ Dispatcher


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--mock" in sys.argv:
        command_center.configure(llms="mock", ui=True, timeout=120)

    elif "--live" in sys.argv:
        import importlib.util
        from pathlib import Path

        def _load(name: str):
            spec = importlib.util.spec_from_file_location(
                name, Path(__file__).parent / f"{name}.py"
            )
            assert spec and spec.loader, f"Could not load {name}.py"
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod

        _gmail = _load("gmail_client")
        _gcal  = _load("google_calendar_client")
        backend = make_openai_backend(
            api_key="ollama",
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434/v1",
            max_tokens=512,
            timeout=120,
        )
        command_center.configure(backend=backend, ui=True, timeout=600)

    else:
        backend = make_openai_backend(
            api_key="ollama",
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434/v1",
            max_tokens=512,
            timeout=120,
        )
        command_center.configure(backend=backend, ui=True, timeout=600)

    result = command_center()
    print(f"\nResult: {result}")
    input("ZipperChat running at http://localhost:8765. Press Enter to exit. ")
