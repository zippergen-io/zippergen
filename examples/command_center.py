# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false, reportArgumentType=false

"""Personal command center: three parallel streams sharing lifelines.

Three independent streams all route decisions through a single shared User lifeline:

  Email branch    — triage incoming emails:
                      spam / quick_reply / scheduling_reply / task / careful_reply
  Calendar branch — prepare briefings for upcoming meetings (Researcher → User)
  Chat branch     — Telegram as control interface (all messages are owner commands):
                      schedule_meeting / cancel_meeting / create_task
                      create_event / draft_email / general

Adding a new stream never touches existing ones — ZipperGen's projection ensures
each lifeline receives only the messages it needs.  Writer handles both email
drafts and Chat replies; User is the human-in-the-loop for all three streams;
Calendar cross-participates in scheduling emails and meeting prep.

Modes
-----
--mock   Fake inbox + meetings + chat messages, mock LLM responses.
--live   All three streams connect to real Google APIs (Gmail, Calendar, Chat,
         Tasks).  One-time setup per service:
           python examples/gmail_client.py --setup
           python examples/google_calendar_client.py --setup
           python examples/google_tasks_client.py --setup
           # Telegram: message @BotFather → /newbot → set ZIPPERGEN_TELEGRAM_TOKEN
"""

from zippergen import Lifeline, Var, branch, fragment, llm, parallel, pure, workflow
from zippergen.actions import human
from zippergen.backends import make_openai_backend

_gmail:   object = None
_gcal:    object = None
_gchat:   object = None
_gtasks:  object = None

_briefed_meetings:    set[str] = set()
_chat_meta:           dict[str, dict] = {}  # formatted text → ChatMeta
_cancel_meta:         dict[str, str]  = {}  # cancel_matches text → event_id

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

Dispatcher = Lifeline("Dispatcher")
Writer     = Lifeline("Writer")
Researcher = Lifeline("Researcher")
User       = Lifeline("User")
Mailbox    = Lifeline("Mailbox")
Calendar   = Lifeline("Calendar")
TasksTool  = Lifeline("TasksTool")
Chat       = Lifeline("Chat")

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

# Calendar / meeting-prep branch
meeting          = Var("meeting",          str)
briefing         = Var("briefing",         str)
briefing_ack     = Var("briefing_ack",     bool)
sched_context    = Var("sched_context",    str)  # Calendar-local; updated by scheduling_reply

# Event creation (scheduling_reply branch, Calendar lifeline)
today            = Var("today",            str)
event_details    = Var("event_details",    str)
sched_event      = Var("sched_event",      str)
event_summary    = Var("event_summary",    str)
ack              = Var("ack",              bool)

email_summary    = Var("email_summary",    str)

# Cancellation branch — email stream (exclusive to email branch)
cancel_query    = Var("cancel_query",    str)
cancel_matches  = Var("cancel_matches",  str)
cancel_confirm  = Var("cancel_confirm",  bool)
cancel_status   = Var("cancel_status",   str)

# Task branch (within email stream)
task_title   = Var("task_title",   str)
task_notes   = Var("task_notes",   str)
task_edit    = Var("task_edit",    str)
task_status  = Var("task_status",  str)

# Chat stream
chat_msg     = Var("chat_msg",     str)
chat_route   = Var("chat_route",   str)
chat_draft   = Var("chat_draft",   str)
chat_edit    = Var("chat_edit",    str)
chat_reply   = Var("chat_reply",   str)
chat_status  = Var("chat_status",  str)

# cancel_meeting branch — chat stream (exclusive)
chat_cancel_query    = Var("chat_cancel_query",    str)
chat_cancel_matches  = Var("chat_cancel_matches",  str)
chat_cancel_confirm  = Var("chat_cancel_confirm",  bool)
chat_cancel_status   = Var("chat_cancel_status",   str)

# schedule_meeting / create_event branches — chat stream (exclusive)
chat_today           = Var("chat_today",           str)
chat_event_details   = Var("chat_event_details",   str)
chat_sched_event     = Var("chat_sched_event",      str)
chat_confirmed_event = Var("chat_confirmed_event",  bool)

# create_task_from_chat (exclusive to chat branch)
chat_task_title  = Var("chat_task_title",  str)
chat_task_notes  = Var("chat_task_notes",  str)
chat_task_edit   = Var("chat_task_edit",   str)
chat_task_status = Var("chat_task_status", str)

# draft_email_from_chat (exclusive to chat branch, touches Writer/User/Mailbox)
chat_email_draft  = Var("chat_email_draft",  str)
chat_email_edit   = Var("chat_email_edit",   str)
chat_email_reply  = Var("chat_email_reply",  str)
chat_mail_status  = Var("chat_mail_status",  str)

_  = Var("_", str)  # throwaway output for wait_briefly()

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

INBOX = [
    "Hi, can we move our Friday 2pm catch-up to Monday at 10am? Thanks, Sarah",
    "CONGRATULATIONS! You have been selected for a $1,000,000 prize. Click now!",
    "Could we find some time next week for a quick sync? I'm flexible on timing.",
    "Please review the proposal doc and share your comments by end of week — I need them for the board meeting.",
    "Hey, could you cancel our Thursday standup? I have a conflict that day.",
    (
        "Dear candidate, following up on your application: could you elaborate "
        "on your experience with distributed systems and formal verification? "
        "We are particularly interested in your approach to message-passing correctness."
    ),
]

MEETINGS = [
    "Standup — 09:30\nAttendees: Sarah, John, Alice\nRoom: Zoom",
    "Product review — 14:00\nAttendees: Sarah, Product Team\nRoom: Conference A",
]

CHAT_MESSAGES = [
    "From: Alice\n\nCan you send me the Q1 report when you get a chance?",
    "From: Bob\n\nIs the team meeting tomorrow still on?",
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


@pure(visible=False)
def wait_briefly() -> str:
    import time
    time.sleep(2)
    return ""


@pure
def email_stream_done() -> str:
    return "Email stream done."


# ---------------------------------------------------------------------------
# Infrastructure — Google Tasks
# ---------------------------------------------------------------------------

@pure
def find_matching_events(cancel_query: str) -> str:
    if _gcal is not None:
        events = _gcal.find_events(cancel_query)  # type: ignore[union-attr]
        if not events:
            return "No matching events found."
        event = events[0]
        text = f"{event['summary']} — {event['start']}"
        _cancel_meta[text] = event['id']
        return text
    text = "Thursday Standup — Thu 29 May at 09:30 (mock)"
    _cancel_meta[text] = "mock_event_id"
    return text


@pure
def do_delete_event(cancel_matches: str) -> str:
    if _gcal is not None and cancel_matches in _cancel_meta:
        _gcal.delete_event(_cancel_meta[cancel_matches])  # type: ignore[union-attr]
        return f"Deleted: {cancel_matches}"
    print(f"[Calendar] Mock: would delete '{cancel_matches}'")
    return f"Deleted (mock): {cancel_matches}"


@pure
def skip_cancellation() -> str:
    return "Cancellation declined by user — event kept."


@pure
def decline_creation() -> str:
    return "Got it — no event was created."


@pure
def create_draft_from_instruction(chat_msg: str, reply: str) -> str:
    import re
    print(f"[Mailbox] Draft from instruction: {reply[:80]}…")
    if _gmail is not None:
        to_match   = re.search(r'^To:\s*(.+)$',      reply, re.MULTILINE | re.IGNORECASE)
        subj_match = re.search(r'^Subject:\s*(.+)$', reply, re.MULTILINE | re.IGNORECASE)
        recipient = to_match.group(1).strip()   if to_match   else ""
        subject   = subj_match.group(1).strip() if subj_match else "Draft from Telegram"
        try:
            draft_id = _gmail.create_draft(recipient, subject, reply)  # type: ignore[union-attr]
            return f"Draft saved (id: {draft_id})"
        except Exception as exc:
            print(f"[Mailbox] Error saving draft: {exc}")
            return f"error: {exc}"
    return "draft_saved_mock"


@pure
def add_task(task_title: str, task_notes: str) -> str:
    if _gtasks is not None:
        try:
            return _gtasks.create_task(task_title, task_notes)  # type: ignore[union-attr]
        except Exception as exc:
            print(f"[Tasks] Error creating task: {exc}")
            return f"error: {exc}"
    print(f"[Tasks] Mock: would create '{task_title}'")
    return "task_created_mock"


# ---------------------------------------------------------------------------
# Infrastructure — Telegram
# ---------------------------------------------------------------------------

def chat_present() -> bool:
    if _gchat is not None:
        return _gchat.count_unread_messages() > 0  # type: ignore[union-attr]
    return bool(CHAT_MESSAGES)


@pure
def pop_pending_chat() -> str:
    if _gchat is not None:
        meta = _gchat.fetch_one_message()  # type: ignore[union-attr]
        if meta is None:
            return ""
        text = f"From: {meta['sender']}\n\n{meta['text']}"
        _chat_meta[text] = meta
        return text
    return CHAT_MESSAGES.pop(0)


@pure
def send_chat_reply(chat_msg: str, reply: str) -> str:
    print(f"[Telegram] Reply: {reply[:80]}…")
    if _gchat is not None and chat_msg in _chat_meta:
        meta = _chat_meta[chat_msg]
        return _gchat.send_message(meta['chat_id'], reply)  # type: ignore[union-attr]
    return "sent_mock"


# ---------------------------------------------------------------------------
# Infrastructure — calendar / meeting prep
# ---------------------------------------------------------------------------

def meeting_soon() -> bool:
    if _gcal is not None:
        meta = _gcal.fetch_next_meeting(window_minutes=30)
        return meta is not None and meta['id'] not in _briefed_meetings
    return bool(MEETINGS)


@pure
def pop_next_meeting() -> str:
    if _gcal is not None:
        meta = _gcal.fetch_next_meeting(window_minutes=30)
        if meta is None:
            return ""
        text = (
            f"{meta['summary']}\nWhen: {meta['start']} – {meta['end']}"
            f"\nOrganizer: {meta['organizer']}"
            + (f"\n\n{meta['description']}" if meta['description'] else "")
        )
        _briefed_meetings.add(meta['id'])
        return text
    return MEETINGS.pop(0)


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
def init_sched_context() -> str:
    return ""


@pure
def remember_scheduling_reply(email: str, reply: str, sched_context: str) -> str:
    snippet = reply.strip().replace("\n", " ")[:100]
    entry = f'• Replied to "{email[:60]}…": "{snippet}"'
    return (sched_context + "\n" + entry).strip() if sched_context else entry


@pure
def slot_from_confirm(availability: str, confirmed: bool) -> str:
    return availability if confirmed else "None of these — please suggest another time."


@pure
def todays_date() -> str:
    from datetime import date
    return date.today().strftime("%A %d %B %Y")


@llm(
    system="""
You are a calendar assistant. Given an email requesting a meeting and the chosen
time slot, extract the event details and output a JSON object with exactly these keys:
  "title":    a short meeting title derived from the email content
  "start":    ISO 8601 datetime string (e.g. "2026-05-25T10:00:00")
  "end":      ISO 8601 datetime string (default: 1 hour after start)
  "attendee": the sender's email address extracted from the email headers
Output only the JSON object, no other text.
If no clear time can be determined, output an empty JSON object {}.
""".strip(),
    user="""
Today is {today}.

Email:
{email}

Chosen slot:
{choice}
""".strip(),
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
    system="""
You are an email triage assistant.
Classify the email as exactly one of six labels:
  spam              — unsolicited or promotional
  quick_reply       — simple request answerable in one sentence
  scheduling_reply  — requests a meeting or asks about availability
  cancellation      — requests cancellation of an existing meeting or event
  task              — asks you to do something that should be tracked as a to-do
  careful_reply     — requires research or careful composition
Reply with the single label and nothing else.
""".strip(),
    user="{email}",
    parse="text",
    outputs=(("route", str),),
)
def classify(email: str) -> None: ...


@llm(
    system="""
You are a scheduling assistant.
Classify the scheduling request as exactly one of two labels:
  check_slot    — the sender proposes a specific time and asks if it works
  propose_slots — the sender asks for available times without proposing one
Reply with the single label and nothing else.
""".strip(),
    user="{email}",
    parse="text",
    outputs=(("sched_kind", str),),
)
def classify_scheduling_request(email: str) -> None: ...


@llm(
    system="""
You are a professional email assistant.
Write a concise, polite reply in two sentences or fewer.
""".strip(),
    user="{email}",
    parse="text",
    outputs=(("draft", str),),
)
def write_draft(email: str) -> None: ...


@llm(
    system="""
You are a research assistant. Given an email, provide 2–3 sentences
of relevant background context the writer should know before replying.
""".strip(),
    user="{email}",
    parse="text",
    outputs=(("context", str),),
)
def research(email: str) -> None: ...


@llm(
    system="""
You are a professional email assistant.
Read the email and plan your reply: identify the key points to address,
the appropriate tone, and the reply structure.
Write a brief outline (3–5 bullet points), not the full reply text.
""".strip(),
    user="{email}",
    parse="text",
    outputs=(("outline", str),),
)
def sketch_reply(email: str) -> None: ...


@llm(
    system="""
You are a professional email assistant.
Write a complete, polished reply using the reply outline and the background context.
Keep it professional and under four sentences.
""".strip(),
    user="""
Email:
{email}

Reply outline:
{outline}

Context:
{context}
""".strip(),
    parse="text",
    outputs=(("reply", str),),
)
def write_reply(email: str, outline: str, context: str) -> None: ...


@llm(
    system="""
You are a professional email assistant.
Write a concise, polite scheduling reply confirming the chosen slot or
proposing it to the sender. Keep it under three sentences.
""".strip(),
    user="""
Email:
{email}

Scheduling decision:
{choice}
""".strip(),
    parse="text",
    outputs=(("reply", str),),
)
def write_scheduling_reply(email: str, choice: str) -> None: ...


@llm(
    system="""
You are a meeting preparation assistant.
Given a meeting title, time, and attendees, write a concise briefing:
- Inferred objective (1 sentence)
- Key attendees and their likely roles
- 2–3 suggested talking points or questions to raise
- Any preparation recommended
Keep it under 150 words.
""".strip(),
    user="{meeting}",
    parse="text",
    outputs=(("briefing", str),),
)
def prepare_meeting_briefing(meeting: str) -> None: ...


@llm(
    system="""
You are an assistant that extracts action items from emails.
Given an email, identify the single most important task for the recipient.
Return a JSON object with exactly these keys:
  "task_title": a short actionable title (verb phrase, under 60 chars)
  "task_notes": key details or deadline from the email (1-2 sentences)
""".strip(),
    user="{email}",
    parse="json",
    outputs=(("task_title", str), ("task_notes", str)),
)
def extract_task(email: str) -> None: ...


@llm(
    system="""
You are an assistant that classifies commands sent to a personal Telegram bot.
All messages are from the bot owner giving instructions. Classify as exactly one of six labels:
  schedule_meeting — schedule a meeting with someone (creates event + drafts invitation)
  cancel_meeting   — cancel or remove an existing calendar event
  create_task      — create a task or reminder
  create_event     — add a calendar event without sending an invitation
  draft_email      — draft an email and save it to Gmail
  general          — anything else (questions, status queries, etc.)
Reply with the single label and nothing else.
""".strip(),
    user="{chat_msg}",
    parse="text",
    outputs=(("chat_route", str),),
)
def classify_chat(chat_msg: str) -> None: ...


@llm(
    system="""
You are a professional assistant replying to Telegram messages.
Write a concise, friendly reply in one or two sentences.
""".strip(),
    user="{chat_msg}",
    parse="text",
    outputs=(("chat_draft", str),),
)
def draft_chat_reply(chat_msg: str) -> None: ...


@llm(
    system="""
You are an assistant extracting a calendar search query from a cancellation request.
Given a message asking to cancel a meeting, return a short search string
(keywords, day, or title) to find that event in the calendar.
Return only the search query, nothing else.
""".strip(),
    user="{message}",
    parse="text",
    outputs=(("cancel_query", str),),
)
def extract_cancel_query(message: str) -> None: ...


@llm(
    system="""
You are a professional assistant drafting a brief reply to a meeting cancellation request.
If the event was deleted, confirm it warmly. If not, apologise and explain.
Keep it to one or two sentences.
""".strip(),
    user="""
Original message: {message}

Cancellation result: {cancel_status}
""".strip(),
    parse="text",
    outputs=(("draft", str),),
)
def write_cancellation_reply(message: str, cancel_status: str) -> None: ...


@llm(
    system="""
You are a professional email assistant.
Draft a complete email based on the instruction given.
Format your response as:
  To: <recipient name or email if mentioned, otherwise leave blank>
  Subject: <concise subject line>

  <email body, professional and under 5 sentences>
""".strip(),
    user="{chat_msg}",
    parse="text",
    outputs=(("draft", str),),
)
def draft_email_from_instruction(chat_msg: str) -> None: ...


@llm(
    system="""
You are a calendar assistant.
Extract event details from a direct scheduling command and output a JSON object with exactly these keys:
  "title":    a short meeting title
  "start":    ISO 8601 datetime string (e.g. "2026-05-25T10:00:00")
  "end":      ISO 8601 datetime string (default: 1 hour after start)
  "attendee": email address if mentioned, otherwise empty string
Output only the JSON object, no other text.
If no clear time can be determined, output an empty JSON object {}.
""".strip(),
    user="""
Today is {today}.

Command: {chat_msg}
""".strip(),
    parse="text",
    outputs=(("event_details", str),),
)
def extract_event_details_from_command(today: str, chat_msg: str) -> None: ...


@llm(
    system="""
You are an assistant that extracts action items from messages.
Given a task creation request, identify the task.
Return a JSON object with exactly these keys:
  "task_title": a short actionable title (verb phrase, under 60 chars)
  "task_notes": key details or deadline (1-2 sentences)
""".strip(),
    user="{chat_msg}",
    parse="json",
    outputs=(("task_title", str), ("task_notes", str)),
)
def extract_task_from_chat(chat_msg: str) -> None: ...


@llm(
    system="""
You are a professional email assistant.
Draft a meeting invitation email based on the scheduling command and event details.
Format your response as:
  To: <attendee email or name>
  Subject: Meeting Invitation: <meeting title>

  <friendly invitation body mentioning the time and asking the recipient to confirm>

Keep it professional and concise (under 5 sentences).
""".strip(),
    user="""
Command: {chat_msg}

Event details: {chat_event_details}
""".strip(),
    parse="text",
    outputs=(("draft", str),),
)
def draft_meeting_invitation(chat_msg: str, chat_event_details: str) -> None: ...


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
    kind="ack",
    context="{briefing}",
    instruction="{meeting}",
    outputs=["briefing_ack: bool"],
    submit_label="Got it",
    visible=False,
)
def acknowledge_briefing(meeting: str, briefing: str): pass


@human(
    kind="ack",
    context="{event_summary}",
    instruction="Calendar event created.",
    outputs=["ack: bool"],
    submit_label="Noted",
    visible=False,
)
def acknowledge_event_created(event_summary: str): pass


@human(
    kind="confirm",
    context="{cancel_matches}",
    instruction="Delete this calendar event?",
    outputs=["cancel_confirm: bool"],
    submit_label="Delete →",
    cancel_label="Keep",
)
def confirm_cancellation(cancel_matches: str): pass


@human(
    kind="edit",
    context="{email}",
    prefill="{task_title}",
    instruction="Task notes: {task_notes}",
    outputs=["task_edit: str"],
    submit_label="Create task →",
    cancel_label="Skip",
)
def confirm_task(email: str, task_title: str, task_notes: str): pass


@human(
    kind="edit",
    context="{chat_msg}",
    prefill="{chat_draft}",
    instruction="Edit the reply or submit as-is",
    outputs=["chat_edit: str"],
    submit_label="Send →",
    cancel_label="Skip",
)
def approve_or_edit_chat(chat_msg: str, chat_draft: str): pass


@human(
    kind="confirm",
    context="{chat_event_details}",
    instruction="Create this calendar event?",
    outputs=["chat_confirmed_event: bool"],
    submit_label="Create →",
    cancel_label="Skip",
)
def confirm_event(chat_event_details: str): pass


@human(
    kind="edit",
    context="{chat_msg}",
    prefill="{chat_task_title}",
    instruction="Task notes: {chat_task_notes}",
    outputs=["chat_task_edit: str"],
    submit_label="Create task →",
    cancel_label="Skip",
)
def confirm_task_from_chat(chat_msg: str, chat_task_title: str, chat_task_notes: str): pass


# ---------------------------------------------------------------------------
# Fragments — reusable coordination sub-sequences
# ---------------------------------------------------------------------------

@fragment
def approve_email_reply(email, reply):
    Writer(email, reply) >> User(email, reply)
    User: edit = approve_or_edit(email, reply)
    User: reply = accept_edit(edit, reply)
    User(email, reply) >> Mailbox(email, reply)
    Mailbox: mail_status = create_draft(email, reply)


@fragment
def approve_chat_reply(chat_msg, chat_draft):
    Writer(chat_msg, chat_draft) >> User(chat_msg, chat_draft)
    User: chat_edit = approve_or_edit_chat(chat_msg, chat_draft)
    User: chat_reply = accept_edit(chat_edit, chat_draft)
    User(chat_msg, chat_reply) >> Chat(chat_msg, chat_reply)
    Chat: chat_status = send_chat_reply(chat_msg, chat_reply)


@fragment
def create_calendar_event(source_msg, choice):
    Calendar: today = todays_date()
    Calendar: event_details = extract_event_details(today, source_msg, choice)
    Calendar: sched_event = create_scheduled_event(event_details)
    Calendar: event_summary = format_event_confirmation(event_details)
    Calendar(event_summary) >> User(event_summary)
    User: ack = acknowledge_event_created(event_summary)


@fragment
def task_branch(email):
    Dispatcher(email) >> Writer(email)
    Writer: (task_title, task_notes) = extract_task(email)
    Writer(email, task_title, task_notes) >> User(email, task_title, task_notes)
    User: task_edit = confirm_task(email, task_title, task_notes)
    User: task_title = accept_edit(task_edit, task_title)
    User(task_title, task_notes) >> TasksTool(task_title, task_notes)
    TasksTool: task_status = add_task(task_title, task_notes)


@fragment
def cancellation_branch(email):
    Dispatcher(email) >> Calendar(email)
    Calendar: cancel_query = extract_cancel_query(email)
    Calendar: cancel_matches = find_matching_events(cancel_query)
    Calendar(email, cancel_matches) >> User(email, cancel_matches)
    User: cancel_confirm = confirm_cancellation(cancel_matches)
    if cancel_confirm @ User:
        Calendar: cancel_status = do_delete_event(cancel_matches)
        Calendar(email, cancel_status) >> Writer(email, cancel_status)
    else:
        User: cancel_status = skip_cancellation()
        User(email, cancel_status) >> Writer(email, cancel_status)
    Writer: draft = write_cancellation_reply(email, cancel_status)
    approve_email_reply(email, draft)


@fragment
def cancellation_chat_branch(chat_msg):
    Writer(chat_msg) >> Calendar(chat_msg)
    Calendar: chat_cancel_query = extract_cancel_query(chat_msg)
    Calendar: chat_cancel_matches = find_matching_events(chat_cancel_query)
    Calendar(chat_msg, chat_cancel_matches) >> User(chat_msg, chat_cancel_matches)
    User: chat_cancel_confirm = confirm_cancellation(chat_cancel_matches)
    if chat_cancel_confirm @ User:
        Calendar: chat_cancel_status = do_delete_event(chat_cancel_matches)
        Calendar(chat_msg, chat_cancel_status) >> Writer(chat_msg, chat_cancel_status)
    else:
        User: chat_cancel_status = skip_cancellation()
        User(chat_msg, chat_cancel_status) >> Writer(chat_msg, chat_cancel_status)
    Writer: chat_draft = write_cancellation_reply(chat_msg, chat_cancel_status)
    approve_chat_reply(chat_msg, chat_draft)


@fragment
def schedule_meeting_from_chat(chat_msg):
    Writer(chat_msg) >> Calendar(chat_msg)
    Calendar: chat_today = todays_date()
    Calendar: chat_event_details = extract_event_details_from_command(chat_today, chat_msg)
    Calendar(chat_event_details) >> User(chat_event_details)
    User: chat_confirmed_event = confirm_event(chat_event_details)
    if chat_confirmed_event @ User:
        User(chat_event_details) >> Calendar(chat_event_details)
        Calendar: chat_sched_event = create_scheduled_event(chat_event_details)
        Calendar(chat_msg, chat_event_details) >> Writer(chat_msg, chat_event_details)
        Writer: chat_email_draft = draft_meeting_invitation(chat_msg, chat_event_details)
        Writer(chat_msg, chat_email_draft) >> User(chat_msg, chat_email_draft)
        User: chat_email_edit = approve_or_edit_chat(chat_msg, chat_email_draft)
        User: chat_email_reply = accept_edit(chat_email_edit, chat_email_draft)
        User(chat_msg, chat_email_reply) >> Mailbox(chat_msg, chat_email_reply)
        Mailbox: chat_mail_status = create_draft_from_instruction(chat_msg, chat_email_reply)
        Mailbox(chat_mail_status) >> Chat(chat_mail_status)
        Chat: chat_status = send_chat_reply(chat_msg, chat_mail_status)
    else:
        User: chat_sched_event = decline_creation()
        User(chat_msg, chat_sched_event) >> Chat(chat_msg, chat_sched_event)
        Chat: chat_status = send_chat_reply(chat_msg, chat_sched_event)


@fragment
def create_task_from_chat(chat_msg):
    Writer: (chat_task_title, chat_task_notes) = extract_task_from_chat(chat_msg)
    Writer(chat_msg, chat_task_title, chat_task_notes) >> User(chat_msg, chat_task_title, chat_task_notes)
    User: chat_task_edit = confirm_task_from_chat(chat_msg, chat_task_title, chat_task_notes)
    User: chat_task_title = accept_edit(chat_task_edit, chat_task_title)
    User(chat_msg, chat_task_title, chat_task_notes) >> TasksTool(chat_msg, chat_task_title, chat_task_notes)
    TasksTool: chat_task_status = add_task(chat_task_title, chat_task_notes)
    TasksTool(chat_msg, chat_task_status) >> Chat(chat_msg, chat_task_status)
    Chat: chat_status = send_chat_reply(chat_msg, chat_task_status)


@fragment
def create_event_from_chat(chat_msg):
    Writer(chat_msg) >> Calendar(chat_msg)
    Calendar: chat_today = todays_date()
    Calendar: chat_event_details = extract_event_details_from_command(chat_today, chat_msg)
    Calendar(chat_event_details) >> User(chat_event_details)
    User: chat_confirmed_event = confirm_event(chat_event_details)
    if chat_confirmed_event @ User:
        User(chat_event_details) >> Calendar(chat_event_details)
        Calendar: chat_sched_event = create_scheduled_event(chat_event_details)
        Calendar(chat_msg, chat_sched_event) >> Chat(chat_msg, chat_sched_event)
        Chat: chat_status = send_chat_reply(chat_msg, chat_sched_event)
    else:
        User: chat_sched_event = decline_creation()
        User(chat_msg, chat_sched_event) >> Chat(chat_msg, chat_sched_event)
        Chat: chat_status = send_chat_reply(chat_msg, chat_sched_event)


@fragment
def draft_email_from_chat(chat_msg):
    Writer: chat_email_draft = draft_email_from_instruction(chat_msg)
    Writer(chat_msg, chat_email_draft) >> User(chat_msg, chat_email_draft)
    User: chat_email_edit = approve_or_edit_chat(chat_msg, chat_email_draft)
    User: chat_email_reply = accept_edit(chat_email_edit, chat_email_draft)
    User(chat_msg, chat_email_reply) >> Mailbox(chat_msg, chat_email_reply)
    Mailbox: chat_mail_status = create_draft_from_instruction(chat_msg, chat_email_reply)
    Mailbox(chat_mail_status) >> Chat(chat_mail_status)
    Chat: chat_status = send_chat_reply(chat_msg, chat_mail_status)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow
def command_center():
    Calendar: sched_context = init_sched_context()

    with parallel:
        with branch:
            # ── Email stream ──────────────────────────────────────────────
            while True @ Dispatcher:
                if mail_present() @ Dispatcher:
                    Dispatcher: email = pop_pending_email()
                    Dispatcher: route = classify(email)
                    Dispatcher: route = normalize_route(route)

                    if (route == "spam") @ Dispatcher:
                        Dispatcher(email) >> Mailbox(email)
                        Mailbox: mail_status = mark_as_spam(email)

                    elif (route == "quick_reply") @ Dispatcher:
                        Dispatcher(email) >> Writer(email)
                        Writer: draft = write_draft(email)
                        approve_email_reply(email, draft)

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

                        approve_email_reply(email, reply)

                        # Cross-stream: Calendar appends context and creates the calendar event.
                        User(email, choice, reply) >> Calendar(email, choice, reply)
                        Calendar: sched_context = remember_scheduling_reply(email, reply, sched_context)
                        create_calendar_event(email, choice)

                    elif (route == "cancellation") @ Dispatcher:
                        cancellation_branch(email)

                    elif (route == "task") @ Dispatcher:
                        task_branch(email)

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
                        approve_email_reply(email, reply)

                else:
                    Dispatcher: _ = wait_briefly()

        with branch:
            # ── Meeting prep stream ───────────────────────────────────────
            # When a meeting is approaching, Researcher prepares a briefing
            # and User is notified — independently of the email stream.
            while True @ Calendar:
                if meeting_soon() @ Calendar:
                    Calendar: meeting = pop_next_meeting()
                    Calendar(meeting) >> Researcher(meeting)
                    Researcher: briefing = prepare_meeting_briefing(meeting)
                    Researcher(meeting, briefing) >> User(meeting, briefing)
                    User: briefing_ack = acknowledge_briefing(meeting, briefing)
                else:
                    Calendar: _ = wait_briefly()

        with branch:
            # ── Chat stream ───────────────────────────────────────────────
            # All Telegram messages are commands from the bot owner.
            # Telegram is the control interface; side effects go to the
            # right lifeline (Calendar, Mailbox, TasksTool).
            while True @ Chat:
                if chat_present() @ Chat:
                    Chat: chat_msg = pop_pending_chat()
                    Chat(chat_msg) >> Writer(chat_msg)
                    Writer: chat_route = classify_chat(chat_msg)
                    Writer: chat_route = normalize_route(chat_route)

                    if (chat_route == "schedule_meeting") @ Writer:
                        schedule_meeting_from_chat(chat_msg)
                    elif (chat_route == "cancel_meeting") @ Writer:
                        cancellation_chat_branch(chat_msg)
                    elif (chat_route == "create_task") @ Writer:
                        create_task_from_chat(chat_msg)
                    elif (chat_route == "create_event") @ Writer:
                        create_event_from_chat(chat_msg)
                    elif (chat_route == "draft_email") @ Writer:
                        draft_email_from_chat(chat_msg)
                    else:
                        Writer: chat_draft = draft_chat_reply(chat_msg)
                        approve_chat_reply(chat_msg, chat_draft)
                else:
                    Chat: _ = wait_briefly()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--mock" in sys.argv:
        command_center.configure(llms="mock", ui=True, timeout=3600)

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

        _gmail   = _load("gmail_client")
        _gcal    = _load("google_calendar_client")
        _gchat   = _load("telegram_client")
        _gtasks  = _load("google_tasks_client")
        backend = make_openai_backend(
            api_key="ollama",
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434/v1",
            max_tokens=512,
            timeout=120,
        )
        command_center.configure(backend=backend, ui=True, timeout=3600)

    else:
        backend = make_openai_backend(
            api_key="ollama",
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434/v1",
            max_tokens=512,
            timeout=120,
        )
        command_center.configure(backend=backend, ui=True, timeout=3600)

    command_center()
    input("ZipperChat running at http://localhost:8765. Press Enter to exit. ")
