# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false, reportArgumentType=false

"""Personal command center: two parallel streams sharing lifelines.

Two independent streams both route decisions through a single shared User lifeline:

  Email branch — triage incoming emails:
                   spam / reply / scheduling / cancellation
  Chat branch  — Telegram as control interface (all messages are owner commands):
                   schedule_meeting / cancel_meeting
                   create_event / draft_email / general

Adding a new stream never touches existing ones — ZipperGen's projection ensures
each lifeline receives only the messages it needs.  Writer handles both email
drafts and Chat replies; User is the human-in-the-loop for both streams.

Modes
-----
--mock    Fake inbox + chat messages, mock LLM responses.
--openai  Live services + OpenAI (reads OPENAI_API_KEY and OPENAI_MODEL).
--live    Live services + local Ollama model.  One-time setup per service:
           python examples/gmail_client.py --setup
           python examples/google_calendar_client.py --setup
           # Telegram: message @BotFather → /newbot → set ZIPPERGEN_TELEGRAM_TOKEN
"""

from zippergen import Lifeline, Var, branch, fragment, llm, parallel, pure, workflow
from zippergen.actions import human
from zippergen.backends import make_openai_backend

_gmail:   object = None
_gcal:    object = None
_gchat:   object = None

_chat_meta:           dict[str, dict] = {}  # formatted text → ChatMeta
_cancel_meta:         dict[str, str]  = {}  # cancel_matches text → event_id

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

Dispatcher = Lifeline("Dispatcher")
Writer     = Lifeline("Writer")
User       = Lifeline("User")
Mailbox    = Lifeline("Mailbox")
Calendar   = Lifeline("Calendar")
Chat       = Lifeline("Chat")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

# Email branch
email        = Var("email",        str)
route        = Var("route",        str)
draft        = Var("draft",        str)
reply        = Var("reply",        str)
edit         = Var("edit",         str)
mail_status  = Var("mail_status",  str)

# Scheduling sub-branch (owned by Calendar and User within the email branch)
sched_kind    = Var("sched_kind",    str)
availability  = Var("availability",  str)
confirmed     = Var("confirmed",     bool)   # check_slot: User confirms/declines the proposed slot
choice        = Var("choice",        str)    # unified slot choice going to Writer

sched_context    = Var("sched_context",    str)  # Calendar-local; updated by scheduling_reply

# Event creation (scheduling_reply branch, Calendar lifeline)
today            = Var("today",            str)
event_details    = Var("event_details",    str)
sched_event      = Var("sched_event",      str)
event_summary    = Var("event_summary",    str)
ack              = Var("ack",              bool)

# Cancellation branch — email stream (exclusive to email branch)
cancel_query    = Var("cancel_query",    str)
cancel_matches  = Var("cancel_matches",  str)
cancel_confirm  = Var("cancel_confirm",  bool)
cancel_status   = Var("cancel_status",   str)

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
    "From: Sarah Müller <sarah.mueller@example.com>\nSubject: Moving Friday catch-up\n\nHi, can we move our Friday 2pm catch-up to Monday at 10am? Thanks, Sarah",
    "From: noreply@prize-alert.com\nSubject: CONGRATULATIONS — you have been selected!\n\nCONGRATULATIONS! You have been selected for a $1,000,000 prize. Click now!",
    "From: Tom Nakamura <tom.nakamura@example.com>\nSubject: Quick sync next week?\n\nCould we find some time next week for a quick sync? I'm flexible on timing.",
    "From: Priya Sharma <priya.sharma@example.com>\nSubject: Proposal doc review\n\nPlease review the proposal doc and share your comments by end of week — I need them for the board meeting.",
    "From: Alex Rivera <alex.rivera@example.com>\nSubject: Cancel Thursday standup\n\nHey, could you cancel our Thursday standup? I have a conflict that day.",
    (
        "From: recruiting@researchlab.example.com\nSubject: Follow-up on your application\n\n"
        "Dear candidate, following up on your application: could you elaborate "
        "on your experience with distributed systems and formal verification? "
        "We are particularly interested in your approach to message-passing correctness."
    ),
]

CHAT_MESSAGES = [
    "Cancel the team meeting tomorrow",
    "Draft an email to Alice saying I will send the report by Friday",
    "Schedule a meeting with Alice for next Monday at 10am",
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
# Infrastructure — Calendar
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
    if not reply.strip():
        return "Draft skipped."
    print(f"[Mailbox] Draft from instruction: {reply[:80]}…")
    if _gmail is not None:
        to_match   = re.search(r'^To:\s*(.+)$',      reply, re.MULTILINE | re.IGNORECASE)
        subj_match = re.search(r'^Subject:\s*(.+)$', reply, re.MULTILINE | re.IGNORECASE)
        raw_to    = to_match.group(1).strip()   if to_match   else ""
        recipient = raw_to if "@" in raw_to else ""   # Gmail rejects non-email To headers
        subject   = subj_match.group(1).strip() if subj_match else "Draft from Telegram"
        try:
            draft_id = _gmail.create_draft(recipient, subject, reply)  # type: ignore[union-attr]
            return f"Draft saved (id: {draft_id})"
        except Exception as exc:
            print(f"[Mailbox] Error saving draft: {exc}")
            return f"error: {exc}"
    return "draft_saved_mock"



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


# ---------------------------------------------------------------------------
# LLM actions
# ---------------------------------------------------------------------------

@llm(
    system="""
You are an email triage assistant.
Classify the email as exactly one of five labels:
  spam         — unsolicited or promotional
  reply        — needs a written reply (any complexity)
  scheduling   — requests a meeting or asks about availability
  cancellation — requests cancellation of an existing meeting or event
Reply with the single label and nothing else.
""".strip(),
    user="{email}",
    parse="text",
    outputs=(("route", str),),
)
def classify_email(email: str) -> None: ...


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
You are an assistant that classifies commands sent to a personal Telegram bot.
All messages are from the bot owner giving instructions. Classify as exactly one of five labels:
  schedule_meeting — schedule a meeting with someone (creates event + drafts invitation)
  cancel_meeting   — cancel or remove an existing calendar event
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
Draft a polite meeting cancellation email.
Format your response as:
  To: <attendee name or email if identifiable, otherwise leave blank>
  Subject: Cancellation: <event name>

  <brief, polite notice that the meeting has been cancelled, 1-3 sentences>
""".strip(),
    user="""
Original request: {chat_msg}

Cancellation result: {chat_cancel_status}
""".strip(),
    parse="text",
    outputs=(("draft", str),),
)
def write_cancellation_email(chat_msg: str, chat_cancel_status: str) -> None: ...


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
    submit_label="Save draft →",
    cancel_label="Skip",
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
    context="{chat_msg}",
    prefill="{chat_draft}",
    instruction="Edit the reply or submit as-is",
    outputs=["chat_edit: str"],
    submit_label="Send →",
    cancel_label="Skip",
)
def approve_or_edit_chat(chat_msg: str, chat_draft: str): pass


@human(
    kind="edit",
    context="{chat_msg}",
    prefill="{chat_email_draft}",
    instruction="Edit the email draft or approve as-is",
    outputs=["chat_email_edit: str"],
    submit_label="Save draft",
    cancel_label="Skip",
)
def approve_email_draft_from_chat(chat_msg: str, chat_email_draft: str): pass


@human(
    kind="confirm",
    context="{chat_event_details}",
    instruction="Create this calendar event?",
    outputs=["chat_confirmed_event: bool"],
    submit_label="Create →",
    cancel_label="Skip",
)
def confirm_event(chat_event_details: str): pass



# ---------------------------------------------------------------------------
# Fragments — reusable coordination sub-sequences
# ---------------------------------------------------------------------------

@fragment
def approve_email_reply(email, reply):
    Writer(email, reply) >> User(email, reply)
    User: edit = approve_or_edit(email, reply)
    if edit @ User:
        User(email, edit) >> Mailbox(email, edit)
        Mailbox: mail_status = create_draft(email, edit)


@fragment
def approve_chat_reply(chat_msg, chat_draft):
    Writer(chat_msg, chat_draft) >> User(chat_msg, chat_draft)
    User: chat_edit = approve_or_edit_chat(chat_msg, chat_draft)
    if chat_edit @ User:
        User(chat_msg, chat_edit) >> Chat(chat_msg, chat_edit)
        Chat: chat_status = send_chat_reply(chat_msg, chat_edit)


@fragment
def create_calendar_event(source_msg, choice):
    Calendar: today = todays_date()
    Calendar: event_details = extract_event_details(today, source_msg, choice)
    Calendar: sched_event = create_scheduled_event(event_details)
    Calendar: event_summary = format_event_confirmation(event_details)
    Calendar(event_summary) >> User(event_summary)
    User: ack = acknowledge_event_created(event_summary)


@fragment
def reply_branch(email):
    Dispatcher(email) >> Writer(email)
    Writer: reply = write_draft(email)
    approve_email_reply(email, reply)


@fragment
def scheduling_branch(email):
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
    User(email, choice, reply) >> Calendar(email, choice, reply)
    Calendar: sched_context = remember_scheduling_reply(email, reply, sched_context)
    create_calendar_event(email, choice)



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
    Calendar: chat_cancel_query = extract_cancel_query(chat_msg)
    Calendar: chat_cancel_matches = find_matching_events(chat_cancel_query)
    Calendar(chat_msg, chat_cancel_matches) >> User(chat_msg, chat_cancel_matches)
    User: chat_cancel_confirm = confirm_cancellation(chat_cancel_matches)
    if chat_cancel_confirm @ User:
        Calendar: chat_cancel_status = do_delete_event(chat_cancel_matches)
        Calendar(chat_msg, chat_cancel_status) >> Writer(chat_msg, chat_cancel_status)
        Writer: chat_email_draft = write_cancellation_email(chat_msg, chat_cancel_status)
        Writer(chat_msg, chat_email_draft) >> User(chat_msg, chat_email_draft)
        User: chat_email_edit = approve_email_draft_from_chat(chat_msg, chat_email_draft)
        if chat_email_edit @ User:
            User(chat_msg, chat_email_edit) >> Mailbox(chat_msg, chat_email_edit)
            Mailbox: chat_mail_status = create_draft_from_instruction(chat_msg, chat_email_edit)
            Mailbox(chat_mail_status) >> Chat(chat_mail_status)
            Chat: chat_status = send_chat_reply(chat_msg, chat_mail_status)
        else:
            User(chat_msg, chat_cancel_status) >> Chat(chat_msg, chat_cancel_status)
            Chat: chat_status = send_chat_reply(chat_msg, chat_cancel_status)
    else:
        User: chat_cancel_status = skip_cancellation()
        User(chat_msg, chat_cancel_status) >> Chat(chat_msg, chat_cancel_status)
        Chat: chat_status = send_chat_reply(chat_msg, chat_cancel_status)


@fragment
def schedule_meeting_from_chat(chat_msg):
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
        User: chat_email_edit = approve_email_draft_from_chat(chat_msg, chat_email_draft)
        if chat_email_edit @ User:
            User(chat_msg, chat_email_edit) >> Mailbox(chat_msg, chat_email_edit)
            Mailbox: chat_mail_status = create_draft_from_instruction(chat_msg, chat_email_edit)
            Mailbox(chat_mail_status) >> Chat(chat_mail_status)
            Chat: chat_status = send_chat_reply(chat_msg, chat_mail_status)
        else:
            User(chat_msg, chat_sched_event) >> Chat(chat_msg, chat_sched_event)
            Chat: chat_status = send_chat_reply(chat_msg, chat_sched_event)
    else:
        User: chat_sched_event = decline_creation()
        User(chat_msg, chat_sched_event) >> Chat(chat_msg, chat_sched_event)
        Chat: chat_status = send_chat_reply(chat_msg, chat_sched_event)



@fragment
def create_event_from_chat(chat_msg):
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
    User: chat_email_edit = approve_email_draft_from_chat(chat_msg, chat_email_draft)
    if chat_email_edit @ User:
        User(chat_msg, chat_email_edit) >> Mailbox(chat_msg, chat_email_edit)
        Mailbox: chat_mail_status = create_draft_from_instruction(chat_msg, chat_email_edit)
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
            while True @ Mailbox:
                if mail_present() @ Mailbox:
                    Mailbox: email = pop_pending_email()
                    Mailbox(email) >> Dispatcher(email)
                    Dispatcher: route = classify_email(email)
                    Dispatcher: route = normalize_route(route)

                    if (route == "spam") @ Dispatcher:
                        Dispatcher(email) >> Mailbox(email)
                        Mailbox: mail_status = mark_as_spam(email)

                    elif (route == "scheduling") @ Dispatcher:
                        scheduling_branch(email)

                    elif (route == "cancellation") @ Dispatcher:
                        cancellation_branch(email)

                    else:
                        reply_branch(email)

                else:
                    Mailbox: _ = wait_briefly()

        with branch:
            # ── Chat stream ───────────────────────────────────────────────
            # All Telegram messages are commands from the bot owner.
            # Telegram is the control interface; side effects go to the
            # right lifeline (Calendar, Mailbox).
            while True @ Chat:
                if chat_present() @ Chat:
                    Chat: chat_msg = pop_pending_chat()
                    Chat(chat_msg) >> Dispatcher(chat_msg)
                    Dispatcher: chat_route = classify_chat(chat_msg)
                    Dispatcher: chat_route = normalize_route(chat_route)

                    if (chat_route == "schedule_meeting") @ Dispatcher:
                        Dispatcher(chat_msg) >> Calendar(chat_msg)
                        schedule_meeting_from_chat(chat_msg)
                    elif (chat_route == "cancel_meeting") @ Dispatcher:
                        Dispatcher(chat_msg) >> Calendar(chat_msg)
                        cancellation_chat_branch(chat_msg)
                    elif (chat_route == "create_event") @ Dispatcher:
                        Dispatcher(chat_msg) >> Calendar(chat_msg)
                        create_event_from_chat(chat_msg)
                    elif (chat_route == "draft_email") @ Dispatcher:
                        Dispatcher(chat_msg) >> Writer(chat_msg)
                        draft_email_from_chat(chat_msg)
                    else:
                        Dispatcher(chat_msg) >> Writer(chat_msg)
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

    elif "--openai" in sys.argv:
        import importlib.util
        import os
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
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        print(f"Using OpenAI model: {model}")
        backend = make_openai_backend(api_key=api_key, model=model, max_tokens=1024)
        command_center.configure(backend=backend, ui=True, timeout=3600)

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
        backend = make_openai_backend(
            api_key="ollama",
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434/v1",
            max_tokens=512,
            timeout=120,
        )
        command_center.configure(backend=backend, ui=True, timeout=3600, show_decisions=False)

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
