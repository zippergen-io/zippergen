# Connector helpers

Use this reference when writing effects for Gmail, Google Sheets, Google Calendar
or Telegram.
These are the supported Python helpers, not raw provider clients. Import them
from `zippergen` and construct them inside an effect with
`Class.from_requirement("logical-name")`. The name must match a declared
`ConnectorRequirement` and the effect's `connector=`. Keep credentials and
resource IDs in connector configuration.

## GmailMailbox

Declare `kind="gmail"`. Reads can use `access="read-only"`. Draft creation,
sending and marking processed require write access.

| Method | Return | Capability / effect operation |
|---|---|---|
| `inspect()` | `{"email": str, "messages": int, "threads": int}` | Configuration check |
| `count_unread()` | `int`, Gmail's estimated number of query matches | `count-unread` |
| `fetch_one_unread()` | Message dictionary below, or `None` | `fetch-one-unread` |
| `mark_processed(meta: dict[str, object] \| str)` | `None` | `mark-processed` |
| `create_draft(meta: dict[str, object], subject: str, body: str)` | Gmail draft ID, `str` | `create-draft` |
| `send_email(meta: dict[str, object], subject: str, body: str)` | Sent Gmail message ID, `str` | `send-message` |

Despite their names, `count_unread` and `fetch_one_unread` use the configured
query as written. They do not append `is:unread`. The default query is
`is:unread in:inbox`. Fetching neither claims nor modifies a message. A positive
count does not guarantee the next fetch returns a message, so handle `None`.

The fetched dictionary always has these keys. Missing text headers become
empty strings:

| Keys | Type and meaning |
|---|---|
| `gmail_id` | `str`, Gmail message ID. Use this to identify the incoming message. |
| `thread_id` | `str`, Gmail thread ID |
| `message_id` | `str`, RFC Message-ID header, distinct from `gmail_id` |
| `internal_date_ms` | `int \| None`, Gmail timestamp in milliseconds |
| `date` | `str`, untrusted sender-supplied Date header |
| `from`, `sender` | `str`, identical copies of the From header |
| `to`, `cc`, `delivered_to`, `x_original_to`, `envelope_to` | `str`, address headers |
| `in_reply_to`, `references` | `str`, threading headers |
| `subject`, `body` | `str`, subject and extracted message text |

Use `internal_date_ms` when comparing arrival times, with a policy for `None`.
Do not use the sender's `date` as trusted inbox ordering. `body` is extracted
text, not an attachment collection or the full MIME message. The helper does
not return a Reply-To header. Replies currently target `sender`/`from`.

Pass the fetched dictionary as `meta` to preserve the thread and reply headers.
`create_draft` and `send_email` add `Re:` when needed. `mark_processed` accepts
that dictionary or its `gmail_id` string. It **only removes the UNREAD label**.
It does not archive the message, delete it or add a custom processed label.
There is no public add-label, arbitrary-ID fetch or paginated message-list
helper. Do not invent one. If preserving unread mail prevents progress with a
one-message fetch, explain the limitation and agree on a selection strategy
before adding custom Gmail API code.

Removing UNREAD can be repeated, but it also overrides someone marking the
message unread again. Sending and draft creation have no idempotency key.
If a request succeeds remotely but its result is not persisted, replay can
send another email or create another draft. Recording success after sending
does not close that window. For a non-idempotent write, an application can
persist a pending operation before the call and stop for reconciliation or
human review when the outcome is uncertain. A Gmail lookup or other evidence
must establish the outcome before retrying. Do not silently treat an uncertain
send as either success or failure.

## GoogleSheetsTable

Declare `kind="google-sheets"`. Reads can use `access="read-only"`. Upserts and
replacement require write access. Configure the spreadsheet ID and tab outside
workflow code. Define `columns` as an ordered sequence of unique, non-empty
column names, such as `("request_id", "status", "notes")`.

Prefer these top-level helpers for JSON text passed through workflow variables:

| Helper | Return |
|---|---|
| `read_json_rows(requirement: str, *, columns: Sequence[str])` | `str`, a JSON array of row objects, not a Python list |
| `upsert_json_row(requirement: str, record_json: str, *, columns: Sequence[str], key_field: str)` | `str`, `"created"` or `"updated"` |

Import both helpers from `zippergen`. `record_json` must encode one JSON object,
not an array. The [authoring reference](dsl-and-cli.md) shows their use inside
effects with a declared connector requirement.

For Python row dictionaries, use
`GoogleSheetsTable.from_requirement("logical-name")` and these methods:

| Method | Return |
|---|---|
| `inspect()` | `{"title": str, "tab": str, "tabs": list[str]}`. Raises if the configured tab does not exist. |
| `read_rows(columns: Sequence[str])` | `list[dict[str, object]]` |
| `upsert_row(record: Mapping[str, object], *, columns: Sequence[str], key_field: str)` | `str`, `"created"` or `"updated"` |
| `replace_rows(rows: Sequence[Mapping[str, object]], *, columns: Sequence[str])` | `None` |

Reads cover columns starting at A on the configured tab. Row 1 must match
`columns` exactly in that range, in the same order. An empty range returns
`[]`. Data rows become dictionaries keyed by column name, with missing cells
filled by `""`. Blank row positions are preserved. Reads request unformatted
values, so application code must handle the returned cell types rather than
assume every cell is a string. JSON-looking cell text is not automatically
decoded into a nested object.

Writes use RAW input. Strings are not interpreted as formulas. `None` becomes
an empty cell, scalar strings/numbers/booleans are retained, and other values
are JSON-encoded into cell text. Only declared columns are written. Extra
record keys are ignored and missing fields become empty cells, so an upsert
replaces the managed row rather than patching only the supplied fields.

`key_field` must name a declared column and the record must have a non-empty
key. Upsert compares keys as strings, updates the first matching row, or
appends a new row if none matches. It writes the header if the read has no
data rows. Keep keys stable and unique, and use one writer for the managed
table. Read-then-write is not atomic: concurrent inserts can duplicate keys,
and row insertion or sorting between the read and write can change the target.
A sequential retry can find a row already written under the same key, but may
return `"updated"` after an earlier `"created"`. Do not use that status as an
exactly-once trigger for another external effect.

`replace_rows` validates the replacement before clearing the managed columns,
then writes the header and new rows. Clear and write are separate requests.
A failure between them can leave the table empty. Resume with the same saved
replacement data, with exclusive ownership of the managed columns, rather
than reconstructing the replacement from the now-empty sheet. There are no
public append-only, delete-row or formatting helpers. Do not call the private
`_append` or `_update` methods from workflows.

## GoogleCalendar

Declare `kind="google-calendar"`. Reading needs `list-events` and can use
`access="read-only"`. Creation needs `create-event` and write access.

```python
calendar = GoogleCalendar.from_requirement("agenda")
calendar.inspect()  # {"calendar_id": str, "title": str, "time_zone": str}
calendar.list_events(start, end, max_events=10000)  # list[dict[str, Any]]
calendar.create_event(
    request_id,
    summary=summary,
    start=start,
    end=end,
    description="",
    location="",
    attendees=(),
)  # str: Google event ID
```

`start` and `end` are RFC3339 strings with an explicit UTC offset or `Z`, and
`end` must be later than `start`. `list_events` returns events overlapping
`[start, end)`, expands recurring events and follows pagination. It raises if
the result exceeds `max_events`, rather than silently truncating it.

Events are Google's dictionaries, not normalized busy intervals. Fields can
be absent. Common fields are `id`, `summary`, `status`, `transparency`,
`start`, `end` and `attendees`. Timed events use
`start/end: {"dateTime": "...", "timeZone": "..."}` with optional `timeZone`.
All-day events instead use `{"date": "YYYY-MM-DD"}`, with an exclusive end
date. Convert them using the calendar's time zone and decide how transparent
events or declined invitations affect your busy-time policy. Reading does not
reserve slots or check all attendees' calendars.

`create_event` creates a single timed event. All arguments are strings except
`attendees`, a sequence of email strings. Attendees receive invitations.
Generate one stable `request_id` per intended meeting before the effect and
retain it with the approved proposal. Retrying the same ID and proposal on the
same calendar recovers the original creation. Changed content or a cancelled
event is rejected. Later manual edits are not undone. This is not an atomic
availability check and reservation, nor an unconditional exactly-once
notification guarantee. Serialize attempts for the same request. There are no
update-event, delete-event or free/busy helpers.

## TelegramChat

For one-way announcements, declare `kind="telegram"` with write access:

```python
chat = TelegramChat.from_requirement("alerts")
chat.send("Processing finished.")  # None, not a Telegram message ID
```

`send(text: str) -> None` rejects empty text and posts to the configured chat.
It does not collect an answer and has no idempotency key, so a replay can
duplicate the announcement. Do not use this helper to implement an approval
protocol. Declare a `@human` action and route its participant or action with
`zg connector assign` instead. That route does not need a separate
`ConnectorRequirement` just to deliver human tasks.
