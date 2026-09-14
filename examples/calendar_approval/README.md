# Create a Google Calendar event after approval

This example reads the proposed time range, shows the proposal and existing
events, and asks you before creating the event. It needs no model or OpenAI key.
It uses real Google Calendar data. Start with a test calendar and no attendees.

Run these commands from this directory, using a ZipperGen installation that
includes the Calendar connector and the `google` extra.

```sh
zg init
zg workflow select workflow.py:calendar_approval
zg provider configure google-work google
zg connector configure work-calendar google-work google-calendar --calendar-id primary
zg connector assign agenda work-calendar
zg provider authorize google-work
zg connector check --strict
```

If you already have a Google connection, reuse its name instead of
`google-work`. Enable the Google Calendar API in the Google Cloud project that
owns your OAuth Desktop app client. Authorization needs Calendar permission
even if Gmail or Sheets already work. `zg provider authorize` asks for the
scopes required by the selected workflow and its assigned connectors. Keep
credentials in the terminal setup, never in workflow code or chat.

Use a calendar's ID from its Google Calendar settings instead of `primary` to
select a test or shared calendar. Your Google account must have access to it.
The connector asks for `calendar.events.readonly` for a reader, or
`calendar.events` when event creation is declared. Live readiness checks read
the calendar without creating an event. They do not prove write permission.

For a server without a browser, run `zg provider authorize google-work
--handoff` on your computer with the project configuration, then
`zg provider accept google-work` in the server's terminal.

## Try a proposal

Choose a future time and a new request ID for your own test. This command reads
the calendar and asks for approval before writing:

```sh
zg run --input 'proposal={"request_id":"calendar-demo-1","summary":"ZipperGen test","start":"2026-10-01T14:00:00+02:00","end":"2026-10-01T14:30:00+02:00"}'
```

For a persistent run, add `--durable`. To use Telegram, assign an existing
Telegram connector to `Requester`. For a managed service, use `zg deploy` and
enter the proposal when prompted. This example handles one proposal per run.

## Connector API

Call the connector inside an `@effect` with `connector="agenda"` and an explicit
operation, as shown in `workflow.py`:

```python
calendar = GoogleCalendar.from_requirement("agenda")
events = calendar.list_events(start, end)
event_id = calendar.create_event(
    request_id, summary="Project meeting", start=start, end=end,
    attendees=("colleague@example.org",),
)
```

Reading follows all result pages and expands recurring events. All-day entries
retain Google's `date` fields. An excessive result raises an error instead of
silently omitting events. Creation supports single timed events with explicit
UTC offsets. Recurring creation, all-day creation, event updates, deletion and
Meet links are outside this first version.

Attendees receive invitations. The approval must therefore cover both the
meeting details and the attendee list. Reading a time range does not reserve it.

## Recovery

The connector derives a Google event ID from the stable request ID and records
a fingerprint of the original proposal in the event's private properties.
A retry looks up that ID. It returns the existing ID if the creation receipt
matches, and refuses conflicting content or a cancelled event. Never generate
a new request ID inside the creation effect. Persist it with the proposal.
Changing the calendar or Google identity during recovery can target a different
resource and is not safe.

The receipt records the original creation. It does not prove that nobody has
edited the event since then, and replay does not undo later edits. Google does
not promise immediate detection of concurrent ID collisions, so this is not an
unconditional exactly-once guarantee. Serialize creation attempts for the same
request. Invitation delivery is also controlled by Google.

Google's references: [event creation](https://developers.google.com/workspace/calendar/api/v3/reference/events/insert)
and [event listing](https://developers.google.com/workspace/calendar/api/v3/reference/events/list).
