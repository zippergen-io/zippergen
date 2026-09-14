# Calendar approval

The Requester supplies one meeting proposal with a stable request_id, summary,
start, end, and optional description, location and attendee email addresses.
Start and end include a UTC offset. The Calendar participant reads existing
Google Calendar events that overlap the proposed interval. The Requester sees
both the exact proposal and the existing events, then approves or rejects it.

Approval allows Calendar to create that same proposal. Rejection creates
nothing. The workflow returns the Google event ID or `rejected`. Attendees
receive invitations when an event is created. No model generates the proposal
in this first example.

A retry uses the same request ID and proposal, including after a restart.
The connector recovers a prior creation rather than generating another ID.
An ID reused for different content, a cancelled event, or an unverifiable
existing event stops execution with an error. The request ID must be unique
for each intended meeting on the target calendar. Keep the configured calendar
and Google identity unchanged during recovery.

The event list is an observation, not a reservation. Other people can change
the calendar while approval is pending, and overlapping meetings are allowed.
This example does not promise that the slot remains free or check other
attendees' availability. Approval is for the exact proposal, not a future
model revision. Calendar access and event visibility depend on Google permissions.
