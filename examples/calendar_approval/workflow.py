"""Review a proposed meeting and existing events before creating it."""

from zippergen import (
    ConnectorRequirement, DeploymentField, DeploymentSpec, GoogleCalendar,
    Json, Lifeline, effect, human, pure, workflow,
)

Requester = Lifeline("Requester")
Calendar = Lifeline("Calendar")

zippergen_connectors = (
    ConnectorRequirement(
        "agenda", "google-calendar", "Calendar",
        capabilities=("list-events", "create-event"), access="read-write",
        description="Read the proposed time range and create the approved meeting.",
    ),
)
zippergen_deployment = DeploymentSpec(
    description="Review one meeting proposal and create it after approval.",
    fields=(DeploymentField("proposal", "Meeting proposal (JSON)", target="input", required=True),),
    files=("workflow.py",),
)


@effect(connector="agenda", operation="list-events")
def existing_events(proposal: Json) -> Json:
    return GoogleCalendar.from_requirement("agenda").list_events(proposal["start"], proposal["end"])


@human(
    kind="confirm",
    instruction="Create this meeting? Attendees will receive invitations. The time is not reserved.",
    context="Proposed meeting: {proposal}\nExisting events: {events}",
    outputs=["approved: bool"],
)
def approve_meeting(proposal: Json, events: Json) -> None: ...


@effect(connector="agenda", operation="create-event")
def create_meeting(proposal: Json) -> str:
    return GoogleCalendar.from_requirement("agenda").create_event(
        proposal["request_id"], summary=proposal["summary"],
        start=proposal["start"], end=proposal["end"],
        description=proposal.get("description", ""), location=proposal.get("location", ""),
        attendees=proposal.get("attendees", []),
    )


@pure
def rejected() -> str:
    return "rejected"


@workflow
def calendar_approval(proposal: Json @ Requester) -> str:
    Requester(proposal) >> Calendar(proposal)
    Calendar: events = existing_events(proposal)
    Calendar(events) >> Requester(events)
    Requester: approved = approve_meeting(proposal, events)
    if approved @ Requester:
        Calendar: result = create_meeting(proposal)
        Calendar(result) >> Requester(result)
    else:
        Requester: result = rejected()
    return result @ Requester
