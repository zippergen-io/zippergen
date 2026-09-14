import json
import subprocess
import sys

import pytest

from zippergen import GoogleCalendar, GoogleCalendarError, run
from zippergen.google_auth import google_scopes_cover, google_scope_for_access, parse_google_scopes

START = "2026-10-01T14:00:00+02:00"
END = "2026-10-01T14:30:00+02:00"
PROPOSAL = dict(summary="Review", start=START, end=END)


class Response:
    def __init__(self, status, value):
        self.status_code, self.value = status, value

    def json(self):
        return self.value


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return next(self.responses)

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        response = next(self.responses)
        if response is None:
            return Response(200, kwargs["json"])
        return response


@pytest.fixture
def calendar():
    return GoogleCalendar("agenda", "private/calendar@example.org", "secret", "read-write")


def test_binding_and_scope(monkeypatch):
    monkeypatch.setenv("CALENDAR_SECRET", "private")
    monkeypatch.setenv("ZIPPERGEN_CONNECTORS_JSON", json.dumps({"requirement:agenda": {
        "kind": "google-calendar", "calendar_id": "primary", "credential_env": "CALENDAR_SECRET",
    }}))
    calendar = GoogleCalendar.from_requirement("agenda")
    assert calendar.calendar_id == "primary"
    assert calendar.access == "read-only"
    read = google_scope_for_access("google-calendar", "read-only")
    write = google_scope_for_access("google-calendar", "read-write")
    assert parse_google_scopes("calendar.events") == (write,)
    assert google_scopes_cover([write], [read])
    assert not google_scopes_cover([read], [write])
    assert not google_scopes_cover(parse_google_scopes("spreadsheets"), [read])
    monkeypatch.delenv("CALENDAR_SECRET")
    with pytest.raises(GoogleCalendarError, match="credential is missing"):
        GoogleCalendar.from_requirement("agenda")


def test_list_follows_empty_pages_and_preserves_all_day_events(calendar, monkeypatch):
    event = {"id": "1", "start": {"date": "2026-10-01"}}
    session = Session(Response(200, {"nextPageToken": "next"}), Response(200, {"items": [event]}))
    monkeypatch.setattr(calendar, "_session", lambda: session)
    assert calendar.list_events(START, END) == [event]
    assert "%2F" in session.calls[0][1]
    assert session.calls[0][2]["params"]["singleEvents"] == "true"
    assert session.calls[1][2]["params"]["pageToken"] == "next"


@pytest.mark.parametrize("responses,limit", [
    ([Response(200, {"items": [{}, {}]})], 1),
    ([Response(200, {"items": "bad"})], 10),
    ([Response(200, {"nextPageToken": "same"})] * 2, 10),
    ([Response(403, {"secret": "should not appear"})], 10),
])
def test_list_never_returns_partial_or_failed_results(calendar, monkeypatch, responses, limit):
    monkeypatch.setattr(calendar, "_session", lambda: Session(*responses))
    with pytest.raises(GoogleCalendarError) as error:
        calendar.list_events(START, END, max_events=limit)
    assert "should not appear" not in str(error.value)


@pytest.mark.parametrize("start,end", [(START, START), (END, START),
    ("2026-10-01T14:00:00", END), ("2026-10-01", END)])
def test_invalid_times_fail_before_network(calendar, start, end):
    with pytest.raises(ValueError):
        calendar.list_events(start, end)
    with pytest.raises(ValueError):
        calendar.create_event("request", summary="Review", start=start, end=end)


def test_creation_replay_and_changed_proposal(calendar, monkeypatch):
    session = Session(Response(404, {}), None)
    monkeypatch.setattr(calendar, "_session", lambda: session)
    event_id = calendar.create_event("request", **PROPOSAL, attendees=["alice@example.org"])
    payload = session.calls[1][2]["json"]
    assert event_id == payload["id"]
    assert session.calls[1][2]["params"] == {"sendUpdates": "all"}
    recovered = Session(Response(200, payload))
    monkeypatch.setattr(calendar, "_session", lambda: recovered)
    assert calendar.create_event("request", **PROPOSAL, attendees=["alice@example.org"]) == event_id
    assert [call[0] for call in recovered.calls] == ["get"]
    monkeypatch.setattr(calendar, "_session", lambda: Session(Response(200, payload)))
    with pytest.raises(GoogleCalendarError, match="different content"):
        calendar.create_event("request", **PROPOSAL)
    payload["status"] = "cancelled"
    with pytest.raises(GoogleCalendarError, match="cancelled"):
        calendar.create_event("request", **PROPOSAL, attendees=["alice@example.org"])


def test_conflict_recovers_only_matching_creation(calendar, monkeypatch):
    first = Session(Response(404, {}), None)
    monkeypatch.setattr(calendar, "_session", lambda: first)
    expected = calendar.create_event("request", **PROPOSAL)
    payload = first.calls[1][2]["json"]
    concurrent = Session(Response(404, {}), Response(409, {}), Response(200, payload))
    monkeypatch.setattr(calendar, "_session", lambda: concurrent)
    assert calendar.create_event("request", **PROPOSAL) == expected
    assert len(concurrent.calls) == 3
    monkeypatch.setattr(calendar, "_session", lambda: Session(Response(200, {"id": expected})))
    with pytest.raises(GoogleCalendarError, match="different content"):
        calendar.create_event("request", **PROPOSAL)


def test_readonly_and_empty_request_are_rejected_before_network(calendar):
    calendar.access = "read-only"
    with pytest.raises(GoogleCalendarError, match="read-only"):
        calendar.create_event("request", **PROPOSAL)
    calendar.access = "read-write"
    with pytest.raises(ValueError, match="request ID"):
        calendar.create_event("", **PROPOSAL)


def test_network_errors_do_not_expose_request_secrets(calendar, monkeypatch):
    class Broken:
        def get(self, *args, **kwargs):
            raise RuntimeError("Authorization: private-token event: private-title")
    monkeypatch.setattr(calendar, "_session", Broken)
    with pytest.raises(GoogleCalendarError, match="could not be reached") as error:
        calendar.create_event("request", **PROPOSAL)
    assert "private" not in str(error.value)


def test_fresh_process_recovers_creation_after_crash(tmp_path):
    # The fake remote stores the successful insertion before killing the first
    # client. The next interpreter has no memory of its returned event ID.
    code = '''
import json, os, sys
from pathlib import Path
from zippergen import GoogleCalendar
store = Path(sys.argv[1])
class Response:
    def __init__(self, status, value): self.status_code, self.value = status, value
    def json(self): return self.value
class Remote:
    def get(self, *args, **kwargs):
        return Response(200, json.loads(store.read_text())) if store.exists() else Response(404, {})
    def post(self, *args, **kwargs):
        assert not store.exists(), "duplicate insertion"
        store.write_text(json.dumps(kwargs["json"]))
        os._exit(9)
calendar = GoogleCalendar("agenda", "primary", "secret", "read-write")
calendar._session = Remote
print(calendar.create_event("persisted-request", summary="Review", start="2026-10-01T12:00:00Z", end="2026-10-01T12:30:00Z"))
'''
    command = [sys.executable, "-c", code, str(tmp_path / "remote.json")]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 9, first.stderr
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == json.loads((tmp_path / "remote.json").read_text())["id"]


@pytest.mark.parametrize("approve", [True, False])
def test_workflow_creates_only_the_reviewed_proposal(monkeypatch, approve):
    from examples.calendar_approval.workflow import calendar_approval, Requester, Calendar
    calls = []
    proposal = {"request_id": "stable-1", **PROPOSAL, "attendees": ["alice@example.org"]}
    class FakeCalendar:
        def list_events(self, start, end):
            assert (start, end) == (START, END)
            return [{"summary": "Existing event"}]
        def create_event(self, request_id, **kwargs):
            calls.append((request_id, kwargs))
            return "event-1"
    monkeypatch.setattr(GoogleCalendar, "from_requirement", lambda name: FakeCalendar())
    def human(action, inputs):
        assert inputs["proposal"] == proposal
        assert inputs["events"] == [{"summary": "Existing event"}]
        return {action.output: approve}
    result = run(calendar_approval, [Requester, Calendar], {"Requester": {"proposal": proposal}},
                 human_backend=human)
    assert result == ("event-1" if approve else "rejected")
    assert calls == ([("stable-1", {**PROPOSAL, "description": "", "location": "",
                                 "attendees": ["alice@example.org"]})] if approve else [])


def test_readiness_checks_events_without_writing(calendar, monkeypatch):
    from zippergen.connectors import ConnectorRequirement, connector_kind_spec
    requirement = ConnectorRequirement("agenda", "google-calendar", "Calendar", access="read-write")
    adapter = connector_kind_spec("google-calendar")
    binding = {"calendar_id": "primary", "credential_env": "CREDENTIAL"}
    assert adapter.readiness(requirement, binding, {}, False).status == "fail"
    assert adapter.readiness(requirement, binding, {"CREDENTIAL": "private"}, False).status == "ok"
    session = Session(Response(200, {"summary": "Test calendar"}))
    monkeypatch.setattr(GoogleCalendar, "_session", lambda self: session)
    assert adapter.readiness(requirement, binding, {"CREDENTIAL": "private"}, True).status == "ok"
    assert [call[0] for call in session.calls] == ["get"]
    assert session.calls[0][1].endswith("/primary/events")


def test_deployment_wiring_keeps_calendar_credentials_private(tmp_path):
    from pathlib import Path
    from zippergen.workspace import Workspace
    from zippergen.workflow_io import load_workflow_spec
    from zippergen.connector_wiring import connector_runtime, ConnectorWiringError
    root = tmp_path / "project"
    root.mkdir()
    source = Path(__file__).resolve().parents[1] / "examples/calendar_approval/workflow.py"
    (root / "workflow.py").write_text(source.read_text())
    workspace = Workspace(root, home=tmp_path / "private")
    workspace.initialize_project()
    scope = google_scope_for_access("google-calendar", "read-write")
    workspace.save_provider_connection("google-work", {"kind": "google", "granted_scopes": json.dumps([scope])})
    workspace.save_provider_secret("google-work", "authorized_user_json", "calendar-secret")
    workspace.save_connector_configuration("my-calendar", {
        "connection": "google-work", "kind": "google-calendar", "calendar_id": "primary",
    })
    entry = "workflow.py:calendar_approval"
    workspace.bind_connector(entry, "agenda", "my-calendar")
    workflow, module = load_workflow_spec(str(root / entry))
    snapshot, environment = connector_runtime(workspace, entry, workflow, module)
    route = snapshot["requirement:agenda"]
    assert route["calendar_id"] == "primary"
    assert environment[route["credential_env"]] == "calendar-secret"
    assert "calendar-secret" not in json.dumps(snapshot)
    assert "calendar-secret" not in (root / "zippergen.toml").read_text()
    workspace.save_provider_connection("google-work", {
        "kind": "google", "granted_scopes": json.dumps(parse_google_scopes("calendar.events.readonly")),
    })
    with pytest.raises(ConnectorWiringError, match="does not cover"):
        connector_runtime(workspace, entry, workflow, module)
