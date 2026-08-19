"""A configured connector must actually reach the deployment.

`deploy` reads the project configuration directly and snapshots the resulting
routing without exposing credentials.

The invariant throughout: the snapshot is durable routing and is committed with
the deployment; the credential values live only in the environment.
"""

import json
import shutil
from pathlib import Path

import pytest

from zippergen.connector_wiring import (
    ConnectorWiringError,
    connector_runtime,
    human_connector_factory,
)
from zippergen.serve import load_workflow_spec
from zippergen.workspace import Workspace

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "email_approval.py"
ENTRY = "workflow.py:email_approval"
TOKEN = "secret-bot-token"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="demo")
    return workspace


def _wire(workspace):
    workflow, module = load_workflow_spec(
        str(workspace.absolute_spec(ENTRY))
    )
    return connector_runtime(workspace, ENTRY, workflow, module)


def _telegram(workspace, name="approvals", chat="4242"):
    if "approval-bot" not in workspace.provider_connections():
        workspace.save_provider_connection("approval-bot", {"kind": "telegram"})
    workspace.save_connector_configuration(
        name,
        {"connection": "approval-bot", "kind": "telegram", "chat_id": chat},
    )
    workspace.save_provider_secret("approval-bot", "bot_token", TOKEN)


def test_a_project_with_no_connectors_wires_nothing(project):
    assert _wire(project) == ({}, {})


def test_an_assigned_participant_is_routed_to_its_chat(project):
    _telegram(project)
    project.save_connector_assignment_profile(
        ENTRY, lifelines={"Mailbox": "approvals"}, actions={}
    )

    snapshot, environment = _wire(project)

    route = snapshot["human:Mailbox"]
    assert route["kind"] == "telegram"
    assert route["connection"] == "approval-bot"
    assert route["chat_id"] == "4242"
    assert route["participant"] == "Mailbox"
    assert environment[route["token_env"]] == TOKEN


def test_the_token_is_never_in_the_snapshot(project):
    """The snapshot ships with the deployment; the token must not."""

    _telegram(project)
    project.save_connector_assignment_profile(
        ENTRY, lifelines={"Mailbox": "approvals"}, actions={}
    )

    snapshot, environment = _wire(project)

    assert TOKEN not in json.dumps(snapshot)
    assert TOKEN in environment.values()


def test_a_single_action_can_be_routed_on_its_own(project):
    _telegram(project)
    project.save_connector_assignment_profile(
        ENTRY, lifelines={}, actions={"Mailbox.approve_reply": "approvals"}
    )

    snapshot, _environment = _wire(project)

    route = snapshot["human:Mailbox.approve_reply"]
    assert route["participant"] == "Mailbox"
    assert route["action"] == "approve_reply"


def test_assigning_a_participant_with_no_human_action_is_refused(project):
    """The Writer has no `@human` action, so it cannot be asked anything."""

    _telegram(project)
    project.save_connector_assignment_profile(
        ENTRY, lifelines={"Writer": "approvals"}, actions={}
    )

    with pytest.raises(ConnectorWiringError, match="no human action"):
        _wire(project)


def test_a_missing_token_is_refused_before_deploying(project):
    project.save_provider_connection("approval-bot", {"kind": "telegram"})
    project.save_connector_configuration(
        "approvals",
        {"connection": "approval-bot", "kind": "telegram", "chat_id": "1"},
    )
    project.save_connector_assignment_profile(
        ENTRY, lifelines={"Mailbox": "approvals"}, actions={}
    )

    with pytest.raises(ConnectorWiringError, match="bot token.*is missing"):
        _wire(project)


def test_distinct_telegram_connections_build_distinct_pollers(tmp_path):
    snapshot = {
        "human:User.approve": {
            "type": "human",
            "target": "User.approve",
            "configuration": "approvals",
            "connection": "approval-bot",
            "chat_id": "1",
            "token_env": "TOKEN_A",
        },
        "human:User.notify": {
            "type": "human",
            "target": "User.notify",
            "configuration": "notifications",
            "connection": "notification-bot",
            "chat_id": "2",
            "token_env": "TOKEN_B",
        },
    }
    factory = human_connector_factory(
        snapshot, {"TOKEN_A": "secret-a", "TOKEN_B": "secret-b"}
    )
    assert factory is not None

    group = factory(str(tmp_path / "store.sqlite"))

    assert [notifier.connection for notifier in group.notifiers] == [
        "approval-bot",
        "notification-bot",
    ]


def test_a_non_human_connector_cannot_answer_a_human_action(project):
    project.save_provider_connection("google-work", {"kind": "google"})
    project.save_connector_configuration(
        "records",
        {
            "connection": "google-work",
            "kind": "google-sheets",
            "spreadsheet_id": "1",
            "tab": "T",
        },
    )
    project.save_provider_secret(
        "google-work", "authorized_user_json", "{}"
    )
    project.save_connector_assignment_profile(
        ENTRY, lifelines={"Mailbox": "records"}, actions={}
    )

    with pytest.raises(ConnectorWiringError, match="ask a person"):
        _wire(project)


def test_reconfiguring_a_deployment_ignores_the_surrounding_project(
    project, tmp_path, monkeypatch
):
    """A deployment carries its own workflow; the ambient directory is not it.

    `zg configure NAME` names an existing deployment. Wiring it from whatever
    project the shell is standing in attaches the wrong connectors — or refuses
    over requirements the deployment never had.
    """

    from zippergen import serve

    _telegram(project)
    project.save_connector_assignment_profile(
        ENTRY, lifelines={"Mailbox": "approvals"}, actions={}
    )

    other = tmp_path / "unrelated"
    other.mkdir()
    shutil.copy(EXAMPLE, other / "workflow.py")
    unrelated = Workspace(other, home=tmp_path / "home")
    unrelated.initialize_project(name="unrelated")
    monkeypatch.chdir(other)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    class Args:
        target = "some-deployment"
        project = None
        connectors_json = None

    # Standing in `unrelated`, the deployment's own workflow must still win.
    snapshot, _environment = serve._project_connector_runtime(
        Args(),
        deployed_workflow=ENTRY,
        deployed_project=str(project.root),
    )

    assert "human:Mailbox" in snapshot


def test_a_default_routes_every_participant_that_asks_a_human(project):
    """One chat for everything is the common case, so it must be sayable once."""

    _telegram(project)
    project.save_connector_assignment_profile(
        ENTRY, default="approvals", lifelines={}, actions={}
    )

    snapshot, environment = _wire(project)

    route = snapshot["human:Mailbox"]
    assert route["configuration"] == "approvals"
    assert route["chat_id"] == "4242"
    assert environment[route["token_env"]] == TOKEN


def test_a_named_participant_beats_the_default(project):
    """The default catches what nothing more specific claimed."""

    _telegram(project)
    _telegram(project, name="escalations", chat="99")
    project.save_connector_assignment_profile(
        ENTRY,
        default="approvals",
        lifelines={"Mailbox": "escalations"},
        actions={},
    )

    snapshot, _environment = _wire(project)

    assert snapshot["human:Mailbox"]["configuration"] == "escalations"
    assert snapshot["human:Mailbox"]["chat_id"] == "99"
