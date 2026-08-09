"""Configuring a connector must not need the deleted Studio shell.

This is what the Studio removal genuinely cost: creating and binding
connectors lived in `studio_connectors.py` and nothing in the CLI replaced it.
Telegram blocked the tutorial; Gmail and Sheets blocked `call_intake`. All
three are back as ordinary commands over the same workspace methods.

The split is the same everywhere: portable fields — which chat, which
spreadsheet, which mailbox query — are committed to `zippergen.toml`;
credentials are not.
"""

import subprocess
import sys
import tomllib
import os
from pathlib import Path

from zippergen.workspace import Workspace


def _run(directory: Path, *arguments: str, token: str = "bot-token"):
    environment = dict(os.environ)
    environment["ZIPPERGEN_HOME"] = str(directory.parent / "home")
    return subprocess.run(
        [sys.executable, "-m", "zippergen.serve", *arguments],
        capture_output=True,
        text=True,
        cwd=directory,
        input=f"{token}\n",
        env=environment,
        check=False,
    )


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(
        [sys.executable, "-m", "zippergen.serve", "init"],
        cwd=root,
        capture_output=True,
        env={**os.environ, "ZIPPERGEN_HOME": str(tmp_path / "home")},
        check=True,
    )
    return root


def test_it_saves_a_configuration_the_project_can_commit(tmp_path):
    root = _project(tmp_path)

    result = _run(root, "connector", "configure", "telegram", "approvals", "--chat-id", "4242")

    assert result.returncode == 0, result.stderr
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    configuration = manifest["connectors"]["configurations"]["approvals"]
    assert configuration == {
        "provider": "telegram",
        "kind": "telegram",
        "chat_id": "4242",
    }


def test_the_bot_token_never_reaches_the_manifest(tmp_path):
    """The chat id is portable; the token belongs to one machine."""

    root = _project(tmp_path)

    _run(root, "connector", "configure", "telegram", "approvals",
         "--chat-id", "4242", token="secret-bot-token")

    assert "secret-bot-token" not in (root / "zippergen.toml").read_text()
    # It is readable back on this machine, which is where it belongs.
    assert Workspace(root, home=root.parent / "home").connector_provider_secret(
        "telegram", "bot_token"
    ) == "secret-bot-token"


def test_the_token_is_not_passed_as_an_argument(tmp_path):
    """It is typed, so it stays out of shell history and any transcript."""

    root = _project(tmp_path)

    result = _run(root, "connector", "configure", "telegram", "approvals", "--chat-id", "1")

    assert "--token" not in result.stderr
    assert "hidden" in result.stdout or "hidden" in result.stderr


def test_an_existing_token_is_reused_rather_than_asked_for_again(tmp_path):
    root = _project(tmp_path)
    _run(root, "connector", "configure", "telegram", "one", "--chat-id", "1", token="tok")

    second = _run(root, "connector", "configure", "telegram", "two", "--chat-id", "2", token="")

    assert second.returncode == 0, second.stderr
    assert "already saved" in second.stdout


def test_binding_without_a_workflow_says_so(tmp_path):
    root = _project(tmp_path)

    result = _run(root, "connector", "configure", "telegram", "approvals",
                  "--chat-id", "1", "--bind", "human-approval")

    assert result.returncode != 0
    assert "none was found" in result.stderr.lower()


def test_a_sheets_configuration_records_the_spreadsheet_and_tab(tmp_path):
    root = _project(tmp_path)

    result = _run(root, "connector", "configure", "google-sheets", "records",
                  "--spreadsheet-id", "1AbC_xyz", "--tab", "Calls")

    assert result.returncode == 0, result.stderr
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["connectors"]["configurations"]["records"] == {
        "provider": "google",
        "kind": "google-sheets",
        "spreadsheet_id": "1AbC_xyz",
        "tab": "Calls",
    }


def test_a_gmail_configuration_records_the_mailbox_and_query(tmp_path):
    root = _project(tmp_path)

    result = _run(root, "connector", "configure", "gmail", "inbox",
                  "--query", "is:unread from:clients@example.com")

    assert result.returncode == 0, result.stderr
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    configuration = manifest["connectors"]["configurations"]["inbox"]
    assert configuration["kind"] == "gmail"
    assert configuration["query"] == "is:unread from:clients@example.com"
    assert configuration["account"] == "me"


def test_google_kinds_say_when_authorization_is_still_missing(tmp_path):
    """The configuration is portable; the credential is not, and it is separate."""

    root = _project(tmp_path)

    result = _run(root, "connector", "configure", "google-sheets", "records",
                  "--spreadsheet-id", "1", "--tab", "T")

    assert "connector authorize google" in result.stdout


def test_binding_works_against_a_real_workflow_requirement(tmp_path):
    """`call_intake` declares call-mailbox and call-records."""

    root = _project(tmp_path)
    example = Path(__file__).resolve().parents[1] / "examples" / "call_intake.py"
    (root / "workflow.py").write_text(example.read_text())
    Workspace(root, home=root.parent / "home").select_workflow(
        "workflow.py:call_intake", cwd=root
    )

    first = _run(root, "connector", "configure", "gmail", "inbox",
                 "--bind", "call-mailbox")
    second = _run(root, "connector", "configure", "google-sheets", "records",
                  "--spreadsheet-id", "1", "--tab", "Calls",
                  "--bind", "call-records")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["connectors"]["bindings"] == {
        "call-mailbox": "inbox",
        "call-records": "records",
    }
