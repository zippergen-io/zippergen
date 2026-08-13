"""Connector configuration is an ordinary project and CLI operation.

The split is the same everywhere: portable fields — which chat, which
spreadsheet, which mailbox query — are committed to `zippergen.toml`;
credentials are not.
"""

import subprocess
import sys
import tomllib
import os
from pathlib import Path

from zippergen.serve import main
from zippergen.workspace import Workspace


def _run(
    directory: Path,
    *arguments: str,
    token: str = "bot-token",
    input_text: str | None = None,
):
    environment = dict(os.environ)
    environment["ZIPPERGEN_HOME"] = str(directory.parent / "home")
    return subprocess.run(
        [sys.executable, "-m", "zippergen.serve", *arguments],
        capture_output=True,
        text=True,
        cwd=directory,
        input=input_text if input_text is not None else f"{token}\n",
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


def _provider(root: Path, name: str, kind: str, credential: str | None = None):
    configured = _run(root, "provider", "configure", name, kind, input_text="")
    assert configured.returncode == 0, configured.stderr
    if credential is not None:
        saved = _run(
            root,
            "provider",
            "credential",
            name,
            input_text=credential + "\n",
        )
        assert saved.returncode == 0, saved.stderr


def test_it_saves_a_configuration_the_project_can_commit(tmp_path):
    root = _project(tmp_path)
    _provider(root, "approval-bot", "telegram")

    result = _run(root, "connector", "configure", "approval-chat", "approval-bot", "telegram", "--chat-id", "4242")

    assert result.returncode == 0, result.stderr
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    configuration = manifest["connectors"]["configurations"]["approval-chat"]
    assert configuration == {
        "connection": "approval-bot",
        "kind": "telegram",
        "chat_id": "4242",
    }


def test_telegram_setup_collects_the_chat_id_in_the_human_terminal(
    tmp_path, monkeypatch
):
    root = _project(tmp_path)
    home = root.parent / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.chdir(root)
    Workspace(root, home=home).save_provider_connection(
        "approval-bot", {"kind": "telegram"}
    )
    monkeypatch.setattr("zippergen.serve.sys.stdin.isatty", lambda: True)
    answers = iter(["4242"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    result = main(
        ["connector", "configure", "approval-chat", "approval-bot", "telegram"]
    )

    assert result == 0
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["connectors"]["configurations"]["approval-chat"][
        "chat_id"
    ] == "4242"


def test_the_bot_token_never_reaches_the_manifest(tmp_path):
    """The chat id is portable; the token belongs to one machine."""

    root = _project(tmp_path)
    _provider(root, "approval-bot", "telegram", "secret-bot-token")

    _run(root, "connector", "configure", "approval-chat", "approval-bot", "telegram",
         "--chat-id", "4242")

    assert "secret-bot-token" not in (root / "zippergen.toml").read_text()
    # It is readable back on this machine, which is where it belongs.
    assert Workspace(root, home=root.parent / "home").provider_secret(
        "approval-bot", "bot_token"
    ) == "secret-bot-token"


def test_the_token_is_not_passed_as_an_argument(tmp_path):
    """It is typed, so it stays out of shell history and any transcript."""

    root = _project(tmp_path)
    _provider(root, "approval-bot", "telegram")

    result = _run(root, "provider", "credential", "approval-bot", input_text="tok\n")

    assert "--token" not in result.stderr
    assert "hidden" in result.stdout or "hidden" in result.stderr


def test_one_provider_credential_is_shared_by_two_connector_configurations(tmp_path):
    root = _project(tmp_path)
    _provider(root, "approval-bot", "telegram", "tok")
    first = _run(root, "connector", "configure", "one", "approval-bot", "telegram", "--chat-id", "1")

    second = _run(root, "connector", "configure", "two", "approval-bot", "telegram", "--chat-id", "2", input_text="")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    workspace = Workspace(root, home=root.parent / "home")
    assert workspace.provider_secret("approval-bot", "bot_token") == "tok"


def test_binding_without_a_workflow_says_so(tmp_path):
    root = _project(tmp_path)
    _provider(root, "approval-bot", "telegram")

    configured = _run(
        root,
        "connector",
        "configure",
        "approval-chat",
        "approval-bot",
        "telegram",
        "--chat-id",
        "1",
    )
    assert configured.returncode == 0, configured.stderr
    result = _run(
        root,
        "connector",
        "bind",
        "human-approval",
        "approval-chat",
    )

    assert result.returncode != 0
    assert "none was found" in result.stderr.lower()


def test_a_sheets_configuration_records_the_spreadsheet_and_tab(tmp_path):
    root = _project(tmp_path)
    _provider(root, "google-work", "google")

    result = _run(root, "connector", "configure", "records", "google-work", "google-sheets",
                  "--spreadsheet-id", "1AbC_xyz", "--tab", "Calls")

    assert result.returncode == 0, result.stderr
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["connectors"]["configurations"]["records"] == {
        "connection": "google-work",
        "kind": "google-sheets",
        "spreadsheet_id": "1AbC_xyz",
        "tab": "Calls",
    }


def test_a_sheets_configuration_names_its_required_fields(tmp_path):
    root = _project(tmp_path)
    _provider(root, "google-work", "google")

    result = _run(root, "connector", "configure", "records", "google-work", "google-sheets")

    assert result.returncode != 0
    assert "--spreadsheet-id ID --tab TAB" in result.stderr


def test_a_gmail_configuration_records_the_mailbox_and_query(tmp_path):
    root = _project(tmp_path)
    _provider(root, "google-work", "google")

    result = _run(root, "connector", "configure", "inbox", "google-work", "gmail",
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
    _provider(root, "google-work", "google")

    result = _run(root, "connector", "configure", "records", "google-work", "google-sheets",
                  "--spreadsheet-id", "1", "--tab", "T")

    assert "provider authorize google-work" in result.stdout


def test_binding_works_against_a_real_workflow_requirement(tmp_path):
    """`call_intake` declares call-mailbox and call-records."""

    root = _project(tmp_path)
    example = Path(__file__).resolve().parents[1] / "examples" / "call_intake.py"
    (root / "workflow.py").write_text(example.read_text())
    workspace = Workspace(root, home=root.parent / "home")
    workspace.initialize_project()
    _provider(root, "google-work", "google")

    first = _run(root, "connector", "configure", "inbox", "google-work", "gmail")
    first_binding = _run(
        root, "connector", "bind", "call-mailbox", "inbox"
    )
    second = _run(root, "connector", "configure", "records", "google-work", "google-sheets",
                  "--spreadsheet-id", "1", "--tab", "Calls")
    second_binding = _run(
        root, "connector", "bind", "call-records", "records"
    )

    assert first.returncode == 0, first.stderr
    assert first_binding.returncode == 0, first_binding.stderr
    assert second.returncode == 0, second.stderr
    assert second_binding.returncode == 0, second_binding.stderr
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["connectors"]["bindings"] == {
        "call-mailbox": "inbox",
        "call-records": "records",
    }


def test_binding_rejects_the_wrong_connector_kind(tmp_path):
    root = _project(tmp_path)
    example = Path(__file__).resolve().parents[1] / "examples" / "call_intake.py"
    (root / "workflow.py").write_text(example.read_text())
    Workspace(root, home=root.parent / "home").initialize_project()
    _provider(root, "approval-bot", "telegram")
    configured = _run(
        root,
        "connector",
        "configure",
        "approval-chat",
        "approval-bot",
        "telegram",
        "--chat-id",
        "1",
    )

    result = _run(root, "connector", "bind", "call-mailbox", "approval-chat")

    assert configured.returncode == 0, configured.stderr
    assert result.returncode != 0
    assert "needs gmail" in result.stderr
