"""Configuring a Telegram connector must not need the deleted Studio shell.

This was the one capability the Studio removal genuinely cost: setting up and
binding a human-approval connector lived in `studio_connectors.py` and nothing
in the CLI replaced it. The tutorial workflow needs it, so it is back as an
ordinary command over the same workspace methods.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

from zippergen.workspace import Workspace


def _run(directory: Path, *arguments: str, token: str = "bot-token"):
    return subprocess.run(
        [sys.executable, "-m", "zippergen.serve", *arguments],
        capture_output=True,
        text=True,
        cwd=directory,
        input=f"{token}\n",
        check=False,
    )


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(
        [sys.executable, "-m", "zippergen.serve", "init"],
        cwd=root,
        capture_output=True,
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
    assert Workspace(root).connector_provider_secret(
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
    assert "no workflow yet" in result.stderr
