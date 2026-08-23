import os
from pathlib import Path

import pytest

from zippergen.private_files import write_private_text


def test_private_writer_replaces_a_permissive_file_atomically(tmp_path):
    target = tmp_path / "token.json"
    target.write_text("old")
    target.chmod(0o644)

    previous = os.umask(0o022)
    try:
        write_private_text(target, "new-secret")
    finally:
        os.umask(previous)

    assert target.read_text() == "new-secret"
    assert target.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".token.json.*.tmp"))


def test_private_writer_refuses_a_symlink(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("keep")
    link = tmp_path / "token.json"
    try:
        link.symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(OSError, match="symlinked private file"):
        write_private_text(link, "replacement")

    assert victim.read_text() == "keep"


def test_a_workspace_secrets_file_is_owner_only_after_a_write(tmp_path):
    """Credentials land here, so the guarantee belongs to the library.

    It used to be checked on the example Google clients instead, which meant
    it held only for code somebody chose to copy. The guarantee belongs here,
    where every credential is actually written, and it must survive a
    permissive umask.
    """

    from zippergen.workspace import Workspace

    workspace = Workspace(tmp_path / "project", home=tmp_path / "home")
    workspace.initialize_project(name="secretive")

    previous = os.umask(0o022)
    try:
        workspace.save_secrets({"OPENAI_API_KEY": "sk-do-not-share"})
    finally:
        os.umask(previous)

    assert workspace.secrets_path.is_file()
    assert workspace.secrets_path.stat().st_mode & 0o077 == 0
    assert "sk-do-not-share" in workspace.secrets_path.read_text()
