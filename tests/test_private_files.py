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


def test_shipped_google_examples_use_the_private_token_writer():
    root = Path(__file__).resolve().parents[1]
    # Every shipped Google client, so a new one cannot quietly write a token
    # with default permissions. Kept as a list rather than a directory scan:
    # the point is that each shipped client was looked at.
    for name in (
        "call_intake_email_client.py",
        "call_intake_sheets_client.py",
    ):
        source = (root / "examples" / name).read_text()
        assert "write_private_text(TOKEN_PATH" in source
        assert "TOKEN_PATH.write_text" not in source
