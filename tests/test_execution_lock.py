from __future__ import annotations

import os

import pytest

from zippergen.execution_lock import (
    ExecutionLockError,
    active_execution,
    execution_lock,
)


def test_execution_lock_has_process_lifetime_and_reports_owner(tmp_path):
    path = tmp_path / "private" / "execution.lock"

    assert active_execution(path) is None
    with execution_lock(path, owner="durable run"):
        active = active_execution(path)
        assert active is not None
        assert active.owner == "durable run"
        assert active.pid == os.getpid()
        assert active.started_at is not None
        with pytest.raises(ExecutionLockError) as caught:
            with execution_lock(path, owner="foreground run"):
                pass
        assert caught.value.active == active

    assert active_execution(path) is None


@pytest.mark.parametrize("operation", ["inspect", "acquire"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_lock_refuses_substituted_files_without_modifying_them(tmp_path, operation, kind):
    victim = tmp_path / "unrelated.txt"
    victim.write_text("keep this content")
    victim.chmod(0o640)
    path = tmp_path / "execution.lock"
    if kind == "symlink":
        path.symlink_to(victim)
    elif kind == "hardlink":
        os.link(victim, path)
    else:
        os.mkfifo(path, 0o640)

    with pytest.raises(OSError):
        if operation == "inspect":
            active_execution(path)
        else:
            with execution_lock(path, owner="test"):
                pytest.fail("unsafe lock was acquired")

    assert victim.read_text() == "keep this content"
    assert victim.stat().st_mode & 0o777 == 0o640
    if kind == "fifo":
        assert path.stat().st_mode & 0o777 == 0o640
