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
