"""Process-lifetime ownership for one project's active execution."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActiveExecution:
    """Metadata written by the process that currently holds a project lock."""

    owner: str
    pid: int | None = None
    started_at: str | None = None


class ExecutionLockError(RuntimeError):
    """Raised when another process already owns a project's execution lock."""

    def __init__(self, active: ActiveExecution) -> None:
        self.active = active
        detail = active.owner
        if active.pid is not None:
            detail += f" (PID {active.pid})"
        super().__init__(f"Project already has an active {detail}.")


def execution_lock_path(home: str | Path, project_key: str) -> Path:
    """Return the private lock shared by a project and its deployment."""

    return Path(home).expanduser() / "workspaces" / project_key / "execution.lock"


def _read_active(fd: int) -> ActiveExecution:
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        raw = json.loads(os.read(fd, 4096).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    pid = raw.get("pid")
    return ActiveExecution(
        owner=str(raw.get("owner") or "execution"),
        pid=pid if isinstance(pid, int) else None,
        started_at=(
            str(raw["started_at"]) if raw.get("started_at") else None
        ),
    )


def _try_lock(fd: int) -> bool:
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _unlock(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def active_execution(path: str | Path) -> ActiveExecution | None:
    """Return the current owner, or ``None`` when the project is idle."""

    lock_path = Path(path)
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(fd, 0o600)
        if not _try_lock(fd):
            return _read_active(fd)
        _unlock(fd)
        return None
    finally:
        os.close(fd)


@contextmanager
def execution_lock(
    path: str | Path,
    *,
    owner: str,
) -> Iterator[None]:
    """Hold one project's execution lock until this process leaves the block."""

    lock_path = Path(path)
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        os.fchmod(fd, 0o600)
        locked = _try_lock(fd)
        if not locked:
            raise ExecutionLockError(_read_active(fd))
        record = json.dumps(
            {
                "owner": owner,
                "pid": os.getpid(),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.ftruncate(fd, 0)
        os.write(fd, record)
        os.fsync(fd)
        yield
    finally:
        if locked:
            _unlock(fd)
        os.close(fd)
