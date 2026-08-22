"""Small, auditable primitives for writing local credentials safely."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def ensure_private_directory(path: Path) -> None:
    """Create *path* and ensure only its owner can traverse it."""

    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise OSError(f"Private directory is not a real directory: {path}")
    path.chmod(0o700)


def write_private_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace a regular file with owner-only contents."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(mode):
            raise OSError(f"Refusing to replace a symlinked private file: {path}")
        if not stat.S_ISREG(mode):
            raise OSError(f"Private file is not a regular file: {path}")

    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_private_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file with owner-only contents."""

    write_private_bytes(path, text.encode("utf-8"))


__all__ = [
    "ensure_private_directory",
    "write_private_bytes",
    "write_private_text",
]
