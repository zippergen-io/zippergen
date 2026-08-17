"""One reader per Telegram bot, and a durable inbox everyone else reads.

Telegram's ``getUpdates`` is a single queue per bot with one server-side
cursor, and reading it is destructive: an update is confirmed, and forgotten,
as soon as a later call supplies a higher offset. Several ZipperGen
deployments may legitimately share one bot, and they are separate processes
with separate stores, so letting each of them poll would mean several
independent readers of a single-consumer queue. Whichever polled first would
confirm the others' updates out of existence.

So the bot's cursor is shared state, because the bot is shared:

    poll:     take the lock, fetch from the offset, write the updates and the
              new offset in one transaction, release the lock
    consume:  every process reads the inbox and takes the updates that belong
              to its own durable tasks

The lock is a plain advisory file lock, held only across one fetch. The kernel
releases it if the holder dies, which is why there is no lease, no heartbeat
and no stale-owner handling here. A process that cannot take the lock simply
does not fetch: someone else is already fetching for it, and it still reads
the inbox, so it remains independently useful.

The ownership rule, stated once:

    Telegram owns an update until it is durably in this inbox. ZipperGen owns
    it from then until the deployment it belongs to has absorbed it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inbox (
  update_id   INTEGER PRIMARY KEY,   -- Telegram's own id, so a retry is a no-op
  received_at REAL NOT NULL,
  payload     TEXT NOT NULL
);
"""


def bot_fingerprint(token: str) -> str:
    """Name a bot without storing its token.

    The token is the identity that matters -- two provider connections holding
    the same token are the same bot -- but it is a credential, so the shared
    file is named after a hash of it rather than the token itself.
    """

    return hashlib.sha256(token.encode()).hexdigest()[:16]


def _connectors_directory() -> Path:
    from zippergen.deployment_platform import zippergen_home

    directory = zippergen_home() / "connectors"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    return directory


def inbox_path(fingerprint: str) -> Path:
    return _connectors_directory() / f"telegram-{fingerprint}.sqlite"


def lock_path(fingerprint: str) -> Path:
    return _connectors_directory() / f"telegram-{fingerprint}.lock"


@contextmanager
def poll_lock(fingerprint: str):
    """Yield True to whoever may fetch from Telegram right now.

    Non-blocking on purpose. Fetching is work done on everyone's behalf, not
    something a process needs for itself, so a process that loses the race
    carries on and reads the inbox instead of queueing behind a long poll.

    The lock file is never deleted. An advisory lock lives on the inode, not
    the path, so unlinking it while another process holds it would let the next
    process lock a fresh inode and believe it had exclusive access.
    """

    path = lock_path(fingerprint)
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def open_inbox(fingerprint: str) -> sqlite3.Connection:
    path = inbox_path(fingerprint)
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.close(handle)
    path.chmod(0o600)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=5.0)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.executescript(SCHEMA)
    return conn


def read_offset(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key='offset'").fetchone()
    return 0 if row is None else int(row[0])


def record_updates(
    conn: sqlite3.Connection,
    updates: list[dict],
    *,
    offset: int,
) -> int:
    """Take ownership of a batch, in one transaction.

    The offset only moves in the same commit that stores the updates, and only
    the *next* fetch confirms them to Telegram. So a crash before this commit
    leaves Telegram holding them, and a crash after it leaves them here. There
    is no window in which both sides believe the other has them.
    """

    if not updates:
        return 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        stored = 0
        now = time.time()
        for update in updates:
            cursor = conn.execute(
                "INSERT INTO inbox(update_id, received_at, payload) "
                "VALUES(?,?,?) ON CONFLICT(update_id) DO NOTHING",
                (int(update["update_id"]), now, json.dumps(update)),
            )
            stored += 1 if cursor.rowcount else 0
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('offset',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(offset),),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return stored


def list_updates(conn: sqlite3.Connection) -> list[tuple[int, dict]]:
    return [
        (int(row[0]), json.loads(row[1]))
        for row in conn.execute(
            "SELECT update_id, payload FROM inbox ORDER BY update_id"
        ).fetchall()
    ]


def remove_update(conn: sqlite3.Connection, update_id: int) -> None:
    conn.execute("DELETE FROM inbox WHERE update_id=?", (int(update_id),))


def count_stale_updates(conn: sqlite3.Connection, *, older_than_days: float) -> int:
    cutoff = time.time() - older_than_days * 86400.0
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM inbox WHERE received_at < ?", (cutoff,)
        ).fetchone()[0]
    )


def prune_updates(conn: sqlite3.Connection, *, older_than_days: float) -> int:
    """Drop updates nobody claimed.

    An update stays until the deployment it belongs to absorbs it, because that
    deployment may simply be stopped and come back later. What is left after
    that is addressed to a task that no longer exists -- a reset store, a
    removed project -- and only age can tell us so.
    """

    cutoff = time.time() - older_than_days * 86400.0
    cursor = conn.execute("DELETE FROM inbox WHERE received_at < ?", (cutoff,))
    return int(cursor.rowcount) if cursor.rowcount and cursor.rowcount > 0 else 0
