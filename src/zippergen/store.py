"""SQLite-backed durable event store: transport, replay log, and observation
stream in one append-only table. All writes serialize through one file, so
`rowid` is a global total order consistent with causality.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  rowid        INTEGER PRIMARY KEY,
  sender       TEXT NOT NULL,
  receiver     TEXT,
  channel      TEXT,
  kind         TEXT NOT NULL,       -- 'seed'|'msg'|'ctrl'|'act'|'decision'|'effect'
  payload      BLOB,
  causal_stamp BLOB
);
CREATE INDEX IF NOT EXISTS events_by_channel
  ON events(receiver, sender, channel, rowid);

CREATE TABLE IF NOT EXISTS cursors (
  role     TEXT NOT NULL,
  chan_key TEXT NOT NULL,           -- "sender|receiver|channel"
  consumed INTEGER NOT NULL,        -- highest rowid consumed on this key
  PRIMARY KEY (role, chan_key)
);

CREATE TABLE IF NOT EXISTS snapshots (
  role    TEXT PRIMARY KEY,
  env     BLOB NOT NULL,            -- json-encoded local env (scalars)
  locator BLOB NOT NULL,            -- json-encoded child-index path to the loop node
  floor   BLOB NOT NULL            -- json-encoded per-channel replay floor
);
"""


def open_store(path: str) -> sqlite3.Connection:
    # isolation_level=None -> autocommit; we drive BEGIN/COMMIT explicitly.
    # check_same_thread=False: a role's connection is created by the supervisor
    # and driven from the role's loop (a different thread in tests / deployment);
    # each connection is still used by only one thread at a time.
    #
    # timeout= at connect time + busy_timeout pragma, both set BEFORE any
    # lock-taking pragma/statement, so SQLite's busy handler covers ordinary
    # lock contention (e.g. the SCHEMA executescript below racing a peer's
    # first-open transaction).
    conn = sqlite3.connect(
        path, isolation_level=None, check_same_thread=False, timeout=5.0
    )
    conn.execute("PRAGMA busy_timeout=5000")  # wait, don't fail, on concurrent writers

    # Switching journal_mode to WAL requires SQLite to upgrade its lock, and
    # SQLite deliberately does NOT invoke the busy-timeout handler for that
    # upgrade (doing so risks deadlock between two connections both trying to
    # upgrade). So when two connections open the same brand-new database file
    # at the same time, the loser can get `database is locked` from this
    # PRAGMA immediately, regardless of busy_timeout. WAL is a persistent,
    # file-level property, so it's always safe to retry (or discover it's
    # already WAL because the peer won the race) — wrap it in a short bounded
    # retry loop instead of letting the race surface as a hard failure.
    for attempt in range(50):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as e:
            if "database is locked" not in str(e) or attempt == 49:
                raise
            time.sleep(0.05)

    conn.executescript(SCHEMA)
    return conn


def chan_key(sender: str, receiver: str, channel: str) -> str:
    return f"{sender}|{receiver}|{channel}"


def write_snapshot(conn, role: str, env: dict, locator: list, floor: dict) -> None:
    # Serialize BEFORE opening the transaction so a non-serializable env raises
    # here (caller skips the snapshot) without leaving a dangling transaction.
    payload = (role, json.dumps(env), json.dumps(locator), json.dumps(floor))
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO snapshots(role, env, locator, floor) VALUES(?,?,?,?) "
            "ON CONFLICT(role) DO UPDATE SET env=excluded.env, "
            "locator=excluded.locator, floor=excluded.floor",
            payload,
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def load_snapshot(conn, role: str) -> dict | None:
    row = conn.execute(
        "SELECT env, locator, floor FROM snapshots WHERE role=?", (role,)
    ).fetchone()
    if row is None:
        return None
    return {"env": json.loads(row[0]), "locator": json.loads(row[1]),
            "floor": json.loads(row[2])}


Item = tuple[int, tuple, "dict | None", "dict | None", "dict | None"]


class DurableChannel:
    """Channel backed by the shared event store, with replay/live semantics.

    Same put/try_get/get surface as InProcessChannel. Consumption is tentative
    until commit_txn(): only then does the durable consume-cursor advance, in the
    same transaction as any emitted sends. On restart the constructor rebuilds
    per-key replay queues from the committed log so re-execution reserves recorded
    sends (no re-INSERT) and re-serves recorded recvs (no live read).
    """

    def __init__(self, conn: sqlite3.Connection, role: str, since: dict | None = None) -> None:
        self.conn = conn
        self.role = role
        self.since = since   # None => full replay; else {"out": int, "cursors": {chan_key: int}}
        self._consumed: dict[tuple[str, str, str], int] = {}
        self._tentative: dict[tuple[str, str, str], int] = {}
        self._replay_outbox: deque = deque()
        self._replay_inbox: dict[tuple[str, str, str], deque] = {}
        self._load_cursors()
        self._load_replay()

    # ---- startup reconstruction -------------------------------------------
    def _load_cursors(self) -> None:
        for ck, consumed in self.conn.execute(
            "SELECT chan_key, consumed FROM cursors WHERE role=?", (self.role,)
        ).fetchall():
            sender, receiver, channel = ck.split("|")
            self._consumed[(sender, receiver, channel)] = consumed

    def _load_replay(self) -> None:
        # With a snapshot floor (self.since), replay only the tail: own sends after
        # floor["out"], and inbound consumed after each channel's floor cursor.
        # With since=None, out_floor=0 and per-channel lo=0 -> full history (as before).
        out_floor = self.since["out"] if self.since else 0
        for rowid, receiver, channel in self.conn.execute(
            "SELECT rowid, receiver, channel FROM events "
            "WHERE sender=? AND kind IN ('msg','ctrl') AND rowid>? ORDER BY rowid",
            (self.role, out_floor),
        ).fetchall():
            self._replay_outbox.append((rowid, receiver, channel))
        cursor_floors = self.since["cursors"] if self.since else {}
        for (sender, receiver, channel), consumed in self._consumed.items():
            lo = cursor_floors.get(chan_key(sender, receiver, channel), 0)
            rows = self.conn.execute(
                "SELECT rowid, payload, causal_stamp FROM events "
                "WHERE sender=? AND receiver=? AND channel=? AND rowid>? AND rowid<=? "
                "ORDER BY rowid",
                (sender, receiver, channel, lo, consumed),
            ).fetchall()
            if rows:
                self._replay_inbox[(sender, receiver, channel)] = deque(rows)

    def replaying(self) -> bool:
        return bool(self._replay_outbox) or any(self._replay_inbox.values())

    def position(self) -> dict:
        """The committed replay floor: own-send high-water + per-channel cursors."""
        row = self.conn.execute(
            "SELECT MAX(rowid) FROM events WHERE sender=? AND kind IN ('msg','ctrl')",
            (self.role,),
        ).fetchone()
        return {
            "out": row[0] or 0,
            "cursors": {chan_key(*key): rowid for key, rowid in self._consumed.items()},
        }

    # ---- interpreter-facing surface ---------------------------------------
    def put(self, sender: str, receiver: str, channel: str, values: tuple,
            vc: dict | None = None, view: dict | None = None,
            field_view: dict | None = None) -> int:
        if self._replay_outbox:
            rowid, _r, _c = self._replay_outbox.popleft()
            return rowid
        cur = self.conn.execute(
            "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
            "VALUES(?,?,?,?,?,?)",
            (sender, receiver, channel, "msg", json.dumps(list(values)),
             json.dumps(vc) if vc is not None else None),
        )
        return int(cur.lastrowid)

    def try_get(self, sender: str, receiver: str, channel: str) -> Item | None:
        key = (sender, receiver, channel)
        dq = self._replay_inbox.get(key)
        if dq:
            rowid, payload, stamp = dq.popleft()
            return self._row_to_item(rowid, payload, stamp)
        floor = self._tentative.get(key, self._consumed.get(key, 0))
        row = self.conn.execute(
            "SELECT rowid, payload, causal_stamp FROM events "
            "WHERE sender=? AND receiver=? AND channel=? AND rowid>? ORDER BY rowid LIMIT 1",
            (sender, receiver, channel, floor),
        ).fetchone()
        if row is None:
            return None
        rowid, payload, stamp = row
        self._tentative[key] = rowid
        return self._row_to_item(rowid, payload, stamp)

    def get(self, sender: str, receiver: str, channel: str, *,
            stop: threading.Event | None = None) -> Item:
        while True:
            item = self.try_get(sender, receiver, channel)
            if item is not None:
                return item
            if stop is not None and stop.is_set():
                raise RuntimeError("Workflow cancelled")
            time.sleep(0.02)

    @staticmethod
    def _row_to_item(rowid: int, payload, stamp) -> Item:
        values = tuple(json.loads(payload)) if payload is not None else ()
        vc = json.loads(stamp) if stamp is not None else None
        return (rowid, values, vc, None, None)

    # ---- transaction lifecycle (driven by the per-role loop) --------------
    def commit_txn(self) -> None:
        for key, rowid in self._tentative.items():
            self.conn.execute(
                "INSERT INTO cursors(role, chan_key, consumed) VALUES(?,?,?) "
                "ON CONFLICT(role, chan_key) DO UPDATE SET consumed=excluded.consumed",
                (self.role, chan_key(*key), rowid),
            )
        self.conn.execute("COMMIT")
        self._consumed.update(self._tentative)
        self._tentative.clear()

    def rollback_txn(self) -> None:
        self.conn.execute("ROLLBACK")
        self._tentative.clear()
