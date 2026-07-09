"""SQLite-backed durable event store: transport, replay log, and observation
stream in one append-only table. All writes serialize through one file, so
`rowid` is a global total order consistent with causality.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from collections import deque


class ReplayMismatch(Exception):
    """A step re-executing during replay diverged from the committed log
    (different payload/locator/kind). Raised loudly rather than corrupting state."""


def _lastrowid(cur) -> int:
    rowid = cur.lastrowid
    if rowid is None:
        raise RuntimeError("SQLite did not return a lastrowid for an inserted event.")
    return int(rowid)


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  rowid        INTEGER PRIMARY KEY,
  sender       TEXT NOT NULL,
  receiver     TEXT,
  channel      TEXT,
  kind         TEXT NOT NULL,       -- 'seed'|'msg'|'ctrl'|'act'|'decision'|'trace'|'effect'
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

CREATE TABLE IF NOT EXISTS human_tasks (
  task_id    TEXT PRIMARY KEY,
  role       TEXT NOT NULL,
  locator    BLOB NOT NULL,
  action     TEXT NOT NULL,
  input_hash TEXT,
  inputs     BLOB NOT NULL,
  spec       BLOB NOT NULL,
  status     TEXT NOT NULL,
  result     BLOB,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS human_tasks_by_status
  ON human_tasks(status, updated_at);

CREATE TABLE IF NOT EXISTS human_task_tokens (
  token      TEXT PRIMARY KEY,
  task_id    TEXT NOT NULL,
  channel    TEXT NOT NULL,
  created_at REAL NOT NULL,
  used_at    REAL,
  UNIQUE(task_id, channel)
);
CREATE INDEX IF NOT EXISTS human_task_tokens_by_task
  ON human_task_tokens(task_id);

CREATE TABLE IF NOT EXISTS human_task_notifications (
  task_id     TEXT NOT NULL,
  channel     TEXT NOT NULL,
  target      TEXT NOT NULL,
  external_id TEXT,
  sent_at     REAL NOT NULL,
  PRIMARY KEY(task_id, channel, target)
);
CREATE INDEX IF NOT EXISTS human_task_notifications_by_channel
  ON human_task_notifications(channel, target, sent_at);

CREATE TABLE IF NOT EXISTS adapter_state (
  key        TEXT PRIMARY KEY,
  value      BLOB NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_results (
  workflow   TEXT PRIMARY KEY,
  value      BLOB NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
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


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def write_workflow_result(conn, workflow: str, value: object) -> None:
    payload = json.dumps(_json_safe(value))
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO workflow_results(workflow, value, created_at, updated_at) "
            "VALUES(?,?,?,?) "
            "ON CONFLICT(workflow) DO UPDATE SET "
            "value=excluded.value, updated_at=excluded.updated_at",
            (workflow, payload, now, now),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def load_workflow_result(conn, workflow: str) -> object | None:
    row = conn.execute(
        "SELECT value FROM workflow_results WHERE workflow=?", (workflow,)
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def list_workflow_results(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT workflow, value, created_at, updated_at "
        "FROM workflow_results ORDER BY updated_at, workflow"
    ).fetchall()
    return [
        {
            "workflow": row[0],
            "value": json.loads(row[1]),
            "created_at": row[2],
            "updated_at": row[3],
        }
        for row in rows
    ]


def record_trace_event(conn, role: str, event: dict) -> int:
    cur = conn.execute(
        "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
        "VALUES(?,?,?,?,?,?)",
        (role, None, None, "trace", json.dumps(_json_safe(event)), None),
    )
    return _lastrowid(cur)


def list_trace_events(conn, after_rowid: int = 0) -> list[dict]:
    rows = conn.execute(
        "SELECT rowid, payload FROM events "
        "WHERE kind='trace' AND rowid>? ORDER BY rowid",
        (after_rowid,),
    ).fetchall()
    return [
        {"rowid": row[0], "event": json.loads(row[1])}
        for row in rows
    ]


def human_task_id(
    role: str,
    locator: list,
    input_hash: str | None,
    journal_after: int,
) -> str:
    payload = {
        "role": role,
        "locator": locator,
        "input_hash": input_hash,
        "journal_after": journal_after,
    }
    import hashlib
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def ensure_human_task(
    conn,
    *,
    task_id: str,
    role: str,
    locator: list,
    action: str,
    input_hash: str | None,
    inputs: dict,
    spec: dict,
) -> tuple[dict, bool]:
    """Create a pending human task if absent; return (task, created)."""
    now = time.time()
    cur = conn.execute(
        "INSERT OR IGNORE INTO human_tasks("
        "task_id, role, locator, action, input_hash, inputs, spec, status, result, created_at, updated_at"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            task_id,
            role,
            json.dumps(locator),
            action,
            input_hash,
            json.dumps(inputs),
            json.dumps(spec),
            "pending",
            None,
            now,
            now,
        ),
    )
    task = load_human_task(conn, task_id)
    assert task is not None
    return task, cur.rowcount == 1


def complete_human_task(conn, task_id: str, result: dict) -> dict:
    """Mark a pending task done without overwriting an already-completed answer."""
    now = time.time()
    conn.execute(
        "UPDATE human_tasks SET status='done', result=?, updated_at=? "
        "WHERE task_id=? AND status='pending'",
        (json.dumps(result), now, task_id),
    )
    task = load_human_task(conn, task_id)
    if task is None:
        raise KeyError(f"human task {task_id!r} not found")
    return task


def ensure_human_task_token(conn, task_id: str, *, channel: str = "default") -> dict:
    """Return a durable random token for a human task/channel pair.

    Tokens are intended for external adapters such as email, Telegram, or Slack.
    The raw task id is stable but not meant to be the only approval credential
    outside trusted local CLI use.
    """

    channel = str(channel or "default")
    if load_human_task(conn, task_id) is None:
        raise KeyError(f"human task {task_id!r} not found")

    row = conn.execute(
        "SELECT token, task_id, channel, created_at, used_at "
        "FROM human_task_tokens WHERE task_id=? AND channel=?",
        (task_id, channel),
    ).fetchone()
    if row is not None:
        return _token_row(row)

    now = time.time()
    for _attempt in range(10):
        token = "zg_" + secrets.token_urlsafe(18)
        cur = conn.execute(
            "INSERT OR IGNORE INTO human_task_tokens(token, task_id, channel, created_at, used_at) "
            "VALUES(?,?,?,?,NULL)",
            (token, task_id, channel, now),
        )
        if cur.rowcount == 1:
            return {
                "token": token,
                "task_id": task_id,
                "channel": channel,
                "created_at": now,
                "used_at": None,
            }
        row = conn.execute(
            "SELECT token, task_id, channel, created_at, used_at "
            "FROM human_task_tokens WHERE task_id=? AND channel=?",
            (task_id, channel),
        ).fetchone()
        if row is not None:
            return _token_row(row)
    raise RuntimeError("could not generate unique human task token")


def _token_row(row) -> dict:
    return {
        "token": row[0],
        "task_id": row[1],
        "channel": row[2],
        "created_at": row[3],
        "used_at": row[4],
    }


def load_human_task_token(conn, token: str) -> dict | None:
    row = conn.execute(
        "SELECT token, task_id, channel, created_at, used_at "
        "FROM human_task_tokens WHERE token=?",
        (token,),
    ).fetchone()
    return _token_row(row) if row is not None else None


def mark_human_task_token_used(conn, token: str) -> dict:
    now = time.time()
    conn.execute(
        "UPDATE human_task_tokens SET used_at=COALESCE(used_at, ?) WHERE token=?",
        (now, token),
    )
    record = load_human_task_token(conn, token)
    if record is None:
        raise KeyError(f"human task token {token!r} not found")
    return record


def record_human_task_notification(
    conn,
    task_id: str,
    *,
    channel: str,
    target: str,
    external_id: str | None = None,
) -> dict:
    """Record that an external adapter notified a target about a human task."""

    if load_human_task(conn, task_id) is None:
        raise KeyError(f"human task {task_id!r} not found")
    channel = str(channel or "default")
    target = str(target)
    now = time.time()
    conn.execute(
        "INSERT INTO human_task_notifications(task_id, channel, target, external_id, sent_at) "
        "VALUES(?,?,?,?,?) "
        "ON CONFLICT(task_id, channel, target) DO UPDATE SET "
        "external_id=COALESCE(excluded.external_id, human_task_notifications.external_id), "
        "sent_at=excluded.sent_at",
        (task_id, channel, target, external_id, now),
    )
    record = load_human_task_notification(conn, task_id, channel=channel, target=target)
    assert record is not None
    return record


def _notification_row(row) -> dict:
    return {
        "task_id": row[0],
        "channel": row[1],
        "target": row[2],
        "external_id": row[3],
        "sent_at": row[4],
    }


def load_human_task_notification(
    conn,
    task_id: str,
    *,
    channel: str,
    target: str,
) -> dict | None:
    row = conn.execute(
        "SELECT task_id, channel, target, external_id, sent_at "
        "FROM human_task_notifications WHERE task_id=? AND channel=? AND target=?",
        (task_id, str(channel or "default"), str(target)),
    ).fetchone()
    return _notification_row(row) if row is not None else None


def load_adapter_state(conn, key: str, default=None):
    row = conn.execute("SELECT value FROM adapter_state WHERE key=?", (key,)).fetchone()
    return default if row is None else json.loads(row[0])


def write_adapter_state(conn, key: str, value) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO adapter_state(key, value, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, json.dumps(_json_safe(value)), now),
    )


def load_human_task(conn, task_id: str) -> dict | None:
    row = conn.execute(
        "SELECT task_id, role, locator, action, input_hash, inputs, spec, status, result, "
        "created_at, updated_at FROM human_tasks WHERE task_id=?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    result = json.loads(row[8]) if row[8] is not None else None
    return {
        "task_id": row[0],
        "role": row[1],
        "locator": json.loads(row[2]),
        "action": row[3],
        "input_hash": row[4],
        "inputs": json.loads(row[5]),
        "spec": json.loads(row[6]),
        "status": row[7],
        "result": result,
        "created_at": row[9],
        "updated_at": row[10],
    }


Item = tuple[int, tuple, "dict | None", "dict | None", "dict | None"]


def _encode_causal_stamp(
    vc: dict | None,
    view: dict | None,
    field_view: dict | None,
) -> str | None:
    if vc is None and view is None and field_view is None:
        return None
    return json.dumps({"vc": vc, "view": view, "field_view": field_view})


def _decode_view(view: dict | None) -> dict | None:
    if view is None:
        return None
    return {
        str(lifeline): {int(formula_id): bool(value) for formula_id, value in values.items()}
        for lifeline, values in view.items()
    }


def _decode_causal_stamp(stamp) -> tuple[dict | None, dict | None, dict | None]:
    if stamp is None:
        return None, None, None
    data = json.loads(stamp)
    if (
        isinstance(data, dict)
        and set(data.keys()) <= {"vc", "view", "field_view"}
        and (data.get("vc") is None or isinstance(data.get("vc"), dict))
    ):
        return data.get("vc"), _decode_view(data.get("view")), data.get("field_view")
    # Backward compatibility: old stores kept only the vector clock in this column.
    return data, None, None


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
        self._journal_consumed: int = (since or {}).get("journal", 0)
        self._journal_floor: int = self._journal_consumed
        self._journal_seen: set[int] = set()
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
            "journal": self._journal_consumed,
        }

    def journal_position(self) -> int:
        return self._journal_consumed

    # ---- role-local journal (external act outputs + owner decisions) --------
    def record_act(self, payload: dict) -> int:
        """INSERT an act-journal row. Does NOT advance the journal cursor — the
        result is applied by a separate consume pass (apply-after-commit)."""
        cur = self.conn.execute(
            "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
            "VALUES(?,?,?,?,?,?)",
            (self.role, None, None, "act", json.dumps(payload), None),
        )
        return _lastrowid(cur)

    def record_decision(self, payload: dict) -> int:
        """INSERT a decision-journal row and advance the cursor past it (the
        value is recorded and consumed in one step; no separate consume pass)."""
        cur = self.conn.execute(
            "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
            "VALUES(?,?,?,?,?,?)",
            (self.role, None, None, "decision", json.dumps(payload), None),
        )
        rowid = _lastrowid(cur)
        self._journal_seen.add(rowid)
        self._journal_consumed = max(self._journal_consumed, rowid)
        return rowid

    def consume_journal(self, expected_kind: str, locator: list,
                        input_hash: str | None = None, *,
                        strict: bool = True) -> dict | None:
        """Return a committed journal row for this role.

        ``strict=True`` preserves the simple FIFO replay invariant used by unit
        tests and non-parallel reasoning. Runtime replay uses ``strict=False``
        because a single lifeline can own multiple local parallel branches whose
        enabled order can differ while still referring to the same committed
        action/decision rows.
        """
        if strict:
            row = self.conn.execute(
                "SELECT rowid, kind, payload FROM events "
                "WHERE sender=? AND kind IN ('act','decision') AND rowid>? "
                "ORDER BY rowid LIMIT 1",
                (self.role, self._journal_consumed),
            ).fetchone()
            if row is None:
                return None
            candidates = [row]
        else:
            candidates = self.conn.execute(
                "SELECT rowid, kind, payload FROM events "
                "WHERE sender=? AND kind IN ('act','decision') AND rowid>? "
                "ORDER BY rowid",
                (self.role, self._journal_floor),
            ).fetchall()
        for rowid, kind, payload in candidates:
            if rowid in self._journal_seen:
                continue
            data = json.loads(payload)
            if kind != expected_kind or data.get("locator") != locator:
                if strict:
                    raise ReplayMismatch(
                        f"journal diverged at rowid {rowid}: recorded {kind}/{data.get('locator')}, "
                        f"executing {expected_kind}/{locator}")
                continue
            if input_hash is not None and data.get("input_hash") not in (None, input_hash):
                if strict:
                    raise ReplayMismatch(
                        f"journal input_hash diverged at rowid {rowid}: "
                        f"recorded {data.get('input_hash')!r}, recomputed {input_hash!r}")
                continue
            self._journal_seen.add(rowid)
            self._journal_consumed = max(self._journal_consumed, rowid)
            return data
        return None

    # ---- interpreter-facing surface ---------------------------------------
    def put(self, sender: str, receiver: str, channel: str, values: tuple,
            vc: dict | None = None, view: dict | None = None,
            field_view: dict | None = None) -> int:
        if self._replay_outbox:
            rowid, exp_receiver, exp_channel = self._replay_outbox.popleft()
            if receiver != exp_receiver or channel != exp_channel:
                raise ReplayMismatch(
                    f"send target diverged: replay expected {exp_receiver}/{exp_channel}, "
                    f"got {receiver}/{channel}")
            recorded = self.conn.execute(
                "SELECT payload FROM events WHERE rowid=?", (rowid,)).fetchone()[0]
            if json.loads(recorded) != list(values):
                raise ReplayMismatch(
                    f"send payload diverged at rowid {rowid}: "
                    f"recorded {recorded!r}, recomputed {list(values)!r}")
            return rowid
        cur = self.conn.execute(
            "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
            "VALUES(?,?,?,?,?,?)",
            (sender, receiver, channel, "msg", json.dumps(list(values)),
             _encode_causal_stamp(vc, view, field_view)),
        )
        return _lastrowid(cur)

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
        vc, view, field_view = _decode_causal_stamp(stamp)
        return (rowid, values, vc, view, field_view)

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
