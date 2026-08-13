"""SQLite-backed durable state: what the computation *is*, not what it did.

The store holds the current live state of the distributed computation:

* ``role_state``            one row per lifeline: variables, control state, monitor
* ``outstanding_messages``  sends that no receiver has absorbed yet
* ``human_tasks`` (+ tokens, notifications)  asynchronous requests that outlive
  the process
* ``adapter_state``         connector bookkeeping
* ``workflow_results``      the value a finished workflow returned
* ``history``               optional, for inspection only

Recovery is: read ``role_state``, read ``outstanding_messages``, continue. There
is no log to replay, no snapshot to validate, and nothing to compact. Deleting
every row of ``history`` cannot affect whether a deployment resumes.

The crash rule, stated once and relied on everywhere:

    The durable role state describes what is known to have completed. Whatever
    the control state points at has not necessarily completed, and may run
    again after a crash.
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path


SCHEMA_VERSION = 2

HISTORY_RETENTION_KEEP = 10_000
HISTORY_RETENTION_BATCH = 1_000


class StoreSchemaError(Exception):
    """The store on disk does not match the durable model this code implements."""


class RoleStateConflict(Exception):
    """Another runner advanced a role from the state this runner loaded."""


def _lastrowid(cur) -> int:
    rowid = cur.lastrowid
    if rowid is None:
        raise RuntimeError("SQLite did not return a lastrowid for an inserted row.")
    return int(rowid)


SCHEMA = """
CREATE TABLE IF NOT EXISTS store_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS role_state (
  role       TEXT PRIMARY KEY,
  env        TEXT NOT NULL,       -- json object of variable values
  control    TEXT NOT NULL,       -- json control state (see control.py)
  monitor    TEXT,                -- json CPL monitor state incl. vector clock
  steps      INTEGER NOT NULL,    -- committed steps; distinguishes loop visits
  status     TEXT NOT NULL,       -- running|blocked|waiting_receive|waiting_human|...
  detail     TEXT NOT NULL,       -- json, non-sensitive status metadata
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS outstanding_messages (
  id           INTEGER PRIMARY KEY,   -- send order; the only ordering fact kept
  sender       TEXT NOT NULL,
  receiver     TEXT NOT NULL,
  channel      TEXT NOT NULL,
  payload      TEXT NOT NULL,
  causal_stamp TEXT
);
CREATE INDEX IF NOT EXISTS outstanding_by_route
  ON outstanding_messages(receiver, sender, channel, id);

CREATE TABLE IF NOT EXISTS human_tasks (
  task_id    TEXT PRIMARY KEY,
  role       TEXT NOT NULL,
  locator    TEXT NOT NULL,
  action     TEXT NOT NULL,
  input_hash TEXT,
  inputs     TEXT NOT NULL,
  spec       TEXT NOT NULL,
  status     TEXT NOT NULL,
  result     TEXT,
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
  value      TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_results (
  workflow   TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

-- Optional. Never read during recovery. Prunable at will.
CREATE TABLE IF NOT EXISTS history (
  id      INTEGER PRIMARY KEY,
  role    TEXT NOT NULL,
  payload TEXT NOT NULL
);
"""


def open_store(path: str) -> sqlite3.Connection:
    connection_path = path
    if path != ":memory:" and not path.startswith("file:"):
        store_path = Path(path).expanduser()
        # The store holds workflow variables and human approval tokens. Create it
        # owner-private before SQLite first opens it rather than relying on umask.
        fd = os.open(store_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(fd)
        store_path.chmod(0o600)
        connection_path = str(store_path)

    # isolation_level=None -> autocommit; transactions are driven explicitly.
    # check_same_thread=False: a role's connection is created by the supervisor and
    # driven from that role's thread; only one thread uses it at a time.
    conn = sqlite3.connect(
        connection_path,
        isolation_level=None,
        check_same_thread=False,
        timeout=5.0,
    )
    conn.execute("PRAGMA busy_timeout=5000")

    # Switching to WAL takes a lock upgrade that SQLite deliberately does not run
    # the busy handler for, so two processes opening a fresh file together can see
    # "database is locked" here regardless of busy_timeout. WAL is a persistent
    # file property, so retrying is always safe.
    for attempt in range(50):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc) or attempt == 49:
                raise
            time.sleep(0.05)

    # State the durability contract rather than inheriting a compile-time default.
    conn.execute("PRAGMA synchronous=FULL")

    _reject_replay_era_store(conn)
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO store_meta(key,value) VALUES('schema_version',?) "
        "ON CONFLICT(key) DO NOTHING",
        (str(SCHEMA_VERSION),),
    )
    if path != ":memory:" and not path.startswith("file:"):
        for family in (
            Path(connection_path),
            Path(f"{connection_path}-wal"),
            Path(f"{connection_path}-shm"),
        ):
            if family.exists():
                family.chmod(0o600)
    return conn


def _reject_replay_era_store(conn: sqlite3.Connection) -> None:
    """Refuse a store written by the replay/snapshot design.

    Its recovery state lives in an event log and per-role snapshots that no
    longer exist. There is nothing to migrate to, because the old store kept
    positions into a log rather than the interpreter state itself.
    """

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "('events','snapshots','cursors','recovery_high_water') LIMIT 1"
    ).fetchone()
    if row is not None:
        raise StoreSchemaError(
            "This durable store was written by an older ZipperGen that recovered "
            "by replaying an event log. Its state cannot be carried over. Reset "
            "the deployment with 'zg deploy reset', or delete the run store and "
            "start again."
        )


# ---------------------------------------------------------------------------
# Store identity
# ---------------------------------------------------------------------------


def read_meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM store_meta WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def write_meta(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO store_meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


class WorkflowIdentityError(Exception):
    """Durable state belongs to a different program than the one being resumed."""


def claim_workflow_identity(conn, workflow: str, fingerprint: str) -> None:
    """Bind a store to one workflow and one projected program, or refuse it.

    Control state is child-index paths into the projected programs, so resuming
    under changed code would silently mean something else. This is checked once,
    explicitly, at startup rather than being inferred from a divergence later.
    """

    conn.execute("BEGIN IMMEDIATE")
    try:
        stored_workflow = read_meta(conn, "workflow")
        stored_fingerprint = read_meta(conn, "workflow_fingerprint")
        if stored_workflow is None and stored_fingerprint is None:
            write_meta(conn, "workflow", workflow)
            write_meta(conn, "workflow_fingerprint", fingerprint)
            conn.execute("COMMIT")
            return
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    if stored_workflow != workflow:
        raise WorkflowIdentityError(
            f"This store holds durable state for workflow {stored_workflow!r}, "
            f"not {workflow!r}. Reset the deployment to start a new one."
        )
    if stored_fingerprint != fingerprint:
        raise WorkflowIdentityError(
            "The workflow changed since this durable state was written, so the "
            "stored control positions no longer mean the same thing. Reset the "
            "deployment with 'zg deploy reset' to start fresh, or restore the "
            "previous version of the workflow to resume it."
        )


# ---------------------------------------------------------------------------
# Role state: the whole of a lifeline's recoverable position
# ---------------------------------------------------------------------------


def load_role_state(conn, role: str) -> dict | None:
    row = conn.execute(
        "SELECT env, control, monitor, steps, status, detail FROM role_state WHERE role=?",
        (role,),
    ).fetchone()
    if row is None:
        return None
    return {
        "env": json.loads(row[0]),
        "control": json.loads(row[1]),
        "monitor": json.loads(row[2]) if row[2] is not None else None,
        "steps": int(row[3]),
        "status": row[4],
        "detail": json.loads(row[5]),
    }


def write_role_state(
    conn,
    role: str,
    *,
    env: dict,
    control: dict,
    monitor: dict | None,
    steps: int,
    status: str,
    detail: dict | None = None,
    expected_steps: int | None = None,
) -> None:
    """Write a role's whole durable position. Caller owns the transaction.

    With no ``expected_steps`` this creates the role's initial row and refuses
    to overwrite an existing one. Later writes are compare-and-swap: a runner
    keeps its residual in memory between commits, so the committed step count
    must still equal the value it loaded. If another runner advanced the same
    role, fail loudly; the caller rolls back every message change made by the
    stale step in the same transaction.
    """

    values = (
        json.dumps(env),
        json.dumps(control),
        None if monitor is None else json.dumps(monitor),
        int(steps),
        status,
        json.dumps(_json_safe(detail or {})),
        time.time(),
    )
    if expected_steps is None:
        conn.execute(
            "INSERT INTO role_state(role,env,control,monitor,steps,status,detail,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (role, *values),
        )
        return

    cursor = conn.execute(
        "UPDATE role_state SET env=?, control=?, monitor=?, steps=?, status=?, "
        "detail=?, updated_at=? WHERE role=? AND steps=?",
        (*values, role, int(expected_steps)),
    )
    if cursor.rowcount != 1:
        raise RoleStateConflict(
            f"Role {role!r} was advanced by another runner. Only one durable "
            "supervisor may execute a store at a time."
        )


def set_role_status(
    conn,
    role: str,
    status: str,
    detail: dict | None = None,
    *,
    expected_steps: int | None = None,
) -> None:
    """Update only the diagnostic status. Short standalone transaction."""

    conn.execute("BEGIN IMMEDIATE")
    try:
        sql = "UPDATE role_state SET status=?, detail=?, updated_at=? WHERE role=?"
        params: tuple = (
            status,
            json.dumps(_json_safe(detail or {})),
            time.time(),
            role,
        )
        if expected_steps is not None:
            sql += " AND steps=?"
            params += (int(expected_steps),)
        cursor = conn.execute(sql, params)
        if expected_steps is not None and cursor.rowcount != 1:
            raise RoleStateConflict(
                f"Role {role!r} was advanced by another runner. Only one durable "
                "supervisor may execute a store at a time."
            )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def list_role_states(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT role,control,steps,status,detail,updated_at FROM role_state ORDER BY role"
    ).fetchall()
    return [
        {
            "role": row[0],
            "control": json.loads(row[1]),
            "steps": int(row[2]),
            "status": row[3],
            "detail": json.loads(row[4]),
            "updated_at": row[5],
        }
        for row in rows
    ]


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


# ---------------------------------------------------------------------------
# Workflow results
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# History: inspection only, never consulted by recovery
# ---------------------------------------------------------------------------


def prune_history(conn, *, keep: int = HISTORY_RETENTION_KEEP) -> int:
    if keep < 0:
        raise ValueError("keep must be zero or greater")
    if keep == 0:
        cursor = conn.execute("DELETE FROM history")
        return int(cursor.rowcount) if cursor.rowcount >= 0 else 0
    cutoff = conn.execute(
        "SELECT id FROM history ORDER BY id DESC LIMIT 1 OFFSET ?", (keep - 1,)
    ).fetchone()
    if cutoff is None:
        return 0
    cursor = conn.execute("DELETE FROM history WHERE id<?", (int(cutoff[0]),))
    return int(cursor.rowcount) if cursor.rowcount >= 0 else 0


def record_history(conn, role: str, event: dict) -> int:
    cur = conn.execute(
        "INSERT INTO history(role,payload) VALUES(?,?)",
        (role, json.dumps(_json_safe(event))),
    )
    rowid = _lastrowid(cur)
    if rowid % HISTORY_RETENTION_BATCH == 0:
        prune_history(conn, keep=HISTORY_RETENTION_KEEP)
    return rowid


def list_history(conn, after_id: int = 0) -> list[dict]:
    rows = conn.execute(
        "SELECT id, payload FROM history WHERE id>? ORDER BY id", (after_id,)
    ).fetchall()
    return [{"rowid": row[0], "event": json.loads(row[1])} for row in rows]


# ---------------------------------------------------------------------------
# Human tasks: genuinely asynchronous, and outlive the interpreter process
# ---------------------------------------------------------------------------


def human_task_id(role: str, locator: list, input_hash: str | None, nonce: object) -> str:
    payload = {
        "role": role,
        "locator": locator,
        "input_hash": input_hash,
        "nonce": nonce,
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

    Tokens are for external adapters such as email, Telegram, or Slack. The raw
    task id is stable but is not meant to be the only approval credential
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


def load_human_task_notification_by_external(
    conn,
    *,
    channel: str,
    target: str,
    external_id: str,
) -> dict | None:
    """Resolve a provider reply to the durable task it answers."""

    row = conn.execute(
        "SELECT task_id, channel, target, external_id, sent_at "
        "FROM human_task_notifications "
        "WHERE channel=? AND target=? AND external_id=?",
        (str(channel), str(target), str(external_id)),
    ).fetchone()
    return _notification_row(row) if row is not None else None


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
    return {
        "task_id": row[0],
        "role": row[1],
        "locator": json.loads(row[2]),
        "action": row[3],
        "input_hash": row[4],
        "inputs": json.loads(row[5]),
        "spec": json.loads(row[6]),
        "status": row[7],
        "result": json.loads(row[8]) if row[8] is not None else None,
        "created_at": row[9],
        "updated_at": row[10],
    }


# ---------------------------------------------------------------------------
# Outstanding messages
# ---------------------------------------------------------------------------


Item = tuple[int, tuple, "dict | None", "dict | None", "dict | None"]


def _encode_causal_stamp(vc, view, field_view) -> str | None:
    if vc is None and view is None and field_view is None:
        return None
    return json.dumps({"vc": vc, "view": view, "field_view": field_view})


def _decode_view(view: dict | None) -> dict | None:
    if view is None:
        return None
    return {
        str(lifeline): {int(formula): bool(value) for formula, value in values.items()}
        for lifeline, values in view.items()
    }


def _decode_causal_stamp(stamp) -> tuple[dict | None, dict | None, dict | None]:
    if stamp is None:
        return None, None, None
    data = json.loads(stamp)
    return data.get("vc"), _decode_view(data.get("view")), data.get("field_view")


class DurableChannel:
    """Communication that has not yet been absorbed by a receiver.

    A send inserts a row. A receive takes the lowest-id row on its route and
    hands it to the interpreter. The row is not deleted immediately: it is held
    as *taken* until the role commits, so that consuming a message and advancing
    the receiver's state are one transaction. If that transaction rolls back,
    the message is still outstanding and nothing was lost.

    ``id`` is the only ordering fact kept. It gives FIFO per route and a
    deterministic winner for a coregion receive across routes.
    """

    def __init__(self, conn: sqlite3.Connection, role: str) -> None:
        self.conn = conn
        self.role = role
        self._taken: list[int] = []

    # ---- interpreter-facing surface ---------------------------------------
    def put(self, sender: str, receiver: str, channel: str, values: tuple,
            vc: dict | None = None, view: dict | None = None,
            field_view: dict | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO outstanding_messages(sender,receiver,channel,payload,causal_stamp) "
            "VALUES(?,?,?,?,?)",
            (
                sender,
                receiver,
                channel,
                json.dumps(list(values)),
                _encode_causal_stamp(vc, view, field_view),
            ),
        )
        return _lastrowid(cur)

    def try_get(self, sender: str, receiver: str, channel: str) -> Item | None:
        clause, params = self._not_taken()
        row = self.conn.execute(
            "SELECT id, payload, causal_stamp FROM outstanding_messages "
            f"WHERE sender=? AND receiver=? AND channel=?{clause} ORDER BY id LIMIT 1",
            (sender, receiver, channel, *params),
        ).fetchone()
        if row is None:
            return None
        self._taken.append(int(row[0]))
        return self._row_to_item(*row)

    def try_get_any(
        self,
        receiver: str,
        senders: set[str],
        channel: str,
    ) -> tuple[str, Item] | None:
        """Take the earliest available message across candidate senders.

        Each route is FIFO on its own. Across routes the send order (``id``) is
        the deterministic tie-break, so a restart makes the same choice.
        """

        if not senders:
            return None
        names = sorted(senders)
        senders_in = ",".join("?" for _ in names)
        clause, params = self._not_taken()
        row = self.conn.execute(
            "SELECT id, sender, payload, causal_stamp FROM outstanding_messages "
            f"WHERE receiver=? AND channel=? AND sender IN ({senders_in}){clause} "
            "ORDER BY id LIMIT 1",
            (receiver, channel, *names, *params),
        ).fetchone()
        if row is None:
            return None
        self._taken.append(int(row[0]))
        return str(row[1]), self._row_to_item(row[0], row[2], row[3])

    def get(self, sender: str, receiver: str, channel: str, *,
            stop: threading.Event | None = None) -> Item:
        while True:
            item = self.try_get(sender, receiver, channel)
            if item is not None:
                return item
            if stop is not None and stop.is_set():
                raise RuntimeError("Workflow cancelled")
            time.sleep(0.02)

    def _not_taken(self) -> tuple[str, tuple]:
        """Exclude rows already taken in this transaction.

        The clause is omitted entirely when nothing is taken: `id NOT IN (NULL)`
        is unknown rather than true in SQL, so an empty list must not be spelled
        as a NULL placeholder.
        """

        if not self._taken:
            return "", ()
        placeholders = ",".join("?" for _ in self._taken)
        return f" AND id NOT IN ({placeholders})", tuple(self._taken)

    @staticmethod
    def _row_to_item(rowid, payload, stamp) -> Item:
        values = tuple(json.loads(payload)) if payload is not None else ()
        vc, view, field_view = _decode_causal_stamp(stamp)
        return (int(rowid), values, vc, view, field_view)

    # ---- transaction lifecycle (driven by the role loop) -------------------
    def delete_taken(self) -> None:
        """Remove consumed messages. Caller is inside the committing transaction."""

        if not self._taken:
            return
        placeholders = ",".join("?" for _ in self._taken)
        self.conn.execute(
            f"DELETE FROM outstanding_messages WHERE id IN ({placeholders})",
            tuple(self._taken),
        )

    def clear_taken(self) -> None:
        self._taken.clear()


def list_outstanding_messages(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, sender, receiver, channel, payload FROM outstanding_messages "
        "ORDER BY id"
    ).fetchall()
    return [
        {
            "id": row[0],
            "sender": row[1],
            "receiver": row[2],
            "channel": row[3],
            "payload": json.loads(row[4]),
        }
        for row in rows
    ]
