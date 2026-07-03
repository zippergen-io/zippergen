# Deployable Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a checked workflow as supervised, crash-survivable per-role processes that coordinate through one shared SQLite log and resume from the last committed step after a crash.

**Architecture:** Split today's fused `run()` into (1) a `Channel` abstraction with an in-process implementation (unchanged behavior) and a new SQLite-backed durable implementation, (2) a per-role event-sourced loop with replay/live modes, and (3) a `zippergen serve --role R` entry point. Projection, the local-statement interpreter (`_step`), and the CPL monitor are reused unchanged. The store's single append-only `events` table is transport, replay log, and observation stream at once.

**Tech Stack:** Python 3.11+ stdlib only — `sqlite3`, `argparse`, `json`, `collections`, `threading`. Tests: `pytest`. Supervision: systemd unit template / container (documented, not code).

## Global Constraints

- Python **3.11+**, stdlib only. **No new third-party dependencies** (verbatim project rule).
- **Zero behavior change to the in-process path.** Every existing example/test that runs through `run()` must behave exactly as today, at every commit. The durable runtime is strictly additive.
- The in-process default remains `run()` with the thread-per-lifeline model.
- FIFO must be preserved per `(sender, receiver, channel)` key.
- Design corresponds to spec `docs/superpowers/specs/2026-07-02-deployable-runtime-design.md`.
- Follow existing repo conventions: frozen-dataclass IR is never mutated; `match`/`isinstance` chains cover all union members.
- **Agents MUST NOT run `git commit` or `git push`, ever, under any circumstances.** Leave every change in the working tree for the human to review and commit. Do not add any AI author or `Co-Authored-By` attribution. This is a hard, non-negotiable constraint — a checkpoint step that says "do NOT commit" means exactly that.

---

## File Structure

- `src/zippergen/channels.py` — **new.** `_SeqQueue` (moved from `runtime.py`), the `Channel` interface (duck-typed), and `InProcessChannel`.
- `src/zippergen/store.py` — **new.** SQLite schema, `open_store()`, JSON row helpers, and `DurableChannel` (send=INSERT, recv=cursor read, replay queues).
- `src/zippergen/serve.py` — **new.** `run_role()` (replay→live loop with per-step transactions), seed handling, and the `main()` CLI entry point.
- `src/zippergen/runtime.py` — **modify.** Remove `_SeqQueue`/`Channels`; route all channel access through the `Channel` interface; build `InProcessChannel()` in `run()`.
- `deploy/zippergen@.service` — **new.** systemd unit template.
- `tests/test_channels.py`, `tests/test_store.py`, `tests/test_serve_replay.py` — **new.** Unit + kill-and-resume tests.
- `tests/test_examples_regression.py` — **new.** Characterization gate over existing examples.

Deferred to a follow-on plan (spec §8): `snapshots` table + residual serialization for non-terminating workflows.

---

## Task 0: Regression characterization gate

Locks in "zero behavior change" before any shared code is touched. Must stay green through every later task.

**Files:**
- Test: `tests/test_examples_regression.py`

**Interfaces:**
- Consumes: existing `run()` path via a simple deterministic workflow built with the IR directly (no randomness).
- Produces: `test_inprocess_two_role_branch_golden()` — the canonical finite fixture reused by later tasks.

- [ ] **Step 1: Write the characterization test**

```python
# tests/test_examples_regression.py
"""Characterization gate: the in-process run() path must not change behavior.

Uses a deterministic two-role, one-exchange, one-branch workflow built directly
from the IR (no builder, no randomness) so the assertion is exact.
"""
from zippergen.syntax import (
    Workflow, Lifeline, Var, VarExpr, LitExpr, MsgStmt, IfStmt, SeqStmt, EmptyStmt,
)
from zippergen.actions import pure
from zippergen.runtime import run

A = Lifeline("A")
B = Lifeline("B")
x = Var("x", int)
ok = Var("ok", bool)

@pure
def is_positive(x: int) -> bool:
    return x > 0

def _two_role_branch_workflow() -> Workflow:
    # A sends x to B; B decides ok = is_positive(x) and both learn the outcome.
    body = SeqStmt(
        MsgStmt(A, (VarExpr(x),), B, (VarExpr(x),)),
        SeqStmt(
            # act B: ok = is_positive(x)
            __import__("zippergen.syntax", fromlist=["ActStmt"]).ActStmt(
                B, is_positive, (VarExpr(x),), (VarExpr(ok),)
            ),
            IfStmt(
                condition=lambda _e: _e.ok,
                owner=B,
                branch_true=MsgStmt(B, (LitExpr(True),), A, (VarExpr(ok),)),
                branch_false=MsgStmt(B, (LitExpr(False),), A, (VarExpr(ok),)),
            ),
        ),
    )
    return Workflow(
        name="two_role_branch",
        body=body,
        inputs=(("x", int, A),),
        outputs=((ok, A),),
        ns={"x": x, "ok": ok},
    )

def test_inprocess_two_role_branch_golden():
    wf = _two_role_branch_workflow()
    result = run(wf, [A, B], {"A": {"x": 7}}, timeout=10)
    assert result is True

def test_inprocess_two_role_branch_false():
    wf = _two_role_branch_workflow()
    result = run(wf, [A, B], {"A": {"x": -3}}, timeout=10)
    assert result is False
```

- [ ] **Step 2: Run it to verify it passes on today's code**

Run: `python -m pytest tests/test_examples_regression.py -v`
Expected: PASS (2 passed). If the `Workflow`/`ActStmt` construction signature differs, adjust the fixture to match the real dataclass fields — do not change `runtime.py`.

- [ ] **Step 3: Run the full suite to record the green baseline**

Run: `python -m pytest tests/ -q`
Expected: all pass. This is the baseline every later task must preserve.

- [ ] **Step 4: Checkpoint — do NOT commit**

Do not run `git commit` or `git push`. Leave changes in the working tree for human review. Files touched: `tests/test_examples_regression.py`.

---

## Task 1: Channel abstraction (behavior-preserving extraction)

**Files:**
- Create: `src/zippergen/channels.py`
- Modify: `src/zippergen/runtime.py` (remove `_SeqQueue`/`Channels`; route channel access through the interface; build `InProcessChannel()`)
- Test: `tests/test_channels.py`

**Interfaces:**
- Produces:
  - `class InProcessChannel` with:
    - `put(self, sender: str, receiver: str, channel: str, values: tuple, vc: dict|None=None, view: dict|None=None, field_view: dict|None=None) -> int`
    - `try_get(self, sender: str, receiver: str, channel: str) -> tuple|None` (item is `(seq, values, vc, view, field_view)`, or `None` when empty)
    - `get(self, sender: str, receiver: str, channel: str, *, stop=None) -> tuple`
  - `_SeqQueue` (moved verbatim from `runtime.py`).
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_channels.py
import queue
import pytest
from zippergen.channels import InProcessChannel

def test_put_then_try_get_roundtrip():
    ch = InProcessChannel()
    seq = ch.put("A", "B", "main", (1, 2))
    assert seq == 0
    item = ch.try_get("A", "B", "main")
    assert item is not None
    got_seq, values, vc, view, field_view = item
    assert got_seq == 0 and values == (1, 2) and vc is None

def test_try_get_empty_returns_none():
    ch = InProcessChannel()
    assert ch.try_get("A", "B", "main") is None

def test_fifo_order_per_channel():
    ch = InProcessChannel()
    ch.put("A", "B", "main", (1,))
    ch.put("A", "B", "main", (2,))
    assert ch.try_get("A", "B", "main")[1] == (1,)
    assert ch.try_get("A", "B", "main")[1] == (2,)

def test_channels_are_isolated_by_key():
    ch = InProcessChannel()
    ch.put("A", "B", "main", (1,))
    assert ch.try_get("A", "C", "main") is None
    assert ch.try_get("A", "B", "ctrl") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_channels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zippergen.channels'`.

- [ ] **Step 3: Create `channels.py` (move `_SeqQueue`, add `InProcessChannel`)**

```python
# src/zippergen/channels.py
"""Channel abstraction shared by the in-process and durable runtimes.

The interpreter (`_step` / `_exec`) touches channels only through three
operations: put (send), try_get (non-blocking recv), get (blocking recv).
Items are 5-tuples ``(seq, values, vc, view, field_view)``; vc/view/field_view
are the sender's monitor snapshot, or None when monitoring is inactive.
"""
from __future__ import annotations

import queue
import threading
from collections import defaultdict

Item = tuple[int, tuple, "dict | None", "dict | None", "dict | None"]


class _SeqQueue:
    """FIFO queue that auto-stamps each item with a per-channel sequence number."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._next = 0

    def put(self, values: tuple, vc: dict | None = None, view: dict | None = None,
            field_view: dict | None = None) -> int:
        seq = self._next
        self._next += 1
        self._q.put((seq, values, vc, view, field_view))
        return seq

    def get(self, *, stop: threading.Event | None = None) -> Item:
        if stop is None:
            return self._q.get()
        while True:
            try:
                return self._q.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    raise RuntimeError("Workflow cancelled: another lifeline failed")

    def get_nowait(self) -> Item:
        return self._q.get_nowait()


class InProcessChannel:
    """In-memory channel map keyed by (sender, receiver, channel)."""

    def __init__(self) -> None:
        self._qs: dict[tuple[str, str, str], _SeqQueue] = defaultdict(_SeqQueue)

    def put(self, sender: str, receiver: str, channel: str, values: tuple,
            vc: dict | None = None, view: dict | None = None,
            field_view: dict | None = None) -> int:
        return self._qs[(sender, receiver, channel)].put(values, vc, view, field_view)

    def try_get(self, sender: str, receiver: str, channel: str) -> Item | None:
        try:
            return self._qs[(sender, receiver, channel)].get_nowait()
        except queue.Empty:
            return None

    def get(self, sender: str, receiver: str, channel: str, *,
            stop: threading.Event | None = None) -> Item:
        return self._qs[(sender, receiver, channel)].get(stop=stop)
```

- [ ] **Step 4: Run channel test to verify it passes**

Run: `python -m pytest tests/test_channels.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Route `runtime.py` through the interface**

In `src/zippergen/runtime.py` make exactly these edits:

1. Delete the `_SeqQueue` class definition and the `Channels = defaultdict[...]` alias. Add to the imports near the top:

```python
from zippergen.channels import _SeqQueue, InProcessChannel  # noqa: F401
```

2. Replace the body of `_try_channel_get` with a delegation:

```python
def _try_channel_get(ch, sender: str, receiver: str, channel: str):
    return ch.try_get(sender, receiver, channel)
```

3. In `_receive_any`, replace the inner queue access:

```python
# was: return sender, ch[(sender, receiver, channel)].get_nowait()
item = ch.try_get(sender, receiver, channel)
if item is not None:
    return sender, item
```

(Keep the surrounding `for sender in sorted(...)` loop and the `time.sleep(0.01)` fallback unchanged.)

4. In `_exec`, `SendStmt` case, replace the two `ch[(A.name, B.name, channel)].put(...)` calls:

```python
if monitor:
    monitor.on_event("send", env)
    seq = ch.put(A.name, B.name, channel, values,
                 monitor.snapshot_vc(), monitor.snapshot_view(), monitor.snapshot_field_view())
else:
    seq = ch.put(A.name, B.name, channel, values)
```

5. In `_exec`, replace every blocking `ch[(B.name, A.name, channel)].get(stop=stop)` (in `RecvStmt`, `IfRecvStmt`, `WhileRecvStmt`, and the `ReceiveAnyStmt` via `_receive_any` already handled) with:

```python
seq, values, recv_vc, recv_view, recv_field_view = ch.get(B.name, A.name, channel, stop=stop)
```

6. In `run()`, replace the channel construction:

```python
# was: channels: Channels = defaultdict(_SeqQueue)
channels = InProcessChannel()
```

- [ ] **Step 6: Run the regression gate and full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass, including `tests/test_examples_regression.py` and existing `tests/test_runtime.py`. If anything fails, the extraction changed behavior — fix until identical. Do not proceed otherwise.

- [ ] **Step 7: Checkpoint — do NOT commit**

Do not run `git commit` or `git push`. Leave changes in the working tree for human review. Files touched: `src/zippergen/channels.py`, `src/zippergen/runtime.py`, `tests/test_channels.py`.

---

## Task 2: SQLite store schema and helpers

**Files:**
- Create: `src/zippergen/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces:
  - `open_store(path: str) -> sqlite3.Connection` — WAL mode, autocommit (`isolation_level=None`), schema ensured.
  - `chan_key(sender: str, receiver: str, channel: str) -> str` — `"sender|receiver|channel"`.
  - `SCHEMA: str` — the DDL.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
import sqlite3
from zippergen.store import open_store, chan_key

def test_open_store_creates_tables(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"events", "cursors"} <= names

def test_open_store_is_wal(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

def test_insert_event_autoincrements_rowid(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    conn.execute("BEGIN")
    c1 = conn.execute("INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp)"
                      " VALUES('A','B','main','msg','[1]',NULL)")
    c2 = conn.execute("INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp)"
                      " VALUES('A','B','main','msg','[2]',NULL)")
    conn.execute("COMMIT")
    assert c2.lastrowid == c1.lastrowid + 1

def test_chan_key_roundtrip():
    assert chan_key("A", "B", "main") == "A|B|main"
    assert chan_key("A", "B", "main").split("|") == ["A", "B", "main"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zippergen.store'`.

- [ ] **Step 3: Create `store.py`**

```python
# src/zippergen/store.py
"""SQLite-backed durable event store: transport, replay log, and observation
stream in one append-only table. All writes serialize through one file, so
`rowid` is a global total order consistent with causality.
"""
from __future__ import annotations

import sqlite3

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
"""


def open_store(path: str) -> sqlite3.Connection:
    # isolation_level=None -> autocommit; we drive BEGIN/COMMIT explicitly.
    # check_same_thread=False: the connection is created by the supervisor and
    # driven from the role's loop thread; each connection is used by one thread
    # at a time. Required for the per-role loop and the threaded Task 4 tests.
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")  # wait, don't fail, on concurrent writers
    conn.executescript(SCHEMA)
    return conn


def chan_key(sender: str, receiver: str, channel: str) -> str:
    return f"{sender}|{receiver}|{channel}"
```

- [ ] **Step 4: Run store test to verify it passes**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Checkpoint — do NOT commit**

Do not run `git commit` or `git push`. Leave changes in the working tree for human review. Files touched: `src/zippergen/store.py`, `tests/test_store.py`.

---

## Task 3: DurableChannel (send=INSERT, recv=cursor, replay queues)

**Files:**
- Modify: `src/zippergen/store.py` (add `DurableChannel`)
- Test: `tests/test_store.py` (extend)

**Interfaces:**
- Produces `class DurableChannel`:
  - `__init__(self, conn: sqlite3.Connection, role: str)` — loads durable cursors and builds replay queues from the committed log.
  - `put/try_get/get` — same signatures as `InProcessChannel` (Task 1 interface). In replay mode `put` does not INSERT and `try_get` serves recorded rows; in live mode `put` INSERTs and `try_get` reads the next uncommitted-cursor row.
  - `replaying(self) -> bool` — True while any recorded I/O remains to re-consume.
  - `commit_txn(self)` — flush tentative consumed-cursors then `COMMIT`; promote tentative→durable.
  - `rollback_txn(self)` — `ROLLBACK`; drop tentative.
- Consumes: `open_store`, `chan_key` (Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store.py  (append)
from collections import deque
from zippergen.store import DurableChannel

def test_durable_put_live_inserts_and_recv_reads(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); a.put("A", "B", "main", (5,)); a.commit_txn()
    conn.execute("BEGIN")
    item = b.try_get("A", "B", "main")
    assert item is not None and item[1] == (5,)
    b.commit_txn()
    # cursor is durable: a fresh DurableChannel does not re-serve it live
    b2 = DurableChannel(conn, "B")
    conn.execute("BEGIN")
    assert b2.try_get("A", "B", "main") is None
    b2.rollback_txn()

def test_durable_rollback_does_not_advance_cursor(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (9,)); a.commit_txn()
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); assert b.try_get("A", "B", "main")[1] == (9,); b.rollback_txn()
    # not consumed -> re-delivered on a fresh receiver
    b2 = DurableChannel(conn, "B")
    conn.execute("BEGIN"); assert b2.try_get("A", "B", "main")[1] == (9,); b2.rollback_txn()

def test_durable_replay_reserves_recorded_sends_and_recvs(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    # Simulate a prior committed history: A sent (1,), and B consumed it.
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); b.try_get("A", "B", "main"); b.commit_txn()
    # Restart B: it must replay its already-consumed recv, not re-read live.
    b_restart = DurableChannel(conn, "B")
    assert b_restart.replaying() is True
    item = b_restart.try_get("A", "B", "main")   # served from replay queue, no txn
    assert item[1] == (1,)
    assert b_restart.replaying() is False
    # Restart A: it must NOT re-INSERT its recorded send.
    a_restart = DurableChannel(conn, "A")
    assert a_restart.replaying() is True
    a_restart.put("A", "B", "main", (1,))        # reserved from replay, no INSERT
    assert a_restart.replaying() is False
    count = conn.execute("SELECT COUNT(*) FROM events WHERE sender='A'").fetchone()[0]
    assert count == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_store.py -k durable -v`
Expected: FAIL with `ImportError: cannot import name 'DurableChannel'`.

- [ ] **Step 3: Implement `DurableChannel` in `store.py`**

```python
# src/zippergen/store.py  (append)
import json
import time
import threading
from collections import deque

Item = tuple[int, tuple, "dict | None", "dict | None", "dict | None"]


class DurableChannel:
    """Channel backed by the shared event store, with replay/live semantics.

    Same put/try_get/get surface as InProcessChannel. Consumption is tentative
    until commit_txn(): only then does the durable consume-cursor advance, in the
    same transaction as any emitted sends. On restart the constructor rebuilds
    per-key replay queues from the committed log so re-execution reserves recorded
    sends (no re-INSERT) and re-serves recorded recvs (no live read).
    """

    def __init__(self, conn: sqlite3.Connection, role: str) -> None:
        self.conn = conn
        self.role = role
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
        # Recorded outbound sends by this role, in commit order.
        for rowid, receiver, channel in self.conn.execute(
            "SELECT rowid, receiver, channel FROM events "
            "WHERE sender=? AND kind IN ('msg','ctrl') ORDER BY rowid", (self.role,)
        ).fetchall():
            self._replay_outbox.append((rowid, receiver, channel))
        # Recorded inbound rows already consumed (rowid <= durable cursor).
        for (sender, receiver, channel), consumed in self._consumed.items():
            rows = self.conn.execute(
                "SELECT rowid, payload, causal_stamp FROM events "
                "WHERE sender=? AND receiver=? AND channel=? AND rowid<=? ORDER BY rowid",
                (sender, receiver, channel, consumed),
            ).fetchall()
            if rows:
                self._replay_inbox[(sender, receiver, channel)] = deque(rows)

    def replaying(self) -> bool:
        return bool(self._replay_outbox) or any(self._replay_inbox.values())

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
```

- [ ] **Step 4: Run the durable tests**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS (all, including the three new `durable` tests).

- [ ] **Step 5: Checkpoint — do NOT commit**

Do not run `git commit` or `git push`. Leave changes in the working tree for human review. Files touched: `src/zippergen/store.py`, `tests/test_store.py`.

---

## Task 4: Per-role loop (`run_role`) with replay→live and per-step transactions

**Files:**
- Create: `src/zippergen/serve.py`
- Test: `tests/test_serve_replay.py`

**Interfaces:**
- Produces `run_role(conn, role: str, local_stmt, env: dict, ns: dict, *, llm_backend=None, human_backend=None, trace=None) -> dict`
  - Drives `_step` (from `runtime`) to completion. Replay phase: no transactions, no trace, channel reserves recorded I/O. Live phase: each progressing step runs inside one `BEGIN…COMMIT`; blocked steps `ROLLBACK` and poll. Returns the final `env`.
- Consumes: `_step` from `zippergen.runtime`, `DurableChannel` (Task 3), `EmptyStmt` from `zippergen.syntax`, `mock_llm` from `zippergen.runtime`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serve_replay.py
"""run_role drives one role over the durable store, and a fresh run_role on the
same store resumes to the identical final env without duplicating events."""
from zippergen.syntax import Lifeline, Var, VarExpr, LitExpr, MsgStmt, IfStmt, SeqStmt, ActStmt
from zippergen.actions import pure
from zippergen.projection import project
from zippergen.store import open_store
from zippergen.serve import run_role
from tests.test_examples_regression import _two_role_branch_workflow, A, B

def _run_both(conn_a, conn_b, wf, seed):
    la = project(wf, A); lb = project(wf, B)
    import threading
    envs = {}
    def go(conn, role, local, seed_env):
        envs[role] = run_role(conn, role, local, dict(seed_env), wf.ns)
    ta = threading.Thread(target=go, args=(conn_a, "A", la, seed))
    tb = threading.Thread(target=go, args=(conn_b, "B", lb, {}))
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)
    return envs

def test_run_role_completes_two_role_branch(tmp_path):
    path = str(tmp_path / "s.sqlite")
    wf = _two_role_branch_workflow()
    envs = _run_both(open_store(path), open_store(path), wf, {"x": 7})
    assert envs["A"]["ok"] is True

def test_run_role_replay_is_idempotent(tmp_path):
    path = str(tmp_path / "s.sqlite")
    wf = _two_role_branch_workflow()
    conn_a, conn_b = open_store(path), open_store(path)
    envs1 = _run_both(conn_a, conn_b, wf, {"x": 7})
    before = conn_a.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    # Re-run both roles on the same store: pure replay, no new events.
    la = project(wf, A); lb = project(wf, B)
    env_a = run_role(open_store(path), "A", la, {"x": 7}, wf.ns)
    env_b = run_role(open_store(path), "B", lb, {}, wf.ns)
    after = conn_a.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert env_a["ok"] is True
    assert after == before  # replay reserved every recorded send; nothing new inserted
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_serve_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zippergen.serve'`.

- [ ] **Step 3: Implement `run_role` in `serve.py`**

```python
# src/zippergen/serve.py
"""Per-role durable runtime: project one role, replay its committed history,
then run live, persisting each step atomically."""
from __future__ import annotations

import time

from zippergen.syntax import EmptyStmt
from zippergen.runtime import _step, mock_llm
from zippergen.store import DurableChannel


def run_role(conn, role: str, local_stmt, env: dict, ns: dict, *,
             llm_backend=None, human_backend=None, trace=None) -> dict:
    if llm_backend is None:
        llm_backend = mock_llm
    if human_backend is None:
        from zippergen.human_backends import make_cli_human_backend
        human_backend = make_cli_human_backend()

    ch = DurableChannel(conn, role)
    residual = local_stmt

    # ---- replay: reconstruct (env, residual) from committed history --------
    # No transactions, no trace: put reserves recorded sends, try_get serves
    # recorded recvs. Determinism of local steps reproduces the exact boundary.
    while ch.replaying() and not isinstance(residual, EmptyStmt):
        residual, progressed = _step(
            residual, env, ch, ns, llm_backend, human_backend, None, None, {}, None)
        if not progressed:
            break  # blocked on live input at the replay boundary

    # ---- live: one transaction per progressing step ------------------------
    while not isinstance(residual, EmptyStmt):
        conn.execute("BEGIN")
        new_residual, progressed = _step(
            residual, env, ch, ns, llm_backend, human_backend, None, trace, {}, None)
        if progressed:
            ch.commit_txn()
            residual = new_residual
        else:
            ch.rollback_txn()
            time.sleep(0.02)
    return env
```

- [ ] **Step 4: Run the replay tests**

Run: `python -m pytest tests/test_serve_replay.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Confirm no regression**

Run: `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Checkpoint — do NOT commit**

Do not run `git commit` or `git push`. Leave changes in the working tree for human review. Files touched: `src/zippergen/serve.py`, `tests/test_serve_replay.py`.

---

## Task 5: `zippergen serve` CLI, seed handling, and systemd unit

**Files:**
- Modify: `src/zippergen/serve.py` (add `load_workflow`, seed persistence, `main`)
- Modify: `pyproject.toml` (add console entry point `zippergen`)
- Create: `deploy/zippergen@.service`
- Test: `tests/test_serve_replay.py` (extend with a CLI-level seed test)

**Interfaces:**
- Produces:
  - `load_workflow(module_path: str, role_name: str) -> tuple[Workflow, Lifeline]` — imports a `.py` workflow module, finds the module-level `Workflow` and the `Lifeline` whose `.name == role_name`.
  - `seed_env(conn, role, wf, inputs: dict) -> dict` — on first start writes a `kind='seed'` event carrying this role's inputs and returns them; on restart reads the seed event back (ignores `inputs`), guaranteeing an identical seed.
  - `main(argv=None) -> int` — CLI: `serve --workflow PATH --role NAME --store PATH [--input k=v ...]`.
- Consumes: `run_role` (Task 4), `open_store` (Task 2), `project` (`zippergen.projection`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serve_replay.py  (append)
import json
from zippergen.store import open_store
from zippergen.serve import seed_env
from tests.test_examples_regression import _two_role_branch_workflow, A

def test_seed_env_persists_then_reads_back(tmp_path):
    path = str(tmp_path / "s.sqlite")
    wf = _two_role_branch_workflow()
    conn1 = open_store(path)
    got1 = seed_env(conn1, "A", wf, {"x": 42})
    assert got1 == {"x": 42}
    # Restart: different inputs are ignored; the recorded seed wins.
    conn2 = open_store(path)
    got2 = seed_env(conn2, "A", wf, {"x": -1})
    assert got2 == {"x": 42}
    rows = conn2.execute("SELECT COUNT(*) FROM events WHERE kind='seed' AND sender='A'").fetchone()[0]
    assert rows == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_serve_replay.py -k seed -v`
Expected: FAIL with `ImportError: cannot import name 'seed_env'`.

- [ ] **Step 3: Add `load_workflow`, `seed_env`, `main` to `serve.py`**

```python
# src/zippergen/serve.py  (append)
import argparse
import importlib.util
import json
import os
import sys

from zippergen.syntax import Workflow, Lifeline
from zippergen.projection import project
from zippergen.store import open_store


def load_workflow(module_path: str, role_name: str) -> tuple[Workflow, Lifeline]:
    spec = importlib.util.spec_from_file_location("_zippergen_wf", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wf = next(v for v in vars(module).values() if isinstance(v, Workflow))
    lifelines = {ll.name: ll for ll in _workflow_lifelines(wf)}
    if role_name not in lifelines:
        raise SystemExit(f"role {role_name!r} not in workflow lifelines {sorted(lifelines)}")
    return wf, lifelines[role_name]


def _workflow_lifelines(wf: Workflow) -> tuple[Lifeline, ...]:
    from zippergen.syntax import _ordered_workflow_lifelines
    return _ordered_workflow_lifelines(wf)


def seed_env(conn, role: str, wf: Workflow, inputs: dict) -> dict:
    row = conn.execute(
        "SELECT payload FROM events WHERE kind='seed' AND sender=? ORDER BY rowid LIMIT 1",
        (role,),
    ).fetchone()
    if row is not None:
        return json.loads(row[0])
    conn.execute("BEGIN")
    conn.execute(
        "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
        "VALUES(?,?,?,?,?,?)",
        (role, None, None, "seed", json.dumps(inputs), None),
    )
    conn.execute("COMMIT")
    return dict(inputs)


def _parse_inputs(pairs: list[str]) -> dict:
    out: dict = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)      # 7 -> int, "true" via JSON, '"s"' -> str
        except json.JSONDecodeError:
            out[k] = v                  # bare string fallback
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zippergen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="run one role as a durable process")
    sv.add_argument("--workflow", required=True)
    sv.add_argument("--role", required=True)
    sv.add_argument("--store", required=True)
    sv.add_argument("--input", action="append", default=[], metavar="k=v")
    args = ap.parse_args(argv)

    wf, role_ll = load_workflow(args.workflow, args.role)
    conn = open_store(args.store)
    env = seed_env(conn, args.role, wf, _parse_inputs(args.input))
    local = project(wf, role_ll)
    final = run_role(conn, args.role, local, env, wf.ns)
    print(json.dumps({k: v for k, v in final.items()
                      if isinstance(v, (bool, int, float, str, type(None)))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the seed test**

Run: `python -m pytest tests/test_serve_replay.py -k seed -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Register the console entry point**

In `pyproject.toml`, add under `[project.scripts]` (create the table if absent):

```toml
[project.scripts]
zippergen = "zippergen.serve:main"
```

Run: `pip install -e .` then `zippergen serve --help`
Expected: usage text prints; exit 0.

- [ ] **Step 6: Add the systemd unit template**

```ini
# deploy/zippergen@.service
[Unit]
Description=ZipperGen role %i
After=network.target

[Service]
Type=simple
Environment=ZG_STORE=/var/lib/zippergen/workflow.sqlite
Environment=ZG_WORKFLOW=/opt/zippergen/workflow.py
ExecStart=/usr/bin/zippergen serve --workflow ${ZG_WORKFLOW} --role %i --store ${ZG_STORE}
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 7: Run full suite and commit**

Run: `python -m pytest tests/ -q`
Expected: all pass.

Do not run `git commit` or `git push`. Leave changes in the working tree for human review. Files touched: `src/zippergen/serve.py`, `pyproject.toml`, `deploy/zippergen@.service`, `tests/test_serve_replay.py`.

---

## Task 6: Kill-and-resume integration test

Validates the one correctness-critical property: a role killed mid-protocol resumes from its last committed step, with no lost or duplicated events.

**Files:**
- Test: `tests/test_serve_replay.py` (extend)

**Interfaces:**
- Consumes: `run_role`, `open_store`, `project`, `_two_role_branch_workflow`.

- [ ] **Step 1: Write the crash-injection test**

```python
# tests/test_serve_replay.py  (append)
import threading
from zippergen.projection import project
from zippergen.store import open_store, DurableChannel
from zippergen.serve import run_role
from tests.test_examples_regression import _two_role_branch_workflow, A, B

class _Crash(Exception):
    pass

def _run_role_crash_after_n_commits(path, role, local, seed, ns, n):
    """Run a role but raise _Crash right after the n-th live commit."""
    conn = open_store(path)
    ch = DurableChannel(conn, role)
    from zippergen.syntax import EmptyStmt
    from zippergen.runtime import _step, mock_llm
    from zippergen.human_backends import make_cli_human_backend
    hb = make_cli_human_backend()
    env = dict(seed); residual = local; commits = 0
    while ch.replaying() and not isinstance(residual, EmptyStmt):
        residual, prog = _step(residual, env, ch, ns, mock_llm, hb, None, None, {}, None)
        if not prog:
            break
    while not isinstance(residual, EmptyStmt):
        conn.execute("BEGIN")
        new_r, prog = _step(residual, env, ch, ns, mock_llm, hb, None, None, {}, None)
        if prog:
            ch.commit_txn(); residual = new_r; commits += 1
            if commits == n:
                raise _Crash()
        else:
            ch.rollback_txn()
    return env

def test_kill_and_resume_reaches_same_state(tmp_path):
    path = str(tmp_path / "s.sqlite")
    wf = _two_role_branch_workflow()
    la = project(wf, A); lb = project(wf, B)

    # B runs to completion in a background thread; A crashes after its 1st commit.
    envs = {}
    def run_b():
        envs["B"] = run_role(open_store(path), "B", lb, {}, wf.ns)
    tb = threading.Thread(target=run_b); tb.start()

    try:
        _run_role_crash_after_n_commits(path, "A", la, {"x": 7}, wf.ns, n=1)
    except _Crash:
        pass

    # Supervisor "restarts" A: fresh process, same store, no --input needed
    # because the seed was recorded. It must replay past the committed send.
    env_a = run_role(open_store(path), "A", la, {"x": 7}, wf.ns)
    tb.join(timeout=10)

    assert env_a["ok"] is True
    # Exactly one send from A survived (its first message), never duplicated.
    conn = open_store(path)
    a_sends = conn.execute(
        "SELECT COUNT(*) FROM events WHERE sender='A' AND kind='msg'").fetchone()[0]
    assert a_sends == 1
```

- [ ] **Step 2: Run to verify it fails or passes**

Run: `python -m pytest tests/test_serve_replay.py -k kill_and_resume -v`
Expected: PASS. If it fails on a duplicated send, the replay-reserve path in `DurableChannel.put` is not being hit on restart — verify `_load_replay` reads committed `sender=A` rows.

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 4: Checkpoint — do NOT commit**

Do not run `git commit` or `git push`. Leave changes in the working tree for human review. Files touched: `tests/test_serve_replay.py`.

---

## Deferred (follow-on plan)

- **Snapshots (spec §8)** — `snapshots` table + residual `LocalStmt` (de)serialization so non-terminating workflows (`command_center`) replay only the tail. Required before reactive deployment; not needed to validate durable resume.
- **External-effect exactly-once (spec §7)** — memoized `effect` events + idempotency keys, once LLM/email actions are in scope.
- **Reactive v2 (spec §14)** — `instance_id` column + triggering for repeated protocol instances.
- **Latency:** replace polling with a unix-socket / filesystem-watch wake-up (optimization only; correctness already holds via the store).

## Self-Review Notes

- **Spec coverage:** §2 architecture → Tasks 1–5; §4 channel abstraction → Task 1; §5 schema → Task 2; §6 replay/live → Tasks 3–4; §7 internal effect idempotence → Task 3 (`put` reserve) + Task 6; §9 commit invariant → Task 4; §10 no codegen → Task 5 (`project` at startup); §3 supervision → Task 5 unit; §11 monitor/dashboard readers → served by the `events` table (no code, read-only); §12 testing → Tasks 0 + 6; backward-compat goal → Task 0 gate held through Task 1. §8 snapshots explicitly deferred.
- **Type consistency:** `put/try_get/get` signatures identical across `InProcessChannel` and `DurableChannel`; `run_role` signature identical in Task 4 and reused in Task 6; item shape `(seq, values, vc, view, field_view)` uniform.
