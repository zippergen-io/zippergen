# Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound restart replay length for long-running `while`-loop workflows by checkpointing `(env, program-position, per-channel replay floor)` at loop boundaries and replaying only the tail on restart.

**Architecture:** Never serialize the program (its residual holds lambdas and action objects). Instead persist cheap JSON state plus a *locator* (child-index path) that re-derives program position by re-projecting the workflow; and a per-channel *floor* (own-send high-water + consumed cursors) so replay covers only events after the boundary. A snapshot is a rebuildable cache — any doubt on restart discards it and falls back to today's full-replay-from-seed, which is already proven correct.

**Tech Stack:** Python 3.11+ stdlib only (`sqlite3`, `json`). Tests: `pytest`. Builds on the durable runtime in `channels.py`/`store.py`/`serve.py`.

## Global Constraints

- Python **3.11+**, stdlib only. **No new third-party dependencies.**
- **NEVER `git commit`/`push`/`add`** — leave all changes in the working tree; no AI/`Co-Authored-By` attribution anywhere. A checkpoint step that says "do NOT commit" means exactly that.
- **A snapshot must never weaken durable resume.** On any invalid/absent/stale snapshot, fall back to full replay from seed (`since=None`). Correctness never depends on a snapshot being present or loadable.
- `since=None` / no-snapshot path must reproduce today's behavior **exactly** — the full deployable-runtime suite stays green at every step.
- Spec of record: `docs/superpowers/specs/2026-07-03-snapshots-design.md`.
- Follow repo conventions: frozen-dataclass IR is never mutated; `match`/`isinstance` cover union members; transactions use `BEGIN IMMEDIATE` (see existing `seed_env`/`run_role`).

---

## File Structure

- `src/zippergen/store.py` — **modify.** Add `snapshots` table to `SCHEMA`; add `write_snapshot`/`load_snapshot`; add `DurableChannel.position()` and an optional `since` param that filters `_load_replay` to the tail.
- `src/zippergen/locator.py` — **new.** `loop_node_paths(root)` and `resolve_path(root, path)` — pure tree helpers, no runtime deps.
- `src/zippergen/serve.py` — **modify `run_role` only.** Build the loop-path map, load+validate a snapshot (restore or seed), and write a snapshot at each loop boundary. `seed_env`/`load_workflow`/`main` unchanged.
- `tests/test_locator.py` — **new.**
- `tests/test_store.py` — **extend** (snapshot helpers, `position`, `since` filtering).
- `tests/test_snapshots.py` — **new** (integration: snapshot appears, restart-matches-full-replay, crash-after-snapshot, stale fallback, no-snapshot unchanged).

---

## Task 1: `snapshots` table + read/write helpers

**Files:**
- Modify: `src/zippergen/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces:
  - `write_snapshot(conn, role: str, env: dict, locator: list[int], floor: dict) -> None` — upsert (latest-only); JSON-encodes args *before* opening the transaction so a non-serializable `env` raises without leaving an open transaction.
  - `load_snapshot(conn, role: str) -> dict | None` — returns `{"env": dict, "locator": list, "floor": dict}` or `None`.
- Consumes: `open_store` (existing).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py  (append)
from zippergen.store import write_snapshot, load_snapshot

def test_snapshot_roundtrip(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    assert load_snapshot(conn, "A") is None
    write_snapshot(conn, "A", {"n": 3}, [1], {"out": 5, "cursors": {"B|A|main": 4}})
    snap = load_snapshot(conn, "A")
    assert snap == {"env": {"n": 3}, "locator": [1], "floor": {"out": 5, "cursors": {"B|A|main": 4}}}

def test_snapshot_is_latest_only(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    write_snapshot(conn, "A", {"n": 1}, [], {"out": 0, "cursors": {}})
    write_snapshot(conn, "A", {"n": 2}, [], {"out": 9, "cursors": {}})
    assert load_snapshot(conn, "A")["env"] == {"n": 2}
    assert conn.execute("SELECT COUNT(*) FROM snapshots WHERE role='A'").fetchone()[0] == 1

def test_write_snapshot_nonserializable_raises_no_open_txn(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    import pytest
    with pytest.raises(TypeError):
        write_snapshot(conn, "A", {"bad": object()}, [], {"out": 0, "cursors": {}})
    # connection is still usable (no dangling transaction)
    write_snapshot(conn, "A", {"n": 1}, [], {"out": 0, "cursors": {}})
    assert load_snapshot(conn, "A")["env"] == {"n": 1}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_store.py -k snapshot -v`
Expected: FAIL (`ImportError: cannot import name 'write_snapshot'`).

- [ ] **Step 3: Implement in `store.py`**

Add the table to `SCHEMA` (inside the existing triple-quoted string, after the `cursors` table):

```sql
CREATE TABLE IF NOT EXISTS snapshots (
  role    TEXT PRIMARY KEY,
  env     BLOB NOT NULL,
  locator BLOB NOT NULL,
  floor   BLOB NOT NULL
);
```

Add the helpers (near `chan_key`):

```python
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
```

(`json` is already imported in `store.py`.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_store.py -k snapshot -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (existing count + 3).

- [ ] **Step 6: Checkpoint — do NOT commit**

Do not run `git commit`/`push`. Leave changes in the working tree. Files touched: `src/zippergen/store.py`, `tests/test_store.py`.

---

## Task 2: Loop-node locator (`locator.py`)

**Files:**
- Create: `src/zippergen/locator.py`
- Test: `tests/test_locator.py`

**Interfaces:**
- Produces:
  - `loop_node_paths(root) -> dict[int, list[int]]` — maps `id(node)` → child-index path for every `WhileStmt`/`WhileRecvStmt` reachable from `root`.
  - `resolve_path(root, path: list[int])` — walks the path via canonical child ordering; returns the node, or `None` if any index is out of range.
- Consumes: IR node types from `zippergen.syntax`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_locator.py
from zippergen.syntax import (
    SeqStmt, WhileStmt, EmptyStmt, SendStmt, Lifeline, VarExpr, Var, seq,
)
from zippergen.locator import loop_node_paths, resolve_path

A = Lifeline("A"); B = Lifeline("B")
x = Var("x", int)

def _while(body):
    return WhileStmt(condition=lambda _e: True, owner=A, body=body, exit_body=EmptyStmt())

def test_whole_program_is_loop_has_empty_path():
    w = _while(SendStmt(A, (VarExpr(x),), B))
    paths = loop_node_paths(w)
    assert paths == {id(w): []}
    assert resolve_path(w, []) is w

def test_prefix_then_loop_path_is_index_one():
    w = _while(SendStmt(A, (VarExpr(x),), B))
    prog = seq(SendStmt(A, (VarExpr(x),), B), w)   # SeqStmt(first=send, second=while)
    paths = loop_node_paths(prog)
    assert paths[id(w)] == [1]
    assert resolve_path(prog, [1]) is w

def test_resolve_out_of_range_returns_none():
    w = _while(SendStmt(A, (VarExpr(x),), B))
    assert resolve_path(w, [5]) is None
    assert resolve_path(w, [0, 0, 0]) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_locator.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'zippergen.locator'`).

- [ ] **Step 3: Create `src/zippergen/locator.py`**

```python
"""Locate a loop node in a projected local program by a child-index path.

Used by snapshots: the residual at a loop-iteration boundary is (by identity)
one of these nodes, and the path re-finds it in a freshly-projected program so
the (unserializable) continuation never has to be persisted.
"""
from __future__ import annotations

from zippergen.syntax import (
    SeqStmt, IfStmt, IfRecvStmt, WhileStmt, WhileRecvStmt,
)


def _children(node) -> list:
    # Canonical, stable child ordering per node type. Leaf nodes have no children.
    match node:
        case SeqStmt(first=a, second=b):
            return [a, b]
        case IfStmt(branch_true=t, branch_false=f):
            return [t, f]
        case IfRecvStmt(branch_true=t, branch_false=f):
            return [t, f]
        case WhileStmt(body=b, exit_body=x):
            return [b, x]
        case WhileRecvStmt(body=b, exit_body=x):
            return [b, x]
        case _:
            return []


def loop_node_paths(root) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}

    def walk(node, path: list[int]) -> None:
        if isinstance(node, (WhileStmt, WhileRecvStmt)):
            out[id(node)] = list(path)
        for i, child in enumerate(_children(node)):
            walk(child, path + [i])

    walk(root, [])
    return out


def resolve_path(root, path: list[int]):
    node = root
    for i in path:
        children = _children(node)
        if not isinstance(i, int) or i < 0 or i >= len(children):
            return None
        node = children[i]
    return node
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_locator.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Checkpoint — do NOT commit**

Do not run `git commit`/`push`. Leave changes in the working tree. Files touched: `src/zippergen/locator.py`, `tests/test_locator.py`.

---

## Task 3: `DurableChannel.position()` + `since` (tail-only replay)

**Files:**
- Modify: `src/zippergen/store.py` (`DurableChannel`)
- Test: `tests/test_store.py`

**Interfaces:**
- Produces (on `DurableChannel`):
  - `__init__(self, conn, role, since: dict | None = None)` — `since=None` = today's full replay; a floor dict `{"out": int, "cursors": {chan_key: int}}` restricts `_load_replay` to the tail.
  - `position(self) -> dict` — `{"out": <max rowid of this role's own committed sends>, "cursors": {chan_key: consumed}}`, computed from committed state.
- Consumes: `chan_key` (existing).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store.py  (append)
def test_position_reports_out_and_cursors(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); r = a.put("A", "B", "main", (1,)); a.commit_txn()
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); b.try_get("A", "B", "main"); b.commit_txn()
    assert a.position()["out"] == r
    assert b.position()["cursors"]["A|B|main"] == r

def test_since_none_replays_all_history(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    a2 = DurableChannel(conn, "A", since=None)
    assert a2.replaying() is True  # full history reserved

def test_since_floor_replays_only_tail(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); r1 = a.put("A", "B", "main", (1,)); a.commit_txn()
    conn.execute("BEGIN"); r2 = a.put("A", "B", "main", (2,)); a.commit_txn()
    # Resume with a floor at r1: only the r2 send is in the tail.
    a2 = DurableChannel(conn, "A", since={"out": r1, "cursors": {}})
    a2.put("A", "B", "main", (2,))   # reserves r2 (no new insert)
    assert a2.replaying() is False
    assert conn.execute("SELECT COUNT(*) FROM events WHERE sender='A'").fetchone()[0] == 2

def test_since_floor_filters_inbound_tail(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    conn.execute("BEGIN"); r2 = a.put("A", "B", "main", (2,)); a.commit_txn()
    b = DurableChannel(conn, "B")
    conn.execute("BEGIN"); b.try_get("A", "B", "main"); b.try_get("A", "B", "main"); b.commit_txn()
    # B resumes with a floor at the first consumed rowid: only the 2nd is replayed.
    b2 = DurableChannel(conn, "B", since={"out": 0, "cursors": {"A|B|main": r2 - 1}})
    item = b2.try_get("A", "B", "main")     # replays r2
    assert item[0] == r2
    assert b2.replaying() is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_store.py -k "position or since" -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'since'` and `AttributeError: position`).

- [ ] **Step 3: Modify `DurableChannel`**

Change `__init__` to accept and store `since`:

```python
    def __init__(self, conn: sqlite3.Connection, role: str, since: dict | None = None) -> None:
        self.conn = conn
        self.role = role
        self.since = since
        self._consumed: dict[tuple[str, str, str], int] = {}
        self._tentative: dict[tuple[str, str, str], int] = {}
        self._replay_outbox: deque = deque()
        self._replay_inbox: dict[tuple[str, str, str], deque] = {}
        self._load_cursors()
        self._load_replay()
```

Replace `_load_replay` body to honor `since` (default `None` == all history):

```python
    def _load_replay(self) -> None:
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
```

Add `position` (place near `replaying`):

```python
    def position(self) -> dict:
        row = self.conn.execute(
            "SELECT MAX(rowid) FROM events WHERE sender=? AND kind IN ('msg','ctrl')",
            (self.role,),
        ).fetchone()
        return {
            "out": row[0] or 0,
            "cursors": {chan_key(*key): rowid for key, rowid in self._consumed.items()},
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS (all store tests, including the 4 new ones).

- [ ] **Step 5: Full suite (the `since=None` path must be unchanged)**

Run: `python -m pytest tests/ -q`
Expected: all pass — the deployable-runtime suite is the proof that `since=None` reproduces prior behavior.

- [ ] **Step 6: Checkpoint — do NOT commit**

Do not run `git commit`/`push`. Leave changes in the working tree. Files touched: `src/zippergen/store.py`, `tests/test_store.py`.

---

## Task 4: `run_role` snapshotting + resume

**Files:**
- Modify: `src/zippergen/serve.py` (`run_role` only)
- Test: `tests/test_snapshots.py`

**Interfaces:**
- Consumes: `loop_node_paths`/`resolve_path` (Task 2), `write_snapshot`/`load_snapshot`/`DurableChannel.position`/`since` (Tasks 1, 3), `_step`/`mock_llm` (existing), `WhileStmt`/`WhileRecvStmt`/`EmptyStmt` (syntax).
- Produces: unchanged `run_role` signature; new module-private helpers `_floor_coherent`, `_try_resume`, `_maybe_snapshot`.

- [ ] **Step 1: Write the fixture and a failing test**

First, a deterministic two-role bounded `while`-loop fixture. **Build it with the DSL and VERIFY it against the in-process `run()` before using it for snapshot tests** (the DSL syntax below is a starting point — consult `examples/diagnosis.py` for the exact `while cond @ Owner:` form and adjust until `run()` yields the asserted result; do NOT change source to fit the fixture):

```python
# tests/loop_fixture.py
"""A deterministic two-role bounded loop: A counts up to `limit`, exchanging the
counter with B each iteration. Owner of the loop guard is A. No LLM, no kpar."""
from zippergen import Lifeline, Var, workflow
from zippergen.actions import pure

A = Lifeline("A")
B = Lifeline("B")

@pure
def add_one(n: int) -> int:
    return n + 1

@pure
def relay(m: int) -> int:
    return m

@workflow
def counter_loop(n: int @ A, limit: int @ A):
    while (n < limit) @ A:
        A(n) >> B(m)
        with B:
            ack = relay(m)
        B(ack) >> A(got)
        with A:
            n = add_one(n)
    return n @ A
```

```python
# tests/test_snapshots.py
from zippergen.runtime import run
from tests.loop_fixture import counter_loop, A, B

def test_fixture_runs_inprocess_to_known_result():
    # Sanity-check the fixture on the thread-based path before durable tests.
    assert run(counter_loop, [A, B], {"A": {"n": 0, "limit": 3}}, timeout=10) == 3
```

- [ ] **Step 2: Run the fixture sanity test; fix the fixture until it passes**

Run: `python -m pytest tests/test_snapshots.py::test_fixture_runs_inprocess_to_known_result -v`
Expected: PASS returning `3`. If it errors, adjust the DSL in `tests/loop_fixture.py` to match the real builder syntax (compare to `examples/diagnosis.py`). The acceptance criterion: a 2-role, A-owned, deterministic bounded `while` loop whose result is the final counter.

- [ ] **Step 3: Write the snapshot integration tests (failing)**

```python
# tests/test_snapshots.py  (append)
import threading
from zippergen.projection import project
from zippergen.store import open_store, load_snapshot
from zippergen.serve import run_role
from tests.loop_fixture import counter_loop, A, B

def _run_both(path, seed):
    la = project(counter_loop, A); lb = project(counter_loop, B)
    envs = {}
    def go(role, local, s):
        envs[role] = run_role(open_store(path), role, local, dict(s), counter_loop.ns)
    ta = threading.Thread(target=go, args=("A", la, seed))
    tb = threading.Thread(target=go, args=("B", lb, {}))
    ta.start(); tb.start(); ta.join(timeout=15); tb.join(timeout=15)
    return envs

def test_durable_run_matches_and_writes_snapshot(tmp_path):
    path = str(tmp_path / "s.sqlite")
    envs = _run_both(path, {"n": 0, "limit": 3})
    assert envs["A"]["n"] == 3
    # A owns the loop, so A snapshots at boundaries.
    snap = load_snapshot(open_store(path), "A")
    assert snap is not None and isinstance(snap["locator"], list)
    assert "out" in snap["floor"] and "cursors" in snap["floor"]

def test_resume_from_snapshot_matches_full_run(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _run_both(path, {"n": 0, "limit": 3})
    # Re-run A from its snapshot on the same store: must reach the same final env
    # and (idempotent replay) insert no new events.
    conn = open_store(path)
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    la = project(counter_loop, A)
    env_a = run_role(open_store(path), "A", la, {"n": 0, "limit": 3}, counter_loop.ns)
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert env_a["n"] == 3
    assert after == before
```

- [ ] **Step 4: Run to verify they fail**

Run: `python -m pytest tests/test_snapshots.py -v`
Expected: `test_durable_run_matches_and_writes_snapshot` FAILS (`snap is None` — no snapshotting yet).

- [ ] **Step 5: Modify `run_role` in `serve.py`**

Add imports at the top of `serve.py`:

```python
from zippergen.syntax import EmptyStmt, WhileStmt, WhileRecvStmt
from zippergen.store import DurableChannel, load_snapshot, write_snapshot
from zippergen.locator import loop_node_paths, resolve_path
```

(Keep the existing `from zippergen.syntax import EmptyStmt` — merge it into the line above; keep the existing `from zippergen.store import DurableChannel` — merge likewise.)

Add helpers above `run_role`:

```python
def _floor_coherent(conn, role: str, floor: dict) -> bool:
    """A floor is coherent only if it does not point past the committed log."""
    try:
        out = floor["out"]; cursors = floor["cursors"]
    except (KeyError, TypeError):
        return False
    max_out = conn.execute(
        "SELECT MAX(rowid) FROM events WHERE sender=? AND kind IN ('msg','ctrl')",
        (role,),
    ).fetchone()[0] or 0
    if out > max_out:
        return False
    durable = {ck: c for ck, c in conn.execute(
        "SELECT chan_key, consumed FROM cursors WHERE role=?", (role,)).fetchall()}
    return all(v <= durable.get(k, 0) for k, v in cursors.items())


def _try_resume(conn, role: str, local_stmt, env: dict):
    """Return (env, residual, since) from a valid snapshot, else (env, local_stmt, None)."""
    snap = load_snapshot(conn, role)
    if snap is None:
        return env, local_stmt, None
    node = resolve_path(local_stmt, snap["locator"])
    if isinstance(node, (WhileStmt, WhileRecvStmt)) and _floor_coherent(conn, role, snap["floor"]):
        return dict(snap["env"]), node, snap["floor"]
    return env, local_stmt, None   # stale/invalid -> full replay from seed


def _maybe_snapshot(conn, role: str, env: dict, locator: list, ch) -> None:
    try:
        write_snapshot(conn, role, env, locator, ch.position())
    except (TypeError, ValueError):
        pass   # best-effort: env not JSON-serializable this iteration
```

Replace the body of `run_role` (keep the backend defaulting at the top unchanged) from `ch = DurableChannel(...)` onward with:

```python
    loop_paths = loop_node_paths(local_stmt)
    env, residual, since = _try_resume(conn, role, local_stmt, env)
    ch = DurableChannel(conn, role, since=since)

    # ---- replay: reconstruct (env, residual) from the tail (or full history) --
    while ch.replaying() and not isinstance(residual, EmptyStmt):
        residual, progressed = _step(
            residual, env, ch, ns, llm_backend, human_backend, None, None, {}, None)
        if not progressed:
            break

    # ---- live: one transaction per step; snapshot at loop boundaries ----------
    while not isinstance(residual, EmptyStmt):
        conn.execute("BEGIN IMMEDIATE")
        new_residual, progressed = _step(
            residual, env, ch, ns, llm_backend, human_backend, None, trace, {}, None)
        if progressed:
            ch.commit_txn()
            residual = new_residual
            if id(residual) in loop_paths:
                _maybe_snapshot(conn, role, env, loop_paths[id(residual)], ch)
        else:
            ch.rollback_txn()
            time.sleep(0.02)
    return env
```

- [ ] **Step 6: Run the snapshot tests**

Run: `python -m pytest tests/test_snapshots.py -v`
Expected: PASS (fixture sanity + 2 integration tests).

- [ ] **Step 7: Full suite (no-snapshot path unchanged)**

Run: `python -m pytest tests/ -q`
Expected: all pass — the deployable-runtime tests (which never write a snapshot and resume with `since=None`) confirm the additive change didn't regress anything.

- [ ] **Step 8: Checkpoint — do NOT commit**

Do not run `git commit`/`push`. Leave changes in the working tree. Files touched: `src/zippergen/serve.py`, `tests/loop_fixture.py`, `tests/test_snapshots.py`.

---

## Task 5: Crash-after-snapshot + stale-fallback validation

Proves the two properties that make snapshots safe: a crash *past* a snapshot resumes correctly from the tail, and a corrupt/stale snapshot degrades to full replay.

**Files:**
- Test: `tests/test_snapshots.py`

**Interfaces:**
- Consumes: `run_role`, `open_store`, `write_snapshot`, `project`, the loop fixture.

- [ ] **Step 1: Write the crash-after-snapshot and stale-fallback tests**

```python
# tests/test_snapshots.py  (append)
from zippergen.syntax import EmptyStmt, WhileStmt, WhileRecvStmt
from zippergen.runtime import _step, mock_llm
from zippergen.store import DurableChannel, load_snapshot, write_snapshot
from zippergen.locator import loop_node_paths
from zippergen.human_backends import make_cli_human_backend

class _Crash(Exception):
    pass

def _run_role_crash_after_k_snapshots(path, role, local, seed, ns, k):
    """Mirror run_role, but raise _Crash right after writing the k-th snapshot."""
    from zippergen.serve import _try_resume, _maybe_snapshot
    conn = open_store(path)
    loop_paths = loop_node_paths(local)
    env, residual, since = _try_resume(conn, role, local, dict(seed))
    ch = DurableChannel(conn, role, since=since)
    hb = make_cli_human_backend()
    while ch.replaying() and not isinstance(residual, EmptyStmt):
        residual, prog = _step(residual, env, ch, ns, mock_llm, hb, None, None, {}, None)
        if not prog:
            break
    snaps = 0
    while not isinstance(residual, EmptyStmt):
        conn.execute("BEGIN IMMEDIATE")
        new_r, prog = _step(residual, env, ch, ns, mock_llm, hb, None, None, {}, None)
        if prog:
            ch.commit_txn(); residual = new_r
            if id(residual) in loop_paths:
                _maybe_snapshot(conn, role, env, loop_paths[id(residual)], ch)
                snaps += 1
                if snaps == k:
                    raise _Crash()
        else:
            ch.rollback_txn()
    return env

def test_crash_after_snapshot_resumes_from_tail(tmp_path):
    path = str(tmp_path / "s.sqlite")
    la = project(counter_loop, A); lb = project(counter_loop, B)
    envs = {}
    def run_b():
        envs["B"] = run_role(open_store(path), "B", lb, {}, counter_loop.ns)
    tb = threading.Thread(target=run_b); tb.start()
    try:
        _run_role_crash_after_k_snapshots(path, "A", la, {"n": 0, "limit": 3}, counter_loop.ns, k=1)
    except _Crash:
        pass
    # Supervisor restarts A: resumes from the snapshot + tail, finishes the loop.
    env_a = run_role(open_store(path), "A", la, {"n": 0, "limit": 3}, counter_loop.ns)
    tb.join(timeout=15)
    assert env_a["n"] == 3
    # A's sends are not duplicated by the snapshot-resume.
    conn = open_store(path)
    a_sends = conn.execute(
        "SELECT COUNT(*) FROM events WHERE sender='A' AND kind='msg'").fetchone()[0]
    assert a_sends == 3

def test_stale_snapshot_falls_back_to_full_replay(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _run_both(path, {"n": 0, "limit": 3})   # completes; writes a real snapshot
    # Corrupt the snapshot with an unresolvable locator; run_role must discard it
    # and replay from seed, still reaching the correct result.
    conn = open_store(path)
    write_snapshot(conn, "A", {"n": 999}, [7, 7, 7], {"out": 0, "cursors": {}})
    la = project(counter_loop, A)
    env_a = run_role(open_store(path), "A", la, {"n": 0, "limit": 3}, counter_loop.ns)
    assert env_a["n"] == 3   # seed replay wins, not the bogus env n=999
```

- [ ] **Step 2: Run to verify (they should pass against Task 4's implementation)**

Run: `python -m pytest tests/test_snapshots.py -k "crash or stale" -v`
Expected: PASS. If `test_crash_after_snapshot_resumes_from_tail` fails on `a_sends != 3`, the tail-replay reserve path (Task 3 `since` filtering) is dropping or duplicating a send — investigate there, do NOT weaken the assertion.

- [ ] **Step 3: Stress the crash test (it uses threads + polling)**

Run: `for i in $(seq 15); do python -m pytest tests/test_snapshots.py -k crash -q || echo "FAIL $i"; done`
Expected: 15/15 pass, each finishing in a few seconds (no hang).

- [ ] **Step 4: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Checkpoint — do NOT commit**

Do not run `git commit`/`push`. Leave changes in the working tree. Files touched: `tests/test_snapshots.py`.

---

## Self-Review Notes

- **Spec coverage:** §2 schema/floor → Task 1 (`snapshots` table) + Task 3 (`floor` shape via `position`); §3 locator → Task 2; §4 write path → Task 4 (`_maybe_snapshot` at `id(residual) in loop_paths`); §5 restart/validate → Task 4 (`_try_resume`/`_floor_coherent`); §6 `DurableChannel` `position`+`since` → Task 3; §7 testing → Tasks 4–5 (appears, matches-full-run, crash-after-snapshot, stale-fallback, no-snapshot-unchanged via full suite). §8 out-of-scope items are not implemented, by design.
- **Type consistency:** `floor` is `{"out": int, "cursors": {chan_key: int}}` everywhere (`write_snapshot`, `position`, `since`, `_floor_coherent`); `locator` is `list[int]` in `write_snapshot`/`loop_node_paths`/`resolve_path`; `_try_resume` returns `(env, residual, since)` consumed directly by `run_role`.
- **Fixture risk:** Task 4 Step 2 gates on the fixture running under `run()` first, so DSL-syntax drift is caught before it can mask a snapshot bug — the same technique that de-risked the prior plan's Task 0.
