# Durable-Deploy Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `zippergen serve` correct for LLM- and human-in-the-loop coordination (incl. non-terminating approval loops and parallel regions) by journaling non-deterministic results, keeping blocking external calls off the SQLite write lock, and asserting replay consistency loudly.

**Architecture:** Extend the event-sourced per-role runtime. Add a role-local **journal** (`kind='act'`/`'decision'` rows in the existing `events` table) consumed FIFO on replay via a log cursor. `_step` gains an opt-in durable mode (a `journal` context); external acts surface as `PendingExternal` so `serve` can run the backend outside any transaction, then journal+commit, then apply. Parallel becomes a true single durable step. In-process `run()` is untouched.

**Tech Stack:** Python 3.11+ stdlib only (`sqlite3`, `json`, `hashlib`, `dataclasses`).

**Spec:** `docs/superpowers/specs/2026-07-04-durable-deploy-hardening-design.md`. One refinement vs. the spec's §4: the journal reader is implemented as a **log cursor** (`_journal_consumed` high-water + `rowid > floor` query), not a pre-loaded deque. This realizes the same contract (FIFO consumption, explicit consumed-floor) more simply and makes the external-act two-pass fall out for free — pass 2's `consume_journal` reads the row pass 1's `record_act` just committed.

## Global Constraints

- **Stdlib only.** No new dependencies.
- **In-process `run()` / `_exec` behavior is byte-for-byte unchanged.** Durable semantics activate only when a non-`None` `journal` is passed to `_step`. `_exec` and `run()` always pass `journal=None`.
- **Deterministic replay contract.** Every source of non-determinism — inbound messages, reserved sends, external act outputs, owner decisions — is reserved from the committed log. `@pure` is deterministic by contract and recomputed (never journaled).
- **At-least-once external effects (v1).** A crash between an external call returning and its journal row committing re-executes the act. Documented, not fixed.
- **Cache-never-a-dependency.** A stale/absent/incoherent snapshot or floor always falls back to full replay from the seed and never changes the result. A snapshot floor lacking the `"journal"` key is incoherent (forces full replay).
- **Journaled statements:** `ActStmt` with `LLMAction`/`HumanAction`/`PlannerAction`, and owner `IfStmt`/`WhileStmt` decisions. **Not journaled:** `PureAction`, `SelfAssignStmt`, receiver branches (`IfRecvStmt`/`WhileRecvStmt`).
- **`status` field:** every journal row carries `"status":"done"`; v1 asserts it but implements no pending rows.
- **No AI attribution** in commits/files. One-line commit messages. Commit only when told (this plan's per-task commits are pre-authorized by the user choosing to execute it).
- **Full suite green on 3.11/3.12/3.13** at the end of every task.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/zippergen/store.py` | Event store + `DurableChannel` | Add `ReplayMismatch`; send-reserve assertion; journal cursor (`consume_journal`, `record_act`, `record_decision`, `_journal_consumed`, `position()` floor) |
| `src/zippergen/locator.py` | Child-index paths | Extend `_children` for `ParallelLocalStmt`; add `action_node_paths` |
| `src/zippergen/runtime.py` | `_step` interpreter | Add `PendingExternal`; `_step(journal=None)`; external-act & owner-decision durable paths; parallel single-step; `external_out_map` helper |
| `src/zippergen/serve.py` | Durable driver / CLI | `JournalContext`; run_role two-pass external-act handling & wiring; `_floor_coherent` journal key; default seeding |
| `tests/…` | Tests | New unit + integration tests per task |

---

## Task 1: `ReplayMismatch` + send-reserve assertion

**Files:**
- Modify: `src/zippergen/store.py` (add exception near top; `DurableChannel.put` reserve branch ~`store.py:184`)
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `class ReplayMismatch(Exception)`; `DurableChannel.put(sender, receiver, channel, values, …)` raises `ReplayMismatch` when the reserved recorded send does not match the outgoing `(receiver, channel, values)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py  (append)
from zippergen.store import ReplayMismatch

def test_reserved_send_payload_mismatch_raises(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    # Restart A: the recorded send (1,) is reserved on replay.
    a2 = DurableChannel(conn, "A")
    assert a2.replaying() is True
    with pytest.raises(ReplayMismatch):
        a2.put("A", "B", "main", (2,))          # diverged payload

def test_reserved_send_match_ok(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.put("A", "B", "main", (1,)); a.commit_txn()
    a2 = DurableChannel(conn, "A")
    assert a2.put("A", "B", "main", (1,))        # matches -> reserved, no raise
    assert a2.replaying() is False
```

Add `import pytest` at the top of `tests/test_store.py` if not present.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py::test_reserved_send_payload_mismatch_raises -v`
Expected: FAIL (`ImportError: cannot import name 'ReplayMismatch'`).

- [ ] **Step 3: Implement**

In `src/zippergen/store.py`, add after the imports (before `SCHEMA`):

```python
class ReplayMismatch(Exception):
    """A step re-executing during replay diverged from the committed log
    (different payload/locator/kind). Raised loudly rather than corrupting state."""
```

Replace the reserve branch of `DurableChannel.put` (currently `store.py:184-186`):

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS (both new tests + existing store tests).

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/store.py tests/test_store.py
git commit -m "Add ReplayMismatch and assert reserved-send target/payload on durable replay"
```

---

## Task 2: `action_node_paths` locator index

**Files:**
- Modify: `src/zippergen/locator.py`
- Test: `tests/test_locator.py`

**Interfaces:**
- Produces: `action_node_paths(root) -> dict[int, list[int]]` mapping `id(node) -> child-index path` for every `ActStmt`, `IfStmt`, `WhileStmt` in a projected local program. `_children` additionally descends into `ParallelLocalStmt.branches`.
- Consumes: existing `resolve_path(root, path)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_locator.py  (append)
from zippergen.locator import action_node_paths, resolve_path
from zippergen.syntax import (
    ActStmt, SeqStmt, WhileStmt, ParallelLocalStmt, Lifeline, Var, VarExpr,
)
from zippergen.actions import pure

A = Lifeline("A")

@pure
def f(x: int) -> int:
    return x

def _act():
    x = Var("x", int)
    return ActStmt(A, f, (VarExpr(x),), (x,))

def test_action_paths_resolve_back_to_same_nodes():
    a1, a2 = _act(), _act()
    root = SeqStmt(a1, SeqStmt(a2, ParallelLocalStmt((_act(),), (0,))))
    paths = action_node_paths(root)
    # every recorded id resolves back to the identical object via its path
    for node_id, path in paths.items():
        assert id(resolve_path(root, path)) == node_id
    # all three acts are indexed (incl. the one inside the parallel branch)
    assert sum(isinstance(resolve_path(root, p), ActStmt) for p in paths.values()) == 3

def test_action_paths_index_owner_loop():
    body = _act()
    root = WhileStmt(condition=lambda _e: True, owner=A, body=body, exit_body=_act())
    paths = action_node_paths(root)
    assert id(root) in paths                    # owner WhileStmt indexed (decision)
    assert id(body) in paths                    # act inside the loop body indexed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_locator.py::test_action_paths_resolve_back_to_same_nodes -v`
Expected: FAIL (`ImportError: cannot import name 'action_node_paths'`).

- [ ] **Step 3: Implement**

In `src/zippergen/locator.py`, import `ActStmt`, `WhileStmt`, `ParallelLocalStmt` and extend:

```python
from zippergen.syntax import (
    SeqStmt, IfStmt, IfRecvStmt, WhileStmt, WhileRecvStmt,
    ActStmt, ParallelLocalStmt,
)
```

Add a `ParallelLocalStmt` case to `_children` (before the `case _`):

```python
        case ParallelLocalStmt(branches=branches):
            return list(branches)
```

Add:

```python
def action_node_paths(root) -> dict[int, list[int]]:
    """Map id(node) -> child-index path for every act and owner if/while node.

    These are the statements whose non-deterministic result is journaled; the
    path re-finds the node in a freshly-projected program (same trick as
    loop_node_paths). Leaf/owner identity survives _step, so id() is a stable key
    for the journal locator."""
    out: dict[int, list[int]] = {}

    def walk(node, path: list[int]) -> None:
        if isinstance(node, (ActStmt, IfStmt, WhileStmt)):
            out[id(node)] = list(path)
        for i, child in enumerate(_children(node)):
            walk(child, path + [i])

    walk(root, [])
    return out
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_locator.py -v`
Expected: PASS (new + existing locator tests).

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/locator.py tests/test_locator.py
git commit -m "Add action_node_paths locator index for journal locators"
```

---

## Task 3: Journal cursor on `DurableChannel`

**Files:**
- Modify: `src/zippergen/store.py` (`DurableChannel.__init__`, add methods, `position()`)
- Test: `tests/test_store.py`

**Interfaces:**
- Produces on `DurableChannel`:
  - `record_act(payload: dict) -> int` — INSERT `kind='act'`, does **not** advance the journal cursor.
  - `record_decision(payload: dict) -> int` — INSERT `kind='decision'`, **advances** the cursor to the new rowid.
  - `consume_journal(expected_kind: str, locator: list, input_hash: str | None = None) -> dict | None` — next journal row with `rowid > _journal_consumed`; asserts `kind`/`locator`/`input_hash` (raises `ReplayMismatch`), advances the cursor, returns the decoded payload; `None` if no such row (⇒ live path).
  - `position()` now returns `{"out":…, "cursors":…, "journal": self._journal_consumed}`.
- Consumes: `ReplayMismatch` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py  (append)
def test_journal_record_and_consume_fifo(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN")
    a.record_act({"status": "done", "locator": [0], "action": "llm", "input_hash": "h", "outputs": {"y": 1}})
    a.record_decision({"status": "done", "locator": [1], "value": True})
    a.commit_txn()
    # A fresh channel replays the journal in FIFO order via the cursor.
    a2 = DurableChannel(conn, "A")
    p0 = a2.consume_journal("act", [0], "h")
    assert p0["outputs"] == {"y": 1}
    p1 = a2.consume_journal("decision", [1])
    assert p1["value"] is True
    assert a2.consume_journal("act", [2]) is None          # nothing left -> live path
    assert a2.position()["journal"] == a2._journal_consumed

def test_journal_locator_mismatch_raises(tmp_path):
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); a.record_act({"status": "done", "locator": [0], "action": "llm", "outputs": {}}); a.commit_txn()
    a2 = DurableChannel(conn, "A")
    with pytest.raises(ReplayMismatch):
        a2.consume_journal("act", [9])                     # wrong locator

def test_record_act_does_not_advance_cursor(tmp_path):
    # act rows are consumed by a separate pass; recording must leave the cursor
    # below the new row so the next consume_journal picks it up.
    conn = open_store(str(tmp_path / "s.sqlite"))
    a = DurableChannel(conn, "A")
    conn.execute("BEGIN"); rid = a.record_act({"status": "done", "locator": [0], "action": "llm", "outputs": {"y": 2}}); a.commit_txn()
    assert a._journal_consumed < rid
    got = a.consume_journal("act", [0])
    assert got["outputs"] == {"y": 2} and a._journal_consumed == rid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py::test_journal_record_and_consume_fifo -v`
Expected: FAIL (`AttributeError: 'DurableChannel' object has no attribute 'record_act'`).

- [ ] **Step 3: Implement**

In `DurableChannel.__init__` (after `self._replay_inbox = {}` line, `store.py:131`), add:

```python
        self._journal_consumed: int = (since or {}).get("journal", 0)
```

Add methods (near the interpreter-facing surface, after `get`):

```python
    # ---- role-local journal (external act outputs + owner decisions) --------
    def record_act(self, payload: dict) -> int:
        """INSERT an act-journal row. Does NOT advance the journal cursor — the
        result is applied by a separate consume pass (apply-after-commit)."""
        cur = self.conn.execute(
            "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
            "VALUES(?,?,?,?,?,?)",
            (self.role, None, None, "act", json.dumps(payload), None),
        )
        return int(cur.lastrowid)

    def record_decision(self, payload: dict) -> int:
        """INSERT a decision-journal row and advance the cursor past it (the
        value is recorded and consumed in one step; no separate consume pass)."""
        cur = self.conn.execute(
            "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
            "VALUES(?,?,?,?,?,?)",
            (self.role, None, None, "decision", json.dumps(payload), None),
        )
        self._journal_consumed = int(cur.lastrowid)
        return int(cur.lastrowid)

    def consume_journal(self, expected_kind: str, locator: list,
                        input_hash: str | None = None) -> dict | None:
        """Return the next committed journal row for this role (FIFO by rowid),
        asserting it matches the statement re-executing here; None if none left."""
        row = self.conn.execute(
            "SELECT rowid, kind, payload FROM events "
            "WHERE sender=? AND kind IN ('act','decision') AND rowid>? "
            "ORDER BY rowid LIMIT 1",
            (self.role, self._journal_consumed),
        ).fetchone()
        if row is None:
            return None
        rowid, kind, payload = row
        data = json.loads(payload)
        if kind != expected_kind or data.get("locator") != locator:
            raise ReplayMismatch(
                f"journal diverged at rowid {rowid}: recorded {kind}/{data.get('locator')}, "
                f"executing {expected_kind}/{locator}")
        if input_hash is not None and data.get("input_hash") not in (None, input_hash):
            raise ReplayMismatch(
                f"journal input_hash diverged at rowid {rowid}: "
                f"recorded {data.get('input_hash')!r}, recomputed {input_hash!r}")
        self._journal_consumed = rowid
        return data
```

Update `position()` (`store.py:169-178`) to add the journal key:

```python
        return {
            "out": row[0] or 0,
            "cursors": {chan_key(*key): rowid for key, rowid in self._consumed.items()},
            "journal": self._journal_consumed,
        }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/store.py tests/test_store.py
git commit -m "Add journal cursor (record_act/record_decision/consume_journal) to DurableChannel"
```

---

## Task 4: Journal floor coherence + snapshot round-trip

**Files:**
- Modify: `src/zippergen/serve.py` (`_floor_coherent`, `serve.py:14-28`)
- Test: `tests/test_snapshots.py`

**Interfaces:**
- Consumes: `position()["journal"]` (Task 3), `write_snapshot`/`load_snapshot`.
- Produces: `_floor_coherent` requires a `"journal"` key and validates it against the committed log; a floor without `"journal"` is incoherent (forces full replay).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_snapshots.py  (append)
from zippergen.serve import _floor_coherent
from zippergen.store import open_store as _open

def test_floor_without_journal_key_is_incoherent(tmp_path):
    conn = _open(str(tmp_path / "s.sqlite"))
    assert _floor_coherent(conn, "A", {"out": 0, "cursors": {}}) is False   # missing journal

def test_floor_journal_past_log_is_incoherent(tmp_path):
    conn = _open(str(tmp_path / "s.sqlite"))
    assert _floor_coherent(conn, "A", {"out": 0, "cursors": {}, "journal": 5}) is False

def test_floor_with_valid_journal_is_coherent(tmp_path):
    conn = _open(str(tmp_path / "s.sqlite"))
    assert _floor_coherent(conn, "A", {"out": 0, "cursors": {}, "journal": 0}) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_snapshots.py::test_floor_without_journal_key_is_incoherent -v`
Expected: FAIL (returns `True`, expected `False`).

- [ ] **Step 3: Implement**

Replace `_floor_coherent` in `src/zippergen/serve.py`:

```python
def _floor_coherent(conn, role: str, floor: dict) -> bool:
    """A floor is coherent only if it carries all three keys and none points past
    the committed log. A floor lacking "journal" predates journaling -> incoherent
    (forces full replay from seed)."""
    try:
        out = floor["out"]; cursors = floor["cursors"]; journal = floor["journal"]
    except (KeyError, TypeError):
        return False
    max_out = conn.execute(
        "SELECT MAX(rowid) FROM events WHERE sender=? AND kind IN ('msg','ctrl')",
        (role,),
    ).fetchone()[0] or 0
    if out > max_out:
        return False
    max_journal = conn.execute(
        "SELECT MAX(rowid) FROM events WHERE sender=? AND kind IN ('act','decision')",
        (role,),
    ).fetchone()[0] or 0
    if journal > max_journal:
        return False
    durable = {ck: c for ck, c in conn.execute(
        "SELECT chan_key, consumed FROM cursors WHERE role=?", (role,)).fetchall()}
    return all(v <= durable.get(k, 0) for k, v in cursors.items())
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_snapshots.py -v`
Expected: PASS (new + existing; existing snapshot tests still pass because `position()` now always includes `"journal"`).

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/serve.py tests/test_snapshots.py
git commit -m "Validate journal floor in snapshot coherence check"
```

---

## Task 5: `PendingExternal` + `_step` durable external-act path

**Files:**
- Modify: `src/zippergen/runtime.py` (`_step` signature `runtime.py:359-370`; `ActStmt` handling `runtime.py:384-386`; add `PendingExternal` + `external_out_map`)
- Test: `tests/test_runtime.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class PendingExternal: node: ActStmt; inputs: dict`
  - `_step(stmt, env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop, journal=None) -> tuple[LocalStmt | PendingExternal, bool]`
  - `external_out_map(action, named_inputs, outs, llm_backend, human_backend) -> dict` — computes an external action's output env-delta (the LLM/Human/Planner branch of `_exec`'s act logic).
  - `JournalContext` is any object with `.channel` (a `DurableChannel`) and `.act_paths` (`dict[int, list[int]]`); passed as `journal`. Defined in Task 7 (`serve.py`); `_step` only duck-types it.
- Consumes: `action_node_paths` locators (Task 2) via `journal.act_paths`; `consume_journal` (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime.py  (append)
import types
from zippergen.runtime import _step, PendingExternal, mock_llm
from zippergen.syntax import ActStmt, Lifeline, Var, VarExpr
from zippergen.actions import pure

A = Lifeline("A")
_x = Var("x", int); _y = Var("y", int)

@pure
def inc(x: int) -> int:
    return x + 1

def _llm_act():
    # Construct the LLMAction IR node directly (no @llm decorator needed for a unit test).
    from zippergen.syntax import LLMAction
    action = LLMAction("ask", (("x", int),), (("y", int),), "s", "{x}", "text")
    return ActStmt(A, action, (VarExpr(_x),), (_y,))

class _FakeJournal:
    def __init__(self, channel, act_paths):
        self.channel = channel; self.act_paths = act_paths

class _FakeChannel:
    def __init__(self, result=None):
        self._result = result
    def consume_journal(self, kind, locator, input_hash=None):
        return self._result

def test_step_external_act_live_returns_pending():
    act = _llm_act()
    j = _FakeJournal(_FakeChannel(result=None), {id(act): [0]})
    env = {"x": 5}
    out, progressed = _step(act, env, None, {}, mock_llm, None, None, None, {}, None, journal=j)
    assert isinstance(out, PendingExternal) and out.node is act
    assert out.inputs == {"x": 5} and progressed is False
    assert env == {"x": 5}                       # nothing mutated

def test_step_external_act_replay_consumes_no_backend():
    act = _llm_act()
    result = {"status": "done", "locator": [0], "outputs": {"y": 99}}
    j = _FakeJournal(_FakeChannel(result=result), {id(act): [0]})
    env = {"x": 5}
    def boom(*a, **k):  # backend must NOT be called on replay
        raise AssertionError("backend called during replay")
    out, progressed = _step(act, env, None, {}, boom, None, None, None, {}, None, journal=j)
    assert progressed is True and env["y"] == 99

def test_step_pure_act_inline_even_in_durable_mode():
    act = ActStmt(A, inc, (VarExpr(_x),), (_y,))
    j = _FakeJournal(_FakeChannel(result=None), {id(act): [0]})
    from zippergen.channels import InProcessChannel
    env = {"x": 5}
    out, progressed = _step(act, env, InProcessChannel(), {}, mock_llm, None, None, None, {}, None, journal=j)
    assert progressed is True and env["y"] == 6  # pure recomputed, not journaled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime.py::test_step_external_act_live_returns_pending -v`
Expected: FAIL (`ImportError: cannot import name 'PendingExternal'`).

- [ ] **Step 3: Implement**

In `src/zippergen/runtime.py`, add near the top (after imports):

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PendingExternal:
    """Durable-mode signal: a live external act needs its backend run OUTSIDE the
    write transaction. Carries the node (for its action/outputs) and evaluated
    inputs. Returned in the residual slot with progressed=False; never produced
    when journal is None."""
    node: object
    inputs: dict
```

Add the helper (near `_exec`'s act logic):

```python
def external_out_map(action, named_inputs, outs, llm_backend, human_backend) -> dict:
    """Output env-delta for an external (LLM/Human/Planner) action — the same
    computation _exec performs for these branches, factored for the durable driver."""
    if isinstance(action, PlannerAction):
        return {outs[0].name: _exec_planner(action, named_inputs, llm_backend, None, _next_act_seq())}
    if isinstance(action, HumanAction):
        if not action.visible:
            default = True if action.output_type is bool else ""
            return {outs[0].name: default}
        named_outputs = human_backend(action, named_inputs)
        return {outs[0].name: named_outputs[action.output]}
    named_outputs = llm_backend(action, named_inputs)   # LLMAction
    return {var.name: named_outputs.get(aname) for (aname, _), var in zip(action.outputs, outs)}
```

Change `_step`'s signature (`runtime.py:359-370`) to add `journal=None` as the final parameter, and update the docstring return type to `tuple[LocalStmt | PendingExternal, bool]`.

Split the combined act case (`runtime.py:384-386`). Replace:

```python
        case SendStmt() | SelfAssignStmt() | ActStmt():
            _exec(stmt, env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
            return EmptyStmt(), True
```

with:

```python
        case SendStmt() | SelfAssignStmt():
            _exec(stmt, env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
            return EmptyStmt(), True

        case ActStmt(lifeline=_, action=action, inputs=ins, outputs=outs):
            if journal is None or isinstance(action, PureAction):
                # in-process, or a pure (deterministic, non-journaled) act
                _exec(stmt, env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
                return EmptyStmt(), True
            locator = journal.act_paths[id(stmt)]
            in_vals = tuple(_eval(x, env) for x in ins)
            named_inputs = {name: val for (name, _), val in zip(action.inputs, in_vals)}
            recorded = journal.channel.consume_journal("act", locator, _input_hash(named_inputs))
            if recorded is not None:                 # replay: apply, no backend call
                env.update(recorded["outputs"])
                return EmptyStmt(), True
            return PendingExternal(stmt, named_inputs), False   # live: serve resolves it
```

Add the input-hash helper (near `external_out_map`):

```python
import hashlib

def _input_hash(named_inputs: dict) -> str | None:
    try:
        return hashlib.sha1(json.dumps(named_inputs, sort_keys=True).encode()).hexdigest()[:16]
    except TypeError:
        return None   # non-serializable inputs -> skip hash (locator+kind still assert)
```

Thread `journal=journal` through the recursive `_step` calls **inside `_step`** — the `SeqStmt` case (`runtime.py:436-439`) and (Task 8) the parallel case. For `SeqStmt`:

```python
            new_first, progressed = _step(
                first, env, ch, ns, llm_backend, human_backend, monitor, trace,
                formula_conditions, stop, journal=journal,
            )
            if isinstance(new_first, PendingExternal):
                return new_first, False
            if not progressed:
                return stmt, False
            return cast(LocalStmt, seq(new_first, second)), True
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: PASS (new tests + existing runtime tests; in-process path unchanged since `journal` defaults to `None`).

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/runtime.py tests/test_runtime.py
git commit -m "Add PendingExternal and durable external-act handling in _step"
```

---

## Task 6: `_step` owner-decision journaling

**Files:**
- Modify: `src/zippergen/runtime.py` (`IfStmt` case `runtime.py:444-478`; `WhileStmt` case `runtime.py:500-536`)
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `journal.act_paths`, `record_decision`/`consume_journal` (Task 3), `PendingExternal`-free (decisions never block).
- Produces: in durable mode, owner `IfStmt`/`WhileStmt` steps journal (live) or consume (replay) the decision `value` instead of trusting a re-evaluated guard.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime.py  (append)
from zippergen.syntax import IfStmt, SkipStmt

class _RecordingChannel:
    def __init__(self, result=None):
        self._result = result; self.decided = None
    def consume_journal(self, kind, locator, input_hash=None):
        return self._result
    def record_decision(self, payload):
        self.decided = payload; return 1

def test_owner_decision_live_journals_and_uses_value():
    t, f = SkipStmt(A), SkipStmt(Lifeline("B"))
    node = IfStmt(condition=lambda _e: True, owner=A, branch_true=t, branch_false=f)
    ch = _RecordingChannel(result=None)
    j = _FakeJournal(ch, {id(node): [7]})
    out, progressed = _step(node, {}, None, {}, mock_llm, None, None, None, {}, None, journal=j)
    assert progressed is True and out is t
    assert ch.decided == {"status": "done", "locator": [7], "value": True}

def test_owner_decision_replay_uses_recorded_value_no_guard():
    t, f = SkipStmt(A), SkipStmt(Lifeline("B"))
    def boom(_e):
        raise AssertionError("guard evaluated during replay")
    node = IfStmt(condition=boom, owner=A, branch_true=t, branch_false=f)
    ch = _RecordingChannel(result={"status": "done", "locator": [7], "value": False})
    j = _FakeJournal(ch, {id(node): [7]})
    out, progressed = _step(node, {}, None, {}, mock_llm, None, None, None, {}, None, journal=j)
    assert progressed is True and out is f      # recorded False, guard never called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime.py::test_owner_decision_live_journals_and_uses_value -v`
Expected: FAIL (guard re-evaluated / decision not recorded).

- [ ] **Step 3: Implement**

Add a durable short-circuit at the **top** of the `IfStmt` case (`runtime.py:444`, before the `cached_formula` logic):

```python
        case IfStmt(condition=c, owner=B, branch_true=t, branch_false=f):
            if journal is not None:
                loc = journal.act_paths[id(stmt)]
                rec = journal.channel.consume_journal("decision", loc)
                if rec is not None:
                    return cast(LocalStmt, t if rec["value"] else f), True
                flag = bool(c(_CondEnv(env, ns)))
                journal.channel.record_decision({"status": "done", "locator": loc, "value": flag})
                return cast(LocalStmt, t if flag else f), True
            # --- in-process path unchanged below ---
            cached_formula = formula_conditions.get(id(c))
            ...
```

Add the identical short-circuit at the top of the `WhileStmt` case (`runtime.py:500`):

```python
        case WhileStmt(condition=c, owner=B, body=body, exit_body=exit_b):
            if journal is not None:
                loc = journal.act_paths[id(stmt)]
                rec = journal.channel.consume_journal("decision", loc)
                if rec is not None:
                    flag = rec["value"]
                else:
                    flag = bool(c(_CondEnv(env, ns)))
                    journal.channel.record_decision({"status": "done", "locator": loc, "value": flag})
                if not flag:
                    return cast(LocalStmt, exit_b), True
                return cast(LocalStmt, seq(body, stmt)), True
            # --- in-process path unchanged below ---
            cached_formula = formula_conditions.get(id(c))
            ...
```

Note: the durable `WhileStmt` reconstructs the loop residual as `seq(body, stmt)` so the next iteration re-enters the same `WhileStmt` node (`stmt`), preserving `id(stmt)` for the loop-boundary snapshot and the next decision locator. Confirm the in-process `WhileStmt` case (below the short-circuit) already returns an equivalent residual; if it differs, match its exact shape so behavior is identical — read `runtime.py:500-536` before editing and mirror its true/false residual construction, only swapping the guard source for the journaled value.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/runtime.py tests/test_runtime.py
git commit -m "Journal owner if/while decisions in _step durable mode"
```

---

## Task 7: `serve` durable driver — `JournalContext`, two-pass external acts, wiring

**Files:**
- Modify: `src/zippergen/serve.py` (`run_role`, `serve.py:52-94`; imports)
- Test: `tests/test_serve_journal.py` (new)

**Interfaces:**
- Produces: `@dataclass class JournalContext: channel; act_paths` in `serve.py`; `run_role` builds it and passes `journal=jctx` to every `_step`; handles `PendingExternal` via the two-pass flow.
- Consumes: `_step`/`PendingExternal`/`external_out_map`/`_input_hash` (Tasks 5–6), `action_node_paths` (Task 2), `record_act`/`consume_journal` (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serve_journal.py  (new)
import threading
from zippergen import Lifeline, Var, workflow
from zippergen.actions import llm as llm_deco, pure
from zippergen.projection import project
from zippergen.store import open_store
from zippergen.serve import run_role

A = Lifeline("A"); B = Lifeline("B")
n = Var("n", int, default=0); m = Var("m", int, default=0)
label = Var("label", int, default=0); got = Var("got", int, default=0)

CALLS = {"n": 0}
def counting_backend(action, inputs):
    CALLS["n"] += 1
    return {"label": CALLS["n"] * 10}          # non-deterministic across calls

# Match the exact @llm signature used in examples/diagnosis.py and tests/ when
# writing this fixture; `parse` and `outputs` keyword shapes come from actions.py.
classify = llm_deco(system="s", user="{m}", parse="json", outputs=[("label", int)])
@classify
def classify_fn(m: int) -> int: ...

@pure
def relay(m: int) -> int:
    return m

@workflow
def one_round(n: int @ A):
    A(n) >> B(m)
    B: label = classify_fn(m)
    B(label) >> A(got)
    return got @ A

def _run(path, role, local, seed):
    return run_role(open_store(path), role, local, dict(seed), one_round.ns,
                    llm_backend=counting_backend)

def test_external_act_memoized_across_restart(tmp_path):
    path = str(tmp_path / "s.sqlite")
    la, lb = project(one_round, A), project(one_round, B)
    envs = {}
    tb = threading.Thread(target=lambda: envs.__setitem__("B", _run(path, "B", lb, {})))
    tb.start()
    envs["A"] = _run(path, "A", la, {"n": 1})
    tb.join(timeout=15)
    first_calls = CALLS["n"]
    assert first_calls == 1 and envs["A"]["got"] == 10
    # Re-run B from the committed log: the classify LLM must NOT be called again.
    env_b2 = _run(path, "B", lb, {})
    assert CALLS["n"] == first_calls               # memoized, no re-invocation
    assert env_b2["label"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_serve_journal.py -v`
Expected: FAIL (backend re-invoked on re-run / `CALLS["n"]` increments, or the LLM act holds the write lock and the run hangs).

- [ ] **Step 3: Implement**

In `src/zippergen/serve.py`, add imports and the context type:

```python
from dataclasses import dataclass
from zippergen.runtime import _step, mock_llm, PendingExternal, external_out_map, _input_hash
from zippergen.locator import loop_node_paths, resolve_path, action_node_paths

@dataclass
class JournalContext:
    channel: object          # DurableChannel
    act_paths: dict          # id(node) -> child-index path
```

Rewrite `run_role` (`serve.py:52-94`):

```python
def run_role(conn, role: str, local_stmt, env: dict, ns: dict, *,
             llm_backend=None, human_backend=None, trace=None) -> dict:
    if llm_backend is None:
        llm_backend = mock_llm
    if human_backend is None:
        from zippergen.human_backends import make_cli_human_backend
        human_backend = make_cli_human_backend()

    loop_paths = loop_node_paths(local_stmt)
    env, residual, since = _try_resume(conn, role, local_stmt, env)
    ch = DurableChannel(conn, role, since=since)
    jctx = JournalContext(ch, action_node_paths(local_stmt))

    def step(r, tr):
        return _step(r, env, ch, ns, llm_backend, human_backend, None, tr, {}, None, journal=jctx)

    # ---- replay: reconstruct (env, residual) from the tail (or full history) --
    while ch.replaying() and not isinstance(residual, EmptyStmt):
        out, progressed = step(residual, None)
        if isinstance(out, PendingExternal):
            break            # unreached during replay: committed acts consume, not pend
        if not progressed:
            break
        residual = out

    # ---- live: one transaction per progressing step ------------------------
    while not isinstance(residual, EmptyStmt):
        conn.execute("BEGIN IMMEDIATE")
        out, progressed = step(residual, trace)
        if isinstance(out, PendingExternal):
            conn.execute("ROLLBACK")                 # release the write lock first
            outs = out.node.outputs
            out_map = external_out_map(out.node.action, out.inputs, outs,
                                       llm_backend, human_backend)   # OUTSIDE any txn
            loc = jctx.act_paths[id(out.node)]
            conn.execute("BEGIN IMMEDIATE")
            ch.record_act({"status": "done", "locator": loc,
                           "action": out.node.action.name,
                           "input_hash": _input_hash(out.inputs), "outputs": out_map})
            conn.execute("COMMIT")
            # pass 2 (no txn): consume the just-committed act row, apply env, advance
            residual, _ = step(residual, trace)
            if id(residual) in loop_paths:
                _maybe_snapshot(conn, role, env, loop_paths[id(residual)], ch)
            continue
        if progressed:
            ch.commit_txn()
            residual = out
            if id(residual) in loop_paths:
                _maybe_snapshot(conn, role, env, loop_paths[id(residual)], ch)
        else:
            ch.rollback_txn()
            time.sleep(0.02)
    return env
```

Remove the now-duplicate top-of-file imports of `_step`/`mock_llm`/`loop_node_paths`/`resolve_path` (they move into the block above; keep a single import site). Verify `EmptyStmt`, `WhileStmt`, `WhileRecvStmt`, `DurableChannel`, `load_snapshot`, `write_snapshot` imports remain.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_serve_journal.py tests/test_snapshots.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/serve.py tests/test_serve_journal.py
git commit -m "Wire journal into serve with two-pass external-act execution off the write lock"
```

---

## Task 8: Parallel as a true durable single step

**Files:**
- Modify: `src/zippergen/runtime.py` (`_step` `ParallelLocalStmt` case `runtime.py:560-562`)
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `_step(journal=…)`, `PendingExternal` (Task 5).
- Produces: `_step(ParallelLocalStmt, journal=<ctx>)` advances one enabled branch and returns the rebuilt region; `journal=None` still delegates to `_exec`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime.py  (append)
from zippergen.syntax import ParallelLocalStmt, SkipStmt as _Skip

def test_parallel_durable_single_step_advances_one_branch():
    b0 = SeqStmt(_Skip(A), _Skip(A))
    b1 = _Skip(A)
    region = ParallelLocalStmt((b0, b1), (0, 1))
    j = _FakeJournal(_FakeChannel(result=None), {})
    out, progressed = _step(region, {}, None, {}, mock_llm, None, None, None, {}, None, journal=j)
    assert progressed is True and isinstance(out, ParallelLocalStmt)
    # exactly one branch advanced; region not run to completion in one step
    assert out is not region

def test_parallel_durable_completes_and_inprocess_delegates(monkeypatch):
    region = ParallelLocalStmt((_Skip(A),), (0,))
    j = _FakeJournal(_FakeChannel(result=None), {})
    out, progressed = _step(region, {}, None, {}, mock_llm, None, None, None, {}, None, journal=j)
    from zippergen.syntax import EmptyStmt
    assert isinstance(out, EmptyStmt) and progressed is True
    # journal=None path must still call _exec (in-process behavior preserved)
    called = {"exec": False}
    import zippergen.runtime as rt
    real_exec = rt._exec
    def spy(stmt, *a, **k):
        if isinstance(stmt, ParallelLocalStmt):
            called["exec"] = True
        return real_exec(stmt, *a, **k)
    monkeypatch.setattr(rt, "_exec", spy)
    from zippergen.channels import InProcessChannel
    rt._step(ParallelLocalStmt((_Skip(A),), (0,)), {}, InProcessChannel(), {},
             mock_llm, None, None, None, {}, None, journal=None)
    assert called["exec"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime.py::test_parallel_durable_single_step_advances_one_branch -v`
Expected: FAIL (current `_step` delegates to `_exec` regardless of `journal`, returning `EmptyStmt`).

- [ ] **Step 3: Implement**

Replace the `ParallelLocalStmt` case in `_step` (`runtime.py:560-562`):

```python
        case ParallelLocalStmt(branches=branches, branch_indices=labels) if journal is None:
            _exec(stmt, env, ch, ns, llm_backend, human_backend, monitor, trace, formula_conditions, stop)
            return EmptyStmt(), True

        case ParallelLocalStmt(branches=branches, branch_indices=labels):
            residuals = list(branches)
            for i, branch in enumerate(residuals):
                if isinstance(branch, EmptyStmt):
                    continue
                new_branch, progressed = _step(
                    branch, env, ch, ns, llm_backend, human_backend, monitor, trace,
                    formula_conditions, stop, journal=journal)
                if isinstance(new_branch, PendingExternal):
                    return new_branch, False           # propagate up; serve resolves
                residuals[i] = new_branch
                if progressed:
                    if all(isinstance(b, EmptyStmt) for b in residuals):
                        return EmptyStmt(), True
                    return ParallelLocalStmt(tuple(residuals), labels), True
            if all(isinstance(b, EmptyStmt) for b in residuals):
                return EmptyStmt(), True
            return stmt, False
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/runtime.py tests/test_runtime.py
git commit -m "Make ParallelLocalStmt a true single durable step in _step"
```

---

## Task 9: `serve.main` default seeding

**Files:**
- Modify: `src/zippergen/serve.py` (`main`, `serve.py:170-174`)
- Test: `tests/test_serve_journal.py`

**Interfaces:**
- Consumes: `wf.ns`, `_parse_inputs`, `seed_env`.
- Produces: the seed persisted by `serve.main` includes `{Var: default}` overlaid with `--input` (parity with `runtime.py:1014`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serve_journal.py  (append)
def test_default_seed_inputs_overlay():
    from zippergen.serve import _seed_inputs
    merged = _seed_inputs(one_round, {"n": 3})
    assert merged["n"] == 3          # caller input wins
    assert merged["m"] == 0          # Var default carried through when not supplied
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_serve_journal.py::test_default_seed_inputs_overlay -v`
Expected: FAIL (`ImportError: cannot import name '_seed_inputs'`).

- [ ] **Step 3: Implement**

Add to `src/zippergen/serve.py`:

```python
def _seed_inputs(wf: Workflow, inputs: dict) -> dict:
    """Var defaults from the workflow namespace, overlaid by caller inputs —
    parity with the in-process run() seeding (runtime.py:1014)."""
    from zippergen.syntax import Var
    env = {k: v.default for k, v in wf.ns.items() if isinstance(v, Var)}
    env.update(inputs)
    return env
```

In `main` (`serve.py:172`), change:

```python
    env = seed_env(conn, args.role, wf, _seed_inputs(wf, _parse_inputs(args.input)))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_serve_journal.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/serve.py tests/test_serve_journal.py
git commit -m "Seed Var defaults in serve for parity with in-process run()"
```

---

## Task 10: Integration — at-least-once window, parallel two-process crash, full regression

**Files:**
- Create: `tests/test_deploy_integration.py`
- Create: `tests/fixtures/parallel_deploy.py` (a two-role workflow with a parallel region, pure actions only)
- Test: the two files above

**Interfaces:**
- Consumes: everything above; `run_role`, `serve.main` via subprocess, `open_store`.

- [ ] **Step 1: Write the failing/behavioral tests**

```python
# tests/fixtures/parallel_deploy.py  (new)
from zippergen import Lifeline, Var, workflow
from zippergen.actions import pure

A = Lifeline("A"); B = Lifeline("B")
x = Var("x", int, default=0); y = Var("y", int, default=0)
u = Var("u", int, default=0); v = Var("v", int, default=0)

@pure
def bump(n: int) -> int:
    return n + 1

@workflow
def par_flow(x: int @ A, y: int @ A):
    with parallel:
        with branch:
            A(x) >> B(u)
            B: u = bump(u)
            B(u) >> A(x)
        with branch:
            A(y) >> B(v)
            B: v = bump(v)
            B(v) >> A(y)
    return x @ A
```

```python
# tests/test_deploy_integration.py  (new)
import json, os, sqlite3, subprocess, sys, threading, time
import pytest
from zippergen import Lifeline, Var, workflow
from zippergen.actions import llm as llm_deco
from zippergen.projection import project
from zippergen.store import open_store
import zippergen.store as store_mod
from zippergen.serve import run_role

A = Lifeline("A")
n = Var("n", int, default=0); label = Var("label", int, default=0)

# Single-role workflow with one external (LLM) act — self-contained, no peer —
# so the crash-injection and lock tests need no thread orchestration.
classify = llm_deco(system="s", user="{n}", parse="json", outputs=[("label", int)])
@classify
def classify_fn(n: int) -> int: ...

@workflow
def solo(n: int @ A):
    A: label = classify_fn(n)
    return label @ A


def test_at_least_once_replays_act_on_crash_before_commit(tmp_path, monkeypatch):
    """Crash after the external call returns but before the act row commits ->
    on restart the act re-executes (at-least-once); final state is correct and
    the log holds exactly one committed act row."""
    path = str(tmp_path / "s.sqlite")
    calls = {"n": 0}
    def backend(action, inputs):
        calls["n"] += 1
        return {"label": 42}                          # deterministic result
    la = project(solo, A)

    # First attempt: raise right after record_act's INSERT, before COMMIT persists.
    orig_record = store_mod.DurableChannel.record_act
    def crash_record(self, payload):
        orig_record(self, payload)                    # INSERT into the open txn
        raise sqlite3.OperationalError("simulated crash before act commit")
    monkeypatch.setattr(store_mod.DurableChannel, "record_act", crash_record)

    conn1 = open_store(path)
    with pytest.raises(sqlite3.OperationalError):
        run_role(conn1, "A", la, {"n": 1}, solo.ns, llm_backend=backend)
    conn1.close()                                     # release the uncommitted txn's lock
    assert calls["n"] == 1

    # Restart cleanly: the act row was never committed -> re-execute, then finish.
    monkeypatch.setattr(store_mod.DurableChannel, "record_act", orig_record)
    env = run_role(open_store(path), "A", la, {"n": 1}, solo.ns, llm_backend=backend)
    assert calls["n"] == 2 and env["label"] == 42
    acts = open_store(path).execute(
        "SELECT COUNT(*) FROM events WHERE sender='A' AND kind='act'").fetchone()[0]
    assert acts == 1                                  # exactly one committed act row


def test_blocking_external_act_does_not_hold_write_lock(tmp_path):
    """While role A is inside a slow external act, a second connection can still
    take the write lock — proof the lock is released across the blocking call."""
    path = str(tmp_path / "s.sqlite")
    started = threading.Event()
    def slow_backend(action, inputs):
        started.set()
        time.sleep(0.5)
        return {"label": 7}
    la = project(solo, A)
    t = threading.Thread(target=lambda: run_role(
        open_store(path), "A", la, {"n": 1}, solo.ns, llm_backend=slow_backend))
    t.start()
    assert started.wait(timeout=5)                    # A is now inside the slow act
    other = open_store(path)
    other.execute("BEGIN IMMEDIATE")                  # would block/raise if A held the lock
    other.execute("INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp)"
                  " VALUES('B',NULL,NULL,'msg','[1]',NULL)")
    other.execute("COMMIT")
    t.join(timeout=10)


def test_parallel_two_process_kill9(tmp_path):
    store = str(tmp_path / "par.sqlite")
    wf = os.path.abspath("tests/fixtures/parallel_deploy.py")
    def serve(role, inputs):
        cmd = [sys.executable, "-m", "zippergen.serve", "serve",
               "--workflow", wf, "--role", role, "--store", store]
        for k, val in inputs.items():
            cmd += ["--input", f"{k}={val}"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    b = serve("B", {}); a = serve("A", {"x": 0, "y": 0})
    time.sleep(0.6); a.kill(); a.wait()
    a2 = serve("A", {"x": 0, "y": 0})
    out_a, _ = a2.communicate(timeout=40); out_b, _ = b.communicate(timeout=40)
    assert a2.returncode == 0 and b.returncode == 0
    result = json.loads(out_a.strip().splitlines()[-1])
    assert result["x"] == 1                            # branch completed exactly once
```

Note: the parallel DSL surface is `with parallel:` containing `with branch:` blocks (verified against `tests/test_builder.py:139-147` and `examples/parallel.py`). If projection rejects this two-way cross-branch fixture, simplify to one message per branch (mirror `examples/parallel.py`), keeping one branch that reassigns `x`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_deploy_integration.py -v`
Expected: the parallel kill-9 test FAILS before Task 8 is merged (deadlock/timeout); after Tasks 1–9 it should be the last piece. If both tasks 8 and this are done, run to confirm PASS.

- [ ] **Step 3: Implement**

No product code — these are behavioral tests over the finished feature. If a test surfaces a defect, fix in the owning module (`runtime.py`/`serve.py`/`store.py`) and note it.

- [ ] **Step 4: Run the full suite on all supported interpreters**

Run:
```bash
python -m pytest tests/ -q
~/.pyenv/versions/3.12.10/bin/python -m pytest tests/ -q   # (or CI) verify 3.12
```
Expected: all green; run the 3.12 full suite ≥5× to confirm no residual flakiness.

- [ ] **Step 5: Commit**

```bash
git add tests/test_deploy_integration.py tests/fixtures/parallel_deploy.py
git commit -m "Add durable-deploy integration tests (at-least-once, parallel crash-resume)"
```

---

## Task 11: Document the v1 limitations

**Files:**
- Modify: `src/zippergen/serve.py` (module docstring)
- Modify: `zippergen/CLAUDE.md` (a short "Durable deployment (serve)" note)

**Interfaces:** none (docs only).

- [ ] **Step 1: Write the docs**

Add to `serve.py`'s module docstring a "Limitations (v1)" paragraph, verbatim intent from the spec:

```
Limitations (v1): external effects are at-least-once (a crash between an external
call and its journal commit re-runs the act); irreversible effects (e.g. sending
mail) are not exactly-once. No durable human task queue yet (a role that is down
cannot have its human prompt answered out-of-band; the `status` journal field is
reserved for it). No snapshot fires inside a parallel region, so a crash replays
the region from the enclosing loop boundary. A blocking external act in one
parallel branch stalls that role's other branches until it returns.
```

Add a matching 4-5 line "Durable deployment (`serve`)" subsection to `zippergen/CLAUDE.md` pointing at the spec path and summarizing the journal model.

- [ ] **Step 2: Verify**

Run: `python -c "import zippergen.serve"` and `python -m pytest tests/ -q`
Expected: import clean, suite green.

- [ ] **Step 3: Commit**

```bash
git add src/zippergen/serve.py zippergen/CLAUDE.md
git commit -m "Document durable-deploy v1 limitations"
```

---

## Notes for the implementer

- **Read before editing the `WhileStmt`/`IfStmt` cases** (`runtime.py:444-536`): the in-process branches use CPL `_Formula` guards and monitors. The durable short-circuit goes *above* that logic and must not disturb it; a durable owner never has a monitor, so `c(_CondEnv(env, ns))` is the correct guard evaluation there.
- **`_input_hash` and `record_act`/`consume_journal` must agree**: the act is hashed from `named_inputs` at both journal time (`serve`) and replay time (`_step`). Keep the exact dict shape identical (`{param_name: value}` from `action.inputs`).
- **Threading `journal` through every recursive `_step`**: `SeqStmt` (Task 5) and `ParallelLocalStmt` (Task 8) are the only internal recursion sites. Miss one and a nested external act silently reverts to in-process behavior.
- **`_exec` is never given a `journal`** — do not add the parameter to `_exec`. Durable semantics live entirely in `_step`.
