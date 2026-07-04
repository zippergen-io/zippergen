# Durable-Deploy Hardening — Design Spec

**Goal:** Make the `zippergen serve` durable runtime correct for the workflows it is actually meant to run — LLM- and human-in-the-loop coordination, including long-running non-terminating approval loops and parallel regions — by journaling non-deterministic results, keeping blocking external calls off the SQLite write lock, and asserting replay consistency loudly.

**Architecture:** Extend the existing event-sourced per-role runtime (`serve.py`, `runtime.py`, `store.py`, `locator.py`). One new concept — a **role-local journal** of non-deterministic results (external action outputs and owner branch decisions), stored as new event kinds in the same `events` table and consumed FIFO on replay. No new processes, no new dependencies.

**Tech Stack:** Python 3.11+ stdlib only (`sqlite3`, `json`, `hashlib`). No third-party packages.

## Global Constraints

- **Stdlib only.** No new dependencies (`zippergen` is stdlib-only by contract).
- **Deterministic replay contract.** After this change, a role's local replay is fully deterministic: every source of non-determinism — inbound messages, reserved sends, **external action outputs, and owner decisions** — is reserved from the committed log. `@pure` actions are deterministic *by contract* and recomputed on replay (never journaled).
- **At-least-once external effects (v1).** An external call may re-execute after a crash between the call completing and its journal row committing. Acceptable for LLM (paid twice) and human (re-prompted). Irreversible effects (e.g. sending mail) are **not** made exactly-once here.
- **Cache-never-a-dependency.** Snapshots and replay floors are rebuildable caches; a stale/absent/invalid one always falls back to full replay from the seed and must never change the result.
- **No AI attribution** anywhere (commit messages, files, contributor lists). Commit messages are one line. Do not commit unless explicitly instructed.
- **In-process `run()` semantics are unchanged.** `_exec` (thread-per-lifeline) is not modified except where a shared `_step` helper is. Existing examples and the full test suite stay green on 3.11 / 3.12 / 3.13.

---

## Background — the four findings

A code review of the durable runtime (commit `3fb2267`, hardened by `6965ecf` + `8432a6c`) surfaced four issues, all verified against the code:

1. **Replay divergence for non-deterministic actions/guards.** `run_role` replays by calling `_step` with the *live* backends (`serve.py:68`); `_step`→`_exec` recomputes actions (`runtime.py:664-710`) and re-evaluates guards (`runtime.py:453`); `DurableChannel.put` during replay only pops the recorded rowid without re-executing or checking (`store.py:184`). A restarted role can move forward with newly computed values while the store/peer hold the old committed ones.
2. **Parallel holds the write lock while blocked.** `_step(ParallelLocalStmt)` delegates to `_exec`'s blocking scheduler (`runtime.py:561` → `runtime.py:734-756`), which spins to completion. Under `serve` that runs inside `BEGIN IMMEDIATE` (`serve.py:81`), so the role holds the write lock while waiting for a peer who needs it — deadlock.
3. **`serve` omits Var defaults.** `run()` preloads `{Var: default}` then overlays inputs (`runtime.py:1014`); `serve.main` seeds only parsed `--input` (`serve.py:172`). Guards reading a defaulted-but-unpassed Var behave differently under deployment.
4. **Silent replay divergence.** Reserved sends are popped without a payload check, so a divergent recompute corrupts silently.

Rejecting LLM/human workflows is **not** an option — human approval and LLM coordination are the framework's purpose (the canonical deploy target is the non-terminating command_center approval loop). The fix journals non-determinism rather than excluding it.

---

## Design

### Concept: the role-local journal

Durable execution (à la Temporal) separates deterministic control flow — replayed — from non-deterministic *results* — recorded once and replayed from the log. ZipperGen already reserves inbound messages and own sends from the `events` log; this adds the missing streams: **external action outputs** and **owner branch decisions**.

Distinction from `κ_ctrl^P` control tags: those are *cross-process coordination artifacts* — independent roles must compute the same tag to route control messages. The journal is *one role's linear replay*. So FIFO ordering is the backbone; the structural key is only an assertion that the journal still lines up with the program.

### 1. Journal schema (`store.py`)

New event rows in the existing `events` table (no schema/column change — `payload` is opaque JSON):

- `kind='act'`, `sender=<role>`, `receiver=NULL`, `channel=NULL`, payload:
  ```json
  {"status":"done","locator":[0,2],"action":"approve_plan","input_hash":"<sha1-16>","outputs":{"label":"urgent"}}
  ```
- `kind='decision'`, `sender=<role>`, payload:
  ```json
  {"status":"done","locator":[3],"value":true}
  ```

`status` is **reserved now.** v1 writes `status:"done"` on every journal row and asserts `status=="done"` on consume, but implements **no** pending rows. A later pending/task-queue feature adds `status ∈ {"pending","done","failed"}` with the same `locator`/`action`/`input_hash` shape and teaches the consumer to distinguish them — additive and migration-free, since `payload` is opaque JSON.

**What is journaled**

| Statement | Journaled? | Replay behavior |
|---|---|---|
| `ActStmt` with `LLMAction`/`HumanAction`/`PlannerAction` | **act row** | consume row, apply `outputs`; **no backend call** |
| Owner `IfStmt`/`WhileStmt` decision | **decision row** | consume row, apply `value`; **no guard recompute** |
| `ActStmt` with `PureAction` / `SelfAssignStmt` | no | recomputed deterministically |
| `RecvStmt`/`IfRecvStmt`/`WhileRecvStmt` (receiver branch) | no | already driven by the reserved control recv from the inbox |
| `SendStmt` | (existing outbox reservation) | reserve recorded rowid; **assert receiver/channel/payload** |

`input_hash` = `sha1(json.dumps(named_inputs, sort_keys=True))[:16]`, best-effort (inputs are JSON-scalar/tuple env values); stored on act rows and asserted when present.

### 2. Locator index (`locator.py`)

Add `action_node_paths(root) -> {id(node): path}` covering `ActStmt`, `IfStmt`, `WhileStmt` leaf/owner nodes, built the same way as the existing `loop_node_paths`. This is valid because leaf/owner node identity survives `_step`: `SeqStmt` steps its `first` child in place and returns the original `second` when the first empties (`runtime.py:431-442`); loop bodies reuse the same body node each iteration, so a repeated locator plus FIFO gives the occurrence count implicitly. The executing act/decision's `locator` is a dict lookup on `id(stmt)`.

### 3. Execution flow & the transaction boundary (`serve.py`, `runtime.py`)

**Execution modes — the `_step` contract.** `_step` is shared by the durable driver (`serve`) and the in-process scheduler (`_exec(ParallelLocalStmt)` steps branches via `_step`, `runtime.py:746`). Durable semantics are therefore **opt-in**, so in-process `run()` is byte-for-byte unchanged:

- **Signature:** `_step(…, journal=None) -> tuple[LocalStmt | PendingExternal, bool]`.
- **In-process mode (`journal=None`, default):** exactly today's behavior — external acts execute inline via the live backend and return `(residual, progressed)`; `ParallelLocalStmt` delegates to `_exec`. `_exec` and `run()` always pass `journal=None`, so they never observe `PendingExternal` and the `(residual, progressed)` shape they destructure is unchanged.
- **Durable mode (`journal=<JournalContext>`):** an external act with no journaled result returns `(PendingExternal(node, named_inputs), False)`; owner decisions and external acts consult/append the journal; `ParallelLocalStmt` uses the single-branch-step form (section 6), threading the same `journal` into each branch `_step`.
- **`PendingExternal`** is a frozen dataclass `PendingExternal(node: ActStmt, inputs: dict)`, returned in the residual slot with `progressed=False`. Only durable-mode callers receive it; they resolve it (backend call outside txn → journal → commit) and re-step. It never appears when `journal=None`.
- **`JournalContext`** is the `DurableChannel`'s journal reader (`consume_journal`/`record_journal`, section 4); passing it both selects durable mode and carries the reader — no separate flag.

**Apply-after-commit (journaled steps).** For an external act or an owner decision, the in-memory `env`/residual is mutated **only after** the journal row commits:

```
compute inputs (env read only)
result = <journaled outputs/value if replaying, else external call / guard eval>
BEGIN IMMEDIATE
  INSERT journal row
COMMIT                      # on failure: raise BEFORE mutating; role aborts → supervisor restart → replay
env.update(result); advance residual
```

Deterministic steps (pure act, send, recv) keep the existing mutate-inside-txn-then-commit path, because replay recomputes them identically.

**External acts — blocking call off the write lock (two-pass).** The backend call must not run inside `BEGIN IMMEDIATE`. `_step`, on reaching a live external `ActStmt` with **no** journaled result available, returns a sentinel `PendingExternal(node, named_inputs)` and mutates nothing (pass 1). The `serve` driver then:

1. runs the backend **outside any transaction** (write lock not held) to get `outputs`;
2. `BEGIN IMMEDIATE` → `ch.record_journal('act', {locator, action, input_hash, outputs})` → `COMMIT`;
3. re-enters `_step` with no transaction open (pass 2); it consumes the just-committed journal row like a replay row — `env.update(outputs)` and advance residual — so the mutation happens strictly **after** the commit succeeded.

During replay the result is already in the journal reader, so pass 1 immediately consumes it and there is no pass-2 round trip and no backend call (no re-prompt).

**Owner decisions — single txn (non-blocking eval).** A guard is a boolean lambda over `env` with no external call, so no two-pass is needed. Live: `BEGIN IMMEDIATE` → evaluate guard (env read-only) → `ch.record_journal('decision', {locator, value})` → `COMMIT` → **then** advance residual (pick the branch). Replay: consume the decision row and pick the branch from `value`, never re-evaluating the guard.

**Recursive dispatch (not a top-level `isinstance`).** `_step`'s existing structural descent through `SeqStmt`, loop bodies, and `ParallelLocalStmt` branches is what reaches the enabled leaf; the `PendingExternal` signal therefore surfaces from arbitrarily nested positions. `serve` never inspects the residual's top-level type to classify a step.

**Intra-role limitation (documented):** because a role executes its own steps sequentially, a blocking external act in one parallel branch stalls that role's *other* branches until it returns — there is no intra-role concurrency of external calls. The deferred durable human task-queue is what removes this; acceptable for v1.

### 4. Journal reader & consumed floor (`store.py`)

`DurableChannel` gains a journal reader parallel to the outbox/inbox:

- On construction (with optional `since`), load committed `act`/`decision` rows for the role with `rowid > since["journal"]` (or all, if `since is None`) into a FIFO deque, in `rowid` order.
- `consume_journal(expected_kind, locator) -> dict` pops the next row, **asserts** `kind`/`locator` (+`input_hash` when present) via the shared assertion helper, advances an explicit `_journal_consumed` high-water, and returns the payload. Raises if the deque is empty when a journaled step expected a row (⇒ live path).
- `record_journal(kind, payload_dict) -> int` INSERTs a journal row, advances `_journal_consumed` to its rowid, and enqueues the row at the head of the reader so the immediate pass-2 `consume_journal` returns it (the mechanism that keeps env mutation strictly after commit).
- `position()` returns `{"out":…, "cursors":…, "journal": self._journal_consumed}` — the **consumed/reserved** high-water, not `MAX(rowid)`, mirroring how inbox cursors are tracked. `_floor_coherent` (`serve.py:14`) is extended to validate the `journal` floor against durable state the same way it validates `out`/`cursors`.

### 5. Snapshot integration (`serve.py`, `store.py`)

`write_snapshot`/`load_snapshot` already persist `position()` as the floor; adding the `journal` key flows through automatically. A loop-boundary snapshot now bounds journal replay exactly as it bounds outbox/inbox. Stale/absent/incoherent snapshot → `_try_resume` returns `since=None` → full journal replay from the seed. No behavior change to the snapshot lifecycle itself.

### 6. Parallel as a true durable step (`runtime.py`)

Replace the `_step(ParallelLocalStmt)` delegation (`runtime.py:561`) with a real single step (durable mode only; `journal=None` still delegates to `_exec`, per section 3):

```python
case ParallelLocalStmt(branches=branches, branch_indices=labels) if journal is None:
    _exec(stmt, env, ch, …)          # unchanged in-process behavior
    return EmptyStmt(), True

case ParallelLocalStmt(branches=branches, branch_indices=labels):   # durable
    residuals = list(branches)
    for i, branch in enumerate(residuals):
        if isinstance(branch, EmptyStmt):
            continue
        new_branch, progressed = _step(branch, env, ch, …, journal=journal)
        if isinstance(new_branch, PendingExternal):
            return new_branch, False          # propagate up; serve resolves it
        residuals[i] = new_branch
        if progressed:
            if all(isinstance(b, EmptyStmt) for b in residuals):
                return EmptyStmt(), True
            return ParallelLocalStmt(tuple(residuals), labels), True
    if all(isinstance(b, EmptyStmt) for b in residuals):
        return EmptyStmt(), True
    return stmt, False   # blocked; caller commits/sleeps and retries
```

Advance exactly one enabled branch, return the rebuilt region; block (`False`) only when no branch can progress. A branch's `PendingExternal` propagates up unchanged as the region's outcome, and `serve` resolves it exactly as for a top-level external act. `_exec`'s scheduler (in-process `run()`) is untouched.

**Documented limitation:** the rebuilt `ParallelLocalStmt` residual changes object identity every step, so no snapshot fires *inside* a parallel region — a crash mid-region replays it from the enclosing loop boundary. Bounded replay inside parallel is a later refinement.

### 7. Default seeding (`serve.py`)

`serve.main` builds `env = {k: v.default for k, v in wf.ns.items() if isinstance(v, Var)}` then overlays `_parse_inputs(args.input)` (parity with `runtime.py:1014`), and passes the **merged** env to `seed_env` so the persisted seed already contains defaults. On restart the recorded seed still wins (unchanged).

### 8. Reserved-entry assertions & `ReplayMismatch` (`store.py`)

One `ReplayMismatch(Exception)` type, raised at every reserved-consumption site:

- **Send** (outbox reserve, `store.py:184`): assert the reserved row's `receiver`/`channel` (from the outbox tuple) match the `SendStmt`, then re-read the row's `payload` by `rowid`, **JSON-decode it, and compare the decoded value to `list(values)`** — a canonical value comparison, never a raw-string one, so JSON key order / whitespace / float formatting cannot false-fail. No locator (msg rows carry none).
- **Act / decision** (`consume_journal`): assert `kind` and `locator` match the executing node (+`input_hash` on acts when present).

A mismatch means the program diverged from committed history (a non-deterministic `@pure`, an edited workflow, a corrupted store) — fail loudly rather than corrupt silently.

---

## Error handling

- **Commit failure on a journaled step:** raise before mutating env/residual; the role process exits and the supervisor restarts it; replay reconstructs from the committed log (external act re-runs — at-least-once).
- **Journal deque empty when a row was expected:** normal at the replay→live boundary; switch to the live path (call backend / eval guard).
- **`ReplayMismatch`:** uncaught by design — surfaces to the operator; not swallowed by `_maybe_snapshot`'s best-effort handler.
- **Non-serializable act outputs:** `record_journal` raises `TypeError` (JSON) inside its own txn; treated as a workflow bug, not caught.

## Testing

- **Memoization / no re-execution:** a workflow with a call-counting "LLM" backend (returns an incrementing value); run to a snapshot, restart, assert the backend call count does **not** increase on resume and the final env matches a single clean run.
- **Loud mismatch:** corrupt a journal row's `locator`; assert `ReplayMismatch` on resume. Corrupt a reserved send's payload; assert `ReplayMismatch`.
- **At-least-once window:** inject a crash between the external call returning and the journal commit; on restart the act re-runs and the final state is still correct (idempotent test action).
- **Parallel durable:** two real `serve` processes running a workflow with a parallel region, `kill -9` one mid-region, restart; no deadlock, correct result. Extends `scratchpad/orchestrate.py` into a committed integration test.
- **Blocking-call lock release:** a role in a long external act does not block a peer's committed sends (assert the peer progresses while the act is outstanding).
- **Defaults parity:** serve a workflow whose guard reads a defaulted Var not passed via `--input`; result matches in-process `run()`.
- **Regression:** full suite green on 3.11 / 3.12 / 3.13; in-process `run()` behavior unchanged.

## Out of scope (documented limitations)

- **Exactly-once irreversible external effects** — needs effect-level idempotency keys; v1 is at-least-once.
- **Durable human task queue / pending records** — the reserved `status` field is the seam; the natural next step for the human-approval deploy target (lets the UI own prompt/result durably while the role process is down).
- **Bounded replay inside a parallel region** — no snapshot fires inside parallel; crash replays the region from the enclosing boundary.
- **Intra-role concurrency of external calls** — a blocking external act stalls that role's other parallel branches until it returns.

## Open risks

- `PendingExternal` threading through `_step`'s recursion (esp. nested in parallel) is the subtlest part; the two-pass "signal then resolve" must leave env/residual untouched on the signal pass. Covered by the memoization and parallel tests.
- `input_hash` requires JSON-serializable inputs; non-serializable inputs skip the hash (assertion degrades to locator+kind), never crash.
