# Deployable Runtime Design

**Date:** 2026-07-02
**Status:** Draft — pending review

## Overview

Today a checked workflow can only be executed by `run()`, which spawns one
thread per lifeline inside a single Python process, wires them with in-memory
FIFO queues, runs the protocol to completion, and returns. It is ephemeral: a
crash loses everything, and the operational surface is "activate a venv and
invoke the interpreter."

This design adds a **deployment target**: a way to run a checked workflow as a
set of standing, supervised, crash-survivable services — one process per role —
without operating a Python interpreter by hand. Correctness of the *interaction
structure* is already established at compile time (acyclicity, path-keyed
channels, sibling-disjoint channel sets, Theorem 3.1 / Corollary 3.1); the
runtime does not re-derive coordination guarantees. Its one new
correctness-critical property is **durable resume**: after a crash, a role
resumes from its last committed step, never from zero.

The design reuses the existing projection engine, local-statement interpreter,
and CPL monitor unchanged. The new surface is exactly three things: a durable
**transport + log**, an **execution mode split** (replay vs. live), and a
**service/CLI wrapper**.

---

## 1. Goals and Non-Goals

**Goals**

- Run a workflow as standing, supervised services — one OS process per role.
- No manual `python`: the operator interacts via a service manager
  (`systemctl start zippergen@LLM1`) or a container (`docker run … --role LLM1`).
- Automatic crash restart (OS-level supervision) with **durable resume**: a
  restarted role reconstructs its exact `(env, program-position)` and continues.
- Support **non-terminating** workflows (e.g. `command_center.py`) in a
  crash-survivable, observable way — replay length must not grow without bound.
- One append-only event stream that simultaneously serves as transport, replay
  log, and the causally-ordered stream read by the CPL monitor and dashboard.
- **Zero behavior change to the existing in-process path.** Every current
  example and test that runs through `run()` must behave *exactly* as it does
  today, throughout development. The durable runtime is strictly additive; the
  thread-based path is the default and is never regressed.

**Non-Goals (this version)**

- No elimination of the Python runtime. Python is present but operationally
  hidden (approach "a", not ahead-of-time compilation to a standalone binary).
- No `kpar` / parallel branches. No LLM calls. No external effects (email, HTTP).
- No multi-instance multiplexing: a process runs a **single** protocol instance.
  (A non-terminating instance is still a single instance — see §9.)
- No new third-party dependencies (see §14 / Open Questions).

**v1 validation target.** Two roles, one sequential message exchange, one branch
with an explicit `else` (both outcomes communicated to the non-deciding role).
The property under test is durable resume in isolation: kill a role mid-protocol
and confirm it resumes exactly, with no lost, duplicated, or re-ordered
messages.

---

## 2. Architecture

The deployment unit is **one role, one process, one long-lived loop**. A
deployment of an N-role workflow is N supervised processes that coordinate
*only* through a single shared SQLite event store on the local machine.

```
  ┌── process: role A ──┐        ┌── process: role B ──┐
  │  project(wf, A)     │        │  project(wf, B)     │
  │  interpreter loop   │        │  interpreter loop   │
  │  Channel(sqlite) ───┼──┐  ┌──┼─── Channel(sqlite)  │
  └─────────────────────┘  │  │  └─────────────────────┘
                           ▼  ▼
                    ┌─────────────────┐
                    │  events.sqlite  │  append-only, rowid = total order
                    └─────────────────┘
                           ▲  ▲
                ┌──────────┘  └──────────┐
        ┌───────────────┐        ┌───────────────┐
        │  CPL monitor  │        │   dashboard   │   (read-only consumers)
        └───────────────┘        └───────────────┘
```

Today's `run()` fuses three concerns: project all lifelines, spawn one thread
each, and wire them with an in-process queue map. This design **splits** those:

- **Project one role.** `project(wf, role)` already yields exactly one local
  program. No change.
- **Abstract the channel** behind an interface with two implementations
  (§4): the existing in-process queue (kept for dev/test/UI) and a new durable
  SQLite-backed channel used for deployment.
- **Persist around the loop.** The durable channel is the log; the loop
  replays on startup and runs live thereafter (§6–§9).

The thread-based `run()` is **retained** as the fast path for local development,
tests, and the single-process UI. Same interpreter, two transports.

---

## 3. Deployment Artifact and Supervision

A thin entry point is added to the package:

```
zippergen serve --workflow examples/diagnosis.py --role LLM1 --store /var/lib/zippergen/diagnosis.sqlite
```

It imports the workflow module, resolves the named `Lifeline`, projects the
local program for that role, opens the shared store, and runs the per-role loop
(§9). Startup inputs for a role (the workflow's `initial_envs` for that
lifeline) are supplied via `--input k=v` or a small JSON file, and are recorded
into the log as the role's seed event so replay is self-contained.

Supervision is **OS-level**, not application-level:

- **systemd**: a templated unit `zippergen@.service` with
  `Restart=always`; instantiate one per role (`zippergen@LLM1`, `zippergen@User`).
- **container**: one image, `restart: always`, `--role` per container.

The operator never invokes Python directly. `stdlib`-only holds: `argparse` for
the CLI, `sqlite3` for the store, no runtime dependency added.

---

## 4. The Channel Abstraction

The interpreter already touches channels through a narrow surface: `.put(values,
stamp)` on send, and `_try_channel_get(sender, receiver, channel)` on receive
(which returns `None` when nothing is available, driving the `progressed=False`
wait). We introduce a `Channel` protocol with exactly those operations and two
implementations:

- `InProcessChannel` — the current `defaultdict[_SeqQueue]`. Unchanged behavior;
  used by `run()`, tests, and UI.
- `DurableChannel` — backed by the shared SQLite store. `send` is an `INSERT`;
  `try_recv` is a `SELECT` of the next unconsumed row on the channel key.

The interpreter (`_step` / `_exec`) is otherwise untouched. Because `try_recv`
already has "nothing yet → wait" semantics identical to `queue.Empty`, the
blocking `queue.get` on the deployment path is replaced by polling the store
(short sleep between empty reads). A socket / filesystem-watch wake-up may be
added later purely as a latency optimization; it is **not** required for
correctness because the store, not any wire, is the source of truth.

---

## 5. Store Schema

One append-only table is the entire coordination substrate. All writes serialize
through the single SQLite file (WAL mode), so `rowid` is a global total order
consistent with causality — this *is* the "shared causally-ordered event
stream."

```sql
CREATE TABLE events (
  rowid        INTEGER PRIMARY KEY,   -- global total order == causal order
  sender       TEXT NOT NULL,
  receiver     TEXT,                  -- NULL for role-local events (act/decision)
  channel      TEXT,                  -- the (sender,receiver,channel) FIFO key
  kind         TEXT NOT NULL,         -- 'seed'|'msg'|'ctrl'|'act'|'decision'|'snapshot'|'effect'
  payload      BLOB,                  -- json-encoded values / outcome
  causal_stamp BLOB                   -- monitor vc/view snapshot (nullable)
);

-- Per-role durable cursor: how far this role has consumed each channel.
CREATE TABLE cursors (
  role     TEXT NOT NULL,
  channel  TEXT NOT NULL,
  consumed INTEGER NOT NULL,          -- highest rowid consumed on this channel
  PRIMARY KEY (role, channel)
);

-- Periodic per-role state checkpoints (see §8).
CREATE TABLE snapshots (
  role      TEXT NOT NULL,
  at_rowid  INTEGER NOT NULL,         -- log position this snapshot reflects
  env       BLOB NOT NULL,            -- json-encoded local env
  residual  BLOB NOT NULL,            -- serialized residual LocalStmt (program position)
  PRIMARY KEY (role, at_rowid)
);
```

**FIFO per channel is preserved for free.** The correctness proofs assume FIFO on
each `(sender, receiver, channel)` key; reading rows for a given `channel` in
`rowid` order *is* that FIFO. Reading nothing means "not yet arrived."

The private per-role durable state is deliberately tiny: a **consume cursor** per
channel plus periodic snapshots. Everything else lives in the shared table.

---

## 6. Event-Sourcing and the Two Execution Modes

The store is the single source of truth. A role's state is a **fold** over its
events; env is *derived*, not primarily persisted (snapshots in §8 are an
optimization, not the ground truth).

The interpreter runs in one of two modes, and the boundary between them is the
heart of crash-correctness:

- **Replay mode** — active from startup until execution reaches the tail of the
  log (the last committed event for this role). In replay mode the interpreter
  is **effect-suppressed**: it re-executes deterministic local logic to
  reconstruct `env` and the residual continuation, but it does **not** re-perform
  any effect that is already committed. A `send` in replay mode does not
  `INSERT`; it recognizes "already committed as row N" and advances the cursor
  past it. A recorded action/effect returns its **logged outcome** rather than
  re-running.

- **Live mode** — active once execution passes the last committed event. Now
  `send` actually `INSERT`s, actions actually run, and outcomes are recorded.

Because v1 local steps are deterministic (pure actions, no LLM), replaying the
committed events reconstructs `(env, residual)` **exactly**. The transition
replay → live happens exactly once per process lifetime, when the fold catches
up to the present.

---

## 7. Effect Idempotence (the crux)

Restart must never re-perform an effect that is already durable. There are two
categories, handled differently:

**Internal role→role messages.** The send *is* the `INSERT`, so effect and
record are the same atomic act. During replay these are never re-inserted — they
are read and skipped. Double-send is therefore **structurally impossible** for
internal messages. This is the entirety of v1's effect surface, so v1 achieves
exactly-once internally by construction.

**External, non-idempotent effects (future: real LLM call, email send).** These
are not under our control and cannot be un-done. The pattern is **memoize the
outcome**: on first execution, perform the effect and record its result as an
`effect` event; on replay, return the logged result instead of re-performing.
A snapshot taken *after* an effect's event is committed begins replay *past* that
event, so it is never re-issued — this is precisely the "a snapshot containing a
send must not re-send on replay" concern, and it is handled by the replay-past-
committed-events rule, not by special-casing snapshots.

The one irreducible risk is a crash in the window **between performing an
external effect and committing its outcome**. No log erases this; it is closed
either with an idempotency key the external system honors, or by consciously
accepting at-least-once / at-most-once semantics for that effect. **v1 has no
such window** (all sends are internal and transactional), but the design names it
explicitly because `command_center` and real LLM calls will introduce it. See
Open Questions.

---

## 8. Snapshots and Non-Terminating Workflows

Pure event-sourcing replays the whole log on restart. For a finite protocol that
is trivial; for a non-terminating workflow (`command_center`) the log grows
without bound and "replay from the top" becomes "re-execute all of history to
reboot." Snapshots bound replay length.

Periodically (every K committed steps, and/or on a timer) a role writes a
`snapshots` row: its serialized `env` and residual `LocalStmt` at a known log
position `at_rowid`. On restart, a role:

1. loads its latest snapshot (if any) → `(env, residual, at_rowid)`;
2. enters **replay mode** and folds only the events *after* `at_rowid`;
3. transitions to **live mode** at the tail.

Serializing the residual `LocalStmt` is tractable — the IR is frozen dataclasses
— but couples the snapshot format to IR shape; snapshots are therefore treated as
a *cache* that can always be rebuilt from the event log, never as the sole source
of truth. The finite path (no snapshots, full replay) is built first; snapshots
are the second increment, required before `command_center`-style deployment.

---

## 9. The Per-Role Loop and Commit Discipline

```
open store; ensure schema
load latest snapshot for role (env, residual, at_rowid) or seed from --input
# ---- replay mode ----
fold events after at_rowid:
    for each already-committed step, reconstruct env / advance cursor;
    do NOT re-perform effects (§7)
# ---- live mode ----
loop:
    step = interpreter.step(residual, env, durable_channel)
    if step blocked on receive:
        poll store for next row on the channel; sleep briefly if none
        continue
    # step produced effects (a send, an act) and a new (env, residual)
    BEGIN TRANSACTION
        INSERT any events the step emitted
        UPDATE cursors for any rows the step consumed
        (periodically) INSERT snapshot(env, residual)
    COMMIT
    if residual is terminal: exit (or idle) — supervision will not restart a clean exit
```

**The commit invariant.** A step's *effects* and the *cursor advance / state*
that depend on them are written in **one transaction**. This is what makes
"consumed" well-defined and gives clean crash semantics:

- **Delivery vs. consumption are distinct.** A row is *delivered* the instant it
  is inserted (visible to its receiver). It is *consumed* by role B only once B
  has advanced its cursor past it **and** committed the step that used it, in the
  same transaction.
- A crash **before** COMMIT loses the in-flight step entirely; on restart the
  cursor and state are exactly as before the step, and the step re-runs cleanly
  in live mode (no effect leaked, because nothing was committed).
- A crash **after** COMMIT means the step is fully durable; on restart it is part
  of the replayed prefix and is not re-performed.

Either way the role resumes from its last committed step, never from zero, and no
message is lost, duplicated, or re-ordered.

---

## 10. Projection Mapping — No Code Generation

Resolves the "one generated module per role vs. single role-tagged file"
question: **neither.** `project(wf, role)` produces the local program *in memory*
at process startup; there is no file-generation step, and adding one would create
a second artifact to keep consistent with the source workflow. Instead, a
**single role-parametrized entry point** (`zippergen serve --role R`) imports the
workflow, projects role `R` at startup, and runs it. One code path, projection as
the single source of truth, identical to how `run()` obtains local programs
today.

Ahead-of-time code generation (emitting a standalone per-role artifact) belongs
only to the deferred "no Python runtime at all" goal (approach "b") and is
explicitly out of scope here.

---

## 11. CPL Monitor and Dashboard

Both are **read-only consumers** of the same `events` table, tailing it in
`rowid` order. The `causal_stamp` column carries the monitor's vector-clock /
view snapshot exactly as `_SeqQueue` items carry `vc`/`view` today, so the
monitor's guard evaluation is unchanged — it now reads stamps from the table
instead of from queue items. The dashboard renders the same ordered stream. No
separate observation channel is written; the transport log *is* the observation
stream.

---

## 12. Testing Strategy

**Regression gate (runs first, stays green throughout).** Before any shared-code
change (notably the §4 channel extraction), a characterization harness runs the
existing examples/tests on the in-process path and records their behavior; that
harness must stay green at every step. Because some examples use randomized mock
outputs, the harness asserts structural invariants (completes without error;
produces the expected output shape / lifeline set) and seeds randomness where an
exact-value comparison is wanted. This is the concrete mechanism that enforces
the "zero behavior change" goal.

The property under test is durable resume, so tests exercise crash + restart, not
just happy-path completion:

- **Golden run.** Two-role, one-exchange, one-branch (both outcomes) workflow
  runs to completion on the durable channel; assert final envs match the
  thread-based `run()` result for the same inputs.
- **Kill-and-resume matrix.** Kill a role process at each step boundary (before
  send-commit, after send-commit, before/after the branch decision, before/after
  the receiver consumes control) and assert the resumed run produces the same
  final state and the same total event stream (modulo nothing — no duplicates,
  no gaps).
- **Crash-in-transaction.** Simulate a crash mid-transaction (kill between INSERT
  and COMMIT via a fault hook) and assert the step re-runs cleanly with no leaked
  effect.
- **FIFO / ordering.** Assert per-channel `rowid` order equals send order.
- **Determinism of replay.** Replaying a completed log reconstructs `(env,
  residual)` bit-identically to the live run.

---

## 13. Out of Scope (named, not solved)

- `kpar` / parallel branches; sibling-disjoint durable channels.
- LLM calls and other external effects (drives the §7 exactly-once window).
- Multi-instance multiplexing / triggering (reactive daemon handling repeated
  instances) — would add an `instance_id` column; the schema extends cleanly.
- Ahead-of-time codegen / Python-free artifact (approach "b").
- Distribution across machines (this design is single-machine, one SQLite file).

---

## 14. Open Questions

1. **stdlib-only relaxation.** The entire deployment layer is stdlib without
   effort (`sqlite3`, `argparse`, OS-level supervision), so nothing here forces a
   dependency. Whether to relax the project-wide stdlib-only rule *elsewhere*
   (e.g. official LLM SDKs in place of hand-rolled `urllib` backends) is a
   separate strategic decision and is **not** settled by this design. Recommend:
   hold the line for the runtime; decide the SDK question independently.
2. **External-effect exactly-once (§7).** When LLM/email effects arrive, choose
   per-effect between idempotency-key exactly-once and accepted at-least/at-most-
   once. Not needed for v1.
3. **Reactive v2.** If always-on, repeated-instance behavior is wanted later,
   add `instance_id` and a triggering mechanism. The v1 schema and loop are
   designed to extend to it, not to be rewritten.
