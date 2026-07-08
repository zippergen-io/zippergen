# SQLite-First Execution Runtime - Design Spec

**Date:** 2026-07-08
**Status:** Draft

## Overview

ZipperGen currently has two execution shapes:

- the in-process runner, where `workflow(...)` starts one thread per lifeline
  and connects roles with in-memory queues;
- the durable deployment runner, where `zippergen serve --role ...` runs one
  role per process and coordinates through a shared SQLite event store.

This split is useful during development, but it creates two runtime surfaces to
maintain. Every feature has to be kept consistent across both: messages,
control broadcasts, parallel regions, external actions, human actions, UI trace
events, snapshots, replay, and CPL monitoring.

The target architecture is simpler:

> All workflow execution goes through an event store. SQLite is the source of
> truth for role communication, durable replay, UI observation, and human task
> results.

The public APIs can stay friendly. `workflow(...)` should still run a workflow
locally. `workflow.configure(ui=True)` should still open the browser UI.
`zippergen serve` should still run one deployed role. Internally, these modes
should share the same SQLite-backed execution engine.

## Goals

- Use one communication mechanism for all runtime modes: the SQLite event log.
- Keep the current user-facing local API: `workflow(...)`.
- Keep the current deployed API: `zippergen serve --workflow ... --role ...`.
- Make local runs behave like deployed runs, including durable replay semantics.
- Make the browser UI read from SQLite instead of relying on an in-memory trace
  callback as the source of truth.
- Make human prompts durable by recording pending tasks and answers in SQLite.
- Preserve stdlib-only runtime dependencies.
- Migrate incrementally, keeping the existing test suite green at every step.

## Non-Goals

- Do not remove `workflow(...)` as a convenient API.
- Do not require users to start N shell processes for local examples.
- Do not introduce a broker, server database, Redis, or message queue.
- Do not make the UI correctness-critical. If the UI process crashes, workflow
  execution must continue or resume from SQLite.
- Do not delete the current in-memory runner until feature parity is proven.

## Core Principle

SQLite should be the only runtime coordination substrate.

That means:

- Role-to-role messages are SQLite rows.
- Control broadcasts are SQLite rows.
- External action results are SQLite journal rows.
- Owner branch decisions are SQLite journal rows.
- Snapshots and replay floors are SQLite rows.
- Human prompts and answers are SQLite rows.
- UI state is derived by reading SQLite rows.

The UI may still use HTTP, SSE, or WebSockets to update the browser, but those
are presentation channels only. They must not be needed to reconstruct the
workflow state.

## Target Architecture

```
              local API                         deployed API
        workflow(...)                    zippergen serve --role A
             |                                   |
             v                                   v
   local SQLite supervisor                single-role service loop
   starts all role loops                  runs one role loop
             \                                   /
              \                                 /
               v                               v
                    SQLite event store
        events, cursors, snapshots, journals, human tasks
                         |
                         v
              dashboard / monitor readers
```

There is one execution model:

1. Project the workflow to local programs, one per role.
2. Seed role inputs into the store.
3. Run role loops that step local programs against the store.
4. Append messages, decisions, journal rows, task rows, and snapshots.
5. Derive results and UI state from the store.

The difference between local and deployed execution is only how role loops are
hosted:

- local mode hosts all roles in one Python process, probably in threads, over a
  temporary SQLite store;
- deployed mode hosts each role in its own supervised OS process over a
  persistent SQLite store.

## Public Surfaces

### Local execution

The existing local API should remain:

```python
result = workflow(input_a=1, input_b=2)
```

Internally, this should eventually do:

1. create a temporary SQLite store;
2. seed inputs for every role;
3. project the workflow to every role;
4. start one role loop per lifeline in local threads;
5. wait for completion or timeout;
6. collect the workflow output from the output role's final environment;
7. keep or delete the temp store depending on debug settings.

### Local execution with UI

The existing UI API should remain:

```python
workflow.configure(ui=True)
result = workflow(...)
```

Internally, this should:

1. run the workflow through the same local SQLite supervisor;
2. start a dashboard server;
3. have the dashboard tail or poll SQLite;
4. render event rows instead of depending on in-memory trace callbacks.

### Deployed execution

The existing deployed API should remain:

```bash
zippergen serve --workflow workflow.py --role Planner --store run.sqlite
zippergen serve --workflow workflow.py --role Reviewer --store run.sqlite
zippergen serve --workflow workflow.py --role Executor --store run.sqlite
```

This path already uses SQLite. It should become one hosting mode of the shared
engine, not a special runtime with separate semantics.

### Dashboard

Add an explicit store-backed dashboard surface:

```bash
zippergen dashboard --store run.sqlite
```

For local examples, `configure(ui=True)` can start this automatically against
the temporary store.

## Store Model

The existing `events`, `cursors`, and `snapshots` tables remain the base.

Current rows already cover:

- `seed`
- `msg`
- `ctrl`
- `act`
- `decision`

The SQLite-first runtime should add durable human task state. This can be done
either as new event kinds in `events` or as a separate table. A separate table is
clearer operationally, while still treating SQLite as the only source of truth.

Candidate table:

```sql
CREATE TABLE IF NOT EXISTS human_tasks (
  task_id      TEXT PRIMARY KEY,
  role         TEXT NOT NULL,
  locator      BLOB NOT NULL,
  action       TEXT NOT NULL,
  inputs       BLOB NOT NULL,
  status       TEXT NOT NULL,  -- 'pending'|'done'|'failed'|'cancelled'
  result       BLOB,
  created_at   REAL NOT NULL,
  updated_at   REAL NOT NULL
);
```

The exact schema can change during implementation. The important rule is that a
role waiting for a human result must be able to crash and later resume by
reading only SQLite.

## Execution Engine Shape

Introduce a shared engine layer around the current durable pieces:

```text
ExecutionStore
  open SQLite
  seed inputs
  load snapshots
  append/read events
  append/read journal rows
  append/read human task rows

RoleRunner
  one role
  one local residual
  one local env
  replay phase
  live phase
  step loop

LocalSupervisor
  one workflow
  one temp or persistent store
  one RoleRunner per lifeline
  result collection
  timeout/cancellation

DeployedServe
  one RoleRunner
  persistent store
  process supervisor outside Python

Dashboard
  read-only store reader
  optional human-task writer
```

The existing `run_role` is close to the `RoleRunner` concept. The main change is
to make it a reusable engine component rather than logic owned only by
`zippergen serve`.

## Replay Semantics

Replay semantics should be identical in local and deployed modes:

- Recorded sends are reserved, not inserted again.
- Recorded receives are served from the committed log.
- Recorded external action results are consumed from journal rows.
- Recorded owner decisions are consumed from journal rows.
- Snapshot floors bound replay but are never correctness-critical.
- A mismatch raises loudly instead of corrupting the run.

This means local examples will exercise durable semantics by default. That is
good: bugs become visible earlier.

## UI Semantics

The current `WebTrace` UI is callback-driven. That should become a compatibility
layer or be replaced by a store-backed dashboard.

Target behavior:

- The dashboard reads events from SQLite in `rowid` order.
- Late browser connections can reconstruct the full visible state from SQLite.
- Restarting the dashboard does not affect workflow execution.
- Replaying a workflow after a role crash does not duplicate UI-visible actions;
  the dashboard renders committed history.
- Browser "run again" for local demos should create a fresh run/store or a fresh
  instance id, not reuse in-memory state.

The dashboard can still push to the browser with SSE. The key rule is that SSE
is derived from SQLite, not the authority.

## Human Actions

Human actions should become durable tasks.

Live path:

1. Role reaches a visible `HumanAction`.
2. Role inserts or reserves a `human_tasks` row with `status='pending'`.
3. Role waits by polling SQLite, without holding the write lock.
4. UI writes the answer into SQLite and marks the task `done`.
5. Role consumes the answer, records/applies the action result, and advances.

Replay path:

1. If the human result was already committed, the role consumes it without
   re-prompting.
2. If only a pending task exists, the restarted role waits for the same task.

This is the main missing piece for making deployment and UI feel like one
system.

## CPL Monitoring

The durable path currently does not wire CPL Formula monitors. SQLite-first
execution should fix this by making monitor state and observation derive from
the same committed event stream.

Implementation options:

- keep per-role monitor state in memory during live execution, reconstructing it
  during replay from committed events;
- periodically snapshot monitor state alongside env/residual snapshots;
- expose monitor observations to the dashboard as derived rows or derived
  read-side state.

The first implementation should prefer correctness and simplicity over speed:
reconstruct from committed events, then optimize with snapshots later.

## Result Collection

The local `workflow(...)` API needs a clear way to return a result.

Possible first version:

- keep the final `env` returned by each local `RoleRunner`;
- return the configured workflow output from the output lifeline's final env;
- optionally write a terminal row per role for dashboard/debugging.

Later, result rows can be stored in SQLite so a completed run is inspectable
without keeping Python objects alive.

## Instance Identity

The current deployment model is one workflow instance per store. SQLite-first
local execution can keep that model initially by creating one temp store per
run.

If multi-instance stores are needed later, add an `instance_id` column or encode
instance identity in every table. Do not mix multiple instances in one store
without an explicit instance key.

## Migration Plan

### Phase 1: Specify and stabilize the target

- Write this design spec.
- Document that SQLite is the intended single execution substrate.
- Leave current behavior unchanged.

### Phase 2: Extract a reusable role runner

- Refactor `run_role` into a shared `RoleRunner` or equivalent helper.
- Keep `zippergen serve` as a thin wrapper around it.
- Preserve all durable tests.

### Phase 3: Add a local SQLite supervisor

- Add an opt-in local runner that creates a temp SQLite store and starts all
  roles through the shared role runner.
- Run it in tests alongside the current in-memory runner.
- Compare outputs and important trace/event properties.

### Phase 4: Store-backed dashboard

- Add a dashboard reader that tails SQLite.
- Make `configure(ui=True)` able to use the store-backed dashboard for the new
  local SQLite runner.
- Keep the old callback UI path until parity is reached.

### Phase 5: Durable human tasks

- Add pending/done human task rows.
- Teach role runners to wait for task completion through SQLite.
- Teach the dashboard to write answers through SQLite.

### Phase 6: CPL monitor parity

- Reconstruct monitor state on the SQLite path.
- Add tests comparing Formula guard behavior on in-memory and SQLite execution.

### Phase 7: Switch the default

- Make `workflow(...)` use the SQLite supervisor by default.
- Keep an escape hatch for the old in-memory runner temporarily.
- Remove or demote the old in-memory channel runner after a deprecation window.

## Testing Strategy

Add parity tests before changing defaults:

- sequential send/receive output parity;
- owner `if` and receiver branch parity;
- owner `while` and loop snapshot parity;
- parallel region parity;
- external LLM/human/planner journal replay parity;
- replay idempotence: re-running a completed store inserts no duplicate events;
- crash/restart parity for local SQLite supervisor;
- UI reconstruction from store after dashboard restart;
- human task completion after role restart;
- CPL Formula guard parity.

The full existing suite must stay green. During migration, run both execution
engines for key fixtures until the SQLite path becomes the default.

## Operational Rules

- SQLite remains local-machine coordination. A shared network filesystem should
  not be assumed safe unless tested explicitly.
- Role processes should use one connection each.
- Long external calls must never hold the SQLite write lock.
- UI readers should avoid blocking writers.
- Snapshots remain rebuildable caches. Deleting snapshots must never change the
  final result.
- Store corruption or replay mismatch should fail loudly.

## Open Questions

- Should human tasks live in `events` as `kind='human_task'` /
  `kind='human_result'`, or in a separate `human_tasks` table?
- Should local temp stores be deleted by default, or kept when `ui=True` /
  debug mode is enabled?
- Should `workflow(...)` expose a `store=` argument for reproducible local runs?
- How should the dashboard represent durable replay versus ordinary live
  execution?
- When should result rows become part of the store schema?
- Is one workflow instance per store sufficient for the next milestone?

## Summary

The desired endpoint is not "only `zippergen serve`." The desired endpoint is
"only SQLite-backed execution."

`workflow(...)`, `configure(ui=True)`, `zippergen serve`, and the future
dashboard should be different launch and presentation modes over the same
event-store runtime. That gives ZipperGen one execution semantics to maintain
while preserving the simple local developer experience.
