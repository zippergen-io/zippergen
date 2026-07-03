# Snapshots — Bounded Replay for Long-Running Workflows

**Date:** 2026-07-03
**Status:** Draft — pending review
**Follows:** `2026-07-02-deployable-runtime-design.md` (§8, deferred there)

## Overview

The durable runtime resumes a crashed role by replaying its committed event
history from the seed. For a finite protocol that is trivial; for a long-running
or non-terminating `while`-loop workflow the history grows without bound, so
"replay from the top" becomes "re-execute all of history to reboot." Snapshots
bound replay length: a role periodically records a checkpoint at a loop-iteration
boundary, and on restart replays only the tail of events since the last
checkpoint.

The design's central constraint, discovered while scoping this work: a role's
"program position" is its residual continuation, and that continuation cannot be
serialized — `WhileStmt`/`IfStmt` conditions are `lambda` closures and `ActStmt`s
carry live action objects (functions). So this design **never serializes the
program.** It persists only cheap, JSON-safe state (`env`) plus a *locator* that
re-derives position by re-projecting the workflow (which regenerates the lambdas
and functions). A snapshot is a **rebuildable cache**, never a correctness
dependency: a missing or stale snapshot degrades to today's full-replay behavior,
which is already proven correct.

---

## 1. Goals and Non-Goals

**Goals**
- Bound restart replay length for long-running / non-terminating `while`-loop
  workflows: replay only events since the last checkpoint, not from seed.
- Never weaken the durable-resume guarantee. A snapshot can only make replay
  *shorter*; it can never cause a lost, duplicated, or reordered message, and can
  never corrupt resume — because on any doubt it is discarded and full replay
  from seed takes over.
- Reuse the existing store, `DurableChannel`, and `run_role` with the smallest
  possible surface change.

**Non-Goals**
- No serialization of the residual `LocalStmt` / IR / action objects / lambdas.
- No snapshots at arbitrary mid-iteration points — only at outer-loop boundaries
  (see §3). Nested-loop and mid-body snapshots are out of scope.
- No change to the in-process `run()` path.
- Not addressed here (still deferred from the parent spec): external-effect
  memoization (§7 there), `kpar`, CPL monitor wiring, reactive multi-instance.

---

## 2. Snapshot Storage

One row per role, upserted so the table cannot grow (latest snapshot only):

```sql
CREATE TABLE IF NOT EXISTS snapshots (
  role    TEXT PRIMARY KEY,
  env     BLOB NOT NULL,   -- json-encoded local env (scalars)
  locator BLOB NOT NULL,   -- json-encoded child-index path to the loop node
  floor   BLOB NOT NULL    -- json-encoded channel replay-floor (see below)
);
```

`env` is restored verbatim on resume; `locator` says where in the (re-projected)
program to resume; `floor` is the replay window boundary.

**The replay floor is per-channel, not a single rowid.** An inbound message's
`rowid` is assigned when the *peer sends* it, not when this role *consumes* it, so
in a producer-push loop a low-`rowid` message can be consumed *after* the
boundary. A single `at_rowid > X` tail filter would wrongly drop it and silently
diverge env. The floor therefore records the `DurableChannel`'s position at the
boundary:

```
floor = {
  "out": <highest rowid of this role's own committed sends at the boundary>,
  "cursors": { "<chan_key>": <consumed rowid on that inbound channel at the boundary>, ... }
}
```

Tail replay is then: own sends with `rowid > floor.out`, and inbound rows with
`floor.cursors[k] < rowid <= current_consumed_cursor[k]` per channel `k`.

---

## 3. Snapshot Points and the Locator

A snapshot is taken **only at an outer-loop iteration boundary** — the point
where `run_role`'s residual is structurally the original `WhileStmt` /
`WhileRecvStmt` node. This is a real, recurring, identifiable point: the
interpreter forms `seq(body, whileNode)` to continue a loop, and when the body
collapses to `EmptyStmt` the residual becomes exactly `whileNode` again. That
node is regenerated identically by re-projection, so it *can* be located without
serialization.

The **locator** is a child-index path from the projected program's root
`LocalStmt` to the loop node — e.g. `[]` means the whole program is the loop,
`[1]` means "the second child of the top-level `SeqStmt`." A helper walks the
projected tree once to build a map `{id(loop_node): path}`; at snapshot time the
current residual's path is looked up; at restart the path is walked from the
freshly-projected root to recover the node. Determinism of projection makes the
path stable across processes.

Eligibility test at snapshot time: the residual is the *same object* as a loop
node recorded in that map (identity within the running process). If the residual
is anything else (mid-body, a non-loop program, already terminal), no snapshot is
taken.

---

## 4. Write Path

In `run_role`'s live loop, after a step commits, if the new residual is a
snapshot-eligible loop node, upsert the snapshot in its own transaction:

```
env_json = json(env)              # skip snapshot if this raises (best-effort)
floor    = channel.position()     # {"out": out_hwm, "cursors": {chan_key: consumed}}
locator  = loop_paths[id(residual)]
UPSERT snapshots(role, env_json, json(locator), json(floor))
```

`DurableChannel` exposes `position()` returning its committed replay floor: the
high-water mark of the role's own `put`-inserted send rowids (`out`) and a copy of
its per-channel consumed cursors (`cursors`). Because the snapshot is written
after a commit and reflects only committed cursor/send state, it is always
consistent with the log.

Cadence (v1): at every eligible boundary crossing, latest-only (the upsert
replaces the prior row). A throttle knob (min committed-events, or min seconds,
between snapshots) is a trivial later addition and is intentionally omitted now.

---

## 5. Restart Path

At `run_role` startup, before building the replay window:

1. Load the role's snapshot row, if any.
2. **Validate it** (all must hold, else discard and fall through):
   - the `locator` path resolves to a `WhileStmt`/`WhileRecvStmt` node in the
     freshly-projected program;
   - `env` and `floor` decode from JSON;
   - `floor.out` ≤ this role's current max own-send rowid, and every
     `floor.cursors[k]` ≤ the current durable consumed cursor for `k` (a floor
     ahead of the log is incoherent — discard).
3. If valid: set `env` from the snapshot, set `residual` to the located loop
   node, and construct `DurableChannel(conn, role, since=floor)` so replay covers
   only the tail (own sends after `floor.out`; inbound consumed after each
   channel's `floor.cursors[k]`). Then run the normal replay→live loop.
4. If absent or invalid: `since = None`, `residual = project(wf, role)`,
   `env = seed` — i.e. exactly today's behavior. Full replay from seed.

Because env is restored to the boundary state and only the tail is replayed, the
pre-snapshot acts are never re-run, which is the whole point. Correctness is
unaffected: the tail events re-executed against the restored env reconstruct the
exact crash-point state, identical to what a full replay would produce.

---

## 6. `DurableChannel` Changes

Two additions, both backward-compatible:

1. **`position() -> dict`** returns the current replay floor:
   `{"out": <max rowid this channel has live-`put`-inserted>, "cursors": <copy of
   the per-channel `_consumed` dict keyed by `chan_key` string>}`. The channel
   tracks `out` as it inserts sends (the max `lastrowid` from `put`).

2. **`__init__` gains an optional `since: dict | None = None`.** When `None`
   (default), `_load_replay` behaves exactly as today (all history). When a floor
   dict is given, it builds the replay queues from the tail only:
   - outbox: `sender=role AND kind IN ('msg','ctrl') AND rowid > since["out"]`
   - inbox per channel `k`: consumed rows with `since["cursors"].get(k,0) < rowid
     <= consumed_cursor[k]`

`since = None` reproduces current behavior exactly, so all existing callers and
tests (including the whole deployable-runtime suite) are unaffected. Nothing else
changes; live mode (`try_get` floor, `put` INSERT, `commit_txn`/`rollback_txn`) is
untouched.

---

## 7. Testing

A long two-role `while`-loop workflow (a genuinely multi-iteration protocol):

- **Snapshot appears at boundaries.** After running, the `snapshots` row exists,
  with `at_rowid` at a loop boundary and a resolvable locator.
- **Restart-from-snapshot replays only the tail and matches.** Restart a role
  from its snapshot; assert its final env equals a full-from-seed replay's final
  env, and that the number of events *replayed* (reconstructable from
  `since_rowid`) is bounded by one iteration, not the whole history.
- **Crash a few iterations after a snapshot resumes correctly.** Inject a crash
  N iterations past the last snapshot; the restart (snapshot + tail replay)
  reaches the same final state as an uninterrupted run, no duplicated sends.
- **Stale/corrupt snapshot falls back to full replay.** Hand-write a snapshot
  with an unresolvable locator (or bad env); assert `run_role` discards it,
  replays from seed, and still succeeds. This proves the cache-not-dependency
  property.
- **No-snapshot path unchanged.** The existing durable-runtime suite (the
  `since_rowid=0` path) stays green.

---

## 8. Out of Scope (named, not solved)

- Serializing continuations / IR / action objects — ruled out by design.
- Snapshots at nested-loop or mid-iteration points.
- Snapshot cadence throttling knobs (trivial follow-up).
- Everything the parent spec deferred (external-effect exactly-once, `kpar`, CPL
  monitor wiring, reactive multi-instance).

## 9. Open Questions

1. **Roles with no outer loop.** A finite role simply never becomes eligible and
   never snapshots — correct and needs no special case. No reactive role in scope
   lacks a locatable outer loop.
2. **`out` high-water on restart.** After resuming from a snapshot, the channel's
   `out` must continue from the actual max own-send rowid in the log (not from
   the snapshot's `floor.out`), so later `position()` calls stay correct. Seed
   `out` from `SELECT MAX(rowid) WHERE sender=role AND kind IN ('msg','ctrl')` at
   construction. Settle in planning.
