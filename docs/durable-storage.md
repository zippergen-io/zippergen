# Durable storage

How a ZipperGen workflow survives a crash, and why the design is a consequence
of the MSC formalism rather than a separate mechanism bolted on top.

This document is the reference of record for the durable store. It states the
invariants the implementation relies on, names where each one is enforced, and
explains what an operator must do when the file grows.

Everything below describes one SQLite file. A durable run owns one. A
deployment owns one.

---

## 1. What the store holds

### 1.1 The `events` table

One append-only table carries the transport, the replay log, and the
observation stream. Nothing in it is ever updated in place.

```sql
CREATE TABLE events (
  rowid        INTEGER PRIMARY KEY,
  sender       TEXT NOT NULL,
  receiver     TEXT,
  channel      TEXT,
  kind         TEXT NOT NULL,
  payload      BLOB,
  causal_stamp BLOB
);
```

| `kind` | Written by | Meaning |
|---|---|---|
| `seed` | `role_runner.seed_env` | A role's initial inputs. Written once, never collected. |
| `msg` | `DurableChannel.put` | A projected send, `send A(x) -> B`. |
| `ctrl` | `DurableChannel.put` | A control message, the branch decision the paper writes as κ_ctrl. |
| `act` | `DurableChannel.record_act` | The result of a non-deterministic action: an LLM call, an assistant call, a human answer, an effect. |
| `decision` | `DurableChannel.record_decision` | An owner evaluating its own guard. |
| `trace` | `store.record_trace_event` | Diagnostics. Never read during recovery. |

`payload` is JSON. `causal_stamp` is JSON holding the vector clock and, when a
run is monitored, the formula views.

> **Note.** There is no separate row kind for `@effect`. An effect's result is
> a non-deterministic outcome like any other, so it is recorded as `act` and
> replayed by the same rule (I7).

### 1.2 Side tables

| Table | Key | Holds |
|---|---|---|
| `cursors` | (role, channel key) | The highest `rowid` this role has durably consumed on that channel. |
| `snapshots` | role | `env`, `locator`, `floor`, and optional `monitor`. Latest only. |
| `recovery_high_water` | role | Monotonic `out` and `journal` marks that survive compaction. |
| `execution_states` | role | Non-sensitive diagnostic position for `zg inspect`. |
| `human_tasks` | task id | A pending or answered human action. |
| `human_task_tokens` | token | A durable random credential for answering a task from outside the CLI. |
| `human_task_notifications` | (task, channel, target) | Which adapter told whom, so a provider reply resolves back to a task. |
| `adapter_state` | key | Connector bookkeeping, for example a mail cursor. |
| `workflow_results` | workflow | The value a completed workflow returned. |
| `maintenance_state` | key | Counters driving online trace pruning. |

A channel key is the string `sender|receiver|channel`.

---

## 2. The write protocol

**I1. One file, one total order.**
Every write goes through a single SQLite file, so `events.rowid` is a global
total order consistent with causality. This is the property the rest of the
design spends. It is what lets a receive-any construct pick a deterministic
winner (`DurableChannel.try_get_any` takes the smallest head rowid), and it is
what makes "in the past of both endpoints" a comparison of two integers.

**I2. A step is atomic.**
Every live step runs inside `BEGIN IMMEDIATE ... COMMIT`. The rows a step emits
and the consume-cursor advances it makes commit together, or neither does.
Consumption is tentative in memory until `DurableChannel.commit_txn` writes the
cursors in the same transaction.
*Enforced in* `role_runner.RoleRunner.run_live`, `DurableChannel.commit_txn`,
`DurableChannel.rollback_txn`.

This is the whole reason replay is exact. A crash can never leave a message
delivered but unconsumed, or consumed but its consequences unwritten.

**I3. The write lock is taken up front.**
`BEGIN IMMEDIATE`, never a deferred `BEGIN`. A receive step reads with a
`SELECT` before it writes its cursor. A deferred begin would make that a
read-to-write upgrade, which SQLite fails immediately with `database is locked`
and does **not** route through `busy_timeout`. Taking the write lock first
serializes cleanly.
*Enforced in* `role_runner._begin_immediate`.

**I4. Durability is stated, not inherited.**
`journal_mode=WAL` and `synchronous=FULL` are set explicitly on every
connection. `FULL` is usually SQLite's default, but it is a per-connection
setting whose compile-time default varies, so the contract is written down
rather than assumed.
*Enforced in* `store.open_store`.

**I5. The store is owner-private from creation.**
The file is created with `O_CREAT, 0o600` before SQLite first opens it, and the
`-wal` and `-shm` files are tightened too. It holds workflow data and human
approval tokens, so it is never briefly world-readable while a umask is
consulted.
*Enforced in* `store.open_store`.

**I6. A seed is written once.**
A role's initial inputs are inserted under `kind='seed'` only when absent. On
restart the recorded seed is returned, not the caller's arguments. A resumed
run therefore cannot silently start from different inputs.
*Enforced in* `role_runner.seed_env`.

**I7. A non-deterministic result is journalled before it is applied.**
`record_act` inserts the row but deliberately does **not** advance the journal
cursor. A second pass consumes it and applies it to the environment. A crash
between the two leaves the result recorded and unconsumed, so replay applies it
without calling the model again.
*Enforced in* `DurableChannel.record_act`, `role_runner.RoleRunner.run_live`.

This is the invariant that makes an expensive LLM call at-most-once in effect.

**I8. An owner's decision is recorded and consumed in one step.**
`record_decision` advances the cursor immediately, because the value came from
the role's own environment rather than from the outside world. There is nothing
to lose by re-deriving it, and nothing to gain by a second pass.
*Enforced in* `DurableChannel.record_decision`.

---

## 3. Recovery

Recovery is replay, not repair. A role re-executes its own local program from a
known point, with the outside world served from the log.

**I9. Replay reserves, it does not re-emit.**
While a role is replaying, `put` pops the recorded send and returns its
original `rowid` without inserting anything. `try_get` serves recorded rows from
the rebuilt replay queues. `consume_journal` returns the recorded outputs
instead of calling the model or asking the human.
*Enforced in* `DurableChannel.put`, `.try_get`, `.consume_journal`.

**I10. Divergence is loud.**
Any mismatch raises `ReplayMismatch`: a different send target, a different send
payload, a different journal kind or locator, a different input hash. The run
stops rather than writing over committed state.
*Enforced in* `DurableChannel.put`, `.consume_journal`.

This is the invariant that catches you editing a workflow under a live store.

**I11. Journal replay matches by locator, not by position, when branches run
concurrently.**
Inside a `parallel` region the order in which branches become enabled can differ
between the original run and the replay. Strict FIFO matching would call that a
divergence and stop a healthy run. So runtime replay passes `strict=False`: a
recorded row is matched by kind, locator, and input hash anywhere above the
replay floor, and `_journal_seen` keeps each row consumable at most once. Strict
FIFO is retained for non-parallel reasoning and for the unit tests.
*Enforced in* `DurableChannel.consume_journal`, called from `runtime`.

This is why an `act` row carries its locator and an input hash rather than
relying on arrival order.

**I12. A snapshot is a cache, never a source of truth.**
Writing one is best effort and swallows a non-serializable environment or a
transient lock error. Deleting every snapshot costs a full replay from seed and
nothing else.
*Enforced in* `role_runner._maybe_snapshot`.

**I13. A snapshot is only trusted at a loop node.**
On resume the locator is resolved against the projected program and must land on
a `WhileStmt` or a `WhileRecvStmt`. Anything else falls back to full replay from
seed.
*Enforced in* `role_runner._try_resume`.

This is not a heuristic. See section 4.

**I14. A floor is coherent on all three axes or not at all.**
A floor carries `out`, `cursors`, and `journal`. None may point past the
committed log. A floor missing `journal` predates journalling and is rejected
outright.
*Enforced in* `role_runner._floor_coherent`.

**I15. Snapshots are taken exactly at loop-iteration boundaries.**
After a step commits, if the new residual is (by object identity) a loop node of
the projected program, the role checkpoints its environment and its position
there.
*Enforced in* `role_runner.RoleRunner.run_live` via `loop_paths`.

### What a restart actually does

1. Open the store. Re-run the schema, which is idempotent.
2. `seed_env` returns the recorded seed (I6).
3. `_try_resume` loads the snapshot, checks the floor (I14), resolves the
   locator (I13). Success gives `(env, residual, since)`. Failure gives the
   whole program and `since=None`, meaning full replay.
4. `DurableChannel.__init__` rebuilds the replay queues from the log tail above
   the floor. With `since=None` the floor is zero and the tail is all history.
5. `replay_committed` steps the role until the replay queues drain. No
   transactions, no traces, no outside calls.
6. `run_live` takes over at the exact boundary where the log ran out.

---

## 4. Why MSCs make this sound

The store is small because the formalism did the hard part first. Five
consequences, in order of how much work each saves.

**Projection makes a lifeline checkpointable on its own.**
π_A(P) yields one local program for lifeline A, and a role checkpoints it with a
single `(env, locator)` pair. There is no distributed snapshot protocol here, no
Chandy-Lamport marker algorithm, no global barrier. Each role checkpoints
itself, alone, whenever it likes. This is the single largest simplification in
the system and it is a gift from the projection.

The first-class `parallel` operator complicates this, and the design answers it
cleanly. Inside a `ParallelLocalStmt` a role has several branches in flight at once,
so its position is a *set* of locators, not one. That is why `execution_states`
stores `locators` in the plural. A snapshot, however, stores one locator, and
stays correct because of I15: a checkpoint is taken only when the residual *is*
a loop node, and the residual is the role's entire remaining program. A role
inside a parallel region has a `ParallelLocalStmt` as its residual, so it does not
checkpoint there. One locator is enough precisely because the moments it is
written are the moments nothing else is in flight.

**The grammar supplies the checkpoint set.**
A local program's only re-entry points are its syntactic `while` nodes. That is
why I13 refuses any snapshot whose locator resolves elsewhere. The set of legal
checkpoints is *derived from the grammar*, not chosen by judgement. A reader can
verify the rule by looking at the local syntax in the paper:

```
S ::= ... | while c@A do S | while A(y) <- B do S
```

Nothing else can be re-entered, so nothing else can be a resume point.

**Control messages make branch decisions replayable across roles.**
A decision is evaluated once by its owner and broadcast to exactly the
participation set L(P). Recipients receive it as a `ctrl` event on an ordinary
FIFO channel with a reserved tag, so it is logged, ordered, and replayed by the
same machinery as user messages. No role ever re-decides. Replay determinism
across roles comes free, without consensus.

**Compaction is a consistent cut, provably.**
An event is collectable only when the sender is past it and the receiver has
consumed it (I16). That is precisely the MSC condition that the event lies in
the past of both endpoints. So the surviving log is always a valid MSC prefix,
and the floors define a consistent cut across the whole chart. The safety
argument is one line of the formalism, not a case analysis over interleavings.

**Deadlock-freedom carries to recovery.**
Corollary 3.1 gives deadlock-freedom by structural induction on the global
program. Replay re-executes those same local programs against a log that is a
valid prefix of a real execution. A resumed run therefore cannot deadlock
either. Recovery inherits the theorem instead of needing its own.

---

## 5. Growth, compaction, pruning

### 5.1 What grows and what does not

**Traces prune themselves.** Every 1,000 trace rows written, the store keeps
only the newest 10,000 and resets the counter. Constants live in `store.py` as
`TRACE_RETENTION_BATCH` and `TRACE_RETENTION_KEEP`. You do not manage this.

**Everything else grows without bound** until you compact. A long-running
mailbox service accumulates one `msg` per send, one `act` per model call, and
one `decision` per guard, forever.

### 5.2 Compacting

```bash
zg deploy compact
```

If the deployment is running, `compact` stops it first and restarts it when it
is done, because deleting rows and then vacuuming contends with live roles.

It runs an integrity check before touching anything and refuses to proceed if
that fails. It then deletes, inside one transaction:

- a `msg` or `ctrl` row when `rowid <= sender.floor.out` **and**
  `rowid <= receiver.cursors[channel]`;
- an `act` or `decision` row when `rowid <= sender.floor.journal`.

Then it checkpoints the WAL with `TRUNCATE` and runs `VACUUM`.

**I16. An event is collectable only when it is in the past of both endpoints.**
*Enforced in* `storage_maintenance._collectable_counts` and `compact_store`,
which recomputes the same predicate inside the write transaction rather than
trusting the plan it showed you.

**I17. A role with no snapshot blocks collection.**
If the sender has no floor, nothing it sent is collectable. If the receiver has
no floor, no message to it is collectable. `inspect_store_storage` reports these
as `roles_without_snapshot`.

**I18. Seed rows are never collected.** Full replay from seed must stay possible
for any role whose snapshot turns out to be unusable (I13, I14).

**I19. Rowids are stable across `VACUUM`.** `events.rowid` is an explicit
`INTEGER PRIMARY KEY`, so vacuuming preserves the identifiers that floors and
cursors point at. A plain `rowid` table would renumber and silently corrupt
every floor.

**I20. High-water marks survive their events.** `backfill_recovery_high_water`
runs inside the compaction transaction *before* the delete, recording each
role's maximum `out` and `journal` in `recovery_high_water`. Floors therefore
stay checkable after the events they refer to are gone.

### 5.3 The two things that surprise people

**Nothing is collectable in a workflow without a loop.** Snapshots are written
only at loop boundaries (I15), and no snapshot means no floor (I17). A long
straight-line workflow compacts to zero. This is correct, not a bug: without a
loop node there is no sound resume point, so the whole log is still needed. If
you want a service to compact, make it non-terminating with a loop, which is
what the tutorial workflow does.

**Deleting rows does not always shrink the file.** SQLite frees whole pages. A
real run on a small store:

```
before:  {'act': 1, 'msg': 1, 'seed': 2, 'trace': 5}
plan:    messages=1 journal=1 blocked=()
deleted: 1 message, 1 journal
bytes:   155648 -> 155648
after:   {'seed': 2, 'trace': 5}
```

Two rows went away and the file did not move, because there was not a full free
page to release. Judge compaction by `removed events`, not by bytes, until the
store is large.

### 5.4 Looking before you act

```bash
zg deploy status      # events, last event, pending human tasks, results
zg deploy check       # includes the SQLite integrity check
```

`plan_store_compaction` gives the dry-run counts and the blocking roles, and is
what to reach for from Python when diagnosing a store that will not shrink.

### 5.5 Starting over

```bash
zg deploy reset
```

Replaces the store with an empty one and keeps an archive. It checks the
service state, and restarts the service afterward if it was running. Use it
when the state is wrong, not when the file is merely large.

---

## 6. Notes for the reader of the code

**`@effect` results are recorded as `act`.** There is no separate `effect` row
kind and there never was. An earlier schema comment listed one, which was
misleading to anyone auditing the on-disk format, so it is gone.

**Compaction manages the service for you.** `zg deploy compact` stops a running
deployment, compacts, and restarts it, the same lifecycle `zg deploy reset`
uses. Before that it vacuumed under the live service and then hit the log
rotation's own running-service refusal, leaving the command half done.
