# Durable storage

A ZipperGen deployment keeps its state in one SQLite file. That file holds the
**current state of the computation**, not a history of how it got there.

Recovery is: read the state, continue. There is no log to replay, no snapshot to
validate, no compaction, and nothing whose correctness needs an argument.

You should be able to understand the whole model from the schema below, eight
invariants, and the twenty-line loop in `role_runner.py`.

---

## 1. The state

```sql
role_state            -- one row per lifeline: its whole position
  role, env, control, monitor, steps, status, detail, updated_at

outstanding_messages  -- sends nobody has absorbed yet
  id, sender, receiver, channel, payload, causal_stamp

human_tasks           -- questions that outlive the process
human_task_tokens     -- credentials for answering from outside
human_task_notifications

adapter_state         -- connector bookkeeping
workflow_results      -- what a finished workflow returned
store_meta            -- schema version, workflow identity, operator settings

history               -- optional. Never read by recovery.
```

That is all of it. Nine tables. The first two are the projected lifelines'
computation; pending `human_tasks` also participate in recovery, while their
tokens and notifications are supporting audit and delivery records.

- `env` is the role's variables, as JSON.
- `control` is where the role is in its program. Section 2.
- `monitor` is CPL monitor state, which includes the vector clock.
- `steps` counts committed steps. It is what tells two visits to the same
  human action in a loop apart.
- `status` and `detail` feed the run and deployment observation commands,
  including `status` and `inspect`. Recovery ignores them.

**The durable state is `role_state` plus `outstanding_messages` plus
`human_tasks`.** Everything else is either derived or optional.

## 2. Control state

A role's position is the residual program it has left to run. Most residual
nodes are nodes of the static projected program, so a child-index path names
them exactly. The interpreter builds only three shapes fresh, so the whole
control language is five constructors:

| Encoded | Meaning |
|---|---|
| `{"k":"done"}` | nothing left to run |
| `{"k":"at","p":[1,0]}` | run the static statement at this path |
| `{"k":"seq","a":…,"b":…}` | run `a`, then `b` |
| `{"k":"par","b":[…],"i":[0,1]}` | a parallel region, one control per branch |

It is closed under one interpreter step. That closure is the reason storing the
current state is sufficient and replay is unnecessary.

Real examples, straight out of a store:

```
A finished role:      {"k": "done"}
A role at the start:  {"k": "at", "p": []}
Mid-sequence:         {"k": "seq", "a": {"k":"at","p":[0]}, "b": {"k":"at","p":[1]}}
```

### Parallel

Inside `parallel` a lifeline has **one position per unfinished branch**, so a
single locator is not enough. The `par` constructor holds one control per
branch, and a finished branch is `done`. Join is implicit: the region is over
when every branch is `done`.

## 3. The loop

From `role_runner.py`:

```
while there is work left:
    BEGIN IMMEDIATE
    take one interpreter step
    if it needs the outside world:
        ROLLBACK, call out with no transaction open,
        then BEGIN and commit (result + next control state) together
    elif it progressed:
        delete the messages it consumed,
        compare-and-swap the new role state by committed step count, COMMIT
    else:
        ROLLBACK and wait
```

**I1. One commit writes a role's whole position.** Variables, control state and
monitor go in one row, in one transaction. There is no second place to look and
no way for them to disagree.

**I2. Consuming a message and advancing the receiver are the same
transaction.** A receive takes the lowest-id row on its route, and that row is
deleted in the same commit that advances the receiver. If the transaction rolls
back, the message is still outstanding and nothing was lost.

**I2b. Emitting a message and advancing the sender are the same transaction.**
The mirror of I2, and what makes "a send is never duplicated" true:

```
not committed:  no message,  old sender position
committed:      message,     new sender position
```

**I3. No transaction is open across the outside world.** An LLM call, a human
question, a network effect: the transaction is dropped first, the call is made,
and a second short transaction commits the result together with the successor
control state.

**I4. The write lock is taken up front.** `BEGIN IMMEDIATE`, never a deferred
`BEGIN`. A receive reads before it writes, and a deferred begin makes that a
read-to-write upgrade, which SQLite fails immediately instead of routing
through `busy_timeout`.

**I5. Durability is stated, not inherited.** `journal_mode=WAL` and
`synchronous=FULL` are set explicitly on every connection, and the file is
created `0600` before SQLite opens it.

**I6. History is never read by recovery.** Progress events share the state
transaction and are lost when it rolls back. An outside-world action's start,
retry, and best-effort failure events are recorded with no transaction open,
because that is also when the call occurs. Its successful completion shares
the transaction that advances control. Either way, history is observation
rather than state. There is a test that deletes all of it and then resumes.

**I7. A stale runner cannot commit.** A runner loads a role position and its
committed `steps` count together. Every later state write includes
`WHERE steps = <loaded value>`. If another supervisor advanced that role, the
update affects no row and the stale transaction rolls back, including any send
or receive it attempted. This turns accidental concurrent execution into a
loud ownership error instead of duplicated protocol work.

The compare-and-swap is a consistency check, not an ownership lease. Normal
execution also holds the project's single-supervisor lock. If two supervisors
are started by bypassing that CLI boundary, both can reach the same external
call before one loses the later compare-and-swap. The stale result cannot enter
durable state, but the remote call may already have happened twice. The crash
guarantees below therefore assume one supervisor per project, which is the
execution model the CLI enforces.

### A guard decides; it does not compute

The loop above drops the transaction before every external action, and that is
the whole reason an LLM call or a connector request cannot hold the write lock.
A condition has no way to ask for the same treatment: deciding a branch is part
of taking one step, so it is evaluated between `BEGIN IMMEDIATE` and `COMMIT`.

So a guard must decide from values the workflow already has. `@workflow` refuses
one that calls anything:

```python
Mailbox: has_mail = mailbox_has_mail()   # an action -- outside the transaction
if has_mail @ Mailbox:                   # a guard -- inside it, and cheap
```

`if mailbox_has_mail() @ Mailbox:` is rejected at build time. A deployed mailbox
poller written that way made a Gmail request inside the transaction on every
cycle, holding the write lock for its duration and blocking all four
participants. Causal-past formulas — `At[A](phi)`, `Y(phi)` and the rest — are
still allowed: they build a formula the monitor evaluates from state the runtime
already holds, and reach nothing outside.

## 4. Crash guarantees

One rule covers everything:

> The committed role state describes what is known to have completed. Whatever
> the control state points at has **not** necessarily completed, and may run
> again after a crash.

| What | Guarantee |
|---|---|
| **A local step** | Reruns if its state did not commit. Deterministic, so no visible difference. |
| **A message** | Never lost and never consumed twice. Consumption and advancement are atomic (I2). |
| **A send** | Never duplicated. The sender's control state is past it once committed. |
| **An LLM call** | May be called again if its result did not commit. That can cost money and can give a different answer. |
| **An `@effect`** | May run again if its result did not commit, **including its outside-world side effect**. An email can be sent twice. |
| **A human task** | One durable task identity, surviving restart, accepting one answer. The *notification* that announces it may be sent more than once. |

The LLM and effect guarantee is deliberate, and it is the honest one. The
window is real: the provider may have completed the request while ZipperGen
died before recording the answer. No amount of local bookkeeping closes it. An
earlier design wrote a journal row and claimed at-most-once; that claim was
false, because the journal row is also written after the call returns.

We do not claim at-most-once or exactly-once for anything external. If you need
it, the outside system has to offer idempotency, and today ZipperGen does not
thread an idempotency key through `@effect`.

### Retries live in memory, not in the store

An `@llm` action may declare `retries=` and a `fallback=`. The whole attempt is
retried -- request, parse, coercion and the declared output types -- until it
produces a usable answer, exhausts the budget, or hits a permanent failure. The
role's control state does **not** advance while this happens. A fallback, when
one is declared, commits exactly like any other action result: one transaction,
one step forward.

Durable coordination values use one tagged JSON encoding, so tuples and lists
remain distinct across role state, causal field views, outstanding messages,
workflow results, managed-run records, and deployment profiles. The files and
SQLite columns remain ordinary JSON; the tags carry only the container type
information that plain JSON lacks. Non-finite floats, cycles, non-string
dictionary keys, and unsupported Python objects are rejected before they cross
a durable boundary.

The counter is a local variable on the stack of the invocation, deliberately.
Nothing about retrying is written to the store: no event log, no attempt rows,
no schema change. That buys a simple, inspectable store, and it costs one
honest limitation:

> **A finite retry budget starts again if the process itself crashes.** An
> action declaring `retries=3` that fails twice, crashes, and resumes will make
> up to four further attempts. The budget bounds one invocation, not the total
> work done across a crash.

This is the same boundary every external action already has, and for the same
reason: the only way to bound attempts across a crash is to record them before
making the call, which is another write that can itself be interrupted.
Cancellation is separate and exact -- a stopped run interrupts the wait
immediately and never takes the fallback, because stopping is not failing.

### Why human actions are different

A human question can sit unanswered for days while nothing is running. So it
has durable identity of its own: a `human_tasks` row with a stable id derived
from the role, the position, the inputs, and `steps`. A restarted role
recomputes the same id and re-finds its pending question instead of asking
twice. `complete_human_task` only writes over a `pending` row, so a second
answer cannot overwrite the first.

`steps` is the count of committed steps, and it is there because position alone
is not enough:

```python
while ...:
    ask_human(...)      # same statement, same inputs, two real questions
```

That is the one place a durable action record is genuinely necessary, and it is
deliberately not generalised to model calls or effects, which do not outlive
the process.

**Sending the notification is a separate, weaker thing.** Telling someone about
the task over Telegram, email or Slack is an ordinary external effect with the
same window as any other:

```
provider accepted the notification
crash
the notification row never committed
```

so on restart the notification can go out again. `human_task_notifications`
records `(task, channel, target)` and an optional provider message id, which
lets a reply be resolved back to the task and lets an adapter recognise what it
already sent — but unless the provider offers idempotency, a duplicate
notification is possible. The task itself is still asked once and answered
once; only the announcement may repeat.

## 5. CPL and causal state

Removing the log did not remove what CPL needs, because CPL never needed the
past — it needs the current summary of it.

- The monitor's state, vector clock included, is a column on `role_state`. It
  commits with everything else, so it can never drift from the variables.
- The snapshot names every subformula by a deterministic semantic fingerprint,
  not its display text or a callable representation. Python cannot generally
  determine whether a predicate's helpers, globals, or external dependencies
  still mean the same thing, so low-level durable atoms require
  `atom(..., version="descriptive-v1")`. Change that identity whenever the
  predicate's meaning changes. Structural field-term formulas assign their
  identity automatically.

Both sides are atomic, which is what makes the picture local:

```
send:     sender's new state  +  the stamped outstanding message   (one commit)
receive:  stamped message + receiver's state -> receiver's new state,
          and the message is deleted                               (one commit)
```

A crash can never leave a stamped message whose sender did not record the
event, nor a receiver that absorbed a stamp while the message survives. Both
directions have a test.

The model in one line: **the relevant past lives in the current causal state;
the message table is communication not yet absorbed.**

This is also a closer fit to the MSC semantics than the old design was. A
channel is exactly its outstanding sends, and a lifeline is exactly its local
state — which is what the chart says in the first place.

## 6. Workflow identity

Control state is child-index paths, so resuming under changed code would mean
something else. `store_meta` records the workflow name and a structural
fingerprint of the projected programs, checked once at startup.

Three cases, kept distinct:

- **Restart of the same deployment** — fingerprints match, resume.
- **A fresh start** — `zg deploy reset`, new store, new claim.
- **An incompatible edit** — refused with a clear error naming `zg deploy reset`.

### Nothing is written until the store is identified

Opening a store reads what it is before changing anything about it. Both the
journal mode and the file permissions are persistent properties, so an
installation that merely looked at a store written by a newer ZipperGen would
leave it altered — and an installation cannot acquire that restraint later,
only ship with it.

Identification has three outcomes and no others:

- **no tables** — a store about to be created, safe to configure;
- **a stated version** — compared, and a newer one refused;
- **it cannot be read** — refused, because a locked store and a newer store are
  indistinguishable from here and only one of them is safe to modify.

That last case is why database errors are not swallowed. A locked store reads
as "no version", and treating that as "nothing to protect" is how a newer store
came to be converted to WAL by a reader that then refused it.

### Version gates

The store is never migrated: a control position only means something under the
program that wrote it, so an incompatible edit is refused and you reset.

The first release applies the same strict rule to the project manifest,
workspace state, run records, and deployment profiles. There is no earlier
released format to migrate. A record with an older internal prerelease schema
is refused with replacement instructions. A record written by a newer
ZipperGen is refused with an instruction to upgrade, and is never rewritten by
the older reader. The unstamped project manifest is the one explicitly
supported initial layout.

Future releases may add an explicit metadata migration for a format that was
actually released. No empty migration table ships in anticipation of one: the
version gate remains the extension point, and the release that changes a format
must add its migration and an old-format fixture together.

For a profile that can be read, `zg deploy remove` unregisters the service and
moves the profile, durable store, and log together under
`$ZIPPERGEN_HOME/trash/deployments/`. It deletes credentials, the managed
environment, source bundle, and service files because those are either private
or rebuildable. The archive is no longer an active deployment: deploying again
creates a new store. `remove --purge` keeps no archive at all.

`tests/test_upgrade_path.py` exercises every version gate with deliberately
mismatched records. That test remains the place to add a real previous-release
fixture when a future format changes.

### What the fingerprint covers, exactly

It answers one question: **does the stored durable state still mean the same
thing to this program?** That includes both control paths and the names and
types through which statements read and write the durable environment:

| Change | Detected |
|---|---|
| a statement added, removed or moved | yes |
| a different statement kind at a position | yes |
| a different lifeline, sender, receiver or channel | yes |
| a different action kind, name or declared interface | yes |
| a renamed or retyped payload, input, output or binding | yes |
| a changed literal used by a statement | yes |
| a rewritten `@effect` or `@pure` **body** | **no** |
| a rewritten LLM **prompt** | **no** |
| a plain Python guard computing something different | **no** |
| a CPL atom's declared semantic `version` | **yes, for live monitor state** |

Action bodies, prompts, and plain Python guards change future computation but
do not change how committed state is decoded. Excluding them is deliberate:
fixing a typo in a prompt should not force a reset that throws away live state.
Expressions are different. They are part of the choreography, and a renamed
expression variable could otherwise read a default instead of the committed
value under its old name. CPL formulas are different too: their accumulated
truth values are durable state, so changing an atom's semantics must invalidate
that state rather than reinterpret it.

There is a test that pins both halves of this, so the guarantee is executable
rather than a claim in prose.

Hashing `repr(program)` instead would embed each guard closure's memory
address, which is stable inside one process and different in the next. An
earlier version did exactly that, which made every restart look like an
incompatible edit.

## 7. Growth

`role_state` has one row per lifeline. `outstanding_messages` holds only what
has not been absorbed. Neither grows with how long the workflow has run.

`history` accumulates and is pruned online. Each store records its own budget
under `history_keep` in `store_meta`; a store that says nothing keeps the newest
10,000 rows. Trimming happens in batches of a tenth of the budget, capped at
1,000 events, rather than on every write, so the table sits above budget in
between trims.

The budget is a FIFO row count, not a time window or a size. A workflow whose
events carry large values — a whole email, a long model response — will hold far
more bytes at 10,000 rows than one passing short strings. Frequent routine
events also evict older incidents exactly like important events. For example, a
four-participant poller measured at 14 events per minute fills the default in
about 12 hours even when almost every poll finds nothing.

So the trace is a **window, not an archive**, and the window is measured in
events rather than in days. A long-running deployment always has a horizon
behind which it can no longer say what happened, and a busy period moves that
horizon closer: the same incident worth investigating is also what evicts the
evidence fastest. Nothing warns you when a particular event ages out.

Two things follow for operators. First, capture a trace worth keeping at the
time, rather than expecting to find it later:

```bash
zg deploy trace --json --tail 5000 > "incident-$(date +%F).json"
```

Second, do not size the budget for normal operation. Size it so the noisiest
plausible day still leaves the window longer than the time it takes you to
notice a problem and go looking. A poller idling at 14 events per minute needs
roughly 20,000 rows per day; the same workflow under a burst of real mail needs
several times that.

Raising the budget is the only lever that reliably widens the window.
`visible=False` suppresses just that action's own start and completion events,
so it removes a small fraction of a polling loop's traffic — the surrounding
decisions, control sends, and receives are still recorded.

None of this touches recovery. `history` is never read to restore a workflow, so
an evicted event costs observability and nothing else: a deployment whose whole
trace was discarded still resumes exactly where it was.

Measure the workflow's real event rate and payload-size distribution before
choosing a value. A very large budget is not free: the periodic prune uses an
offset scan and runs on the writer path. The row ceiling also does not bound the
whole SQLite file family exactly because pages and the WAL have their own
overhead.

```bash
zg deploy --history-keep 50000       # a bigger window for a quiet workflow
zg deploy --history-keep 0           # record no trace at all
zg deploy compact --set-history-keep 2000   # change it later, and apply it now
zg run --durable --history-keep 500  # the same choice for one durable run
```

Which command writes which is fixed, so there is one place to look:

| command | what it writes |
|---|---|
| `zg deploy --history-keep N` | the budget on the profile and the current store; trims immediately |
| `zg deploy reset` | a new store, carrying the profile's budget |
| `zg deploy compact --set-history-keep N` | the budget on an existing store, and trims it |

An ordinary `zg deploy` does not open an existing store while changing
configuration. Passing `--history-keep` is the explicit exception: that option
owns the store setting, so it updates and trims the current store as well as the
profile. An incompatible store therefore does not block an ordinary redeploy;
`reset` remains the way through it.

A budget of zero writes nothing, so it costs nothing rather than writing and
deleting. Setting a budget also trims immediately: a budget nobody has reached
yet would otherwise be a promise rather than a fact, and with a budget of zero
no later write would ever trim what was already there. `zg deploy status` reports
the budget together with how much of it is in use.

The budget is recorded on the deployment as well as in the store, so
`zg deploy reset` — which archives the store and starts an empty one — does not
quietly put the trace back to the default.

Event numbers restart if a budget of zero empties the table. `history` uses a
plain `INTEGER PRIMARY KEY`, so SQLite hands out 1 again once nothing is left,
and `trace --after N` then skips new events until the counter catches up. The
highest number reached is recorded in `store_meta` under `history_high_water`
so a reader can tell that happened.

Numbering is deliberately **not** forced to continue. Doing so means choosing
the id in Python — `SELECT MAX(id)`, then inserting it — and one role per thread
writes here concurrently, each on its own connection. Two threads read the same
maximum and the second insert fails with a `UNIQUE` violation, which surfaces as
a killed lifeline in a running workflow. Only the database can hand out an id
without racing its own writers, and a restarted trace counter after a deliberate
operator action is a far smaller cost than that.

Each new history event carries a wall-clock `recorded_at` timestamp for the
human-facing trace table. That timestamp is observational: row ids and causal
stamps remain the ordering facts, and recovery never reads the timestamp.
Visible outside-world actions record an `act_start`, including a fresh
`attempt_id`, after the transaction is released and immediately before the
call. A successful `act` or best-effort `act_failed` terminal event carries the
same attempt id and a measured `duration_ms`, so the trace can pair concurrent
and restarted attempts without guessing from sequence numbers. Cancellation is
not recorded as failure. If the process dies before a terminal event commits,
the start remains unmatched; that means only **no completion was recorded**.
It does not prove whether the remote call succeeded, failed, or was still
running when the process disappeared. A static trace labels that row
`incomplete`; `zg deploy status` is the source for current live activity.

`visible=False` on a `@pure`, `@effect`, or `@assistant` is a persistence choice, not merely a
display filter: that action's own start, completion, and failure events are not
written. It does not suppress decisions or control sends and receives around
the action, and recovery is unchanged either way.

Human authority is separate from trace visibility. A `@human` action waits for
a durable answer by default. Only `kind="ack"` accepts `required=False`; that
non-blocking notice resolves to true without creating a human task and is still
recorded. Confirmations, edits, selections, and inputs cannot be auto-completed.

That automatic pruning is recovery-independent. The combined maintenance
command also rotates deployment logs, so it requires the deployment to be
stopped and refuses before changing either resource:

```bash
zg deploy stop
zg deploy compact
```

With no arguments `compact` trims history to the store's own budget. It does not
empty the store: throwing away the only record of what ran is a separate request,
made with `--set-history-keep 0`.

Dropping history is a retention choice, not a recovery operation. The explicit
stop makes the combined command predictable and its log rotation lossless.

Completed human tasks, answer tokens, and notification records are retained as
audit records. They are bounded by the number of human interactions, not by the
live computation, and currently have no automatic retention policy. This is a
deliberate visibility choice, not recovery state; operators with a formal
retention requirement should account for it when sizing and backing up the
store rather than assuming every table is bounded.

## 8. Filesystem, sensitivity, and backup

The SQLite file is local-machine coordination. Keep it on a local filesystem
with reliable POSIX locking and `fsync`; do not put a live store on NFS, a
cloud-synchronised directory, or a volume whose locking guarantees are
unclear. ZipperGen currently supports macOS and Linux for managed execution;
the execution lock uses `fcntl`, so Windows is not a supported runtime host.

The store is created owner-only (`0600`) because it can contain workflow
inputs, model outputs, message payloads, human-task context, approval tokens,
and connector bookkeeping. Do not edit rows directly. SQLite's online backup
API or its `.backup` command can copy a live store consistently; protect the
copy with the same permissions.

Managed deployment directories are owner-only (`0700`), and profiles, logs,
stores, service templates, and credential files are owner-only (`0600`). The
generated systemd and launchd services also set an owner-only umask. `zg deploy
check` reports weaker permissions; `zg deploy check --repair-permissions`
repairs managed paths without changing external files referenced by a legacy
or manually edited profile.

`zg deploy remove` owns only canonical files created under the configured
ZipperGen home, plus its installed service registration. A profile's external
`store`, `log`, or `secrets_file` reference is never sufficient authority to
move or delete that external path. `reset` and `compact` refuse external store
or log references instead of treating them as managed state.

Restoring an older backup restores an older belief about the outside world.
Any LLM call or `@effect` completed after that backup may run again on resume,
including email sends and connector writes. Review that replay window before
starting the restored store, and make irreversible effects idempotent where
the external system supports it.

## 9. What this replaced

The previous design recovered by replaying an append-only event log, with
per-role snapshots as checkpoints and a compaction pass whose safety depended
on snapshot floors. Deleted with it:

event replay, replay queues, `ReplayMismatch`, recovery from seed, snapshots,
snapshot floors, floor coherence checks, loop-only checkpoints,
`recovery_high_water` and its backfill, consumption cursors, collectability
predicates, and the stop/compact/restart lifecycle.

A store written by that version is refused at open, with instructions to reset.
There is no migration, because the old store recorded positions into a log
rather than the interpreter's state.
