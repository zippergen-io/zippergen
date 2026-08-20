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
store_meta            -- schema version, workflow identity

history               -- optional. Never read by recovery.
```

That is all of it. Nine tables, and only the first two are the computation.

- `env` is the role's variables, as JSON.
- `control` is where the role is in its program. Section 2.
- `monitor` is CPL monitor state, which includes the vector clock.
- `steps` counts committed steps. It is what tells two visits to the same
  human action in a loop apart.
- `status` and `detail` are for `zg run inspect`. Recovery ignores them.

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
| `{"k":"any","p":[…],"s":["B","C"]}` | a coregion receive with senders still pending |
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

A coregion receive is the other shape that needed care. The interpreter cannot
just shrink the static node, because the smaller node would have no path. It
keeps a reference to the static node plus the set of senders still outstanding
(`PartialReceiveAny`), so the position stays exactly representable.

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

**I6. History is never read by recovery.** It is written inside whatever
transaction is open and lost on rollback, and that is fine. There is a test
that deletes all of it and then resumes.

**I7. A stale runner cannot commit.** A runner loads a role position and its
committed `steps` count together. Every later state write includes
`WHERE steps = <loaded value>`. If another supervisor advanced that role, the
update affects no row and the stale transaction rolls back, including any send
or receive it attempted. This turns accidental concurrent execution into a
loud ownership error instead of duplicated protocol work.

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

Durable coordination values use a tagged JSON encoding, so tuples and lists
remain distinct across role state, outstanding messages, and workflow results.
The SQLite columns remain ordinary JSON text; the tags carry only the container
type information that plain JSON lacks.

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
| a guard computing something different | **no** |

The last three change future computation but do not change how committed state
is decoded. Excluding them is deliberate: fixing a typo in a prompt should not
force a reset that throws away live state. Expressions are different. They are
part of the choreography, and a renamed expression variable could otherwise
read a default instead of the committed value under its old name.

There is a test that pins both halves of this, so the guarantee is executable
rather than a claim in prose.

Hashing `repr(program)` instead would embed each guard closure's memory
address, which is stable inside one process and different in the next. An
earlier version did exactly that, which made every restart look like an
incompatible edit.

## 7. Growth

`role_state` has one row per lifeline. `outstanding_messages` holds only what
has not been absorbed. Neither grows with how long the workflow has run.

`history` accumulates and is pruned online, keeping the newest 10,000 rows.
That automatic pruning is recovery-independent. The combined maintenance
command also rotates deployment logs, so it requires the deployment to be
stopped and refuses before changing either resource:

```bash
zg deploy stop
zg deploy compact
```

Dropping history is a retention choice, not a recovery operation. The explicit
stop makes the combined command predictable and its log rotation lossless.

## 8. What this replaced

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
