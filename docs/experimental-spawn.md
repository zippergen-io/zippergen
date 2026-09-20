# Experimental spawn

`parallel` runs concurrent branches within one session. `spawn` starts another
session while the current one continues. Both use the same participants.

```python
with spawn(fresh=(Mailbox.request, Writer.request, Writer.draft)):
    Mailbox(request) >> Writer(request)
    Writer: draft = draft_reply(request)
```

There is no initiating participant, handle or join. A child can continue after
its parent finishes, including beyond an enclosing parallel region. Existing
parallel joins remain participant-local.

## State and communication

Every listed field gets a new participant-local cell. Its value is copied at
that participant's local fork. A field without an initialized value stays
uninitialized. Other fields keep referring to the same participant-local cell.
This copy is not a snapshot taken at one global instant.

`zg validate` warns when a spawned body assigns a field omitted from `fresh`,
including message receive targets. The warning names the participant and field.
For example, selecting `Mailbox.draft` does not silence a warning about
`Writer.draft`. Add fields that belong to one request to `fresh`.

Intentional shared accumulators are allowed and still produce a warning. There
is no separate sharing annotation or suppression option yet. These checks cover
explicit assignments, not reads of changing parent state or mutations hidden
inside Python actions. A warning-free workflow is not a proof of race freedom.

An outer intake loop can therefore copy `Mailbox.request` and overwrite its
own request for the next iteration. Other participants learn the request
through messages. Their local forks need not happen at the same time.

Nested children inherit their parent's cell bindings unless they explicitly
freshen them. An inherited cell remains available after its allocating session
has finished. Completed session controls are removed and cells are retained
while a remaining session refers to them.

Logical routes include the session and the parallel branch namespace. Messages
for different sessions cannot be consumed by one another. Generated decision
variables also have separate storage in each session.

## Projection and execution

Each involved participant receives a local fork instruction with the same
static source-tree location. The local child name combines parent session,
spawn location and visit counter. Enclosing decisions include participants
that occur only in a spawned body. They therefore see the choices needed to
reach the same dynamic spawn visit.

The runtime keeps a control position and counters for each local session.
It tries sessions in round-robin order. A receive that cannot proceed leaves
other sessions runnable. Events at a participant remain sequential.

The durable human-task backend publishes a request, commits it, and resumes
other enabled work. An answer completes that same request later. Task identity
includes the session and dynamic position. CLI approval and Telegram delivery
use the existing task store.

Durable runs and deployed services use that task queue, including workflows
with pools and no human connector. Use `zg run tasks` / `zg run approve` for a
durable run, or `zg deploy tasks` / `zg deploy approve` for a deployment. A plain
local run without a human connector continues to prompt in the terminal.

Ordinary external effects, model calls and terminal human prompts remain
synchronous at their participant. They can delay its other sessions until they
return. Spawn does not change the retry or idempotency guarantees of connectors.

The existing finite-run supervisor waits until all participants have exhausted
all their sessions before collecting workflow outputs. That is the lifetime of
the run, not a join inserted into the parent's program. A service or durable
run stopped with pending children resumes them from its store.

## Persistence and compatibility

The existing role-state transaction stores a versioned session registry,
control positions, copied cells, binding references, fork counters, scheduler
cursor and pending human requests. Child creation and parent advancement
commit together. A crash before commit retries the same fork. A crash after
commit resumes its existing child.

No database schema migration is needed. Workflows without spawn retain their
existing projection, fingerprints and control encoding. Spawn workflows use
source-tree locations for control tags and parallel namespaces so identical
fragments at different sites remain distinct.

Changing a workflow that already has durable sessions still requires the
existing deployment compatibility checks. Do not resume its store using an
older ZipperGen version that cannot decode session controls.

Source-tree locations depend on the program structure. Inserting a statement
can change later spawn sites and invalidate saved control positions, even when
the edit looks small. The fingerprint check rejects an incompatible resume.
Restore the previous workflow to continue its pending work, or explicitly reset
only after deciding what to do with that work and its external effects.

## Try it

See [the draft-approval example](../examples/spawn/specification.md). It creates
three approval sessions with fixed drafts and no external services. Durable
execution exposes all three tasks so they can be answered in any order.
`zg show` displays the spawn body globally and the local forks in projections.
History events include a session ID and message traces use the actual scoped
route. Position inspection shows all active local program points, which may
repeat when several sessions are at the same instruction.

## Current limits

- The implementation is experimental. Tests exercise the proposed semantics
  but do not constitute the paper's projection and progress proof.
- CPL guards are rejected in workflows containing spawn. Cross-session causal
  queries have not been assigned a semantics.
- There is no join, wait, cancellation, participant creation, session limit or
  automatic backpressure. Bound intake while evaluating it.
- The store remains local to the existing service. Spawn does not add remote
  transport or isolation between administrative domains.
- External effects retain their existing replay window. Copying a work-item
  identity does not atomically claim that item in an external mailbox or queue.
