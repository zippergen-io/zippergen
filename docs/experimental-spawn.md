# Experimental spawn

`parallel` runs concurrent branches within one session. `spawn` starts another
session while the current one continues. The inherited form uses the current
participants; the bound form below selects endpoints for a new session.

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

For a child whose entire protocol state should be private, use:

```python
with spawn(fresh="all"):
    Mailbox(request) >> Writer(request)
    Writer: draft = draft_reply(request)
```

This is shorthand for all participant-qualified fields read or written in the
body, including receive targets, guard inputs and nested fragments/spawns.
It expands to the ordinary explicit field list before projection. `zg show`
and semantic snapshots display that expanded list. Adding a field to the body
therefore includes it automatically. This does not copy Python globals,
credentials or external resources. To inherit selected fields while isolating
everything else, use:

```python
from zippergen import all_except

with spawn(fresh=all_except(Writer.preferences)):
    Mailbox(request) >> Writer(request)
    Writer: draft = draft_reply(request, preferences)
```

`Writer.preferences` keeps its current cell, shared with sessions inheriting
that cell. Each read sees that cell's value at the time of the action. A nested
child inherits its parent's binding, not necessarily the root session's cell.
Exclusions must name participant-qualified fields used by the child. Unused,
duplicate or malformed exclusions fail during construction. `all_except()`
with no exclusions is equivalent to `"all"`. Both forms expand to ordinary
explicit freshness lists, so they introduce no new runtime semantics.

The explicit tuple remains available. Omitting `fresh` still means inheritance.
Low-level callable guards without source metadata require an explicit field
list because the builder cannot infer their reads safely.

`zg validate` warns when a spawned body assigns a field omitted from `fresh`,
including message receive targets. The warning names the participant and field.
For example, selecting `Mailbox.draft` does not silence a warning about
`Writer.draft`. Add fields that belong to one request to `fresh`.

Intentional shared accumulators are allowed and still produce a warning,
including writes to fields excluded with `all_except`. The selector does not
suppress warnings. These checks cover
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

A short SQLite-backed polling effect may raise `EffectPending(retry_after=1)`
when no data is ready. Its continuation stays at the same action while other
sessions and parallel branches can run. See [polling effects](polling-effects.md).
This does not make arbitrary long-running calls asynchronous.

The existing finite-run supervisor waits until all participants have exhausted
all their sessions before collecting workflow outputs. That is the lifetime of
the run, not a join inserted into the parent's program. A service or durable
run stopped with pending children resumes them from its store.

## Dynamic role bindings

Keep a named `@fragment` as the static protocol and choose its endpoints in an
owner-local value:

```python
Manager: bindings = select_reviewer(request)
spawn(AccessReview, bindings=bindings @ Manager,
      retain=(Manager, IT), fresh="all")
```

The binding is a JSON mapping such as
`{"Manager": "Manager", "Reviewer": "reviewer-east", "IT": "IT"}`.
It must name exactly the child roles, with distinct endpoint IDs. Retained
roles keep their parent addresses. Omitted `retain` means no retained roles
in this form. Omitting `bindings` keeps ordinary inherited spawn.

Register additional endpoint IDs with `endpoints=` when calling `run` or
`run_sqlite`. Root roles retain their same-name endpoints. An endpoint can
host different roles in different concurrent sessions, but cannot play two
roles in the same session. The current backend hosts these workers in the
existing local runtime/store; it does not add network deployment or independent
administrative stores. Registration and selection do not create remote agents.
Additional endpoint registration is currently a Python runner API; `zg run`
and deployment configuration have no corresponding registry option yet. Use
the executable example below to run recruited endpoints. Selecting an
unregistered endpoint fails explicitly.

Use `endpoint_backends={"reviewer-east": {"llm": review_backend}}` to select
backends for a particular endpoint. Supported keys are `llm`, `human`, and
`assistant`; omitted keys use the runner's corresponding default backend.
These are Python callables, not remote endpoint addresses. Endpoint-specific
backends apply to every role hosted there; each action still identifies its
static role. Configure a separate backend when endpoints need separate provider
credentials or human-task delivery. Endpoint-specific human callbacks must be
worker-safe; a main-thread-only human backend can instead be supplied as the
runner's shared default.

Role programs are projected before endpoint values are known. The owner
records a binding, sends descriptors to retained roles and invitations to
recruits, then forks locally if retained. Other retained roles receive their
descriptor and fork. A recruited role starts after its invitation is accepted,
even if it had no parent control or the parent has finished. Each role can
proceed once its own activation permits it; messages may wait for a recipient
that has not activated yet. There is no global activation barrier.

Recruited fields start fresh and uninitialized, including when the endpoint
already serves another session. They do not copy that endpoint's root values
or `Var` defaults. Retained freshness still copies at the local fork, not when
the owner selects the binding. `fresh="all"` selects retained fields;
recruited fields are fresh independently. `all_except(...)` may preserve only
retained fields. Inputs for recruits must arrive by actions or messages.

Selection, fan-out progress and activation are durable runtime state. Once a
child binding commits, recovery reuses it. This does not make a preceding
external reviewer-selection action exactly-once; its ordinary effect replay
window remains. The paper's symbolic invitation calculus assumes reliable
queues and excludes crashes, so runtime recovery tests are an additional
engineering guarantee, not its projection theorem.

The configured endpoint registry is part of the durable run's identity. Resume
with the same registry: adding or removing configured endpoints is currently
rejected, even if an endpoint has not yet been selected. Changing endpoint
membership for a live durable run needs a separate lifecycle design.

See [the executable access-review example](../examples/dynamic_spawn/specification.md).
The exact local view for Reviewer shows its statically compiled child template;
the global view preserves source-level binding and retention choices.

## Persistence and compatibility

The existing role-state transaction stores a versioned session registry,
control positions, copied cells, binding references, fork counters, scheduler
cursor and pending human requests. Child creation and parent advancement
commit together. A crash before commit retries the same fork. A crash after
commit resumes its existing child.

External intake acknowledgement is a separate boundary. Keep a claim
redeliverable until its identity and the intake participant's fork are durable.
Then acknowledge intake in the parent before claiming another item. A crash
before acknowledgement can resume the saved child, and a crash after it must
be safe to repeat the acknowledgement. Do not acknowledge inside an effect
whose returned claim has not yet committed. The external service must support
the required redelivery and idempotent acknowledgement. This is not an atomic
transaction across a mailbox and the workflow store.

Opening a schema-4 store for execution upgrades it transactionally to schema 5
by adding the spawn-instance and activation ledgers. Existing workflow state
is preserved; read-only inspection does not upgrade it. Older incompatible
recovery schemas still require an explicit reset. Workflows without spawn
retain their existing projection, fingerprints and control encoding. Spawn
workflows use source-tree locations for control tags and parallel namespaces
so identical fragments at different sites remain distinct.

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
