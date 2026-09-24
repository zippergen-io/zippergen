# Experimental spawn

`parallel` runs concurrent branches within one session. `spawn` starts another
session while the current one continues. The inherited form uses the current
participants; the initiated form below allocates fresh endpoints for child roles
that are not retained.

```python
with spawn(fresh=(Mailbox.request, Writer.request, Writer.draft)):
    Mailbox(request) >> Writer(request)
    Writer: draft = draft_reply(request)
```

This inherited form has no initiating participant, handle or join. A child can continue after
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

In inherited spawn, each involved participant receives a local fork instruction with the same
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

## Fresh participants

Keep the roles static and let spawn allocate the new participants:

```python
spawn(Onboarding, at=HR, retain=(HR,),
      fresh=(HR.hire, HR.confirmation))
```

`Onboarding` is an ordinary `@fragment` with roles HR and Employee. HR initiates
creation and retains its endpoint. Employee receives a fresh endpoint on every
visit. The runtime installs Employee's already compiled role program there.
The initiator must be a child role listed explicitly in `retain`. Omitting
`retain` or excluding the initiator is an error. Without `at`, all child roles
inherit their parent endpoints and no distinguished initiator is required.

There is no application binding expression or endpoint registry. Arbitrary
recruitment of an existing service is not supported. Each endpoint keeps its
role across the sessions that retain it. Fresh endpoints are local execution
workers in the current backend; allocating one does not create a machine,
credential, account, or human being. Human actions and other external actions
connect that worker to the appropriate external service or person.

Each retained role computes the child session and binding from its own local
spawn-site visit, then forks without receiving a descriptor from the initiator.
The initiator computes the same binding, sends initialization only to fresh
roles, and then always forks its own child role. Each send allocates its target endpoint; receiving it activates the
child program. A retained child can send before the initiator reaches the
spawn: the message waits at the future address, without creating a worker or
running recipient code. There is no global activation barrier or child join.

If all child roles are retained, there are no initialization sends: everyone
forks locally, just as with inherited spawn. Initiated empty bodies are invalid
because the initiator must participate in the child.

A dispatcher can participate trivially without adding any message or join:

```python
with spawn(at=Dispatcher, retain=(Dispatcher,)):
    skip(Dispatcher)  # import skip from zippergen
    Review()         # a fragment whose other roles get fresh endpoints
```

The dispatcher initializes the fresh roles, forks a child that immediately
finishes, and continues its parent program independently of Review.

### Endpoint lifetime

An endpoint worker exits when all of its local sessions have finished, including
retained descendants. It releases its thread and, in durable execution, its
SQLite connection without waiting for other endpoints to finish or consume
already-sent messages. A newly allocated worker must first receive its
initialization; having no active session before that does not mean completion.
Pending human decisions and external actions also keep the local session alive.

Completed session controls are collected, and cells remain live only while a
local session references them. Completion and activation records are retained
for restart and duplicate suppression. A late initialization retry may briefly
wake a worker to acknowledge the duplicate (or reject conflicting contents),
but never starts the completed role again. Resume skips completed fresh workers
unless initialization is pending. No schema or protocol-fingerprint change is
required: existing session-state records contain the necessary evidence.

Retirement does not impose admission limits or prune durable history, endpoint
rows, or allocation records. In-memory runs also retain completed endpoint
records until the run ends. It bounds idle worker threads, not total historical
storage or the number of simultaneously active sessions.

### Initializing child state

Session freshness, endpoint freshness, and field freshness are distinct.
Selected retained fields copy at that role's local fork. Unselected retained
fields keep their parent cell references. New endpoints have fresh,
uninitialized protocol fields, including fields with module-level `Var`
defaults: those defaults initialize only the original participants. Initialize
new fields by protocol messages or actions before reading them.

For example, HR copies its current `hire` into the child. Then
`HR(hire) >> Employee(details)` initializes Employee.details with that hire.
An Employee action can compute Employee.response, which returns in an explicit
message. No cross-participant cell sharing or copying is implied by matching
field names. `fresh="all"` and `all_except(...)` select retained fields only;
new endpoints are fresh regardless of that selection.

Copying takes place at each retained role's local fork, rather than at the
initiator's initialization sends. An uninitialized retained field stays
uninitialized when copied. Nested inherited spawns preserve current endpoint
and cell bindings unless their field selector requests new cells.

### Allocation and recovery

Endpoint identities are derived from the fresh child session identity and the
static role in a reserved namespace. Concurrent visits and nested spawns thus
have different identities without querying a global allocator. The current
runtime scopes these identities to a workflow execution/store.

Local creation records preserve the session and derived binding. They can
exist before the initiator runs and do not themselves allocate any worker.
The separate allocation ledger commits each fresh endpoint together with its
initialization send and the initiator's successor control. A retained fork
commits its binding, copied cells, activation and successor control together.
Recovery creates only allocated workers, resumes partial initialization, and
deduplicates activation without resetting completed child state. The endpoint
population grows automatically as the protocol spawns new participants.
Existing effect replay rules still apply to arrival sources and human delivery;
allocation does not atomically claim an item in an external system.

See [the executable onboarding example](../examples/dynamic_spawn/specification.md).
The exact Employee view shows the statically compiled child program and no
parent intake loop. Global views and semantic snapshots show the initiator,
retained roles and roles allocated fresh endpoints.

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

Opening a schema-4 or schema-5 store for execution upgrades it transactionally
to schema 6, adding the missing creation, activation and allocation ledgers.
Existing workflow state is preserved; read-only inspection does not upgrade it. Older incompatible
recovery schemas still require an explicit reset. Workflows without spawn
retain their existing projection, fingerprints and control encoding. Spawn
workflows use source-tree locations for control tags and parallel namespaces
so identical fragments at different sites remain distinct.

Changing a workflow that already has durable sessions still requires the
existing deployment compatibility checks. Do not resume its store using an
older ZipperGen version that cannot decode session controls.

Initiated spawn now requires its initiator to be retained. Existing programs
that already satisfy this rule keep their projections and durable identities.
Programs using an external or freshly allocated initiator must be updated;
finish their saved executions on the original runtime before changing source.
Adding a trivial retained role changes the protocol and does not migrate a live
store.

Initiated-spawn templates use `fresh-v2` to identify the independent local
fork semantics. Existing `fresh-v1` runs must finish with their original
runtime, or be archived/reset before a fresh run. The workflow fingerprint
rejects resuming their old controls under this semantics; adding a store table
is not a migration of those controls. Inherited-only workflows keep their
previous fingerprints and control representation.

The removed `bindings=` version is not an alternative syntax for allocation.
Existing runs of that version need their original workflow and runtime to
finish; changing them to `at=` does not migrate their saved controls or endpoint
assignments. The compatibility check rejects such a resume.

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
- CPL guards use [current-session scope](cpl-spawn-design.md): `At[Reviewer]`
  refers to this session's Reviewer. Children start empty histories even when
  their cells are copied or their endpoint is shared. Same-session messages
  carry causal summaries; SQLite persists monitors with local control.
  Out-of-session roles are rejected. Cross-session queries and combining
  spawn-aware CPL with pool handoffs are unsupported. Low-level atoms require
  a semantic `version=`; field comparisons supply it automatically.
- There is no join, wait, cancellation, arbitrary existing-endpoint recruitment,
  session limit or automatic backpressure. Bound intake while evaluating it.
- The store remains local to the existing service. Spawn does not add remote
  transport or isolation between administrative domains.
- Finished endpoint workers retire automatically. Durable endpoint rows and
  creation/activation records remain; unbounded intake still needs storage
  retention and admission policies.
- External effects retain their existing replay window. Copying a work-item
  identity does not atomically claim that item in an external mailbox or queue.
