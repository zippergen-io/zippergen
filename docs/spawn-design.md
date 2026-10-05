# Concurrent workflow instances

Status: revised design for review, 18 September 2026. Design only. No new
construct, schema migration, or correctness theorem is implemented.

This revision replaces the participant-owned spawn and explicit wait proposal.
The current distinction is:

> `parallel` splits the current session and joins locally.
> `spawn(P)` is a global program construct that creates concurrent session P.
> `run_all` is owned by one participant and waits for the sessions it starts.

The semantic clauses and their projection must be established before extending
the supported grammar. The notation below is proposed, not executable alpha-5
syntax.

## 1. Three constructs

| Construct | Ownership | Continuation | Completion dependency |
| --- | --- | --- | --- |
| Existing `parallel` | Global program construct | Branches of the current session | Existing participant-local joins |
| `spawn(P)` | Global program construct | Parent and P continue concurrently | No source-level join |
| `A: results = run_all(...)` | Owned by A | A's calling continuation is parked | Every participant in each directly requested session must finish |

For discussion, use the mathematical term `spawn(P)` for a global subprogram
P. A possible Python-facing form is:

```python
spawn(P, fresh=(Writer.draft, User.approved))
```

No participant annotation, participant-owned return handle, or separate source
`wait`/`join` accompanies this construct. Runtime inspection can still expose
session identifiers. Exact workflow/fragment argument syntax remains open.
There is no need to introduce Python async functions.

For owned invocation, the illustrative form is:

```python
Dispatcher: results = run_all(
    P(first_request),
    Q(second_request),
)
```

Targets are statically known initially. The nested call expressions describe
workflow instances, not ordinary eager Python calls. Arguments to this owned
operation must be available at Dispatcher. Start with a finite group, typed
results in request order, and one exported owned value or no value per target.
No implicit environment export is intended.

Global spawn does not gather all inputs at one participant. Variables retain
their ordinary owners. A value needed elsewhere requires normal communication.
Any future input-binding syntax must say where the value is captured and must
not introduce an invisible remote read or a global snapshot.

## 2. Parallel remains unchanged

```python
with parallel:
    with branch:
        A: x = quick()
    with branch:
        B: y = slow()
A: z = next_step(x)
```

A may execute `next_step` while B is still working. Branches share a session,
its participant-local cells, and the containing protocol's lifetime. Each
participant joins only its own branch projections. Branch channels stay scoped
as in the parallel paper. No global barrier is added.

Spawn differs at that join boundary. In schematic notation:

```text
parallel(spawn(P); Q, R); S
```

P may continue after the parent reaches S. Replacing the first branch by
`parallel(P, Q)` would make P part of that branch's local join and is incorrect.

The guiding fork law is:

```text
spawn(P) with continuing context K
    creates P in a new session alongside K in the parent session
```

Here K includes the existing parent control structure, including surrounding
branches and joins. It is not merely the remaining statements in the nearest
source block. The concurrency on the right is between sessions. It is not a
textual rewrite using an enclosing source `parallel` with its implicit join.

## 3. Candidate projection for global spawn

A global construct does not imply one simultaneous global runtime event.
The candidate is one local fork at each participating lifeline. Schematically:

```text
project_A(spawn^o(P), parent_session) =
    fork_local(child_session, project_A(P, child_session))  if A participates in P
    empty                                                 otherwise
```

The session name is shared across these local forks. `o` is a stable source
occurrence label. This is a candidate rule, not an implemented projection.

At A, the fork adds the child's projected control to A's session registry and
advances A's parent continuation. The two then become schedulable at the same
persistent participant. This is not a joining `ParallelLocalStmt`. Another
participant B can reach its corresponding fork earlier or later. There is no
implicit barrier between their fork events and no designated spawn owner.
Child events at A follow A's fork, and parent events after the construct at A
also follow that fork. No additional cross-participant launch edge is assumed.

Participation follows the same structural rule as existing parallel: the
participants of the contained program participate in the construct. Thus:

```text
L(spawn(P)) = L(P)
```

In `if condition @ A: spawn(P)`, a participant B in P receives the branch
decision through the existing conditional projection, exactly as it would for
a parallel branch containing its actions. Enclosing loops use the existing
loop-decision mechanism too. This requires no separate spawn-choice protocol
and adds no global barrier. The earlier owned-spawn participation rule is
superseded because spawn is now a global construct.

In contrast, the parent-level footprint of an owned `run_all` is its caller.
Its child templates are discovered for deployment and started by that caller's
invocation protocol. They do not automatically become recipients of every
parent branch decision. This is a real projection distinction between the two
constructs, not just different spellings for a local helper.

## 4. Session identities and channels

All projected forks for one source occurrence must identify the same child.
A candidate identity is:

```text
(parent session, spawn source site, dynamic occurrence)
```

Occurrence bookkeeping must distinguish loop visits and nested branch paths
and must survive restart. The agreement lemma is still required. Independent
random IDs at each role, wall-clock order, or local total step counts are not
sufficient. Ordinary decision communication supplies relevant control history,
but the construction must prove that local occurrence tracking agrees.

A child endpoint can send before another endpoint has reached its local fork.
The receiving route must retain that message until the endpoint is activated.
A late or replayed fork must neither create another child nor clear its queue.

Ordinary and decision channels use:

```text
(session, parallel branch path, sender, receiver)
```

FIFO holds only within that complete route. Physical connections may multiplex
routes without promising FIFO across sessions. A spawn starts a fresh session
namespace. Parallel extends the branch path within its existing namespace.

## 5. Participant-local state

For global spawn, retain the requested selected-freshness rule. Source fields
belong to one participant. For review, identify a field by participant and
source name within the deployment and reject incompatible types/defaults.
Name collisions across reusable templates are a usability issue to assess.

Let `rho_p(A,x)` identify the current A-owned cell in parent session p. A child
s with fresh set F uses:

```text
rho_s(A,x) = (s,A,x)      if (A,x) is in F
rho_s(A,x) = rho_p(A,x)   otherwise
```

Only bindings are inherited, not snapshots of values. Each role initializes
its fresh cells at its own fork, unset or with copied declared defaults.
Inherited defaults are not reset. A nested spawn inherits the current binding,
including any ancestor-allocated cell, unless it freshens that field again.
Cells referenced by surviving children remain alive after the parent finishes.

`fresh=(Writer.draft, User.approved)` names fields, not remote value reads.
It is not a sufficient isolation list for a complete meeting workflow. Inputs,
receiver copies, decisions, and intermediate values must also be considered.
Exact input initialization remains part of the syntax/semantics review.

Ordinary local actions remain sequential at an actual participant. This does
not isolate several actions across sessions:

```text
s1 at Writer: draft = "reply to Alice"
s2 at Writer: draft = "reply to Bob"
s1 at Writer: send(draft)             # reads the Bob draft if the cell is shared
```

There is no directly shared cell between different participants. Messages copy
values. Runtime serialization is still needed, and shared external resources
retain their own concurrency rules. Generated control positions and decision
temporaries are always session-specific. Credentials remain configuration.

For `run_all`, the current intention is independent workflow instances with
fresh local data and channels, without implicit access to the caller's cells.
Whether their similarly named roles share actual participant executors or are
fresh role instances remains a reviewer decision. Do not silently settle that
question by equating owned invocation with global spawn. Configuration reuse
alone implies neither shared cells nor shared lifelines.

## 6. Completion and run_all

For a successfully completed session s:

```text
complete(s) iff every participant projection in s has terminated
```

Required protocol traffic and result publication must be accounted for at these
terminal states. A role is not finished merely because another role produced
a result. The result owner snapshots its exported value at its local terminal
transition, so later writes to inherited cells cannot change that result.
Current workflow `return` records output metadata. Do not introduce an early
exit or global barrier under that spelling as part of this change.

An owned `run_all` requires a concrete invocation/completion protocol:

1. The caller captures its local arguments and starts each requested instance.
2. Child endpoints receive their inputs and enable their projected control.
3. Each terminal endpoint supplies a durable completion receipt, with the
   frozen value from the declared result owner.
4. The caller resumes only after all endpoints of every requested instance have
   finished and results are available. It writes results and advances atomically.

Generated deliveries must appear in enriched MSCs and detailed views. A
one-machine registry can implement part of this protocol, but reading a global
"done" flag is not itself a projection proof. Other parent participants need
normal communication if their continuation must depend on the caller's result.

Completion is session-local. If P starts T with global spawn and finishes its
own projections, `run_all(P, Q)` can return after P and Q finish while T runs.
If P instead invokes and awaits work through its own `run_all`, that dependency
holds indirectly. There is no recursive descendant-completion rule.

Thus `run_all` does not always enclose all work transitively created by its
targets. A proof using only closed child MSCs must either first consider targets
without unjoined spawning or account for escaping sessions. No separate source
wait/join or transferable future is proposed in the current three-construct
model. Standalone spawned results are observable through runtime inspection,
not a later source join.

## 7. Inductive MSC semantics remains the main gate

The parallel paper uses enriched MSCs, componentwise sequencing, and filtered
shuffle. Its existing proof cannot be assumed to cover spawn.

If a spawn statement's denotation includes the whole child chart, ordinary
componentwise concatenation places the continuation after the child's events
on shared participants. This creates an unwanted join. A bare statement that
spawn means parallel with the remaining context is not yet an inductive rule.

A candidate domain separates the current-session tuple `M_s`, a family B of
spawned-session charts, and scope/instance metadata E. Schematically, with
compatible bindings and fresh names consistently renamed:

```text
(M_s, B, E) ; (N_s, C, F) = (M_s concat N_s, B union C, E union F)
```

Only current-session words are concatenated. Parallel retains its current
local branch joins. Spawn contributes correlated local fork markers to the
current session and a separately named child. These markers are not a broadcast
from one chosen participant. Child control at each role begins after its fork.
Owned `run_all` contributes its caller's invocation and result dependency plus
the separately running instance charts and completion protocol.

This is a construction to formalize. A continuation/context formulation may be
cleaner. Whichever domain is chosen must specify composition independently of
the projected runtime and must handle nested parallel, loops, nested spawns,
and escaping sessions. Session-completion markers belong to the ends of that
session's own projections, not after all its outstanding descendants.

On each actual participant, events from shared sessions form one local word.
Keep scoped FIFO and generated control edges, check enriched MSC consistency,
and only then apply the chosen observation/erasure. Do not rename one shared
participant into independent lifelines just to make the proof easier.

The paper's action and guard labels are symbolic. A structural theorem at that
level is distinct from concrete feasibility of shared-cell computations. The
latter must evaluate the combined participant store, not each session in
isolation before shuffling. Human request/answer phases require their own
refinement account.

Before implementation, establish:

- Exact inductive clauses and matching local projection rules.
- Agreement of instance identities across projected local forks.
- Associativity of sequence and existing semantics when no new constructs occur.
- Both directions of complete enriched execution correspondence, up to names.
- A separate prefix/progress statement with explicit assumptions.

Acyclicity of complete charts alone does not establish progress. The old Zipper
lemma's completion premise cannot mean all work at a participant has finished
when a parent continuation has started and children remain active.

## 8. Scheduling, recovery, and lifetime

Share scoped queues, residual encoding, enabled-leaf scheduling, and occurrence
persistence. Keep distinct AST/control nodes for parallel, global spawn, and
owned run_all. Do not assume the current synchronous external-action driver
already supplies the necessary scheduler.

A blocked receive, run_all, or human answer parks only its continuation.
Other sessions and enabled parallel branches must remain schedulable. Ordinary
actions at a shared participant stay sequential. Unbounded ordinary effects
can still delay that participant and need an explicit application policy.

Human interaction has a durable request phase and answer phase. Capture the
request's inputs and destination cells once. Consume its answer once and
advance only that continuation. Task identifiers include session and occurrence.
Never restore a stale whole environment over another session's updates.
Retryable external actions also need frozen operands where shared cells can
change between attempts. Existing external-effect replay limitations remain.

Global spawn requires **per-endpoint atomicity**: register that role's child
control and cell bindings, initialize once, and advance that role's parent fork
in one durable transition. All roles use the same logical instance key and
register idempotently. There is no simultaneous all-role parent advancement.
Record expected roles so an endpoint that has not yet forked is not mistaken
for a completed one. A restart must preserve messages awaiting late endpoints.

Owned run_all can use caller-side atomic group creation, input snapshots,
activation outboxes, and parked control, followed by idempotent result delivery.
Persist session/cell bindings, residuals, occurrence counters, pending tasks,
queues, membership, terminal receipts, and frozen results. Pin code versions.
Admission failure must be consistent for a logical instance or group. The exact
bounded-runtime failure protocol remains work, not a proved language property.

Sessions belong to the host service. Parent termination does not cancel spawned
sessions. Stop/restart preserves them. Host reset explicitly handles all stored
session state. A foreground host may drain active sessions, which is distinct
from a parent program join. Referenced cells and results are retained until
archival initially. Failure does not undo shared-cell or external-effect writes.

## 9. Review examples and deliberate exclusions

The review must cover:

- P and its continuation sharing a participant, with either advancing first.
- Spawn inside a parallel branch, with the child outliving that branch's join.
- Conditional and loop spawning with roles that occur only in the child.
- Different roles reaching corresponding forks at different times, including
  a message arriving before the receiver's fork and recovery in that gap.
- Repeated/nested spawning with distinct channels and consistent instance IDs.
- Fresh data isolation and intentional inherited-cell sharing.
- A run_all result becoming available before another required endpoint finishes.
- A run_all target spawning an unjoined descendant that survives its completion.
- Several human approvals answered out of order across a restart.

Postpone source join/wait, cancellation, role-renaming syntax, dynamic/recursive
targets, cross-session pool operations, direct inter-session messaging,
multi-machine deployment, live protocol migration, automatic cleanup, and agent
memory policies. Decide the independent-instance role model for run_all before
claiming its semantics settled.

CPL is separate. Shared lifeline histories and fresh fields make session-relative
causality nontrivial. Reject unsupported CPL/monitor use in the first session
mode rather than silently choose a meaning. Existing non-session behavior stays
unchanged.

## Source basis

Relevant current sources are [projection](../src/zippergen/projection.py),
[syntax](../src/zippergen/syntax.py), [builder](../src/zippergen/builder.py),
[availability](../src/zippergen/availability.py),
[runtime](../src/zippergen/runtime.py),
[role driver](../src/zippergen/role_runner.py),
[supervisor](../src/zippergen/sqlite_runner.py),
[store](../src/zippergen/store.py), [control](../src/zippergen/control.py),
and [monitor](../src/zippergen/monitor.py). The sibling parallel paper's
camera-ready source supplied the existing sequencing, shuffle, and proof rules.

See the [reviewer handoff](reviews/2026-09-18-spawn-design.md). The README's
formal-coverage gate remains applicable. This document records design choices
and open proof obligations, not supported features or new guarantees.
