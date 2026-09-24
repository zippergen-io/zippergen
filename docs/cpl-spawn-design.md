# CPL with spawn: current-session scope

Status: implemented experimentally in the in-memory and SQLite runners.
Monitors are scoped to session-role instances and recovered with their local
control and cells. This document specifies the observation boundary; it is not
a claim of a new mechanized monitor-adequacy theorem.

## Scope and role references

Every session `s` has a statically known role interface `R_s` and an immutable
role-to-endpoint binding. At an event in `s`, `At[A]` identifies `(s,A)`, provided
`A` belongs to `R_s`. It does not identify the physical endpoint across all its
sessions. Two sessions retain separate histories even when the same endpoint
executes their Reviewer roles.

References to a role outside `R_s` are validation errors. Validate each session
body against its own interface, stopping at nested spawn bodies and checking
those separately. A role belonging to the interface but not yet activated is
a valid reference; it has no visible event yet. The same applies to a role
whose first event has not reached the observer through the protocol.

The first version has no parent/ancestor references, variable-valued role
references, endpoint-history queries, or equality-key historical lookup.
Existing `At`, `Here`, `Y`, `since`, `P`, and `on` retain their meanings on the
session-local event structure defined below. `At[A]` selects the latest visible
event of A; field comparisons do not search for an older matching event.

## Which history CPL observes

For session `s`, form an event structure from:

- its ordinary action, send, receive, and choice events, including generated
  decision traffic already observed by the existing CPL monitor;
- local order within each session-role `(s,A)`;
- matching send-to-receive edges of messages in `s`.

Take the reflexive transitive closure of these edges to define session-local
causal visibility. At event `e`, `At[A](phi)` evaluates `phi` at the latest
`(s,A)` event visible to `e`. If A owns e, this is e itself, as in current CPL.
`Y` and `since` follow the local event sequence of `(s,A)`; `P` ranges over
this session-local causal past, including the current event.

This is an explicit observation boundary. It is not the full physical causal
order restricted to events tagged `s`: paths through other sessions do not
import evidence. Sharing an endpoint, scheduling another session between two
events, or reading a shared cell adds no cross-session CPL edge. A proof using
this logic must state this session-local interpretation, not claim a property
of all endpoint history.

Fork, visit, initialization-send, and activation events remain coordination
events but are not CPL observation events in this first version. They do not
advance a parent's CPL clock, create a child's first CPL event, or transport
monitor summaries. No synthetic approval or other event is generated at startup.

For a role with no visible event, `At[A](phi)` is false. Missing fields retain
the existing field-term behavior: a direct comparison involving a missing
field is false. Ordinary Boolean negation still applies; absence is not a new
three-valued logic. Policies must explicitly require positive evidence where
needed. A choice is observed before its guard is evaluated, as today, so a
non-strict past formula may already hold at a role's first choice event.

## Creation and data

At each retained role's local fork, the child gets a new logical monitor:
its clock is zero, and its formula and field views are empty. Do not clone the
parent monitor. At a fresh endpoint, create the same empty monitor at activation.
The runtime materializes it lazily before the first attempted child step.
All monitors in the child use the child interface and the same static formula
indexing. Different roles may activate at different times; no barrier is added.

Existing field initialization is unchanged:

- selected retained fields copy at the local fork;
- unselected retained fields continue to share their existing cells;
- fresh endpoint fields start uninitialized.

Initialization does not itself create a CPL field snapshot. At each subsequent
observed event, `Here.x` uses the session's actual local cell binding, and the
monitor snapshots that event's values. Remote `At[A].x` remains a snapshot at
A's latest visible session event, not a live read of A's store.

Copying `approved = True` therefore copies data, not the parent's approval
history. A later child event can nevertheless satisfy the state predicate
`Here.approved == True`. That predicate alone does not establish that a human
performed an approval in the child. Likewise, a shared field may change due to
another session without importing the event that changed it. Correlation and
approval-event policies must express the evidence they actually require.

Ordinary child messages propagate only child-session summaries. Sending a
Boolean or certificate as application data does not automatically reconstruct
the parent events that justified it. Parent-history import is outside this
version.

## Monitoring and recovery requirements

Keep a monitor per active `(session, role)`, not one per physical endpoint.
Use the existing static monitor algorithm within each session. Compile the
formula closure for each session body independently; nested child guards must
not introduce out-of-interface references into the parent monitor.

Message metadata must identify the session and compatible formula layout.
Never merge a sibling or parent clock merely because its role names match.
Repeated spawn visits and nested instances have separate monitors. Control
messages propagate session-local summaries just as application messages do.

Persist monitor state with the corresponding session control, cells, and message
updates under the existing transaction boundary. Resume restores the monitor;
it does not restart a live child's history. Duplicate initialization must not
reset or recreate a completed monitor. Worker retirement releases execution
resources only when all local sessions finish; outstanding messages and peers
retain the snapshots they already received. Recovery and activation evidence
remain available after retirement.

The implementation rejects combining spawn-aware CPL with pool
causal handoffs until those handoffs have an explicit session-scope rule. The
existing pool stamps can cross sessions and must not be silently merged or
discarded. This restriction does not change non-spawn CPL/pool behavior
or spawn workflows that do not use CPL.

Session CPL adds a versioned formula-layout identity to the durable workflow
fingerprint and each transmitted summary. A changed layout cannot resume an old
store. Session snapshots also validate their session identity and formula layout.
Non-spawn workflows and spawn workflows without CPL keep their existing identity.

Low-level `atom` predicates in spawn workflows require `version=`, including
in-memory runs, so message layouts have stable semantic identities. Structural
field formulas supply this automatically. `At` and `on` references are checked
statically; remote roles accessed inside a Python atom must be declared with
`roles=("Reviewer",)`. Arbitrary Python predicates remain trusted user code:
the validator cannot infer undeclared reads hidden inside them.

## Regression coverage

1. Alice and Bob share physical endpoints but have distinct sessions: Reviewer
   activity in Bob's session neither supplies nor overwrites Alice's CPL view.
2. A copied or shared value is readable under existing data rules, while parent
   temporal history is absent from the child's monitor. Test state predicates
   separately from predicates about approval events.
3. A valid but inactive Reviewer has no visible event; an absent Reviewer fails
   validation. Check parent and child interfaces separately.
4. A same-session message imports its sender's causal summary, including
   transitively learned evidence. No message imports another session's summary.
5. Repeated and nested spawns start distinct histories; initialization retries
   do not reset histories or duplicate events.
6. First-event/non-strict behavior and `Y`/`since` agree with existing CPL inside
   one session. Creation events do not accidentally become previous events.
7. Shared-cell writes from another session affect subsequent local values but
   add no hidden history edge; remote field views remain event snapshots.
8. Fresh-process recovery preserves guard results around action, receive, fork,
   and initialization commits. Retirement preserves already-sent evidence.
9. Non-spawn CPL behavior is unchanged. Unsupported scope and pool combinations
   fail explicitly rather than silently dropping causal metadata.

The proof obligation is monitor adequacy for this session-local semantics and
its interaction with projected execution. Existing symbolic spawn correspondence
does not by itself prove data-dependent guard correctness or completion.
