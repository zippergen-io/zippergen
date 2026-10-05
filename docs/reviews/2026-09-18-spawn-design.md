# Concurrent workflow design review handoff

Date: 18 September 2026. Design only. No new syntax is implemented.
The current proposal is [docs/spawn-design.md](../spawn-design.md).
Source baseline: `31f31b70acdb2893edb814b9ed6cdaf866f9f385`.

## Current direction

The proposal now distinguishes three constructs:

- Existing global `parallel`, with its unchanged participant-local joins.
- Global `spawn(P)`, without an owner annotation or source-level join.
- Participant-owned `run_all`, which starts workflow instances and waits for
  every endpoint in each directly requested instance to finish.

The earlier `A: s = spawn(P)` plus `A: result = wait(s)` design is superseded.
Do not review its creator-owned handles or its single-owner launch event as
requirements of the current global spawn. There is no source wait/join in the
current proposal. Runtime instance identities remain necessary.

The main gate is a simple inductive MSC semantics with a matching projection.
Runtime plausibility is insufficient. No new theorem is claimed.

## Direct answers and remaining decisions

1. **Syntax.** Global `spawn(P)` with optional selected freshness. Owned
   `A: results = run_all(P(args), Q(args))` for a finite set of statically known
   targets. Python-facing program/argument syntax is illustrative and must be
   finalized with ownership and binding rules.

2. **Parallel versus spawn.** Parallel divides current-session control and
   locally joins its branches. Spawn adds child-session control alongside the
   continuing parent context, without making that child an obligation of the
   enclosing parallel join. The current participants remain persistent.

3. **Common machinery.** Scoped routes, residual encoding, session registries,
   scheduling, durable occurrences, and inspection. Separate AST/control nodes
   are appropriate. Global spawn and owned invocation have different projection
   rules even when the runtime reuses lower-level components.

4. **Channels.** `(session, branch-path, sender, receiver)`, FIFO only within
   that full route. Messages can arrive before the receiving role reaches its
   local fork and must be retained for the correct endpoint.

5. **Variables.** For global spawn, inherit the parent's field-to-cell bindings
   and allocate new cells for fields listed in `fresh`. Cells remain owned by
   one participant. Nested spawning inherits the current binding, including an
   ancestor's fresh cell. Defaults do not reset inherited cells. Input binding
   must not imply an invisible remote read or a global snapshot.

6. **Session state.** Control positions, generated temporaries, binding maps,
   routes, pending actions/tasks, and terminal records are session-qualified.
   Source cells may be shared according to the explicit freshness policy.
   Sequential owner actions do not prevent logical interference across sessions.

7. **Completion.** Every participant projection in the named session must end.
   Snapshot an exported result when its owner terminates, not later from a
   mutable cell. `run_all` waits for its requested sessions, not every unjoined
   descendant. A nested run_all creates an explicit indirect dependency.

8. **Projection.** Global spawn proposes a local fork at every role in P,
   with the same logical child identity and no simultaneous launch barrier.
   Participation follows the ordinary structural rule `L(spawn(P)) = L(P)`.
   Existing if/while decision projection then works just as for parallel. No
   additional spawn-choice protocol is proposed. Owned run_all instead uses
   its caller's activation/completion protocol, with only that caller's parent
   continuation blocked.

9. **Scheduling.** One actual participant can host several session residuals.
   Blocked receives, run_all calls, and human responses park their continuation.
   Other enabled work remains schedulable. Human interaction needs separate
   durable request/answer phases with captured operands and destination cells.

10. **Recovery.** Global spawn commits each role's child registration, local
    initialization, and parent advancement together. This is per-endpoint
    atomicity, not a simultaneous transaction over all parent roles. Persist
    common occurrence identities, expected endpoints, binding aliases, queues,
    pending operations, and completion/results. run_all additionally needs a
    durable caller group and idempotent result delivery.

11. **Postponed or unsettled.** Source join/wait, cancellation, dynamic or
    recursive targets, multi-machine deployment, live migration, automatic
    cleanup, cross-session pools, and CPL extensions. For run_all, independent
    state/channels are intended, but whether role instances share actual
    participant executors remains an explicit design choice.

## Why the projection distinction matters

The global spawn rule is a candidate structural clause:

```text
project_A(spawn(P)) = fork_local(child_id, project_A(P)) if A participates in P
                     empty                             otherwise
```

The new local fork adds a residual to the participant's session registry and
continues the parent. It is not a joining local parallel block. Participating
roles may reach it at different times. Therefore agreement on child identity
requires a lemma, for example using parent session, stable source site, and
matching dynamic occurrence. Different random IDs or unrelated local step
counts would be incorrect.

For a conditional global spawn, a role occurring only in P is a participant
of that source construct. Normal decision broadcasts tell it whether to fork.
This is precisely the existing parallel/conditional participation pattern,
not a new all-role readiness handshake.

## MSC construction to assess

A complete child chart cannot simply be concatenated with the parent remainder:
that would order shared-participant continuation events after child completion.
The current candidate distinguishes a current-session chart, outstanding
session charts, and scope/lifecycle metadata. Sequence concatenates only the
current-session chart. A continuation/context formulation may be simpler.

Global spawn contributes corresponding local fork markers, not one owner event
with launch edges to all other roles. Own child events follow each role's fork.
The assembled family retains actual participant words, scoped FIFO, and generated
control communication. Check enriched consistency before erasing controls.

Require an independently defined source semantics, associative composition,
conservativity for the existing language, and both directions of correspondence
with projection. Prefix completion needs its own argument. The old premise that
starting a continuation implies preceding local completion must distinguish
current-session obligations from children that remain active.

Unjoined descendants can escape run_all too. It cannot generally be modeled
as a closed parallel region containing all transitively generated work.

## Priority reviewer questions

1. Is the local-fork rule compatible with a simple inductive global semantics?
   Does it preserve existing local joins when spawn occurs inside parallel?
2. Does normal decision projection suffice for occurrence agreement through
   nested branches and loops? Specify the identity construction precisely.
3. Can each endpoint fork at its own local point without hidden synchronization
   or unobservable reads of another endpoint's control state?
4. Which independent-instance role model should run_all use? Define its actual
   invocation/completion protocol and the required all-done dependency.
5. Are selected freshness, initialization, and inherited cell lifetime coherent
   with fork timing? Check both intentional sharing and omitted-fresh faults.
6. What completion/progress statement holds for nested unjoined children,
   parked human tasks, external actions, and bounded runtime admission?

Please prioritize these mathematical decisions over additional runtime policy
or convenience syntax. Do not implement the constructs during this review.

## Evidence and scope

Read the current projection and participation rules, builder output handling,
control encoding, runtime/store, and the sibling parallel paper's camera-ready
semantics and proof. Relevant source links are in the main proposal.

Earlier in this design exercise, five parallel projection tests, two runtime
tests, and a temporary two-role workflow probe passed. The probe showed A's
post-parallel action following only A's own branch. These establish existing
parallel behavior, not the new constructs. They were not rerun for this prose
revision. The new documents receive syntax/link/whitespace checks only.

No framework source, skill, tutorial, credential, deployment, commit, or push
was changed. No spawn/run_all proof or implementation test exists yet.
Prior documents are saved locally under
`/private/tmp/zg-spawn-global-construct-20260918/` for comparison.
