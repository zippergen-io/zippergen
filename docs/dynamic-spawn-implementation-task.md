# Implementation contract: spawn with fresh participants

This contract supersedes the previous owner-supplied binding design. Protocol
roles and their projections remain static; spawn creates session instances and,
when requested, fresh endpoints. Arbitrary existing-endpoint recruitment is
not supported in this version.

## Source API

Inherited spawn remains unchanged:

```python
spawn(Onboarding)
spawn(Onboarding, fresh="all")
spawn(Onboarding, fresh=all_except(HR.preferences))
```

Every child role retains its parent's endpoint. No initiator is required.

Fresh-participant spawn uses:

```python
spawn(Onboarding, at=HR, retain=(HR,), fresh=(HR.hire, HR.confirmation))
```

The equivalent `with spawn(at=HR, ...):` block is supported. `at` names a
parent role. `retain` lists child roles also available in the parent; omitted
`retain` is empty. All other child roles receive distinct fresh endpoints.
The initiator need not occur in the child or be retained. Reject the former
`bindings=` argument instead of silently treating it as inherited spawn.

Source code does not construct descriptors or name session/endpoint IDs.

## Static support and projection

For child interface R_P and parent interface R:

- inherited spawn requires R_P contained in R and has support R_P;
- initiated spawn requires D in R and I contained in R_P intersect R, with
  current-session support I union {D}.

Fresh roles receive no enclosing parent choices. Compile every child role
program before endpoint identities are known. Templates retain stable site
identities through nested and repeated spawns. Preserve inherited projection,
control encoding and fingerprints. Incompatible old bound-spawn controls must
fail explicitly on durable resume.

## Allocation, initialization and execution

Every dynamic spawn has a fresh child session identity. The initiated form
records a complete immutable role-to-endpoint binding: retained roles keep
parent addresses; other roles get globally fresh addresses within the run.
Use a collision-free namespace derived from child identity and role, disjoint
from initial endpoints; no globally consulted name registry is needed.

Retained roles derive the child name and binding independently at their local
fork. No descriptor traffic between parent roles is needed. Each initiator
initialization send allocates that one fresh endpoint and registers it for a
worker; its receive activates the compiled role program. Application messages
may queue for deterministically named endpoints before allocation; naming an
address is not execution there. Different
roles can start at different times. There is no global activation barrier or
child completion join. Parent completion must not terminate pending children.

An endpoint keeps its role in every session that retains it. Role backend
configuration must continue to work when a fresh endpoint's identity differs
from its role name. The current backend hosts workers locally; machines,
credentials, human accounts and remote transport remain external concerns.

## Field initialization

Keep three distinct notions: new session, new endpoint, new field cell.
Selected retained fields copy from one local pre-fork store at the retained
role's fork; unselected fields keep their current cell references. Missing
values remain missing. Fresh endpoint fields start uninitialized and receive
initial application data through explicit messages or actions. Neither a
module-level default nor another participant's same-named cell supplies their
values. `fresh="all"` and `all_except(...)` select retained fields only.
Descendants may keep inherited cells after their allocating session finishes.

## Recovery

Persist each allocation and initialization send atomically with its initiating
local step. A retained fork independently commits its local creation state; it
does not require or publish endpoint allocation. A committed child keeps its endpoint identities across retries. Resume
partial initialization, recreate recorded workers and deduplicate activation,
including stale initialization messages after completion. Retained copy and
local installation commit together. Completion checks must account for every
allocated worker and outstanding initialization/application message.

A persistent allocation ledger is bookkeeping for recovery, not a distributed
consensus/name-allocation service. External effects keep their existing replay
contract. Do not claim atomicity with a mailbox, human delivery service, or
remote machine provisioner.

## Validation and acceptance

Validate parent/child role interfaces, initiator availability, retained role
membership, retained-field selectors, variable availability at fresh endpoints,
template compatibility, and reserved endpoint namespaces. Keep CPL-with-spawn
rejection. Update views, snapshots, fingerprints and relevant CLI paths.

Exercise:

- inherited default/explicit/all/all-except behavior;
- owner absent from child, empty retention, owner not retained;
- repeated, nested and concurrent allocations with distinct identities;
- new endpoint fields initialized by messages/actions, copied retained inputs;
- overlapping child sessions on retained endpoints;
- parent continuation before child completion and fair initialization;
- crash/restart before and after allocation and partial initialization;
- idempotent stale initialization and rejection of conflicts;
- static-role model/human routing at dynamically created endpoints;
- non-spawn compatibility and schema upgrades.

The running example is HR onboarding Alice and Bob: copy HR.hire at each
local fork, send it to Employee.details, compute Employee.response, and send
it to child-local HR.confirmation. Two sessions share HR but use distinct
new Employee endpoints. Arrival and human behavior are represented by fake
local actions in the executable example.

Validate the example, inspect global and exact role views, compare semantic
snapshots, run focused tests and the full suite, and mirror packaged workflow
skill updates. Do not change unrelated Lean experiments or draft documents.

## Formal boundary

The paper models fresh endpoint allocation, session-role words, final endpoint assembly and activation
causality explicitly. Recheck allocation freshness, descriptor agreement,
activation-before-execution, endpoint assembly, erasure and inherited-fragment
conservativity. Separate mathematical arguments from checked Lean results;
the old fixed-participant mechanization is not a proof of this extension.
