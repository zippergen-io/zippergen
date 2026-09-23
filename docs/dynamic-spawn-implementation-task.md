# Implementation task: dynamically bound spawn

Extend ZipperGen's existing spawn implementation with runtime role-to-endpoint
binding. Keep the source API small and preserve inherited spawn behavior.
Implement the feature; do not stop at parsing or an architectural proposal.

Semantic reference: `../paper-spawn/paper/main.tex`, reviewed at paper commit
`975650a26bfea6ad0ca738058965c7fd71fc0dc7`. Read its role-interface, admissible
binding, initialization, projection and operational definitions before coding.
Record the actual paper revision used and identify any subsequent semantic
changes. The paper's symbolic model excludes crashes and duplicate sends;
recovery below is an engineering refinement, not a proved paper guarantee.

## 1. Surface API: extend the existing Python DSL

The paper's `spawn P with b @D` is explanatory notation. Do not add a second
text parser or a new `protocol` declaration language for this task. ZipperGen
already parses Python ASTs and supports named `@fragment` bodies:

```python
spawn(AccessReview)
spawn(AccessReview, fresh=(Manager.request, IT.request))
spawn(AccessReview, fresh="all")
spawn(AccessReview, fresh=all_except(Manager.preferences))
```

Keep those forms and the existing `with spawn(...):` body form. The proposed
new form is:

```python
Manager: bindings = select_reviewer(request)
spawn(
    AccessReview,
    bindings=bindings @ Manager,
    retain=(Manager, IT),
    fresh="all",
)
```

`bindings @ Manager` is DSL syntax for a value owned by Manager, consistent
with existing owner annotations. Extend AST rewriting so it is evaluated at
runtime, not during Python workflow construction. Use an equally small
Python-compatible variation only if the architecture requires it; document
why. Reuse `@fragment` rather than introducing independently deployed child
workflow declarations. Derive and compile each child template statically.

Without `bindings`, all child roles are retained. Omit `retain`; if accepted
explicitly, it must equal the entire child interface. With `bindings`, omitted
`retain` means the empty set: every child role is recruited. Reject `retain`
without a binding if it would change inherited semantics. Omitted `fresh`
continues to mean no selected retained cells are copied.

Descriptors, invitations, fork counters and session IDs are runtime internals,
not author-facing requirements. Exact local/debug views may expose generated
instructions when useful for inspection; ordinary global views should show
source-level spawn and its parameters.

## 2. Inspect and preserve the current architecture

Read the workflow skill and its DSL reference, then inspect the builder, AST,
validation, projection, session scheduler, channels, role runner, store,
control serialization, human-task identity, views and existing spawn tests.
Current starting points include `builder.py:spawn`, `syntax.py:SpawnStmt`,
`projection.py`, `sessions.py`, `role_runner.py` and `store.py`.

The current runtime equates a static lifeline with a physical endpoint.
`Session.bindings` currently maps variable names to cell allocations: it is
not a role-to-endpoint binding. Keep these concepts distinct in the code.
The current durable control decoder locates child programs in one root AST;
it must also identify installed role templates correctly.

Preserve existing inherited spawn, `fresh="all"`, `all_except(...)`, sharing
warnings and participant-local parallel joins. Preserve non-spawn projection
and behavior. Explicitly version any changed durable encoding and either
migrate supported old states or reject incompatible resume clearly; never
silently reinterpret persisted state.

## 3. Roles, endpoints and templates

Separate static protocol roles from concrete endpoint identities. A binding
is a serializable map with exactly the child interface's role keys and
canonical endpoint identifiers. It is total, injective within one session and
immutable after selection. Retained roles must preserve their parent address.
Endpoint identity is not a display label, secret or arbitrary Python object.

Supply a minimal endpoint/template registration or resolution mechanism using
the existing managed runtime and transports. An endpoint can exist without
having a parent-session role. An invitation activates precompiled role code;
it does not ship executable code or recompile for an address. Bind descriptors
to the static template/site and compatible protocol version.

For the first implementation, configured endpoints hosted by the existing
runtime are sufficient. Exercise real logical routing and endpoint-local
state, not a mock binding table that still runs everything in fixed role
slots. Do not add a remote network backend merely for this task. State this
backend boundary clearly: it is not yet isolation across administrative
organizations. Explain how action/backend configuration and credentials resolve
for the bound endpoint; do not accidentally reuse another endpoint's settings
just because the role name matches.

Keep the root's default binding compatible with existing workflows. A single
endpoint may execute multiple sessions, including different roles in different
sessions. It may be recruited despite appearing in the parent under another
role. Retention is explicit, never inferred from equal addresses.

The binding owner belongs to the parent but need not belong to the child.
The paper also permits an owner whose role name occurs in the child but is
not retained; preserve this case, including self-invitation when applicable.

## 4. Creation and local progress

Inherited spawn keeps its existing initiator-free local forks and introduces
no descriptor traffic or new global barrier.

For a bound spawn:

1. The owner records the child identity and validated binding.
2. It sends that same descriptor to retained roles other than itself and
   invitations to recruited roles in a fixed deterministic order.
3. If retained, the owner forks after its outgoing initialization sends.
4. Other retained roles receive and validate their descriptor, then fork.
5. A recruited endpoint accepts its invitation and installs its child role.

Preserve the paper's per-site visit agreement and single counter increment;
a subsequent fork must not increment an already recorded bound-spawn visit.
A retained receive must check the expected child visit, site/template and
address. Reject conflicting descriptors rather than installing inconsistent
bindings.

A child role starts after its own activation, without a global activation
barrier. A parent continues after its own creation obligations; it does not
wait for all invitations to arrive or for child completion. Application
messages sent before their destination activates must remain queued.

Invitation acceptance must remain runnable when an endpoint has no parent
role or its other controls are blocked/finished. Parent termination must not
terminate children, pending invitations or initialization delivery. Finite-run
completion must account for outstanding activation work.

## 5. Routing, state and participant support

Keep logical routes session- and branch-scoped with static sender/receiver
roles. Resolve physical delivery through the installed immutable binding.
Keep initialization routes distinct from application and decision routes.
Controls, counters, pending actions and human tasks must distinguish execution,
session and role as needed; equal source locations or input values in different
sessions must not collide.

Separate endpoint-local stores and role/session cell allocations. Same field
names at different roles or allocations must not alias. Retained unselected
fields keep the actual inherited allocation, including an ancestor's cell.
Selected retained fields are copied from one pre-fork local store at that
role's fork, not at the owner's binding selection. This is not a global
snapshot. Preserve deep-copy and missing-value behavior and cell lifetimes.

Every recruited protocol field starts fresh and uninitialized at acceptance.
Do not fall back to endpoint/root cells, existing `Var` defaults or another
session's fields. Pass inputs through ordinary actions/messages before reads.
Persistent recruited-endpoint settings are not protocol-cell inheritance and
are outside this first calculus.

For bound spawn, `fresh="all"` selects child-relevant fields of retained roles;
recruited fields are fresh independently. Preserve `all_except(...)` for
retained fields; reject attempts to inherit recruited fields through exclusions.
Field inference must respect nested session/retention boundaries rather than
blindly collecting every syntactically occurring role/field.

Implement current-session support exactly: inherited spawn contributes the
child interface; bound spawn contributes retained roles plus the binding
owner. Recruits do not receive enclosing parent decisions merely because they
occur inside a child. Validate parent availability separately from child roles.

## 6. Durable creation, not an external exactly-once claim

Define one durable creation record per logical child, containing its stable
identity, immutable binding, template/version and creation progress. Commit
selection with the owner's counter/continuation before publishing any
initialization traffic that can activate that child. Resume partial fan-out
using that record, never a freshly selected binding.

Commit each retained fork and its copy with parent advancement. Commit an
acceptance with installation of its child control and cells. Retries must not
recopy changed parent values or reset an already installed child's state.
Reuse the existing transactional message conventions where possible; use
durable outbox/receipt state if required. Duplicate identical activation must
not create a second instance, including after the original child finishes;
conflicting reuse of an identity must fail clearly.

The guarantee is one committed selection/activation per logical instance.
Pure evaluation before a failed commit may repeat. An external selector in a
preceding action still has the existing effect replay boundary; do not claim
that this feature makes discovery/claims exactly-once across external stores.
Prefer the surface binding operand to read an already recorded owner-local
value. Do not reevaluate a previously committed binding during retry.

## 7. Validation and observable behavior

Reject missing/extra/unknown role keys, invalid endpoint values, noninjective
bindings, retained addresses that changed, invalid retain sets, freshness or
exclusions naming recruited roles, an unavailable owner, and nonlocal binding
reads when statically detectable. Check child inputs under fresh recruited
state, not under parent availability. Detect missing/incompatible templates.
Distinguish an invalid endpoint from a valid temporarily unavailable endpoint;
the latter must not produce false successful completion.

Update projection views, semantic snapshots/diffs, fingerprints, validation
and relevant inspection output. Keep existing CPL-with-spawn rejection; this
task does not establish a CPL semantics for dynamic endpoints.

## 8. Acceptance tests and example

Retain all existing spawn regression tests. Add focused in-memory and durable
coverage, using fake services and no live credentials:

- Inherited explicit/default/all/all-except freshness remains compatible.
- One/multiple recruits; mixed roles; empty retain; owner absent from child;
  owner role recruited rather than retained; self-invitation where admissible.
- Different sessions share endpoints under different role names without
  mixing messages, state, human answers or control decisions.
- A recruited endpoint can own a child decision, but receives no enclosing
  parent decision. Nested and repeated-site spawns agree on distinct identities.
- A recruit reads no pre-existing endpoint protocol state; nested retained
  children can inherit its newly allocated cells correctly.
- Child communication can precede acceptance at the destination; blocked or
  completed parent controls do not prevent acceptance/child progress.
- Crash injection before/after selection commit, partial fan-out, retained
  descriptor receipt, local fork/copy and recruited activation. Resume in a
  fresh process. Test duplicates/conflicts and stale delivery after completion.
- Invalid bindings/retention and incompatible durable/template versions fail
  clearly. Existing non-spawn behavior stays compatible.

Provide an executable `@fragment AccessReview` example with Manager, Reviewer
and IT. Manager sends the request to Reviewer; Reviewer owns approval and sends
the approved request to IT; IT performs a fake provisioning action. A parent
selects different Reviewer endpoints for overlapping children, retaining
Manager and IT with `fresh="all"`. Reviewer gets its inputs by message and is
not a participant of the parent control flow. Exercise approval and rejection.

Validate the example, inspect global and each exact local view, and inspect
semantic snapshot/diff output. Update example specification, user documentation
and the packaged workflow skill plus its repository mirror. Run focused tests,
then the required broader checks appropriate to the runtime/schema change.

## 9. Delivery and exclusions

Implement in reviewable increments: API/IR and role support; endpoint/template
execution and activation; durable recovery; views/docs/integration validation.
An intermediate stub is not completion of this task. Report the final concrete
syntax, changed state/identity model, recovery boundaries, backend coverage,
checks run and deviations from the paper. Tests are not a machine-checked
refinement proof of the implementation.

Do not add bags/pools, discovery protocols, distributed binding construction,
rebinding, migration, mid-session joining, capability negotiation, cancellation,
child result handles, implicit return channels, CPL support or a new scheduler
policy. Endpoint admission/discovery and remote transport remain separate work.
