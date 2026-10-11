# Finite-prefix completion in optional-review workflows

ZipperGen experimental report, 11 October 2026. Runs conducted from
30 September to 4 October 2026.

This pilot measured the effort required by two coding agents to construct Lean
proofs of completion for a family of distributed reviewer workflows. The property
is inherited from a general projection theorem when the workflows are specified
as well-typed choreographies. The principal measurements below concern recovering
that property directly from the projected local programs.

## Workflow and property

Each execution handles one request. An Agent starts a Worker and independently
decides whether each of *k* reviewers is required. Required reviewers assess the
request and send verdicts to a Finalizer. The Finalizer receives the required
verdicts in reviewer-index order, then the Worker's result, and resolves the
request. Worker preparation and reviewer actions can proceed independently,
subject to the endpoint programs and asynchronous FIFO communication. Projection
provides the control messages needed to communicate branch decisions.

The target, [CompletionGoal](frozen/k1/CompletionGoal.lean), states that every
valid finite distributed execution prefix has a complete extension preserving
each participant's existing sequence of events. In a complete execution all local
programs have terminated and all sent messages have been received. This is an
existential completion property: it does not assert that every execution
terminates, or that a human eventually responds. Assessment and resolution are
opaque actions; the theorem does not establish review quality or compliance with
verdict values.

The family has fixed receive order and no repeated requests, spawning or
cancellation. Results from other workflow families are not included.

## Method

There was one fresh attempt per model, size and proof setting, with a 45-minute
limit. The configured models were `gpt-6-sol` and `claude-opus-5-5`, both at high
reasoning effort. Opus also had a $25 limit per attempt. The initial sizes were
*k* = 1, 4, 8; *k* = 16 was added subsequently. Subjects ran in isolated workspaces
with the frozen task files and Lean library. Previous submissions and organizer
reference proofs were inaccessible.

In the local setting, subjects received the generated endpoint programs and their
semantics. They could develop endpoint, trace and queue invariants, but could not
use global realization, projection completion or related decomposition results,
including through transitive proof dependencies. The [prompts](frozen/prompts)
and [dependency policy](tools/route_policy.py) specify the restrictions. Final
submissions were checked against pristine task files; only the standard Lean
axioms `propext`, `Classical.choice` and `Quot.sound` were permitted.

Elapsed subject time includes model responses and tool calls, including builds
during the attempt. It excludes task preparation, initial build warming,
isolation checks and final independent checking. It measures elapsed proof
construction time, not human effort or kernel checking time alone.

## Local proof results

Times are minutes, rounded from the recorded seconds in [results.csv](results.csv).
Each linked submission has its original checker verdict alongside it.

| Reviewers | GPT-6-Sol | Outcome | Claude Opus 5.5 | Outcome |
|---:|---:|---|---:|---|
| 1 | [24.1](runs/review-sol-k1-local-e1/Solution.lean) | Accepted | [4.5](runs/review-claude-k1-local-e1/Solution.lean) | Accepted |
| 4 | [30.0](runs/review-sol-k4-local-e1/Solution.lean) | Accepted | [5.4](runs/review-claude-k4-local-e1/Solution.lean) | Accepted |
| 8 | [41.8](runs/review-sol-k8-local-e1/Solution.lean) | Accepted | [9.8](runs/review-claude-k8-local-e1/Solution.lean) | Accepted |
| 16 | [44.9](runs/review-sol-k16-local-e1/Solution.lean) | Not accepted¹ | [12.8](runs/review-claude-k16-local-e1/Solution.lean) | Accepted |

¹ The Sol submission at *k* = 16 established the stated theorem in Lean, but the
dependency check rejected its use of
`_private.MSCAgents.Correctness.0.concatList_apply`, an elementary list identity
covered by the experiment's module-based exclusion rule. It is recorded as an
unaccepted attempt, rather than evidence that Sol could not prove the property.

The smaller local proofs use explicit finite traces or finite enumeration. At
larger sizes, both models developed parameterized constructions: complete the
Agent's decisions, extend the other participants consistently, and establish
channel matching and causal order. In particular, the successful Opus proofs at
*k* = 8 and 16 avoid enumerating all combinations of review decisions.

## Choreographic guarantee

The [general theorem](pinned-lib/MSCAgents/DeadlockFreeness.lean)
`projectDist_deadlockFree` establishes the same completion property for projections
of well-typed choreographies. In the global setting, subjects received the source,
its checked well-typedness and the checked equality between its projection and
the endpoint programs. All eight submissions were accepted. Their elapsed times
are retained here for completeness.

| Reviewers | GPT-6-Sol (min) | Claude Opus 5.5 (min) |
|---:|---:|---:|
| 1 | 2.17 | 0.63 |
| 4 | 2.07 | 0.69 |
| 8 | 2.67 | 0.53 |
| 16 | 1.07 | 0.54 |

These proofs rewrite the endpoints using the projection equality and apply the
general theorem; [this submission](runs/review-sol-k16-global-e1/Solution.lean)
is representative. The theorem and the supplied certificates settle the essential
argument before the subject starts. Their development and preparation are outside
the timing measurements. The experiment therefore measures reuse of an established
guarantee versus direct local recovery, rather than an isolated effect of notation
under equal access to proof results.

## Scope of the evidence

The observations illustrate proof work that verification by construction can
avoid, and show that both models can discover useful local invariants. One attempt
per condition, on a single regular family, does not establish a scaling law or a
general ranking of models. These measurements do not support a claim of
exponential local proof cost. Less regular workflows and local proofs allowed to
recover and reuse global structure remain to be investigated.

## Materials and checking

The [artifact guide](ARTIFACTS.md) describes the frozen inputs, submitted proofs,
recorded outcomes and commands for independent checking. The archive includes all
16 timed attempts, including the unaccepted submission. The original timing
measurements are retained; rebuilding a proof does not reproduce model latency.

Suggested citation: ZipperGen. *Finite-prefix completion in optional-review
workflows*. Experimental report, 2026. Cite this file through a GitHub commit
permalink to identify the version used.
