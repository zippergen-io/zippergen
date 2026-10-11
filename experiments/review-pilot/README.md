# Finite-prefix completion in optional-review workflows

ZipperGen experimental report, 11 October 2026. Runs took place from
30 September to 4 October 2026.

We asked two coding agents to prove completion in Lean for a family of reviewer
workflows. Well-typed choreographies already provide this guarantee through a
general projection theorem. This pilot measured how long the agents took to prove
the same property directly from the projected local programs.

## Workflow and property

Each execution handles one request. An Agent starts a Worker and independently
decides whether each of *k* reviewers is required. Required reviewers assess the
request and send verdicts to a Finalizer. The Finalizer receives the required
verdicts in reviewer order, then the Worker's result, and resolves the request.
The Worker can prepare while reviews are pending. Participants follow their local
programs and communicate through asynchronous FIFO channels. Projection adds the
control messages that tell participants which branches to take.

The target, [CompletionGoal](frozen/k1/CompletionGoal.lean), states that every
valid finite distributed execution prefix can be extended to a complete execution
without changing any participant's past events. A complete execution ends all
local programs and leaves no messages unreceived. The property says that completion
is possible. It does not say that every execution terminates or that a human
eventually responds. Assessment and resolution are opaque actions. The theorem
does not establish review quality or whether the final action follows the verdicts.

These workflows use a fixed receive order. They have no repeated requests,
spawning or cancellation. This report covers only this family.

## Method

Each model made one fresh attempt per size and proof setting, with a 45-minute
limit. We used `gpt-6-sol` and `claude-opus-5-5`, both at high
reasoning effort. Opus also had a $25 limit per attempt. The initial sizes were
*k* = 1, 4, 8. We added *k* = 16 later. Each agent ran in an isolated workspace
with fixed task files and a fixed Lean library. It could not access previous
submissions or organizer reference proofs.

For local proofs, the agents received the generated endpoint programs and their
semantics. They could develop endpoint, trace and queue invariants. They could not
use global realization, projection completion or related decomposition results,
even indirectly through other lemmas. The [prompts](frozen/prompts) and
[dependency policy](tools/route_policy.py) give the full restrictions. We checked
each submission against fresh copies of the task files. The only permitted
axioms were `propext`, `Classical.choice` and `Quot.sound`.

We timed the agent's attempt, including model responses, tool calls and builds.
The timer excluded task preparation, initial compilation, isolation checks and
the final independent check. These times measure proof construction by the agent.
They do not measure human effort or just Lean's checking time.

## Local proof results

Times are in minutes, rounded from the seconds recorded in [results.csv](results.csv).
Each link opens the submitted proof. Its checker verdict is in the same directory.

| Reviewers | GPT-6-Sol | Outcome | Claude Opus 5.5 | Outcome |
|---:|---:|---|---:|---|
| 1 | [24.1](runs/review-sol-k1-local-e1/Solution.lean) | Accepted | [4.5](runs/review-claude-k1-local-e1/Solution.lean) | Accepted |
| 4 | [30.0](runs/review-sol-k4-local-e1/Solution.lean) | Accepted | [5.4](runs/review-claude-k4-local-e1/Solution.lean) | Accepted |
| 8 | [41.8](runs/review-sol-k8-local-e1/Solution.lean) | Accepted | [9.8](runs/review-claude-k8-local-e1/Solution.lean) | Accepted |
| 16 | [44.9](runs/review-sol-k16-local-e1/Solution.lean) | Not accepted¹ | [12.8](runs/review-claude-k16-local-e1/Solution.lean) | Accepted |

¹ Sol proved the theorem in Lean at *k* = 16, but the
dependency check rejected its use of
`_private.MSCAgents.Correctness.0.concatList_apply`, an elementary list identity
excluded by the rule covering its module. We therefore record the attempt as
unaccepted. This rejection does not show that Sol was unable to prove completion.

The smaller local proofs enumerate traces or cases. At larger sizes, both models
used parameterized proofs. They complete the Agent's decisions, extend the other
participants' traces consistently, and prove that messages match and respect
causal order. The successful Opus proofs at
*k* = 8 and 16 avoid enumerating all combinations of review decisions.

## Choreographic guarantee

The [general theorem](pinned-lib/MSCAgents/DeadlockFreeness.lean)
`projectDist_deadlockFree` establishes the same completion property for projections
of well-typed choreographies. For global proofs, the agents received the
choreography and checked proofs that it was well-typed and projected to the given
endpoint programs. All eight submissions were accepted. The table below gives
their elapsed times.

| Reviewers | GPT-6-Sol (min) | Claude Opus 5.5 (min) |
|---:|---:|---:|
| 1 | 2.17 | 0.63 |
| 4 | 2.07 | 0.69 |
| 8 | 2.67 | 0.53 |
| 16 | 1.07 | 0.54 |

These proofs rewrite the endpoints using the projection equality and apply the
general theorem, as in [this submission](runs/review-sol-k16-global-e1/Solution.lean).
The supplied theorem and certificates already settle the main proof obligation.
We did not time their development or preparation. The experiment compares reusing
this guarantee with proving completion directly from local programs. It does not
isolate the effect of notation, since the two settings allow different proof results.

## Scope of the evidence

These runs show the proof work that a general choreographic guarantee can avoid.
They also show that both models can find useful local invariants. With one attempt
per condition on one regular family, we cannot infer a scaling law or rank the
models generally. The results do not show exponential growth in local proof cost.
Further experiments should test less regular workflows and allow local proofs to
recover and reuse global structure.

## Materials and checking

The [artifact guide](ARTIFACTS.md) explains how to inspect the inputs, proofs and
outcomes and check the submissions yourself. The archive contains all 16 timed
attempts, including the unaccepted submission. It retains the original times.
Rebuilding a proof checks its validity but does not reproduce the model's running
time.

Suggested citation: ZipperGen. *Finite-prefix completion in optional-review
workflows*. Experimental report, 2026. Use a GitHub commit permalink to cite the
version you consulted.
