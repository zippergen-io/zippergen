# Natural-language workflow generation

This small experiment asks a coding agent to translate two product-level
descriptions into ZipperGen workflows. The first task requires a remote-field
comparison. The second adds a local temporal condition. Both use the
application-message chain `L0 -> L1 -> L2 -> L3`.

We ran the task three times with Codex CLI 0.153.4 and model `gpt-6-astra` at
low reasoning effort. Each run began in a fresh directory produced by
`zippergen init`. The agent was asked to read the standard `zippergen skill`,
implement `specification.md`, and validate its result. No extra DSL primer,
paper text, or reference guard was supplied. The installed skill contains a
generic example of remote field access but no temporal operator. In the first
two temporal runs, the agent queried the installed Python API for `since`. In
the third, validation exposed its initial use of the unavailable name `Since`.

The first task's generated projects are under `results/run1` through
`results/run3`. The temporal task and its results are under `temporal/`.
The two `experiment.json` files record the prompt and tool versions. Recheck
the results from the ZipperGen repository root with:

```console
python3 experiments/icfem-2026-cpl/workflow-generation/assess.py
python3 experiments/icfem-2026-cpl/workflow-generation/temporal/assess.py
```

The generation runs used ZipperGen revision
`7a171946e5f3b415dcaf67947acb8eb4c62488f1`. The assessment scripts use
the `zippergen` executable on `PATH` by default. For an exact recheck, install
that revision separately and pass its executable with `--zippergen PATH`.
The monitor microbenchmark used the later revision recorded in its
`results/environment.json`.

The scripts automatically check validation, message structure, guard
ownership, and the expected outputs. For the temporal task they also run all
27 input histories. The remaining criteria in `reference.json`, such as the
absence of a precomputed summary, were checked by reading the six saved
programs. The scripts print JSON and return a nonzero status if an automatic
check fails.

All three runs produced the `L3`-owned guard
`At[L0].outcome == "heads"`, preserved the three token messages, and passed
ZipperGen validation. Both coin outcomes returned the expected result. This
task concerns the simplest guard shape.

For the temporal task, `L0` records three successive status updates before
sending the token. All three final workflows produced
`At[L0](since(Here.status != "failed", Here.status == "passed"))` and passed
all 27 three-update histories. In the third run, the first generated guard put
`since` outside `At[L0](...)`. The agent's tests exposed the error and the
agent corrected it before finishing. Episode identity is not tested.
