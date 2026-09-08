# ICFEM 2026 CPL experiments

This directory contains the CPL monitor microbenchmark and two small workflow-
generation checks. They are described in
[`workflow-generation/README.md`](workflow-generation/README.md).

## CPL monitor microbenchmark

The benchmark measures the runtime and message data used by the CPL monitor
implemented in ZipperGen. Each iteration processes one event through the
monitor. Model calls, tool calls, and network transmission are outside the
measurement.

The benchmark varies one parameter at a time:

- the complete code-review guard from the paper, with the Orchestrator,
  TestRunner, Security, and Committer lifelines;
- 2--32 lifelines, with 31 subformulas and 4 variable names;
- 7--127 monitored subformulas, with 8 lifelines and 4 variable names;
- 1--128 variable names in cross-lifeline terms, with 8 lifelines and 255
  subformulas.

The code-review guard compares the candidates visible at TestRunner and
Security with the Committer's candidate and evaluates the two episode-aware
`Since` conditions. It has 16 distinct subformulas and one remote field name,
`candidate`.

The scaling cases use lifelines `L0`, ..., `L(l-1)`. Events are processed on
the last lifeline with the synthetic guard
`(At[L0].field_0 == 0) & (At[L0].field_1 == 1) & ...`. When the number of
atoms exceeds the number of field names, the field names repeat cyclically;
the integer constants keep the atoms distinct. A conjunction of `n` atoms has
`2n-1` subformulas. Every reported field occurs in the guard. The larger fixed
guard in the field series allows all 128 fields to occur while keeping the
formula size unchanged: 128 atoms give 255 subformulas.

For each configuration it reports the median cost of:

- `local action`: update the local clock and evaluate the formulas;
- `send`: process a send event and construct the monitor metadata attached to
  the message;
- `receive`: merge newer metadata from a received message and evaluate the
  formulas;
- `message metadata`: the number of bytes attached to the message when all
  monitor entries are populated, using ZipperGen's durable JSON encoding.

## Run

From the ZipperGen repository root, run:

```sh
python3 experiments/icfem-2026-cpl/benchmark.py
```

For a quick smoke test:

```sh
python3 experiments/icfem-2026-cpl/benchmark.py \
  --iterations 10 --repeats 2 --output /tmp/cpl-monitor-smoke
```

The default run writes:

- `results/latest.csv`: one row per benchmark configuration;
- `results/latest.md`: a compact Markdown table;
- `results/environment.json`: Python, platform, processor, and the Git source
  state recorded before measurement.

The recorded measurements used ZipperGen revision
`b7cacf32f220cb9ae11f5cda05fc60fc3327f028`. The checked-in
`environment.json` retains the paper-repository revision from the original run.
New runs record the ZipperGen revision and worktree status, excluding the
result files being written, together with a hash of the benchmark script.
An empty `porcelain` status with no error means that the checked source was
clean. Diff hashes cover tracked changes only.

The default source path is this repository's `src` directory. Use
`--zippergen-src PATH` to benchmark another ZipperGen checkout. Run
`python3 benchmark.py --help` for all options. On systems that restrict CPU
identification, `--cpu-model NAME` records the processor name explicitly.
At least two repetitions are required to compute the interquartile range.

## Interpretation

The timings cover these monitor operations, measured in one process with
`time.perf_counter_ns`. Each reported value is the median of the mean
per-operation times from several repetitions. The CSV also records the
interquartile range and full range of these means. Receive measurements
exercise a fully populated incoming view and advance every remote clock
component, so all remote entries are refreshed. The byte count uses the same
metadata encoder as ZipperGen's durable channel. The harness fills the view
tables directly with alternating truth values; their contents do not affect the
amount of work done by an update.
