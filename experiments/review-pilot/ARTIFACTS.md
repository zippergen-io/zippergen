# Artifact guide

This directory contains the materials supporting the [report](README.md).

- `frozen/`: original task files, prompts and specifications for *k* = 1, 4, 8,
  with their original SHA-256 manifests.
- `frozen-k16/`: the subsequently added *k* = 16 task and its manifests.
- `pinned-lib/`: the Lean source files identified by
  `frozen/library-manifest.json`, including the Lake configuration and toolchain.
- `runs/`: each submitted `Solution.lean` and its original `check.json`.
- `results.csv`: selected fields from the retained run metadata, including raw
  elapsed seconds, model settings and outcomes. `claude` in run identifiers
  denotes Claude Opus 5.5. Recorded start times are in Europe/Paris local time;
  the original Sol timestamps include the UTC offset.
- `tools/`: the original packet builder, submission checker and dependency policy,
  with two packaging changes: the library path points inside this archive, and
  the checker uses Python's platform-default temporary directory.

The frozen inputs, library sources, submissions and verdict files are unchanged.
All 16 submissions were rechecked from this archive on 11 October 2026. Their
acceptance outcomes, axiom lists and forbidden-dependency lists matched the
original records: 15 accepted submissions and the same Sol local rejection at
*k* = 16.

The archive omits model conversation transcripts, account configuration, build
caches and machine-specific workspace metadata. It supports checking the submitted
proofs and their dependency restrictions; it is not a complete replay harness for
the model sessions. No new model runs were used to prepare this report.

## Check a submission

Install Python 3 and Lean's `elan` toolchain manager, then run from the repository
root:

```sh
python3 experiments/review-pilot/tools/check_submission.py 8 local \
  experiments/review-pilot/runs/review-claude-k8-local-e1/Solution.lean
```

The pinned `lean-toolchain` selects the Lean version. The first check may download
that toolchain and compile the bundled library. No model API access is needed.
The checker verifies the frozen input and library hashes, builds a fresh task
workspace, checks the exact theorem statement and permitted axioms, and traverses
proof dependencies. Its JSON output includes `proved`, `axioms` and
`banned_dependencies`.

Use `global` in place of `local` to check the global setting, and select the size
and corresponding submission path. Fifteen checks should report `proved: true`.
The Sol local submission at *k* = 16 should return a nonzero exit code and identify
the forbidden dependency recorded in its original verdict. The `build_ok` field
includes the dependency audit: its false value for that submission is not a claim
that the theorem failed Lean's kernel check.

The task manifests can also be checked without invoking Lean:

```sh
python3 - <<'PY'
import sys
sys.path.insert(0, 'experiments/review-pilot/tools')
import packet
for k in packet.ALL_SIZES:
    packet.verify_size(k)
print('All frozen inputs and pinned library hashes match.')
PY
```
