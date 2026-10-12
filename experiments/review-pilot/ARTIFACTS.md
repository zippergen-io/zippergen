# Artifact guide

This directory contains the inputs, proofs and results for the [report](README.md).

- `frozen/`: original task files, prompts and specifications for *k* = 1, 4, 8,
  with their original SHA-256 manifests.
- `frozen-k16/`: the *k* = 16 task added later, with its manifests.
- `pinned-lib/`: the Lean source files identified by
  `frozen/library-manifest.json`, including the Lake configuration and toolchain.
- `runs/`: each submitted `Solution.lean` and its original `check.json`.
- `results.csv`: model settings, start times, elapsed seconds and outcomes from
  the run records. `claude` in a run name means Claude Opus 5.5. Start times use
  Europe/Paris local time. Sol and Astra timestamps also include the UTC offset.
- `tools/`: the original packet builder, submission checker and dependency policy,
  with two packaging changes: the library path points inside this archive, and
  the checker uses Python's platform-default temporary directory.

We copied the task files, library sources, submissions and verdicts without
changing them. We rechecked the 16 Sol and Opus submissions from this archive on
11 October 2026 and the four Astra submissions on 12 October 2026. The outcomes,
axioms and forbidden dependencies matched the original records. Nineteen
submissions were accepted. The Sol local proof at *k* = 16 was rejected for the
same dependency.

The archive does not include model conversations, account settings, build caches
or machine-specific workspace records. You can check the proofs and their
dependencies, but you cannot replay the model sessions from these files alone.
Packaging and rechecking the proofs did not involve new model calls.

## Check a submission

Install Python 3 and Lean's `elan` toolchain manager, then run from the repository
root:

```sh
python3 experiments/review-pilot/tools/check_submission.py 8 local \
  experiments/review-pilot/runs/review-claude-k8-local-e1/Solution.lean
```

The `lean-toolchain` file selects the original Lean version. The first check may
download that toolchain and compile the library. You do not need model API access.
The checker verifies the input and library hashes and builds a fresh workspace.
It then checks the theorem statement, axioms and proof dependencies. Its JSON
output includes `proved`, `axioms` and `banned_dependencies`.

To check another submission, change the size, `local` or `global`, and the file
path. Nineteen checks should report `proved: true`. The Sol local submission at
*k* = 16 should return a nonzero exit code and report the original forbidden
dependency. Its `build_ok` field is false because this field includes the
dependency check. The theorem itself passes Lean's kernel check.

You can also check the input hashes without running Lean:

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
