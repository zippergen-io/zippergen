# Experimental spawn implementation handoff

This implements the detached, global spawn discussed in the paper. It does not
implement the earlier handle/wait or run_all proposals. The earlier untracked
design notes have been left untouched.

## Surface and semantic choices

`with spawn(fresh=(A.x, B.y)):` wraps ordinary choreography statements.
Existing fragments work inside it. The optional functional form accepts a
zero-argument fragment. Fields copy at the owning participant's local fork.
Unselected fields retain their existing local cells. Parent termination does
not join or cancel children.

## Files to review

- `syntax.py`, `builder.py`: global spawn, qualified fields, transformed block.
- `projection.py`: source-tree site IDs, local forks, choice dissemination,
  branch routes and rejection of unsupported CPL guards.
- `sessions.py`: local session controls, copied and inherited cells, scoped
  queues, round-robin scheduling, persistence and finished-state collection.
- `runtime.py`, `role_runner.py`, `control.py`: single-step integration,
  transactional forks, resumable controls and asynchronous durable human tasks.
- `durable_runs.py`: spawn runs select the task queue instead of inline prompts.
  Existing workflows keep their current terminal interaction.
- `serve.py`: deployed runs retain the durable human-task backend even with
  pools. Disposable runs without a connector keep terminal prompts.
- `execution_inspection.py`: action and decision trace rows retain session
  identities, and sequence-based action pairing includes the session.
- `availability.py`, `validation.py`, `semantic.py`, `view.py`, `locator.py`:
  availability, discovery, semantic snapshots, projections and inspection.

## Main review obligations

1. Endpoints agree on visit counts at a static spawn site, including under
   loops and decisions where some roles appear only in the child.
2. A parent and its children cannot consume each other's messages or overwrite
   each other's selected cells or generated decision bindings.
3. A completed parent cannot cause a live descendant's inherited cells to be
   removed. A local parallel join never includes a detached child.
4. Fork copies, counters and parent advancement commit together. Human request
   creation also commits before another external action starts.
5. Resolving an external action advances that same session and instruction,
   even if another parallel branch became enabled during the call.
6. Workflows without spawn retain their old projection and durable identity.

## Evidence and limits

`tests/test_spawn.py` exercises both in-memory and SQLite execution, repeated
and nested children, conditional and zero activation, local-join escape,
several pending approvals, rejection, out-of-order answers, restart, and real
subprocess death immediately before and after a fork commit. Another subprocess
case leaves a grandchild waiting after its ancestors finish and resumes it
using the inherited cell.

A CLI subprocess test lists all three tasks, inspects their positions and
completes a mixture of approvals and rejections with the normal run commands.
Representative loop and parallel projections, initial control encodings and
fingerprints were also compared with the committed pre-spawn code and matched.

The self-contained example in `examples/spawn` has global and local views and
requires no external account. The shipped skill reference documents syntax,
freshness and current limits.

The existing synchronous external-action boundary remains. This is not a
general asynchronous action executor. There is no admission control for an
unbounded stream of new sessions. CPL and the general projection/progress
theorem remain work for review, not claims made by this implementation.

## Reviewer findings addressed

1. Local communication views omit forks with no visible communication and use
   `with fork_session(...):` for retained forks. Both full and filtered local
   views are parsed in regression tests.
2. A non-spawn durable parallel regression test makes a receive available
   while a model call runs, stops after the result commit, then resumes. It
   fails against the pre-spawn committed code with only the message persisted,
   and passes with the fix without repeating the model call. The changelog
   now describes this change separately from spawn.
3. Validation warns about participant-qualified inherited writes, including
   receive targets. Nested freshening is respected. Intentional sharing remains
   legal and still warns, with no additional syntax or suppression option in
   this revision. The example's `Mailbox.completed` warning is documented.
   Reads of parent state and mutations hidden in Python are outside this check.
4. The deployed run entry path now keeps the SQLite human-task backend even
   with pools. An integration test waits for three pending approvals before
   answering any of them. Its disposable-run counterpart verifies terminal
   prompts remain usable without a persistent task store.
5. History tables show session IDs on action, retry and decision rows. Action
   duration matching also distinguishes sessions. Missing cell allocations and
   unsupported monitors now fail explicitly. Control-tag and session-counter
   comments reflect the implementation. Compatibility notes explain why tree
   position changes can invalidate a resume.

The superseded `docs/spawn-design.md` and
`docs/reviews/2026-09-18-spawn-design.md` remain untouched and untracked. Exclude
them when staging the implementation.

## Local verification

- Full suite after the review fixes: 1,576 passed, 1 skipped on Python 3.12.
- Focused review checks: 127 passed across spawn, durable state, history and
  code views, and pools. The non-spawn result-commit test was also run against
  the old committed code and failed on the expected lost-result assertion.
- Pyright: no errors or warnings.
- Example: validation succeeds with its documented shared-accumulator warning.
  The global and local views and semantic snapshot were checked during the
  implementation. The CLI regression test exposes three tasks and completes
  two approvals and one rejection. The new deployed-run test also verifies
  three concurrent tasks for a workflow using pools.
- Skill validation and packaged/check-out reference equality passed.

No live service was changed and no version was published. These are local
checks, not a new Linux service-lifecycle run or CI result for this change.
