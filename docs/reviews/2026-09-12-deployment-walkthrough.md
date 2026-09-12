# Deployment walkthrough

Run on 12 September 2026 against runtime commit `0c19e23`, after pushing the
[operational reliability fixes](2026-09-12-operational-reliability.md).

## Setup

The walkthrough used macOS 26.6.2 and Python 3.12.8. A fresh virtual environment
installed the built wheel outside the source checkout. Deployment then built
its own Python environment, copied the workflow into a managed source bundle,
and registered a real launchd service.

The project, ZipperGen home, mailbox, and launchd plist directory were temporary.
The private home and mailbox paths included spaces. A local HTTP server
returned simulated model replies through the normal local-provider backend.
Approvals used the CLI. No real credentials or external business services
were involved.

The fixture processed numbered input files through two participants, Mailbox
and Writer. Mailbox read a request, Writer drafted a reply, and Mailbox waited
for approval. A local output file recorded either sent or rejected. Request
numbers lived in durable workflow state. Repeating an output write with the
same number and content left one outcome file.

This was a purpose-built fixture. The stock email tutorial was inspected and
tested separately, which exposed the issue described below.

Wheel SHA-256:
`01f6a24b442e0394f01fa65c6c330c893a7834fa4c69cf16a5946897f5d2fa40`.
This identifies the artifact used for the service exercise, before the
tutorial and documentation changes made afterward.

## Observed results

| Check | Result |
| --- | --- |
| Fresh deployment | Managed installation succeeded, the service ran, and its installed `zg` command worked |
| Stop and resume while awaiting approval | Same task ID, no repeated draft request |
| Forced process crash | launchd restarted the managed workflow process, with the same approval and no repeated draft request |
| Positive approval | One sent outcome appeared after approval |
| Negative approval | One rejected outcome appeared, with no sent outcome for that request |
| Prompt update with an outstanding approval | Redeployment kept the existing task and its original draft. The next request used the new prompt |
| Incompatible protocol update | Deployment returned exit code 1 and preserved the profile and SQLite database byte for byte |
| Resume after the refused update | Restoring compatible source allowed the existing request to finish |
| Cleanup | Service unregistered, plist removed, deployment archived, local provider stopped |

Four requests finished with the intended outcomes: three approved and one
rejected. The HTTP server received four draft requests and one explicit
connectivity check. The stop/start and forced-crash checks added no draft
requests. These results concern a crash while waiting for approval, after the
draft had committed. They do not establish exactly-once model calls when a
process dies between receiving a response and saving it.

Validation, global views, and both participant views passed. Semantic
snapshots and diffs recorded the prompt update and protocol change.
`zg check --strict`, deployment checks, status, inspection, tasks, logs, and
trace all worked during the exercise.

The incompatible candidate added an action before reading the next request.
The readiness check reported:

```text
FAIL workflow state compatibility: WorkflowIdentityError:
The workflow changed since this durable state was written, so the stored
control positions no longer mean the same thing.
```

The message also gave reset and restore instructions. The old approval remained
available. No reset was needed to resume the previously deployed workflow.
This checks compatibility of saved control positions. A compatible prompt or
action-body edit still needs review for its business meaning.

## Tutorial issue found and fixed

The original `examples/email_approval.py` renamed a request from `.txt` to
`.done` inside its initial read action. A process could die after that rename
but before saving the action's result. On replay, the next input would be
selected and the first request would never reach approval.

A two-process reproduction confirmed this. The first process exited with
code 99 after reading and renaming `01.txt`, without returning the result to
the runtime. The next process's read returned the second request. The first
request was already marked done.

The tutorial now reads without removing the input. A durable item record
carries the filename and text until an explicit completion action runs after
approval or rejection. Completion recognizes a rename that already succeeded.
The optional message budget now uses a durable processed count instead of a
mutable Python global. Rejections count toward the budget too.

Six regression cases run the real SQLite workflow in fresh processes. They
cover both approval outcomes with a crash after the read, after the completion
rename but before its result commits, and after completion commits. Each case
resumes the first request, drafts it once, and leaves the second request
untouched when the budget is one.

The README and first-workflow tutorial now show the revised protocol. The PDF
was rebuilt. Saved runs of the old example must finish with the old source or
be reset before deploying the changed protocol. The runtime storage format is
unchanged.

The file example assumes one consumer per mailbox and unique input filenames.
Inputs must stay unchanged while awaiting review. Its simulated send is a
print, which can repeat after an uncommitted action. A real delivery connector
must use the external service's idempotency mechanism where available.

## Validation after the tutorial fix

| Check | Result |
| --- | --- |
| Full suite on Python 3.11 | 1,413 passed, 2 skipped |
| Full suite on Python 3.12 | 1,413 passed, 2 skipped |
| Full suite on Python 3.13 | 1,413 passed, 2 skipped |
| Pyright with the Google extra installed | 0 errors, 0 warnings |
| Documentation build | Passed |
| Source distribution, wheel, and Twine checks | Passed |
| Documented two-message CLI sample | One approval and one rejection, result 1, both files completed |

The skipped checks were the opt-in live systemd test and a stale local build
tree check. The final package was built and checked separately afterward.

## Limits

This exercise provides evidence for the local deployment lifecycle. It does
not compare ZipperGen with another framework or measure model quality.
Linux systemd execution, logout and host reboot, remote HTTPS, live credentials,
credential refresh, and Telegram or Google delivery remain separate checks.
The full service walkthrough used a simulated provider over loopback HTTP.
