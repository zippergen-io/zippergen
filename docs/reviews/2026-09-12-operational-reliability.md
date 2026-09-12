# Operational reliability review

Reviewed on 12 September 2026, starting from commit `38ee438`.

This review follows the path from project configuration to a running service.
It covers credential storage and routing, approvals, SQLite startup and
recovery, deployment preparation and publication, and updates with saved state.
It is a focused review of these paths. It does not cover every framework module.

## Findings and fixes

### 1. Missing credentials could destroy the previous development run

Priority: high. Fixed in `durable_runs.py`.

Starting a new durable run discarded the selected run before resolving required
environment fields. A missing credential then stopped preparation after the old
record and SQLite store had been deleted.

Required environment and provider configuration now resolve before replacement.
The regression uses a real missing required credential and checks that the
previous record, store bytes, and selected run remain unchanged. A successfully
prepared new run still replaces the previous run as documented. This change
does not provide rollback once the new workflow starts executing.

Test: `test_missing_credential_keeps_the_previous_durable_run` in
[`test_durable_runs.py`](../../tests/test_durable_runs.py).

### 2. A deployment could consume another deployment's text approval

Priority: high. Fixed in `telegram_notify.py`.

Two deployments can share a Telegram bot and chat while trusting different
approvers. The text handler checked its local approver policy before checking
whether it owned the task. A rejected actor caused the update to be removed
from the shared inbox, including when the answer belonged to another workflow.
That workflow then kept waiting. Button approvals already checked ownership
first.

Text commands and direct replies now establish ownership before applying the
actor policy. Tests cover foreign and owned tasks through buttons, commands,
and replies. Unauthorized answers to owned tasks remain rejected.

Tests: `test_a_foreign_token_stays_in_the_shared_inbox_under_every_actor_policy`
and `test_a_token_this_deployment_owns_still_obeys_its_actor_policy` in
[`test_telegram_notify.py`](../../tests/test_telegram_notify.py).

### 3. A worker startup failure could leave the service waiting indefinitely

Priority: high. Fixed in `sqlite_runner.py`.

A role opened its SQLite connection outside the worker's error handler. If
opening failed, the role thread exited without notifying the supervisor. With
the unlimited timeout used for services, another role could wait indefinitely
for its messages.

Opening the connection now shares the worker's error handler. The supervisor
cancels its peers and reports the original cause. The regression injects a
connection failure into one role of a two-role workflow with an unlimited
timeout and checks that supervision finishes with that cause.

Test: `test_role_store_open_failure_stops_the_supervisor` in
[`test_sqlite_runner.py`](../../tests/test_sqlite_runner.py).

### 4. Moving a prepared Python environment broke installed commands

Priority: medium. Fixed in `deployment_environment.py`.

Deployment built a virtual environment in a temporary directory, installed its
packages, then renamed the directory. Installed scripts retain absolute paths
to their interpreter. A script could therefore exist in the published
environment and still fail with a missing-interpreter error. Python documents
this restriction in its [virtual environment guidance](https://docs.python.org/3/library/venv.html#how-venvs-work).

Each environment is now built at its final, unique generation path. The active
profile still selects the previous environment until publication. Failed
preparation removes the candidate. The regression creates a real virtual
environment with a representative installed command and executes the command
before and after publication.

Test: `test_managed_environment_console_script_survives_publication` in
[`test_deployment_environment.py`](../../tests/test_deployment_environment.py).

### 5. Linux service templates interpreted parts of filesystem paths

Priority: medium. Fixed in `deployment_publication.py`.

The generated `ExecStart` value was unquoted. A space in the ZipperGen home
split the executable path. Percent signs in executable, working-directory, or
log paths could also be interpreted as systemd specifiers.

Executable paths now use systemd command quoting with environment substitution
disabled. Path settings escape literal percent signs. The implementation was
checked against the upstream [command parsing documentation](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml)
and [path-setting parsers](https://github.com/systemd/systemd/blob/main/src/core/load-fragment.c).
Tests check spaces, quotes, backslashes, percent signs, and dollar expressions
in generated files, including unchanged literal paths in launchd plists.

These are template tests. Running the generated unit under a live systemd user
manager remains part of the Linux release check.

Test: `test_service_template_preserves_literal_paths` in
[`test_deployments.py`](../../tests/test_deployments.py).

### 6. Failed store initialization leaked its connection

Priority: medium. Fixed in `store.py`.

Once SQLite connected, an error in identification or setup could leave the
connection open. Retaining the exception could retain the connection too.
The opener now closes the connection on every initialization failure and
preserves the original exception.

Test: `test_failed_store_open_closes_its_connection` in
[`test_store.py`](../../tests/test_store.py).

## Other paths reviewed

- Credential entry uses hidden input. Managed secret files use owner-only
  permissions. Provider credentials and connector routing are stored separately.
- Approval answers are validated centrally. Completion preserves the first
  accepted answer, and token use shares the answer transaction.
- Role state, message consumption, and message emission have explicit transaction
  boundaries. External effects can repeat if the process dies before recording
  their result, as documented.
- Deployment readiness checks workflow identity before publication. Runtime
  startup checks it again. Removal and reset constrain the paths they can modify.
- Candidate environments and secrets are selected through the deployment
  profile. Existing rollback and compatibility tests remain part of validation.

No workflow syntax, store schema, profile schema, or workflow fingerprint was
changed by these fixes. Existing deployments need a redeploy to use the updated
runtime and generated service files. Their saved state remains compatible with
these review changes. Earlier unreleased pool changes have their own
compatibility note in the changelog.

## Validation

| Check | Result |
| --- | --- |
| Full suite on Python 3.11 | 1,407 passed, 2 skipped |
| Full suite on Python 3.12 | 1,407 passed, 2 skipped |
| Full suite on Python 3.13 | 1,407 passed, 2 skipped |
| Durable-run suite after the final type-check adjustment | 21 passed |
| Pyright with the Google extra installed | 0 errors, 0 warnings |
| Source distribution and wheel build, plus Twine metadata checks | Passed |
| Wheel installed into a separate environment outside the checkout | Import, CLI help, deployment help, and packaged skill passed |
| Whitespace and patch checks | Passed |

The skipped tests were the live systemd integration and a check for a stale
local build tree, which was absent when the test run reached it. The final
distribution was built and checked separately afterwards.

The review used temporary stores and fake provider responses. No real
credentials, live approvals, or external business actions were used.

A live Linux service-manager test and real provider credential refresh remain
for the fresh-install release walkthrough. This review ran on macOS, and a
Linux container daemon was unavailable.
