# Changelog

## Unreleased

- The email approval tutorial now keeps each request available until it has
  been handled and preserves its message budget across restart. Completion
  recognizes a repeated call after a crash. This changes the example's
  protocol. Finish saved executions with the previous example, or reset them
  before deploying the updated version.
- A new durable run now checks its required environment and credentials before
  discarding the previous run. Missing configuration leaves the previous run
  record and store intact.
- Telegram text approvals now check task ownership before applying the local
  approver policy. Deployments sharing a bot and chat no longer consume text
  approvals addressed to another deployment.
- SQLite worker startup failures now stop the supervisor and report the
  original error. Failed store initialization also closes its connection.
- Managed Python environments are built at their final generation path, so
  installed command-line scripts remain usable after deployment publication.
- Generated systemd units now quote executable paths and escape literal
  percent signs in paths. Environment substitution is disabled for the
  executable command.
- Added execution-local FIFO work pools: `Pool("jobs")` provides `put`,
  nonblocking `try_claim`, `ack`, and `release` effect actions. Stable operation
  receipts survive recovery, claims have fixed leases, and stale acknowledgements
  are rejected. Pools use the managed SQLite store. Includes a two-worker
  example and updated workflow-authoring guidance.
- Pool jobs now carry CPL context: completed puts and explicit releases publish
  it, successful claims import it, and replay uses the original saved context.
  Empty claims and lease expiry add no dependency on other workers. Includes
  an approval-and-correlation example. This changes pool workflow identity:
  saved executions from the earlier unreleased pool implementation must finish
  on that runtime or be archived/reset before starting fresh. No-pool workflow
  identities are unchanged.

- Deployment readiness now checks saved workflow identity before publishing
  an update or starting a service. An incompatible protocol is rejected with
  reset/restore guidance; the previous deployment and durable state are kept.
- HTTPS certificate verification failures in model calls now report a permanent
  error with certificate-configuration guidance instead of retrying indefinitely
  under `retries="forever"`. TLS verification remains enabled.
- Default pytest discovery now targets the framework suite, matching CI, so
  archived research programs with separate dependencies are not collected.

## 0.1.0a3 — 2026-09-02

This prerelease replaces the original example-oriented package with the
project, configuration, durable-run, and managed-deployment surfaces described
in the current README and manuals.

Highlights:

- Project-local workflows, semantic validation, snapshots, and diffs.
- Named model, assistant, provider, and connector configurations.
- Durable SQLite execution with crash recovery, human tasks, and explicit
  reset/archival operations.
- Managed systemd and launchd deployments with status, freshness, logs,
  inspection, trace streaming, maintenance, and lifecycle commands.
- Crash-safe deployment publication through immutable bundle, environment,
  and secret generations selected by one atomic profile update.
- Gmail, Google Sheets, Telegram, and coding-assistant integrations.
- Coding-assistant results use each CLI's structured success envelope, so an
  interrupted assistant cannot be committed as ordinary output merely because
  its process exited zero.
- External-action attempt, duration, and failure diagnostics, including honest
  incomplete attempts after process death.
- Project manifests, workspace state, run records, deployment profiles, and the
  durable store use strict version gates. A file written by a newer ZipperGen
  is refused rather than rewritten. There are no migrations because there is
  no earlier released format.

Removed before release:

- `with coregion:` is gone, along with `CoregionStmt`, `ReceiveAnyStmt`, and
  the `any` durable control state. It was the one construct outside the
  published results -- the ISoLA paper lists coregions as future work -- so it
  had to be documented as not carrying the deadlock-freedom guarantee. Every
  remaining construct is covered by a proof.
- Use `with parallel:` where a relaxed order is wanted. It is not identical:
  a coregion accepted from several senders on one thread, while parallel runs
  concurrent branches on separate channels. The outcome is the same; the
  execution shape is heavier.

Operational notes:

- Durable store compatibility remains strict. An incompatible workflow change
  requires `zg deploy reset`, which archives and replaces the current recovery
  state and leaves the deployment stopped.
- Existing workspace and deployment addresses are retained across the switch
  to identity-based lookup; new names keep a readable project-directory prefix.
  A legacy project moved before its address is recorded now refuses with a
  recovery instruction instead of silently selecting an empty workspace.
- `zg deploy remove` archives the profile, store, and log by default, deletes
  credentials and rebuildable artifacts, and unregisters the service.
- Trace retention remains a per-store row budget, defaulting to 10,000 rows.
  It is not a time or disk-size guarantee; high-frequency routine events can
  evict older incidents quickly.
- Managed deployments use at-least-once external effects. Effects must be
  idempotent because a process can die after the outside world changed but
  before the successor control position was committed.
- Human actions now require a response unless they are explicitly declared as
  `kind="ack", required=False`. The former `@human(visible=False)` shortcut is
  removed because it could silently grant approval; `visible=False` remains a
  trace-persistence choice for `@pure`, `@effect`, and `@assistant`.

## 0.1.0a2 — 2026-06-29

Second public alpha.

## 0.1.0a1 — 2026-03-14

Initial public alpha.
