# Changelog

## Unreleased

- Named assistant configurations now accept `--model` and `--reasoning-effort`
  (`--effort` alias), so participants and actions can select different models
  within the same CLI backend. Pins are passed on every turn and session resume,
  shown in configuration reports, and saved in durable-run/deployment records.
  Existing configurations that only select a backend keep their behavior.

- Assistant actions now retain one CLI conversation per participant, backend,
  workspace, and login location within a workflow execution. Durable resume
  restores conversation IDs from the store; native CLI session files must
  remain available. Use `session="fresh"` for an independent action. This
  policy is part of workflow semantics, so existing durable assistant workflows
  must finish with their original runtime or start a new execution.

- Initiated spawn now requires `at` to be a retained child role. The initiator
  initializes fresh roles, then always forks; the separate visit marker is
  removed. Use a retained `skip(Dispatcher)` role for fire-and-forget dispatch.
  Existing spawns already satisfying this restriction keep their durable identity.

- CPL now supports spawn with current-session role scope. Each session-role has
  an independent monitor; children start empty histories and message summaries
  are checked against the session and formula layout. SQLite commits monitor
  state with session control and cells. Missing roles fail validation; pool
  causal handoffs with spawn-aware CPL remain unsupported. Low-level atoms need
  semantic versions, and can declare remote references with `roles=`.

- Completed endpoint workers now retire while other sessions continue, releasing
  their threads and SQLite connections. Retained descendants keep the endpoint
  alive; uninitialized fresh endpoints still wait for activation. Durable resume
  skips finished endpoints, while late initialization retries are acknowledged
  without rerunning role code. Completion and activation records remain stored.

- Retained roles in initiated spawn now derive child bindings locally and fork
  independently, without descriptor messages. Messages may queue before fresh
  endpoints are allocated; each initialization send allocates only its target.
  Durable allocation records preserve this behavior across restart. Existing
  initiated-spawn runs require their original runtime (`fresh-v1`); the new
  projection is fingerprinted as `fresh-v2`.

- Experimental fresh-participant spawn allocates endpoints while keeping role
  projections static: `spawn(Onboarding, at=HR, retain=(HR,), fresh="all")`.
  Retained roles fork locally; other roles start on newly created endpoints
  with uninitialized cells. The earlier arbitrary `bindings=`/endpoint-registry
  API is removed. Global/local views and snapshots expose allocation choices.
  See [fresh participants](docs/experimental-spawn.md#fresh-participants).
- Durable stores upgrade from schemas 4/5 to 6 by adding missing spawn creation,
  activation and allocation ledgers transactionally. Existing workflow state is preserved;
  read-only inspection leaves old stores unchanged.
- Added `spawn(fresh=all_except(Writer.preferences))` to infer private child
  fields while retaining selected existing participant-local cells. Invalid
  exclusions fail during construction. Nested exclusions inherit the current
  binding, and views retain the expanded ordinary freshness list.
- Added `spawn(fresh="all")` as shorthand for the participant-local fields
  used in the child body. Views and semantic snapshots retain the expanded
  list. Explicit tuples and default inheritance keep their meaning.
- SQLite polling effects can raise `EffectPending(retry_after=...)` to yield
  without committing an empty result. Other sessions and parallel branches
  remain runnable. Quiet polls use `visible=False`. See
  [polling effects](docs/polling-effects.md) for retry and runner limits.
- Unsupported participant assignments such as `Mailbox: settled = True` now
  fail during workflow construction instead of disappearing from the protocol.
- Spawn freshness now accepts module-level fields used only in guards.
- Fixed durable parallel execution losing an external action's result when a
  receive in another branch became enabled during the call. The returned
  result now commits to the action that requested it, without repeating the
  call. This also fixes workflows that do not use spawn.
- Experimental `spawn` starts detached sessions using the same participants.
  Selected local fields copy at each participant's fork. Sessions have separate
  controls and message channels and resume from the durable store.
- Durable spawn runs keep human approvals in the task queue so other sessions
  can continue. Use `zg run tasks` and `zg run approve --task TASK_ID --yes`
  or `--no`, or the existing Telegram connector.
- Deployed workflows using pools now retain the durable human-task backend
  instead of opening terminal prompts. Plain local runs still prompt inline.
- Spawn validation warns about writes to inherited fields omitted from `fresh`,
  including receive targets. Intentional sharing remains allowed.
- Added a self-contained concurrent approval example and workflow skill
  guidance. See [experimental spawn](docs/experimental-spawn.md) for its
  semantics and limits. CPL guards and session admission limits are not yet
  supported. External actions retain their existing replay behavior.

## 0.1.0a5 - 2026-09-18

This alpha makes local model setup easier and gives coding agents more precise
workflow and connector instructions.

- Guided local model setup shows the server URL, offers available models and
  participant assignment, and explains connection failures and SSH tunnels.
  Explicit configuration commands remain offline. The workflow skill now starts
  from the user's existing model server and separates server setup from routing.
- Expanded the workflow skill with Gmail, Google Sheets, Calendar and Telegram
  connector signatures and return shapes, loop initialization guidance, and a
  runnable crash-recovery test using fake services. Clarified when successful
  checks need to be repeated.
- Updated the development and deployment guide for the guided model setup and
  remote model connections. The first tutorial prompt now stays on one page
  and can be copied without a shell prompt marker.

Upgrade notes:

- Workflow syntax, the durable-store schema and deployment command meanings
  are unchanged from 0.1.0a4. No state reset is needed for this package update.
- In an existing coding-agent conversation, run `zg skill` again after
  upgrading so the agent receives the updated instructions.
- Existing deployments keep their managed runtime until redeployed. Stop the
  deployment and run `zg deploy` to apply the updated runtime while preserving
  compatible saved state.
- The [security limits](docs/security.md) are unchanged. Workflows are trusted
  Python code, assistant access needs suitable operating-system isolation, and
  external effects may repeat after a crash.

## 0.1.0a4 - 2026-09-16

This alpha adds durable work pools, CPL context across pool handoffs and a
Google Calendar connector. It also improves deployment recovery, approval
routing and credential handling.

Security fixes:

- The Google extra requires `cryptography>=50.0.0`, which includes the fix for
  [GHSA-g6cj-pr64-35w5](https://github.com/pyca/cryptography/security/advisories/GHSA-g6cj-pr64-35w5).
  The affected PKCS#7 decryption APIs are not used by ZipperGen's connectors.
- Execution locks reject symbolic links, multiply linked files and special
  files before changing their permissions or contents.
- Google authorization handoffs require refreshable credentials, matching
  client IDs and valid scope metadata before replacing saved credentials.
  Malformed encoded data and checksums are rejected with a controlled error.
- Assistant environment filtering now lists standard locale variables
  explicitly. Arbitrary variables beginning with `LC_` are no longer inherited.

Security limits:

- Workflows and deployment setup are trusted Python code. This release does
  not provide a sandbox for arbitrary third-party workflows.
- Assistant environment filtering does not isolate credential files readable
  by the same operating-system account. Stronger separation requires an
  assistant execution service, account or container with restricted file access.
- Google handoff encoding is not encryption, and its checksum is not a
  signature. Accept a handoff only from your own trusted authorization session.
- External effects can repeat after a crash unless the external service
  supports idempotency. Private files use filesystem permissions, not encryption
  supplied by ZipperGen. See [security notes](docs/security.md).

Upgrade notes:

- These security fixes do not change workflow syntax, the durable-store schema
  or deployment command meanings. They do not require resetting saved state.
- An installed package upgrade does not replace an existing deployment's
  managed runtime. Stop the deployment and run `zg deploy` to apply the new
  runtime. Compatible saved state is preserved.
- Protocol changes still require a compatibility check. In particular, the
  updated email approval example and earlier unreleased pool workflows have
  the compatibility limits described below. Do not reset a live workflow
  without first deciding how to handle pending work and external effects.

Other changes:

- Google and Telegram connector errors no longer include upstream response
  bodies or chained transport exceptions in ordinary diagnostics. Google
  credential fields are excluded from object representations. HTTP status
  codes remain available for troubleshooting.
- Google Sheets rejects malformed row collections before a keyed update, so
  skipping a bad row cannot shift the write to another row. Table replacement
  validates its complete payload before clearing existing cells. Replacement
  still uses two requests, and keyed upserts still require a single writer.

- Added a Google Calendar connector for reading events and creating single timed
  events with stable request IDs for recovery. It shares Google authorization,
  connector configuration and deployment wiring. Includes an approval example.

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
