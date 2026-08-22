# Changelog

## 0.1.0a3 — 2026-08-22

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
- Gmail, Google Sheets, Telegram, and coding-assistant integrations.
- External-action attempt, duration, and failure diagnostics, including honest
  incomplete attempts after process death.
- Upgrade paths for project, workspace, run, and deployment-profile metadata.

Operational notes:

- Durable store compatibility remains strict. An incompatible workflow change
  requires `zg deploy reset`, which archives and replaces the current recovery
  state and leaves the deployment stopped.
- `zg deploy remove` archives the profile, store, and log by default, deletes
  credentials and rebuildable artifacts, and unregisters the service.
- Trace retention remains a per-store row budget, defaulting to 10,000 rows.
  It is not a time or disk-size guarantee; high-frequency routine events can
  evict older incidents quickly.
- Managed deployments use at-least-once external effects. Effects must be
  idempotent because a process can die after the outside world changed but
  before the successor control position was committed.

## 0.1.0a2 — 2026-06-29

Second public alpha.

## 0.1.0a1 — 2026-03-14

Initial public alpha.
