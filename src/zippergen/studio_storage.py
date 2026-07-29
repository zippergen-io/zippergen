"""Studio deployment storage inspection and safe maintenance."""

from __future__ import annotations

from pathlib import Path

from zippergen.store import (
    RECOVERY_COMPACTION_VERSION,
    TRACE_RETENTION_BATCH,
    TRACE_RETENTION_KEEP,
    TRACE_RETENTION_VERSION,
)

# This mixin uses Studio's rendering, selection, and confirmation interface.
# Keeping the storage surface here prevents the main command shell from
# becoming the durable-store maintenance subsystem.
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false


class StudioStorageMixin:
    @staticmethod
    def _storage_size(value: int) -> str:
        size = float(max(0, value))
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        unit = units[0]
        for candidate in units:
            unit = candidate
            if size < 1024 or candidate == units[-1]:
                break
            size /= 1024
        return (
            f"{int(size)} {unit}"
            if unit == "B"
            else f"{size:.1f} {unit}"
        )

    def show_deployment_storage(self, args: list[str]) -> None:
        if len(args) > 1:
            raise SystemExit("Use deploy storage [NAME].")
        name = self._deployment_name(args[0] if args else None)
        from zippergen.serve import (
            _load_deployment_profile,
            _slug,
            _zippergen_home,
        )
        from zippergen.storage_maintenance import (
            inspect_store_storage,
            plan_store_compaction,
        )

        profile = _load_deployment_profile(name)
        store_path = Path(str(profile.get("store") or "")).expanduser()
        log_path = Path(str(profile.get("log") or "")).expanduser()
        report = inspect_store_storage(store_path)
        log_bytes = log_path.stat().st_size if log_path.is_file() else 0
        archive_root = _zippergen_home() / "trash" / "deployment-logs"
        archives = (
            list(archive_root.glob(f"{_slug(name)}-*.log"))
            if archive_root.is_dir()
            else []
        )
        archive_bytes = sum(
            path.stat().st_size
            for path in archives
            if path.is_file()
        )
        self._emit_table(
            "Deployment storage",
            [
                ("Deployment", name, None),
                ("Durable store", report.path, None),
                ("SQLite", self._storage_size(report.database_bytes), None),
                ("SQLite WAL", self._storage_size(report.wal_bytes), None),
                (
                    "SQLite shared memory",
                    self._storage_size(report.shm_bytes),
                    None,
                ),
                (
                    "Reusable SQLite space",
                    self._storage_size(report.reusable_bytes),
                    None,
                ),
                ("Store total", self._storage_size(report.total_bytes), None),
                ("Active log", self._storage_size(log_bytes), None),
                (
                    "Archived logs",
                    f"{len(archives)} · {self._storage_size(archive_bytes)}",
                    None,
                ),
            ],
        )
        if report.event_counts:
            self._emit_columns(
                "Durable event inventory",
                ("Kind", "Rows"),
                [
                    (kind, count)
                    for kind, count in sorted(report.event_counts.items())
                ],
                right_aligned=frozenset({1}),
            )
        else:
            self._emit_table(
                "Durable event inventory",
                [("Events", "none", "warning")],
            )
        if store_path.is_file():
            plan = plan_store_compaction(store_path)
            compaction_ready = (
                profile.get("recovery_compaction_version")
                == RECOVERY_COMPACTION_VERSION
            )
            trace_retention_ready = (
                profile.get("trace_retention_version")
                == TRACE_RETENTION_VERSION
            )
            snapshot_total = (
                len(report.snapshot_roles)
                + len(report.roles_without_snapshot)
            )
            self._emit_table(
                "Recovery and retention",
                [
                    (
                        "Snapshots",
                        f"{len(report.snapshot_roles)} of "
                        f"{snapshot_total} roles",
                        (
                            "success"
                            if snapshot_total
                            and not report.roles_without_snapshot
                            else "warning"
                        ),
                    ),
                    (
                        "Without snapshot",
                        (
                            ", ".join(report.roles_without_snapshot)
                            if report.roles_without_snapshot
                            else "none"
                        ),
                        (
                            "warning"
                            if report.roles_without_snapshot
                            else "success"
                        ),
                    ),
                    (
                        "Trace retention",
                        (
                            f"automatic online · target "
                            f"{TRACE_RETENTION_KEEP:,} · batch "
                            f"{TRACE_RETENTION_BATCH:,}"
                            if trace_retention_ready
                            else "redeploy once to enable online retention"
                        ),
                        "success" if trace_retention_ready else "warning",
                    ),
                    (
                        "Recovery-safe events",
                        f"{plan.removable_core:,}",
                        "warning" if plan.removable_core else None,
                    ),
                    (
                        "Safe compaction",
                        (
                            "available"
                            if compaction_ready
                            else "redeploy once to enable it"
                        ),
                        "success" if compaction_ready else "warning",
                    ),
                    (
                        "Human tasks",
                        f"{report.pending_tasks} pending",
                        "warning" if report.pending_tasks else None,
                    ),
                    (
                        "Task audit",
                        f"{report.completed_tasks} completed · "
                        f"{report.task_tokens} tokens · "
                        f"{report.task_notifications} notifications · "
                        "retained by design",
                        None,
                    ),
                ],
            )
        if (
            profile.get("trace_retention_version")
            != TRACE_RETENTION_VERSION
        ):
            self._emit_next(f"deploy {name} · deploy show {name}")
        elif (
            profile.get("recovery_compaction_version")
            == RECOVERY_COMPACTION_VERSION
        ):
            self._emit_next(
                f"deploy storage compact {name} · deploy show {name}"
            )
        else:
            self._emit_next(f"deploy {name} · deploy show {name}")

    def compact_deployment_storage(self, args: list[str]) -> None:
        yes = False
        names: list[str] = []
        for value in args:
            if value == "--yes":
                yes = True
            elif value.startswith("-"):
                raise SystemExit(f"Unknown storage option: {value}")
            else:
                names.append(value)
        if len(names) > 1:
            raise SystemExit(
                "Use deploy storage compact [NAME] [--yes]."
            )
        name = self._deployment_name(names[0] if names else None)
        from zippergen.serve import (
            _deployment_service_status,
            _load_deployment_profile,
            _slug,
            _zippergen_home,
        )
        from zippergen.storage_maintenance import (
            compact_store,
            plan_store_compaction,
        )

        profile = _load_deployment_profile(name)
        if (
            profile.get("recovery_compaction_version")
            != RECOVERY_COMPACTION_VERSION
        ):
            raise SystemExit(
                f"Deployment {name} predates recovery-safe compaction. "
                "Redeploy it once, then run this command again."
            )
        store_path = Path(str(profile.get("store") or "")).expanduser()
        if not store_path.is_file():
            raise SystemExit(
                f"Deployment {name} has no durable store to compact."
            )
        plan = plan_store_compaction(store_path)
        log_path = Path(str(profile.get("log") or "")).expanduser()
        log_bytes = log_path.stat().st_size if log_path.is_file() else 0
        archive_root = _zippergen_home() / "trash" / "deployment-logs"
        archive_count = (
            sum(
                1
                for path in archive_root.glob(f"{_slug(name)}-*.log")
                if path.is_file() and not path.is_symlink()
            )
            if archive_root.is_dir()
            else 0
        )
        prunable_archives = max(0, archive_count + bool(log_bytes) - 3)
        self._emit_table(
            "Safe storage compaction",
            [
                ("Deployment", name, None),
                ("Store", store_path, None),
                (
                    "Remove communications",
                    f"{plan.removable_messages:,}",
                    None,
                ),
                (
                    "Remove action journal",
                    f"{plan.removable_journal:,}",
                    None,
                ),
                (
                    "Rotate active log",
                    self._storage_size(log_bytes),
                    None,
                ),
                ("Keep log archives", "3 most recent", None),
                (
                    "Remove old log archives",
                    str(prunable_archives),
                    None,
                ),
                (
                    "Blocked roles",
                    (
                        ", ".join(plan.roles_without_snapshot)
                        if plan.roles_without_snapshot
                        else "none"
                    ),
                    (
                        "warning"
                        if plan.roles_without_snapshot
                        else "success"
                    ),
                ),
                (
                    "Safety",
                    "seed inputs, pending work, and every event beyond a "
                    "durable recovery floor remain",
                    "success",
                ),
                (
                    "Traces",
                    f"pruned online around {TRACE_RETENTION_KEEP:,} · "
                    f"never more than "
                    f"{TRACE_RETENTION_KEEP + TRACE_RETENTION_BATCH - 1:,} · "
                    "no service stop required",
                    "success",
                ),
            ],
        )
        if (
            plan.removable_core == 0
            and log_bytes == 0
            and prunable_archives == 0
        ):
            self._success(
                "Nothing is currently safe and eligible to remove."
            )
            return
        service = _deployment_service_status(name)
        if service["state"] not in {"not-loaded", "completed"}:
            raise SystemExit(
                f"Stop deployment {name} before compacting its durable "
                f"store. Current service state: {service['detail']}"
            )
        if not yes and not self._confirm_action(
            f"Compact durable storage for {name}? [y/N]: ",
            cancel_message=(
                "Storage compaction cancelled; nothing was changed."
            ),
            default=False,
        ):
            return
        result = compact_store(store_path)
        from zippergen.studio_deployments import compact_deployment_logs

        log_result = compact_deployment_logs(
            name,
            profile,
            keep_archives=3,
        )
        self._success(f"Deployment storage compacted: {name}")
        self._emit_table(
            "Compaction result",
            [
                (
                    "Removed communications",
                    f"{result.deleted_messages:,}",
                    "success",
                ),
                (
                    "Removed action journal",
                    f"{result.deleted_journal:,}",
                    "success",
                ),
                ("Before", self._storage_size(result.before_bytes), None),
                ("After", self._storage_size(result.after_bytes), "success"),
                (
                    "Reusable SQLite space",
                    self._storage_size(result.reusable_after_bytes),
                    "success",
                ),
                (
                    "Log archived",
                    self._storage_size(log_result.archived_bytes),
                    "success",
                ),
                (
                    "Old log archives removed",
                    f"{log_result.removed_archives} · "
                    f"{self._storage_size(log_result.removed_archive_bytes)}",
                    "success",
                ),
            ],
        )
        self._emit_next(f"deploy storage {name} · deploy start {name}")
