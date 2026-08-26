"""Command-line execution, inspection, and local deployment entry point.

The per-role durable runtime lives in :mod:`zippergen.role_runner`; this module
parses ordinary CLI commands and coordinates the supporting subsystems.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CLI:
#   `zippergen run [--llm SPEC] [--input k=v]`
#   `zippergen run --durable` / `zippergen run --resume`
# ---------------------------------------------------------------------------
import argparse
import getpass
import json
import math
import os
import plistlib
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zippergen.models import ModelSettings

from zippergen.process_environment import temporary_environment
from zippergen.workspace import Workspace

from zippergen.deployment import (
    DeploymentField,
    DeploymentSetup,
    DeploymentSpec,
    deployment_spec_from_module,
    normalize_deployment_spec,
)
from zippergen.deployment_platform import (
    deployment_launchd_path as _deployment_launchd_path,
    deployment_profile_path as _deployment_profile_path,
    deployment_script_path as _deployment_script_path,
    deployment_secrets_dir as _deployment_secrets_dir,
    deployment_secrets_path as _deployment_secrets_path,
    deployment_service_path as _deployment_service_path,
    deployments_dir as _deployments_dir,
    installed_launchd_path as _installed_launchd_path,
    installed_systemd_service_path as _installed_systemd_service_path,
    launchctl_command as _launchctl_command,
    launchctl_domain as _launchctl_domain,
    launchd_label as _launchd_label,
    launchd_service_status as _launchd_service_status,
    run_launchctl as _run_launchctl,
    run_systemctl as _run_systemctl,
    enforce_deploy_requirement,
    require_service_stopped,
    ServiceIsLiveError,
    service_is_running,
    service_manager as _service_manager,
    slug as _slug,
    systemctl_command as _systemctl_command,
    systemd_unit_name as _systemd_unit_name,
    zippergen_home as _zippergen_home,
)
from zippergen.deployment_environment import (
    bundle_deployment as _bundle_deployment,
    prepare_deployment_environment as _prepare_deployment_environment,
)
from zippergen.execution_lock import (
    ActiveExecution,
    ExecutionLockError,
    active_execution,
    execution_lock,
    execution_lock_path,
)
from zippergen.deployment_profiles import (
    DEPLOYMENT_PROFILE_SCHEMA_VERSION,
    _default_deployment_log_path,
    _default_deployment_store_path,
    _deployment_environment,
    _field_enabled,
    _load_deployment_profile,
    _load_deployment_secrets,
    _profile_environment,
    _profile_mapping,
    _profile_options,
)
from zippergen.value_codec import encode_value
from zippergen.private_files import (
    ensure_private_directory,
    write_private_bytes,
    write_private_text,
)
from zippergen.deployment_checks import (
    DoctorConfig,
    _call_doctor_hook,
    _doctor_check,
    _doctor_checks,
    deployment_freshness_checks,
    _launchd_active_check,
    _path_parent_check,
    _safe_json_loads,
    _store_status,
    _systemd_active_check,
)
from zippergen.connector_wiring import (
    _start_deployment_connector_workers,
)
from zippergen.connectors import (
    CONNECTOR_KINDS,
    CONNECTOR_SETTING_SPECS,
    connector_kind_spec,
    connector_requirements_from_module,
)
from zippergen.models import (
    apply_model_overrides,
    effective_llm_routes,
    normalize_llm_overrides,
    project_model_routing,
    selected_llm_specs,
)
from zippergen.assistant_configuration import (
    apply_assistant_overrides,
    normalize_assistant_overrides,
    project_assistant_routing,
)
from zippergen.view import DETAILS, ViewOptions, workflow_view_data
from zippergen.human_tasks import (
    human_task_result_from_value,
    validate_human_task_result,
)
from zippergen.workflow_io import (
    RunConfig,
    _call_setup_hook,
    _looks_like_path,
    load_workflow_spec,
)
from zippergen.validation import (
    assistant_actions as _assistant_actions,
    validate_workflow as _validate_workflow,
)
from zippergen.semantic import (
    read_semantic_snapshot,
    render_semantic_diff,
    semantic_diff_models,
    semantic_snapshot,
    workflow_semantics,
)
from zippergen.assistant_backends import (
    ASSISTANT_BACKENDS,
    assistant_backend_spec,
)
from zippergen.deployments import (
    _deployment_inventory,
    _prune_shared_connector_inboxes,
)
from zippergen.deployment_publication import (
    _apply_existing_history_keep,
    _run_deployment_setup,
    _setup_enabled,
    _write_deployment_secrets,
    _deployment_command,
    _initialize_deployment_store,
    _install_launchd_agent,
    _install_systemd_unit,
    _prepare_managed_home,
    _profile_history_keep,
    _write_deployment_artifacts,
)
from zippergen.execution_inspection import (
    _control_values,
    _trace_duration,
    _trace_fields,
    _trace_time,
    _load_trace_events,
    _trace_row,
    _trace_rows,
    _trace_seconds,
    _trace_value,
)
from zippergen.syntax import Workflow
from zippergen.store import (
    complete_human_task,
    ensure_human_task_token,
    list_workflow_results,
    load_human_task,
    load_human_task_token,
    mark_human_task_token_used,
    open_store,
    open_store_readonly,
    read_history_keep,
    write_history_keep,
)


def _parse_inputs(pairs: list[str]) -> dict:
    out: dict = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        if not k or not _:
            raise SystemExit(f"Invalid --input {p!r}; expected name=value.")
        try:
            out[k] = json.loads(v)      # 7 -> int, "true" via JSON, '"s"' -> str
        except json.JSONDecodeError:
            out[k] = v                  # bare string fallback
    return out


def _nonnegative_int_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int_argument(value: str) -> int:
    parsed = _nonnegative_int_argument(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float_argument(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number zero or greater")
    return parsed


def _positive_float_argument(value: str) -> float:
    parsed = _nonnegative_float_argument(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_input_json(text: str | None) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--input-json must be valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SystemExit("--input-json must be a JSON object.")
    return value


def _parse_options(pairs: list[str]) -> dict:
    return _parse_inputs(pairs)


def _profile_model_settings(
    profile: Mapping[str, object],
) -> dict[str, "ModelSettings"]:
    """Read the unified model settings from a deployment profile."""

    from zippergen.models import model_settings_from_mapping

    stored = profile.get("llm_settings")
    if isinstance(stored, Mapping):
        return {
            str(target): model_settings_from_mapping(value, subject=str(target))
            for target, value in stored.items()
        }
    return {}


def _profile_idle_timeouts(profile: Mapping[str, object]) -> dict[str, float]:
    """The idle-release times a profile carries, whichever shape it uses."""

    return {
        target: chosen.idle_timeout
        for target, chosen in _profile_model_settings(profile).items()
        if chosen.idle_timeout is not None
    }


def _idle_timeout_settings(pairs: list[str]) -> dict[str, "ModelSettings"]:
    """Turn --llm-idle-timeout-for pairs into per-target model settings."""

    from zippergen.models import ModelSettings

    return {
        target: ModelSettings(idle_timeout=seconds)
        for target, seconds in _parse_llm_idle_timeouts(pairs).items()
    }


def _parse_llm_idle_timeouts(pairs: list[str]) -> dict[str, float]:
    values = _parse_inputs(pairs)
    timeouts: dict[str, float] = {}
    for target, value in values.items():
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "--llm-idle-timeout-for requires "
                "PARTICIPANT_OR_ACTION=SECONDS."
            ) from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise SystemExit("LLM idle release seconds must be non-negative.")
        timeouts[str(target)] = seconds
    return timeouts


def _jsonable_kv_pairs(values: Mapping[str, object]) -> list[str]:
    return [f"{key}={json.dumps(value, default=str)}" for key, value in sorted(values.items())]


def _run_args_from_deployment(profile: dict[str, object]):
    timeout_raw = profile.get("timeout", 0.0)
    timeout = float(timeout_raw) if isinstance(timeout_raw, (int, float, str)) else 0.0
    return argparse.Namespace(
        workflow=str(profile["workflow"]),
        llm=profile.get("llm") or None,
        llm_for=_jsonable_kv_pairs(normalize_llm_overrides(profile.get("llms"))),
        llm_idle_timeout=profile.get("llm_idle_timeout"),
        llm_idle_timeout_for=_jsonable_kv_pairs(
            _profile_idle_timeouts(profile)  # type: ignore[arg-type]
        ),
        llm_settings=_profile_model_settings(profile),
        assistant=profile.get("assistant") or None,
        assistants=normalize_assistant_overrides(profile.get("assistants")),
        store=str(profile["store"]),
        input=[],
        input_json=json.dumps(profile.get("inputs") or {}, default=str),
        option=_jsonable_kv_pairs(profile.get("options") or {}),  # type: ignore[arg-type]
        timeout=timeout,
        execution=str(profile.get("execution", "sqlite")),
    )


def _deployment_lifecycle_command(args, action: str) -> int:
    from zippergen.deployment_platform import deployment_service_status

    profile = _load_deployment_profile(args.name)
    name = str(profile["name"])
    # Read the state once, up front. Every decision below and the sentence this
    # prints at the end are about the difference between before and after.
    before = deployment_service_status(name)
    if action == "start" and not args.dry_run:
        # "start" means make sure it is running, so a running service is
        # already the answer. Bringing it down and up again would interrupt
        # whatever step is in flight, and an interrupted model call or effect
        # can run a second time.
        if service_is_running(before):
            if not bool(before.get("healthy")):
                print(
                    f"Deployment {name} is already restarting but is not "
                    f"healthy ({before.get('detail')}). Inspect "
                    "'zippergen deploy logs'."
                )
                return 1
            print(
                f"Deployment {name} is already running ({before.get('detail')})."
            )
            if getattr(args, "enable", False):
                print(
                    "  Autostart was left unchanged. Run 'zippergen deploy stop' "
                    "then 'zippergen deploy start --enable' to set it."
                )
            return 0
    if action == "start" and not args.dry_run:
        _require_deployment_execution_slot(profile)
        _initialize_deployment_store(profile)
    if (
        action == "start"
        and not args.dry_run
        and not getattr(args, "skip_readiness", False)
    ):
        checks = _doctor_checks(name, include_systemd=False, before_start=True)
        failures = [
            check for check in checks if check.get("status") == "fail"
        ]
        if failures:
            _print_doctor_summary(name, checks)
            print(
                f"Deployment {name} was not {action}ed because readiness "
                "checks found failures."
            )
            return 1
    manager = _service_manager()
    if manager == "systemd":
        unit = _systemd_unit_name(name)
        if action == "start":
            target = _install_systemd_unit(profile, dry_run=args.dry_run)
            if not args.dry_run:
                print(f"Installed systemd unit: {target}")
            _run_systemctl(_systemctl_command("daemon-reload"), dry_run=args.dry_run)
            if args.enable:
                _run_systemctl(_systemctl_command("enable", unit), dry_run=args.dry_run)
        # systemctl rejects `stop` for a unit that is not installed. That is
        # already the requested outcome, just as a missing launchd agent is,
        # so do not turn an idempotent stop into a command failure.
        if not (
            action == "stop"
            and not args.dry_run
            and str(before.get("state") or "unknown") == "not-loaded"
        ):
            _run_systemctl(_systemctl_command(action, unit), dry_run=args.dry_run)
        service = unit
    else:
        label = _launchd_label(name)
        domain = _launchctl_domain()
        service = f"{domain}/{label}"
        if action == "start":
            target = _install_launchd_agent(profile, dry_run=args.dry_run)
            if not args.dry_run:
                print(f"Installed launchd agent: {target}")
            # A stale agent may still be loaded after a crash, so clear it
            # before bootstrapping. A missing prior agent is expected.
            _run_launchctl(
                _launchctl_command("bootout", service),
                dry_run=args.dry_run,
                check=False,
            )
            _run_launchctl(
                _launchctl_command("bootstrap", domain, str(target)),
                dry_run=args.dry_run,
            )
        else:
            # A missing agent is the outcome stop is asking for, not a failure.
            # What actually happened is decided from the state afterwards.
            _run_launchctl(
                _launchctl_command("bootout", service),
                dry_run=args.dry_run,
                check=False,
            )
    if args.dry_run:
        return 0
    # Report what is true afterwards, not which verb was typed. "Stopped" on a
    # service that was already stopped reads as though something happened, and
    # sends people looking for what.
    after = deployment_service_status(name)
    running_before = service_is_running(before)
    running_after = service_is_running(after)
    if action == "start":
        healthy_after = bool(after.get("healthy"))
        if healthy_after and running_after and running_before:
            print(f"Deployment {name} was already running ({service}).")
        elif healthy_after and running_after:
            print(f"Started deployment {name} ({service}).")
        elif healthy_after and str(after.get("state")) == "completed":
            print(f"Deployment {name} completed successfully ({service}).")
        else:
            print(
                f"Deployment {name} did not become healthy ({service}): "
                f"{after['detail']}"
            )
            return 1
    else:
        after_state = str(after.get("state") or "unknown")
        before_state = str(before.get("state") or "unknown")
        if after_state == "unknown":
            print(
                f"Could not confirm deployment {name} stopped ({service}): "
                f"{after.get('detail') or 'service state is unknown'}"
            )
            return 1
        if running_after:
            print(f"Deployment {name} is still running ({service}): {after['detail']}")
            return 1
        if before_state == "unknown":
            print(f"Deployment {name} is stopped ({service}).")
        elif not running_before:
            print(f"Deployment {name} was already stopped ({service}).")
        else:
            print(f"Stopped deployment {name} ({service}).")
    return 0


def _logs_command(args) -> int:
    if args.tail <= 0:
        raise SystemExit("--tail must be greater than 0.")
    if args.interval is not None and not args.follow:
        raise SystemExit("--interval requires --follow.")
    interval = 1.0 if args.interval is None else args.interval
    profile = _load_deployment_profile(args.name)
    log_path = Path(str(profile.get("log") or _default_deployment_log_path(args.name))).expanduser()
    if not log_path.exists():
        print(f"Log does not exist yet: {log_path}")
        return 0

    raw_offset = profile.get("log_generation_offset")
    initial_offset = (
        raw_offset
        if isinstance(raw_offset, int)
        and 0 <= raw_offset <= log_path.stat().st_size
        else 0
    )

    def visible_lines() -> list[str]:
        content = log_path.read_bytes()
        offset = initial_offset if initial_offset <= len(content) else 0
        return content[offset:].decode(errors="replace").splitlines()

    def print_tail() -> int:
        lines = visible_lines()
        for line in lines[-args.tail:]:
            print(line)
        return len(lines)

    seen = print_tail()
    if not args.follow:
        return 0
    while True:
        time.sleep(interval)
        lines = visible_lines()
        for line in lines[seen:]:
            print(line)
        seen = len(lines)


def _print_doctor(name: str, checks: list[dict[str, object]]) -> None:
    print(f"Doctor: {name}")
    for check in checks:
        status = str(check.get("status", "warn")).upper()
        print(f"{status:4} {check.get('name')}: {check.get('detail')}")
    counts = {
        status: sum(1 for check in checks if check.get("status") == status)
        for status in ("ok", "warn", "fail")
    }
    print(f"Summary: {counts['ok']} ok, {counts['warn']} warn, {counts['fail']} fail")


def _print_doctor_summary(
    name: str,
    checks: list[dict[str, object]],
) -> None:
    counts = {
        status: sum(1 for check in checks if check.get("status") == status)
        for status in ("ok", "warn", "fail")
    }
    print(
        f"Readiness {name}: {counts['ok']} ok, "
        f"{counts['warn']} warning(s), {counts['fail']} failure(s)"
    )
    for check in checks:
        if check.get("status") == "fail":
            print(f"FAIL {check.get('name')}: {check.get('detail')}")


def _repair_deployment_permissions(
    name: str,
    profile: Mapping[str, object],
) -> tuple[Path, ...]:
    """Repair only files ZipperGen owns under its managed home."""

    home = _zippergen_home().resolve()
    ensure_private_directory(home)
    repaired: list[Path] = [home]

    directories = {
        _deployments_dir(),
        Path(str(profile.get("store") or "")).expanduser().parent,
        Path(str(profile.get("log") or "")).expanduser().parent,
    }
    for directory in directories:
        try:
            directory.resolve().relative_to(home)
        except ValueError:
            continue
        ensure_private_directory(directory)
        repaired.append(directory)

    private_files = {
        _deployment_profile_path(name): 0o600,
        _deployment_service_path(name): 0o600,
        _deployment_launchd_path(name): 0o600,
        _deployment_script_path(name): 0o700,
    }
    for field in ("store", "log", "secrets_file"):
        raw_path = profile.get(field)
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        try:
            path.resolve().relative_to(home)
        except ValueError:
            continue
        private_files[path] = 0o600
        if field == "store":
            private_files[Path(str(path) + "-wal")] = 0o600
            private_files[Path(str(path) + "-shm")] = 0o600

    for path, mode in private_files.items():
        if path.exists() and not path.is_symlink() and path.is_file():
            path.chmod(mode)
            repaired.append(path)
    return tuple(repaired)


def _doctor_command(args) -> int:
    repaired: tuple[Path, ...] = ()
    if args.repair_permissions:
        profile = _load_deployment_profile(args.name)
        repaired = _repair_deployment_permissions(args.name, profile)
    checks = _doctor_checks(
        args.name,
        include_systemd=not args.no_systemd,
        check_store_integrity=True,
    )
    if repaired:
        checks.append(_doctor_check(
            "ok",
            "permissions repaired",
            f"secured {len(repaired)} managed path(s)",
        ))
    if args.json:
        print(json.dumps({"deployment": args.name, "checks": checks}, default=str, sort_keys=True))
    else:
        _print_doctor(args.name, checks)
    return _check_exit_code(
        args,
        ready=not any(check.get("status") == "fail" for check in checks),
    )


@dataclass(frozen=True)
class _ExecutionReference:
    """One durable execution selected by its owning CLI family."""

    store: str
    subject: str
    status: str | None = None
    updated_at: str | None = None
    service: Mapping[str, object] | None = None
    freshness: tuple[dict[str, object], ...] = ()


def _resolve_execution_reference(args) -> _ExecutionReference:
    """Resolve durable state and retain enough context to identify it."""

    from zippergen.workspace import Workspace

    if getattr(args, "execution_owner", "run") == "deploy":
        from zippergen.deployment_platform import deployment_service_status

        profile = _load_deployment_profile(_resolved_deployment_name(args))
        return _ExecutionReference(
            store=str(profile["store"]),
            subject="project deployment",
            updated_at=(
                str(profile["updated_at"])
                if profile.get("updated_at")
                else None
            ),
            service=deployment_service_status(str(profile["name"])),
            freshness=tuple(deployment_freshness_checks(profile)),
        )
    workspace = Workspace(getattr(args, "project", None))
    record = workspace.current_run()
    if record is None:
        raise SystemExit(
            "There is no current durable run. Start one with "
            "'zippergen run --durable'."
        )
    return _ExecutionReference(
        store=str(record["store"]),
        subject=f"durable run {record['run_id']}",
        status=str(record.get("status") or "unknown"),
        updated_at=(
            str(record["updated_at"])
            if record.get("updated_at")
            else None
        ),
    )


def _resolve_store_arg(args) -> str:
    """Resolve durable state through its owning run or deployment."""

    return _resolve_execution_reference(args).store


def _print_execution_reference(reference: _ExecutionReference) -> None:
    from zippergen.rendering import TerminalRenderer

    renderer = TerminalRenderer()
    renderer.section("Execution")
    renderer.emit(f"Subject: {reference.subject}")
    if reference.status is not None:
        renderer.emit(f"Status: {reference.status}")
    if reference.updated_at is not None:
        renderer.emit(f"Updated: {reference.updated_at}")
    if reference.service is not None:
        renderer.emit(f"Service: {_service_summary(reference.service)}")
    for check in reference.freshness:
        marker = "OK" if check.get("status") == "ok" else "WARN"
        renderer.emit(
            f"{marker} {check.get('name', 'freshness')}: "
            f"{check.get('detail', 'no detail')}"
        )
    renderer.emit(f"Store: {reference.store}")
    renderer.emit()


def _fmt_time(ts: float | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _load_human_tasks(
    store_path: str,
    *,
    status: str | None = "pending",
    limit: int | None = None,
    with_tokens: bool = False,
    token_channel: str = "cli",
) -> list[dict]:
    conn = open_store(str(Path(store_path).expanduser()))
    try:
        query = (
            "SELECT task_id FROM human_tasks "
            + ("WHERE status=? " if status is not None else "")
            + "ORDER BY updated_at DESC, task_id"
        )
        params: tuple[object, ...] = (status,) if status is not None else ()
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        rows = conn.execute(query, params).fetchall()
        tasks = []
        for row in rows:
            task = load_human_task(conn, row[0])
            if task is not None:
                if with_tokens:
                    record = ensure_human_task_token(conn, task["task_id"], channel=token_channel)
                    task["token"] = record["token"]
                    task["token_channel"] = record["channel"]
                tasks.append(task)
        return tasks
    finally:
        conn.close()


def _short_text(value: object, *, limit: int = 120) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _print_trace_events(events: list[dict]) -> None:
    from zippergen.rendering import TerminalRenderer

    renderer = TerminalRenderer()
    count = len(events)
    renderer.columns(
        f"Trace ({count} event{'s' if count != 1 else ''})",
        ("Time", "#", "Participant", "Event", "Detail"),
        _trace_rows(events),
    )


_TRACE_HEADERS = ("Time", "#", "Participant", "Event", "Detail")


def _print_trace_follow_table(events: list[dict]) -> tuple[int, ...]:
    """Print and retain one table layout for the whole followed trace."""

    from zippergen.rendering import TerminalRenderer

    renderer = TerminalRenderer()
    rows = _trace_rows(events)
    largest_rowid = max((int(event["rowid"]) for event in events), default=0)
    id_digits = max(4, len(str(largest_rowid)) + 1)
    layout_hint = (
        "0000-00-00 00:00:00.000+00:00",
        "#" + "0" * id_digits,
        "Participant",
        "assistant failed",
        "D" * renderer.data_output_columns(),
    )
    widths = renderer.column_widths(_TRACE_HEADERS, [*rows, layout_hint])
    count = len(events)
    renderer.columns(
        f"Trace ({count} event{'s' if count != 1 else ''})",
        _TRACE_HEADERS,
        rows,
        widths=widths,
    )
    return widths


def _print_trace_follow_events(
    events: list[dict],
    widths: tuple[int, ...],
) -> None:
    """Append trace rows without reprinting the table banner each poll."""

    from zippergen.rendering import TerminalRenderer

    # A completion commonly lands in the next polling batch.  A newly seen
    # start is therefore only a start, not evidence of an incomplete attempt.
    # Static snapshots still mark starts whose terminal event is absent.
    rows = _trace_rows(events, mark_unmatched_incomplete=False)
    renderer = TerminalRenderer(output=lambda value: print(value, flush=True))
    renderer.column_rows(_TRACE_HEADERS, rows, widths)


def _print_tasks(tasks: list[dict], *, heading: str) -> None:
    print(f"{heading}: {len(tasks)}")
    for task in tasks:
        spec = task.get("spec") or {}
        rendered = spec.get("rendered") or {}
        output = spec.get("output")
        output_type = spec.get("output_type")
        print(
            f"{task['task_id']} {task['role']}.{task['action']} "
            f"{spec.get('kind', 'human')} -> {output}: {output_type} "
            f"status={task['status']} updated={_fmt_time(task['updated_at'])}"
        )
        if task.get("token"):
            print(f"  token[{task.get('token_channel', 'default')}]: {task['token']}")
        instruction = rendered.get("instruction")
        context = rendered.get("context")
        prefill = rendered.get("prefill")
        if instruction:
            print(f"  instruction: {_short_text(instruction)}")
        if context:
            print(f"  context: {_short_text(context)}")
        if prefill:
            print(f"  prefill: {_short_text(prefill)}")


def _notify_stdout_task(task: dict) -> None:
    spec = task.get("spec") or {}
    rendered = spec.get("rendered") or {}
    print("=" * 72)
    print(f"Human task: {task['task_id']}")
    print(f"Action: {task['role']}.{task['action']} ({spec.get('kind', 'human')})")
    instruction = rendered.get("instruction")
    context = rendered.get("context")
    prefill = rendered.get("prefill")
    if instruction:
        print("\nInstruction:")
        print(instruction)
    if context:
        print("\nContext:")
        print(context)
    if prefill:
        print("\nPrefill:")
        print(prefill)
    task_id = task["task_id"]
    kind = spec.get("kind")
    if kind == "confirm":
        print("\nApprove:")
        print(f"  zippergen deploy approve --task {task_id} --yes")
        print("Decline:")
        print(f"  zippergen deploy approve --task {task_id} --no")
    elif kind == "ack":
        print("\nAcknowledge:")
        print(f"  zippergen deploy approve --task {task_id}")
    else:
        print("\nRespond:")
        print(
            "  zippergen deploy approve "
            f"--task {task_id} --value '<value>'"
        )


def _print_status(status: dict[str, object]) -> None:
    print(f"Store: {status['store']}")
    # "Durable state", not "State": a deployment also has a service state, and
    # two lines both called State is exactly the confusion to avoid.
    print(f"Durable state: {status['state']} ({status['summary']})")
    if not status.get("exists"):
        return

    roles = status.get("roles")
    if isinstance(roles, list):
        print(f"Roles: {len(roles)}")
        for role in roles:
            detail = role.get("detail") or {}
            action = detail.get("action") if isinstance(detail, dict) else None
            activity = f" in {action}" if action else ""
            duration = (
                f" for {_execution_age(role.get('updated_at'))}"
                if str(role.get("status") or "").startswith("running_")
                else ""
            )
            print(
                f"  {role['role']}: {role['status']}{activity}{duration} "
                f"after {role['steps']} step(s)"
            )

    last_failure = status.get("last_failure")
    if isinstance(last_failure, dict):
        role = last_failure.get("role") or "unknown lifeline"
        error = last_failure.get("error") or "Error"
        message = last_failure.get("message") or "no detail recorded"
        occurred = _fmt_time(last_failure.get("recorded_at"))
        if last_failure.get("historical") is True:
            recovered = _fmt_time(last_failure.get("recovered_at"))
            print(
                f"Earlier failure (recovered): {role} · {error}: {message} · "
                f"{occurred}; workflow completed {recovered}"
            )
        else:
            print(f"Last failure: {role} · {error}: {message} · {occurred}")

    connectors = status.get("connectors")
    if isinstance(connectors, list) and connectors:
        for connector in connectors:
            if connector.get("healthy"):
                print(f"Connector {connector['connector']}: reaching its provider")
            else:
                print(
                    f"Connector {connector['connector']}: FAILING since "
                    f"{_fmt_time(connector.get('since'))} "
                    f"- {connector.get('detail')}"
                )

    outstanding = status.get("outstanding_messages")
    if isinstance(outstanding, list):
        print(f"Outstanding messages: {len(outstanding)}")
        for message in outstanding[:10]:
            print(
                f"  #{message['id']} {message['sender']}->{message['receiver']} "
                f"on {message['channel']}"
            )

    history = status.get("history")
    if isinstance(history, dict):
        keep = int(history.get("keep") or 0)
        rows = int(history.get("rows") or 0)
        if keep == 0:
            print("Trace history: off (this store records none)")
        else:
            print(f"Trace history: {rows:,} of {keep:,} rows kept")

    tasks = status.get("pending_human_tasks")
    if isinstance(tasks, list):
        print(f"Pending human tasks: {len(tasks)}")
        for task in tasks[:10]:
            print(
                f"  {task['task_id']} {task['role']}.{task['action']} "
                f"updated {_fmt_time(task['updated_at'])}"
            )

    results = status.get("workflow_results")
    if isinstance(results, list):
        print(f"Workflow results: {len(results)}")
        for result in results[:10]:
            print(
                f"  {result['workflow']} = {json.dumps(result['value'], default=str)} "
                f"updated {_fmt_time(result['updated_at'])}"
            )


def _deployment_name_for_project(args) -> str | None:
    """Return this project's deployment name, or ``None`` when it has none.

    The stable workspace name belongs to the checkout's local identity.
    Reinitializing a path therefore cannot inherit the previous project's
    deployment.  A missing profile is the only absence.  An unreadable or
    mismatched profile is an error, because it does not prove that no service
    is running.
    """

    from zippergen.workspace import Workspace

    workspace = Workspace(getattr(args, "project", None))
    name = workspace.directory.name
    path = _deployment_profile_path(name)
    if not path.exists():
        return None
    profile = _load_deployment_profile(name)
    source = profile.get("source_cwd")
    profile_project_id = str(profile.get("project_id") or "")
    project_id = str(workspace.project_manifest().get("project_id") or "")
    if project_id and profile_project_id != project_id:
        raise SystemExit(
            "This deployment belongs to an earlier project identity at the "
            "same path. Inspect it with 'zg deploy list' and remove orphaned "
            "deployments with 'zg deploy prune'."
        )
    if not project_id and (
        profile_project_id
        or not source
        or Path(str(source)).resolve() != workspace.root
    ):
        raise SystemExit(
            f"Deployment profile {path} does not belong to this project. "
            "Remove it and deploy again."
        )
    return name


def _resolved_deployment_name(args) -> str:
    """Return the deployment name for commands that require one."""

    name = _deployment_name_for_project(args)
    if name is None:
        raise SystemExit(
            "This project has no deployment yet. Create one with "
            "'zippergen deploy'."
        )
    return name


def _resolved_workflow_spec(args) -> str:
    """Use the project's workflow when the command line does not name one.

    Convention for the simple case, explicit configuration for the ambiguous
    one, in that order: an explicit argument, then `workflow_entry`, then the
    single workflow in the project if there is exactly one. `zg init` runs
    before any workflow exists, so a beginner should not have to record an
    entry by hand before the first `zg validate`.

    Inference is a convenience only. Nothing here writes to the manifest.
    """

    named = getattr(args, "workflow", None) or getattr(args, "target", None)
    if named:
        return str(named)
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    try:
        resolved = workspace.resolve_workflow(str(named) if named else None)
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    return str(workspace.absolute_spec(resolved))


def _require_project(args):
    """Resolve and require an initialized project for project-scoped commands."""

    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    try:
        workspace.require_project()
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    return workspace


def _execution_conflict_message(
    active: ActiveExecution,
    *,
    requested: str,
) -> str:
    detail = active.owner
    if active.pid is not None:
        detail += f" (PID {active.pid})"
    return (
        f"This project already has an active {detail}. Stop it before "
        f"starting {requested}. Only one run or deployment may execute per "
        "project; observation commands remain available."
    )


@contextmanager
def _hold_project_execution(workspace, *, owner: str):
    """Hold the execution slot shared by a project and its deployment."""

    path = execution_lock_path(workspace.home, workspace.directory.name)
    try:
        with execution_lock(path, owner=owner):
            yield
    except ExecutionLockError as exc:
        raise SystemExit(
            _execution_conflict_message(exc.active, requested=owner)
        ) from None


@contextmanager
def _hold_deployment_execution(profile: Mapping[str, object]):
    """Hold the source project's slot for one supervised deployment process."""

    path = execution_lock_path(_zippergen_home(), str(profile["name"]))
    deadline = time.monotonic() + 30.0
    stack = ExitStack()
    while True:
        try:
            stack.enter_context(
                execution_lock(path, owner="project deployment")
            )
            break
        except ExecutionLockError as exc:
            # Bare deploy starts the supervised process before releasing its
            # publication lock. Waiting here closes the start/redeploy gap:
            # the service sees the published profile, then takes the same
            # execution slot immediately after publication completes.
            if (
                exc.active.owner == "a deployment update"
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
                continue
            raise SystemExit(
                _execution_conflict_message(
                    exc.active,
                    requested="the project deployment",
                )
            ) from None
    try:
        yield
    finally:
        stack.close()


def _require_deployment_execution_slot(
    profile: Mapping[str, object],
) -> None:
    """Reject a public start when a foreground run owns the project."""

    active = active_execution(
        execution_lock_path(_zippergen_home(), str(profile["name"]))
    )
    if (
        active is not None
        and active.owner == "a deployment update"
        and active.pid == os.getpid()
    ):
        return
    if active is not None and active.owner != "project deployment":
        raise SystemExit(
            _execution_conflict_message(
                active,
                requested="the project deployment",
            )
        )


@contextmanager
def _hold_deployment_mutation(name: str, *, owner: str):
    """Hold the same slot the supervised process needs for a mutation."""

    path = execution_lock_path(_zippergen_home(), name)
    try:
        with execution_lock(path, owner=owner):
            yield
    except ExecutionLockError as exc:
        raise SystemExit(
            _execution_conflict_message(exc.active, requested=owner)
        ) from None


def _run_workflow_command(args) -> int:
    from zippergen.workflow_io import project_directory
    from zippergen.workspace import Workspace

    workspace = (
        Workspace(Path.cwd())
        if getattr(args, "store", None)
        else Workspace(getattr(args, "project", None))
    )
    with project_directory(workspace.root):
        if getattr(args, "store", None):
            return _run_workflow_from_project(args, workspace)
        with _hold_project_execution(workspace, owner="foreground run"):
            return _run_workflow_from_project(args, workspace)


def _run_workflow_from_project(args, workspace) -> int:
    """Execute a plain run with project-relative paths anchored to its root."""

    args.workflow = _resolved_workflow_spec(args)
    wf, module = load_workflow_spec(args.workflow)
    from zippergen.durable_runs import default_llm_spec
    inputs = _parse_input_json(args.input_json)
    inputs.update(_parse_inputs(args.input))
    # Apply declared defaults in every mode. Ask for a truly missing value only
    # in an interactive terminal, and otherwise fail before starting threads.
    from zippergen.durable_runs import collect_workflow_inputs

    inputs = collect_workflow_inputs(
        wf,
        module,
        inputs,
        interactive=(
            sys.stdin.isatty() and not getattr(args, "yes", False)
        ),
        input_func=input,
        output_func=print,
        workspace=workspace,
    )
    options = _parse_options(args.option)
    routing = project_model_routing(
        workspace,
        workspace.canonical_spec(args.workflow, cwd=workspace.root),
        wf,
        fallback_default=default_llm_spec(module),
    )
    routing = apply_model_overrides(
        routing,
        default_spec=args.llm,
        overrides=_parse_inputs(args.llm_for),
        settings=_idle_timeout_settings(args.llm_idle_timeout_for),
    )
    selected_llm = routing.default_spec
    llms = routing.overrides
    llm_settings = routing.settings
    from zippergen.models import effective_llm_routes, fake_model_notice

    notice = fake_model_notice(effective_llm_routes(wf, selected_llm, llms))
    if notice:
        # Standard output carries the run's result, and a caller may pipe it
        # into a parser. A warning is for the person, so it goes beside it.
        print(notice, file=sys.stderr)
    assistant_routing = project_assistant_routing(
        workspace,
        workspace.canonical_spec(args.workflow, cwd=workspace.root),
        wf,
        module=module,
    )
    assistant_routing = apply_assistant_overrides(
        assistant_routing,
        default_backend=getattr(args, "assistant", None),
        overrides=normalize_assistant_overrides(
            getattr(args, "assistants", None)
        ),
        workflow=wf,
        module=module,
    )

    from zippergen.connector_wiring import (
        ConnectorWiringError,
        connector_runtime,
        human_connector_factory,
    )

    canonical = workspace.canonical_spec(args.workflow, cwd=workspace.root)
    internal_store = getattr(args, "store", None)
    try:
        if internal_store:
            connector_snapshot, connector_environment = {}, {}
        else:
            connector_snapshot, connector_environment = connector_runtime(
                workspace, canonical, wf, module
            )
    except ConnectorWiringError as exc:
        raise SystemExit(str(exc)) from exc
    if connector_snapshot:
        connector_environment["ZIPPERGEN_CONNECTORS_JSON"] = json.dumps(
            connector_snapshot
        )
    runtime_environment = workspace.development_provider_environment(
        selected_llm_specs(selected_llm, llms)
    )
    runtime_environment.update(connector_environment)
    connector_factory = human_connector_factory(
        connector_snapshot, connector_environment
    )

    # A public plain run remains disposable. A configured asynchronous human
    # connector needs SQLite coordination while the process is alive, so use
    # a private temporary store without turning the run into resumable state.
    execution = str(getattr(args, "execution", None) or "memory")
    store_path = internal_store
    connector_thread = None
    connector_stop = None
    temporary_root = None
    if connector_factory is not None:
        temporary_root = Path(tempfile.mkdtemp(prefix="zippergen-run-"))
        store_path = str(temporary_root / "run.sqlite")
        execution = "sqlite"
        connector = connector_factory(store_path)
        connector_stop = threading.Event()
        connector_thread = threading.Thread(
            target=connector.run_forever,
            kwargs={"poll_timeout": 2.0, "stop_event": connector_stop},
            name="connector-human",
            daemon=True,
        )

    config = RunConfig(
        workflow_spec=args.workflow,
        workflow=wf,
        module=module,
        llm=selected_llm,
        llms=llms,
        assistant=assistant_routing.default_backend,
        assistants=assistant_routing.overrides,
        llm_idle_timeout=args.llm_idle_timeout,
        llm_settings=llm_settings,
        store_path=store_path,
        inputs=inputs,
        options=options,
        timeout=args.timeout,
        execution=execution,
    )
    configure_kwargs = {
        "timeout": args.timeout,
        "llm_idle_timeout": args.llm_idle_timeout,
        "llm_settings": {
            target: chosen.as_dict() for target, chosen in llm_settings.items()
        },
        "execution": execution,
        "store_path": store_path,
        "assistant_root": str(workspace.root),
    }
    if connector_factory is not None:
        from zippergen.human_backends import make_sqlite_human_backend

        configure_kwargs["human_backend"] = make_sqlite_human_backend()
    from zippergen.assistant_backends import make_cli_assistant_backend

    configure_kwargs["assistant_backend"] = (
        make_cli_assistant_backend(
            assistant_routing.default_backend,
            project_root=workspace.root,
            routes=assistant_routing.overrides,
        )
        if assistant_routing.overrides
        else make_cli_assistant_backend(
            assistant_routing.default_backend,
            project_root=workspace.root,
        )
    )
    try:
        with temporary_environment(runtime_environment):
            _call_setup_hook(module, config)
            if llms:
                wf.configure(
                    effective_llm_routes(wf, selected_llm, llms),
                    **configure_kwargs,
                )
            else:
                wf.configure(selected_llm, **configure_kwargs)
            if connector_thread is not None:
                connector_thread.start()
                print("External human connector started for this run.")
            result = wf(**inputs)
    finally:
        if connector_stop is not None:
            connector_stop.set()
        if connector_thread is not None:
            connector_thread.join(timeout=15)
        if (
            temporary_root is not None
            and (connector_thread is None or not connector_thread.is_alive())
        ):
            shutil.rmtree(temporary_root, ignore_errors=True)
    print(json.dumps({"result": result}, default=str))
    return 0


def _durable_run_command(args) -> int:
    """Run with a recorded, resumable run.

    Same execution as a plain run; what differs is the bookkeeping. The run is
    registered in the project so `--resume` has something to continue, and any
    inputs the workflow needs but the command line did not supply are asked
    for.
    """

    from zippergen.workflow_io import project_directory
    from zippergen.workspace import Workspace

    workspace = Workspace(getattr(args, "project", None))
    with project_directory(workspace.root):
        with _hold_project_execution(workspace, owner="durable run"):
            return _durable_run_from_project(args, workspace)


def _durable_run_from_project(args, workspace) -> int:
    """Execute a durable run with project-relative paths anchored to its root."""

    from zippergen.connector_wiring import (
        ConnectorWiringError,
        connector_environment_from_snapshot,
        connector_runtime,
        human_connector_factory,
    )
    from zippergen.durable_runs import run_durable
    from zippergen.workspace import WorkspaceError
    inputs = _parse_input_json(args.input_json)
    inputs.update(_parse_inputs(args.input))
    try:
        if args.resume:
            record = workspace.current_run()
            if record is None:
                snapshot: dict[str, object] = {}
            else:
                snapshot = dict(record.get("connectors") or {})
            connector_environment = connector_environment_from_snapshot(
                workspace, snapshot
            )
        else:
            selected = workspace.resolve_workflow(args.workflow)
            workflow, module = load_workflow_spec(
                str(workspace.absolute_spec(selected))
            )
            snapshot, connector_environment = connector_runtime(
                workspace, selected, workflow, module
            )
    except (ConnectorWiringError, WorkspaceError) as exc:
        raise SystemExit(str(exc)) from exc
    if snapshot:
        connector_environment["ZIPPERGEN_CONNECTORS_JSON"] = json.dumps(
            snapshot
        )
    if args.history_keep is not None and args.history_keep < 0:
        raise SystemExit("--history-keep must be zero or greater.")
    run_durable(
        workspace,
        workflow_spec=args.workflow,
        resume=args.resume,
        history_keep=args.history_keep,
        provided_inputs=inputs,
        llm=args.llm,
        llms=normalize_llm_overrides(_parse_inputs(args.llm_for)),
        llm_idle_timeout=args.llm_idle_timeout,
        llm_settings=_idle_timeout_settings(
            args.llm_idle_timeout_for
        ),
        assistant=None,
        assistants=normalize_assistant_overrides(
            getattr(args, "assistants", None)
        ),
        options=_parse_options(args.option),
        timeout=args.timeout,
        interactive=not args.yes and sys.stdin.isatty(),
        input_func=input,
        output_func=print,
        human_connector_factory=human_connector_factory(
            snapshot, connector_environment
        ),
        connector_environment=connector_environment,
        connector_snapshot=snapshot if not args.resume else None,
    )
    return 0


def _view_options_from_args(args) -> ViewOptions:
    agents = tuple(
        name.strip()
        for name in str(args.agents or "").split(",")
        if name.strip()
    )
    return ViewOptions(
        detail=args.detail,
        communications_only=args.communications,
        agent=args.agent,
        agents=agents,
    )


def _show_command(args) -> int:
    args.workflow = _resolved_workflow_spec(args)
    workflow, module = load_workflow_spec(args.workflow)
    options = _view_options_from_args(args)
    try:
        data = workflow_view_data(workflow, module, options=options)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.format == "json":
        print(json.dumps(data, indent=2, default=str))
    else:
        print(str(data["code"]))
    return 0


def _validate_command(args) -> int:
    args.workflow = _resolved_workflow_spec(args)
    workflow, module = load_workflow_spec(args.workflow)
    result = _validate_workflow(workflow, module)
    from zippergen.project_configuration import configuration_report
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    try:
        project_workflow = workspace.resolve_workflow()
    except WorkspaceError:
        project_workflow = None
    validated_workflow = workspace.canonical_spec(args.workflow, cwd=Path.cwd())
    if project_workflow == validated_workflow:
        try:
            report = configuration_report(workspace, include_site_checks=False)
        except WorkspaceError as exc:
            configuration_checks = [
                {
                    "status": "fail",
                    "name": "project configuration",
                    "detail": str(exc),
                }
            ]
        else:
            raw_configuration_checks = report["checks"]
            assert isinstance(raw_configuration_checks, list)
            configuration_checks = [
                item
                for item in raw_configuration_checks
                if isinstance(item, dict)
                and str(item.get("name") or "") != "workflow"
            ]
        checks = result["checks"]
        assert isinstance(checks, list)
        checks.extend(configuration_checks)
        result["valid"] = not any(
            isinstance(item, dict) and item.get("status") == "fail"
            for item in checks
        )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        verdict = "valid" if result["valid"] else "invalid"
        print(f"Workflow {workflow.name}: {verdict}")
        for check in result["checks"]:  # type: ignore[union-attr]
            print(f"{str(check['status']).upper():4} {check['name']}: {check['detail']}")
    return 0 if result["valid"] else 1


def _workflow_command(args) -> int:
    """Show or explicitly select the workflow this project is about."""

    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    if args.workflow_action is None:
        entry = workspace.workflow_entry
        if entry:
            print(entry)
            return 0
        try:
            inferred = workspace.resolve_workflow()
        except WorkspaceError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"{inferred} (inferred)")
        return 0

    try:
        spec = _guided_required_value(
            args.spec,
            label="Workflow",
            command="zg workflow select SPEC",
            choices=_project_choices("workflow-specs", getattr(args, "project", None)),
        )
        selected = workspace.select_workflow(spec, replace=True)
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Project workflow: {selected}")
    return 0


def _apply_configuration_answers(
    workspace: "Workspace",
    *,
    set_pairs: list[str],
    unset_names: list[str],
    as_json: bool = False,
) -> bool:
    """Answer or forget declared configuration questions, and say what changed.

    Answering a project's questions used to require `zippergen deploy`, which
    builds a bundle and a managed environment as a side effect of writing a
    line to a file. Every other thing in the project manifest has a command
    that writes it without building anything; this is that command.
    """

    if not set_pairs and not unset_names:
        return False

    from zippergen.deployment import deployment_spec_from_module

    workflow_spec = workspace.resolve_workflow()
    _workflow, module = load_workflow_spec(
        workspace.absolute_spec(workflow_spec)
    )
    spec = deployment_spec_from_module(module)
    declared = {field.name: field for field in spec.fields}
    answers = dict(workspace.configuration_values())

    # One field cannot be both answered and forgotten. Applying unsets first
    # and reporting them last made the output disagree with the file.
    both = sorted(set(_parse_inputs(set_pairs)) & set(unset_names))
    if both:
        raise SystemExit(
            "These fields are both set and unset in one command: "
            + ", ".join(both)
            + ". Choose one."
        )

    for name in unset_names:
        field = declared.get(name)
        if field is None:
            raise SystemExit(_unknown_configuration_field(name, declared))
        if field.secret:
            # A secret was never in the project file, so "forgetting" it here
            # would report the verb rather than the outcome -- and on a
            # credential, where an operator may read it as deletion.
            raise SystemExit(_secret_not_in_the_project(name))
        answers.pop(name, None)

    for pair in _parse_inputs(set_pairs).items():
        name, value = pair
        field = declared.get(name)
        if field is None:
            raise SystemExit(_unknown_configuration_field(name, declared))
        if field.secret:
            raise SystemExit(_secret_not_in_the_project(name))
        if field.choices and str(value) not in field.choices:
            raise SystemExit(
                f"Configuration field {name!r} must be one of "
                + ", ".join(field.choices)
                + f"; got {value!r}."
            )
        answers[name] = value

    workspace.write_configuration_values(answers)
    if as_json:
        # A machine-readable request stays machine-readable, whether it reads
        # or writes.
        print(json.dumps(
            {
                "manifest": str(workspace.manifest_path),
                "set": {
                    name: answers[name]
                    for name in sorted(_parse_inputs(set_pairs))
                },
                "unset": sorted(set(unset_names)),
                "configuration": answers,
            },
            indent=2,
            default=str,
        ))
        return True
    for name in sorted(set(_parse_inputs(set_pairs))):
        print(f"Set {name} = {answers[name]!r}")
    for name in sorted(set(unset_names)):
        print(f"Forgot {name}")
    print(f"Stored in {workspace.manifest_path}")
    return True


def _secret_not_in_the_project(name: str) -> str:
    """One sentence for both --set and --unset, because it is one fact."""

    return (
        f"Configuration field {name!r} is secret, so it is not stored in the "
        "project and cannot be set or forgotten here. Provide it in the "
        "deployment environment."
    )


def _unknown_configuration_field(name: str, declared: dict) -> str:
    available = ", ".join(sorted(declared)) or "none"
    return (
        f"This workflow declares no configuration field {name!r}. "
        f"Available: {available}."
    )


def _configuration_command(args) -> int:
    from zippergen.project_configuration import (
        configuration_report,
        render_configuration,
    )
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    changed = _apply_configuration_answers(
        workspace,
        set_pairs=getattr(args, "set", []),
        unset_names=getattr(args, "unset", []),
        as_json=getattr(args, "json", False),
    )
    if changed:
        return 0
    try:
        report = configuration_report(
            workspace,
            live=False,
            include_site_checks=True,
            model_names=tuple(workspace.model_configurations()),
            assistant_names=tuple(workspace.assistant_configurations()),
            connector_names=tuple(workspace.connector_configurations()),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, default=str))
    else:
        render_configuration(report, TerminalRenderer())
    return 0


def _check_command(args) -> int:
    """Perform one live, project-wide readiness check."""

    from zippergen.project_configuration import (
        configuration_report,
        render_readiness,
    )
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import WorkspaceError

    workspace = _require_project(args)
    try:
        report = configuration_report(
            workspace,
            live=True,
            include_site_checks=True,
            model_names=tuple(workspace.model_configurations()),
            assistant_names=tuple(workspace.assistant_configurations()),
            connector_names=tuple(workspace.connector_configurations()),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, default=str))
    else:
        render_readiness(report, TerminalRenderer())
    # Reading a report is not an error, so the exit code says whether the check
    # ran, not what it found. Scripts that want a gate ask for one.
    return _check_exit_code(args, ready=bool(report["valid"]))


#: Each family renames the same way: pick an existing name, give a free one.
#: The differences are only which list to offer and which method to call.
_RENAMEABLE = {
    "provider": (
        "provider connection",
        "provider-connections",
        "rename_provider_connection",
        None,
    ),
    "model": (
        "model configuration",
        "model-configurations",
        "rename_model_configuration",
        {"mock"},
    ),
    "connector": (
        "connector configuration",
        "connector-configurations",
        "rename_connector_configuration",
        None,
    ),
    "assistant": (
        "assistant configuration",
        "assistant-configurations",
        "rename_assistant_configuration",
        None,
    ),
}


def _rename_command(args, family: str) -> int:
    """Rename one saved configuration, and everything that referred to it.

    The point of the command is that it is one step. Doing it by hand means
    creating the new name, re-pointing every reference, then removing the old
    one, with the project inconsistent in between -- and for a provider
    connection it also strands the credential, which is keyed by that name.
    """

    from zippergen.workspace import Workspace, WorkspaceError

    subject, completion_kind, method, reserved = _RENAMEABLE[family]
    workspace = Workspace(getattr(args, "project", None))
    choices = _project_choices(completion_kind, args.project)
    try:
        old = _guided_required_value(
            args.name,
            label=subject.capitalize(),
            command=f"zg {family} rename OLD NEW",
            choices=choices,
        )
        new = _guided_required_value(
            args.new_name,
            label="New name",
            command=f"zg {family} rename OLD NEW",
            check=_name_check(subject, reserved),
        )
        saved = getattr(workspace, method)(old, new)
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Renamed {subject} {old} to {saved}.")
    if family == "provider":
        print("Its credential and endpoint moved with it.")
    return 0


def _check_exit_code(args, *, ready: bool) -> int:
    """Every ``check`` command ends here, so they cannot drift apart.

    A check reports; it does not gate. Running one by hand and having the
    shell mark it as failed is misleading: nothing went wrong, the command
    did exactly what was asked and printed what it found. A script that wants
    a gate says so with --strict.
    """

    return 1 if getattr(args, "strict", False) and not ready else 0


def _suggested_configuration_name(
    basis: str,
    existing: Mapping[str, object],
    *,
    check: Callable[[str], str | None],
) -> str | None:
    """Offer a name once the deciding field is known.

    Every one of these dialogues ends by asking for a name, which is the only
    value the user has to invent rather than choose. Once the deciding answer
    is on the screen there is usually an obvious name, so offer it -- but only
    when it is legal and free. Suggesting a name that already exists would
    invite reconfiguring something by accident.
    """

    candidate = basis.strip()
    if check(candidate) is not None or candidate in existing:
        return None
    return candidate


def _name_check(
    subject: str,
    reserved: set[str] | None = None,
) -> Callable[[str], str | None]:
    """Ask the workspace whether a name is acceptable, before saving anything."""

    from zippergen.workspace import configuration_name_problem

    return lambda text: configuration_name_problem(
        text, subject=subject, reserved=reserved
    )


def _guided_required_value(
    value: object,
    *,
    label: str,
    command: str,
    choices: tuple[str, ...] = (),
    default: str | None = None,
    check: Callable[[str], str | None] | None = None,
    enforce_choices: bool = True,
) -> str:
    """Return a required CLI value, prompting only in a human terminal.

    A value passed on the command line is returned as it stands, because the
    save validates it and there is nobody at a prompt to correct it. A value
    typed at a prompt is different: the person is still sitting there, so a
    wrong answer is explained and asked again instead of ending the dialogue
    over a typo. An empty answer, or Ctrl-C, still leaves without saving.
    """

    entered = str(value or "").strip()
    if entered:
        # Reject a bad value now, not after two more questions. The list is
        # empty when the workflow could not be loaded, and an empty list is
        # ignorance rather than a verdict, so it never rejects anything.
        problem = check(entered) if check is not None else None
        if problem is None and enforce_choices and choices and entered not in choices:
            problem = (
                f"Unknown {label.casefold()} {entered!r}. Available: "
                + ", ".join(choices)
            )
        if problem is not None:
            raise SystemExit(problem)
        return entered
    if not sys.stdin.isatty():
        if default:
            return default
        raise SystemExit(
            f"{label} is required. Pass it explicitly with: {command}"
        )
    if choices:
        print(f"Available {label.casefold()}s: {', '.join(choices)}")
        # One candidate is its own suggestion. It is offered, not taken: the
        # value still only applies if the person presses Enter on it.
        if default is None and len(choices) == 1:
            default = choices[0]
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            entered = input(f"{label}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("Cancelled. Nothing was saved.") from None
        if not entered and default:
            entered = default
        if not entered:
            raise SystemExit(f"{label} is required. Nothing was saved.")
        if enforce_choices and choices and entered not in choices:
            problem = (
                f"Unknown {label.casefold()} {entered!r}. Available: "
                + ", ".join(choices)
            )
        else:
            problem = check(entered) if check is not None else None
        if problem is None:
            return entered
        print(problem)


def _project_choices(kind: str, project: str | None) -> tuple[str, ...]:
    """Return live project choices used by both completion and prompts."""

    from zippergen.completion import completion_candidates

    return tuple(completion_candidates(kind, project))


def _required_provider_connections(
    kind: str,
    project: str | None,
    *,
    purpose: str,
    example: str,
) -> tuple[str, ...]:
    """Return compatible connections or explain the prerequisite precisely."""

    choices = _project_choices(kind, project)
    if choices:
        return choices
    raise SystemExit(
        f"No {purpose}-capable provider connection is configured.\n\n"
        "Create one first, for example:\n\n"
        f"  {example}\n\n"
        "Then run this command again."
    )


def _provider_set_credential_command(workspace, connection: object) -> int:
    """Prompt for the private credential owned by one provider connection."""

    from zippergen.provider_connections import (
        provider_credential_field,
        provider_credential_label,
        provider_standard_environment,
    )
    from zippergen.workspace import WorkspaceError

    connections = workspace.provider_connections()
    selected = _guided_required_value(
        connection,
        label="Provider connection",
        command="zg provider set-credential CONNECTION",
        choices=tuple(sorted(connections)),
    )
    profile = connections.get(selected)
    if profile is None:
        raise WorkspaceError(f"Provider connection does not exist: {selected}.")
    kind = str(profile.get("kind") or "")
    field = provider_credential_field(kind)
    label = provider_credential_label(kind)
    if field is None:
        print(f"Provider connection {selected} does not require a credential.")
        return 0
    if kind == "google":
        print("Google uses browser authorization rather than a pasted API key.")
        print(f"Run: zg provider authorize {selected}")
        return 0
    try:
        value = getpass.getpass(f"{label} (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("Cancelled. Nothing was saved.") from None
    if not value:
        raise SystemExit("No credential entered. Nothing was saved.")
    workspace.save_provider_secret(selected, field, value)
    environment = provider_standard_environment(kind)
    print(f"Saved {label} for provider connection {selected!r}.")
    print(f"Private location: {workspace.secrets_path} (owner-only file).")
    if environment:
        print(
            f"When no private value is saved, this connection may instead "
            f"read {environment} from the process environment."
        )
    return 0


def _model_command(args) -> int:
    from zippergen.project_configuration import (
        assign_model,
        configuration_report,
        configuration_scope_valid,
        configure_model,
        render_model_configuration,
    )
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    action = getattr(args, "model_action", None)
    try:
        if action == "configure":
            connections = _required_provider_connections(
                "provider-connections-model",
                args.project,
                purpose="model",
                example="zg provider configure openai-main openai",
            )
            # Which connection, then which model on it, then what to call the
            # pair. The first two are answers about the world; the name is the
            # one thing the user invents, so it is asked once the model it
            # names is on the screen and can supply the default.
            configurations = workspace.model_configurations()
            existing = configurations.get(str(args.name or "")) or {}
            connection = _guided_required_value(
                args.connection,
                label="Provider connection",
                command="zg model configure NAME CONNECTION MODEL",
                choices=connections,
                default=str(existing.get("connection") or "") or None,
            )
            model = _guided_required_value(
                args.model,
                label="Model name or path",
                command="zg model configure NAME CONNECTION MODEL",
                default=str(existing.get("model") or "") or None,
            )
            check = _name_check("model configuration", {"mock"})
            name = _guided_required_value(
                args.name,
                label="Model configuration name",
                command="zg model configure NAME CONNECTION MODEL",
                default=_suggested_configuration_name(
                    model, configurations, check=check
                ),
                check=check,
            )
            existing = configurations.get(name) or existing
            idle_timeout = args.idle_timeout
            if idle_timeout is None and existing.get("idle_timeout") is not None:
                idle_timeout = float(str(existing["idle_timeout"]))
            temperature = args.temperature
            if temperature is None and existing.get("temperature") is not None:
                temperature = float(str(existing["temperature"]))
            max_tokens = args.max_tokens
            if max_tokens is None and existing.get("max_tokens") is not None:
                max_tokens = int(float(str(existing["max_tokens"])))
            timeout = args.timeout
            if timeout is None and existing.get("timeout") is not None:
                timeout = float(str(existing["timeout"]))
            value = configure_model(
                workspace,
                name,
                connection,
                model,
                idle_timeout=idle_timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            print(
                f"Saved model configuration {name}: "
                f"{value['connection']} / {value['model']}"
            )
            return 0
        if action in {"assign", "unassign"}:
            target = _guided_required_value(
                args.target,
                label="Model assignment target",
                command=f"zg model {action} TARGET"
                + (" CONFIGURATION" if action == "assign" else ""),
                choices=_project_choices("model-targets", args.project),
            )
            configuration = None
            if action == "assign":
                configuration = _guided_required_value(
                    args.configuration,
                    label="Model configuration",
                    command="zg model assign TARGET CONFIGURATION",
                    choices=_project_choices(
                        "model-configurations", args.project
                    ),
                )
            assign_model(workspace, target, configuration)
            if configuration is None:
                print(f"Removed model assignment for {target}.")
            else:
                print(f"Assigned {target} to model configuration {configuration}.")
            return 0
        if action == "remove":
            name = _guided_required_value(
                args.name,
                label="Model configuration",
                command="zg model remove NAME",
                choices=_project_choices("model-configurations", args.project),
            )
            workspace.remove_model_configuration(name)
            print(f"Removed model configuration {name}.")
            return 0
        if action == "check" and args.name:
            if args.name not in workspace.model_configurations():
                raise WorkspaceError(
                    f"Model configuration does not exist: {args.name}."
                )
        selected_name = str(getattr(args, "name", "") or "")
        report = configuration_report(
            workspace,
            live=action == "check",
            include_site_checks=True,
            model_names=(selected_name,)
            if selected_name
            else tuple(workspace.model_configurations()),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    render_model_configuration(
        report,
        TerminalRenderer(),
        show_checks=action == "check",
    )
    if action != "check":
        return 0
    return _check_exit_code(
        args, ready=configuration_scope_valid(report, "model")
    )


def _provider_command(args) -> int:
    """Show and manage named provider identities shared by configurations."""

    from zippergen.project_configuration import (
        configuration_report,
        configuration_scope_valid,
        render_provider_configuration,
    )
    from zippergen.provider_connections import PROVIDER_KINDS
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    action = getattr(args, "provider_action", None)
    try:
        if action == "configure":
            # Ask the kind first: it decides whether an endpoint is wanted, and
            # it is the question with a menu. The name is the only invented
            # value here, so it is asked last, once there is something to name
            # and a sensible default to offer.
            connections = workspace.provider_connections()
            named = str(args.name or "")
            existing = connections.get(named) or {}
            kind = _guided_required_value(
                args.kind,
                label="Provider kind",
                command="zg provider configure NAME KIND",
                choices=PROVIDER_KINDS,
                default=str(existing.get("kind") or "") or None,
            )
            check = _name_check("provider connection")
            name = _guided_required_value(
                args.name,
                label="Provider connection name",
                command="zg provider configure NAME KIND",
                default=_suggested_configuration_name(
                    f"{kind}-main", connections, check=check
                ),
                check=check,
            )
            existing = connections.get(name) or existing
            base_url = args.base_url
            if kind == "local":
                base_url = _guided_required_value(
                    base_url,
                    label="OpenAI-compatible base URL",
                    command="zg provider configure NAME local --base-url URL",
                    default=str(existing.get("base_url") or "")
                    or "http://127.0.0.1:11434/v1",
                )
            elif base_url:
                raise WorkspaceError("--base-url is only valid for local providers.")
            saved = workspace.save_provider_connection(
                name,
                {"kind": kind, **({"base_url": base_url} if base_url else {})},
            )
            print(f"Saved provider connection {name}: {saved['kind']}.")
            from zippergen.provider_connections import provider_credential_field

            if provider_credential_field(kind):
                if kind == "google":
                    print(
                        f"Authorize it with: zg provider authorize {name}"
                    )
                else:
                    print(
                        f"Add its credential with: "
                        f"zg provider set-credential {name}"
                    )
            return 0
        if action == "set-credential":
            return _provider_set_credential_command(workspace, args.name)
        if action == "remove":
            name = _guided_required_value(
                args.name,
                label="Provider connection",
                command="zg provider remove NAME",
                choices=_project_choices("provider-connections", args.project),
            )
            # Removing the connection also deletes the credential stored
            # under it. That is the right behaviour -- a credential with no
            # connection is unreachable -- but it is not what "remove an
            # unused connection" leads a person to expect, and an OAuth
            # authorization is not always cheap to redo.
            stored = [
                key
                for key in workspace.load_secrets()
                if key.startswith(f"provider:{name}:")
            ]
            if stored and not args.yes:
                print(
                    f"Provider connection {name} holds a stored credential. "
                    "Removing the connection deletes it."
                )
                if not sys.stdin.isatty():
                    raise SystemExit(
                        "Deleting a stored credential requires confirmation. "
                        "Re-run with --yes."
                    )
                answer = input(
                    f"Remove {name} and delete its credential? [y/N]: "
                ).strip().casefold()
                if answer not in {"y", "yes"}:
                    print("Nothing was changed.")
                    return 1
            workspace.remove_provider_connection(name)
            if stored:
                print(
                    f"Removed provider connection {name} and deleted its "
                    "stored credential."
                )
            else:
                print(f"Removed provider connection {name}.")
            return 0
        if action == "check" and args.name:
            if args.name not in workspace.provider_connections():
                raise WorkspaceError(
                    f"Provider connection does not exist: {args.name}."
                )
        selected = str(getattr(args, "name", "") or "")
        report = configuration_report(
            workspace,
            live=action == "check",
            include_site_checks=True,
            provider_names=(selected,)
            if selected
            else tuple(workspace.provider_connections()),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    render_provider_configuration(
        report,
        TerminalRenderer(),
        show_checks=action == "check",
    )
    if action != "check":
        return 0
    return _check_exit_code(
        args, ready=configuration_scope_valid(report, "provider")
    )


def _assistant_command(args) -> int:
    """Show and manage named coding-assistant backend assignments."""

    from zippergen.project_configuration import (
        assign_assistant,
        assistant_target_problem,
        configuration_report,
        configuration_scope_valid,
        configure_assistant,
        render_assistant_configuration,
    )
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    action = getattr(args, "assistant_action", None)
    try:
        if action == "configure":
            # The backend is the choice; the name only labels it.
            configurations = workspace.assistant_configurations()
            existing = configurations.get(str(args.name or "")) or {}
            backend = _guided_required_value(
                args.backend,
                label="Assistant backend",
                command="zg assistant configure NAME BACKEND",
                choices=ASSISTANT_BACKENDS,
                default=str(existing.get("backend") or "") or None,
            )
            check = _name_check("assistant configuration")
            name = _guided_required_value(
                args.name,
                label="Assistant configuration name",
                command="zg assistant configure NAME BACKEND",
                default=_suggested_configuration_name(
                    f"{backend}-main", configurations, check=check
                ),
                check=check,
            )
            value = configure_assistant(workspace, name, backend)
            print(
                f"Saved assistant configuration {name}: "
                f"{value['backend']}"
            )
            spec = assistant_backend_spec(value["backend"])
            print(
                "Authentication remains managed by "
                f"{spec.label if spec else value['backend']}."
            )
            return 0
        if action in {"assign", "unassign"}:
            target = _guided_required_value(
                args.target,
                label="Assistant assignment target",
                command=f"zg assistant {action} TARGET"
                + (" CONFIGURATION" if action == "assign" else ""),
                choices=_project_choices("assistant-targets", args.project),
                check=lambda entered: assistant_target_problem(
                    workspace, entered
                ),
            )
            configuration = None
            if action == "assign":
                configuration = _guided_required_value(
                    args.configuration,
                    label="Assistant configuration",
                    command="zg assistant assign TARGET CONFIGURATION",
                    choices=_project_choices(
                        "assistant-configurations", args.project
                    ),
                )
            assign_assistant(workspace, target, configuration)
            if configuration is None:
                print(f"Removed assistant assignment for {target}.")
            else:
                print(
                    f"Assigned {target} to assistant configuration "
                    f"{configuration}."
                )
            return 0
        if action == "remove":
            name = _guided_required_value(
                args.name,
                label="Assistant configuration",
                command="zg assistant remove NAME",
                choices=_project_choices(
                    "assistant-configurations", args.project
                ),
            )
            workspace.remove_assistant_configuration(name)
            print(f"Removed assistant configuration {name}.")
            return 0
        if action == "check" and args.name:
            if args.name not in workspace.assistant_configurations():
                raise WorkspaceError(
                    f"Assistant configuration does not exist: {args.name}."
                )
        selected_name = str(getattr(args, "name", "") or "")
        report = configuration_report(
            workspace,
            include_site_checks=True,
            assistant_names=(
                (selected_name,)
                if selected_name
                else tuple(workspace.assistant_configurations())
            ),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    render_assistant_configuration(
        report,
        TerminalRenderer(),
        show_checks=action == "check",
    )
    if action != "check":
        return 0
    return _check_exit_code(
        args, ready=configuration_scope_valid(report, "assistant")
    )


def _connector_management_command(args) -> int:
    from zippergen.project_configuration import (
        assign_connector,
        configuration_report,
        connector_target_problem,
        configuration_scope_valid,
        render_connector_configuration,
    )
    from zippergen.rendering import TerminalRenderer
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    action = getattr(args, "connector_action", None)
    try:
        if action == "unassign":
            target = _guided_required_value(
                args.target,
                label="Connector target",
                command="zg connector unassign TARGET",
                choices=_project_choices("connector-targets", args.project),
                check=lambda entered: connector_target_problem(
                    workspace, entered
                ),
            )
            kind = assign_connector(workspace, target, None)
            print(f"Removed the {kind} assignment for {target}.")
            return 0
        if action == "remove":
            name = _guided_required_value(
                args.name,
                label="Connector configuration",
                command="zg connector remove NAME",
                choices=_project_choices(
                    "connector-configurations", args.project
                ),
            )
            workspace.remove_connector_configuration(name)
            print(f"Removed connector configuration {name}.")
            return 0
        if action == "check" and args.name:
            if args.name not in workspace.connector_configurations():
                raise WorkspaceError(
                    f"Connector configuration does not exist: {args.name}."
                )
        selected_name = str(getattr(args, "name", "") or "")
        report = configuration_report(
            workspace,
            live=action == "check",
            include_site_checks=True,
            connector_names=(selected_name,)
            if selected_name
            else tuple(workspace.connector_configurations()),
        )
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    render_connector_configuration(
        report,
        TerminalRenderer(),
        show_checks=action == "check",
    )
    if action != "check":
        return 0
    return _check_exit_code(
        args, ready=configuration_scope_valid(report, "connector")
    )


SPECIFICATION_TEMPLATE = """# {name}

What should this workflow do? Describe the participants, what they exchange,
and in what order. Plain prose is fine — this is the statement of intent, not
a formal document.
"""


def _remove_command(args) -> int:
    """Delete a deployment. Its managed store survives unless --purge."""

    with _hold_deployment_mutation(
        args.name, owner="a deployment removal"
    ):
        try:
            enforce_deploy_requirement("remove", args.name)
        except ServiceIsLiveError as exc:
            raise SystemExit(str(exc)) from exc
        return _remove_command_locked(args)


def _remove_command_locked(args) -> int:

    from zippergen.deployments import (
        DeploymentRemovalError,
        present_deployment_artifacts,
        remove_deployment_artifacts,
        unregister_deployment_service,
    )

    profile = _load_deployment_profile(args.name)
    try:
        artifacts = present_deployment_artifacts(args.name, profile)
    except DeploymentRemovalError as exc:
        raise SystemExit(str(exc)) from exc

    _print_remove_consequences(
        args.name, artifacts, purge=args.purge, profile=profile
    )

    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                f"Removing {args.name} is permanent. Re-run with --yes."
            )
        typed = input(f"Type {args.name} to remove this deployment: ").strip()
        if typed != args.name:
            print("The name did not match; nothing was changed.")
            return 1

    try:
        service = unregister_deployment_service(args.name)
        result = remove_deployment_artifacts(args.name, profile, purge=args.purge)
    except DeploymentRemovalError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        _remove_outcome(
            result.name,
            planned=len(artifacts),
            removed=result.artifact_count,
            service=service,
        )
    )
    if result.archive is not None:
        # The archive is the only thing this command keeps. Say where it is and
        # how large, so kept state does not become invisible state.
        print(f"  Archive: {result.archive} ({_directory_size(result.archive)})")
        print("  Delete it yourself, or re-run with --purge to keep nothing.")
    return 0


def _format_bytes(total: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024 or unit == "GB":
            return f"{total:.0f} {unit}" if unit == "B" else f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} GB"


def _remove_outcome(
    name: str,
    *,
    planned: int,
    removed: int,
    service: str,
) -> str:
    """Report what removal observed after stopping the service."""

    text = f"Removed {name}: {removed} artifact(s). {service.rstrip('.')}."
    if removed < planned:
        missing = planned - removed
        text += (
            f" After service shutdown, {missing} previously confirmed "
            "artifact(s) were no longer present."
        )
    elif removed > planned:
        appeared = removed - planned
        text += (
            f" {appeared} additional artifact(s) appeared before removal "
            "and were removed too."
        )
    return text


def _directory_size(path: Path) -> str:
    total = 0
    try:
        for item in Path(path).rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        return "size unavailable"
    return _format_bytes(total)


def _deployment_list_command(args) -> int:
    from zippergen.rendering import TerminalRenderer

    rows = _deployment_inventory()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, default=str))
        return 0
    renderer = TerminalRenderer()
    renderer.framed_section("Host deployments")
    if not rows:
        renderer.empty("Deployments", "No deployments on this computer.")
        return 0
    renderer.columns(
        "Deployments",
        ("State", "Deployment", "Service", "Project", "Ownership"),
        [
            (
                renderer.status_mark("warning" if row["orphaned"] else "success"),
                row["name"],
                row["service"],
                row["project"],
                row["ownership"],
            )
            for row in rows
        ],
    )
    return 0


def _deployment_prune_command(args) -> int:
    """Clear what nobody owns: orphaned deployments and stale trash.

    Both are host-wide leftovers. Archives exist so a mistaken removal can be
    undone, so only ones older than the retention window are deleted.
    """

    from zippergen.deployments import list_trash_entries, prune_trash

    orphaned = [row for row in _deployment_inventory() if row["orphaned"]]
    names = [str(row["name"]) for row in orphaned]
    entries = list_trash_entries()
    stale = [entry for entry in entries if entry.age_days >= args.keep_days]
    unclaimed = _prune_shared_connector_inboxes(
        keep_days=args.keep_days, preview=True
    )

    if names:
        print("Orphaned deployments: " + ", ".join(names))
    else:
        print("No orphaned deployments.")
    if entries:
        print(
            f"Trash: {len(entries)} archive(s), {_format_bytes(sum(entry.bytes for entry in entries))}"
            f" total; {len(stale)} older than {args.keep_days:g} day(s)."
        )
    else:
        print("Trash: empty.")
    if unclaimed:
        print(
            f"Connector inboxes: {unclaimed} unclaimed update(s) older than "
            f"{args.keep_days:g} day(s)."
        )
    if not names and not stale and not unclaimed:
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit("Re-run with --yes to remove them.")
        answer = input(
            "Remove the orphaned deployments and delete the stale archives? "
            "[y/N]: "
        ).strip().casefold()
        if answer not in {"y", "yes"}:
            print("Nothing was changed.")
            return 1

    for row in orphaned:
        name = str(row["name"])
        if row.get("profile_loadable", True):
            _remove_command(argparse.Namespace(name=name, purge=False, yes=True))
            continue
        from zippergen.deployments import (
            DeploymentRemovalError,
            present_deployment_artifacts,
            remove_deployment_artifacts,
            unregister_deployment_service,
        )

        fallback_profile: dict[str, object] = {"name": name}
        try:
            artifacts = present_deployment_artifacts(name, fallback_profile)
            _print_remove_consequences(
                name,
                artifacts,
                purge=False,
                profile=fallback_profile,
            )
            service = unregister_deployment_service(name)
            result = remove_deployment_artifacts(
                name,
                fallback_profile,
                purge=False,
            )
        except DeploymentRemovalError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            _remove_outcome(
                name,
                planned=len(artifacts),
                removed=result.artifact_count,
                service=service,
            )
        )
        if result.archive is not None:
            print(f"  Archive: {result.archive} ({_directory_size(result.archive)})")
    dropped = _prune_shared_connector_inboxes(keep_days=args.keep_days)
    if dropped:
        print(
            f"Dropped {dropped} unclaimed connector update(s) older than "
            f"{args.keep_days:g} day(s)."
        )
    if stale:
        outcome = prune_trash(keep_days=args.keep_days)
        print(
            f"Deleted {len(outcome.removed)} archive(s), reclaiming "
            f"{_format_bytes(outcome.removed_bytes)}."
        )
        if outcome.kept:
            print(
                f"  Kept {len(outcome.kept)} archive(s) newer than "
                f"{args.keep_days:g} day(s), {_format_bytes(outcome.kept_bytes)}."
            )
    return 0


def _compact_command(args) -> int:
    """Prune optional history and rotate logs while the service is stopped.

    Durable state is the current state of the computation, so there is nothing
    to compact in it. History is not read by recovery, but pruning it still
    changes the store; the stopped-service precondition also makes log rotation
    lossless and prevents the command from failing after deleting history.
    """

    with _hold_deployment_mutation(
        args.name, owner="a deployment compaction"
    ):
        try:
            enforce_deploy_requirement("compact", args.name)
        except ServiceIsLiveError as exc:
            raise SystemExit(str(exc)) from exc
        return _compact_command_locked(args)


def _compact_command_locked(args) -> int:
    from zippergen.deployment_platform import deployment_service_status
    from zippergen.deployments import DeploymentRemovalError, compact_deployment_logs
    from zippergen.storage_maintenance import (
        prune_store_history,
        set_store_history_keep,
    )

    # The dispatcher has already refused this if a service may hold the store.
    profile = _load_deployment_profile(args.name)
    store = profile.get("store")
    try:
        logs = compact_deployment_logs(
            args.name,
            profile,
            keep_archives=args.keep_archives,
        )
    except DeploymentRemovalError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Logs: rotated, keeping {args.keep_archives} archive(s).")
    if logs.removed_archives:
        print(
            f"  removed archives: {logs.removed_archives} "
            f"({logs.removed_archive_bytes} bytes)"
        )
    if store:
        if args.set_history_keep is not None:
            if args.set_history_keep < 0:
                raise SystemExit("--set-history-keep must be zero or greater.")
            profile["history_keep"] = int(args.set_history_keep)
            _write_deployment_artifacts(profile)
            outcome = set_store_history_keep(str(store), args.set_history_keep)
        else:
            # "Tidy this up" means the store's own budget. It used to mean
            # "delete all of it", which is a surprising thing for a bare command
            # to do to the only record of what ran. Changing the budget is a
            # separate request, made with --set-history-keep.
            outcome = prune_store_history(str(store))
        print(f"Store {store}")
        print(f"  history budget: {_history_budget_text(str(store))}")
        print(f"  removed history rows: {outcome.removed_rows}")
        print(
            "  reclaimed bytes: "
            f"{max(0, outcome.before_bytes - outcome.after_bytes)}"
        )
    return 0


def _history_budget_text(store: str) -> str:
    """Describe a store's history budget the way a person would say it."""

    from zippergen.storage_maintenance import inspect_store_storage

    report = inspect_store_storage(store)
    if report.history_keep == 0:
        return "0 rows (history is off)"
    return f"{report.history_rows:,} of {report.history_keep:,} rows kept"


def _print_reset_consequences(name: str, profile: Mapping[str, object]) -> None:
    """Say what a reset throws away, in the units a person cares about.

    Reset moves the whole SQLite file family aside and starts an empty store,
    so everything in it goes, not only the obvious parts. Every category is
    listed even at zero: an omitted line cannot be told apart from a category
    nobody checked, and this is a prompt people answer 'y' to.
    """

    from zippergen.storage_maintenance import inspect_store_storage

    print(f"Reset deployment {name}")
    store = profile.get("store")
    if not store:
        print("  This deployment has no durable store configured.")
        return
    report = inspect_store_storage(str(store))
    rows = [
        ("participant positions", str(report.roles)),
        ("messages in flight", str(report.outstanding_messages)),
        (
            "human tasks",
            f"{report.pending_tasks} waiting, {report.completed_tasks} answered",
        ),
        ("workflow results", str(report.workflow_results)),
        ("connector progress", str(report.connector_entries)),
    ]
    print("  Archived, then cleared from the live deployment:")
    width = max(len(label) for label, _value in rows)
    for label, value in rows:
        print(f"    {label.ljust(width)}  {value}")
    if report.connector_entries:
        # A mail cursor or chat offset lives here. Losing it is not abstract:
        # the workflow can re-read mail it already handled.
        print(
            "    (connector progress is where a mailbox cursor lives, so the "
            "workflow may re-read messages it already handled)"
        )
    print(
        "  Kept: the deployment itself, its configuration, secrets, bundle "
        "and logs."
    )
    print(
        "  The service is left stopped. The workflow runs from the beginning "
        "the next time you start it."
    )


def _print_remove_consequences(
    name: str,
    artifacts,
    *,
    purge: bool,
    profile: Mapping[str, object] | None = None,
) -> None:
    """Name what removal destroys, not only what it keeps.

    Credentials are the expensive, irreversible loss: they are deleted rather
    than archived, so removing means re-entering a token and redoing any
    authorization. Listing only what survives hides exactly that.
    """

    def describe(item) -> str:
        raw_path = getattr(item, "path", None)
        return (
            f"{item.label} ({Path(raw_path).resolve()})"
            if raw_path is not None
            else item.label
        )

    kept = [describe(item) for item in artifacts if item.retain and not purge]
    lost = [
        describe(item) for item in artifacts if not (item.retain and not purge)
    ]
    print(f"Deployment {name}: {len(artifacts)} artifact(s) currently present.")
    if lost:
        print(f"  Deleted for good: {', '.join(lost)}.")
    if any("secret" in item.label.casefold() for item in artifacts):
        print(
            "    Credentials go with them. You will re-enter tokens and redo "
            "any provider authorization."
        )
    if purge:
        print("  --purge: nothing is kept, including the durable store.")
    elif kept:
        print(
            f"  Archived under {_zippergen_home() / 'trash' / 'deployments'}: "
            f"{', '.join(kept)}."
        )
    owned = tuple(
        (
            Path(item.path).expanduser().resolve(),
            getattr(item, "kind", "file") == "directory",
        )
        for item in artifacts
        if getattr(item, "path", None) is not None
    )
    preserved = []
    for field in ("store", "log", "secrets_file"):
        raw_path = (profile or {}).get(field)
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser().resolve()
        is_owned = any(
            path == root
            or (is_directory and path.is_relative_to(root))
            for root, is_directory in owned
        )
        if path.exists() and not is_owned:
            preserved.append(f"{field} ({path})")
    if preserved:
        print(
            "  Preserved external reference(s), because the profile does not "
            f"confer ownership: {', '.join(preserved)}."
        )
    print("  The service is unregistered, so nothing runs it again.")


def _reset_deployment_command(args) -> int:
    """Replace deployment state with an empty store, keeping an archive."""

    from zippergen.deployment_platform import deployment_service_status
    from zippergen.deployments import (
        DeploymentRemovalError,
        reset_deployment_store,
    )

    profile = _load_deployment_profile(args.name)
    _print_reset_consequences(args.name, profile)
    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Resetting deployment state requires confirmation. "
                "Re-run with --yes."
            )
        answer = input("Discard this state? [y/N]: ").strip().casefold()
        if answer not in {"y", "yes"}:
            print("Nothing was changed.")
            return 1

    service = deployment_service_status(args.name)
    was_running = service_is_running(service)
    needs_stop = service_is_running(service)
    lifecycle = argparse.Namespace(
        name=args.name,
        dry_run=False,
        enable=False,
        skip_readiness=False,
    )
    if needs_stop:
        _deployment_lifecycle_command(lifecycle, "stop")
    with _hold_deployment_mutation(args.name, owner="a deployment reset"):
        try:
            result = reset_deployment_store(args.name, profile)
        except DeploymentRemovalError as exc:
            raise SystemExit(str(exc)) from exc
        _initialize_deployment_store(profile)
    print(f"Reset deployment state: {result.store}")
    if result.archive is not None:
        print(
            f"Archived {result.archived_files} SQLite file(s): "
            f"{result.archive}"
        )
    else:
        print("There was no previous durable state to archive.")
    # Reset never starts the service. It is a state operation, and the service
    # has to be stopped for it, so leaving it stopped is the honest outcome.
    # It also gives you a beat before a from-scratch run: the connector cursor
    # is gone too, so starting may re-read a mailbox you already handled.
    if was_running:
        print("The service was running and is now stopped.")
    print("Start it again when you are ready: zippergen deploy start")
    return 0


def _project_connector_runtime(
    args,
    deployed_workflow: str | None = None,
    deployed_project: str | None = None,
) -> tuple[dict, dict[str, str]]:
    """Build deployment connector routing from the project's configuration.

    `deploy` reads the project directly, so connector configurations and
    assignments reach the deployment without an intermediate UI layer.

    `deployed_workflow` and `deployed_project` are what an existing deployment
    already records about itself. Reconfiguring by name must wire that
    workflow, from that project — not from whatever directory the shell
    happens to be standing in, which may be an unrelated project or none.
    """

    from zippergen.connector_wiring import connector_runtime
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None) or deployed_project)
    # Follow the workflow actually being deployed. Falling back to the
    # project's configured entry silently skipped connectors whenever a
    # workflow was named explicitly.
    target = getattr(args, "target", None)
    entry = target if target and (":" in target or _looks_like_path(target)) else None
    entry = entry or deployed_workflow
    try:
        canonical = workspace.resolve_workflow(entry)
    except WorkspaceError:
        return {}, {}
    workflow, module = load_workflow_spec(str(workspace.absolute_spec(canonical)))
    return connector_runtime(workspace, canonical, workflow, module)


def _provider_accept_google_command(args) -> int:
    """Save a Google authorization produced on a computer with a browser.

    `provider authorize CONNECTION --handoff` runs the browser flow and prints
    an encoded result instead of saving it. This is the other half: it stores
    the credential and the scopes Google actually granted, so a machine with no
    browser — a server — can be authorized from your laptop.
    """

    import getpass

    from zippergen.google_auth import (
        GoogleConnectorError,
        decode_google_authorization,
        google_authorization_summary,
    )
    from zippergen.workspace import Workspace

    args.name = _guided_required_value(
        args.name,
        label="Google connection",
        command="zg provider accept CONNECTION",
        choices=_project_choices(
            "provider-connections-google", getattr(args, "project", None)
        ),
    )
    try:
        encoded = getpass.getpass(
            "Paste the zg-google-v1... result (input hidden): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("Cancelled; nothing was saved.") from None
    if not encoded:
        raise SystemExit("An encoded authorization result is required.")

    try:
        result = decode_google_authorization(encoded)
    except (GoogleConnectorError, ValueError) as exc:
        raise SystemExit(f"That is not a valid authorization result: {exc}") from exc

    workspace = Workspace(getattr(args, "project", None))
    connection = str(args.name)
    if not _save_google_authorization(workspace, connection, result):
        raise SystemExit(
            f"Provider connection {connection!r} does not exist or is not Google."
        )
    return 0


def _save_google_authorization(workspace, connection: str, result) -> bool:
    """Store one finished authorization, or say the connection is not Google.

    Both halves of the handoff end here, and so does authorizing on the
    machine that will use it, so a credential is written exactly one way.
    """

    from zippergen.google_auth import google_authorization_summary

    granted, client, expiry = google_authorization_summary(result)
    profile = workspace.provider_connections().get(connection)
    if profile is None or profile.get("kind") != "google":
        return False
    workspace.save_provider_secret(
        connection, "authorized_user_json", result.authorized_user_json
    )
    workspace.save_provider_connection(
        connection,
        {
            **profile,
            "kind": "google",
            "granted_scopes": json.dumps(list(result.granted_scopes)),
            "client_id": client,
            "credential_expiry": expiry,
        },
    )
    print(
        f"Google authorization saved in {workspace.secrets_path} "
        "(owner-only file)."
    )
    print(f"  Granted: {granted}")
    print(f"  Expiry:  {expiry}")
    return True


def _connector_assign_command(args) -> int:
    """Fill one of the workflow's connector slots with a saved configuration.

    A workflow has two kinds of slot, and this fills either. A declared
    service requirement names something the workflow needs, such as a mailbox
    to read. A `@human` action asks a person something, and the participant
    naming it says where that question is delivered; without it the question
    appears in whichever terminal is running the workflow, which is right for
    development and wrong for a deployment nobody is watching.

    Which kind a name refers to is something the workflow already knows, so
    there is one verb rather than two, and no flag to say which you meant.
    """

    from zippergen.project_configuration import (
        CONNECTOR_REQUIREMENT,
        assign_connector,
        connector_target_problem,
    )
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(getattr(args, "project", None))
    try:
        target = _guided_required_value(
            args.target,
            label="Connector target",
            command="zg connector assign TARGET CONFIGURATION",
            choices=_project_choices("connector-targets", args.project),
            check=lambda entered: connector_target_problem(workspace, entered),
        )
        configuration = _guided_required_value(
            args.configuration,
            label="Connector configuration",
            command="zg connector assign TARGET CONFIGURATION",
            choices=_project_choices(
                "connector-configurations", args.project
            ),
        )
        kind = assign_connector(workspace, target, configuration)
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    if kind == CONNECTOR_REQUIREMENT:
        print(f"{target} will use {configuration}.")
    else:
        print(f"{target} will be asked through {configuration}.")
    return 0


def _connector_configure_command(args) -> int:
    """Save one named connector configuration for this project.

    Portable fields — which chat, spreadsheet, or mailbox query — are separate
    from the provider identity and credential shared by configurations.
    """

    from zippergen.provider_connections import connector_kinds_for_provider
    from zippergen.workspace import Workspace, WorkspaceError

    workspace = Workspace(args.project)
    connections = _required_provider_connections(
        "provider-connections-connector",
        args.project,
        purpose="connector",
        example="zg provider configure approval-bot telegram",
    )
    # The connection is asked first: it decides the kind, and whether the rest
    # of this dialogue is about a chat, a mailbox or a spreadsheet. The name is
    # the only invented value, so it comes once there is something to name.
    configurations = workspace.connector_configurations()
    existing = configurations.get(str(args.name or "")) or {}
    connection = _guided_required_value(
        args.connection,
        label="Provider connection",
        command="zg connector configure NAME CONNECTION [KIND]",
        choices=connections,
        default=str(existing.get("connection") or "") or None,
    )
    provider_profile = workspace.provider_connections().get(connection) or {}
    provider = str(provider_profile.get("kind") or "")
    supported = connector_kinds_for_provider(provider)
    if not supported:
        raise WorkspaceError(
            f"Provider connection {connection!r} ({provider or 'unknown'}) "
            "cannot be used by a connector."
        )
    kind = (
        supported[0]
        if len(supported) == 1 and args.kind is None
        else _guided_required_value(
            args.kind,
            label="Connector kind",
            command="zg connector configure NAME CONNECTION KIND",
            choices=supported,
            default=str(existing.get("kind") or "") or None,
        )
    )
    # Last, because a connector is named after its purpose, and its purpose is
    # only visible once the connection and the kind are on the screen.
    name = _guided_required_value(
        args.name,
        label="Connector configuration name",
        command="zg connector configure NAME CONNECTION [KIND]",
        check=_name_check("connector configuration"),
    )
    existing = configurations.get(name) or existing

    spec = connector_kind_spec(kind)
    if spec is None:  # pragma: no cover - argparse and provider restrict this
        raise SystemExit(f"Unsupported connector kind {kind!r}.")

    accepted = {setting.name for setting in spec.settings}
    unused = [
        setting
        for setting in CONNECTOR_SETTING_SPECS
        if setting.name not in accepted and getattr(args, setting.name, None)
    ]
    if unused:
        options = ", ".join(
            "--" + setting.name.replace("_", "-") for setting in unused
        )
        raise SystemExit(f"{spec.name} configuration does not use {options}.")

    required_usage = " ".join(
        f"--{setting.name.replace('_', '-')} {setting.metavar}"
        for setting in spec.settings
        if setting.required and setting.prompt
    )
    command = (
        f"zg connector configure NAME CONNECTION {spec.name}"
        + (f" {required_usage}" if required_usage else "")
    )
    values = {"connection": connection, "kind": spec.name}
    for setting in spec.settings:
        supplied = getattr(args, setting.name, None)
        fallback = str(existing.get(setting.name) or "")
        if not fallback and setting.default_from:
            fallback = values.get(setting.default_from, "")
        if not fallback:
            fallback = setting.default or ""
        if setting.prompt:
            value = _guided_required_value(
                supplied,
                label=setting.label,
                command=command,
                default=fallback or None,
            )
        else:
            value = str(supplied or fallback).strip()
            if setting.required and not value:
                raise SystemExit(f"{setting.label} is required. Use: {command}")
        if value:
            values[setting.name] = value
    described = spec.describe(values)

    try:
        workspace.save_connector_configuration(name, values)
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Saved connector configuration {name} ({described}).")

    from zippergen.provider_connections import provider_credential_field

    credential_field = provider_credential_field(provider)
    if credential_field and not workspace.provider_secret(connection, credential_field):
        print()
        print(f"Provider connection {connection!r} has no private credential here.")
        if provider == "google":
            print(
                f"Authorize it with: zg provider authorize {connection}"
            )
        else:
            print(f"Add it with: zg provider set-credential {connection}")

    return 0


def _init_command(args) -> int:
    """Create a project and stop. No questionnaire, no configuration.

    Everything else — models, connectors, the workflow itself — is either an
    edit to `zippergen.toml` or something a coding agent writes once you say
    what you want.
    """

    from zippergen.skill import agents_md, claude_md
    from zippergen.workspace import Workspace

    root = Path(args.directory).expanduser().resolve() if args.directory else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(root)
    name = args.name or root.name

    created = workspace.manifest_path.exists()
    manifest = workspace.initialize_project(name=name)
    rows = [("zippergen.toml", "exists" if created else "created")]

    specification = root / str(manifest["specification_file"])
    if specification.exists():
        rows.append((specification.name, "exists"))
    else:
        specification.write_text(
            SPECIFICATION_TEMPLATE.format(name=manifest["name"]),
            encoding="utf-8",
        )
        rows.append((specification.name, "created"))

    agents = root / "AGENTS.md"
    if agents.exists():
        rows.append(("AGENTS.md", "exists; left alone"))
    else:
        agents.write_text(agents_md(str(manifest["name"])), encoding="utf-8")
        rows.append(("AGENTS.md", "created"))

    claude = root / "CLAUDE.md"
    if claude.exists():
        rows.append(("CLAUDE.md", "exists; left alone"))
    else:
        claude.write_text(claude_md(), encoding="utf-8")
        rows.append(("CLAUDE.md", "created"))

    print(f"ZipperGen project: {manifest['name']}")
    for filename, state in rows:
        print(f"  {filename:<18} {state}")
    if agents.exists() and "zippergen skill" not in agents.read_text(encoding="utf-8"):
        print()
        print("AGENTS.md was already here and was not changed. Add this so a")
        print("coding agent finds the ZipperGen instructions:")
        print()
        print("    Before editing workflow code, run `zippergen skill`")
        print("    and follow it completely.")
    if claude.exists() and "@AGENTS.md" not in claude.read_text(encoding="utf-8"):
        print()
        print("CLAUDE.md was already here and was not changed. Add this so")
        print("Claude Code reads the shared ZipperGen instructions:")
        print()
        print("    @AGENTS.md")
    return 0


def _skill_command(args) -> int:
    from zippergen.skill import SkillNotFound, agents_md, load_skill

    if args.agents_md:
        project = args.project or Path.cwd().name
        print(agents_md(project), end="")
        return 0
    try:
        skill = load_skill()
    except SkillNotFound as exc:
        raise SystemExit(str(exc)) from exc
    print(skill.render(include_references=not args.no_references), end="")
    return 0


def _semantic_input(spec: str) -> dict[str, object]:
    candidate = Path(spec).expanduser()
    if candidate.is_file() and candidate.suffix.lower() == ".json":
        try:
            return read_semantic_snapshot(json.loads(candidate.read_text()))
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Invalid semantic snapshot {candidate}: {exc}") from exc
    workflow, module = load_workflow_spec(spec)
    return workflow_semantics(workflow, module)


def _snapshot_command(args) -> int:
    """Write one workflow's semantic baseline."""

    spec = args.workflow or _resolved_workflow_spec(args)
    workflow, module = load_workflow_spec(spec)
    payload = json.dumps(
        semantic_snapshot(workflow, module), indent=2, default=str
    )
    if args.path == "-":
        print(payload)
        return 0
    output_path = Path(args.path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n")
    print(f"Wrote semantic baseline to {output_path}")
    return 0


def _diff_command(args) -> int:
    if not args.before:
        raise SystemExit(
            "Give a baseline to compare against, or create one first with "
            "'zippergen snapshot PATH'."
        )
    # With one argument, compare a saved baseline against the project workflow.
    after = args.after or _resolved_workflow_spec(args)
    result = semantic_diff_models(
        _semantic_input(args.before),
        _semantic_input(after),
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, default=str))
    else:
        print(render_semantic_diff(result))
    return 0


def _parse_guided_value(raw: str, default: object) -> object:
    text = raw.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class _FormatValues(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _resolve_field_default(
    field: DeploymentField,
    current: object,
    values: dict[str, object],
) -> object:
    if current != field.default or not isinstance(current, str) or "{" not in current:
        return current
    try:
        return current.format_map(_FormatValues(values))
    except (KeyError, ValueError):
        return current


def _display_default(value: object, *, secret: bool) -> str:
    if value is None or value == "":
        return ""
    if secret:
        return " [already set]"
    if isinstance(value, (dict, list, tuple, bool, int, float)):
        return f" [{json.dumps(value, default=str)}]"
    return f" [{value}]"


# Where a deployment field's value came from. A configured deployment is the
# result of four sources with a fixed precedence, and reporting only the value
# leaves an operator unable to tell a deliberate setting from a default nobody
# chose. `zg deploy --yes` in particular answers every prompt silently, so the
# source is the only thing that makes it reviewable.
FIELD_SOURCE_PROJECT = "zippergen.toml"
FIELD_SOURCE_ENVIRONMENT = "environment"
FIELD_SOURCE_DEFAULT = "declared default"
FIELD_SOURCE_OVERRIDE = "--set"
FIELD_SOURCE_ENTERED = "entered now"
FIELD_SOURCE_UNSET = "unset"


def _collect_deployment_fields(
    spec: DeploymentSpec,
    profile: dict[str, object],
    *,
    overrides: dict[str, object],
    interactive: bool,
    sources: dict[str, str] | None = None,
    workspace: "Workspace | None" = None,
) -> tuple[dict[str, object], dict[str, str]]:
    declared = {field.name for field in spec.fields}
    unknown = sorted(set(overrides) - declared)
    if unknown:
        available = ", ".join(sorted(declared)) or "none"
        raise SystemExit(
            "Unknown deployment field"
            f"{'s' if len(unknown) != 1 else ''}: {', '.join(unknown)}. "
            f"Available fields: {available}."
        )
    existing_secrets = _load_deployment_secrets(profile)
    # One rule: every non-secret answer a person gives is kept in the visible
    # project file, and the deployment profile is derived from it. Reading from
    # anywhere else is what made "where is the value I typed?" have two answers.
    project_answers = (
        workspace.configuration_values() if workspace is not None else {}
    )
    values: dict[str, object] = {}
    secrets: dict[str, str] = dict(existing_secrets)

    for field in spec.fields:
        if field.secret and field.name in overrides:
            raise SystemExit(
                f"Deployment field {field.name!r} is secret and cannot be "
                "passed with --set. Run interactively or provide "
                f"{field.target_name} in the deployment environment."
            )
        current = None if field.secret else project_answers.get(field.name)
        origin = FIELD_SOURCE_PROJECT
        if current is None and field.target == "env":
            current = os.environ.get(field.target_name)
            origin = FIELD_SOURCE_ENVIRONMENT
        if current is None:
            current = field.default
            origin = FIELD_SOURCE_DEFAULT
        if current is None:
            origin = FIELD_SOURCE_UNSET
        if field.name in overrides:
            current = overrides[field.name]
            origin = FIELD_SOURCE_OVERRIDE
        values[field.name] = current
        if sources is not None:
            sources[field.name] = origin
    values["__llm_specs__"] = selected_llm_specs(
        profile.get("llm"),
        profile.get("llms"),
    )
    values["__llm_field_names__"] = ()

    for field in spec.fields:
        if not _field_enabled(field, values):
            continue
        current = _resolve_field_default(field, values.get(field.name), values)
        values[field.name] = current
        if interactive and field.name not in overrides:
            choices = f" ({'/'.join(field.choices)})" if field.choices else ""
            label = field.prompt + choices + _display_default(current, secret=field.secret) + ": "
            if field.secret:
                entered = getpass.getpass(label)
            else:
                entered = input(label)
            values[field.name] = _parse_guided_value(entered, current)
            if sources is not None and values[field.name] != current:
                sources[field.name] = FIELD_SOURCE_ENTERED
        value = values.get(field.name)
        if field.required and (value is None or str(value).strip() == ""):
            if field.secret:
                raise SystemExit(
                    f"Deployment field {field.name!r} is required. Run "
                    f"interactively or provide {field.target_name} in the "
                    "deployment environment."
                )
            raise SystemExit(
                f"Deployment field {field.name!r} is required. "
                f"Use --set {field.name}=VALUE or run interactively."
            )
        if value is not None and field.choices and str(value) not in field.choices:
            raise SystemExit(
                f"Deployment field {field.name!r} must be one of "
                f"{', '.join(field.choices)}; got {value!r}."
            )
        if value is not None and value != "" and field.path_exists:
            path = Path(str(value)).expanduser()
            if not path.exists():
                raise SystemExit(f"Deployment field {field.name!r} points to a missing path: {path}")
            # Deployment services run from immutable source bundles. Preserve
            # the external resource the user selected rather than re-resolving
            # a relative path inside the bundle at service start.
            values[field.name] = str(path.resolve())

    options: dict[str, object] = _profile_mapping(profile, "options")
    inputs: dict[str, object] = _profile_mapping(profile, "inputs")
    environment = {
        key: str(value)
        for key, value in _profile_mapping(profile, "environment").items()
    }
    for field in spec.fields:
        if not _field_enabled(field, values):
            continue
        value = values.get(field.name)
        if value is None:
            continue
        if field.target == "input":
            inputs[field.target_name] = value
        elif field.target == "option":
            options[field.target_name] = value
        elif field.secret:
            if str(value):
                secrets[field.target_name] = str(value)
        else:
            environment[field.target_name] = str(value)
    profile["options"] = options
    profile["inputs"] = inputs
    profile["environment"] = environment
    _record_answers_in_project(spec, values, workspace)
    return values, secrets


def _record_answers_in_project(
    spec: DeploymentSpec,
    values: Mapping[str, object],
    workspace: "Workspace | None",
) -> None:
    """Persist every non-secret answer where a person can find and edit it.

    This is what makes the deployment profile derived rather than authored: an
    answer typed once is kept in the visible project file, so it survives
    removing the deployment, moving to another machine, and is reviewable in
    version control like every other project choice.
    """

    if workspace is None:
        return
    original_answers = workspace.configuration_values()
    answers = dict(original_answers)
    for field in spec.fields:
        if (
            not field.secret
            and values.get(field.name) is not None
            and _field_enabled(field, dict(values))
        ):
            answers[field.name] = values[field.name]
    if answers != original_answers:
        workspace.write_configuration_values(answers)


def _field_display_value(field: DeploymentField, value: object) -> str:
    """Render one value for a person, without printing a secret."""

    if field.secret:
        text = str(value or "")
        return f"({len(text)} characters, stored privately)" if text else "(not set)"
    if value is None or str(value) == "":
        return "(not set)"
    return str(value)


def _print_deployment_configuration(
    spec: DeploymentSpec,
    values: Mapping[str, object],
    sources: Mapping[str, str],
    *,
    heading: str,
    stored_in: str | None = None,
) -> None:
    """Say what this deployment is configured with, and where each value came from."""

    from zippergen.rendering import TerminalRenderer

    fields = [field for field in spec.fields if _field_enabled(field, dict(values))]
    if not fields:
        return
    rows: list[tuple[object, ...]] = [
        (
            field.name,
            _field_display_value(field, values.get(field.name)),
            sources.get(field.name, ""),
        )
        for field in fields
    ]
    TerminalRenderer().columns(heading, ("Field", "Value", "From"), rows)
    if stored_in:
        print(f"Stored in {stored_in}")
        print(f"Change one with: zippergen deploy --set FIELD=VALUE")


def _deployment_context_from_profile(
    profile: dict[str, object],
    *,
    source: bool = False,
) -> tuple[dict[str, object], Workflow, ModuleType, DeploymentSpec]:
    cwd_key = "source_cwd" if source and profile.get("source_cwd") else "cwd"
    workflow_key = "source_workflow" if source and profile.get("source_workflow") else "workflow"
    cwd = Path(str(profile.get(cwd_key) or ".")).expanduser()
    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        with _profile_environment(profile):
            workflow, module = load_workflow_spec(str(profile[workflow_key]))
    finally:
        os.chdir(old_cwd)
    return profile, workflow, module, deployment_spec_from_module(module)


def _workflow_source_identity(spec: str, cwd: Path) -> str:
    """Identify a workflow by its project-relative path when possible."""

    module_ref, separator, workflow_name = spec.partition(":")
    if not _looks_like_path(module_ref):
        return spec
    path = Path(module_ref).expanduser()
    if not path.is_absolute():
        path = cwd / path
    resolved = path.resolve()
    try:
        source = resolved.relative_to(cwd.resolve()).as_posix()
    except ValueError:
        source = str(resolved)
    return source + (f":{workflow_name}" if separator else "")


def _apply_deploy_arguments(
    profile: dict[str, object],
    args,
    spec: DeploymentSpec,
    workflow: Workflow,
) -> tuple[dict[str, object], dict[str, str]]:
    if args.timeout is not None:
        profile["timeout"] = args.timeout
    runtime_secrets: dict[str, str] = {}
    try:
        snapshot, connector_secrets = _project_connector_runtime(
            args,
            deployed_workflow=str(
                profile.get("source_workflow") or profile.get("workflow") or ""
            )
            or None,
            deployed_project=str(profile.get("source_cwd") or "") or None,
        )
    except Exception as exc:
        from zippergen.connector_wiring import ConnectorWiringError

        if isinstance(exc, ConnectorWiringError):
            raise SystemExit(str(exc)) from exc
        raise
    profile["connectors"] = snapshot
    runtime_secrets.update(connector_secrets)

    from zippergen.workspace import Workspace

    source_workspace = Workspace(str(profile.get("source_cwd") or Path.cwd()))
    model_environment = source_workspace.development_provider_environment(
        selected_llm_specs(profile.get("llm"), profile.get("llms"))
    )
    public_environment = _profile_mapping(profile, "environment")
    for name, value in model_environment.items():
        if name.endswith("_API_KEY"):
            runtime_secrets[name] = value
        else:
            public_environment[name] = value
    profile["environment"] = public_environment

    overrides = _parse_inputs(args.set)
    interactive = not args.yes and sys.stdin.isatty()
    sources: dict[str, str] = {}
    values, secrets = _collect_deployment_fields(
        spec,
        profile,
        overrides=overrides,
        interactive=interactive,
        sources=sources,
        workspace=source_workspace,
    )
    # Every deploy says what it is configured with. A non-interactive `--yes`
    # answers each prompt from an existing deployment, the environment, or a
    # declared default without showing any of them, which is precisely when an
    # operator cannot otherwise tell which.
    _print_deployment_configuration(
        spec, values, sources, heading="Configuration"
    )
    secrets.update(runtime_secrets)
    return values, secrets


# `_finalize_guided_deployment` and `_run_deployment_setup` stay here on
# purpose. They sequence typed domain calls and report progress to the
# terminal, which is exactly what this module is for. The steps they call --
# preparing the home, writing artifacts, initialising the store -- live in
# `deployment_publication.py`.
def _finalize_guided_deployment(
    profile: dict[str, object],
    spec: DeploymentSpec,
    workflow: Workflow,
    values: dict[str, object],
    secrets: dict[str, str],
    args,
) -> int:
    name = str(profile["name"])
    active_profile_existed = _deployment_profile_path(name).exists()
    previous_bundle = str(profile.get("bundle") or "")
    candidate_bundle: Path | None = None
    environment_update = None
    candidate_secrets: Path | None = None
    canonical_secrets = _deployment_secrets_path(name)
    previous_secrets_raw = profile.get("secrets_file")
    previous_secrets = (
        Path(str(previous_secrets_raw)).expanduser()
        if previous_secrets_raw
        else canonical_secrets if canonical_secrets.exists() else None
    )
    history_previous: int | None = None
    store_created = False
    published = False

    secret_fields = [field for field in spec.fields if field.secret]
    if canonical_secrets.is_symlink():
        raise SystemExit(
            f"Refusing a symlinked deployment secrets file: {canonical_secrets}"
        )
    if secret_fields or secrets:
        secrets_dir = _deployment_secrets_dir(name)
        ensure_private_directory(secrets_dir)
        fd, raw_candidate = tempfile.mkstemp(
            prefix="generation-",
            suffix=".json",
            dir=secrets_dir,
        )
        os.close(fd)
        candidate_secrets = Path(raw_candidate)
        _write_deployment_secrets(candidate_secrets, secrets)
        profile["secrets_file"] = str(candidate_secrets)
        profile["secret_names"] = sorted(secrets)
    else:
        profile.pop("secrets_file", None)
        profile["secret_names"] = []
    profile["deployment_spec"] = spec.as_dict()
    profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log_path = Path(
        str(profile.get("log") or _default_deployment_log_path(name))
    ).expanduser()
    try:
        profile["log_generation_offset"] = (
            log_path.stat().st_size if log_path.is_file() else 0
        )
    except OSError:
        profile["log_generation_offset"] = 0
    profile["deployment_generation_at"] = profile["updated_at"]

    history_keep = getattr(args, "history_keep", None)
    if history_keep is not None:
        if history_keep < 0:
            raise SystemExit("--history-keep must be zero or greater.")
        profile["history_keep"] = int(history_keep)

    try:
        if not args.no_bundle:
            _bundle_deployment(profile, spec, workflow)
            raw_bundle = str(profile.get("bundle") or "")
            if raw_bundle and raw_bundle != previous_bundle:
                candidate_bundle = Path(raw_bundle)

        _prepare_managed_home(profile)
        store_created = _initialize_deployment_store(profile)
        environment_update = _prepare_deployment_environment(
            profile,
            spec,
            skip_install=args.no_install,
            defer_cleanup=True,
        )
        _run_deployment_setup(profile, spec, values, skip_setup=args.no_setup)

        if not args.no_doctor:
            checks = _doctor_checks(
                name,
                include_systemd=False,
                before_start=True,
                profile_override=profile,
                check_artifacts=False,
            )
            if getattr(args, "concise", False):
                _print_doctor_summary(name, checks)
            else:
                _print_doctor(name, checks)
            if any(check.get("status") == "fail" for check in checks):
                print(
                    f"Deployment candidate {name} was not applied because the "
                    "checks above found problems. It was not started, and the "
                    "previous deployment was left unchanged."
                )
                return 1

        history_previous = _apply_existing_history_keep(
            profile,
            requested=history_keep,
            store_created=store_created,
        )
        _write_deployment_artifacts(profile)
        published = True
        if environment_update is not None:
            environment_update.commit()
        if previous_secrets is not None and previous_secrets != candidate_secrets:
            managed_secrets = _deployment_secrets_dir(name).resolve()
            canonical = canonical_secrets.resolve(strict=False)
            resolved_previous = previous_secrets.resolve(strict=False)
            if resolved_previous == canonical:
                previous_secrets.unlink(missing_ok=True)
            else:
                try:
                    resolved_previous.relative_to(managed_secrets)
                except ValueError:
                    pass
                else:
                    previous_secrets.unlink(missing_ok=True)
        if candidate_secrets is not None:
            for stale in candidate_secrets.parent.iterdir():
                if stale != candidate_secrets and (
                    stale.is_symlink() or not stale.is_dir()
                ):
                    stale.unlink(missing_ok=True)
    except BaseException:
        if history_previous is not None:
            _apply_existing_history_keep(
                profile,
                requested=history_previous,
                store_created=False,
            )
        raise
    finally:
        if not published:
            if environment_update is not None:
                environment_update.rollback()
            if candidate_secrets is not None:
                candidate_secrets.unlink(missing_ok=True)
            if candidate_bundle is not None:
                shutil.rmtree(candidate_bundle, ignore_errors=True)
            if store_created and not active_profile_existed:
                store = Path(str(profile["store"])).expanduser()
                for path in (
                    store,
                    Path(str(store) + "-wal"),
                    Path(str(store) + "-shm"),
                    Path(str(store) + "-journal"),
                ):
                    path.unlink(missing_ok=True)

    return 0


def _deploy_command(args) -> int:
    from zippergen.workspace import Workspace

    workspace = Workspace(getattr(args, "project", None))
    name = workspace.directory.name
    with _hold_deployment_mutation(name, owner="a deployment update"):
        try:
            enforce_deploy_requirement(None, name)
        except ServiceIsLiveError as exc:
            raise SystemExit(str(exc)) from exc
        result = _deploy_command_locked(args)
        if result != 0:
            return result
        name = str(getattr(args, "name", name))
        if not args.no_start:
            lifecycle_args = argparse.Namespace(
                name=name,
                enable=True,
                dry_run=False,
                skip_readiness=True,
            )
            lifecycle_result = _deployment_lifecycle_command(
                lifecycle_args, "start"
            )
            if lifecycle_result != 0:
                return lifecycle_result
    print(f"Deployment: {name}")
    if not getattr(args, "concise", False):
        print("Status: zippergen deploy status")
        print("Logs: zippergen deploy logs --follow")
        print("Stop: zippergen deploy stop")
    return 0


def _deploy_command_locked(args) -> int:
    # The workflow and the deployment both come from the project. The private
    # deployment identity is the checkout's stable private workspace name and
    # is never typed by the user.
    from zippergen.workspace import Workspace

    args.target = _resolved_workflow_spec(args)
    deployment_workspace = Workspace(getattr(args, "project", None))
    workspace_project_id = str(
        deployment_workspace.project_manifest().get("project_id") or ""
    )
    deployment_name = deployment_workspace.directory.name
    if _deployment_profile_path(deployment_name).exists():
        existing = _resolved_deployment_name(args)
        args.name = existing
        profile = _load_deployment_profile(existing)
        recorded_source = str(
            profile.get("source_workflow") or profile.get("workflow") or ""
        )
        recorded_cwd = Path(
            str(profile.get("source_cwd") or profile.get("cwd") or ".")
        ).expanduser()
        profile_project_id = str(profile.get("project_id") or "")
        source_moved = bool(
            workspace_project_id
            and profile_project_id == workspace_project_id
            and recorded_cwd.resolve() != deployment_workspace.root
        )
        selected = _workflow_source_identity(args.target, deployment_workspace.root)
        recorded = _workflow_source_identity(recorded_source, recorded_cwd)
        if selected != recorded:
            store = _store_status(str(profile.get("store") or ""))
            if store.get("state") not in {"empty", "missing"}:
                raise SystemExit(
                    "This project now selects a different workflow. Changing "
                    "the deployed program while durable state exists would "
                    "give its saved control positions a different meaning. "
                    "Run 'zg deploy reset --yes', then run 'zg deploy' again."
                )
            workflow, module = load_workflow_spec(args.target)
            spec = deployment_spec_from_module(module)
            profile["workflow"] = args.target
            profile["cwd"] = str(deployment_workspace.root)
            profile["source_workflow"] = args.target
            profile["source_cwd"] = str(deployment_workspace.root)
            for key in (
                "bundle",
                "bundled_files",
                "workflow_source",
                "deployment_spec",
                "secrets_file",
                "secret_names",
            ):
                profile.pop(key, None)
            profile["inputs"] = {}
            profile["options"] = {}
            profile["environment"] = {}
        else:
            # Redeploying the same program: retain its answered settings.
            if source_moved:
                profile["workflow"] = args.target
                profile["cwd"] = str(deployment_workspace.root)
                profile["source_workflow"] = args.target
                profile["source_cwd"] = str(deployment_workspace.root)
            profile, workflow, module, spec = _deployment_context_from_profile(
                profile, source=True
            )
    else:
        args.name = deployment_name
        workflow, module = load_workflow_spec(args.target)
        spec = deployment_spec_from_module(module)
        profile = {
            "schema_version": DEPLOYMENT_PROFILE_SCHEMA_VERSION,
            "name": deployment_name,
            "project_id": workspace_project_id,
            "workflow": args.target,
            "cwd": str(Path.cwd()),
            "source_workflow": args.target,
            "source_cwd": str(Path.cwd()),
            "store": _default_deployment_store_path(deployment_name),
            "log": _default_deployment_log_path(deployment_name),
            "llm": None,
            "llms": {},
            "llm_idle_timeout": None,
            "llm_settings": {},
            "assistant": None,
            "assistants": {},
            "options": {},
            "inputs": {},
            "environment": {},
            "timeout": 0.0,
            "execution": "sqlite",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": sys.executable,
        }

    profile["schema_version"] = DEPLOYMENT_PROFILE_SCHEMA_VERSION
    profile["project_id"] = workspace_project_id
    # The service must resolve external executables under the same PATH that
    # passed this deployment's readiness checks.  launchd does not preserve an
    # interactive shell's PATH, so make it part of the immutable profile and
    # let `_deployment_environment` apply it to doctor and runtime alike.
    profile["executable_search_path"] = os.environ.get("PATH", os.defpath)
    # A deployment snapshots the project's current routing.  Re-deploying is
    # what applies later edits to zippergen.toml; configuring an already
    # prepared deployment remains an explicit profile-only operation.
    from zippergen.durable_runs import default_llm_spec
    from zippergen.workspace import Workspace, WorkspaceError

    source_project = str(profile.get("source_cwd") or profile.get("cwd") or "")
    model_workspace = Workspace(
        getattr(args, "project", None) or source_project or None
    )
    source_workflow = str(
        profile.get("source_workflow") or profile.get("workflow") or args.target
    )
    try:
        model_workflow_spec = model_workspace.resolve_workflow(source_workflow)
    except WorkspaceError:
        model_workflow_spec = model_workspace.canonical_spec(
            source_workflow,
            cwd=model_workspace.root,
        )
    model_routing = project_model_routing(
        model_workspace,
        model_workflow_spec,
        workflow,
        fallback_default=default_llm_spec(module),
    )
    profile["llm"] = model_routing.default_spec
    profile["llms"] = model_routing.overrides
    profile["llm_idle_timeout"] = None
    profile["llm_settings"] = {
        target: chosen.as_dict()
        for target, chosen in model_routing.settings.items()
    }
    assistant_routing = project_assistant_routing(
        model_workspace,
        model_workflow_spec,
        workflow,
        module=module,
    )
    profile["assistant"] = assistant_routing.default_backend
    profile["assistants"] = assistant_routing.overrides
    if not args.yes and sys.stdin.isatty() and spec.description:
        print(spec.description)
        print()
    values, secrets = _apply_deploy_arguments(profile, args, spec, workflow)
    return _finalize_guided_deployment(
        profile, spec, workflow, values, secrets, args
    )


def _run_deployment_command(args) -> int:
    profile = _load_deployment_profile(args.name)
    cwd = Path(str(profile.get("cwd") or ".")).expanduser()
    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        with _hold_deployment_execution(profile):
            with _profile_environment(profile):
                _start_deployment_connector_workers(profile)
                return _run_workflow_command(
                    _run_args_from_deployment(profile)
                )
    finally:
        os.chdir(old_cwd)


def _launch_deployment_command(args) -> int:
    """Bootstrap the active profile, then replace this process with its runtime."""

    profile = _load_deployment_profile(args.profile)
    python = Path(str(profile.get("python") or "")).expanduser()
    if not python.is_file():
        raise SystemExit(
            f"Deployment runtime is missing: {python}. Run 'zg deploy' to "
            "rebuild the managed environment."
        )
    arguments = [
        str(python),
        "-m",
        "zippergen.serve",
        "__run-deployment",
        "--profile",
        _slug(str(profile["name"])),
    ]
    os.execve(str(python), arguments, dict(os.environ))
    raise AssertionError("os.execve returned")


# launchd and systemd each have their own words for the same few situations.
# Someone asking "is it running?" should not have to learn either vocabulary.
_SERVICE_WORDS = {
    "running": "running",
    "restarting": "restarting",
    "not-loaded": "stopped",
    "loaded": "stopped",
    "completed": "finished",
    "unknown": "unknown",
}


def _service_summary(service: Mapping[str, object]) -> str:
    """Render the service state in plain words, keeping any real detail."""

    raw = str(service.get("state") or "unknown")
    word = _SERVICE_WORDS.get(raw, raw)
    detail = str(service.get("detail") or "").strip()
    # Drop a detail that only restates the state, such as "not loaded".
    restates = detail.casefold().replace("-", " ") == raw.casefold().replace("-", " ")
    if not detail or restates:
        return word
    return f"{word} ({detail})"


def _status_command(args) -> int:
    """Answer the two questions "deploy status" is asked.

    Is this deployment running, and what state does it hold? Only the second
    used to be reported, which left the obvious question to 'deploy check' or
    'deploy list'.
    """

    from zippergen.deployment_platform import deployment_service_status

    profile = _load_deployment_profile(args.name)
    service = deployment_service_status(args.name)
    status = _store_status(str(profile["store"]))
    status["deployment"] = args.name
    status["service"] = service
    status["freshness"] = deployment_freshness_checks(profile)
    if args.json:
        print(json.dumps(status, default=str))
        return 0
    print(f"Deployment: {args.name}")
    print(f"Service: {_service_summary(service)}")
    workflow = profile.get("workflow")
    if workflow:
        print(f"Workflow: {workflow}")
    for check in status["freshness"]:
        marker = "OK" if check["status"] == "ok" else "WARN"
        print(f"{marker} {check['name']}: {check['detail']}")
    _print_status(status)
    other = _other_execution_line(args, owner="deploy")
    if other:
        print(other)
    # The values themselves belong to the project, and `zippergen config`
    # shows them. What a deployment adds is whether the running service still
    # matches them, which the freshness checks above already report.
    return 0


@dataclass(frozen=True)
class _OtherExecution:
    """What this project's other half is doing, or why we cannot say.

    A project has two places its workflow can run -- a deployed service and a
    selected durable run -- and only one may execute at a time. Each status
    command owns one half, so it must be able to say something about the
    other.

    "Nothing there" and "could not tell" are different answers and are kept
    apart here. Collapsing them let a corrupt profile or an unreachable
    service manager read exactly like a project that simply has no
    deployment, which is the misleading outcome this line exists to prevent.
    """

    #: absent | idle | executing | unreadable
    state: str
    detail: str = ""

    @property
    def line(self) -> str | None:
        """The sentence to print, or None when there is nothing to say."""

        if self.state == "executing":
            return f"Also executing: {self.detail}"
        if self.state == "unreadable":
            return f"WARN other execution: {self.detail}"
        return None


def _other_deployment_execution(args) -> _OtherExecution:
    from zippergen.deployment_platform import (
        deployment_service_status,
        service_is_running,
    )

    try:
        name = _deployment_name_for_project(args)
    except (SystemExit, Exception) as exc:
        return _OtherExecution(
            "unreadable",
            "the project deployment could not be identified "
            f"({type(exc).__name__}: {exc}); see zippergen deploy list.",
        )
    if name is None:
        return _OtherExecution("absent")
    try:
        service = deployment_service_status(name)
    except Exception as exc:
        return _OtherExecution(
            "unreadable",
            f"deployment {name} could not be queried "
            f"({type(exc).__name__}: {exc}); see zippergen deploy status.",
        )
    state = str(service.get("state") or "unknown")
    if state == "unknown":
        return _OtherExecution(
            "unreadable",
            f"deployment {name} state is unknown "
            f"({service.get('detail') or 'the service manager could not be asked'}); "
            "it may be running. See zippergen deploy status.",
        )
    if not service_is_running(service):
        return _OtherExecution("idle")
    return _OtherExecution(
        "executing",
        f"deployment {name} is running. Its state is separate from this "
        "run; see zippergen deploy status.",
    )


def _other_run_execution(args) -> _OtherExecution:
    from zippergen.workspace import Workspace

    try:
        record = Workspace(getattr(args, "project", None)).current_run()
    except (SystemExit, Exception) as exc:
        return _OtherExecution(
            "unreadable",
            f"the selected durable run could not be read "
            f"({type(exc).__name__}: {exc}); see zippergen run status.",
        )
    if record is None:
        return _OtherExecution("absent")
    status = str(record.get("status") or "")
    if status not in {"running", "waiting"}:
        return _OtherExecution("idle")
    waiting = " and is waiting for a person" if status == "waiting" else ""
    return _OtherExecution(
        "executing",
        f"durable run {record['run_id']} is {status}{waiting}. Its state is "
        "separate from this deployment; see zippergen run status.",
    )


def _other_execution_line(args, *, owner: str) -> str | None:
    """Name the project's other execution, or warn that it cannot be read."""

    observed = (
        _other_deployment_execution(args)
        if owner == "run"
        else _other_run_execution(args)
    )
    return observed.line


def _run_status_command(args) -> int:
    """Show the selected durable run and the state it owns."""

    from zippergen.workspace import Workspace

    record = Workspace(getattr(args, "project", None)).current_run()
    if record is None:
        if args.json:
            print(json.dumps({"run": None, "state": "absent"}))
        else:
            print("Current durable run: none")
            print("Start one with: zippergen run --durable")
        return 0
    status = _store_status(str(record["store"]))
    payload = {
        "run": record.get("run_id"),
        "workflow": record.get("workflow_spec"),
        "run_status": record.get("status"),
        "updated_at": record.get("updated_at"),
        "durable_state": status,
    }
    if args.json:
        print(json.dumps(payload, default=str))
    else:
        print(f"Run: {payload['run']}")
        print(f"Status: {payload['run_status']}")
        print(f"Workflow: {payload['workflow']}")
        _print_status(status)
        other = _other_execution_line(args, owner="run")
        if other:
            print(other)
    return 0


def _run_reset_command(args) -> int:
    """Discard the selected durable run, optionally retaining an archive."""

    from zippergen.durable_runs import RunResetError, reset_current_run
    from zippergen.workspace import Workspace

    workspace = Workspace(getattr(args, "project", None))
    record = workspace.current_run()
    if record is None:
        raise SystemExit("There is no current durable run to reset.")
    status = str(record.get("status") or "")
    if status in {"running", "waiting"} and not args.force:
        raise SystemExit(
            f"Run {record['run_id']} is recorded as {status}. Stop its "
            "foreground process with Ctrl-C first. If that process is "
            "already gone, re-run with --force."
        )
    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Resetting a durable run requires confirmation. Re-run with --yes."
            )
        question = (
            "Archive and clear the current durable run? [y/N]: "
            if args.archive
            else "Permanently discard the current durable run? [y/N]: "
        )
        answer = input(question).strip().casefold()
        if answer not in {"y", "yes"}:
            print("Nothing was changed.")
            return 1
    try:
        reset, archive, store_file_count = reset_current_run(
            workspace,
            archive=args.archive,
            force=args.force,
        )
    except RunResetError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Reset durable run: {reset['run_id']}")
    if archive is None:
        print(
            "Permanently discarded its run record and "
            f"{store_file_count} SQLite file(s)."
        )
    else:
        print(
            "Archived its run record and "
            f"{store_file_count} SQLite file(s): {archive}"
        )
    print("Current durable run: none")
    print("Start a new one with: zippergen run --durable")
    return 0


def _execution_age(updated_at: float | None) -> str:
    if updated_at is None:
        return "-"
    seconds = max(0, int(time.time() - updated_at))
    if seconds < 2:
        return "just now"
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h" if hours < 48 else f"{hours // 24}d"


def _inspection_context(args) -> tuple[Workflow, str, str]:
    """Resolve one deployment or durable run without mutating its state."""

    from zippergen.workspace import Workspace, WorkspaceError

    if getattr(args, "execution_owner", "run") == "deploy":
        profile = _load_deployment_profile(_resolved_deployment_name(args))
        workflow_spec = str(profile["workflow"])
        cwd = Path(str(profile.get("cwd") or ".")).expanduser()
        old_cwd = Path.cwd()
        try:
            os.chdir(cwd)
            workflow, _module = load_workflow_spec(workflow_spec)
        finally:
            os.chdir(old_cwd)
        return workflow, str(profile["store"]), "project deployment"

    workspace = Workspace(args.project)
    record = workspace.current_run()
    if record is None:
        raise SystemExit(
            "There is no current durable run. Start one with "
            "'zippergen run --durable'."
        )
    workflow_spec = str(record["workflow_spec"])
    workflow, _module = load_workflow_spec(
        str(workspace.absolute_spec(workflow_spec))
    )
    return workflow, str(record["store"]), f"run {record['run_id']}"


def _inspection_snapshot(workflow: Workflow, store: str, agent: str | None):
    from zippergen.execution_inspection import (
        default_focus,
        participant_positions,
        read_execution_states,
    )

    observed = read_execution_states(store)
    positions = participant_positions(workflow, observed)
    names = [position.participant for position in positions]
    if agent:
        focus = next(
            (name for name in names if name.casefold() == agent.casefold()),
            None,
        )
        if focus is None:
            raise SystemExit(
                f"Unknown participant {agent!r}. Available: "
                f"{', '.join(names) or 'none'}."
            )
    else:
        focus = default_focus(positions)
    return observed, positions, focus


def _render_inspection(
    workflow: Workflow,
    store: str,
    subject: str,
    observed,
    positions,
    focus: str | None,
    renderer,
    *,
    projection_indent: int = 0,
) -> None:
    from zippergen.execution_inspection import state_label
    from zippergen.view import render_local_projection_with_pointers

    renderer.section("Execution positions")
    renderer.emit(f"Subject: {subject}")
    renderer.emit(f"Store: {store}")
    if not observed:
        renderer.status(
            "warning",
            "No position data. The run may not have started yet.",
        )
    renderer.emit()
    renderer.columns(
        "Participants",
        (
            "Focus",
            "Participant",
            "State",
            "Last committed position",
            "Elapsed",
        ),
        [
            (
                "▶" if position.participant == focus else "",
                position.participant,
                state_label(position.state),
                position.location,
                _execution_age(position.updated_at),
            )
            for position in positions
        ],
    )
    if focus is not None:
        selected = next(
            position for position in positions
            if position.participant == focus
        )
        renderer.emit()
        renderer.section(f"{focus} local projection")
        renderer.emit(
            render_local_projection_with_pointers(
                workflow,
                focus,
                selected.locators,
                indent=projection_indent,
            )
        )


def _inspect_command(args) -> int:
    from zippergen.execution_inspection import state_label
    from zippergen.rendering import TerminalRenderer

    if args.watch and args.json:
        raise SystemExit("Use either --watch or --json, not both.")
    if args.interval is not None and not args.watch:
        raise SystemExit("--interval requires --watch.")
    interval = 1.0 if args.interval is None else args.interval
    if not math.isfinite(interval) or interval <= 0:
        raise SystemExit("--interval must be a positive number.")

    workflow, store, subject = _inspection_context(args)
    observed, positions, focus = _inspection_snapshot(
        workflow,
        store,
        args.agent,
    )

    if args.json:
        print(json.dumps({
            "subject": subject,
            "store": store,
            "focus": focus,
            "positions": [
                {
                    "participant": position.participant,
                    "state": position.state,
                    "state_label": state_label(position.state),
                    "locators": [list(path) for path in position.locators],
                    "location": position.location,
                    "updated_at": position.updated_at,
                    "detail": position.detail,
                }
                for position in positions
            ],
        }, indent=2, default=str))
        return 0

    if args.watch:
        from zippergen.live_display import (
            live_display_available,
            watch_frames,
        )

        if not live_display_available():
            raise SystemExit("--watch requires an interactive terminal.")

        def frame(columns: int) -> str:
            live_observed, live_positions, live_focus = _inspection_snapshot(
                workflow,
                store,
                args.agent,
            )
            lines: list[str] = []
            live_renderer = TerminalRenderer(
                output=lines.append,
                color=False,
                columns=lambda: columns,
            )
            _render_inspection(
                workflow,
                store,
                subject,
                live_observed,
                live_positions,
                live_focus,
                live_renderer,
                projection_indent=1,
            )
            return "\n".join(lines)

        watch_frames(frame, interval=interval)
        print(f"Stopped watching {subject}. The execution was not interrupted.")
        return 0

    renderer = TerminalRenderer()
    _render_inspection(
        workflow,
        store,
        subject,
        observed,
        positions,
        focus,
        renderer,
    )
    return 0


def _trace_command(args) -> int:
    if args.interval is not None and not args.follow:
        raise SystemExit("--interval requires --follow.")
    interval = 0.25 if args.interval is None else args.interval
    if args.follow and (
        not math.isfinite(interval) or interval <= 0
    ):
        raise SystemExit("--interval must be greater than 0.")
    execution = _resolve_execution_reference(args)
    events = _load_trace_events(
        execution.store,
        after_rowid=args.after,
        limit=args.tail,
    )
    follow_widths: tuple[int, ...] | None = None
    if args.json and args.follow:
        for event in events:
            print(json.dumps(event, default=str), flush=True)
    elif args.json:
        print(json.dumps(events, default=str))
    else:
        _print_execution_reference(execution)
        if args.follow:
            follow_widths = _print_trace_follow_table(events)
        else:
            _print_trace_events(events)
    if not args.follow:
        return 0

    after_rowid = max(
        (int(event["rowid"]) for event in events),
        default=int(args.after),
    )
    try:
        while True:
            time.sleep(interval)
            while True:
                new_events = _load_trace_events(
                    execution.store,
                    after_rowid=after_rowid,
                    limit=args.tail,
                    newest=False,
                )
                if not new_events:
                    break
                after_rowid = int(new_events[-1]["rowid"])
                if args.json:
                    for event in new_events:
                        print(json.dumps(event, default=str), flush=True)
                else:
                    if follow_widths is None:
                        raise AssertionError("follow table layout was not prepared")
                    _print_trace_follow_events(new_events, follow_widths)
                if len(new_events) < args.tail:
                    break
    except KeyboardInterrupt:
        return 0


def _tasks_command(args) -> int:
    execution = _resolve_execution_reference(args)
    store = execution.store
    if not Path(store).expanduser().exists():
        raise SystemExit(f"Durable store does not exist yet: {store}")
    status = None if args.all else "pending"
    tasks = _load_human_tasks(
        store,
        status=status,
        limit=args.limit,
        with_tokens=args.tokens,
        token_channel=args.channel,
    )
    if args.json:
        print(json.dumps(tasks, default=str))
    else:
        _print_execution_reference(execution)
        _print_tasks(tasks, heading="Human tasks" if args.all else "Pending human tasks")
    return 0


def _approve_result_from_args(task: dict, args) -> dict:
    spec = task.get("spec") or {}

    try:
        if args.result_json is not None:
            result = json.loads(args.result_json)
            return validate_human_task_result(
                spec, result, context="--result-json"
            )

        if args.yes and args.no:
            raise ValueError("Use only one of --yes or --no.")
        if args.value is not None and (args.yes or args.no):
            raise ValueError("Use either --value or --yes/--no, not both.")

        if args.no:
            return human_task_result_from_value(spec, False)
        if args.yes:
            return human_task_result_from_value(spec, True)
        if args.value is not None:
            return human_task_result_from_value(spec, args.value)
        if spec.get("output_type") == "str":
            raise ValueError(
                f"Task {task['task_id']} requires --value for output "
                f"{spec.get('output')!r}."
            )
        return human_task_result_from_value(spec)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--result-json must be valid JSON: {exc.msg}") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _approve_command(args) -> int:
    store_path = str(Path(_resolve_store_arg(args)).expanduser())
    if not Path(store_path).exists():
        raise SystemExit(f"Durable store does not exist yet: {store_path}")
    conn = open_store(store_path)
    try:
        token_record = None
        task_id = args.task
        if args.token_stdin:
            token = sys.stdin.readline().strip()
            if not token:
                raise SystemExit("No human task token was received on stdin.")
            token_record = load_human_task_token(conn, token)
            if token_record is None:
                raise SystemExit("Human task token was not found.")
            task_id = token_record["task_id"]
        task = load_human_task(conn, task_id)
        if task is None:
            raise SystemExit(f"Human task not found: {task_id}")
        if task["status"] != "pending":
            raise SystemExit(f"Human task {task_id} is already {task['status']}.")
        result = _approve_result_from_args(task, args)
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = complete_human_task(conn, task_id, result)
            if token_record is not None:
                mark_human_task_token_used(conn, token_record["token"])
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()

    if args.json:
        print(json.dumps(task, default=str))
    else:
        print(f"Completed human task {task['task_id']}: {json.dumps(task['result'], default=str)}")
    return 0


def _notify_stdout_command(args) -> int:
    store_path = str(Path(args.store).expanduser())
    if not Path(store_path).exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    seen: set[str] = set()
    while True:
        tasks = _load_human_tasks(
            store_path,
            status="pending",
            limit=args.limit,
            with_tokens=False,
            token_channel=args.channel,
        )
        emitted = 0
        for task in tasks:
            task_id = task["task_id"]
            if task_id in seen:
                continue
            _notify_stdout_task(task)
            seen.add(task_id)
            emitted += 1
        if not args.watch:
            if emitted == 0 and not args.quiet:
                print("No pending human tasks.")
            return 0
        time.sleep(args.interval)


def _notify_telegram_command(args) -> int:
    from zippergen.telegram_notify import (
        TelegramAPIError,
        TelegramBotClient,
        TelegramDeploymentNotifier,
        TelegramNotifier,
        load_telegram_chat_id,
        load_telegram_token,
    )

    store_path = str(Path(args.store).expanduser())
    if not Path(store_path).exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    token = load_telegram_token()
    chat_id = load_telegram_chat_id(args.chat_id)
    if not chat_id:
        raise SystemExit("Telegram chat id is required. Set ZIPPERGEN_TELEGRAM_CHAT_ID or pass --chat-id.")
    client = TelegramBotClient(token)
    notifier = TelegramNotifier(
        store_path=store_path,
        client=client,
        chat_id=chat_id,
        allowed_user_id=(
            args.allowed_user_id
            or os.environ.get("ZIPPERGEN_TELEGRAM_ALLOWED_USER_ID")
            or chat_id
        ),
        channel=args.channel,
        limit=args.limit,
    )

    if not args.watch:
        sent = notifier.send_pending_once(resend=args.resend)
        processed = notifier.poll_updates_once(timeout=0)
        if not args.quiet:
            print(f"Telegram: sent {sent} task notification(s), processed {processed} update(s).")
        return 0

    if not args.quiet:
        print(f"Watching Telegram chat {chat_id} for store {store_path}.")
    delay = args.interval
    while True:
        # Same rule as the deployment poller: an outage is expected input, a
        # defect in our own code is not.
        try:
            sent = notifier.send_pending_once(resend=args.resend)
            processed = notifier.poll_updates_once(timeout=args.poll_timeout)
        except TelegramAPIError as exc:
            delay = min(delay * 2, TelegramDeploymentNotifier.MAX_RETRY_DELAY)
            print(f"Telegram retrying in {delay:g}s: {exc}", file=sys.stderr, flush=True)
            time.sleep(delay)
            continue
        delay = args.interval
        if not args.quiet and (sent or processed):
            print(f"Telegram: sent {sent} task notification(s), processed {processed} update(s).")
        time.sleep(delay)


def _add_guided_deployment_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--set", action="append", default=[], metavar="field=value", help="Declared deployment field value.")
    parser.add_argument("--timeout", type=_nonnegative_float_argument, help="Workflow timeout; defaults to 0 (no deadline).")
    parser.add_argument("--yes", action="store_true", help="Accept defaults and existing environment values without prompting.")
    parser.set_defaults(no_install=False, no_setup=False, no_doctor=False)


def _add_owned_execution_commands(subparsers, *, owner: str) -> None:
    """Register observation commands under the run or deployment that owns state."""

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="show the last committed durable program position",
    )
    inspect_parser.set_defaults(execution_owner=owner)
    inspect_parser.add_argument(
        "--project", default=argparse.SUPPRESS, help="Project root."
    )
    inspect_parser.add_argument("--agent", help="Participant whose local projection to show.")
    inspect_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    inspect_parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh the inspection in place until Ctrl-C.",
    )
    inspect_parser.add_argument(
        "--interval",
        type=float,
        help="Refresh interval in seconds for --watch. Default 1.",
    )

    trace_parser = subparsers.add_parser(
        "trace",
        help="show recent events from this execution",
    )
    trace_parser.set_defaults(execution_owner=owner)
    trace_parser.add_argument(
        "--project", default=argparse.SUPPRESS, help="Project root."
    )
    trace_parser.add_argument("--tail", type=_positive_int_argument, default=50, help="Maximum number of trace events to show.")
    trace_parser.add_argument("--after", type=_nonnegative_int_argument, default=0, help="Only show trace events after this event rowid.")
    trace_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    trace_parser.add_argument(
        "--follow",
        action="store_true",
        help="Print newly committed trace events until Ctrl-C.",
    )
    trace_parser.add_argument(
        "--interval",
        type=float,
        help="Polling interval in seconds for --follow. Default 0.25.",
    )

    tasks_parser = subparsers.add_parser(
        "tasks",
        help="list human tasks in this execution",
    )
    tasks_parser.set_defaults(execution_owner=owner)
    tasks_parser.add_argument(
        "--project", default=argparse.SUPPRESS, help="Project root."
    )
    tasks_parser.add_argument("--all", action="store_true", help="Include completed tasks.")
    tasks_parser.add_argument("--limit", type=_positive_int_argument, help="Maximum number of tasks to show.")
    tasks_parser.add_argument("--tokens", action="store_true", help="Generate/show durable approval tokens.")
    tasks_parser.add_argument("--channel", default="cli", help="Token channel name used with --tokens.")
    tasks_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    approve_parser = subparsers.add_parser(
        "approve",
        help="complete a pending human task in this execution",
    )
    approve_parser.set_defaults(execution_owner=owner)
    approve_parser.add_argument(
        "--project", default=argparse.SUPPRESS, help="Project root."
    )
    target = approve_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--task", help="Human task id.")
    target.add_argument(
        "--token-stdin",
        action="store_true",
        help="Read a durable approval token from one line of standard input.",
    )
    approve_parser.add_argument("--yes", action="store_true", default=argparse.SUPPRESS, help="Complete a boolean task with true.")
    approve_parser.add_argument("--no", action="store_true", help="Complete a boolean task with false.")
    approve_parser.add_argument("--value", help="Value for string tasks, or explicit true/false for boolean tasks.")
    approve_parser.add_argument("--result-json", help="Complete with an explicit JSON object result.")
    approve_parser.add_argument("--json", action="store_true", help="Print the completed task as JSON.")


def _reject_deploy_configuration_options(args, action: str) -> None:
    """Keep bare-deploy options from becoming silent no-ops on verbs."""

    used: list[str] = []
    if getattr(args, "set", None):
        used.append("--set")
    if getattr(args, "timeout", None) is not None:
        used.append("--timeout")
    if getattr(args, "no_start", False):
        used.append("--no-start")
    if getattr(args, "history_keep", None) is not None:
        used.append("--history-keep")
    if getattr(args, "yes", False) and action not in {
        "approve",
        "prune",
        "remove",
        "reset",
    }:
        used.append("--yes")
    if not used:
        return
    raise SystemExit(
        f"{', '.join(used)} configure a deployment and cannot be used with "
        f"'zg deploy {action}'. Run them with bare 'zg deploy' instead."
    )


def _project_google_scopes(connection: str) -> tuple[str, ...]:
    """Read the scopes off the workflow, or say plainly that it could not."""

    from zippergen.google_auth import GoogleConnectorError, google_scope_names
    from zippergen.project_configuration import project_google_scopes
    from zippergen.workspace import Workspace, WorkspaceError

    advice = (
        "Could not work out which Google scopes to request. Run this inside a "
        "project whose workflow declares Google connectors assigned to "
        f"{connection!r}, or pass them yourself with --scopes."
    )
    try:
        scopes = project_google_scopes(Workspace(None), connection)
    except (WorkspaceError, GoogleConnectorError, OSError, SystemExit, ValueError):
        raise SystemExit(advice) from None
    if not scopes:
        raise SystemExit(advice)
    print(
        "Scopes this workflow asks for: "
        + ", ".join(google_scope_names(scopes))
    )
    return scopes


def _provider_authorize_google_command(args) -> int:
    """Authorize Google on the computer that owns the browser."""

    from zippergen.google_auth import (
        GoogleConnectorError,
        authorize_google_client_result,
        encode_google_authorization,
        google_authorization_summary,
        google_scope_names,
        google_scopes_cover,
        normalize_google_client_json,
        parse_google_scopes,
    )

    try:
        args.name = _guided_required_value(
            args.name,
            label="Google connection",
            command="zg provider authorize CONNECTION",
            choices=_project_choices(
                "provider-connections-google", getattr(args, "project", None)
            ),
            # The list is a suggestion, not the whole world: authorizing for a
            # server names a connection that exists there, not necessarily on
            # the computer holding the browser.
            enforce_choices=False,
        )
        scopes = (
            parse_google_scopes(args.scopes)
            if args.scopes
            else _project_google_scopes(args.name)
        )
        entered = str(args.client or "").strip()
        if not entered:
            entered = input("Google OAuth Desktop app JSON path: ").strip()
        if not entered:
            raise GoogleConnectorError(
                "Select the OAuth Desktop app JSON downloaded from Google "
                "Cloud."
            )
        path = Path(entered).expanduser().resolve()
        if not path.is_file():
            raise GoogleConnectorError(
                f"Google OAuth desktop client JSON does not exist: {path}"
            )
        client_json = normalize_google_client_json(path.read_text())
        result = authorize_google_client_result(
            client_json,
            scopes=scopes,
        )
        if not google_scopes_cover(result.granted_scopes, scopes):
            missing = [
                name
                for scope, name in zip(
                    scopes, google_scope_names(scopes), strict=True
                )
                if not google_scopes_cover(
                    result.granted_scopes, (scope,)
                )
            ]
            raise GoogleConnectorError(
                "Google authorization did not grant: "
                + ", ".join(missing)
                + ". Run the command again and leave those permissions "
                "selected on Google's consent screen."
            )
        granted, client, expiry = google_authorization_summary(result)
        print("Google authorization completed.")
        print(f"Granted scopes: {granted}")
        print(f"OAuth client: {client}")
        print(f"Credential expiry: {expiry}")
        # Authorizing for the machine you are standing on is the ordinary
        # case, and it has somewhere to put the result. Printing a refresh
        # token for a person to copy back into the same computer puts a live
        # credential through the screen and the shell history for no reason.
        if not args.handoff:
            from zippergen.workspace import Workspace, WorkspaceError

            try:
                workspace = Workspace(None)
            except (WorkspaceError, SystemExit):
                workspace = None
            if workspace is not None and _save_google_authorization(
                workspace, args.name, result
            ):
                return 0
        print(
            f"In the project that will use it, run 'zippergen provider accept "
            f"{args.name}' "
            "and paste the private result below. It contains a refresh token, "
            "so do not share it or save it in shell history."
        )
        print(encode_google_authorization(result))
        return 0
    except (OSError, GoogleConnectorError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


# Registered but kept out of help. `__complete` backs shell completion,
# `__run-deployment` is the generated service entry point, and `notify` is an
# adapter a deployment runs for itself rather than a user task.
HIDDEN_COMMANDS = frozenset({
    "__complete", "__launch-deployment", "__run-deployment", "notify"
})


class _NoAbbrevArgumentParser(argparse.ArgumentParser):
    """Argument parser whose long options must always be spelled exactly."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def _parse_cli_args(
    argv=None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Build the real CLI parser and parse arguments without dispatching."""

    ap = _NoAbbrevArgumentParser(
        prog="zippergen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")

    config = sub.add_parser(
        "config",
        help="show the effective project configuration and local availability",
    )
    config.add_argument("--json", action="store_true", help="Print JSON.")
    config.add_argument("--project", help="Project root.")
    config.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="field=value",
        help=(
            "Answer one of the workflow's declared configuration questions "
            "and store it in zippergen.toml. Repeatable."
        ),
    )
    config.add_argument(
        "--unset",
        action="append",
        default=[],
        metavar="field",
        help="Forget one stored answer. Repeatable.",
    )

    check = sub.add_parser(
        "check",
        help="check the workflow, routing, credentials, and live providers",
    )
    check.add_argument("--json", action="store_true", help="Print JSON.")
    check.add_argument("--project", help="Project root.")
    check.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when something is not ready, for scripts.",
    )

    workflow_parser = sub.add_parser(
        "workflow",
        help="show or select the workflow this project is about",
    )
    workflow_parser.add_argument("--project", help="Project root.")
    workflow_sub = workflow_parser.add_subparsers(dest="workflow_action")
    workflow_select = workflow_sub.add_parser(
        "select",
        help="select the project's workflow entry",
    )
    workflow_select.add_argument(
        "spec", nargs="?", help="Workflow to select, as path.py:name."
    )
    workflow_select.add_argument(
        "--project",
        default=argparse.SUPPRESS,
        help="Project root.",
    )

    provider = sub.add_parser(
        "provider",
        help="show and manage named provider connections and credentials",
        description=(
            "One pattern: configure NAME KIND, add its credential, then let "
            "model and connector configurations reuse that connection."
        ),
    )
    provider_sub = provider.add_subparsers(dest="provider_action")
    provider_configure = provider_sub.add_parser(
        "configure", help="save one named provider connection"
    )
    provider_configure.add_argument("name", nargs="?")
    provider_configure.add_argument(
        "kind",
        nargs="?",
        choices=(
            "openai", "anthropic", "mistral", "local", "scripted",
            "telegram", "google",
        ),
    )
    provider_configure.add_argument(
        "--base-url", help="OpenAI-compatible endpoint for a local connection."
    )
    provider_configure.add_argument("--project", help="Project root.")
    provider_set_credential = provider_sub.add_parser(
        "set-credential",
        help="save this connection's private API key or bot token",
    )
    provider_set_credential.add_argument("name", nargs="?")
    provider_set_credential.add_argument("--project", help="Project root.")
    provider_check = provider_sub.add_parser(
        "check", help="check provider credentials and local readiness"
    )
    provider_check.add_argument("name", nargs="?")
    provider_check.add_argument("--project", help="Project root.")
    provider_check.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when something is not ready, for scripts.",
    )
    provider_rename = provider_sub.add_parser(
        "rename", help="rename a connection and everything that names it"
    )
    provider_rename.add_argument("name", nargs="?")
    provider_rename.add_argument("new_name", nargs="?")
    provider_rename.add_argument("--project", help="Project root.")
    provider_remove = provider_sub.add_parser(
        "remove", help="remove an unused provider connection"
    )
    provider_remove.add_argument("name", nargs="?")
    provider_remove.add_argument("--project", help="Project root.")
    provider_remove.add_argument(
        "--yes",
        action="store_true",
        help="Delete a stored credential without asking.",
    )
    provider_authorize = provider_sub.add_parser(
        "authorize",
        help="authorize Google here, or with --handoff for another computer",
    )
    provider_authorize.add_argument("name", nargs="?", help="Google connection name.")
    provider_authorize.add_argument("--client", help="OAuth Desktop app JSON.")
    provider_authorize.add_argument(
        "--handoff",
        action="store_true",
        help=(
            "Print the result for another computer instead of saving it here. "
            "Use this to authorize a server from your laptop."
        ),
    )
    provider_authorize.add_argument(
        "--scopes",
        help=(
            "Comma-separated Gmail/Sheets scopes. Omit inside a project to "
            "use the scopes its workflow actually asks for."
        ),
    )
    provider_accept = provider_sub.add_parser(
        "accept", help="save a Google authorization produced elsewhere"
    )
    provider_accept.add_argument("name", nargs="?", help="Google connection name.")
    provider_accept.add_argument("--project", help="Project root.")

    model = sub.add_parser(
        "model",
        help="show and manage model configurations and assignments",
        description=(
            "One pattern: configure NAME CONNECTION MODEL, then assign "
            "TARGET NAME. "
            "Run without an action to show everything. In a terminal, omit "
            "required values to be guided."
        ),
    )
    model_sub = model.add_subparsers(dest="model_action")
    model_configure = model_sub.add_parser(
        "configure",
        help="save one named model configuration",
    )
    model_configure.add_argument("name", nargs="?")
    model_configure.add_argument("connection", nargs="?")
    model_configure.add_argument("model", nargs="?")
    model_configure.add_argument(
        "--idle-timeout",
        type=float,
        help=(
            "Unload a local Ollama model after this many seconds without an "
            "active call; 0 unloads after each call."
        ),
    )
    model_configure.add_argument(
        "--temperature",
        type=float,
        help=(
            "Default sampling temperature from 0 to 1; an @llm action may "
            "override it."
        ),
    )
    model_configure.add_argument(
        "--max-tokens",
        type=int,
        help="Most tokens this model may generate in one response.",
    )
    model_configure.add_argument(
        "--timeout",
        type=float,
        help="Seconds to wait for one response before giving up.",
    )
    model_configure.add_argument("--project", help="Project root.")
    model_assign = model_sub.add_parser(
        "assign",
        help="assign a named configuration to a participant or action",
    )
    model_assign.add_argument("target", nargs="?", help="default, Participant, or Participant.action.")
    model_assign.add_argument("configuration", nargs="?")
    model_assign.add_argument("--project", help="Project root.")
    model_unassign = model_sub.add_parser("unassign", help="remove one assignment")
    model_unassign.add_argument("target", nargs="?")
    model_unassign.add_argument("--project", help="Project root.")
    model_check = model_sub.add_parser(
        "check",
        help="check model readiness with a small provider request",
    )
    model_check.add_argument("name", nargs="?")
    model_check.add_argument("--project", help="Project root.")
    model_check.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when something is not ready, for scripts.",
    )
    model_rename = model_sub.add_parser(
        "rename", help="rename a configuration and every assignment naming it"
    )
    model_rename.add_argument("name", nargs="?")
    model_rename.add_argument("new_name", nargs="?")
    model_rename.add_argument("--project", help="Project root.")
    model_remove = model_sub.add_parser("remove", help="remove an unused configuration")
    model_remove.add_argument("name", nargs="?")
    model_remove.add_argument("--project", help="Project root.")

    assistant = sub.add_parser(
        "assistant",
        help="show and manage coding-assistant configurations and assignments",
        description=(
            "One pattern: configure NAME BACKEND, then assign TARGET NAME. "
            "Run without an action to show everything. In a terminal, omit "
            "required values to be guided."
        ),
    )
    assistant_sub = assistant.add_subparsers(dest="assistant_action")
    assistant_configure = assistant_sub.add_parser(
        "configure",
        help="save one named Codex or Claude configuration",
    )
    assistant_configure.add_argument(
        "name",
        nargs="?",
        help="User-defined configuration name, such as coding-agent.",
    )
    assistant_configure.add_argument(
        "backend",
        nargs="?",
        choices=ASSISTANT_BACKENDS,
        help="Coding-assistant CLI selected by this configuration.",
    )
    assistant_configure.add_argument("--project", help="Project root.")
    assistant_assign = assistant_sub.add_parser(
        "assign",
        help="assign a configuration to a participant or exact action",
    )
    assistant_assign.add_argument(
        "target",
        nargs="?",
        help="default, Participant, or Participant.action.",
    )
    assistant_assign.add_argument("configuration", nargs="?")
    assistant_assign.add_argument("--project", help="Project root.")
    assistant_unassign = assistant_sub.add_parser(
        "unassign",
        help="remove one assistant assignment",
    )
    assistant_unassign.add_argument("target", nargs="?")
    assistant_unassign.add_argument("--project", help="Project root.")
    assistant_check = assistant_sub.add_parser(
        "check",
        help="check CLI availability and required safety options",
    )
    assistant_check.add_argument("name", nargs="?")
    assistant_check.add_argument("--project", help="Project root.")
    assistant_check.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when something is not ready, for scripts.",
    )
    assistant_rename = assistant_sub.add_parser(
        "rename", help="rename a configuration and every assignment naming it"
    )
    assistant_rename.add_argument("name", nargs="?")
    assistant_rename.add_argument("new_name", nargs="?")
    assistant_rename.add_argument("--project", help="Project root.")
    assistant_remove = assistant_sub.add_parser(
        "remove",
        help="remove an unused assistant configuration",
    )
    assistant_remove.add_argument("name", nargs="?")
    assistant_remove.add_argument("--project", help="Project root.")

    completion = sub.add_parser(
        "completion",
        help="print shell completion for zsh, bash, or fish",
    )
    completion.add_argument(
        "shell", nargs="?", choices=("zsh", "bash", "fish")
    )
    internal_completion = sub.add_parser("__complete")
    internal_completion.add_argument("kind")
    internal_completion.add_argument("path", nargs="*")
    internal_completion.add_argument("--project")

    connector = sub.add_parser(
        "connector",
        help="show and manage connector configurations and assignments",
        description=(
            "One pattern: configure NAME CONNECTION [KIND], then assign "
            "TARGET NAME. A target is either a service requirement the "
            "workflow declares or a participant whose actions ask a human; "
            "the workflow says which. KIND is inferred when the connection "
            "has only one connector kind. Run without an action to show "
            "everything. In a terminal, omit required values to be guided."
        ),
    )
    connector_sub = connector.add_subparsers(
        dest="connector_action",
    )
    connector_configure = connector_sub.add_parser(
        "configure",
        help="save one named connector configuration",
    )
    connector_configure.add_argument(
        "name",
        nargs="?",
        help="Configuration name, such as approval-chat, inbox, or records.",
    )
    connector_configure.add_argument("connection", nargs="?")
    connector_configure.add_argument(
        "kind",
        nargs="?",
        choices=CONNECTOR_KINDS,
        help="Required only when the connection supports several connector kinds.",
    )
    connector_configure.add_argument(
        "--project",
        help="Project root; defaults to discovery from the current directory.",
    )
    for setting in CONNECTOR_SETTING_SPECS:
        connector_configure.add_argument(
            "--" + setting.name.replace("_", "-"),
            dest=setting.name,
            metavar=setting.metavar,
            help=setting.help,
        )

    connector_assign = connector_sub.add_parser(
        "assign",
        help="route a participant's human actions to a saved connector",
    )
    connector_assign.add_argument(
        "target",
        nargs="?",
        help="Participant, or Participant.action for a single action.",
    )
    connector_assign.add_argument(
        "configuration",
        nargs="?",
        help="Name of a saved connector configuration.",
    )
    connector_assign.add_argument("--project", help="Project root.")

    connector_unassign = connector_sub.add_parser(
        "unassign",
        help="remove one human-action assignment",
    )
    connector_unassign.add_argument("target", nargs="?")
    connector_unassign.add_argument("--project", help="Project root.")


    connector_check = connector_sub.add_parser(
        "check",
        help="check connector readiness by contacting providers",
    )
    connector_check.add_argument("name", nargs="?")
    connector_check.add_argument("--project", help="Project root.")
    connector_check.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when something is not ready, for scripts.",
    )

    connector_rename = connector_sub.add_parser(
        "rename", help="rename a configuration and everything that names it"
    )
    connector_rename.add_argument("name", nargs="?")
    connector_rename.add_argument("new_name", nargs="?")
    connector_rename.add_argument("--project", help="Project root.")
    connector_remove = connector_sub.add_parser(
        "remove",
        help="remove an unused connector configuration",
    )
    connector_remove.add_argument("name", nargs="?")
    connector_remove.add_argument("--project", help="Project root.")

    rn = sub.add_parser(
        "run",
        allow_abbrev=False,
        help="run this project, and inspect a recorded run",
        description=(
            "Run bare for a disposable execution. --durable records a new "
            "run and --resume continues it. The verbs inspect that recorded run."
        ),
    )
    rn.add_argument(
        "--workflow",
        help=(
            "Explicit workflow spec; normally inferred from the project."
        ),
    )
    rn.add_argument("--llm", metavar="SPEC", help="LLM spec: mock, openai:gpt-4o, ollama:qwen2.5:7b, ...")
    rn.add_argument(
        "--llm-for",
        action="append",
        default=[],
        metavar="PARTICIPANT_OR_ACTION=SPEC",
        help="Override the LLM for one participant or exact action; repeat as needed.",
    )
    rn.add_argument("--llm-idle-timeout", type=float, help="Release a managed local LLM after this many idle seconds.")
    rn.add_argument(
        "--llm-idle-timeout-for",
        action="append",
        default=[],
        metavar="PARTICIPANT_OR_ACTION=SECONDS",
        help=argparse.SUPPRESS,
    )
    rn.add_argument(
        "--durable",
        action="store_true",
        help=(
            "Record the run so it can be resumed, and collect any missing "
            "inputs interactively."
        ),
    )
    rn.add_argument(
        "--resume",
        action="store_true",
        help="Continue the project's most recent unfinished run.",
    )
    rn.add_argument(
        "--history-keep",
        type=_nonnegative_int_argument,
        default=None,
        metavar="ROWS",
        help=(
            "How many trace rows this run's store keeps. 0 records none. "
            "Default 10000."
        ),
    )
    rn.add_argument(
        "--project",
        help="Project root for a durable run; defaults to discovery.",
    )
    rn.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt for missing inputs during a durable run.",
    )
    rn.add_argument("--input", action="append", default=[], metavar="name=value", help="Workflow input value.")
    rn.add_argument("--input-json", help="Workflow inputs as a JSON object.")
    rn.add_argument("--option", action="append", default=[], metavar="name=value", help="Option passed to zippergen_setup(config).")
    rn.add_argument("--timeout", type=_nonnegative_float_argument, default=0.0, help="Workflow timeout in seconds. Default 0 (no deadline).")
    run_sub = rn.add_subparsers(dest="run_action")
    run_status = run_sub.add_parser(
        "status",
        help="show the selected durable run and its state",
    )
    run_status.set_defaults(execution_owner="run")
    run_status.add_argument(
        "--project", default=argparse.SUPPRESS, help="Project root."
    )
    run_status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    run_reset = run_sub.add_parser(
        "reset",
        help="discard the selected durable run",
    )
    run_reset.add_argument(
        "--project", default=argparse.SUPPRESS, help="Project root."
    )
    run_reset.add_argument(
        "--yes",
        action="store_true",
        help="Reset without an interactive confirmation.",
    )
    run_reset.add_argument(
        "--archive",
        action="store_true",
        help="Retain the discarded run record and SQLite state in private trash.",
    )
    run_reset.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reset stale running metadata after confirming the owning process "
            "has stopped."
        ),
    )
    _add_owned_execution_commands(run_sub, owner="run")

    show = sub.add_parser("show", help="render a workflow as a code-first semantic view")
    show.add_argument("workflow", nargs="?", help="Workflow spec: module:workflow or path.py:workflow. Defaults to this project's workflow.")
    show.add_argument("--detail", choices=DETAILS, default="protocol", help="Amount of implementation detail to include.")
    focus = show.add_mutually_exclusive_group()
    focus.add_argument("--communications", action="store_true", help="Show communication and control flow only.")
    focus.add_argument("--agent", help="Show the exact local projection for one agent.")
    focus.add_argument("--agents", help="Comma-separated agents to retain in a boundary-aware focus view.")
    show.add_argument("--format", choices=("code", "json"), default="code", help="Output format.")

    validate = sub.add_parser("validate", help="validate loading, projection, rendering, and deployment metadata")
    validate.add_argument("workflow", nargs="?", help="Workflow spec: module:workflow or path.py:workflow. Defaults to this project's workflow.")
    validate.add_argument("--json", action="store_true", help="Print machine-readable validation results.")

    init_parser = sub.add_parser(
        "init",
        help="create a ZipperGen project in this directory",
    )
    init_parser.add_argument(
        "name",
        nargs="?",
        help="Project name; defaults to the directory name.",
    )
    init_parser.add_argument(
        "--directory",
        help="Create the project here instead of the current directory.",
    )

    skill_parser = sub.add_parser(
        "skill",
        help="print the coding-agent skill shipped with this package",
    )
    skill_parser.add_argument(
        "--agents-md",
        action="store_true",
        help="Print an AGENTS.md that points a coding agent at the skill.",
    )
    skill_parser.add_argument(
        "--project",
        help="Project name for --agents-md; defaults to the directory name.",
    )
    skill_parser.add_argument(
        "--no-references",
        action="store_true",
        help="Print SKILL.md alone, without the reference files it links.",
    )

    snapshot_parser = sub.add_parser(
        "snapshot",
        help="write a semantic baseline for later comparison",
    )
    snapshot_parser.add_argument(
        "path",
        help="Output JSON path, or '-' for standard output.",
    )
    snapshot_parser.add_argument(
        "workflow",
        nargs="?",
        help="Workflow spec; defaults to this project's workflow.",
    )

    semantic_diff_parser = sub.add_parser(
        "diff",
        help="compare a workflow against a semantic baseline",
        description=(
            "Create a baseline with 'zg snapshot PATH', then run "
            "'zg diff PATH' after editing to see exactly what changed in "
            "the protocol."
        ),
    )
    semantic_diff_parser.add_argument(
        "before",
        nargs="?",
        help="Original workflow spec or saved baseline JSON.",
    )
    semantic_diff_parser.add_argument(
        "after",
        nargs="?",
        help=(
            "Modified workflow spec or semantic snapshot JSON. Defaults to "
            "this project's workflow, so a saved baseline can be compared "
            "against the working tree with one argument."
        ),
    )
    semantic_diff_parser.add_argument("--format", choices=("code", "json"), default="code", help="Output format.")
    # One family, the same shape as model/assistant/connector: run it bare to
    # see everything, then one verb per thing you can do to a deployment.
    deploy = sub.add_parser(
        "deploy",
        allow_abbrev=False,
        help="deploy this project, and manage what is deployed",
        description=(
            "Run bare to configure, validate and start this project's "
            "deployment. The verbs act on that same deployment, so you do not "
            "normally type its name."
        ),
    )
    _add_guided_deployment_arguments(deploy)
    deploy.set_defaults(no_bundle=False)
    deploy.add_argument("--no-start", action="store_true", help="Configure the deployment without starting its service.")
    deploy.add_argument("--history-keep", type=_nonnegative_int_argument, default=None, metavar="ROWS", help="How many trace rows this deployment's store keeps. 0 records none. Default 10000.")
    deploy_sub = deploy.add_subparsers(
        dest="deploy_action", parser_class=_NoAbbrevArgumentParser
    )

    deploy_list = deploy_sub.add_parser(
        "list",
        help="show every deployment on this computer, including orphans",
    )
    deploy_list.add_argument("--json", action="store_true", help="Print JSON.")

    deploy_prune = deploy_sub.add_parser(
        "prune",
        help=(
            "remove deployments whose owning project is gone, and delete "
            "stale archives"
        ),
    )
    deploy_prune.add_argument(
        "--yes",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Do not ask for confirmation.",
    )
    deploy_prune.add_argument(
        "--keep-days",
        type=_nonnegative_float_argument,
        default=30.0,
        help=(
            "Keep archives newer than this many days, so a mistaken removal "
            "can still be undone. Default 30."
        ),
    )

    deploy_start = deploy_sub.add_parser("start", help="start a deployment as a supervised user service")
    deploy_start.add_argument("--enable", action="store_true", help="Enable the service to start automatically for this user.")
    deploy_start.add_argument("--dry-run", action="store_true", help="Print service-manager commands without running them.")

    deploy_stop = deploy_sub.add_parser("stop", help="stop a supervised deployment")
    deploy_stop.add_argument("--dry-run", action="store_true", help="Print the service-manager command without running it.")


    deploy_remove = deploy_sub.add_parser(
        "remove",
        help=(
            "unregister the service and take the deployment out of use; its "
            "profile, store and log move to trash unless purged"
        ),
    )
    deploy_remove.add_argument("--purge", action="store_true", help="Delete everything permanently, including the profile, store and log. Nothing is archived.")
    deploy_remove.add_argument("--yes", action="store_true", default=argparse.SUPPRESS, help="Do not ask for confirmation.")

    deploy_compact = deploy_sub.add_parser(
        "compact",
        help="drop optional history and rotate logs",
    )
    deploy_compact.add_argument("--keep-archives", type=_nonnegative_int_argument, default=3, help="How many rotated log archives to retain. Default 3.")
    deploy_compact.add_argument("--set-history-keep", type=_nonnegative_int_argument, default=None, metavar="ROWS", help="Change how many history rows this deployment keeps from now on, and apply it. 0 turns the trace off.")

    deploy_logs = deploy_sub.add_parser("logs", help="show logs for a deployment")
    deploy_logs.add_argument("--tail", type=_positive_int_argument, default=80, help="Number of log lines to show.")
    deploy_logs.add_argument("--follow", action="store_true", help="Keep watching the log file.")
    deploy_logs.add_argument("--interval", type=_positive_float_argument, help="Polling interval in seconds for --follow. Default 1.")

    deploy_check = deploy_sub.add_parser("check", help="check a deployment for common problems")
    deploy_check.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    deploy_check.add_argument(
        "--repair-permissions",
        action="store_true",
        help="Make managed deployment directories and files owner-only.",
    )
    deploy_check.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a check fails, for scripts.",
    )
    deploy_check.add_argument(
        "--no-service-check",
        "--no-systemd",
        dest="no_systemd",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    deploy_status = deploy_sub.add_parser("status", help="show durable store status for a deployment")
    deploy_status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    deploy_reset = deploy_sub.add_parser(
        "reset",
        help="archive durable state and start this deployment fresh",
    )
    deploy_reset.add_argument(
        "--yes",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Reset without asking for confirmation.",
    )
    _add_owned_execution_commands(deploy_sub, owner="deploy")
    internal_run = sub.add_parser("__run-deployment")
    internal_run.add_argument("--profile", required=True)
    internal_launch = sub.add_parser("__launch-deployment")
    internal_launch.add_argument("--profile", required=True)

    nt = sub.add_parser("notify", description="Notification adapter a deployment runs for itself.")
    notify_sub = nt.add_subparsers(dest="adapter", required=True)
    out = notify_sub.add_parser("stdout", help="print pending human tasks with approval tokens")
    out.add_argument("--store", required=True, help="SQLite store path.")
    out.add_argument("--channel", default="stdout", help="Approval token channel name.")
    out.add_argument("--watch", action="store_true", help="Keep polling for new pending tasks.")
    out.add_argument("--interval", type=_positive_float_argument, default=2.0, help="Polling interval in seconds for --watch.")
    out.add_argument("--limit", type=_positive_int_argument, help="Maximum number of tasks to notify per poll.")
    out.add_argument("--quiet", action="store_true", help="Suppress the no-pending-tasks message in one-shot mode.")

    tg = notify_sub.add_parser("telegram", help="send and receive human task approvals through Telegram")
    tg.add_argument("--store", required=True, help="SQLite store path.")
    tg.add_argument("--chat-id", help="Telegram chat id. Defaults to ZIPPERGEN_TELEGRAM_CHAT_ID.")
    tg.add_argument(
        "--allowed-user-id",
        help="Telegram user id allowed to answer. Defaults to ZIPPERGEN_TELEGRAM_ALLOWED_USER_ID or the chat id.",
    )
    tg.add_argument("--channel", default="telegram", help="Approval token channel name.")
    tg.add_argument("--watch", action="store_true", help="Keep polling Telegram and the local store.")
    tg.add_argument("--interval", type=_positive_float_argument, default=2.0, help="Delay between store scans in --watch mode.")
    tg.add_argument("--poll-timeout", type=_positive_float_argument, default=20.0, help="Telegram long-poll timeout in seconds.")
    tg.add_argument("--limit", type=_positive_int_argument, help="Maximum number of tasks to notify per poll.")
    tg.add_argument("--resend", action="store_true", help="Resend already-notified pending tasks.")
    tg.add_argument("--quiet", action="store_true", help="Suppress progress messages.")

    # Derive the visible command list from what is actually registered. A
    # hand-written list drifts the moment a command is added or renamed, and
    # this one had drifted.
    sub.metavar = "{" + ",".join(
        name for name in sub.choices if name not in HIDDEN_COMMANDS
    ) + "}"

    from zippergen.cli_help import render_command_tree

    ap.epilog = render_command_tree(ap, hidden=HIDDEN_COMMANDS)

    args = ap.parse_args(argv)
    return ap, args


def main(argv=None) -> int:
    ap, args = _parse_cli_args(argv)

    if args.cmd is None:
        ap.print_help()
        return 0
    project_scoped = args.cmd in {
        "config",
        "check",
        "workflow",
        "provider",
        "model",
        "assistant",
        "run",
        "deploy",
    }
    if args.cmd == "deploy" and getattr(args, "deploy_action", None) in {
        "list",
        "prune",
    }:
        project_scoped = False
    if args.cmd == "provider" and args.provider_action == "authorize":
        project_scoped = False
    if args.cmd == "connector":
        project_scoped = True
    if args.cmd in {"show", "validate"} and not getattr(args, "workflow", None):
        project_scoped = True
    if project_scoped:
        _require_project(args)
    if args.cmd == "config":
        return _configuration_command(args)
    if args.cmd == "check":
        return _check_command(args)
    if args.cmd == "workflow":
        return _workflow_command(args)
    if args.cmd == "provider" and args.provider_action == "authorize":
        return _provider_authorize_google_command(args)
    for family in _RENAMEABLE:
        if args.cmd == family and getattr(args, f"{family}_action", None) == "rename":
            return _rename_command(args, family)
    if args.cmd == "provider" and args.provider_action == "accept":
        return _provider_accept_google_command(args)
    if args.cmd == "provider":
        return _provider_command(args)
    if args.cmd == "model":
        return _model_command(args)
    if args.cmd == "assistant":
        return _assistant_command(args)
    if args.cmd == "completion":
        from zippergen.completion import render_completion

        print(render_completion(_guided_required_value(
            args.shell,
            label="Shell",
            command="zg completion SHELL",
            choices=("zsh", "bash", "fish"),
        )))
        return 0
    if args.cmd == "__complete":
        from zippergen.completion import completion_candidates

        print(
            "\n".join(
                completion_candidates(args.kind, args.project, tuple(args.path))
            )
        )
        return 0
    if args.cmd == "run":
        action = getattr(args, "run_action", None)
        if action == "status":
            return _run_status_command(args)
        if action == "reset":
            return _run_reset_command(args)
        if action == "inspect":
            return _inspect_command(args)
        if action == "trace":
            return _trace_command(args)
        if action == "tasks":
            return _tasks_command(args)
        if action == "approve":
            return _approve_command(args)
        if getattr(args, "durable", False) or getattr(args, "resume", False):
            return _durable_run_command(args)
        if getattr(args, "history_keep", None) is not None:
            raise SystemExit(
                "--history-keep requires --durable or --resume. A plain run "
                "keeps no store to record a trace in."
            )
        return _run_workflow_command(args)
    if args.cmd == "show":
        return _show_command(args)
    if args.cmd == "validate":
        return _validate_command(args)
    if args.cmd == "connector" and args.connector_action == "configure":
        return _connector_configure_command(args)
    if args.cmd == "connector" and args.connector_action == "assign":
        return _connector_assign_command(args)
    if args.cmd == "connector" and args.connector_action in {
        None,
        "unassign",
        "check",
        "remove",
    }:
        return _connector_management_command(args)
    if args.cmd == "init":
        return _init_command(args)
    if args.cmd == "skill":
        return _skill_command(args)
    if args.cmd == "snapshot":
        return _snapshot_command(args)
    if args.cmd == "diff":
        return _diff_command(args)
    if args.cmd == "__run-deployment":
        profile = _load_deployment_profile(args.profile)
        return _run_deployment_command(
            argparse.Namespace(name=str(profile["name"]))
        )
    if args.cmd == "__launch-deployment":
        return _launch_deployment_command(args)
    if args.cmd == "deploy":
        action = getattr(args, "deploy_action", None)
        if action is None:
            return _deploy_command(args)
        _reject_deploy_configuration_options(args, action)
        if action == "list":
            enforce_deploy_requirement(action, "")
            return _deployment_list_command(args)
        if action == "prune":
            enforce_deploy_requirement(action, "")
            return _deployment_prune_command(args)
        # Every public verb acts on this project's deployment. The generated
        # service script alone supplies the hidden profile identity because it
        # runs from an immutable bundle rather than the source project.
        resolved = _resolved_deployment_name(args)
        args.name = resolved
        # `status` reads `deployment`; it no longer takes one as an
        # argument, so the attribute has to be created, not updated.
        args.deployment = resolved
        if action not in {"remove", "compact", "reset"}:
            try:
                enforce_deploy_requirement(action, resolved)
            except ServiceIsLiveError as exc:
                raise SystemExit(str(exc)) from exc
        if action in {"start", "stop"}:
            return _deployment_lifecycle_command(args, action)
        if action == "remove":
            return _remove_command(args)
        if action == "compact":
            return _compact_command(args)
        if action == "reset":
            return _reset_deployment_command(args)
        if action == "logs":
            return _logs_command(args)
        if action == "check":
            return _doctor_command(args)
        if action == "status":
            return _status_command(args)
        if action == "inspect":
            return _inspect_command(args)
        if action == "trace":
            return _trace_command(args)
        if action == "tasks":
            return _tasks_command(args)
        if action == "approve":
            return _approve_command(args)
    if args.cmd == "notify" and args.adapter == "stdout":
        return _notify_stdout_command(args)
    if args.cmd == "notify" and args.adapter == "telegram":
        return _notify_telegram_command(args)

    ap.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
