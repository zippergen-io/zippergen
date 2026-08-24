"""Project-aware durable runs that can be resumed after interruption."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from zippergen.assistant_configuration import (
    apply_assistant_overrides,
    normalize_assistant_overrides,
    project_assistant_routing,
)
from zippergen.deployment import DeploymentField, deployment_spec_from_module
from zippergen.human_backends import (
    make_cli_human_backend,
    make_sqlite_human_backend,
)
from zippergen.activity import ActivityIndicator
from zippergen.models import (
    ModelSettings,
    apply_model_overrides,
    effective_llm_routes,
    normalize_llm_overrides,
    fake_model_notice,
    project_model_routing,
    selected_llm_specs,
)
from zippergen.rendering import StatusKind, TerminalRenderer
from zippergen.semantic import semantic_snapshot, workflow_semantics
from zippergen.syntax import Json, Workflow, validate_zvalue
from zippergen.workspace import Workspace, WorkspaceError
from zippergen.workflow_io import (
    RunConfig,
    _call_setup_hook,
    load_workflow_spec,
    project_directory,
)

InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], object]


class RunResetError(RuntimeError):
    """A durable run could not be reset without risking its state."""


def _sqlite_family(path: Path) -> tuple[Path, ...]:
    return (
        path,
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
        Path(str(path) + "-journal"),
    )


def reset_current_run(
    workspace: Workspace,
    *,
    archive: bool = False,
    force: bool = False,
) -> tuple[dict[str, Any], Path | None, int]:
    """Discard the selected durable run, optionally retaining an archive.

    No run remains selected afterwards. ``force`` is only for stale metadata
    after the owning process is known to have stopped; moving a live SQLite
    family is unsafe.
    """

    record = workspace.current_run()
    if record is None:
        raise RunResetError("There is no current durable run to reset.")
    status = str(record.get("status") or "")
    if status in {"running", "waiting"} and not force:
        raise RunResetError(
            f"Run {record['run_id']} is recorded as {status}. Stop its "
            "foreground process with Ctrl-C first. If that process is "
            "already gone, re-run with --force."
        )

    store = Path(str(record["store"])).expanduser()
    store_files = tuple(
        path
        for path in _sqlite_family(store)
        if path.exists() or path.is_symlink()
    )
    if any(path.is_dir() or path.is_symlink() for path in store_files):
        raise RunResetError(
            f"Expected regular SQLite files for run {record['run_id']}: {store}"
        )
    record_path = workspace.run_path(str(record["run_id"]))
    if not record_path.is_file() or record_path.is_symlink():
        raise RunResetError(f"Expected a regular run record: {record_path}")

    root = workspace.home / "trash" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = "" if archive else ".discarding-"
    base = f"{prefix}{record['run_id']}-{timestamp}"
    destination = root / base
    suffix = 2
    while destination.exists():
        destination = root / f"{base}-{suffix}"
        suffix += 1
    destination.mkdir(mode=0o700)

    moved: list[tuple[Path, Path]] = []
    try:
        for source in (record_path, *store_files):
            target = destination / source.name
            shutil.move(str(source), str(target))
            target.chmod(0o600)
            moved.append((source, target))
        if archive:
            metadata = destination / "reset.json"
            metadata.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run": record["run_id"],
                        "previous_status": status,
                        "store": str(store),
                        "reset_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "files": [target.name for _source, target in moved],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            metadata.chmod(0o600)
        workspace.update(current_run=None)
    except Exception as exc:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
        shutil.rmtree(destination, ignore_errors=True)
        raise RunResetError(
            f"Could not reset run {record['run_id']} safely: {exc}"
        ) from exc

    if not archive:
        shutil.rmtree(destination)
    return record, destination if archive else None, len(store_files)


def semantic_fingerprint(workflow: Workflow, module: ModuleType) -> str:
    """Hash the stable semantic model used to guard durable resume."""

    payload = json.dumps(
        semantic_snapshot(workflow, module),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def active_llm_routes(
    workflow: Workflow,
    module: ModuleType,
    default_spec: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the effective model route for every LLM action site."""

    routes = effective_llm_routes(workflow, default_spec, overrides)
    active: list[tuple[str, str]] = []
    sites = workflow_semantics(workflow, module).get("action_sites") or []
    if isinstance(sites, list):
        for site in sites:
            if not isinstance(site, dict) or site.get("kind") != "llm":
                continue
            participant = str(site.get("lifeline"))
            action = str(site.get("action"))
            target = f"{participant}.{action}"
            item = (participant, target)
            if participant in routes and item not in active:
                active.append(item)
    return {
        target: routes.get(target, routes[participant])
        for participant, target in active
    }


def _deployment_input_fields(module: ModuleType) -> dict[str, DeploymentField]:
    spec = deployment_spec_from_module(module)
    return {
        field.target_name: field
        for field in spec.fields
        if field.target == "input"
    }


def default_llm_spec(module: ModuleType) -> str:
    return "mock"


def _field_enabled(field: DeploymentField, values: dict[str, object]) -> bool:
    if not field.when:
        return True
    candidates = [values.get(field.when)]
    llm_field_names = values.get("__llm_field_names__")
    if (
        field.when == "llm"
        or (
            isinstance(llm_field_names, (list, tuple, set))
            and field.when in llm_field_names
        )
    ):
        configured = values.get("__llm_specs__")
        if isinstance(configured, (list, tuple, set)):
            candidates.extend(configured)
    if not field.when_values:
        return any(bool(current) for current in candidates)
    return any(
        str(current).startswith(expected[:-1])
        if expected.endswith("*")
        else str(current) == expected
        for current in candidates
        for expected in field.when_values
    )


def collect_development_environment(
    module: ModuleType,
    workspace: Workspace,
    *,
    llm: str,
    llms: dict[str, str] | None,
    inputs: dict[str, object],
    options: dict[str, object],
    interactive: bool,
    input_func: InputFunc,
    secret_input_func: InputFunc,
    output_func: OutputFunc,
    renderer: TerminalRenderer | None = None,
) -> dict[str, str]:
    """Resolve declared environment fields, privately persisting secrets."""

    spec = deployment_spec_from_module(module)
    saved_secrets = workspace.load_secrets()
    values: dict[str, object] = {}
    for field in spec.fields:
        if field.target == "input":
            values[field.name] = inputs.get(field.target_name, field.default)
        elif field.target == "option":
            values[field.name] = options.get(field.target_name, field.default)
        elif field.target == "env":
            values[field.name] = (
                os.environ.get(field.target_name)
                or saved_secrets.get(field.target_name)
                or field.default
            )
    values["__llm_field_names__"] = ()
    values["__llm_specs__"] = selected_llm_specs(llm, llms)

    environment: dict[str, str] = {}
    secrets_changed = False
    for field in spec.fields:
        if field.target != "env" or not _field_enabled(field, values):
            continue
        value = values.get(field.name)
        if value is None or str(value).strip() == "":
            if interactive:
                choices = f" ({'/'.join(field.choices)})" if field.choices else ""
                prompt = f"{field.prompt}{choices}: "
                raw = (
                    secret_input_func(prompt)
                    if field.secret
                    else input_func(prompt)
                )
                value = raw.strip()
        if field.required and (value is None or str(value).strip() == ""):
            raise SystemExit(
                f"Development field {field.name!r} is required. Run in an "
                f"interactive terminal or set {field.target_name}."
            )
        if value is None or str(value) == "":
            continue
        if field.choices and str(value) not in field.choices:
            raise SystemExit(
                f"Development field {field.name!r} must be one of "
                f"{', '.join(field.choices)}; got {value!r}."
            )
        environment[field.target_name] = str(value)
        if (
            field.secret
            and field.target_name not in os.environ
            and saved_secrets.get(field.target_name) != str(value)
        ):
            saved_secrets[field.target_name] = str(value)
            secrets_changed = True
            message = (
                f"Saved {field.target_name} in private development secret storage."
            )
            if renderer is None:
                output_func(message)
            else:
                renderer.status("success", message)
    if secrets_changed:
        workspace.save_secrets(saved_secrets)
    return environment


@contextmanager
def _temporary_environment(values: dict[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _parse_guided_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _coerce_input(name: str, value: object, expected: type) -> object:
    if expected is Json:
        try:
            return validate_zvalue(
                value,
                Json,
                context=f"Input {name!r}",
            )
        except TypeError as exc:
            raise SystemExit(str(exc)) from exc
    if expected is str:
        return value if isinstance(value, str) else str(value)
    if expected is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
        raise SystemExit(f"Input {name!r} must be true or false; got {value!r}.")
    if expected is int:
        if isinstance(value, bool):
            raise SystemExit(f"Input {name!r} must be an integer; got {value!r}.")
        try:
            return int(str(value))
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Input {name!r} must be an integer; got {value!r}."
            ) from exc
    if expected is float:
        if isinstance(value, bool):
            raise SystemExit(f"Input {name!r} must be a number; got {value!r}.")
        try:
            result = float(str(value))
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Input {name!r} must be a number; got {value!r}."
            ) from exc
        try:
            return validate_zvalue(result, float, context=f"Input {name!r}")
        except TypeError as exc:
            raise SystemExit(str(exc)) from exc
    if expected is tuple:
        if type(value) is list:
            value = tuple(value)
        try:
            return validate_zvalue(value, tuple, context=f"Input {name!r}")
        except TypeError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        return validate_zvalue(value, expected, context=f"Input {name!r}")
    except TypeError as exc:
        raise SystemExit(str(exc)) from exc


def collect_workflow_inputs(
    workflow: Workflow,
    module: ModuleType,
    provided: dict[str, object] | None = None,
    *,
    interactive: bool,
    input_func: InputFunc = input,
    output_func: OutputFunc = print,
    workspace: "Workspace | None" = None,
) -> dict[str, object]:
    """Collect typed workflow inputs from overrides, the project, and defaults.

    An answer stored in the project's configuration is the default here, and a
    new answer is written back to it. A run and a deployment therefore read the
    same values from the same place: an answer given once does not have to be
    given again, in either command.
    """

    supplied = dict(provided or {})
    expected_names = {name for name, _value_type, _lifeline in workflow.inputs}
    unknown = sorted(set(supplied) - expected_names)
    if unknown:
        raise SystemExit(
            "Unknown workflow input(s): " + ", ".join(unknown) + "."
        )

    fields = _deployment_input_fields(module)
    stored = workspace.configuration_values() if workspace is not None else {}
    collected: dict[str, object] = {}
    missing = [
        name
        for name, _value_type, _lifeline in workflow.inputs
        if name not in supplied
    ]
    if interactive and missing:
        output_func("Workflow inputs")
        output_func("═══════════════")
    for name, value_type, _lifeline in workflow.inputs:
        field = fields.get(name)
        # The project's answer outranks the declared default: a declared
        # default is what to do when nobody has answered, and somebody has.
        answered = field is not None and field.name in stored
        default = (
            stored[field.name]
            if answered and field is not None
            else (
                field.default
                if field is not None and field.default is not None
                else None
            )
        )
        has_default = default is not None
        if name in supplied:
            value = supplied[name]
        elif interactive:
            label = field.prompt if field is not None else name.replace("_", " ").title()
            type_name = getattr(value_type, "__name__", str(value_type))
            default_text = (
                f" [{json.dumps(default, default=str)}]" if has_default else ""
            )
            raw = input_func(f"{label} ({type_name}){default_text}: ")
            if not raw.strip() and has_default:
                value = default
            else:
                value = _parse_guided_value(raw)
        elif has_default:
            value = default
        else:
            raise SystemExit(
                f"Workflow input {name!r} is required. Use --input {name}=VALUE "
                "or run in an interactive terminal."
            )
        collected[name] = _coerce_input(name, value, value_type)
    _record_inputs_in_project(fields, collected, workspace)
    return collected


def _record_inputs_in_project(
    fields: dict[str, DeploymentField],
    collected: Mapping[str, object],
    workspace: "Workspace | None",
) -> None:
    """Keep a run's answers where a deployment would have kept them."""

    if workspace is None or not fields:
        return
    answers = dict(workspace.configuration_values())
    for name, field in fields.items():
        if field.secret or name not in collected:
            continue
        answers[field.name] = collected[name]
    if answers != workspace.configuration_values():
        workspace.write_configuration_values(answers)


def _recorded_model_settings(
    record: Mapping[str, object],
) -> dict[str, "ModelSettings"]:
    """Read the unified model settings from a run record."""

    from zippergen.models import model_settings_from_mapping

    stored = record.get("llm_settings")
    if isinstance(stored, Mapping):
        return {
            str(target): model_settings_from_mapping(value, subject=str(target))
            for target, value in stored.items()
        }
    return {}


def _load_and_validate(workspace: Workspace, stored_spec: str):
    from zippergen.validation import validate_workflow

    workflow, module = load_workflow_spec(workspace.absolute_spec(stored_spec))
    validation = validate_workflow(workflow, module)
    if not validation["valid"]:
        failures = [
            str(check["detail"])
            for check in validation["checks"]  # type: ignore[index]
            if check["status"] == "fail"
        ]
        raise SystemExit(
            f"Workflow {workflow.name!r} is invalid: " + "; ".join(failures)
        )
    return workflow, module


def _run_setup_hook(
    *,
    workflow_spec: str,
    workflow: Workflow,
    module: ModuleType,
    llm: str,
    llms: dict[str, str],
    llm_idle_timeout: float | None,
    llm_settings: Mapping[str, "ModelSettings"],
    assistant: str | None,
    assistants: dict[str, str],
    store_path: str,
    inputs: dict[str, object],
    options: dict[str, object],
    timeout: float,
) -> None:
    _call_setup_hook(
        module,
        RunConfig(
            workflow_spec=workflow_spec,
            workflow=workflow,
            module=module,
            llm=llm,
            llms=llms,
            assistant=assistant,
            assistants=assistants,
            llm_idle_timeout=llm_idle_timeout,
            llm_settings=llm_settings,
            store_path=store_path,
            inputs=inputs,
            options=dict(options),
            timeout=timeout,
            execution="sqlite",
        ),
    )


def run_durable(
    workspace: Workspace,
    *,
    workflow_spec: str | None = None,
    resume: bool = False,
    history_keep: int | None = None,
    provided_inputs: dict[str, object] | None = None,
    llm: str | None = None,
    llms: dict[str, str] | None = None,
    llm_idle_timeout: float | None = None,
    llm_settings: Mapping[str, "ModelSettings"] | None = None,
    assistant: str | None = None,
    assistants: dict[str, str] | None = None,
    options: dict[str, object] | None = None,
    timeout: float = 0.0,
    interactive: bool = True,
    input_func: InputFunc = input,
    secret_input_func: InputFunc | None = None,
    output_func: OutputFunc = print,
    renderer: TerminalRenderer | None = None,
    human_connector_factory: Callable[[str], object] | None = None,
    connector_environment: dict[str, str] | None = None,
    connector_snapshot: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Create or resume one recorded durable run."""

    with project_directory(workspace.root):
        return _run_durable_in_project(
            workspace,
            workflow_spec=workflow_spec,
            resume=resume,
            history_keep=history_keep,
            provided_inputs=provided_inputs,
            llm=llm,
            llms=llms,
            llm_idle_timeout=llm_idle_timeout,
            llm_settings=llm_settings,
            assistant=assistant,
            assistants=assistants,
            options=options,
            timeout=timeout,
            interactive=interactive,
            input_func=input_func,
            secret_input_func=secret_input_func,
            output_func=output_func,
            renderer=renderer,
            human_connector_factory=human_connector_factory,
            connector_environment=connector_environment,
            connector_snapshot=connector_snapshot,
        )


def _run_durable_in_project(
    workspace: Workspace,
    *,
    workflow_spec: str | None,
    resume: bool,
    history_keep: int | None = None,
    provided_inputs: dict[str, object] | None,
    llm: str | None,
    llms: dict[str, str] | None,
    llm_idle_timeout: float | None,
    llm_settings: Mapping[str, "ModelSettings"] | None,
    assistant: str | None,
    assistants: dict[str, str] | None,
    options: dict[str, object] | None,
    timeout: float,
    interactive: bool,
    input_func: InputFunc,
    secret_input_func: InputFunc | None,
    output_func: OutputFunc,
    renderer: TerminalRenderer | None,
    human_connector_factory: Callable[[str], object] | None,
    connector_environment: dict[str, str] | None,
    connector_snapshot: dict[str, object] | None,
) -> dict[str, Any]:
    """Implementation of :func:`run_durable` inside the project directory."""

    if resume and workflow_spec is not None:
        raise SystemExit("Do not pass a workflow when using --resume.")
    if resume and (
        provided_inputs
        or llm is not None
        or llms
        or llm_idle_timeout is not None
        or llm_settings
        or assistant is not None
        or assistants
        or options
        or connector_snapshot is not None
    ):
        raise SystemExit(
            "A resumed run uses its recorded workflow inputs and configuration."
        )

    if secret_input_func is None:
        import getpass

        secret_input_func = getpass.getpass

    if resume:
        selected_run_id = workspace.current_run_id
        if not selected_run_id:
            raise SystemExit("There is no current durable run to resume.")
        record = workspace.load_run(selected_run_id)
        if record.get("status") == "done":
            raise SystemExit(
                f"Run {selected_run_id} is already complete. Start a new run instead."
            )
        stored_spec = str(record["workflow_spec"])
        workflow, module = _load_and_validate(workspace, stored_spec)
        fingerprint = semantic_fingerprint(workflow, module)
        if fingerprint != record.get("fingerprint"):
            raise SystemExit(
                "The workflow meaning changed after this run began. Resume with the "
                "matching source or start a new run; the existing store was preserved."
            )
        inputs = dict(record.get("inputs") or {})
        selected_llm = str(record.get("llm") or "mock")
        selected_llms = normalize_llm_overrides(record.get("llms"))
        selected_idle_timeout = (
            float(record["llm_idle_timeout"])
            if record.get("llm_idle_timeout") is not None
            else None
        )
        selected_settings = _recorded_model_settings(record)
        selected_assistant = (
            str(record["assistant"]) if record.get("assistant") else None
        )
        selected_assistants = normalize_assistant_overrides(
            record.get("assistants")
        )
        run_options = dict(record.get("options") or {})
        if renderer is None:
            output_func(f"Resuming run {selected_run_id}")
        else:
            renderer.status("info", f"Resuming run {selected_run_id}.")
    else:
        try:
            selected = workspace.resolve_workflow(workflow_spec)
        except WorkspaceError as exc:
            raise SystemExit(str(exc)) from exc
        stored_spec = workspace.canonical_spec(selected, cwd=workspace.root)
        workflow, module = _load_and_validate(workspace, stored_spec)
        if renderer is None:
            output_func(f"Workflow {workflow.name}: valid")
        else:
            renderer.status(
                "success",
                f"Workflow {workflow.name} validated.",
            )
        inputs = collect_workflow_inputs(
            workflow,
            module,
            provided_inputs,
            interactive=interactive,
            input_func=input_func,
            output_func=output_func,
            workspace=workspace,
        )
        routing = project_model_routing(
            workspace,
            stored_spec,
            workflow,
            fallback_default=default_llm_spec(module),
        )
        routing = apply_model_overrides(
            routing,
            default_spec=llm,
            overrides=llms,
            settings=llm_settings,
        )
        selected_llm = routing.default_spec
        selected_llms = routing.overrides
        selected_idle_timeout = llm_idle_timeout
        selected_settings = routing.settings
        assistant_routing = project_assistant_routing(
            workspace,
            stored_spec,
            workflow,
            module=module,
        )
        assistant_routing = apply_assistant_overrides(
            assistant_routing,
            default_backend=assistant,
            overrides=assistants,
            workflow=workflow,
            module=module,
        )
        selected_assistant = assistant_routing.default_backend
        selected_assistants = assistant_routing.overrides
        notice = fake_model_notice(
            effective_llm_routes(workflow, selected_llm, selected_llms)
        )
        if notice:
            if renderer is None:
                output_func(notice)
            else:
                renderer.status("warning", notice)
        run_options = dict(options or {})
        previous = workspace.current_run()
        if previous is not None:
            try:
                reset_current_run(workspace)
            except RunResetError as exc:
                raise SystemExit(str(exc)) from exc
            message = f"Discarded previous durable run {previous['run_id']}."
            if renderer is None:
                output_func(message)
            else:
                renderer.status("info", message)
        record = workspace.new_run(
            workflow_spec=stored_spec,
            workflow_name=workflow.name,
            fingerprint=semantic_fingerprint(workflow, module),
            inputs=inputs,
            llm=selected_llm,
            llms=selected_llms,
            llm_idle_timeout=selected_idle_timeout,
            llm_settings=selected_settings,
            assistant=selected_assistant,
            assistants=selected_assistants,
            options=run_options,
            connectors=connector_snapshot,
        )
        selected_run_id = str(record["run_id"])
        if renderer is None:
            output_func(f"Run {selected_run_id}")

    environment = collect_development_environment(
        module,
        workspace,
        llm=selected_llm,
        llms=selected_llms,
        inputs=inputs,
        options=run_options,
        interactive=interactive,
        input_func=input_func,
        secret_input_func=secret_input_func,
        output_func=output_func,
        renderer=renderer,
    )
    provider_environment = workspace.development_provider_environment(
        selected_llm_specs(selected_llm, selected_llms)
    )
    provider_environment.update(environment)
    provider_environment.update(connector_environment or {})
    environment = provider_environment

    store_path = str(record["store"])
    if history_keep is not None:
        # A storage setting, not workflow configuration: it changes how much of
        # the trace is kept, never what the run computes. Safe on a resume too.
        # The store is recorded before anything opens it, so this may be the
        # thing that creates it.
        from zippergen.storage_maintenance import initialize_store_history_keep

        initialize_store_history_keep(store_path, history_keep)
    workspace.update(current_run=selected_run_id)
    workspace.update_run(selected_run_id, status="running", error=None)
    if renderer is not None:
        run_rows: list[tuple[str, object, StatusKind | None]] = [
            ("Status", "running", "success"),
            ("Workflow", stored_spec, None),
            ("Run", selected_run_id, None),
            ("Store", store_path, None),
        ]
        routes = active_llm_routes(
            workflow,
            module,
            selected_llm,
            selected_llms,
        )
        if routes:
            run_rows.extend(
                (f"Model · {participant}", spec, None)
                for participant, spec in routes.items()
            )
            for target, chosen in sorted(selected_settings.items()):
                stated = dict(chosen.as_dict())
                if "idle_timeout" in stated:
                    seconds = stated.pop("idle_timeout")
                    run_rows.append((
                        f"Idle release · {target}",
                        (
                            "after every call"
                            if seconds == 0
                            else f"after {seconds:g} seconds"
                        ),
                        None,
                    ))
                if stated:
                    run_rows.append((
                        f"Model settings · {target}",
                        ", ".join(
                            f"{name}={value:g}"
                            if isinstance(value, float)
                            else f"{name}={value}"
                            for name, value in sorted(stated.items())
                        ),
                        None,
                    ))
        else:
            run_rows.append(("Models", "none; no LLM actions", None))
        renderer.table(
            "Durable run",
            run_rows,
        )

    if human_connector_factory is not None:
        connector = human_connector_factory(store_path)
        threading.Thread(
            target=connector.run_forever,  # type: ignore[attr-defined]
            name="connector-human",
            daemon=True,
        ).start()
        if renderer is not None:
            renderer.status(
                "success",
                "External human connector started for this durable run.",
            )

    terminal_backend = make_cli_human_backend(
        input_func=input_func,
        output_func=output_func,
    )

    # A run that calls a model or an assistant is silent for minutes at a
    # time, and silence is indistinguishable from a hang. The trace already
    # reports every action's start and end, so the same events drive a line
    # saying what is running.
    activity = ActivityIndicator(
        sys.stderr, interactive=renderer is None and sys.stderr.isatty()
    )

    def report_activity(event: object) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind not in {"act_start", "act", "act_failed"}:
            return
        key = id(event.get("seq")) if event.get("seq") is None else int(event["seq"])
        label = f"{event.get('lifeline')} · {event.get('action')}"
        person = event.get("action_kind") == "human"
        if kind == "act_start":
            if person:
                activity.person_waiting(True)
            else:
                activity.started(key, label)
            return
        if person:
            activity.person_waiting(False)
        else:
            activity.finished(key)

    def managed_human_backend(action, action_inputs):
        workspace.update_run(selected_run_id, status="waiting")
        try:
            return terminal_backend(action, action_inputs)
        finally:
            workspace.update_run(selected_run_id, status="running")

    setattr(managed_human_backend, "claims_pending_human_tasks", True)
    setattr(managed_human_backend, "requires_main_thread", True)
    selected_human_backend = (
        make_sqlite_human_backend()
        if human_connector_factory is not None
        else managed_human_backend
    )

    try:
        with _temporary_environment(environment):
            _run_setup_hook(
                workflow_spec=stored_spec,
                workflow=workflow,
                module=module,
                llm=selected_llm,
                llms=selected_llms,
                llm_idle_timeout=selected_idle_timeout,
                llm_settings=selected_settings,
                assistant=selected_assistant,
                assistants=selected_assistants,
                store_path=store_path,
                inputs=inputs,
                options=run_options,
                timeout=timeout,
            )
            llm_config: str | dict[str, str] = selected_llm
            if selected_llms:
                llm_config = effective_llm_routes(
                    workflow,
                    selected_llm,
                    selected_llms,
                )
            from zippergen.assistant_backends import make_cli_assistant_backend

            workflow.configure(
                llm_config,
                llm_idle_timeout=selected_idle_timeout,
                llm_settings=selected_settings,
                execution="sqlite",
                store_path=store_path,
                timeout=timeout,
                mock_delay=(0.0, 0.0),
                trace=report_activity,
                human_backend=selected_human_backend,
                assistant_backend=(
                    make_cli_assistant_backend(
                        selected_assistant,
                        project_root=str(workspace.root),
                        routes=selected_assistants,
                    )
                    if selected_assistants
                    else make_cli_assistant_backend(
                        selected_assistant,
                        project_root=str(workspace.root),
                    )
                ),
            )
            result = workflow(**inputs)
    except KeyboardInterrupt:
        workspace.update_run(selected_run_id, status="interrupted", error="Interrupted")
        raise
    except BaseException as exc:
        workspace.update_run(
            selected_run_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        # However the run ends, the terminal gets its line back.
        activity.close()

    updated = workspace.update_run(
        selected_run_id,
        status="done",
        result=result,
        error=None,
    )
    if renderer is None:
        output_func(f"Result: {result}")
        output_func("Next: show another view, start a new run, or prepare deployment.")
    else:
        renderer.table(
            "Run completed",
            [
                ("Status", "done", "success"),
                ("Run", selected_run_id, None),
                ("Result", result, None),
                ("Store", store_path, None),
                ("Next", "workflow show · run · deploy", None),
            ],
        )
    return updated
