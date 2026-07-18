"""Guided, project-aware durable development runs."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from collections.abc import Callable
from types import ModuleType
from typing import Any

from zippergen.deployment import DeploymentField, deployment_spec_from_module
from zippergen.human_backends import make_cli_human_backend
from zippergen.semantic import semantic_snapshot
from zippergen.syntax import Workflow
from zippergen.workspace import Workspace, WorkspaceError


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], object]


def semantic_fingerprint(workflow: Workflow, module: ModuleType) -> str:
    """Hash the stable semantic model used to guard durable resume."""

    payload = json.dumps(
        semantic_snapshot(workflow, module),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _deployment_input_fields(module: ModuleType) -> dict[str, DeploymentField]:
    spec = deployment_spec_from_module(module)
    return {
        field.target_name: field
        for field in spec.fields
        if field.target == "input"
    }


def default_llm_spec(module: ModuleType) -> str:
    for field in deployment_spec_from_module(module).fields:
        if field.target == "llm" and field.default:
            return str(field.default)
    return "mock"


def _field_enabled(field: DeploymentField, values: dict[str, object]) -> bool:
    if not field.when:
        return True
    current = values.get(field.when)
    if not field.when_values:
        return bool(current)
    text = str(current)
    return any(
        text.startswith(expected[:-1]) if expected.endswith("*") else text == expected
        for expected in field.when_values
    )


def collect_development_environment(
    module: ModuleType,
    workspace: Workspace,
    *,
    llm: str,
    inputs: dict[str, object],
    options: dict[str, object],
    services: str | None,
    interactive: bool,
    input_func: InputFunc,
    secret_input_func: InputFunc,
    output_func: OutputFunc,
) -> dict[str, str]:
    """Resolve declared environment fields, privately persisting secrets."""

    spec = deployment_spec_from_module(module)
    saved_secrets = workspace.load_secrets()
    values: dict[str, object] = {}
    for field in spec.fields:
        if field.target == "llm":
            values[field.name] = llm
        elif field.target == "services":
            values[field.name] = services
        elif field.target == "input":
            values[field.name] = inputs.get(field.target_name, field.default)
        elif field.target == "option":
            values[field.name] = options.get(field.target_name, field.default)
        elif field.target == "env":
            values[field.name] = (
                os.environ.get(field.target_name)
                or saved_secrets.get(field.target_name)
                or field.default
            )

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
            output_func(
                f"Saved {field.target_name} in private development secret storage."
            )
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
            return float(str(value))
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Input {name!r} must be a number; got {value!r}."
            ) from exc
    return value


def collect_workflow_inputs(
    workflow: Workflow,
    module: ModuleType,
    provided: dict[str, object] | None = None,
    *,
    interactive: bool,
    input_func: InputFunc = input,
) -> dict[str, object]:
    """Collect typed workflow inputs from overrides and declared defaults."""

    supplied = dict(provided or {})
    expected_names = {name for name, _value_type, _lifeline in workflow.inputs}
    unknown = sorted(set(supplied) - expected_names)
    if unknown:
        raise SystemExit(
            "Unknown workflow input(s): " + ", ".join(unknown) + "."
        )

    fields = _deployment_input_fields(module)
    collected: dict[str, object] = {}
    for name, value_type, _lifeline in workflow.inputs:
        field = fields.get(name)
        has_default = field is not None and field.default is not None
        default = (
            field.default
            if field is not None and field.default is not None
            else None
        )
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
    return collected


def _load_and_validate(workspace: Workspace, stored_spec: str):
    # Imported lazily to avoid making the state layer depend on the CLI module.
    from zippergen.serve import _validate_workflow, load_workflow_spec

    workflow, module = load_workflow_spec(workspace.absolute_spec(stored_spec))
    validation = _validate_workflow(workflow, module)
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
    store_path: str,
    inputs: dict[str, object],
    options: dict[str, object],
    services: str | None,
    timeout: float,
) -> None:
    from zippergen.serve import RunConfig, _call_setup_hook

    setup_options = dict(options)
    if services is not None:
        setup_options["services"] = services
    _call_setup_hook(
        module,
        RunConfig(
            workflow_spec=workflow_spec,
            workflow=workflow,
            module=module,
            llm=llm,
            llm_idle_timeout=None,
            store_path=store_path,
            inputs=inputs,
            options=setup_options,
            ui=False,
            timeout=timeout,
            execution="sqlite",
            show_decisions=False,
        ),
    )


def run_dev(
    workspace: Workspace,
    *,
    workflow_spec: str | None = None,
    resume: bool = False,
    run_id: str | None = None,
    provided_inputs: dict[str, object] | None = None,
    llm: str | None = None,
    options: dict[str, object] | None = None,
    services: str | None = None,
    timeout: float = 0.0,
    interactive: bool = True,
    input_func: InputFunc = input,
    secret_input_func: InputFunc | None = None,
    output_func: OutputFunc = print,
) -> dict[str, Any]:
    """Create or resume one managed durable development run."""

    if resume and workflow_spec is not None:
        raise SystemExit("Do not pass a workflow when using --resume.")
    if resume and (provided_inputs or llm is not None or options or services is not None):
        raise SystemExit(
            "A resumed run uses its recorded workflow inputs and configuration."
        )

    if secret_input_func is None:
        import getpass

        secret_input_func = getpass.getpass

    if resume:
        selected_run_id = run_id or workspace.current_run_id
        if not selected_run_id:
            raise SystemExit("There is no current development run to resume.")
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
        run_options = dict(record.get("options") or {})
        run_services = record.get("services")
        output_func(f"Resuming run {selected_run_id}")
    else:
        selected = workflow_spec or workspace.current_workflow
        if not selected:
            raise SystemExit(
                "No workflow selected. Pass PATH.py:WORKFLOW or select one in Studio."
            )
        stored_spec = workspace.select_workflow(selected)
        workflow, module = _load_and_validate(workspace, stored_spec)
        output_func(f"Workflow {workflow.name}: valid")
        inputs = collect_workflow_inputs(
            workflow,
            module,
            provided_inputs,
            interactive=interactive,
            input_func=input_func,
        )
        selected_llm = llm or default_llm_spec(module)
        run_options = dict(options or {})
        run_services = services
        record = workspace.new_run(
            workflow_spec=stored_spec,
            workflow_name=workflow.name,
            fingerprint=semantic_fingerprint(workflow, module),
            inputs=inputs,
            llm=selected_llm,
            options=run_options,
            services=run_services,
        )
        selected_run_id = str(record["run_id"])
        output_func(f"Run {selected_run_id}")

    environment = collect_development_environment(
        module,
        workspace,
        llm=selected_llm,
        inputs=inputs,
        options=run_options,
        services=str(run_services) if run_services is not None else None,
        interactive=interactive,
        input_func=input_func,
        secret_input_func=secret_input_func,
        output_func=output_func,
    )

    store_path = str(record["store"])
    workspace.update(
        current_workflow=stored_spec,
        current_run=selected_run_id,
    )
    workspace.update_run(selected_run_id, status="running", error=None)

    terminal_backend = make_cli_human_backend(
        input_func=input_func,
        output_func=output_func,
    )

    def managed_human_backend(action, action_inputs):
        workspace.update_run(selected_run_id, status="waiting")
        try:
            return terminal_backend(action, action_inputs)
        finally:
            workspace.update_run(selected_run_id, status="running")

    setattr(managed_human_backend, "claims_pending_human_tasks", True)

    try:
        with _temporary_environment(environment):
            _run_setup_hook(
                workflow_spec=stored_spec,
                workflow=workflow,
                module=module,
                llm=selected_llm,
                store_path=store_path,
                inputs=inputs,
                options=run_options,
                services=str(run_services) if run_services is not None else None,
                timeout=timeout,
            )
            workflow.configure(
                selected_llm,
                execution="sqlite",
                store_path=store_path,
                timeout=timeout,
                ui=False,
                mock_delay=(0.0, 0.0),
                human_backend=managed_human_backend,
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

    updated = workspace.update_run(
        selected_run_id,
        status="done",
        result=result,
        error=None,
    )
    output_func(f"Result: {result}")
    output_func("Next: show another view, start a new run, or prepare deployment.")
    return updated
