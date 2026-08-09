"""Loading workflows and invoking their optional setup hook."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from zippergen.syntax import Lifeline, Workflow


@dataclass(frozen=True)
class RunConfig:
    """Configuration passed to an optional module-level ``zippergen_setup`` hook."""

    workflow_spec: str
    workflow: Workflow
    module: ModuleType
    llm: str | None
    llms: dict[str, str]
    assistant: str | None
    llm_idle_timeout: float | None
    llm_idle_timeouts: dict[str, float]
    store_path: str | None
    inputs: dict[str, object]
    options: dict[str, object]
    timeout: float
    execution: str

    def option(self, name: str, default: object = None) -> object:
        return self.options.get(name, default)


def _import_module_path(module_path: str) -> ModuleType:
    path = Path(module_path).expanduser().resolve()
    package_parts: list[str] = []
    package_root = path.parent
    while (package_root / "__init__.py").is_file():
        package_parts.insert(0, package_root.name)
        package_root = package_root.parent
    if package_parts:
        module_name = ".".join(
            [
                *package_parts,
                *([] if path.name == "__init__.py" else [path.stem]),
            ]
        )
        import_root = package_root
    else:
        module_name = (
            f"_zippergen_wf_"
            f"{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
        )
        import_root = path.parent
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    modules_before = set(sys.modules)
    sys.path.insert(0, str(import_root))
    try:
        spec.loader.exec_module(module)
    finally:
        for imported_name in set(sys.modules) - modules_before:
            imported = sys.modules.get(imported_name)
            imported_file = getattr(imported, "__file__", None)
            if not imported_file:
                continue
            try:
                local = Path(imported_file).resolve().is_relative_to(
                    import_root
                )
            except OSError:
                local = False
            if local:
                sys.modules.pop(imported_name, None)
        try:
            sys.path.remove(str(import_root))
        except ValueError:
            pass
    return module


def _looks_like_path(name: str) -> bool:
    return name.endswith(".py") or "/" in name or "\\" in name or Path(name).exists()


def _import_workflow_module(module_ref: str) -> ModuleType:
    if _looks_like_path(module_ref):
        return _import_module_path(module_ref)
    return importlib.import_module(module_ref)


def load_workflow_spec(spec_text: str) -> tuple[Workflow, ModuleType]:
    """Load ``module:workflow`` or ``path.py:workflow``.

    If no workflow name is supplied, the module must define exactly one
    ``Workflow`` object.
    """

    module_ref, sep, workflow_name = spec_text.partition(":")
    if not module_ref:
        raise SystemExit("Workflow spec must be MODULE:WORKFLOW or PATH.py:WORKFLOW.")
    module = _import_workflow_module(module_ref)
    if sep:
        try:
            value = getattr(module, workflow_name)
        except AttributeError as exc:
            raise SystemExit(f"Workflow {workflow_name!r} not found in {module_ref!r}.") from exc
        if not isinstance(value, Workflow):
            raise SystemExit(f"{module_ref}:{workflow_name} is not a ZipperGen Workflow.")
        return value, module

    workflows = {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, Workflow)
    }
    if len(workflows) == 1:
        return next(iter(workflows.values())), module
    if not workflows:
        raise SystemExit(f"No ZipperGen Workflow found in {module_ref!r}.")
    names = ", ".join(sorted(workflows))
    raise SystemExit(f"Multiple workflows found in {module_ref!r}: {names}. Use MODULE:WORKFLOW.")


def load_workflow(module_path: str, role_name: str) -> tuple[Workflow, Lifeline]:
    module = _import_module_path(module_path)
    workflows = [value for value in vars(module).values() if isinstance(value, Workflow)]
    if not workflows:
        raise SystemExit(f"No ZipperGen Workflow found in {module_path!r}.")
    wf = workflows[0]
    lifelines = {ll.name: ll for ll in _workflow_lifelines(wf)}
    if role_name not in lifelines:
        raise SystemExit(f"role {role_name!r} not in workflow lifelines {sorted(lifelines)}")
    return wf, lifelines[role_name]


def _workflow_lifelines(wf: Workflow) -> tuple[Lifeline, ...]:
    from zippergen.syntax import _ordered_workflow_lifelines
    return _ordered_workflow_lifelines(wf)


def _call_setup_hook(module: ModuleType, config: RunConfig) -> None:
    setup = getattr(module, "zippergen_setup", None)
    if setup is None:
        return
    if not callable(setup):
        raise SystemExit("zippergen_setup exists but is not callable.")
    setup(config)
