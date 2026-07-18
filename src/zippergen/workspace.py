"""Project-aware state for the prompt-first ZipperGen development experience.

Workspace state is deliberately small, non-secret, and stored below
``ZIPPERGEN_HOME`` rather than in the user's Git checkout.  The regular CLI
remains stateless; ``zippergen studio`` and ``zippergen dev`` use this module to
remember a current workflow and managed development runs.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any


WORKSPACE_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1

_IGNORED_DISCOVERY_PARTS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".zippergen",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


class WorkspaceError(RuntimeError):
    """Workspace state is missing or malformed."""


def zippergen_home() -> Path:
    """Return the configured ZipperGen home without requiring an export."""

    return Path(
        os.environ.get("ZIPPERGEN_HOME", str(Path.home() / ".zippergen"))
    ).expanduser()


def discover_project_root(start: str | Path | None = None) -> Path:
    """Find the containing Git/project root, falling back to the start path."""

    path = Path(start or Path.cwd()).expanduser().resolve()
    if path.is_file():
        path = path.parent
    candidates = (path, *path.parents)
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return path


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-._")
    return value or "project"


def _workspace_key(root: Path) -> str:
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:10]
    return f"{_slug(root.name)}-{digest}"


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _identifier_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        raise WorkspaceError(f"Workspace record does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise WorkspaceError(f"Invalid workspace JSON {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise WorkspaceError(f"Workspace record must be a JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def discover_workflow_specs(root: str | Path) -> list[str]:
    """Discover top-level ``@workflow`` functions without importing modules."""

    project_root = Path(root).expanduser().resolve()
    discovered: list[str] = []
    for path in project_root.rglob("*.py"):
        try:
            relative = path.relative_to(project_root)
        except ValueError:
            continue
        if any(part in _IGNORED_DISCOVERY_PARTS for part in relative.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(_decorator_name(item) == "workflow" for item in node.decorator_list):
                discovered.append(f"{relative.as_posix()}:{node.name}")
    return sorted(set(discovered))


def _looks_like_path(module_ref: str) -> bool:
    return (
        module_ref.endswith(".py")
        or "/" in module_ref
        or "\\" in module_ref
        or Path(module_ref).exists()
    )


class Workspace:
    """Persistent project context and managed development-run records."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        home: str | Path | None = None,
    ) -> None:
        self.root = discover_project_root(root)
        self.home = Path(home).expanduser() if home is not None else zippergen_home()
        self.directory = self.home / "workspaces" / _workspace_key(self.root)
        self.state_path = self.directory / "workspace.json"
        self.secrets_path = self.directory / "development.secrets.json"
        self.runs_directory = self.directory / "runs"
        self.requests_directory = self.directory / "requests"

    def default_state(self) -> dict[str, Any]:
        return {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "project_root": str(self.root),
            "current_workflow": None,
            "current_run": None,
            "last_deployment": None,
            "last_view": "protocol",
            "updated_at": _timestamp(),
        }

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self.default_state()
        state = _read_json(self.state_path)
        if state.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
            raise WorkspaceError(
                f"Unsupported workspace schema in {self.state_path}: "
                f"{state.get('schema_version')!r}"
            )
        if Path(str(state.get("project_root"))).resolve() != self.root:
            raise WorkspaceError(
                f"Workspace {self.state_path} belongs to another project root."
            )
        return state

    def update(self, **changes: object) -> dict[str, Any]:
        state = self.load()
        state.update(changes)
        state["updated_at"] = _timestamp()
        _atomic_write_json(self.state_path, state)
        return state

    @property
    def current_workflow(self) -> str | None:
        value = self.load().get("current_workflow")
        return str(value) if value else None

    @property
    def current_run_id(self) -> str | None:
        value = self.load().get("current_run")
        return str(value) if value else None

    def canonical_spec(self, spec: str, *, cwd: str | Path | None = None) -> str:
        """Store path workflow specs relative to the project when possible."""

        module_ref, separator, workflow_name = spec.partition(":")
        if not _looks_like_path(module_ref):
            return spec
        path = Path(module_ref).expanduser()
        if not path.is_absolute():
            path_from_cwd = Path(cwd or Path.cwd()).expanduser().resolve() / path
            path_from_root = self.root / path
            path = (
                path_from_root
                if not path_from_cwd.exists() and path_from_root.exists()
                else path_from_cwd
            )
        path = path.resolve()
        try:
            display = path.relative_to(self.root).as_posix()
        except ValueError:
            display = str(path)
        return display + (f":{workflow_name}" if separator else "")

    def absolute_spec(self, spec: str) -> str:
        """Resolve a stored path spec for loading from any working directory."""

        module_ref, separator, workflow_name = spec.partition(":")
        if not _looks_like_path(module_ref):
            return spec
        path = Path(module_ref).expanduser()
        if not path.is_absolute():
            path = self.root / path
        value = str(path.resolve())
        return value + (f":{workflow_name}" if separator else "")

    def select_workflow(self, spec: str, *, cwd: str | Path | None = None) -> str:
        canonical = self.canonical_spec(spec, cwd=cwd)
        self.update(current_workflow=canonical)
        return canonical

    def discover_workflows(self) -> list[str]:
        return discover_workflow_specs(self.root)

    def new_run(
        self,
        *,
        workflow_spec: str,
        workflow_name: str,
        fingerprint: str,
        inputs: dict[str, object],
        llm: str,
        options: dict[str, object] | None = None,
        services: str | None = None,
    ) -> dict[str, Any]:
        created_at_ns = time.time_ns()
        base = (
            f"{_slug(workflow_name)}-{_identifier_timestamp()}-"
            f"{created_at_ns % 1_000_000_000:09d}"
        )
        run_id = base
        suffix = 2
        while (self.runs_directory / f"{run_id}.json").exists():
            run_id = f"{base}-{suffix}"
            suffix += 1
        store = self.runs_directory / f"{run_id}.sqlite"
        record: dict[str, Any] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "project_root": str(self.root),
            "workflow_spec": self.canonical_spec(workflow_spec, cwd=self.root),
            "workflow_name": workflow_name,
            "fingerprint": fingerprint,
            "store": str(store),
            "inputs": dict(inputs),
            "llm": llm,
            "options": dict(options or {}),
            "services": services,
            "status": "created",
            "result": None,
            "error": None,
            "created_at": _timestamp(),
            "created_at_ns": created_at_ns,
            "updated_at": _timestamp(),
        }
        self.write_run(record)
        self.update(
            current_workflow=record["workflow_spec"],
            current_run=run_id,
        )
        return record

    def run_path(self, run_id: str) -> Path:
        return self.runs_directory / f"{run_id}.json"

    def load_run(self, run_id: str) -> dict[str, Any]:
        record = _read_json(self.run_path(run_id))
        if record.get("schema_version") != RUN_SCHEMA_VERSION:
            raise WorkspaceError(
                f"Unsupported run schema in {self.run_path(run_id)}: "
                f"{record.get('schema_version')!r}"
            )
        return record

    def current_run(self) -> dict[str, Any] | None:
        run_id = self.current_run_id
        return self.load_run(run_id) if run_id else None

    def write_run(self, record: dict[str, Any]) -> None:
        run_id = str(record.get("run_id") or "")
        if not run_id or _slug(run_id) != run_id:
            raise WorkspaceError(f"Invalid run id: {run_id!r}")
        value = dict(record)
        value["updated_at"] = _timestamp()
        _atomic_write_json(self.run_path(run_id), value)

    def update_run(self, run_id: str, **changes: object) -> dict[str, Any]:
        record = self.load_run(run_id)
        record.update(changes)
        self.write_run(record)
        return self.load_run(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.runs_directory.exists():
            return []
        records = []
        for path in self.runs_directory.glob("*.json"):
            try:
                records.append(self.load_run(path.stem))
            except WorkspaceError:
                continue
        return sorted(
            records,
            key=lambda record: (
                int(record.get("created_at_ns") or 0),
                str(record.get("created_at") or ""),
                str(record.get("run_id") or ""),
            ),
            reverse=True,
        )

    def load_secrets(self) -> dict[str, str]:
        """Load private development secrets without copying them into state."""

        if not self.secrets_path.exists():
            return {}
        values = _read_json(self.secrets_path)
        return {str(name): str(value) for name, value in values.items()}

    def save_secrets(self, values: dict[str, str]) -> None:
        """Persist development secrets with owner-only filesystem permissions."""

        _atomic_write_json(self.secrets_path, dict(values))
        self.secrets_path.chmod(0o600)

    def save_request(
        self,
        *,
        kind: str,
        prompt: str,
        content: str,
        workflow_spec: str | None = None,
    ) -> dict[str, Any]:
        base = f"{_identifier_timestamp()}-{_slug(kind)}"
        request_id = base
        suffix = 2
        while (self.requests_directory / f"{request_id}.json").exists():
            request_id = f"{base}-{suffix}"
            suffix += 1
        record: dict[str, Any] = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "request_id": request_id,
            "kind": kind,
            "project_root": str(self.root),
            "workflow_spec": workflow_spec,
            "prompt": prompt,
            "content_file": str(self.requests_directory / f"{request_id}.md"),
            "created_at": _timestamp(),
        }
        self.requests_directory.mkdir(parents=True, exist_ok=True)
        Path(record["content_file"]).write_text(content.rstrip() + "\n", encoding="utf-8")
        _atomic_write_json(self.requests_directory / f"{request_id}.json", record)
        return record
