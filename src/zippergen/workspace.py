"""Project-aware state for the ZipperGen development experience.

Visible project identity lives in ``zippergen.toml``: one canonical
``specification.md``, one workflow entry point, and portable model, assistant,
and connector configuration can be reviewed, versioned, and recovered from a
clone.
Machine-specific workspace state and its separate owner-only secret file stay
below ``ZIPPERGEN_HOME`` rather than in the user's Git checkout; the ordinary
workspace record is non-secret. The CLI uses this module to manage durable
runs and site-specific configuration.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import tempfile
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any


WORKSPACE_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
PROJECT_SCHEMA_VERSION = 1
PROJECT_MANIFEST_NAME = "zippergen.toml"
SPECIFICATION_FILE_NAME = "specification.md"
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

_MODEL_PROJECT_FIELDS = frozenset({"provider", "model", "spec"})
# Checks are always live, so their results are never stored.  These are the
# fields that describe one machine rather than the project.
_MODEL_SITE_FIELDS = frozenset({"idle_timeout"})
# Results of earlier versions' cached checks, stripped wherever they are read.
_STALE_CHECK_FIELDS = frozenset({"check_status", "check_detail", "checked_at"})
_PROVIDER_PROJECT_FIELDS = frozenset({"kind"})


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
        if (candidate / PROJECT_MANIFEST_NAME).exists():
            return candidate
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


def _workspace_key(root: Path, project_id: str | None = None) -> str:
    identity = f"{root}\0{project_id or ''}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
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


def _atomic_write_text(path: Path, value: str) -> None:
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
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _toml_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _toml_key(value: object) -> str:
    """Render one TOML key without interpreting dots as table separators."""

    return json.dumps(str(value), ensure_ascii=False)


def _string_values(value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise WorkspaceError(f"Project {field} must be a table.")
    return {
        str(key): str(item)
        for key, item in value.items()
        if item is not None
    }


def _named_string_tables(
    value: object,
    *,
    field: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise WorkspaceError(f"Project {field} must be a table.")
    result: dict[str, dict[str, str]] = {}
    for name, raw in value.items():
        result[str(name)] = _string_values(
            raw,
            field=f"{field}.{name}",
        )
    return result


def _object_table(value: object, *, field: str) -> dict[str, object]:
    """Return a shallow string-keyed table with a precise static type."""

    if not isinstance(value, dict):
        raise WorkspaceError(f"Project {field} must be a table.")
    return {str(key): item for key, item in value.items()}


def _safe_project_directory(root: Path, value: object, *, field: str) -> Path:
    raw = str(value).strip()
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise WorkspaceError(
            f"{field} must be a relative directory inside the project; got {raw!r}."
        )
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise WorkspaceError(f"{field} escapes the project root: {raw!r}.")
    return resolved


def _safe_project_file(root: Path, value: object, *, field: str) -> Path:
    raw = str(value).strip()
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise WorkspaceError(
            f"{field} must be a relative file inside the project; got {raw!r}."
        )
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise WorkspaceError(f"{field} escapes the project root: {raw!r}.")
    return resolved


def _without_specification_guide(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("<!-- zippergen:specification-guide"):
        _guide, separator, remainder = stripped.partition("-->")
        if separator:
            return remainder.strip()
    return stripped


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def discover_workflow_specs(
    root: str | Path,
    *,
    ignored_directories: tuple[str, ...] = (),
) -> list[str]:
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
        if any(relative.is_relative_to(Path(item)) for item in ignored_directories):
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
        self.root = (
            Path(root).expanduser().resolve()
            if root is not None
            else discover_project_root()
        )
        self.home = Path(home).expanduser() if home is not None else zippergen_home()

    def _project_id(self) -> str | None:
        """Read only the portable identity needed to locate private state."""

        try:
            raw = tomllib.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (
            FileNotFoundError,
            OSError,
            UnicodeDecodeError,
            tomllib.TOMLDecodeError,
        ):
            return None
        value = str(raw.get("project_id") or "").strip()
        return value or None

    @property
    def directory(self) -> Path:
        return self.home / "workspaces" / _workspace_key(
            self.root, self._project_id()
        )

    @property
    def state_path(self) -> Path:
        return self.directory / "workspace.json"

    @property
    def secrets_path(self) -> Path:
        return self.directory / "development.secrets.json"

    @property
    def runs_directory(self) -> Path:
        return self.directory / "runs"


    @property
    def specification_path(self) -> Path:
        """Return the visible, versionable canonical specification path."""

        manifest = self.project_manifest()
        return _safe_project_file(
            self.root,
            manifest["specification_file"],
            field="specification_file",
        )

    @property
    def manifest_path(self) -> Path:
        return self.root / PROJECT_MANIFEST_NAME

    def project_manifest(self) -> dict[str, object]:
        """Load visible project configuration, or return non-writing defaults."""

        if not self.manifest_path.exists():
            return {
                "schema_version": PROJECT_SCHEMA_VERSION,
                "project_id": None,
                "name": self.root.name,
                "specification_file": SPECIFICATION_FILE_NAME,
                "workflow_entry": None,
                "framework_directory": None,
                "models": {
                    "configurations": {},
                    "assignments": {
                        "default": "mock",
                        "lifelines": {},
                        "actions": {},
                    },
                },
                "assistants": {
                    "configurations": {},
                    "assignments": {
                        "default": "",
                        "lifelines": {},
                        "actions": {},
                    },
                },
                "connectors": {
                    "providers": {},
                    "configurations": {},
                    "bindings": {},
                    "assignments": {"lifelines": {}, "actions": {}},
                },
                "exists": False,
            }
        try:
            manifest = tomllib.loads(self.manifest_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise WorkspaceError(
                f"Invalid project manifest {self.manifest_path}: {exc}"
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise WorkspaceError(
                f"Could not read project manifest {self.manifest_path}: {exc}"
            ) from exc
        if manifest.get("schema_version") != PROJECT_SCHEMA_VERSION:
            raise WorkspaceError(
                f"Unsupported project schema in {self.manifest_path}: "
                f"{manifest.get('schema_version')!r}"
            )
        name = str(manifest.get("name") or "").strip()
        if not name:
            raise WorkspaceError(f"Project name is empty in {self.manifest_path}.")
        specification = str(
            manifest.get("specification_file") or SPECIFICATION_FILE_NAME
        )
        _safe_project_file(
            self.root,
            specification,
            field="specification_file",
        )
        workflow_value = manifest.get("workflow_entry")
        workflow_entry = str(workflow_value).strip() if workflow_value else None
        if workflow_value is not None and not workflow_entry:
            raise WorkspaceError(
                f"Project workflow_entry is empty in {self.manifest_path}."
            )
        if workflow_entry:
            module_ref = workflow_entry.partition(":")[0]
            if _looks_like_path(module_ref):
                _safe_project_file(
                    self.root,
                    module_ref,
                    field="workflow_entry",
                )
        framework_value = manifest.get("framework_directory")
        framework = str(framework_value).strip() if framework_value else None
        if framework:
            _safe_project_directory(
                self.root,
                framework,
                field="framework_directory",
            )
        raw_models = manifest.get("models") or {}
        if not isinstance(raw_models, dict):
            raise WorkspaceError("Project models must be a table.")
        model_configurations = _named_string_tables(
            raw_models.get("configurations") or {},
            field="models.configurations",
        )
        raw_model_assignments = raw_models.get("assignments") or {}
        if not isinstance(raw_model_assignments, dict):
            raise WorkspaceError("Project models.assignments must be a table.")
        model_assignments = {
            "default": str(raw_model_assignments.get("default") or "mock"),
            "lifelines": _string_values(
                raw_model_assignments.get("lifelines") or {},
                field="models.assignments.lifelines",
            ),
            "actions": _string_values(
                raw_model_assignments.get("actions") or {},
                field="models.assignments.actions",
            ),
        }
        raw_assistants = manifest.get("assistants") or {}
        if not isinstance(raw_assistants, dict):
            raise WorkspaceError("Project assistants must be a table.")
        assistant_configurations = _named_string_tables(
            raw_assistants.get("configurations") or {},
            field="assistants.configurations",
        )
        raw_assistant_assignments = raw_assistants.get("assignments") or {}
        if not isinstance(raw_assistant_assignments, dict):
            raise WorkspaceError(
                "Project assistants.assignments must be a table."
            )
        assistant_assignments = {
            "default": str(raw_assistant_assignments.get("default") or ""),
            "lifelines": _string_values(
                raw_assistant_assignments.get("lifelines") or {},
                field="assistants.assignments.lifelines",
            ),
            "actions": _string_values(
                raw_assistant_assignments.get("actions") or {},
                field="assistants.assignments.actions",
            ),
        }
        raw_connectors = manifest.get("connectors") or {}
        if not isinstance(raw_connectors, dict):
            raise WorkspaceError("Project connectors must be a table.")
        connector_providers = _named_string_tables(
            raw_connectors.get("providers") or {},
            field="connectors.providers",
        )
        connector_configurations = _named_string_tables(
            raw_connectors.get("configurations") or {},
            field="connectors.configurations",
        )
        raw_connector_assignments = raw_connectors.get("assignments") or {}
        if not isinstance(raw_connector_assignments, dict):
            raise WorkspaceError(
                "Project connectors.assignments must be a table."
            )
        connector_assignments = {
            "lifelines": _string_values(
                raw_connector_assignments.get("lifelines") or {},
                field="connectors.assignments.lifelines",
            ),
            "actions": _string_values(
                raw_connector_assignments.get("actions") or {},
                field="connectors.assignments.actions",
            ),
        }
        connector_bindings = _string_values(
            raw_connectors.get("bindings") or {},
            field="connectors.bindings",
        )
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": str(manifest.get("project_id") or "").strip() or None,
            "name": name,
            "specification_file": specification,
            "workflow_entry": workflow_entry,
            "framework_directory": framework,
            "models": {
                "configurations": model_configurations,
                "assignments": model_assignments,
            },
            "assistants": {
                "configurations": assistant_configurations,
                "assignments": assistant_assignments,
            },
            "connectors": {
                "providers": connector_providers,
                "configurations": connector_configurations,
                "bindings": connector_bindings,
                "assignments": connector_assignments,
            },
            "exists": True,
        }

    def _write_project_configuration(
        self,
        *,
        models: dict[str, object] | None = None,
        assistants: dict[str, object] | None = None,
        connectors: dict[str, object] | None = None,
    ) -> None:
        """Rewrite visible project configuration in deterministic TOML."""

        self.initialize_project()
        manifest = self.project_manifest()
        model_data = _object_table(
            models if models is not None else manifest["models"],
            field="models",
        )
        assistant_data = _object_table(
            assistants if assistants is not None else manifest["assistants"],
            field="assistants",
        )
        connector_data = _object_table(
            connectors if connectors is not None else manifest["connectors"],
            field="connectors",
        )
        lines = [
            "# Visible, versionable ZipperGen project configuration.",
            f"schema_version = {PROJECT_SCHEMA_VERSION}",
            f"project_id = {_toml_string(manifest['project_id'])}",
            f"name = {_toml_string(manifest['name'])}",
            f"specification_file = {_toml_string(manifest['specification_file'])}",
        ]
        if manifest.get("workflow_entry"):
            lines.append(
                f"workflow_entry = {_toml_string(manifest['workflow_entry'])}"
            )
        if manifest.get("framework_directory"):
            lines.append(
                f"framework_directory = "
                f"{_toml_string(manifest['framework_directory'])}"
            )

        configurations = model_data.get("configurations") or {}
        assert isinstance(configurations, dict)
        for name, raw in sorted(configurations.items()):
            assert isinstance(raw, dict)
            lines.extend(["", f"[models.configurations.{_toml_key(name)}]"])
            lines.extend(
                f"{_toml_key(key)} = {_toml_string(value)}"
                for key, value in sorted(raw.items())
            )
        assignments = model_data.get("assignments") or {}
        assert isinstance(assignments, dict)
        default = str(assignments.get("default") or "mock")
        lifelines = assignments.get("lifelines") or {}
        actions = assignments.get("actions") or {}
        if default != "mock" or lifelines or actions:
            lines.extend(["", "[models.assignments]"])
            lines.append(f"default = {_toml_string(default)}")
        for label, values in (("lifelines", lifelines), ("actions", actions)):
            if values:
                assert isinstance(values, dict)
                lines.extend(["", f"[models.assignments.{label}]"])
                lines.extend(
                    f"{_toml_key(key)} = {_toml_string(value)}"
                    for key, value in sorted(values.items())
                )

        assistant_configurations = assistant_data.get("configurations") or {}
        assistant_assignments = assistant_data.get("assignments") or {}
        assert isinstance(assistant_configurations, dict)
        assert isinstance(assistant_assignments, dict)
        for name, raw in sorted(assistant_configurations.items()):
            assert isinstance(raw, dict)
            lines.extend(
                ["", f"[assistants.configurations.{_toml_key(name)}]"]
            )
            lines.extend(
                f"{_toml_key(key)} = {_toml_string(value)}"
                for key, value in sorted(raw.items())
            )
        assistant_default = str(assistant_assignments.get("default") or "")
        assistant_lifelines = assistant_assignments.get("lifelines") or {}
        assistant_actions = assistant_assignments.get("actions") or {}
        if assistant_default:
            lines.extend(["", "[assistants.assignments]"])
            lines.append(f"default = {_toml_string(assistant_default)}")
        for label, values in (
            ("lifelines", assistant_lifelines),
            ("actions", assistant_actions),
        ):
            if values:
                assert isinstance(values, dict)
                lines.extend(["", f"[assistants.assignments.{label}]"])
                lines.extend(
                    f"{_toml_key(key)} = {_toml_string(value)}"
                    for key, value in sorted(values.items())
                )

        providers = connector_data.get("providers") or {}
        connector_configurations = connector_data.get("configurations") or {}
        bindings = connector_data.get("bindings") or {}
        connector_assignments = connector_data.get("assignments") or {}
        assert isinstance(providers, dict)
        assert isinstance(connector_configurations, dict)
        assert isinstance(bindings, dict)
        assert isinstance(connector_assignments, dict)
        for name, raw in sorted(providers.items()):
            assert isinstance(raw, dict)
            lines.extend(["", f"[connectors.providers.{_toml_key(name)}]"])
            lines.extend(
                f"{_toml_key(key)} = {_toml_string(value)}"
                for key, value in sorted(raw.items())
            )
        for name, raw in sorted(connector_configurations.items()):
            assert isinstance(raw, dict)
            lines.extend(
                ["", f"[connectors.configurations.{_toml_key(name)}]"]
            )
            lines.extend(
                f"{_toml_key(key)} = {_toml_string(value)}"
                for key, value in sorted(raw.items())
            )
        if bindings:
            lines.extend(["", "[connectors.bindings]"])
            lines.extend(
                f"{_toml_key(key)} = {_toml_string(value)}"
                for key, value in sorted(bindings.items())
            )
        for label in ("lifelines", "actions"):
            values = connector_assignments.get(label) or {}
            if values:
                assert isinstance(values, dict)
                lines.extend(["", f"[connectors.assignments.{label}]"])
                lines.extend(
                    f"{_toml_key(key)} = {_toml_string(value)}"
                    for key, value in sorted(values.items())
                )
        _atomic_write_text(self.manifest_path, "\n".join(lines) + "\n")


    def initialize_project(
        self,
        *,
        name: str | None = None,
        specification_file: str = SPECIFICATION_FILE_NAME,
        framework_directory: str | None = None,
    ) -> dict[str, object]:
        """Create the visible project manifest."""

        if self.manifest_path.exists():
            manifest = self.project_manifest()
            self._ensure_project_gitignore(
                str(manifest["framework_directory"])
                if manifest.get("framework_directory")
                else None
            )
            return manifest
        project_name = str(name or self.root.name).strip()
        if not project_name:
            raise WorkspaceError("Project name must not be empty.")
        _safe_project_file(
            self.root,
            specification_file,
            field="specification_file",
        )
        if framework_directory is None and (
            self.root / "zippergen" / "pyproject.toml"
        ).is_file():
            framework_directory = "zippergen"
        if framework_directory:
            _safe_project_directory(
                self.root,
                framework_directory,
                field="framework_directory",
            )
        content = (
            "# Visible, versionable ZipperGen project configuration.\n"
            f"schema_version = {PROJECT_SCHEMA_VERSION}\n"
            f"project_id = {_toml_string(uuid.uuid4().hex)}\n"
            f"name = {_toml_string(project_name)}\n"
            f"specification_file = {_toml_string(specification_file)}\n"
        )
        if framework_directory:
            content += (
                f"framework_directory = {_toml_string(framework_directory)}\n"
            )
        _atomic_write_text(self.manifest_path, content)
        self._ensure_project_gitignore(framework_directory)
        return self.project_manifest()

    def require_project(self) -> dict[str, object]:
        """Return the manifest or reject an accidental non-project directory."""

        if not self.manifest_path.is_file():
            raise WorkspaceError(
                f"Not a ZipperGen project: {self.root}. Run 'zg init' in the "
                "project directory first."
            )
        return self.project_manifest()

    @property
    def workflow_entry(self) -> str | None:
        """Return the versioned workflow entry point from the manifest."""

        value = self.project_manifest().get("workflow_entry")
        return str(value) if value else None

    def select_workflow(
        self,
        spec: str,
        *,
        cwd: str | Path | None = None,
        replace: bool = False,
    ) -> str:
        """Record an explicit workflow entry when convention is ambiguous."""

        canonical = self.canonical_spec(spec, cwd=cwd)
        module_ref = canonical.partition(":")[0]
        if _looks_like_path(module_ref):
            _safe_project_file(
                self.root,
                module_ref,
                field="workflow_entry",
            )
        self.initialize_project()
        existing = self.workflow_entry
        if existing == canonical:
            return canonical
        if existing is not None and not replace:
            raise WorkspaceError(
                f"This project already uses {existing}. One project has one "
                "workflow; 'zippergen workflow select SPEC' replaces it."
            )
        try:
            content = self.manifest_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise WorkspaceError(
                f"Could not read project manifest {self.manifest_path}: {exc}"
            ) from exc
        replacement = f"workflow_entry = {_toml_string(canonical)}"
        updated, replacements = re.subn(
            r"(?m)^workflow_entry\s*=\s*.*$",
            lambda _match: replacement,
            content,
            count=1,
        )
        if replacements == 0:
            specification_line = r"(?m)^(specification_file\s*=\s*.*)$"
            updated, inserted = re.subn(
                specification_line,
                lambda match: f"{match.group(1)}\n{replacement}",
                content,
                count=1,
            )
            if inserted != 1:
                raise WorkspaceError(
                    "Project manifest has no specification_file field beside "
                    "which to record workflow_entry."
                )
        _atomic_write_text(self.manifest_path, updated)
        return canonical

    def _ensure_project_gitignore(self, framework_directory: str | None) -> None:
        if (self.root / ".git").exists():
            gitignore = self.root / ".gitignore"
            existing = (
                gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
            )
            desired = ["/tutorial-runtime/"]
            if framework_directory:
                desired.insert(0, f"/{framework_directory.rstrip('/')}/")
            current = {line.strip() for line in existing.splitlines()}
            missing = [entry for entry in desired if entry not in current]
            if missing:
                separator = "" if not existing or existing.endswith("\n") else "\n"
                _atomic_write_text(
                    gitignore,
                    existing
                    + separator
                    + "# ZipperGen transparent runtime\n"
                    + "\n".join(missing)
                    + "\n",
                )

    def default_state(self) -> dict[str, Any]:
        return {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "project_root": str(self.root),
            "current_run": None,
            "model_configuration_overrides": {},
            "model_site_profiles": {},
            "providers": {},
            "connector_providers": {},
            "connector_configuration_overrides": {},
            "connector_site_bindings": {},
            "connector_site_assignments": {},
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
        # Workspace state is intentionally limited to site facts. Project
        # identity, configurations, and assignments live in zippergen.toml.
        state.setdefault("model_configuration_overrides", {})
        state.setdefault("model_site_profiles", {})
        state.setdefault("providers", {})
        state.setdefault("connector_providers", {})
        state.setdefault("connector_configuration_overrides", {})
        state.setdefault("connector_site_bindings", {})
        state.setdefault("connector_site_assignments", {})
        return state

    def update(self, **changes: object) -> dict[str, Any]:
        state = self.load()
        state.update(changes)
        state["updated_at"] = _timestamp()
        _atomic_write_json(self.state_path, state)
        return state

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

    def model_configurations(self) -> dict[str, dict[str, str]]:
        """Return project model configurations with one site override."""

        state = self.load()
        manifest_models = self.project_manifest().get("models") or {}
        assert isinstance(manifest_models, dict)
        raw_project = manifest_models.get("configurations") or {}
        assert isinstance(raw_project, dict)
        configurations: dict[str, dict[str, str]] = {
            "mock": {
                "provider": "mock",
                "model": "",
                "spec": "mock",
            }
        }
        for name, raw_configuration in raw_project.items():
            if not isinstance(raw_configuration, dict):
                raise WorkspaceError(
                    f"Model configuration {name!r} must be an object."
                )
            configurations[str(name)] = {
                str(key): str(value)
                for key, value in raw_configuration.items()
                if value is not None and str(key) not in _STALE_CHECK_FIELDS
            }
        raw_overrides = state.get("model_configuration_overrides") or {}
        if not isinstance(raw_overrides, dict):
            raise WorkspaceError(
                "Workspace model_configuration_overrides must be an object."
            )
        for name, raw_override in raw_overrides.items():
            if not isinstance(raw_override, dict):
                raise WorkspaceError(
                    f"Model site override {name!r} must be an object."
                )
            if str(name) in configurations:
                configurations[str(name)].update(
                    {
                        str(key): str(value)
                        for key, value in raw_override.items()
                        if value is not None
                    }
                )
        return configurations

    def save_model_configuration(
        self,
        name: str,
        values: dict[str, str],
    ) -> dict[str, str]:
        """Save portable model identity and machine-specific observations."""

        normalized = name.strip()
        if (
            normalized.casefold() == "mock"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", normalized)
        ):
            raise WorkspaceError(
                "A model configuration name must start with a letter or digit, "
                "contain only letters, digits, '.', '_' or '-', and must not "
                "replace the built-in name 'mock'."
            )
        if not values.get("spec"):
            raise WorkspaceError("A model configuration requires a model spec.")
        idle_timeout = str(values.get("idle_timeout") or "").strip()
        if idle_timeout:
            provider = str(values.get("provider") or "").casefold()
            if provider not in {"local", "ollama"}:
                raise WorkspaceError(
                    "Idle release is only available for local Ollama model "
                    "configurations."
                )
            try:
                idle_seconds = float(idle_timeout)
            except ValueError as exc:
                raise WorkspaceError(
                    "A model configuration idle timeout must be a number of "
                    "seconds."
                ) from exc
            if not math.isfinite(idle_seconds) or idle_seconds < 0:
                raise WorkspaceError(
                    "A model configuration idle timeout must be a non-negative "
                    "number of seconds."
                )
            values = {
                **values,
                "idle_timeout": (
                    str(int(idle_seconds))
                    if idle_seconds.is_integer()
                    else str(idle_seconds)
                ),
            }
        state = self.load()
        manifest = self.project_manifest()
        models = _object_table(manifest["models"], field="models")
        raw_project = _object_table(
            models.get("configurations") or {},
            field="models.configurations",
        )
        conflicting = next(
            (
                str(existing)
                for existing in raw_project
                if str(existing).casefold() == normalized.casefold()
                and str(existing) != normalized
            ),
            None,
        )
        if conflicting is not None:
            raise WorkspaceError(
                f"Model configuration {normalized!r} differs only by case "
                f"from existing configuration {conflicting!r}."
            )
        project_configurations = {
            str(key): dict(value)
            for key, value in raw_project.items()
            if isinstance(value, dict)
        }
        project_configuration = {
            str(key): str(value)
            for key, value in values.items()
            if value is not None and str(key) in _MODEL_PROJECT_FIELDS
        }
        project_configurations[normalized] = project_configuration
        models["configurations"] = project_configurations
        self._write_project_configuration(models=models)

        raw_overrides = state.get("model_configuration_overrides") or {}
        if not isinstance(raw_overrides, dict):
            raise WorkspaceError(
                "Workspace model_configuration_overrides must be an object."
            )
        site_overrides = dict(raw_overrides)
        override = {
            str(key): str(value)
            for key, value in values.items()
            if value is not None and str(key) in _MODEL_SITE_FIELDS
        }
        if override:
            site_overrides[normalized] = override
        else:
            site_overrides.pop(normalized, None)
        self.update(model_configuration_overrides=site_overrides)
        return self.model_configurations()[normalized]

    def automatic_model_configuration_name(self, spec: str) -> str:
        """Return a stable unused name for a compact model spec."""

        normalized = spec.strip()
        if normalized == "mock":
            return "mock"
        provider, separator, model = normalized.partition(":")
        provider = {
            "claude": "anthropic",
            "ollama": "local",
        }.get(provider.casefold(), provider.casefold())
        stem = provider if not separator else f"{provider}-{model}"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("._-")
        stem = stem[:56] or "model"
        configurations = self.model_configurations()
        for name, configuration in configurations.items():
            if configuration.get("spec") == normalized:
                return name
        candidate = stem
        suffix = 2
        existing_names = {name.casefold() for name in configurations}
        while candidate.casefold() in existing_names:
            candidate = f"{stem[:59]}-{suffix}"
            suffix += 1
        return candidate

    def model_assignment_profile(
        self,
        workflow_spec: str,
        *,
        default: str = "mock",
        include_site: bool = True,
    ) -> dict[str, Any]:
        """Return project assignments with this site's optional overrides."""

        state = self.load()
        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        manifest_models = self.project_manifest().get("models") or {}
        assert isinstance(manifest_models, dict)
        project = manifest_models.get("assignments") or {}
        assert isinstance(project, dict)
        project_lifelines = project.get("lifelines") or {}
        project_actions = project.get("actions") or {}
        if not isinstance(project_lifelines, dict) or not isinstance(
            project_actions, dict
        ):
            raise WorkspaceError("Project model assignments are malformed.")
        result: dict[str, Any] = {
            "default": str(project.get("default") or default),
            "lifelines": {
                str(name): str(configuration)
                for name, configuration in project_lifelines.items()
            },
            "actions": {
                str(name): str(configuration)
                for name, configuration in project_actions.items()
            },
        }
        if not include_site:
            return result
        raw_site_profiles = state.get("model_site_profiles") or {}
        if not isinstance(raw_site_profiles, dict):
            raise WorkspaceError(
                "Workspace model_site_profiles must be an object."
            )
        site = raw_site_profiles.get(canonical) or {}
        if not isinstance(site, dict):
            raise WorkspaceError(
                f"Model site assignments for {canonical} must be an object."
            )
        if site.get("default"):
            result["default"] = str(site["default"])
        for key in ("lifelines", "actions"):
            values = site.get(key) or {}
            if not isinstance(values, dict):
                raise WorkspaceError(
                    f"Model site {key} for {canonical} must be an object."
                )
            result[key].update(
                {
                    str(name): str(configuration)
                    for name, configuration in values.items()
                }
            )
        return result

    def has_model_assignment_profile(self, workflow_spec: str) -> bool:
        """Whether this workflow has project or site model assignments.

        This check is deliberately read-only.  Runtime commands use it before
        resolving named configurations so a project with no assignments keeps
        the workflow's own default without creating configuration as a side
        effect.
        """

        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        state = self.load()
        if self._is_project_workflow(canonical) and self.manifest_path.exists():
            try:
                raw_manifest = tomllib.loads(
                    self.manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
                raise WorkspaceError(
                    f"Could not read project model assignments: {exc}"
                ) from exc
            raw_models = raw_manifest.get("models") or {}
            if isinstance(raw_models, dict) and "assignments" in raw_models:
                return True

        raw_site = state.get("model_site_profiles") or {}
        if isinstance(raw_site, dict) and raw_site.get(canonical):
            return True
        return False

    def save_model_assignment_profile(
        self,
        workflow_spec: str,
        *,
        default: str,
        lifelines: dict[str, str],
        actions: dict[str, str] | None = None,
        site: bool = False,
    ) -> dict[str, Any]:
        """Persist project assignments or one private site override."""

        configurations = self.model_configurations()
        action_assignments = dict(actions or {})
        names = {default, *lifelines.values(), *action_assignments.values()}
        missing = sorted(names - set(configurations))
        if missing:
            raise WorkspaceError(
                "Unknown model configuration(s): " + ", ".join(missing)
            )
        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        state = self.load()
        if site:
            raw_site_profiles = state.get("model_site_profiles") or {}
            if not isinstance(raw_site_profiles, dict):
                raise WorkspaceError(
                    "Workspace model_site_profiles must be an object."
                )
            site_profiles = dict(raw_site_profiles)
            project = self.model_assignment_profile(
                canonical,
                default=default,
                include_site=False,
            )
            project_lifelines = dict(project.get("lifelines") or {})
            project_actions = dict(project.get("actions") or {})
            site_profile: dict[str, object] = {
                "lifelines": {
                    str(name): str(configuration)
                    for name, configuration in sorted(lifelines.items())
                    if project_lifelines.get(name) != configuration
                },
                "actions": {
                    str(name): str(configuration)
                    for name, configuration in sorted(action_assignments.items())
                    if project_actions.get(name) != configuration
                },
            }
            if str(project.get("default") or "mock") != default:
                site_profile["default"] = default
            if site_profile.get("default") or site_profile["lifelines"] or site_profile["actions"]:
                site_profiles[canonical] = site_profile
            else:
                site_profiles.pop(canonical, None)
            self.update(model_site_profiles=site_profiles)
            return {
                "default": default,
                "lifelines": dict(lifelines),
                "actions": dict(action_assignments),
            }

        manifest = self.project_manifest()
        models = _object_table(manifest["models"], field="models")
        models["assignments"] = {
            "default": default,
            "lifelines": {
                str(name): str(configuration)
                for name, configuration in sorted(lifelines.items())
            },
            "actions": {
                str(name): str(configuration)
                for name, configuration in sorted(action_assignments.items())
            },
        }
        self._write_project_configuration(models=models)
        lifeline_assignments = {
            str(name): str(configuration)
            for name, configuration in sorted(lifelines.items())
        }
        result: dict[str, object] = {
            "default": default,
            "lifelines": lifeline_assignments,
        }
        if action_assignments:
            result["actions"] = {
                str(name): str(configuration)
                for name, configuration
                in sorted(action_assignments.items())
            }
        return result

    def provider_profiles(self) -> dict[str, dict[str, str]]:
        raw = self.load().get("providers") or {}
        if not isinstance(raw, dict):
            raise WorkspaceError("Workspace providers must be an object.")
        profiles: dict[str, dict[str, str]] = {}
        for name, raw_profile in raw.items():
            if not isinstance(raw_profile, dict):
                raise WorkspaceError(f"Provider profile {name!r} must be an object.")
            profiles[str(name)] = {
                str(key): str(value)
                for key, value in raw_profile.items()
                if value is not None
            }
        return profiles

    def save_provider_profile(
        self,
        name: str,
        values: dict[str, str],
    ) -> dict[str, str]:
        """Save machine-specific model-provider connection settings."""

        normalized = name.strip().casefold()
        if not re.fullmatch(r"[a-z][a-z0-9._-]{0,63}", normalized):
            raise WorkspaceError(
                "A model provider name must start with a letter and contain "
                "only letters, digits, '.', '_' or '-'."
            )
        state = self.load()
        raw = state.get("providers") or {}
        if not isinstance(raw, dict):
            raise WorkspaceError("Workspace providers must be an object.")
        profiles = {
            str(key): dict(value)
            for key, value in raw.items()
            if isinstance(value, dict)
        }
        profiles[normalized] = {
            str(key): str(value)
            for key, value in values.items()
            if value is not None and str(value).strip()
        }
        self.update(providers=profiles)
        return self.provider_profiles()[normalized]

    def remove_model_configuration(self, name: str) -> None:
        """Remove one unused named model configuration."""

        normalized = name.strip()
        if normalized == "mock":
            raise WorkspaceError("The built-in mock configuration cannot be removed.")
        configurations = self.model_configurations()
        if normalized not in configurations:
            raise WorkspaceError(f"Model configuration does not exist: {normalized}")
        workflow = self.resolve_workflow()
        profile = self.model_assignment_profile(workflow)
        references: list[str] = []
        for group in ("lifelines", "actions"):
            raw = profile.get(group) or {}
            if not isinstance(raw, dict):
                raise WorkspaceError(
                    "Project model assignments are malformed."
                )
            references.extend(
                str(target)
                for target, selected in raw.items()
                if str(selected) == normalized
            )
        if profile.get("default") == normalized:
            references.insert(0, "default")
        if references:
            raise WorkspaceError(
                f"Model configuration {normalized!r} is still assigned to: "
                + ", ".join(references)
                + ". Unassign it first."
            )
        manifest = self.project_manifest()
        models = _object_table(manifest["models"], field="models")
        project = _object_table(
            models.get("configurations") or {},
            field="models.configurations",
        )
        project.pop(normalized, None)
        models["configurations"] = project
        self._write_project_configuration(models=models)
        state = self.load()
        updates: dict[str, object] = {}
        for key in ("model_configuration_overrides",):
            raw = state.get(key) or {}
            if isinstance(raw, dict):
                values = dict(raw)
                values.pop(normalized, None)
                updates[key] = values
        if updates:
            self.update(**updates)

    def assistant_configurations(self) -> dict[str, dict[str, str]]:
        """Return portable named coding-assistant configurations."""

        raw = self.project_manifest().get("assistants") or {}
        if not isinstance(raw, dict):
            raise WorkspaceError("Project assistants must be a table.")
        configurations = raw.get("configurations") or {}
        if not isinstance(configurations, dict):
            raise WorkspaceError(
                "Project assistants.configurations must be a table."
            )
        result: dict[str, dict[str, str]] = {}
        for name, configuration in configurations.items():
            if not isinstance(configuration, dict):
                raise WorkspaceError(
                    f"Assistant configuration {name!r} must be a table."
                )
            backend = str(configuration.get("backend") or "").strip().casefold()
            if backend not in {"codex", "claude"}:
                raise WorkspaceError(
                    f"Assistant configuration {name!r} must select backend "
                    "'codex' or 'claude'."
                )
            result[str(name)] = {"backend": backend}
        return result

    def save_assistant_configuration(
        self,
        name: str,
        backend: str,
    ) -> dict[str, str]:
        """Save one portable coding-assistant backend selection."""

        normalized = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", normalized):
            raise WorkspaceError(
                "An assistant configuration name must start with a letter or "
                "digit and contain only letters, digits, '.', '_' or '-'."
            )
        selected = backend.strip().casefold()
        if selected not in {"codex", "claude"}:
            raise WorkspaceError(
                "An assistant configuration backend must be 'codex' or "
                "'claude'."
            )
        manifest = self.project_manifest()
        assistants = _object_table(
            manifest["assistants"],
            field="assistants",
        )
        configurations = _object_table(
            assistants.get("configurations") or {},
            field="assistants.configurations",
        )
        conflicting = next(
            (
                str(existing)
                for existing in configurations
                if str(existing).casefold() == normalized.casefold()
                and str(existing) != normalized
            ),
            None,
        )
        if conflicting is not None:
            raise WorkspaceError(
                f"Assistant configuration {normalized!r} conflicts with "
                f"existing name {conflicting!r}."
            )
        configurations[normalized] = {"backend": selected}
        assistants["configurations"] = configurations
        self._write_project_configuration(assistants=assistants)
        return self.assistant_configurations()[normalized]

    def assistant_assignment_profile(
        self,
        workflow_spec: str,
    ) -> dict[str, object]:
        """Return portable default, participant, and action assignments."""

        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        if not self._is_project_workflow(canonical):
            return {"default": "", "lifelines": {}, "actions": {}}
        raw = self.project_manifest().get("assistants") or {}
        if not isinstance(raw, dict):
            raise WorkspaceError("Project assistants must be a table.")
        assignments = raw.get("assignments") or {}
        if not isinstance(assignments, dict):
            raise WorkspaceError(
                "Project assistants.assignments must be a table."
            )
        result: dict[str, object] = {
            "default": str(assignments.get("default") or ""),
            "lifelines": {
                str(target): str(configuration)
                for target, configuration in dict(
                    assignments.get("lifelines") or {}
                ).items()
            },
            "actions": {
                str(target): str(configuration)
                for target, configuration in dict(
                    assignments.get("actions") or {}
                ).items()
            },
        }
        return result

    def has_assistant_assignment_profile(self, workflow_spec: str) -> bool:
        """Whether the project declares any coding-assistant assignment."""

        profile = self.assistant_assignment_profile(workflow_spec)
        return bool(
            profile.get("default")
            or profile.get("lifelines")
            or profile.get("actions")
        )

    def save_assistant_assignment_profile(
        self,
        workflow_spec: str,
        *,
        default: str,
        lifelines: dict[str, str],
        actions: dict[str, str],
    ) -> dict[str, object]:
        """Persist portable coding-assistant assignments."""

        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        if not self._is_project_workflow(canonical):
            raise WorkspaceError(
                "Assistant assignments can only be saved for this project's "
                "workflow."
            )
        configurations = self.assistant_configurations()
        selected = {
            *(value for value in (default,) if value),
            *lifelines.values(),
            *actions.values(),
        }
        missing = sorted(selected - set(configurations))
        if missing:
            raise WorkspaceError(
                "Unknown assistant configuration(s): " + ", ".join(missing)
            )
        manifest = self.project_manifest()
        assistants = _object_table(
            manifest["assistants"],
            field="assistants",
        )
        assistants["assignments"] = {
            "default": default,
            "lifelines": {
                str(target): str(configuration)
                for target, configuration in sorted(lifelines.items())
            },
            "actions": {
                str(target): str(configuration)
                for target, configuration in sorted(actions.items())
            },
        }
        self._write_project_configuration(assistants=assistants)
        return self.assistant_assignment_profile(canonical)

    def remove_assistant_configuration(self, name: str) -> None:
        """Remove one unused named coding-assistant configuration."""

        normalized = name.strip()
        configurations = self.assistant_configurations()
        if normalized not in configurations:
            raise WorkspaceError(
                f"Assistant configuration does not exist: {normalized}"
            )
        workflow = self.resolve_workflow()
        profile = self.assistant_assignment_profile(workflow)
        references: list[str] = []
        for group in ("lifelines", "actions"):
            assignments = profile.get(group)
            if not isinstance(assignments, dict):
                continue
            references.extend(
                str(target)
                for target, selected in assignments.items()
                if selected == normalized
            )
        if profile.get("default") == normalized:
            references.insert(0, "default")
        if references:
            raise WorkspaceError(
                f"Assistant configuration {normalized!r} is still assigned "
                "to: " + ", ".join(references) + ". Unassign it first."
            )
        manifest = self.project_manifest()
        assistants = _object_table(
            manifest["assistants"],
            field="assistants",
        )
        project = _object_table(
            assistants.get("configurations") or {},
            field="assistants.configurations",
        )
        project.pop(normalized, None)
        assistants["configurations"] = project
        self._write_project_configuration(assistants=assistants)

    def development_provider_environment(
        self,
        model_specs: tuple[str, ...],
    ) -> dict[str, str]:
        """Resolve privately configured API keys and local endpoint settings."""

        aliases = {
            "claude": "anthropic",
            "ollama": "local",
        }
        secret_names = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }
        selected: set[str] = set()
        for spec in model_specs:
            raw_provider = spec.partition(":")[0].strip().lower()
            selected.add(aliases.get(raw_provider, raw_provider))
        secrets = self.load_secrets()
        profiles = self.provider_profiles()
        environment: dict[str, str] = {}
        for provider in selected:
            secret_name = secret_names.get(provider)
            if secret_name:
                secret_value = secrets.get(secret_name) or os.environ.get(
                    secret_name
                )
                if secret_value:
                    environment[secret_name] = secret_value
            if provider == "local":
                base_url = profiles.get("local", {}).get("base_url")
                if base_url:
                    environment["OLLAMA_BASE_URL"] = base_url
        return environment

    def discover_workflows(self) -> list[str]:
        framework = self.project_manifest().get("framework_directory")
        ignored = (str(framework),) if framework else ()
        return discover_workflow_specs(self.root, ignored_directories=ignored)

    def resolve_workflow(self, explicit: str | None = None) -> str:
        """Resolve the project's workflow without changing the manifest.

        Convention covers the ordinary one-workflow project. Configuration is
        required only when discovery is ambiguous.
        """

        if explicit:
            return self.canonical_spec(explicit, cwd=self.root)
        if self.workflow_entry:
            return self.canonical_spec(self.workflow_entry, cwd=self.root)
        discovered = self.discover_workflows()
        if len(discovered) == 1:
            return self.canonical_spec(discovered[0], cwd=self.root)
        if discovered:
            raise WorkspaceError(
                "This project has several workflows: "
                + ", ".join(discovered)
                + ". Name one explicitly, or record one with "
                "'zippergen workflow select SPEC'."
            )
        raise WorkspaceError(
            "No workflow was named and none was found in this project. "
            "Write a workflow, or give a workflow spec."
        )

    def _is_project_workflow(self, canonical: str) -> bool:
        """Whether a canonical spec is this one-workflow project's workflow."""

        try:
            return canonical == self.resolve_workflow()
        except WorkspaceError:
            return False

    def new_run(
        self,
        *,
        workflow_spec: str,
        workflow_name: str,
        fingerprint: str,
        inputs: dict[str, object],
        llm: str,
        llms: dict[str, str] | None = None,
        llm_idle_timeout: float | None = None,
        llm_idle_timeouts: dict[str, float] | None = None,
        assistant: str | None = None,
        assistants: dict[str, str] | None = None,
        options: dict[str, object] | None = None,
        connectors: dict[str, object] | None = None,
        store_path: str | None = None,
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
        store = (
            Path(store_path).expanduser().resolve()
            if store_path is not None
            else self.runs_directory / f"{run_id}.sqlite"
        )
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
            "llms": dict(llms or {}),
            "llm_idle_timeout": llm_idle_timeout,
            "llm_idle_timeouts": {
                str(target): float(value)
                for target, value in (llm_idle_timeouts or {}).items()
            },
            "assistant": assistant,
            "assistants": dict(assistants or {}),
            "options": dict(options or {}),
            "connectors": dict(connectors or {}),
            "status": "created",
            "result": None,
            "error": None,
            "created_at": _timestamp(),
            "created_at_ns": created_at_ns,
            "updated_at": _timestamp(),
        }
        self.write_run(record)
        self.update(
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

    def connector_configurations(self) -> dict[str, dict[str, str]]:
        """Return project connector configurations with site observations."""

        state = self.load()
        manifest_connectors = self.project_manifest().get("connectors") or {}
        assert isinstance(manifest_connectors, dict)
        raw_project = manifest_connectors.get("configurations") or {}
        assert isinstance(raw_project, dict)
        configurations: dict[str, dict[str, str]] = {}
        for name, value in raw_project.items():
            if not isinstance(value, dict):
                raise WorkspaceError(
                    f"Connector configuration {name!r} must be an object."
                )
            configurations[str(name)] = {
                str(key): str(item)
                for key, item in value.items()
                if item is not None and str(key) not in _STALE_CHECK_FIELDS
            }
        raw_overrides = state.get("connector_configuration_overrides") or {}
        if not isinstance(raw_overrides, dict):
            raise WorkspaceError(
                "Workspace connector_configuration_overrides must be an object."
            )
        for name, value in raw_overrides.items():
            if isinstance(value, dict) and str(name) in configurations:
                configurations[str(name)].update(
                    {str(key): str(item) for key, item in value.items()}
                )
        return configurations

    def connector_provider_profiles(self) -> dict[str, dict[str, str]]:
        """Return non-secret connector provider connection metadata."""

        raw_site = self.load().get("connector_providers") or {}
        if not isinstance(raw_site, dict):
            raise WorkspaceError(
                "Workspace connector_providers must be an object."
            )
        profiles: dict[str, dict[str, str]] = {}
        manifest_connectors = self.project_manifest().get("connectors") or {}
        assert isinstance(manifest_connectors, dict)
        raw_project = manifest_connectors.get("providers") or {}
        assert isinstance(raw_project, dict)
        for name, value in raw_project.items():
            if isinstance(value, dict):
                profiles[str(name)] = {
                    str(key): str(item) for key, item in value.items()
                }
        for name, value in raw_site.items():
            if not isinstance(value, dict):
                raise WorkspaceError(
                    f"Connector provider {name!r} must be an object."
                )
            profiles.setdefault(str(name), {}).update({
                str(key): str(item)
                for key, item in value.items()
                if item is not None
                and str(key) not in _PROVIDER_PROJECT_FIELDS
            })
        return profiles

    def save_connector_provider_profile(
        self,
        name: str,
        values: dict[str, str],
    ) -> dict[str, str]:
        """Save portable provider kind and machine-specific observations."""

        normalized = name.strip().casefold()
        if not re.fullmatch(r"[a-z][a-z0-9._-]{0,63}", normalized):
            raise WorkspaceError(
                "A connector provider name must start with a letter and "
                "contain only letters, digits, '.', '_' or '-'."
            )
        profile = {
            str(key): str(value)
            for key, value in values.items()
            if value is not None
        }
        manifest = self.project_manifest()
        connectors = _object_table(manifest["connectors"], field="connectors")
        project_providers = _object_table(
            connectors.get("providers") or {},
            field="connectors.providers",
        )
        project_providers[normalized] = {
            key: value
            for key, value in profile.items()
            if key in _PROVIDER_PROJECT_FIELDS
        }
        connectors["providers"] = project_providers
        self._write_project_configuration(connectors=connectors)
        state = self.load()
        raw_site = state.get("connector_providers") or {}
        if not isinstance(raw_site, dict):
            raise WorkspaceError("Workspace connector_providers must be an object.")
        site = dict(raw_site)
        site_values = {
            key: value
            for key, value in profile.items()
            if key not in _PROVIDER_PROJECT_FIELDS
        }
        if site_values:
            site[normalized] = site_values
        else:
            site.pop(normalized, None)
        self.update(connector_providers=site)
        return self.connector_provider_profiles()[normalized]

    @staticmethod
    def connector_provider_secret_name(provider: str, field: str) -> str:
        return f"connector-provider:{provider.casefold()}:{field}"

    def connector_provider_secret(
        self,
        provider: str,
        field: str,
    ) -> str | None:
        return self.load_secrets().get(
            self.connector_provider_secret_name(provider, field)
        )

    def save_connector_provider_secret(
        self,
        provider: str,
        field: str,
        value: str,
    ) -> None:
        secrets = self.load_secrets()
        secrets[self.connector_provider_secret_name(provider, field)] = value
        self.save_secrets(secrets)



    def save_connector_configuration(
        self,
        name: str,
        values: dict[str, str],
    ) -> dict[str, str]:
        """Save a portable connector target plus private site observations."""

        normalized = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", normalized):
            raise WorkspaceError(
                "A connector configuration name must start with a letter or "
                "digit and contain only letters, digits, '.', '_' or '-'."
            )
        kind = str(values.get("kind") or "").strip()
        if not kind:
            raise WorkspaceError("A connector configuration requires a kind.")
        configurations = self.connector_configurations()
        conflicting = next(
            (
                existing
                for existing in configurations
                if existing.casefold() == normalized.casefold()
                and existing != normalized
            ),
            None,
        )
        if conflicting:
            raise WorkspaceError(
                f"Connector configuration {normalized!r} differs only by "
                f"case from existing configuration {conflicting!r}."
            )
        configuration = {
            str(key): str(value)
            for key, value in values.items()
            if value is not None
        }
        manifest = self.project_manifest()
        connectors = _object_table(manifest["connectors"], field="connectors")
        project_configurations = _object_table(
            connectors.get("configurations") or {},
            field="connectors.configurations",
        )
        project_configurations[normalized] = {
            key: value
            for key, value in configuration.items()
            if key not in _STALE_CHECK_FIELDS
        }
        connectors["configurations"] = project_configurations
        self._write_project_configuration(connectors=connectors)
        return self.connector_configurations()[normalized]

    def remove_connector_configuration(self, name: str) -> None:
        """Remove one unused named connector configuration."""

        normalized = name.strip()
        configurations = self.connector_configurations()
        if normalized not in configurations:
            raise WorkspaceError(
                f"Connector configuration does not exist: {normalized}"
            )
        workflow = self.resolve_workflow()
        assignments = self.connector_assignment_profile(workflow)
        bindings = self.connector_binding_profile(workflow)
        references = [
            target
            for group in ("lifelines", "actions")
            for target, selected in assignments[group].items()
            if selected == normalized
        ]
        references.extend(
            f"requirement {requirement}"
            for requirement, selected in bindings.items()
            if selected == normalized
        )
        if references:
            raise WorkspaceError(
                f"Connector configuration {normalized!r} is still used by: "
                + ", ".join(references)
                + ". Unassign or unbind it first."
            )
        manifest = self.project_manifest()
        connectors = _object_table(manifest["connectors"], field="connectors")
        project = _object_table(
            connectors.get("configurations") or {},
            field="connectors.configurations",
        )
        project.pop(normalized, None)
        connectors["configurations"] = project
        self._write_project_configuration(connectors=connectors)
        state = self.load()
        updates: dict[str, object] = {}
        for key in (
            "connector_configuration_overrides",
        ):
            raw = state.get(key) or {}
            if isinstance(raw, dict):
                values = dict(raw)
                values.pop(normalized, None)
                updates[key] = values
        if updates:
            self.update(**updates)

    def connector_binding_profile(
        self,
        workflow_spec: str,
    ) -> dict[str, str]:
        """Return requirement-to-configuration bindings for one workflow."""

        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        state = self.load()
        manifest_connectors = self.project_manifest().get("connectors") or {}
        assert isinstance(manifest_connectors, dict)
        project = manifest_connectors.get("bindings") or {}
        assert isinstance(project, dict)
        result = {str(name): str(value) for name, value in project.items()}
        raw_site = state.get("connector_site_bindings") or {}
        if not isinstance(raw_site, dict):
            raise WorkspaceError(
                "Workspace connector_site_bindings must be an object."
            )
        site = raw_site.get(canonical) or {}
        if not isinstance(site, dict):
            raise WorkspaceError(
                f"Connector site bindings for {canonical!r} must be an object."
            )
        result.update({str(name): str(value) for name, value in site.items()})
        return result

    def bind_connector(
        self,
        workflow_spec: str,
        requirement: str,
        configuration: str,
    ) -> dict[str, str]:
        """Bind one logical workflow requirement to a named configuration."""

        configurations = self.connector_configurations()
        if configuration not in configurations:
            raise WorkspaceError(
                f"Connector configuration does not exist: {configuration}."
            )
        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        manifest = self.project_manifest()
        connectors = _object_table(manifest["connectors"], field="connectors")
        profile = {
            str(name): str(value)
            for name, value in _object_table(
                connectors.get("bindings") or {},
                field="connectors.bindings",
            ).items()
        }
        profile[str(requirement)] = configuration
        connectors["bindings"] = profile
        self._write_project_configuration(connectors=connectors)
        return profile

    def unbind_connector(
        self,
        workflow_spec: str,
        requirement: str,
    ) -> dict[str, str]:
        """Remove one requirement binding from the portable project profile."""

        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        effective = self.connector_binding_profile(canonical)
        if requirement not in effective:
            raise WorkspaceError(
                f"Connector requirement is not bound: {requirement}."
            )
        manifest = self.project_manifest()
        connectors = _object_table(manifest["connectors"], field="connectors")
        profile = {
            str(name): str(value)
            for name, value in _object_table(
                connectors.get("bindings") or {},
                field="connectors.bindings",
            ).items()
        }
        profile.pop(requirement, None)
        connectors["bindings"] = profile
        self._write_project_configuration(connectors=connectors)
        state = self.load()
        raw_site = state.get("connector_site_bindings") or {}
        if isinstance(raw_site, dict):
            site_profiles = dict(raw_site)
            raw_site_profile = site_profiles.get(canonical)
            if isinstance(raw_site_profile, dict):
                site_profile = dict(raw_site_profile)
                site_profile.pop(requirement, None)
                if site_profile:
                    site_profiles[canonical] = site_profile
                else:
                    site_profiles.pop(canonical, None)
                self.update(connector_site_bindings=site_profiles)
        return profile

    def connector_assignment_profile(
        self,
        workflow_spec: str,
    ) -> dict[str, dict[str, str]]:
        """Return participant and action connector assignments."""

        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        state = self.load()
        manifest_connectors = self.project_manifest().get("connectors") or {}
        assert isinstance(manifest_connectors, dict)
        project = manifest_connectors.get("assignments") or {}
        assert isinstance(project, dict)
        result = {
            "lifelines": {
                str(name): str(configuration)
                for name, configuration in dict(
                    project.get("lifelines") or {}
                ).items()
            },
            "actions": {
                str(name): str(configuration)
                for name, configuration in dict(
                    project.get("actions") or {}
                ).items()
            },
        }
        raw_site = state.get("connector_site_assignments") or {}
        if not isinstance(raw_site, dict):
            raise WorkspaceError(
                "Workspace connector_site_assignments must be an object."
            )
        site = raw_site.get(canonical) or {}
        if not isinstance(site, dict):
            raise WorkspaceError(
                f"Connector site assignments for {canonical!r} are malformed."
            )
        for key in ("lifelines", "actions"):
            values = site.get(key) or {}
            if not isinstance(values, dict):
                raise WorkspaceError(
                    f"Connector site {key} for {canonical!r} are malformed."
                )
            result[key].update(
                {
                    str(name): str(configuration)
                    for name, configuration in values.items()
                }
            )
        return result

    def save_connector_assignment_profile(
        self,
        workflow_spec: str,
        *,
        lifelines: dict[str, str],
        actions: dict[str, str] | None = None,
        site: bool = False,
    ) -> dict[str, dict[str, str]]:
        """Persist reusable configuration routes for human actions."""

        action_assignments = dict(actions or {})
        configurations = self.connector_configurations()
        missing = sorted(
            {
                *lifelines.values(),
                *action_assignments.values(),
            }
            - set(configurations)
        )
        if missing:
            raise WorkspaceError(
                "Unknown connector configuration(s): " + ", ".join(missing)
            )
        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        state = self.load()
        profile = {
            "lifelines": {
                str(name): str(configuration)
                for name, configuration in sorted(lifelines.items())
            },
            "actions": {
                str(name): str(configuration)
                for name, configuration in sorted(action_assignments.items())
            },
        }
        if site:
            raw_site = state.get("connector_site_assignments") or {}
            if not isinstance(raw_site, dict):
                raise WorkspaceError(
                    "Workspace connector_site_assignments must be an object."
                )
            site_profiles = dict(raw_site)
            project = self.connector_assignment_profile(canonical)
            raw_existing_site = raw_site.get(canonical) or {}
            if isinstance(raw_existing_site, dict):
                for key in ("lifelines", "actions"):
                    existing_values = raw_existing_site.get(key) or {}
                    if isinstance(existing_values, dict):
                        for target in existing_values:
                            project[key].pop(str(target), None)
            site_profile = {
                key: {
                    target: configuration
                    for target, configuration in profile[key].items()
                    if project[key].get(target) != configuration
                }
                for key in ("lifelines", "actions")
            }
            if any(site_profile.values()):
                site_profiles[canonical] = site_profile
            else:
                site_profiles.pop(canonical, None)
            self.update(connector_site_assignments=site_profiles)
            return profile
        manifest = self.project_manifest()
        connectors = _object_table(manifest["connectors"], field="connectors")
        connectors["assignments"] = profile
        self._write_project_configuration(connectors=connectors)
        return profile

    @staticmethod
    def connector_secret_name(configuration: str, field: str) -> str:
        """Return an internal private-store key; never an environment name."""

        return f"connector:{configuration}:{field}"

    def connector_secret(
        self,
        configuration: str,
        field: str,
    ) -> str | None:
        return self.load_secrets().get(
            self.connector_secret_name(configuration, field)
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

    def development_credential(self, name: str) -> str | None:
        """Read one private credential used by development commands."""

        return self.load_secrets().get(name)

    def save_development_credential(self, name: str, value: str) -> None:
        """Save one development credential without changing project files."""

        secrets = self.load_secrets()
        secrets[name] = value
        self.save_secrets(secrets)
