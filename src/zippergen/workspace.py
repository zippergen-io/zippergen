"""Project-aware state for the prompt-first ZipperGen development experience.

Visible project intent lives in ``zippergen.toml`` and an ordered ``prompts/``
ledger so it can be reviewed and versioned. Machine-specific workspace state
and its separate owner-only secret file stay below ``ZIPPERGEN_HOME`` rather
than in the user's Git checkout; the ordinary workspace record is non-secret.
The regular CLI
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
import tomllib
from pathlib import Path
from typing import Any


WORKSPACE_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
PROJECT_SCHEMA_VERSION = 1
PROMPT_LEDGER_SCHEMA_VERSION = 1
PROJECT_MANIFEST_NAME = "zippergen.toml"
PROMPT_INDEX_NAME = "index.toml"
PROJECT_TASK_DIRECTORY = ".zippergen"
CURRENT_TASK_NAME = "current-task.md"

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


def _prompt_title(content: str) -> str:
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        title = line.lstrip("#").strip() if line.startswith("#") else line
        if title:
            return title[:100]
    return "Untitled prompt"


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
        self.directory = self.home / "workspaces" / _workspace_key(self.root)
        self.state_path = self.directory / "workspace.json"
        self.secrets_path = self.directory / "development.secrets.json"
        self.runs_directory = self.directory / "runs"
        self.requests_directory = self.directory / "requests"

    @property
    def current_task_path(self) -> Path:
        """Return the stable, project-local coding-assistant handoff path."""

        return self.root / PROJECT_TASK_DIRECTORY / CURRENT_TASK_NAME

    @property
    def manifest_path(self) -> Path:
        return self.root / PROJECT_MANIFEST_NAME

    def project_manifest(self) -> dict[str, object]:
        """Load visible project configuration, or return non-writing defaults."""

        if not self.manifest_path.exists():
            return {
                "schema_version": PROJECT_SCHEMA_VERSION,
                "name": self.root.name,
                "prompts_directory": "prompts",
                "framework_directory": None,
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
        prompts = str(manifest.get("prompts_directory") or "prompts")
        _safe_project_directory(
            self.root,
            prompts,
            field="prompts_directory",
        )
        framework_value = manifest.get("framework_directory")
        framework = str(framework_value).strip() if framework_value else None
        if framework:
            _safe_project_directory(
                self.root,
                framework,
                field="framework_directory",
            )
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "name": name,
            "prompts_directory": prompts,
            "framework_directory": framework,
            "exists": True,
        }

    def initialize_project(
        self,
        *,
        name: str | None = None,
        prompts_directory: str = "prompts",
        framework_directory: str | None = None,
    ) -> dict[str, object]:
        """Create the visible project manifest and empty prompt ledger."""

        if self.manifest_path.exists():
            manifest = self.project_manifest()
            self._ensure_project_gitignore(
                str(manifest["framework_directory"])
                if manifest.get("framework_directory")
                else None
            )
            self._ensure_prompt_index()
            return manifest
        project_name = str(name or self.root.name).strip()
        if not project_name:
            raise WorkspaceError("Project name must not be empty.")
        _safe_project_directory(
            self.root,
            prompts_directory,
            field="prompts_directory",
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
            f"name = {_toml_string(project_name)}\n"
            f"prompts_directory = {_toml_string(prompts_directory)}\n"
        )
        if framework_directory:
            content += (
                f"framework_directory = {_toml_string(framework_directory)}\n"
            )
        _atomic_write_text(self.manifest_path, content)
        self._ensure_project_gitignore(framework_directory)
        self._ensure_prompt_index()
        return self.project_manifest()

    def _ensure_project_gitignore(self, framework_directory: str | None) -> None:
        if (self.root / ".git").exists():
            gitignore = self.root / ".gitignore"
            existing = (
                gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
            )
            desired = [f"/{PROJECT_TASK_DIRECTORY}/", "/tutorial-runtime/"]
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
                    + "# ZipperGen project-local tooling and transparent runtime\n"
                    + "\n".join(missing)
                    + "\n",
                )

    @property
    def prompts_directory(self) -> Path:
        manifest = self.project_manifest()
        return _safe_project_directory(
            self.root,
            manifest["prompts_directory"],
            field="prompts_directory",
        )

    @property
    def prompt_index_path(self) -> Path:
        return self.prompts_directory / PROMPT_INDEX_NAME

    def _ensure_prompt_index(self) -> None:
        path = self.prompt_index_path
        if path.exists():
            self.list_prompts()
            return
        content = (
            "# Ordered ZipperGen design intent. Managed by Studio; safe to review.\n"
            f"schema_version = {PROMPT_LEDGER_SCHEMA_VERSION}\n"
        )
        _atomic_write_text(path, content)

    def _prompt_file(self, value: object) -> Path:
        raw = str(value).strip()
        relative = Path(raw)
        if (
            not raw
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.name == PROMPT_INDEX_NAME
        ):
            raise WorkspaceError(f"Invalid prompt ledger file: {raw!r}.")
        path = (self.prompts_directory / relative).resolve()
        if not path.is_relative_to(self.prompts_directory):
            raise WorkspaceError(f"Prompt file escapes the prompt directory: {raw!r}.")
        return path

    def list_prompts(self) -> list[dict[str, object]]:
        """Return the ordered, visible project prompt ledger."""

        path = self.prompt_index_path
        if not path.exists():
            return []
        try:
            index = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise WorkspaceError(f"Invalid prompt index {path}: {exc}") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise WorkspaceError(f"Could not read prompt index {path}: {exc}") from exc
        if index.get("schema_version") != PROMPT_LEDGER_SCHEMA_VERSION:
            raise WorkspaceError(
                f"Unsupported prompt ledger schema in {path}: "
                f"{index.get('schema_version')!r}"
            )
        raw_prompts = index.get("prompts") or []
        if not isinstance(raw_prompts, list):
            raise WorkspaceError(f"Prompt entries must be a list in {path}.")
        records: list[dict[str, object]] = []
        seen: set[str] = set()
        for raw in raw_prompts:
            if not isinstance(raw, dict):
                raise WorkspaceError(f"Every prompt entry in {path} must be a table.")
            prompt_id = str(raw.get("id") or "").upper()
            if not re.fullmatch(r"P[0-9]{3,}", prompt_id) or prompt_id in seen:
                raise WorkspaceError(f"Invalid or duplicate prompt id {prompt_id!r}.")
            seen.add(prompt_id)
            kind = str(raw.get("kind") or "")
            if kind not in {"initial", "refinement"}:
                raise WorkspaceError(
                    f"Prompt {prompt_id} has invalid kind {kind!r}; expected "
                    "'initial' or 'refinement'."
                )
            prompt_file = self._prompt_file(raw.get("file"))
            try:
                content = prompt_file.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                raise WorkspaceError(
                    f"Prompt {prompt_id} file does not exist: {prompt_file}"
                ) from None
            except (OSError, UnicodeDecodeError) as exc:
                raise WorkspaceError(
                    f"Could not read prompt {prompt_id} file {prompt_file}: {exc}"
                ) from exc
            if not content:
                raise WorkspaceError(f"Prompt {prompt_id} file is empty: {prompt_file}")
            record: dict[str, object] = {
                "id": prompt_id,
                "kind": kind,
                "file": prompt_file.relative_to(self.root).as_posix(),
                "ledger_file": prompt_file.relative_to(
                    self.prompts_directory
                ).as_posix(),
                "active": bool(raw.get("active", True)),
                "created_at": str(raw.get("created_at") or ""),
                "workflow_spec": str(raw.get("workflow_spec") or "") or None,
                "replaces": str(raw.get("replaces") or "") or None,
                "title": _prompt_title(content),
                "content": content,
            }
            records.append(record)
        return records

    def _write_prompt_index(self, records: list[dict[str, object]]) -> None:
        lines = [
            "# Ordered ZipperGen design intent. Managed by Studio; safe to review.",
            f"schema_version = {PROMPT_LEDGER_SCHEMA_VERSION}",
        ]
        for record in records:
            lines.extend(
                [
                    "",
                    "[[prompts]]",
                    f"id = {_toml_string(record['id'])}",
                    f"file = {_toml_string(record['ledger_file'])}",
                    f"kind = {_toml_string(record['kind'])}",
                    f"active = {'true' if record.get('active', True) else 'false'}",
                    f"created_at = {_toml_string(record.get('created_at') or '')}",
                ]
            )
            if record.get("workflow_spec"):
                lines.append(
                    f"workflow_spec = {_toml_string(record['workflow_spec'])}"
                )
            if record.get("replaces"):
                lines.append(f"replaces = {_toml_string(record['replaces'])}")
        _atomic_write_text(self.prompt_index_path, "\n".join(lines) + "\n")

    def _next_prompt_id(self, records: list[dict[str, object]]) -> str:
        used = [int(str(record["id"])[1:]) for record in records]
        return f"P{max(used, default=0) + 1:03d}"

    def add_prompt(
        self,
        *,
        kind: str,
        content: str,
        source_path: str | Path | None = None,
        workflow_spec: str | None = None,
        replaces: str | None = None,
        reuse_project_file: bool = True,
    ) -> dict[str, object]:
        """Register design intent and keep its source in the project prompt ledger."""

        if kind not in {"initial", "refinement"}:
            raise WorkspaceError(f"Unsupported prompt kind: {kind!r}.")
        prompt = content.strip()
        if not prompt:
            raise WorkspaceError("Prompt content must not be empty.")
        self.initialize_project()
        records = self.list_prompts()
        prompt_id = self._next_prompt_id(records)
        source = Path(source_path).expanduser().resolve() if source_path else None
        target: Path | None = None
        if source is not None and reuse_project_file:
            if source.is_relative_to(self.prompts_directory) and source.name != PROMPT_INDEX_NAME:
                target = source
                relative = source.relative_to(self.prompts_directory).as_posix()
                for record in records:
                    if record["ledger_file"] != relative:
                        continue
                    if record["kind"] != kind:
                        raise WorkspaceError(
                            f"{source} is already registered as {record['id']} "
                            f"with kind {record['kind']}."
                        )
                    if not record["active"]:
                        record["active"] = True
                        self._write_prompt_index(records)
                    return {**record, "created": False}
        if target is None:
            filename = f"{prompt_id[1:]}-{_slug(_prompt_title(prompt)).lower()}.md"
            target = self.prompts_directory / filename
            suffix = 2
            while target.exists():
                target = self.prompts_directory / (
                    f"{prompt_id[1:]}-{_slug(_prompt_title(prompt)).lower()}-{suffix}.md"
                )
                suffix += 1
            _atomic_write_text(target, prompt.rstrip() + "\n")
        elif target.read_text(encoding="utf-8").strip() != prompt:
            raise WorkspaceError(
                f"Prompt source changed while it was being registered: {target}"
            )
        record: dict[str, object] = {
            "id": prompt_id,
            "kind": kind,
            "file": target.relative_to(self.root).as_posix(),
            "ledger_file": target.relative_to(self.prompts_directory).as_posix(),
            "active": True,
            "created_at": _timestamp(),
            "workflow_spec": workflow_spec,
            "replaces": replaces,
            "title": _prompt_title(prompt),
            "content": prompt,
        }
        records.append(record)
        self._write_prompt_index(records)
        return {**record, "created": True}

    def _resolve_prompt(self, prompt_id: str) -> tuple[int, dict[str, object]]:
        entered = prompt_id.strip().upper()
        if entered.isdigit():
            entered = f"P{int(entered):03d}"
        records = self.list_prompts()
        for index, record in enumerate(records):
            if record["id"] == entered:
                return index, record
        available = ", ".join(str(record["id"]) for record in records) or "none"
        raise WorkspaceError(f"Unknown prompt {prompt_id!r}. Available: {available}.")

    def prompt(self, prompt_id: str) -> dict[str, object]:
        """Return one prompt ledger entry by stable id or numeric shorthand."""

        _index, record = self._resolve_prompt(prompt_id)
        return record

    def set_prompt_active(self, prompt_id: str, *, active: bool) -> dict[str, object]:
        records = self.list_prompts()
        index, _record = self._resolve_prompt(prompt_id)
        records[index]["active"] = active
        self._write_prompt_index(records)
        return records[index]

    def update_prompt_content(
        self,
        prompt_id: str,
        *,
        content: str,
    ) -> dict[str, object]:
        """Update one prompt in place while preserving its stable identity."""

        prompt = content.strip()
        if not prompt:
            raise WorkspaceError("Prompt content must not be empty.")
        records = self.list_prompts()
        index, _record = self._resolve_prompt(prompt_id)
        target = self._prompt_file(records[index]["ledger_file"])
        _atomic_write_text(target, prompt.rstrip() + "\n")
        records[index]["content"] = prompt
        records[index]["title"] = _prompt_title(prompt)
        return records[index]

    def move_prompt(
        self,
        prompt_id: str,
        *,
        relation: str,
        other_id: str,
    ) -> list[dict[str, object]]:
        if relation not in {"before", "after"}:
            raise WorkspaceError("Prompt move relation must be 'before' or 'after'.")
        records = self.list_prompts()
        source_index, source = self._resolve_prompt(prompt_id)
        _other_index, other = self._resolve_prompt(other_id)
        if source["id"] == other["id"]:
            raise WorkspaceError("A prompt cannot be moved relative to itself.")
        records.pop(source_index)
        target_index = next(
            index
            for index, record in enumerate(records)
            if record["id"] == other["id"]
        )
        if relation == "after":
            target_index += 1
        records.insert(target_index, source)
        self._write_prompt_index(records)
        return records

    def replace_prompt(
        self,
        prompt_id: str,
        *,
        content: str,
        source_path: str | Path | None = None,
    ) -> dict[str, object]:
        records = self.list_prompts()
        old_index, old = self._resolve_prompt(prompt_id)
        reuse_project_file = True
        if source_path is not None:
            source = Path(source_path).expanduser().resolve()
            if source.is_relative_to(self.prompts_directory):
                relative = source.relative_to(self.prompts_directory).as_posix()
                reuse_project_file = all(
                    record["ledger_file"] != relative for record in records
                )
        replacement = self.add_prompt(
            kind=str(old["kind"]),
            content=content,
            source_path=source_path,
            workflow_spec=(
                str(old["workflow_spec"]) if old.get("workflow_spec") else None
            ),
            replaces=str(old["id"]),
            reuse_project_file=reuse_project_file,
        )
        records = self.list_prompts()
        replacement_index = next(
            index
            for index, record in enumerate(records)
            if record["id"] == replacement["id"]
        )
        replacement_record = records.pop(replacement_index)
        old_index = next(
            index
            for index, record in enumerate(records)
            if record["id"] == old["id"]
        )
        records[old_index]["active"] = False
        records.insert(old_index + 1, replacement_record)
        self._write_prompt_index(records)
        return replacement_record

    def prompt_context(self) -> str:
        """Render all active design intent in its explicit precedence order."""

        manifest = self.project_manifest()
        active = [record for record in self.list_prompts() if record["active"]]
        lines = [
            f"# ZipperGen design context: {manifest['name']}",
            "",
            "Read active prompts in the order shown. Later prompts take precedence "
            "only where they explicitly change or contradict an earlier requirement; "
            "all other earlier requirements remain in force.",
        ]
        if not active:
            lines.extend(["", "No active prompts."])
            return "\n".join(lines)
        for record in active:
            lines.extend(
                [
                    "",
                    f"## {record['id']} [{record['kind']}] — {record['title']}",
                    f"Source: {record['file']}",
                    "",
                    str(record["content"]),
                ]
            )
        return "\n".join(lines)

    def default_state(self) -> dict[str, Any]:
        return {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "project_root": str(self.root),
            "current_workflow": None,
            "current_run": None,
            "last_deployment": None,
            "last_view": "protocol",
            "current_request": None,
            "editor_command": None,
            "model_profiles": {},
            "providers": {},
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
        # This field was added additively so existing workspaces keep working
        # without a schema migration.
        state.setdefault("current_request", None)
        state.setdefault("editor_command", None)
        state.setdefault("model_profiles", {})
        state.setdefault("providers", {})
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

    def model_profile(
        self,
        workflow_spec: str | None = None,
        *,
        default: str = "mock",
    ) -> dict[str, Any]:
        """Return the persistent model profile for one workflow."""

        selected = workflow_spec or self.current_workflow
        if not selected:
            raise WorkspaceError("No workflow is selected.")
        canonical = self.canonical_spec(selected, cwd=self.root)
        raw_profiles = self.load().get("model_profiles") or {}
        if not isinstance(raw_profiles, dict):
            raise WorkspaceError("Workspace model_profiles must be an object.")
        raw_profile = raw_profiles.get(canonical) or {}
        if not isinstance(raw_profile, dict):
            raise WorkspaceError(f"Model profile for {canonical} must be an object.")
        raw_lifelines = raw_profile.get("lifelines") or {}
        if not isinstance(raw_lifelines, dict):
            raise WorkspaceError(
                f"Model lifeline overrides for {canonical} must be an object."
            )
        return {
            "default": str(raw_profile.get("default") or default),
            "lifelines": {
                str(name): str(spec) for name, spec in raw_lifelines.items()
            },
        }

    def save_model_profile(
        self,
        workflow_spec: str,
        *,
        default: str,
        lifelines: dict[str, str],
    ) -> dict[str, Any]:
        """Persist a non-secret default model and explicit lifeline overrides."""

        canonical = self.canonical_spec(workflow_spec, cwd=self.root)
        state = self.load()
        raw_profiles = state.get("model_profiles") or {}
        if not isinstance(raw_profiles, dict):
            raise WorkspaceError("Workspace model_profiles must be an object.")
        profiles = dict(raw_profiles)
        profile = {
            "default": str(default),
            "lifelines": {
                str(name): str(spec) for name, spec in sorted(lifelines.items())
            },
        }
        profiles[canonical] = profile
        self.update(model_profiles=profiles)
        return profile

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
        profiles = self.provider_profiles()
        profiles[name] = {
            str(key): str(value) for key, value in values.items() if value is not None
        }
        self.update(providers=profiles)
        return profiles[name]

    def remove_provider_profile(self, name: str) -> None:
        profiles = self.provider_profiles()
        profiles.pop(name, None)
        self.update(providers=profiles)

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
            if secret_name and secrets.get(secret_name):
                environment[secret_name] = secrets[secret_name]
            if provider == "local":
                base_url = profiles.get("local", {}).get("base_url")
                if base_url:
                    environment["OLLAMA_BASE_URL"] = base_url
        return environment

    def discover_workflows(self) -> list[str]:
        framework = self.project_manifest().get("framework_directory")
        ignored = (str(framework),) if framework else ()
        return discover_workflow_specs(self.root, ignored_directories=ignored)

    def new_run(
        self,
        *,
        workflow_spec: str,
        workflow_name: str,
        fingerprint: str,
        inputs: dict[str, object],
        llm: str,
        llms: dict[str, str] | None = None,
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
            "llms": dict(llms or {}),
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
        prompt_id: str | None = None,
        active_prompt_ids: tuple[str, ...] = (),
        baseline_file: str | Path | None = None,
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
            "prompt_id": prompt_id,
            "active_prompt_ids": list(active_prompt_ids),
            "baseline_file": str(baseline_file) if baseline_file else None,
            "prompt": prompt,
            "content_file": str(self.requests_directory / f"{request_id}.md"),
            "task_file": str(self.current_task_path),
            "created_at": _timestamp(),
        }
        self.requests_directory.mkdir(parents=True, exist_ok=True)
        task_content = content.rstrip() + "\n"
        _atomic_write_text(Path(record["content_file"]), task_content)
        _atomic_write_json(self.requests_directory / f"{request_id}.json", record)
        _atomic_write_text(self.current_task_path, task_content)
        self.update(current_request=request_id)
        return record

    def request_path(self, request_id: str) -> Path:
        return self.requests_directory / f"{request_id}.json"

    def load_request(self, request_id: str) -> dict[str, Any]:
        record = _read_json(self.request_path(request_id))
        if record.get("schema_version") != REQUEST_SCHEMA_VERSION:
            raise WorkspaceError(
                f"Unsupported assistant request schema in "
                f"{self.request_path(request_id)}: "
                f"{record.get('schema_version')!r}"
            )
        if Path(str(record.get("project_root"))).resolve() != self.root:
            raise WorkspaceError(
                f"Assistant request {request_id} belongs to another project root."
            )
        return record

    def list_requests(self) -> list[dict[str, Any]]:
        """Return archived coding-assistant requests, newest first."""

        if not self.requests_directory.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in self.requests_directory.glob("*.json"):
            try:
                records.append(self.load_request(path.stem))
            except WorkspaceError:
                continue
        return sorted(
            records,
            key=lambda record: (
                str(record.get("created_at") or ""),
                str(record.get("request_id") or ""),
            ),
            reverse=True,
        )

    def current_request(self) -> dict[str, Any] | None:
        """Return the current handoff and restore its stable mirror if needed."""

        request_id = self.load().get("current_request")
        record: dict[str, Any] | None = None
        if request_id:
            try:
                record = self.load_request(str(request_id))
            except WorkspaceError:
                record = None
        if record is None:
            requests = self.list_requests()
            record = requests[0] if requests else None
        if record is None:
            return None
        content_file = Path(str(record.get("content_file") or ""))
        try:
            content = content_file.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
            raise WorkspaceError(
                f"Could not restore current task from {content_file}: {exc}"
            ) from exc
        if not self.current_task_path.exists() or (
            self.current_task_path.read_text(encoding="utf-8") != content
        ):
            _atomic_write_text(self.current_task_path, content)
        if request_id != record.get("request_id"):
            self.update(current_request=str(record["request_id"]))
        return {**record, "task_file": str(self.current_task_path)}
