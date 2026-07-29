"""Project-aware discovery and safe management of durable SQLite stores."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zippergen.store import open_store
from zippergen.workspace import Workspace, WorkspaceError


@dataclass(frozen=True)
class StoreRecord:
    name: str
    path: Path
    owners: tuple[str, ...]
    deployment_names: tuple[str, ...]
    run_ids: tuple[str, ...]
    workflows: tuple[str, ...]
    exists: bool
    state: str
    summary: str
    pending_tasks: int
    size: int
    updated_at: float | None
    project_owned: bool


def store_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    if not name:
        raise WorkspaceError("A store name must contain a letter or number.")
    return name


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve()))


def deployment_profiles(workspace: Workspace) -> list[tuple[Path, dict[str, object]]]:
    directory = workspace.home / "deployments"
    if not directory.exists():
        return []
    profiles: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".secrets.json"):
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(value, dict)
            and value.get("name")
            and value.get("store")
            and _profile_belongs_to_project(workspace, value)
        ):
            profiles.append((path, value))
    return profiles


def _profile_belongs_to_project(
    workspace: Workspace,
    profile: dict[str, object],
) -> bool:
    root = str(workspace.root)
    if str(profile.get("project_root") or "") == root:
        return True
    if str(profile.get("name") or "") == str(
        workspace.load().get("last_deployment") or ""
    ):
        return True
    for key in ("source_cwd", "cwd"):
        raw = profile.get(key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser().resolve()
        if path == workspace.root or path.is_relative_to(workspace.directory):
            return True
    return False


def discover_stores(workspace: Workspace) -> list[StoreRecord]:
    """Discover run, deployment, standalone, and expected-but-missing stores."""

    entries: dict[str, dict[str, Any]] = {}

    def include(path_value: object, *, project_owned: bool = False) -> dict[str, Any]:
        path = Path(str(path_value)).expanduser().resolve()
        key = _path_key(path)
        entry = entries.setdefault(
            key,
            {
                "path": path,
                "owners": set(),
                "deployment_names": set(),
                "run_ids": set(),
                "workflows": set(),
                "project_owned": False,
            },
        )
        entry["project_owned"] = bool(entry["project_owned"] or project_owned)
        return entry

    for run in workspace.list_runs():
        store = run.get("store")
        if not store:
            continue
        entry = include(store, project_owned=True)
        run_id = str(run.get("run_id") or "")
        if run_id:
            entry["run_ids"].add(run_id)
            entry["owners"].add(f"run {run_id}")
        workflow = str(run.get("workflow_spec") or "")
        if workflow:
            entry["workflows"].add(workflow)

    for _profile_path, profile in deployment_profiles(workspace):
        project_owned = _profile_belongs_to_project(workspace, profile)
        entry = include(profile["store"], project_owned=project_owned)
        name = str(profile["name"])
        entry["deployment_names"].add(name)
        entry["owners"].add(f"deployment {name}")
        workflow = str(profile.get("workflow") or "")
        if workflow:
            entry["workflows"].add(workflow)

    for directory, project_owned in ((workspace.runs_directory, True),):
        if not directory.exists():
            continue
        for path in directory.glob("*.sqlite"):
            include(path, project_owned=project_owned)

    from zippergen.serve import _store_status

    records: list[StoreRecord] = []
    for entry in entries.values():
        path = entry["path"]
        exists = path.is_file()
        try:
            status = _store_status(str(path))
            state = str(status["state"])
            summary = str(status["summary"])
            raw_pending = status.get("pending_human_tasks")
            pending = len(raw_pending) if isinstance(raw_pending, list) else 0
        except Exception as exc:
            state = "invalid"
            summary = f"{type(exc).__name__}: {exc}"
            pending = 0
        stat = path.stat() if exists else None
        records.append(
            StoreRecord(
                name=path.stem,
                path=path,
                owners=tuple(sorted(entry["owners"])) or ("standalone",),
                deployment_names=tuple(sorted(entry["deployment_names"])),
                run_ids=tuple(sorted(entry["run_ids"])),
                workflows=tuple(sorted(entry["workflows"])),
                exists=exists,
                state=state,
                summary=summary,
                pending_tasks=pending,
                size=stat.st_size if stat else 0,
                updated_at=stat.st_mtime if stat else None,
                project_owned=bool(entry["project_owned"]),
            )
        )
    return sorted(
        records,
        key=lambda record: (
            not record.project_owned,
            -float(record.updated_at or 0),
            record.name.casefold(),
            str(record.path),
        ),
    )


def resolve_store(
    workspace: Workspace,
    selector: str | None,
) -> StoreRecord:
    records = discover_stores(workspace)
    if not records:
        raise WorkspaceError(
            "No durable state is known. Start a development run or prepare a "
            "deployment first."
        )
    if selector:
        if selector.isdigit():
            index = int(selector)
            if 1 <= index <= len(records):
                return records[index - 1]
            raise WorkspaceError(
                f"Store number must be between 1 and {len(records)}."
            )
        folded = selector.casefold()
        matches = [
            record
            for record in records
            if folded
            in {
                record.name.casefold(),
                str(record.path).casefold(),
                *(value.casefold() for value in record.deployment_names),
                *(value.casefold() for value in record.run_ids),
            }
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise WorkspaceError(
                f"Store name {selector!r} is ambiguous; use its list number or path."
            )
        raise WorkspaceError(
            f"Durable state not found: {selector}. Use 'runs' or "
            "'deploy list' to inspect its owner."
        )

    current_run = workspace.current_run()
    if current_run and current_run.get("store"):
        key = _path_key(Path(str(current_run["store"])))
        match = next(
            (record for record in records if _path_key(record.path) == key),
            None,
        )
        if match is not None:
            return match
    remembered = workspace.load().get("last_deployment")
    if remembered:
        match = next(
            (
                record
                for record in records
                if str(remembered) in record.deployment_names
            ),
            None,
        )
        if match is not None:
            return match
    if len(records) == 1:
        return records[0]
    raise WorkspaceError(
        "No current run or deployment owns the selected durable state."
    )


def create_store(workspace: Workspace, name: str) -> StoreRecord:
    normalized = store_name(name)
    path = (workspace.runs_directory / f"{normalized}.sqlite").resolve()
    if path.exists():
        raise WorkspaceError(f"Store already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = open_store(str(path))
    connection.close()
    return resolve_store(workspace, str(path))


def active_store_deployments(record: StoreRecord) -> tuple[str, ...]:
    from zippergen.serve import _deployment_service_status

    active = []
    for name in record.deployment_names:
        status = _deployment_service_status(name)
        if status.get("state") not in {"not-loaded", "completed"}:
            active.append(name)
    return tuple(active)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
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


def _move_sqlite_family(source: Path, destination: Path) -> list[tuple[Path, Path]]:
    moved: list[tuple[Path, Path]] = []
    for suffix in ("", "-wal", "-shm"):
        old = Path(str(source) + suffix)
        new = Path(str(destination) + suffix)
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            old.replace(new)
            moved.append((old, new))
    return moved


def rename_store(
    workspace: Workspace,
    record: StoreRecord,
    new_name: str,
) -> StoreRecord:
    active = active_store_deployments(record)
    if active:
        raise WorkspaceError(
            "Stop the deployment before renaming its store: " + ", ".join(active)
        )
    normalized = store_name(new_name)
    destination = record.path.with_name(f"{normalized}.sqlite")
    if destination.exists() and destination != record.path:
        raise WorkspaceError(f"Store already exists: {destination}")

    affected_runs = [
        run for run in workspace.list_runs()
        if run.get("store")
        and _path_key(Path(str(run["store"]))) == _path_key(record.path)
    ]
    affected_profiles = [
        (path, profile)
        for path, profile in deployment_profiles(workspace)
        if _path_key(Path(str(profile["store"]))) == _path_key(record.path)
    ]
    moved: list[tuple[Path, Path]] = []
    try:
        moved = _move_sqlite_family(record.path, destination)
        for run in affected_runs:
            workspace.update_run(str(run["run_id"]), store=str(destination))
        for path, profile in affected_profiles:
            profile["store"] = str(destination)
            _atomic_json(path, profile)
    except Exception:
        for old, new in reversed(moved):
            if new.exists() and not old.exists():
                new.replace(old)
        for run in affected_runs:
            workspace.update_run(str(run["run_id"]), store=str(record.path))
        for path, profile in affected_profiles:
            profile["store"] = str(record.path)
            _atomic_json(path, profile)
        raise
    return resolve_store(workspace, str(destination))


def archive_store(workspace: Workspace, record: StoreRecord) -> Path | None:
    active = active_store_deployments(record)
    if active:
        raise WorkspaceError(
            "Stop the deployment before deleting its store: " + ", ".join(active)
        )
    if not record.path.exists():
        return None
    destination = (
        workspace.home
        / "trash"
        / "stores"
        / time.strftime("%Y%m%d-%H%M%S")
        / record.path.name
    )
    suffix = 2
    while destination.exists():
        destination = destination.with_name(
            f"{record.path.stem}-{suffix}{record.path.suffix}"
        )
        suffix += 1
    _move_sqlite_family(record.path, destination)
    return destination
