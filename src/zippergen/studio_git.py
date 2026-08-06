"""Narrow Git support for Studio's portable implementation unit.

Studio does not manage branches, remotes, merges, or history.  This module
only answers whether the files that define one implementation differ from a
clone, and commits exactly those files when the user explicitly asks.
"""

from __future__ import annotations

import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from shutil import which as _which
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from zippergen.workspace import Workspace

_run_process = subprocess.run


@dataclass(frozen=True)
class GitCommitUnit:
    """The project files that must travel together for one implementation."""

    executable: str
    repository_root: Path
    project_root: Path
    repository_paths: tuple[str, ...]
    project_paths: tuple[str, ...]


class GitCommitError(RuntimeError):
    """A requested, project-local Git commit could not be completed."""


def _git_repository(project_root: Path) -> tuple[str, Path] | None:
    """Return Git and its worktree root, or silently report no Git context."""

    executable = _which("git")
    if executable is None:
        return None
    try:
        completed = _run_process(
            [
                executable,
                "-C",
                str(project_root),
                "rev-parse",
                "--show-toplevel",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    raw_root = completed.stdout.strip()
    if not raw_root:
        return None
    try:
        repository_root = Path(raw_root).resolve()
        project_root.resolve().relative_to(repository_root)
    except (OSError, ValueError):
        return None
    return executable, repository_root


def _committed_implementation_files(
    workspace: Workspace,
    *,
    executable: str,
    repository_root: Path,
) -> tuple[str, ...]:
    """Implementation files named by the lock as committed at HEAD.

    Used only to widen the commit unit so removals are staged.  Any failure —
    no commit yet, no committed lock, unreadable TOML — simply contributes no
    extra paths.
    """

    try:
        repository_path = (
            workspace.implementation_lock_path.resolve()
            .relative_to(repository_root)
            .as_posix()
        )
        completed = _run_process(
            [
                executable,
                "-C",
                str(repository_root),
                "show",
                f"HEAD:{repository_path}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError):
        return ()
    if completed.returncode != 0:
        return ()
    try:
        document = tomllib.loads(completed.stdout)
    except tomllib.TOMLDecodeError:
        return ()
    raw = document.get("implementation_files")
    if not isinstance(raw, list):
        return ()
    return tuple(str(value) for value in raw)


def implementation_commit_unit(
    workspace: Workspace,
    *,
    include_manifest: bool | Literal["workflow_identity"],
) -> GitCommitUnit | None:
    """Resolve the exact portable implementation files inside a Git worktree.

    An invalid or absent lock cannot define an implementation unit.  That case
    is already described by the four-valued implementation state and is not a
    Git error.
    """

    repository = _git_repository(workspace.root)
    lock = workspace.implementation_lock()
    if repository is None or lock is None or lock.get("valid") is not True:
        return None
    raw_files = lock.get("implementation_files")
    if not isinstance(raw_files, list):
        return None

    executable, repository_root = repository
    # Union with the committed lock so that a file the new implementation drops
    # has its removal staged.  Naming only the current files would leave the
    # deletion behind, and a clone would carry a file no lock describes.
    names = sorted(
        {
            *(str(value) for value in raw_files),
            *_committed_implementation_files(
                workspace,
                executable=executable,
                repository_root=repository_root,
            ),
        }
    )
    paths = [
        workspace.specification_path,
        *(workspace.root / name for name in names),
        workspace.implementation_lock_path,
    ]
    if include_manifest is True or (
        include_manifest == "workflow_identity"
        and _manifest_changes_workflow_identity(
            workspace,
            executable=executable,
            repository_root=repository_root,
        )
    ):
        paths.append(workspace.manifest_path)

    return _resolved_unit(
        workspace,
        executable=executable,
        repository_root=repository_root,
        paths=paths,
    )


def implementation_status_unit(workspace: Workspace) -> GitCommitUnit | None:
    """Resolve the files whose Git state can change clone-derived state.

    Unlike the commit operation, this keeps the lock path observable when the
    working lock is missing or malformed.  A read-only guard must still report
    that change even though it cannot trust the lock's file list.
    """

    repository = _git_repository(workspace.root)
    if repository is None:
        return None
    executable, repository_root = repository
    paths = [workspace.specification_path, workspace.implementation_lock_path]
    lock = workspace.implementation_lock()
    if lock is not None and lock.get("valid") is True:
        raw_files = lock.get("implementation_files")
        if isinstance(raw_files, list):
            paths.extend(workspace.root / str(value) for value in raw_files)
    elif workspace.workflow_entry is not None:
        source = workspace.absolute_spec(workspace.workflow_entry).partition(":")[0]
        paths.append(Path(source))
    if _manifest_changes_workflow_identity(
        workspace,
        executable=executable,
        repository_root=repository_root,
    ):
        paths.append(workspace.manifest_path)
    return _resolved_unit(
        workspace,
        executable=executable,
        repository_root=repository_root,
        paths=paths,
    )


def _resolved_unit(
    workspace: Workspace,
    *,
    executable: str,
    repository_root: Path,
    paths: list[Path],
) -> GitCommitUnit | None:
    """Normalize project paths for safe path-limited Git commands."""

    resolved_project = workspace.root.resolve()
    resolved: dict[str, tuple[str, str]] = {}
    for path in paths:
        try:
            absolute = path.resolve()
            project_relative = absolute.relative_to(resolved_project).as_posix()
            repository_relative = absolute.relative_to(repository_root).as_posix()
        except (OSError, ValueError):
            return None
        resolved[repository_relative] = (repository_relative, project_relative)
    ordered = [resolved[key] for key in sorted(resolved)]
    return GitCommitUnit(
        executable=executable,
        repository_root=repository_root,
        project_root=resolved_project,
        repository_paths=tuple(value[0] for value in ordered),
        project_paths=tuple(value[1] for value in ordered),
    )


def _manifest_identity(content: str) -> tuple[str, str | None] | None:
    """Read only the manifest fields that determine implementation identity."""

    try:
        raw = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return None
    specification = str(raw.get("specification_file") or "specification.md")
    workflow = raw.get("workflow_entry")
    return specification, str(workflow) if workflow else None


def _manifest_changes_workflow_identity(
    workspace: Workspace,
    *,
    executable: str,
    repository_root: Path,
) -> bool:
    """Whether the working manifest makes a clone derive another workflow."""

    try:
        repository_path = (
            workspace.manifest_path.resolve()
            .relative_to(repository_root)
            .as_posix()
        )
        working = workspace.manifest_path.read_text(encoding="utf-8")
        completed = _run_process(
            [
                executable,
                "-C",
                str(repository_root),
                "show",
                f"HEAD:{repository_path}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    if completed.returncode != 0:
        # An uncommitted manifest (including an initial repository) is the only
        # source of the project's workflow entry on the working tree.
        return True
    return _manifest_identity(working) != _manifest_identity(completed.stdout)


def uncommitted_commit_unit_paths(unit: GitCommitUnit) -> tuple[str, ...] | None:
    """Return dirty project paths, or ``None`` if Git cannot inspect them."""

    changed: list[str] = []
    for repository_path, project_path in zip(
        unit.repository_paths,
        unit.project_paths,
        strict=True,
    ):
        try:
            completed = _run_process(
                [
                    unit.executable,
                    "-C",
                    str(unit.repository_root),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    repository_path,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        if completed.stdout:
            changed.append(project_path)
    return tuple(changed)


def commit_implementation_unit(unit: GitCommitUnit, message: str) -> str:
    """Stage and commit only ``unit``, preserving unrelated staged changes."""

    clean_message = message.strip()
    if not clean_message:
        raise GitCommitError("The Git commit message must not be empty.")

    commands = (
        [
            unit.executable,
            "-C",
            str(unit.repository_root),
            "add",
            "--",
            *unit.repository_paths,
        ],
        [
            unit.executable,
            "-C",
            str(unit.repository_root),
            "commit",
            "--only",
            "-m",
            clean_message,
            "--",
            *unit.repository_paths,
        ],
    )
    for command in commands:
        try:
            completed = _run_process(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise GitCommitError(f"Git could not start: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise GitCommitError(
                "Git could not commit the implementation"
                + (f": {detail}" if detail else ".")
            )

    try:
        completed = _run_process(
            [
                unit.executable,
                "-C",
                str(unit.repository_root),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "created"
    revision = completed.stdout.strip()
    return revision or "created"


__all__ = [
    "GitCommitError",
    "GitCommitUnit",
    "commit_implementation_unit",
    "implementation_commit_unit",
    "implementation_status_unit",
    "uncommitted_commit_unit_paths",
]
