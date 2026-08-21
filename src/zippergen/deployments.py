"""Safe archival and purging of a deployment's artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Iterable

from zippergen.deployment_platform import (
    service_is_live,
    deployment_bundles_dir as _deployment_bundles_dir,
    deployment_environment_dir as _deployment_environment_dir,
    deployment_launchd_path as _deployment_launchd_path,
    deployment_profile_path as _deployment_profile_path,
    deployment_script_path as _deployment_script_path,
    deployment_secrets_path as _deployment_secrets_path,
    deployment_service_path as _deployment_service_path,
    deployment_service_status as _deployment_service_status,
    deployments_dir as _deployments_dir,
    installed_launchd_path as _installed_launchd_path,
    installed_systemd_service_path as _installed_systemd_service_path,
    launchctl_command as _launchctl_command,
    launchctl_domain as _launchctl_domain,
    launchd_label as _launchd_label,
    run_launchctl as _run_launchctl,
    run_systemctl as _run_systemctl,
    service_manager as _service_manager,
    slug as _slug,
    systemctl_command as _systemctl_command,
    systemd_unit_name as _systemd_unit_name,
    zippergen_home as _zippergen_home,
)


class DeploymentRemovalError(RuntimeError):
    """A deployment could not be removed without risking unrelated state."""


@dataclass(frozen=True)
class DeploymentArtifact:
    label: str
    path: Path
    destination: Path
    kind: str
    # Whether removal keeps this in the archive. Only what cannot be got back
    # is kept: the durable store and the log record what actually happened,
    # and the profile says what produced them. Source lives in git, the
    # environment and service files are rebuilt by deploying again, and
    # secrets must not be left behind.
    retain: bool = False


@dataclass(frozen=True)
class DeploymentRemovalResult:
    name: str
    purged: bool
    artifact_count: int
    archive: Path | None

@dataclass(frozen=True)
class DeploymentLogCompactionResult:
    name: str
    log: Path
    archived_bytes: int
    archive: Path | None
    removed_archives: int
    removed_archive_bytes: int


@dataclass(frozen=True)
class DeploymentStoreResetResult:
    """Recoverable replacement of one deployment's durable store."""

    name: str
    store: Path
    archive: Path | None
    archived_files: int


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _sqlite_family(path: Path) -> tuple[Path, ...]:
    return (
        path,
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
        Path(str(path) + "-journal"),
    )


def _artifact(
    label: str,
    path: Path,
    destination: str,
    *,
    kind: str = "file",
    retain: bool = False,
) -> DeploymentArtifact:
    return DeploymentArtifact(
        label=label,
        path=path.expanduser(),
        destination=Path(destination),
        kind=kind,
        retain=retain,
    )


def deployment_artifacts(
    name: str,
    profile: dict[str, object],
) -> tuple[DeploymentArtifact, ...]:
    """Return the exact deployment-owned artifact set, without reading secrets."""

    slug = _slug(name)
    if not slug or slug in {".", ".."}:
        raise DeploymentRemovalError(f"Unsafe deployment name: {name!r}.")

    artifacts = [
        _artifact(
            "Profile",
            _deployment_profile_path(name),
            "profile/deployment.json",
            retain=True,
        ),
        _artifact(
            "Private secrets",
            _deployment_secrets_path(name),
            "profile/secrets.json",
        ),
        _artifact(
            "Run script",
            _deployment_script_path(name),
            "launch/run.sh",
        ),
        _artifact(
            "systemd template",
            _deployment_service_path(name),
            "launch/template.service",
        ),
        _artifact(
            "launchd template",
            _deployment_launchd_path(name),
            "launch/template.plist",
        ),
        _artifact(
            "Installed systemd unit",
            _installed_systemd_service_path(name),
            "launch/installed.service",
        ),
        _artifact(
            "Installed launchd agent",
            _installed_launchd_path(name),
            "launch/installed.plist",
        ),
        _artifact(
            "Managed environment",
            _deployment_environment_dir(name),
            "runtime/environment",
            kind="directory",
        ),
        _artifact(
            "Immutable bundles",
            _deployment_bundles_dir(name),
            "runtime/bundles",
            kind="directory",
        ),
    ]

    secrets = profile.get("secrets_file")
    if secrets:
        artifacts.append(
            _artifact(
                "Private secrets",
                Path(str(secrets)),
                "profile/custom-secrets.json",
            )
        )
    store = profile.get("store")
    if store:
        store_path = Path(str(store)).expanduser()
        for index, path in enumerate(_sqlite_family(store_path)):
            suffix = "" if index == 0 else path.name.removeprefix(store_path.name)
            artifacts.append(
                _artifact(
                    "Durable store" if index == 0 else f"Store sidecar {suffix}",
                    path,
                    f"state/store.sqlite{suffix}",
                    retain=True,
                )
            )
    log = profile.get("log")
    if log:
        artifacts.append(
            _artifact(
                "Deployment log",
                Path(str(log)),
                "logs/deployment.log",
                retain=True,
            )
        )

    unique: list[DeploymentArtifact] = []
    seen: set[Path] = set()
    for item in artifacts:
        key = _resolved(item.path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)


def _profile_references(profile: dict[str, object]) -> tuple[Path, ...]:
    references: list[Path] = []
    for key in ("store", "log", "secrets_file", "bundle", "cwd", "python"):
        value = profile.get(key)
        if value:
            references.append(_resolved(Path(str(value))))
    return tuple(references)


def _other_deployment_references(name: str) -> tuple[tuple[str, Path], ...]:
    references: list[tuple[str, Path]] = []
    target = _slug(name)
    directory = _deployments_dir()
    if not directory.exists():
        return ()
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".secrets.json") or path.stem == target:
            continue
        try:
            profile = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(profile, dict):
            continue
        other_name = str(profile.get("name") or path.stem)
        references.extend(
            (other_name, reference)
            for reference in _profile_references(profile)
        )
    return tuple(references)


def _shared_conflicts(
    name: str,
    artifacts: Iterable[DeploymentArtifact],
) -> tuple[str, ...]:
    conflicts: set[str] = set()
    references = _other_deployment_references(name)
    for artifact in artifacts:
        candidate = _resolved(artifact.path)
        for other_name, reference in references:
            shared = candidate == reference
            if artifact.kind == "directory":
                shared = shared or reference.is_relative_to(candidate)
            if shared:
                conflicts.add(
                    f"{artifact.path} is also used by deployment {other_name}"
                )
    return tuple(sorted(conflicts))


def present_deployment_artifacts(
    name: str,
    profile: dict[str, object],
) -> tuple[DeploymentArtifact, ...]:
    artifacts = tuple(
        item
        for item in deployment_artifacts(name, profile)
        if _path_present(item.path)
    )
    conflicts = _shared_conflicts(name, artifacts)
    if conflicts:
        raise DeploymentRemovalError(
            "Deployment-owned paths overlap another deployment: "
            + "; ".join(conflicts)
        )
    for item in artifacts:
        if (
            item.kind == "file"
            and item.path.is_dir()
            and not item.path.is_symlink()
        ):
            raise DeploymentRemovalError(
                f"Expected a deployment file but found a directory: {item.path}"
            )
    return artifacts


def unregister_deployment_service(name: str) -> str:
    """Stop, disable, and unregister the user service before moving files."""

    installed_systemd = _installed_systemd_service_path(name)
    installed_launchd = _installed_launchd_path(name)
    try:
        manager = _service_manager()
    except SystemExit as exc:
        if _path_present(installed_systemd) or _path_present(installed_launchd):
            raise DeploymentRemovalError(
                "Cannot verify that the installed service is stopped because "
                f"no supported service manager is available: {exc}"
            ) from exc
        return "no installed service registration"

    status = _deployment_service_status(name)
    state = str(status.get("state") or "unknown")
    if state == "unknown" and (
        _path_present(installed_systemd) or _path_present(installed_launchd)
    ):
        raise DeploymentRemovalError(
            "Cannot verify that the installed service is stopped: "
            + str(status.get("detail") or "service state is unknown")
        )

    if manager == "launchd":
        if state != "not-loaded":
            service = f"{_launchctl_domain()}/{_launchd_label(name)}"
            _run_launchctl(
                _launchctl_command("bootout", service),
                check=True,
            )
        if _path_present(installed_launchd):
            installed_launchd.unlink()
        return (
            "launchd service stopped and unregistered"
            if state != "not-loaded"
            else "launchd service was not loaded"
        )

    unit = _systemd_unit_name(name)
    should_manage = state != "not-loaded" or _path_present(installed_systemd)
    if should_manage:
        if state != "not-loaded":
            _run_systemctl(_systemctl_command("stop", unit))
        _run_systemctl(_systemctl_command("disable", unit))
    if _path_present(installed_systemd):
        installed_systemd.unlink()
    if should_manage:
        _run_systemctl(_systemctl_command("daemon-reload"))
    return (
        "systemd service stopped, disabled, and unregistered"
        if should_manage
        else "systemd service was not installed"
    )


def _unique_removal_directory(name: str, *, purging: bool) -> Path:
    root = _zippergen_home() / "trash" / "deployments"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = ".purging-" if purging else ""
    base = f"{prefix}{_slug(name)}-{timestamp}"
    destination = root / base
    suffix = 2
    while destination.exists():
        destination = root / f"{base}-{suffix}"
        suffix += 1
    destination.mkdir(mode=0o700)
    return destination


def _unique_log_archive(name: str) -> Path:
    root = _zippergen_home() / "trash" / "deployment-logs"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{_slug(name)}-{timestamp}.log"
    destination = root / base
    suffix = 2
    while destination.exists():
        destination = root / f"{_slug(name)}-{timestamp}-{suffix}.log"
        suffix += 1
    return destination


TRASH_AREAS = ("deployments", "deployment-stores", "deployment-logs")


@dataclass(frozen=True)
class TrashEntry:
    area: str
    path: Path
    age_days: float
    bytes: int


@dataclass(frozen=True)
class TrashPruneResult:
    removed: tuple[TrashEntry, ...]
    kept: tuple[TrashEntry, ...]

    @property
    def removed_bytes(self) -> int:
        return sum(entry.bytes for entry in self.removed)

    @property
    def kept_bytes(self) -> int:
        return sum(entry.bytes for entry in self.kept)


def _entry_bytes(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(
            item.stat().st_size for item in path.rglob("*") if item.is_file()
        )
    except OSError:
        return 0


def list_trash_entries(*, now: float | None = None) -> tuple[TrashEntry, ...]:
    """Describe everything sitting in this machine's deployment trash.

    Removal archives, reset archives and rotated logs all land here and nothing
    has ever cleaned them up. Their age is what decides whether they are still
    serving as an undo, so report that rather than only their size.
    """

    moment = time.time() if now is None else now
    entries: list[TrashEntry] = []
    for area in TRASH_AREAS:
        root = _zippergen_home() / "trash" / area
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            entries.append(
                TrashEntry(
                    area=area,
                    path=path,
                    age_days=max(0.0, (moment - modified) / 86400.0),
                    bytes=_entry_bytes(path),
                )
            )
    return tuple(entries)


def prune_trash(
    *,
    keep_days: float,
    now: float | None = None,
) -> TrashPruneResult:
    """Delete trash older than ``keep_days``, keeping the recent undo window.

    Removing a deployment archives its durable store precisely so a mistake can
    be undone, so pruning must never take today's archive. Age, not size, is
    the criterion.
    """

    if keep_days < 0:
        raise ValueError("keep_days must be zero or greater")
    removed: list[TrashEntry] = []
    kept: list[TrashEntry] = []
    for entry in list_trash_entries(now=now):
        if entry.age_days < keep_days:
            kept.append(entry)
            continue
        try:
            if entry.path.is_dir():
                shutil.rmtree(entry.path)
            else:
                entry.path.unlink()
        except OSError as exc:
            raise DeploymentRemovalError(
                f"Could not delete {entry.path}: {exc}"
            ) from exc
        removed.append(entry)
    return TrashPruneResult(removed=tuple(removed), kept=tuple(kept))


def _unique_store_archive(name: str) -> Path:
    root = _zippergen_home() / "trash" / "deployment-stores"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{_slug(name)}-{timestamp}"
    destination = root / base
    suffix = 2
    while destination.exists():
        destination = root / f"{base}-{suffix}"
        suffix += 1
    destination.mkdir(mode=0o700)
    return destination


def reset_deployment_store(
    name: str,
    profile: dict[str, object],
) -> DeploymentStoreResetResult:
    """Archive the current store and leave its configured path empty.

    The caller must stop the supervised service first. Every member of the
    SQLite file family is moved as one recoverable unit. A failure part way
    through is rolled back so reset cannot silently leave half a database.
    """

    service = _deployment_service_status(name)
    if service_is_live(service):
        raise DeploymentRemovalError(
            f"Stop deployment {name} before resetting its durable state. "
            f"Current service state: {service['detail']}"
        )
    raw_store = profile.get("store")
    if not raw_store:
        raise DeploymentRemovalError(
            f"Deployment {name} has no durable store configured."
        )
    store = Path(str(raw_store)).expanduser()
    present = tuple(path for path in _sqlite_family(store) if _path_present(path))
    if not present:
        return DeploymentStoreResetResult(name, store, None, 0)
    if any(path.is_dir() or path.is_symlink() for path in present):
        raise DeploymentRemovalError(
            f"Expected regular SQLite files for deployment {name}: {store}"
        )

    destination = _unique_store_archive(name)
    moved: list[tuple[Path, Path]] = []
    try:
        for source in present:
            target = destination / source.name
            shutil.move(str(source), str(target))
            target.chmod(0o600)
            moved.append((source, target))
        metadata = destination / "reset.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "deployment": name,
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
    except Exception as exc:
        for source, target in reversed(moved):
            if _path_present(target) and not _path_present(source):
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
        shutil.rmtree(destination, ignore_errors=True)
        raise DeploymentRemovalError(
            f"Could not reset deployment {name} safely: {exc}"
        ) from exc

    return DeploymentStoreResetResult(
        name=name,
        store=store,
        archive=destination,
        archived_files=len(moved),
    )




def compact_deployment_logs(
    name: str,
    profile: dict[str, object],
    *,
    keep_archives: int = 3,
) -> DeploymentLogCompactionResult:
    """Rotate a stopped deployment log and retain a bounded archive set."""

    if keep_archives < 0:
        raise ValueError("keep_archives must be zero or greater")
    service = _deployment_service_status(name)
    if service_is_live(service):
        raise DeploymentRemovalError(
            f"Stop deployment {name} before rotating its log. "
            f"Current service state: {service['detail']}"
        )

    raw_log = profile.get("log")
    if not raw_log:
        raise DeploymentRemovalError(
            f"Deployment {name} has no log path configured."
        )
    log = Path(str(raw_log)).expanduser()
    if _path_present(log) and (
        not log.is_file() or log.is_symlink()
    ):
        raise DeploymentRemovalError(
            f"Expected a regular deployment log file: {log}"
        )

    archive: Path | None = None
    archived_bytes = 0
    removed_archives = 0
    removed_archive_bytes = 0
    try:
        if log.is_file():
            archived_bytes = log.stat().st_size
        if archived_bytes:
            archive = _unique_log_archive(name)
            shutil.copyfile(log, archive)
            archive.chmod(0o600)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.parent.chmod(0o700)
        log.write_bytes(b"")
        log.chmod(0o600)

        root = _zippergen_home() / "trash" / "deployment-logs"
        candidates = sorted(
            (
                path
                for path in root.glob(f"{_slug(name)}-*.log")
                if path.is_file() and not path.is_symlink()
            ),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        ) if root.is_dir() else []
        for old in candidates[keep_archives:]:
            removed_archive_bytes += old.stat().st_size
            old.unlink()
            removed_archives += 1

        profile["log_generation_offset"] = 0
        profile["log_compacted_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        _deployment_profile_path(name).write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        raise DeploymentRemovalError(
            f"Could not rotate deployment logs safely: {exc}"
        ) from exc

    return DeploymentLogCompactionResult(
        name=name,
        log=log,
        archived_bytes=archived_bytes,
        archive=archive,
        removed_archives=removed_archives,
        removed_archive_bytes=removed_archive_bytes,
    )


def _discard_unretained(
    destination: Path,
    artifacts: Iterable[DeploymentArtifact],
) -> None:
    """Delete the staged artifacts that an archive does not keep."""

    for artifact in artifacts:
        if artifact.retain:
            continue
        staged = destination / artifact.destination
        if staged.is_dir() and not staged.is_symlink():
            shutil.rmtree(staged)
        elif staged.exists() or staged.is_symlink():
            staged.unlink()
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()


def remove_deployment_artifacts(
    name: str,
    profile: dict[str, object],
    *,
    purge: bool,
) -> DeploymentRemovalResult:
    """Archive or permanently purge a deployment after service unregistration.

    Every artifact is moved into a staging directory first so that a failure
    part way through can be rolled back. What survives afterwards is only what
    cannot be got back another way: the durable store, the log, and the
    profile. Secrets, the managed environment, the source bundles, and the
    service files are deleted even when the removal is recoverable.
    """

    artifacts = present_deployment_artifacts(name, profile)
    destination = _unique_removal_directory(name, purging=purge)
    moved: list[tuple[Path, Path]] = []
    try:
        for artifact in artifacts:
            target = destination / artifact.destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(artifact.path), str(target))
            moved.append((artifact.path, target))
        metadata = destination / "removal.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "deployment": name,
                    "removed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "purge_requested": purge,
                    "artifacts": [
                        {
                            "label": artifact.label,
                            "source": str(artifact.path),
                            "archived_as": str(artifact.destination),
                        }
                        for artifact in artifacts
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata.chmod(0o600)
    except Exception as exc:
        for source, target in reversed(moved):
            if _path_present(target) and not _path_present(source):
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
        shutil.rmtree(destination, ignore_errors=True)
        raise DeploymentRemovalError(
            f"Could not remove deployment {name} safely: {exc}"
        ) from exc

    if purge:
        try:
            shutil.rmtree(destination)
        except OSError as exc:
            raise DeploymentRemovalError(
                "Deployment artifacts were isolated from active use, but "
                f"permanent deletion failed at {destination}: {exc}"
            ) from exc
        archive = None
    else:
        try:
            _discard_unretained(destination, artifacts)
        except OSError as exc:
            raise DeploymentRemovalError(
                "Deployment artifacts were isolated from active use, but "
                f"discarding the rebuildable ones failed at {destination}: "
                f"{exc}"
            ) from exc
        archive = destination
    return DeploymentRemovalResult(
        name=name,
        purged=purge,
        artifact_count=len(artifacts),
        archive=archive,
    )
