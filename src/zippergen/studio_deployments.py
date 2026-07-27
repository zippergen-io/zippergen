"""Safe archival and purging of Studio-owned deployment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Iterable

from zippergen.serve import (
    _deployment_bundles_dir,
    _deployment_environment_dir,
    _deployment_launchd_path,
    _deployment_profile_path,
    _deployment_script_path,
    _deployment_secrets_path,
    _deployment_service_path,
    _deployments_dir,
    _installed_launchd_path,
    _installed_systemd_service_path,
    _launchctl_command,
    _launchctl_domain,
    _launchd_label,
    _run_launchctl,
    _run_systemctl,
    _service_manager,
    _slug,
    _systemctl_command,
    _systemd_unit_name,
    _zippergen_home,
    _deployment_service_status,
)


class DeploymentRemovalError(RuntimeError):
    """A deployment could not be removed without risking unrelated state."""


@dataclass(frozen=True)
class DeploymentArtifact:
    label: str
    path: Path
    destination: Path
    kind: str


@dataclass(frozen=True)
class DeploymentRemovalResult:
    name: str
    purged: bool
    artifact_count: int
    archive: Path | None


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
) -> DeploymentArtifact:
    return DeploymentArtifact(
        label=label,
        path=path.expanduser(),
        destination=Path(destination),
        kind=kind,
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
                )
            )
    log = profile.get("log")
    if log:
        artifacts.append(
            _artifact(
                "Deployment log",
                Path(str(log)),
                "logs/deployment.log",
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


def remove_deployment_artifacts(
    name: str,
    profile: dict[str, object],
    *,
    purge: bool,
) -> DeploymentRemovalResult:
    """Archive or permanently purge a deployment after service unregistration."""

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
        archive = destination
    return DeploymentRemovalResult(
        name=name,
        purged=purge,
        artifact_count=len(artifacts),
        archive=archive,
    )
