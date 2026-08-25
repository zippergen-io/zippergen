"""Source bundling and managed Python environments for deployments."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import venv

from zippergen.deployment import DeploymentSpec
from zippergen.deployment_platform import (
    deployment_bundles_dir,
    deployment_environment_releases_dir,
    slug,
)
from zippergen.syntax import Workflow
from zippergen.validation import assistant_actions
from zippergen.private_files import ensure_private_directory


@dataclass
class DeploymentEnvironmentUpdate:
    """An unpublished environment generation that can be kept or discarded."""

    environment: Path
    previous: Path | None
    finished: bool = False

    def commit(self) -> None:
        if self.finished:
            return
        if self.previous is not None:
            if self.previous.is_symlink():
                self.previous.unlink(missing_ok=True)
            else:
                shutil.rmtree(self.previous, ignore_errors=True)
        for stale in self.environment.parent.iterdir():
            if stale == self.environment:
                continue
            if stale.is_symlink() or stale.is_file():
                stale.unlink(missing_ok=True)
            elif stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        self.finished = True

    def rollback(self) -> None:
        if self.finished:
            return
        shutil.rmtree(self.environment, ignore_errors=True)
        self.finished = True


def _deployment_python_path(environment_dir: Path) -> Path:
    if os.name == "nt":
        return environment_dir / "Scripts" / "python.exe"
    return environment_dir / "bin" / "python"


def _bundle_relative_path(source: Path, source_root: Path) -> Path:
    try:
        return source.relative_to(source_root)
    except ValueError:
        digest = hashlib.sha1(str(source).encode()).hexdigest()[:8]
        return Path("external") / f"{digest}-{source.name}"


def _copy_deployment_source(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            target,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "*.pyc"
            ),
        )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _validate_source_symlinks(source: Path) -> None:
    """Reject links that would make a bundle escape its declared source."""

    if not source.is_dir():
        return
    root = source.resolve()
    for path in source.rglob("*"):
        if not path.is_symlink():
            continue
        raw_target = Path(os.readlink(path))
        if raw_target.is_absolute():
            raise SystemExit(
                f"Declared deployment directory contains an absolute symlink: {path}"
            )
        target = path.resolve(strict=False)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise SystemExit(
                f"Declared deployment directory contains a symlink outside "
                f"its root: {path} -> {raw_target}"
            ) from exc
        if not target.exists():
            raise SystemExit(
                f"Declared deployment directory contains a broken symlink: "
                f"{path} -> {raw_target}"
            )


def _deployment_sources(
    profile: dict[str, object],
    spec: DeploymentSpec,
    workflow: Workflow,
) -> tuple[Path, str, Path, list[Path], bool]:
    """Resolve exactly the source paths copied into one deployment bundle."""

    source_cwd = Path(
        str(profile.get("source_cwd") or profile["cwd"])
    ).expanduser().resolve()
    source_workflow = str(profile.get("source_workflow") or profile["workflow"])
    module_ref = source_workflow.partition(":")[0]
    module_path = Path(module_ref).expanduser()
    if not module_path.is_absolute():
        module_path = source_cwd / module_path
    preserve_import = False
    if module_path.exists():
        module_path = module_path.resolve()
        sources = [module_path]
    else:
        try:
            module_spec = importlib.util.find_spec(module_ref)
            top_spec = importlib.util.find_spec(module_ref.split(".", 1)[0])
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            raise SystemExit(
                f"Workflow module {module_ref!r} cannot be located for an "
                "immutable deployment bundle: {exc}"
            ) from None
        if module_spec is None or module_spec.origin is None or top_spec is None:
            raise SystemExit(
                f"Workflow module {module_ref!r} has no bundleable source. "
                "Deploy a project-local Python file or package."
            )
        module_path = Path(module_spec.origin).resolve()
        locations = top_spec.submodule_search_locations
        top_source = (
            Path(next(iter(locations))).resolve()
            if locations
            else Path(str(top_spec.origin)).resolve()
        )
        try:
            top_source.relative_to(source_cwd)
            module_path.relative_to(source_cwd)
        except ValueError as exc:
            raise SystemExit(
                f"Workflow module {module_ref!r} resolves outside the project "
                f"root ({module_path}). Use a project-local module so deploy "
                "can snapshot the exact source."
            ) from exc
        sources = [top_source]
        preserve_import = True
    for declared in spec.files:
        path = Path(declared).expanduser()
        if not path.is_absolute():
            path = source_cwd / path
        if path.is_symlink():
            raise SystemExit(
                f"Declared deployment source must not itself be a symlink: {path}"
            )
        path = path.resolve()
        if not path.exists():
            raise SystemExit(f"Declared deployment file does not exist: {path}")
        if path not in sources:
            sources.append(path)
    for action in assistant_actions(workflow):
        if action.instructions_path is None:
            continue
        path = Path(action.instructions_path).resolve()
        try:
            path.relative_to(source_cwd)
        except ValueError as exc:
            raise SystemExit(
                f"Assistant instruction file for {action.name!r} is outside "
                f"the project root and cannot be bundled portably: {path}"
            ) from exc
        if path not in sources:
            sources.append(path)
    for source in sources:
        _validate_source_symlinks(source)
    return source_cwd, source_workflow, module_path, sources, preserve_import


def _deployment_source_digest(sources: list[Path], source_cwd: Path) -> str:
    """Hash the project inputs that determine one deployment."""

    files: dict[str, Path] = {}
    ignored_parts = {".git", ".venv", "__pycache__"}
    for source in sources:
        target = _bundle_relative_path(source, source_cwd)
        if source.is_dir():
            for path in source.rglob("*"):
                relative = path.relative_to(source)
                if path.is_symlink():
                    files[str(target / relative)] = path
                    continue
                if (
                    not path.is_file()
                    or any(part in ignored_parts for part in relative.parts)
                    or path.suffix == ".pyc"
                ):
                    continue
                files[str(target / relative)] = path
        else:
            files[str(target)] = source
    digest = hashlib.sha256()
    for relative, path in sorted(files.items()):
        digest.update(relative.encode())
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode())
        else:
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def deployment_source_provenance(
    profile: dict[str, object],
    spec: DeploymentSpec,
    workflow: Workflow,
) -> dict[str, str]:
    """Describe the workflow source that a deployment bundle represents."""

    source_cwd, _source_workflow, _module_path, sources, _preserve_import = _deployment_sources(
        profile, spec, workflow
    )
    manifest = source_cwd / "zippergen.toml"
    provenance_sources = [*sources, *([manifest] if manifest.is_file() else [])]
    result = {
        "kind": "source-bundle",
        "source": str(source_cwd),
        "source_sha256": _deployment_source_digest(
            provenance_sources, source_cwd
        ),
    }
    revision = _checkout_revision(source_cwd)
    if revision:
        result["revision"] = revision
    return result


def bundle_deployment(
    profile: dict[str, object],
    spec: DeploymentSpec,
    workflow: Workflow,
) -> None:
    """Snapshot a path-based workflow and every declared deployment file."""

    source_cwd, source_workflow, module_path, sources, preserve_import = _deployment_sources(
        profile, spec, workflow
    )
    module_ref, separator, workflow_name = source_workflow.partition(":")
    version = (
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{time.time_ns() % 1_000_000_000:09d}"
    )
    bundles = deployment_bundles_dir(str(profile["name"]))
    ensure_private_directory(bundles)
    bundle_root = bundles / version
    bundle_root.mkdir(parents=True, exist_ok=False)

    try:
        profile["workflow_source"] = deployment_source_provenance(
            profile, spec, workflow
        )

        copied: dict[Path, Path] = {}
        for source in sources:
            relative = _bundle_relative_path(source, source_cwd)
            _copy_deployment_source(source, bundle_root / relative)
            copied[source] = relative
    except BaseException:
        shutil.rmtree(bundle_root, ignore_errors=True)
        raise

    profile["source_cwd"] = str(source_cwd)
    profile["source_workflow"] = source_workflow
    profile["cwd"] = str(bundle_root)
    if preserve_import:
        profile["workflow"] = source_workflow
    else:
        workflow_relative = copied[module_path]
        profile["workflow"] = str(workflow_relative) + (
            f":{workflow_name}" if separator else ""
        )
    profile["bundle"] = str(bundle_root)
    profile["bundled_files"] = [str(path) for path in copied.values()]


def _installed_zippergen_origin() -> str | None:
    """Where this installed copy came from, as an installable requirement.

    An install records its own origin in ``direct_url.json``, so a copy
    installed from Git can say so instead of naming a version. Without this a
    deployment asks an index for ``zippergen==<version>``, which does not exist
    while the package is unpublished -- so following the documented install
    succeeds and the first deployment fails.
    """

    import json
    from importlib.metadata import Distribution, PackageNotFoundError

    try:
        raw = Distribution.from_name("zippergen").read_text("direct_url.json")
    except (PackageNotFoundError, OSError):
        return None
    if not raw:
        return None
    try:
        recorded = json.loads(raw)
    except ValueError:
        return None
    url = str(recorded.get("url") or "")
    if not url:
        return None
    vcs = recorded.get("vcs_info")
    if isinstance(vcs, dict) and vcs.get("vcs") == "git":
        commit = str(vcs.get("commit_id") or "")
        # Pin the commit: a deployment is an immutable release, and resolving
        # a branch later would install something the checks never saw.
        return f"git+{url}@{commit}" if commit else f"git+{url}"
    if url.startswith("file://"):
        return url[len("file://"):]
    return None


def _zippergen_install_requirement(
    *,
    extras: tuple[str, ...] = (),
) -> str:
    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "pyproject.toml").exists():
        requirement = str(project_root)
    else:
        origin = _installed_zippergen_origin()
        if origin is not None:
            return (
                f"zippergen[{','.join(sorted(set(extras)))}] @ {origin}"
                if extras
                else origin
            )
        try:
            from importlib.metadata import version

            requirement = f"zippergen=={version('zippergen')}"
        except Exception:
            requirement = "zippergen"
    if not extras:
        return requirement
    name, separator, version_spec = requirement.partition("==")
    suffix = ",".join(sorted(set(extras)))
    return (
        f"{name}[{suffix}]=={version_spec}"
        if separator
        else f"{requirement}[{suffix}]"
    )


def _checkout_revision(project_root: Path) -> str | None:
    """Read a checkout revision without invoking Git or contacting a remote."""

    git_marker = project_root / ".git"
    try:
        if git_marker.is_file():
            marker = git_marker.read_text().strip()
            if not marker.startswith("gitdir:"):
                return None
            git_dir = Path(marker.partition(":")[2].strip())
            if not git_dir.is_absolute():
                git_dir = (project_root / git_dir).resolve()
        elif git_marker.is_dir():
            git_dir = git_marker
        else:
            return None
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head if len(head) >= 12 else None
        reference = head.partition(":")[2].strip()
        loose = git_dir / reference
        if loose.is_file():
            return loose.read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                revision, _, name = line.partition(" ")
                if name == reference:
                    return revision
    except OSError:
        return None
    return None


def zippergen_runtime_provenance() -> dict[str, str]:
    """Describe the ZipperGen source selected for a deployment environment."""

    from importlib.metadata import PackageNotFoundError, version

    project_root = Path(__file__).resolve().parents[2]
    try:
        installed_version = version("zippergen")
    except PackageNotFoundError:
        installed_version = "unknown"
    if not (project_root / "pyproject.toml").is_file():
        return {
            "kind": "package",
            "version": installed_version,
            "source": "installed package",
        }

    provenance = {
        "kind": "source-checkout",
        "version": installed_version,
        "source": str(project_root),
    }
    digest = hashlib.sha256()
    package_root = project_root / "src" / "zippergen"
    source_files = (
        [
            path
            for path in package_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ]
        if package_root.is_dir()
        else []
    )
    for path in [project_root / "pyproject.toml", *sorted(source_files)]:
        try:
            relative = path.relative_to(project_root)
            digest.update(str(relative).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        except OSError:
            continue
    provenance["source_sha256"] = digest.hexdigest()
    revision = _checkout_revision(project_root)
    if revision:
        provenance["revision"] = revision
    return provenance


def _deployment_zippergen_extras(
    profile: dict[str, object],
) -> tuple[str, ...]:
    from zippergen.provider_connections import _CONNECTOR_KINDS

    raw = profile.get("connectors") or {}
    bindings = raw if isinstance(raw, dict) else {}
    # Which kinds need the extra is the provider's own business, so ask it
    # rather than keeping a second list here that can fall out of step.
    google_kinds = _CONNECTOR_KINDS["google"]
    if any(
        isinstance(value, dict) and value.get("kind") in google_kinds
        for value in bindings.values()
    ):
        return ("google",)
    return ()


def prepare_deployment_environment(
    profile: dict[str, object],
    spec: DeploymentSpec,
    *,
    skip_install: bool,
    defer_cleanup: bool = False,
) -> DeploymentEnvironmentUpdate | None:
    """Build an immutable environment generation for profile publication.

    The active profile remains the only publication point. Until that profile
    is atomically replaced, a crash can leave an unused generation behind but
    cannot make the previous deployment run with a candidate environment.
    """

    requirements = [package.requirement for package in spec.packages]
    profile["packages"] = requirements
    zippergen_extras = _deployment_zippergen_extras(profile)
    profile["zippergen_extras"] = list(zippergen_extras)
    profile["zippergen_runtime"] = zippergen_runtime_provenance()
    if skip_install:
        profile["python"] = str(profile.get("python") or sys.executable)
        return None

    name = str(profile["name"])
    releases_dir = deployment_environment_releases_dir(name)
    ensure_private_directory(releases_dir)
    version = (
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{time.time_ns() % 1_000_000_000:09d}"
    )
    environment_dir = releases_dir / version
    build_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{slug(name)}-building-",
            dir=releases_dir,
        )
    )
    build_python = _deployment_python_path(build_dir)
    uv = shutil.which("uv")
    phase = "creating the environment"
    print(f"Creating managed Python environment for {name}...")
    try:
        if uv is not None:
            subprocess.run(
                [uv, "venv", "--python", sys.executable, str(build_dir)],
                check=True,
            )
            install = [
                uv,
                "pip",
                "install",
                "--refresh-package",
                "zippergen",
                "--python",
                str(build_python),
                _zippergen_install_requirement(extras=zippergen_extras),
                *requirements,
            ]
        else:
            venv.EnvBuilder(with_pip=True).create(build_dir)
            install = [
                str(build_python),
                "-m",
                "pip",
                "install",
                _zippergen_install_requirement(extras=zippergen_extras),
                *requirements,
            ]
        phase = "installing deployment dependencies"
        print("Installing deployment dependencies...")
        subprocess.run(install, check=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(build_dir, ignore_errors=True)
        outcome = (
            f"signal {-exc.returncode}"
            if exc.returncode < 0
            else f"exit code {exc.returncode}"
        )
        guidance = (
            " ZipperGen found uv and used it instead of ensurepip."
            if uv is not None
            else " Install uv and retry to avoid the standard-library "
            "ensurepip bootstrap."
        )
        raise SystemExit(
            f"Managed environment failed while {phase} ({outcome})."
            f"{guidance} The previous deployment environment, if any, was "
            "left unchanged."
        ) from None
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise SystemExit(
            f"Managed environment failed while {phase}: {exc}. The previous "
            "deployment environment, if any, was left unchanged."
        ) from None
    except KeyboardInterrupt:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise SystemExit(
            f"Managed environment failed while {phase}: {exc}. The previous "
            "deployment environment, if any, was left unchanged."
        ) from None

    try:
        os.replace(build_dir, environment_dir)
    except OSError as exc:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise SystemExit(
            "Managed environment was built but could not publish candidate "
            f"{environment_dir}: {exc}. The previous environment is unchanged."
        ) from None
    # The generation this one replaces, and only when ZipperGen owns it: a
    # first deployment has none, and a path outside the managed root belongs
    # to someone else and is never removed.
    previous_raw = profile.get("environment_dir")
    previous = Path(str(previous_raw)).expanduser() if previous_raw else None
    if previous is not None:
        managed_root = releases_dir.parents[1].resolve()
        try:
            previous.resolve(strict=False).relative_to(managed_root)
        except ValueError:
            previous = None
        else:
            if not (previous.exists() or previous.is_symlink()):
                previous = None
    profile["python"] = str(_deployment_python_path(environment_dir))
    profile["environment_dir"] = str(environment_dir)
    update = DeploymentEnvironmentUpdate(environment_dir, previous)
    if defer_cleanup:
        return update
    update.commit()
    return None


__all__ = [
    "DeploymentEnvironmentUpdate",
    "bundle_deployment",
    "prepare_deployment_environment",
]
