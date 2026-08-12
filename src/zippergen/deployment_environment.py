"""Source bundling and managed Python environments for deployments."""

from __future__ import annotations

import hashlib
import os
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
    deployment_environment_dir,
    slug,
)
from zippergen.syntax import Workflow
from zippergen.validation import assistant_actions


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
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "*.pyc"
            ),
        )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def bundle_deployment(
    profile: dict[str, object],
    spec: DeploymentSpec,
    workflow: Workflow,
) -> None:
    """Snapshot a path-based workflow and every declared deployment file."""

    source_cwd = Path(
        str(profile.get("source_cwd") or profile["cwd"])
    ).expanduser().resolve()
    source_workflow = str(
        profile.get("source_workflow") or profile["workflow"]
    )
    module_ref, separator, workflow_name = source_workflow.partition(":")
    module_path = Path(module_ref).expanduser()
    if not module_path.is_absolute():
        module_path = source_cwd / module_path
    if not module_path.exists():
        # Importable modules are already packaged Python artifacts. Path-based
        # workflows get a concrete source bundle here.
        profile.setdefault("source_cwd", str(source_cwd))
        profile.setdefault("source_workflow", source_workflow)
        return

    version = (
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{time.time_ns() % 1_000_000_000:09d}"
    )
    bundle_root = deployment_bundles_dir(str(profile["name"])) / version
    bundle_root.mkdir(parents=True, exist_ok=False)

    sources = [module_path.resolve()]
    for declared in spec.files:
        path = Path(declared).expanduser()
        if not path.is_absolute():
            path = source_cwd / path
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

    copied: dict[Path, Path] = {}
    for source in sources:
        relative = _bundle_relative_path(source, source_cwd)
        _copy_deployment_source(source, bundle_root / relative)
        copied[source] = relative

    workflow_relative = copied[module_path.resolve()]
    profile["source_cwd"] = str(source_cwd)
    profile["source_workflow"] = source_workflow
    profile["cwd"] = str(bundle_root)
    profile["workflow"] = str(workflow_relative) + (
        f":{workflow_name}" if separator else ""
    )
    profile["bundle"] = str(bundle_root)
    profile["bundled_files"] = [str(path) for path in copied.values()]


def _zippergen_install_requirement(
    *,
    extras: tuple[str, ...] = (),
) -> str:
    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "pyproject.toml").exists():
        requirement = str(project_root)
    else:
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


def _zippergen_runtime_provenance() -> dict[str, str]:
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
    raw = profile.get("connectors") or {}
    bindings = raw if isinstance(raw, dict) else {}
    if any(
        isinstance(value, dict)
        and value.get("kind") in {"gmail", "google-sheets"}
        for value in bindings.values()
    ):
        return ("google",)
    return ()


def prepare_deployment_environment(
    profile: dict[str, object],
    spec: DeploymentSpec,
    *,
    skip_install: bool,
) -> None:
    """Build and atomically install a deployment's private environment."""

    requirements = [package.requirement for package in spec.packages]
    profile["packages"] = requirements
    zippergen_extras = _deployment_zippergen_extras(profile)
    profile["zippergen_extras"] = list(zippergen_extras)
    profile["zippergen_runtime"] = _zippergen_runtime_provenance()
    if skip_install:
        profile["python"] = str(profile.get("python") or sys.executable)
        return

    name = str(profile["name"])
    environment_dir = deployment_environment_dir(name)
    environment_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{slug(name)}-building-",
            dir=environment_dir.parent,
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

    replaced: Path | None = None
    try:
        if environment_dir.exists() or environment_dir.is_symlink():
            replaced = environment_dir.with_name(
                f".{environment_dir.name}-replaced-"
                f"{time.strftime('%Y%m%d-%H%M%S')}-"
                f"{time.time_ns() % 1_000_000_000:09d}"
            )
            os.replace(environment_dir, replaced)
        os.replace(build_dir, environment_dir)
    except OSError as exc:
        if replaced is not None and replaced.exists():
            os.replace(replaced, environment_dir)
        shutil.rmtree(build_dir, ignore_errors=True)
        raise SystemExit(
            "Managed environment was built but could not replace "
            f"{environment_dir}: {exc}. The previous environment was restored."
        ) from None
    if replaced is not None:
        shutil.rmtree(replaced, ignore_errors=True)

    profile["python"] = str(_deployment_python_path(environment_dir))
    profile["environment_dir"] = str(environment_dir)


__all__ = ["bundle_deployment", "prepare_deployment_environment"]
