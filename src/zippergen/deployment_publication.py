"""Publishing a deployment: its home, its artifacts, and its durable store.

Creating the managed home, writing the service script and profile, installing
the unit, and initialising the store are deployment operations, not argument
parsing. They lived in ``serve.py``, which is documented as owning parsing and
dispatch only -- so a contributor following that documentation would edit a
focused module and change nothing about what actually runs.

Ordering matters here and is not obvious: the managed home has to exist, with
the right permissions, before the readiness checks run against it. Two
first-deploy failures came from checks running ahead of preparation, which is
why these steps live together rather than being called ad hoc from the CLI.
"""

from __future__ import annotations

import json
import os
import plistlib
import shlex
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from zippergen.deployment_platform import (
    deployment_launchd_path as _deployment_launchd_path,
    deployment_profile_path as _deployment_profile_path,
    deployment_script_path as _deployment_script_path,
    deployment_service_path as _deployment_service_path,
    deployments_dir as _deployments_dir,
    installed_launchd_path as _installed_launchd_path,
    installed_systemd_service_path as _installed_systemd_service_path,
    launchd_label as _launchd_label,
    slug as _slug,
    zippergen_home as _zippergen_home,
)
from zippergen.private_files import (
    ensure_private_directory,
    write_private_bytes,
    write_private_text,
)
from zippergen.store import open_store, read_history_keep, write_history_keep
from zippergen.value_codec import encode_value


def _prepare_managed_home(profile: Mapping[str, object]) -> None:
    """Make everything ZipperGen owns under its home private, before it is read.

    The managed home is ZipperGen's workspace rather than part of any one
    deployment, so preparing it is a precondition for evaluating a candidate,
    not a step in publishing one. Doing it only at publication time is what
    made the readiness checks demand a state that only publication created, so
    a first deploy failed on the home, and a later one on a log file left
    world-readable by an earlier release.

    Directories and the log file are one job for that reason: the checks ask
    the same question of both, so both are answered in the same place.

    Called more than once per deploy on purpose: every step is idempotent.
    """

    home = _zippergen_home()
    ensure_private_directory(home)
    store_path = Path(str(profile["store"])).expanduser()
    log_path = Path(str(profile["log"])).expanduser()
    if log_path.is_symlink():
        raise SystemExit(f"Refusing a symlinked deployment log: {log_path}")
    for directory in (store_path.parent, log_path.parent, _deployments_dir()):
        directory.mkdir(parents=True, exist_ok=True)
        try:
            directory.resolve().relative_to(home.resolve())
        except ValueError:
            continue
        ensure_private_directory(directory)

    try:
        log_path.resolve().relative_to(home.resolve())
    except ValueError:
        return
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.close(log_fd)
    log_path.chmod(0o600)


def _deployment_command(name: str, *, python_executable: str | None = None) -> str:
    python = python_executable or sys.executable
    return (
        f"{shlex.quote(python)} -m zippergen.serve __launch-deployment "
        f"--profile {shlex.quote(_slug(name))}"
    )


def _write_deployment_artifacts(profile: dict[str, object]) -> None:
    name = str(profile["name"])
    profile_path = _deployment_profile_path(name)
    script_path = _deployment_script_path(name)
    service_path = _deployment_service_path(name)
    launchd_path = _deployment_launchd_path(name)
    _prepare_managed_home(profile)

    stored_profile = dict(profile)
    stored_profile["inputs"] = encode_value(profile.get("inputs") or {})
    write_private_text(
        script_path,
        "#!/bin/sh\n"
        "set -eu\n"
        f"cd {shlex.quote(str(_zippergen_home()))}\n"
        f"exec env ZIPPERGEN_HOME={shlex.quote(str(_zippergen_home()))} "
        f"{_deployment_command(name)}\n"
    )
    script_path.chmod(0o700)
    write_private_text(
        service_path,
        "[Unit]\n"
        f"Description=ZipperGen deployment {name}\n"
        "After=network-online.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        "UMask=0077\n"
        f"WorkingDirectory={_zippergen_home()}\n"
        f"ExecStart={script_path}\n"
        "Restart=on-failure\n"
        "RestartSec=10\n"
        f"StandardOutput=append:{profile['log']}\n"
        f"StandardError=append:{profile['log']}\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    launchd = {
        "Label": _launchd_label(name),
        "ProgramArguments": [str(script_path)],
        "WorkingDirectory": str(_zippergen_home()),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "StandardOutPath": str(profile["log"]),
        "StandardErrorPath": str(profile["log"]),
        "ProcessType": "Background",
        "Umask": 0o077,
    }
    write_private_bytes(launchd_path, plistlib.dumps(launchd, sort_keys=True))
    # The profile is the publication point: the service reads it to decide
    # which immutable bundle and configuration to run. Write every supporting
    # artifact first so a profile never points at files we have not finished.
    write_private_text(
        profile_path,
        json.dumps(stored_profile, indent=2, sort_keys=True) + "\n"
    )


def _install_launchd_agent(profile: dict[str, object], *, dry_run: bool = False) -> Path:
    name = str(profile["name"])
    source = _deployment_launchd_path(name)
    target = _installed_launchd_path(name)
    if dry_run:
        print(f"Install launchd agent: {source} -> {target}")
        return target
    _write_deployment_artifacts(profile)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_private_bytes(target, source.read_bytes())
    return target

def _install_systemd_unit(profile: dict[str, object], *, dry_run: bool = False) -> Path:
    name = str(profile["name"])
    source = _deployment_service_path(name)
    target = _installed_systemd_service_path(name)
    if dry_run:
        print(f"Install systemd unit: {source} -> {target}")
        return target
    _write_deployment_artifacts(profile)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_private_text(target, source.read_text())
    return target


def _profile_history_keep(profile: Mapping[str, object]) -> int | None:
    """The history budget this deployment asked for, if it asked for one."""

    raw = profile.get("history_keep")
    if raw is None:
        return None
    try:
        keep = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise SystemExit(
            f"Deployment history_keep is not a whole number: {raw!r}. "
            "Set it again with 'zg deploy --history-keep N'."
        ) from None
    if keep < 0:
        raise SystemExit(
            f"Deployment history_keep is negative: {keep}. "
            "Set it again with 'zg deploy --history-keep N'."
        )
    return keep


def _initialize_deployment_store(profile: dict[str, object]) -> bool:
    """Allocate one valid durable store for a deployment if it has none.

    An ordinary ``zg deploy`` writes a store only here, immediately after it
    creates it. Deploying is configuration while the store is state, so an
    ordinary redeploy never has to open incompatible recovery state; readiness
    checks report that state instead. The explicit ``--history-keep`` option is
    the narrow exception handled below because that option directly owns one
    store setting.

    A reset archives the old store and lands here with a fresh one, so the
    deployment's history budget is stamped on at creation. Without that, every
    reset would quietly put the trace back to the default.
    """

    path = Path(str(profile["store"])).expanduser()
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = open_store(str(path))
        keep = _profile_history_keep(profile)
        if keep is not None:
            write_history_keep(connection, keep)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"Could not initialize deployment store {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    connection.close()
    return True


def _apply_existing_history_keep(
    profile: dict[str, object],
    *,
    requested: int | None,
    store_created: bool,
) -> int | None:
    """Apply an explicit deploy-time budget to a store that already exists.

    A profile setting is the reset default; the store setting controls the
    running deployment. Recording only the former makes ``--history-keep`` look
    successful while leaving live behavior unchanged. Opening an existing
    store remains conditional on this explicit state-setting request, so an
    ordinary redeploy still never trips over incompatible recovery state.
    """

    if requested is None or store_created:
        return None
    path = Path(str(profile["store"])).expanduser()
    try:
        connection = open_store(str(path))
        try:
            previous = read_history_keep(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                write_history_keep(connection, requested)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()
    except Exception as exc:
        raise SystemExit(
            f"Could not apply --history-keep to existing store {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return previous


def _write_deployment_secrets(path: Path, values: dict[str, str]) -> None:
    ensure_private_directory(path.parent)
    write_private_text(
        path,
        json.dumps(values, indent=2, sort_keys=True) + "\n",
    )
