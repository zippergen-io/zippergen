"""Opt-in lifecycle test against a real systemd user manager.

Run on the Linux deployment host before release with:

    ZIPPERGEN_RUN_SYSTEMD_INTEGRATION=1 pytest -q \
        tests/test_systemd_deploy_integration.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from zippergen.deployment_platform import slug
from zippergen.store import SCHEMA_VERSION
from zippergen.workspace import Workspace


pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or os.environ.get("ZIPPERGEN_RUN_SYSTEMD_INTEGRATION") != "1",
    reason="set ZIPPERGEN_RUN_SYSTEMD_INTEGRATION=1 on a Linux user-systemd host",
)


WORKFLOW = '''
import time
from zippergen import Lifeline, effect, workflow

Worker = Lifeline("Worker")

@effect(visible=False)
def wait_once() -> str:
    time.sleep(0.5)
    return "done"

@workflow
def lifecycle() -> str:
    Worker: result = wait_once()
    return result @ Worker
'''


def _run(project: Path, environment: dict[str, str], *arguments: str):
    return subprocess.run(
        [sys.executable, "-m", "zippergen.serve", *arguments],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
    )


def test_real_systemd_stop_upgrade_reset_start(tmp_path):
    probe = subprocess.run(
        ["systemctl", "--user", "show-environment"],
        text=True,
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.fail(
            "ZIPPERGEN_RUN_SYSTEMD_INTEGRATION=1 was set, but the systemd "
            f"user manager is unavailable: {probe.stderr.strip()}"
        )

    project = tmp_path / "systemd-upgrade"
    project.mkdir()
    (project / "workflow.py").write_text(WORKFLOW)
    home = tmp_path / "zippergen-home"
    workspace = Workspace(project, home=home)
    workspace.initialize_project(name="systemd-upgrade")
    workspace.select_workflow("workflow.py:lifecycle", cwd=project)
    environment = {
        **os.environ,
        "ZIPPERGEN_HOME": str(home),
        "ZIPPERGEN_SERVICE_MANAGER": "systemd",
    }
    name = workspace.directory.name

    try:
        deployed = _run(project, environment, "deploy", "--yes")
        assert deployed.returncode == 0, deployed.stdout + deployed.stderr

        stopped = _run(project, environment, "deploy", "stop")
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr

        profile_path = home / "deployments" / f"{slug(name)}.json"
        profile = json.loads(profile_path.read_text())
        store = Path(profile["store"])
        connection = sqlite3.connect(store)
        connection.execute(
            "UPDATE store_meta SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION - 1),),
        )
        connection.commit()
        connection.close()

        incompatible = _run(
            project, environment, "deploy", "--yes", "--no-start"
        )
        assert incompatible.returncode == 1, incompatible.stdout + incompatible.stderr
        assert "reset" in (incompatible.stdout + incompatible.stderr)

        reset = _run(project, environment, "deploy", "reset", "--yes")
        assert reset.returncode == 0, reset.stdout + reset.stderr

        redeployed = _run(
            project, environment, "deploy", "--yes", "--no-start"
        )
        assert redeployed.returncode == 0, redeployed.stdout + redeployed.stderr
        started = _run(project, environment, "deploy", "start")
        assert started.returncode == 0, started.stdout + started.stderr
    finally:
        _run(project, environment, "deploy", "stop")
        _run(project, environment, "deploy", "remove", "--purge", "--yes")
