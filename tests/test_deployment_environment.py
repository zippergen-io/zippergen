import subprocess
from pathlib import Path

import pytest

from zippergen.deployment import DeploymentSpec
from zippergen.deployment_environment import (
    deployment_source_provenance,
    prepare_deployment_environment as _prepare_deployment_environment,
)
from zippergen.deployment_checks import deployment_freshness_checks
from zippergen.workflow_io import load_workflow_spec


WORKFLOW_SOURCE = """
from zippergen import Lifeline, workflow

Worker = Lifeline("Worker")

@workflow
def sample(value: str @ Worker) -> str:
    return value @ Worker
"""


def test_managed_environment_uses_uv_and_replaces_an_old_environment_atomically(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    environment = home / "environments" / "reviewed-answer"
    environment.mkdir(parents=True)
    (environment / "old-environment").write_text("preserve until success\n")
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(
        "zippergen.deployment_environment.shutil.which",
        lambda name: "/tools/uv" if name == "uv" else None,
    )
    calls: list[list[str]] = []

    def fake_run(arguments, *, check):
        command = [str(value) for value in arguments]
        calls.append(command)
        if command[1] == "venv":
            build = Path(command[-1])
            python = build / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("managed python\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "zippergen.deployment_environment.subprocess.run",
        fake_run,
    )
    profile: dict[str, object] = {"name": "reviewed-answer"}

    _prepare_deployment_environment(
        profile,
        DeploymentSpec(),
        skip_install=False,
    )

    assert calls[0][0:3] == ["/tools/uv", "venv", "--python"]
    assert calls[1][0:3] == ["/tools/uv", "pip", "install"]
    assert calls[1][3:5] == ["--refresh-package", "zippergen"]
    assert Path(str(profile["python"])) == environment / "bin" / "python"
    assert isinstance(profile["zippergen_runtime"], dict)
    assert (environment / "bin" / "python").read_text() == "managed python\n"
    assert not (environment / "old-environment").exists()
    assert not list((home / "environments").glob(".*-building-*"))
    assert not list((home / "environments").glob(".*-replaced-*"))


def test_failed_ensurepip_keeps_the_previous_environment_and_has_guidance(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    environment = home / "environments" / "reviewed-answer"
    environment.mkdir(parents=True)
    sentinel = environment / "existing-environment"
    sentinel.write_text("still usable\n")
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(
        "zippergen.deployment_environment.shutil.which",
        lambda _name: None,
    )

    class BrokenBuilder:
        def create(self, _target):
            raise subprocess.CalledProcessError(
                -6,
                ["python", "-m", "ensurepip"],
            )

    monkeypatch.setattr(
        "zippergen.deployment_environment.venv.EnvBuilder",
        lambda **_kwargs: BrokenBuilder(),
    )
    profile: dict[str, object] = {"name": "reviewed-answer"}

    with pytest.raises(SystemExit) as raised:
        _prepare_deployment_environment(
            profile,
            DeploymentSpec(),
            skip_install=False,
        )

    message = str(raised.value)
    assert "signal 6" in message
    assert "Install uv" in message
    assert "previous deployment environment" in message
    assert sentinel.read_text() == "still usable\n"
    assert not list((home / "environments").glob(".*-building-*"))


def test_google_connector_deployment_installs_the_optional_extra(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(
        "zippergen.deployment_environment.shutil.which",
        lambda name: "/tools/uv" if name == "uv" else None,
    )
    calls: list[list[str]] = []

    def fake_run(arguments, *, check):
        command = [str(value) for value in arguments]
        calls.append(command)
        if command[1] == "venv":
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("managed python\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "zippergen.deployment_environment.subprocess.run",
        fake_run,
    )
    profile: dict[str, object] = {
        "name": "google-service",
        "connectors": {
            "requirement:records": {
                "kind": "google-sheets",
            },
        },
    }

    _prepare_deployment_environment(
        profile,
        DeploymentSpec(),
        skip_install=False,
    )

    install_requirement = calls[1][-1]
    assert install_requirement.endswith("[google]")
    assert profile["zippergen_extras"] == ["google"]


def test_workflow_source_fingerprint_covers_only_bundle_inputs(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    workflow_path = root / "workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    prompt = root / "prompt.txt"
    prompt.write_text("first\n")
    manifest = root / "zippergen.toml"
    manifest.write_text('workflow_entry = "workflow.py:sample"\n')
    unrelated = root / "notes.txt"
    unrelated.write_text("one\n")
    monkeypatch.chdir(root)
    workflow, _module = load_workflow_spec("workflow.py:sample")
    profile: dict[str, object] = {
        "cwd": str(root),
        "workflow": "workflow.py:sample",
    }
    spec = DeploymentSpec(files=("prompt.txt",))

    first = deployment_source_provenance(profile, spec, workflow)
    unrelated.write_text("two\n")
    assert deployment_source_provenance(profile, spec, workflow) == first

    prompt.write_text("second\n")
    assert deployment_source_provenance(profile, spec, workflow) != first

    prompt.write_text("first\n")
    manifest.write_text(
        'workflow_entry = "workflow.py:sample"\n[models.assignments]\n'
    )
    assert deployment_source_provenance(profile, spec, workflow) != first


def test_freshness_distinguishes_runtime_and_workflow_source(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    root.mkdir()
    workflow_path = root / "workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    monkeypatch.chdir(root)
    workflow, _module = load_workflow_spec("workflow.py:sample")
    profile: dict[str, object] = {
        "cwd": str(root),
        "workflow": "workflow.py:sample",
        "source_cwd": str(root),
        "source_workflow": "workflow.py:sample",
        "zippergen_runtime": {"source_sha256": "runtime-a"},
    }
    profile["workflow_source"] = deployment_source_provenance(
        profile, DeploymentSpec(), workflow
    )
    monkeypatch.setattr(
        "zippergen.deployment_environment.zippergen_runtime_provenance",
        lambda: {"source_sha256": "runtime-a"},
    )

    current = deployment_freshness_checks(profile)
    assert [(item["name"], item["freshness"]) for item in current] == [
        ("ZipperGen runtime", "current"),
        ("workflow source", "current"),
    ]

    workflow_path.write_text(WORKFLOW_SOURCE + "\n# changed but not committed\n")
    stale = deployment_freshness_checks(profile)
    assert stale[0]["freshness"] == "current"
    assert stale[1]["freshness"] == "stale"
    assert "immutable workflow bundle" in stale[1]["detail"]
    assert "current source edits are not active" in stale[1]["detail"]

    monkeypatch.setattr(
        "zippergen.deployment_environment.zippergen_runtime_provenance",
        lambda: {"source_sha256": "runtime-b"},
    )
    runtime_stale = deployment_freshness_checks(profile)[0]
    assert runtime_stale["freshness"] == "stale"
    assert "fixes in the current checkout are not active" in runtime_stale["detail"]
    assert "cannot show its severity" in runtime_stale["detail"]
