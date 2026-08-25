import subprocess
from pathlib import Path

import pytest

from zippergen.deployment import DeploymentSpec
from zippergen.deployment_environment import (
    bundle_deployment,
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


def test_managed_environment_uses_an_immutable_generation(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    # A directory ZipperGen does not own. It sits beside the generations and
    # must be left alone: only a recorded previous generation is removed.
    unowned = home / "environments" / "reviewed-answer"
    unowned.mkdir(parents=True)
    (unowned / "not-ours").write_text("left alone\n")
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
    managed = Path(str(profile["environment_dir"]))
    assert managed.parent == home / "environments" / ".releases" / "reviewed-answer"
    assert Path(str(profile["python"])) == managed / "bin" / "python"
    assert isinstance(profile["zippergen_runtime"], dict)
    assert (managed / "bin" / "python").read_text() == "managed python\n"
    assert (unowned / "not-ours").exists(), (
        "publishing a generation must not delete a directory ZipperGen "
        "never recorded as the previous one"
    )
    assert not list(managed.parent.glob(".*-building-*"))


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


def test_deferred_environment_generation_can_be_rolled_back(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    environment = home / "environments" / "reviewed-answer"
    environment.mkdir(parents=True)
    sentinel = environment / "old-environment"
    sentinel.write_text("still active\n")
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.setattr(
        "zippergen.deployment_environment.shutil.which",
        lambda name: "/tools/uv" if name == "uv" else None,
    )

    def fake_run(arguments, *, check):
        command = [str(value) for value in arguments]
        if command[1] == "venv":
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("candidate python\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "zippergen.deployment_environment.subprocess.run",
        fake_run,
    )
    profile: dict[str, object] = {"name": "reviewed-answer"}

    update = _prepare_deployment_environment(
        profile,
        DeploymentSpec(),
        skip_install=False,
        defer_cleanup=True,
    )

    assert update is not None
    candidate = update.environment
    assert sentinel.read_text() == "still active\n"
    assert (candidate / "bin" / "python").is_file()
    update.rollback()
    assert sentinel.read_text() == "still active\n"
    assert not candidate.exists()


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


def test_declared_directory_rejects_a_symlink_outside_its_root(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "project"
    root.mkdir()
    workflow_path = root / "workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    declared = root / "assets"
    declared.mkdir()
    outside = root / "private.txt"
    outside.write_text("do not bundle")
    try:
        (declared / "linked-secret").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.chdir(root)
    workflow, _module = load_workflow_spec("workflow.py:sample")
    profile: dict[str, object] = {
        "cwd": str(root),
        "workflow": "workflow.py:sample",
    }

    with pytest.raises(SystemExit, match="symlink"):
        deployment_source_provenance(
            profile,
            DeploymentSpec(files=("assets",)),
            workflow,
        )


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
    # A deployment is made of three things, and each reports whether what is
    # running matches the project: the runtime, the workflow bundle, and the
    # answers.
    assert [(item["name"], item["freshness"]) for item in current] == [
        ("ZipperGen runtime", "current"),
        ("workflow source", "current"),
        ("configuration", "current"),
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


def test_dotted_project_module_is_bundled_instead_of_left_mutable(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    package = root / "mailflow"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from .workflow import sample\n"
    )
    (package / "workflow.py").write_text(WORKFLOW_SOURCE)
    home = tmp_path / "home"
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    monkeypatch.syspath_prepend(str(root))
    workflow, _module = load_workflow_spec("mailflow.workflow:sample")
    profile: dict[str, object] = {
        "name": "mailflow",
        "cwd": str(root),
        "workflow": "mailflow.workflow:sample",
    }

    bundle_deployment(profile, DeploymentSpec(), workflow)

    bundle = Path(str(profile["bundle"]))
    assert profile["workflow"] == "mailflow.workflow:sample"
    assert (bundle / "mailflow" / "workflow.py").read_text() == WORKFLOW_SOURCE
    assert profile["workflow_source"]["kind"] == "source-bundle"


def test_dotted_module_outside_project_is_refused(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    root.mkdir()
    workflow_path = root / "workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    workflow, _module = load_workflow_spec(f"{workflow_path}:sample")
    profile: dict[str, object] = {
        "name": "external",
        "cwd": str(root),
        "workflow": "zippergen.serve:main",
    }

    with pytest.raises(SystemExit, match="resolves outside the project root"):
        bundle_deployment(profile, DeploymentSpec(), workflow)
