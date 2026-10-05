"""Role model pins survive routing, CLI isolation, and durable recovery."""

import json
import subprocess
import tomllib

import pytest

from zippergen import assistant, make_cli_assistant_backend
from zippergen.assistant_backends import (
    ASSISTANT_BACKEND_SPECS, assistant_settings, check_cli_assistant,
    normalize_assistant_selection,
)
from zippergen.assistant_configuration import (
    apply_assistant_overrides, project_assistant_routing,
)
from zippergen.serve import main, _run_args_from_deployment
from zippergen.workflow_io import load_workflow_spec
from zippergen.workspace import WorkspaceError
from tests.test_project_assistant_routing import (
    _project, _successful_cli_stdout, _supported, _one_deployment,
)


@assistant(instructions="Inspect the request.")
def inspect(request: str) -> str: ...


@assistant(instructions="Implement the result.", access="write")
def implement(request: str) -> str: ...


def _assert_pins(command, backend, model, effort):
    assert command[command.index("--model") + 1] == model
    if backend == "codex":
        assert "model_reasoning_effort=" + json.dumps(effort) in command
        assert "--ignore-user-config" in command
    else:
        assert command[command.index("--effort") + 1] == effort
        assert "--safe-mode" in command


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_pins_are_passed_on_every_turn_and_resume(tmp_path, monkeypatch, backend):
    commands = []
    monkeypatch.setattr("zippergen.assistant_backends.shutil.which", lambda name: f"/tools/{name}")

    def cli(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=_successful_cli_stdout(command))

    monkeypatch.setattr("zippergen.assistant_backends.subprocess.run", cli)
    selected = {"backend": backend, "model": "exact-model", "reasoning_effort": "high"}
    action_selection = {"backend": backend, "model": "review-model", "reasoning_effort": "low"}
    adapter = make_cli_assistant_backend(
        selected, project_root=tmp_path,
        routes={"MainThread.inspect": action_selection},
    )
    adapter(implement, {"request": "first"})
    adapter(inspect, {"request": "second"})
    adapter(implement, {"request": "third"})
    for command, model, effort in zip(commands, ["exact-model", "review-model", "exact-model"], ["high", "low", "high"]):
        _assert_pins(command, backend, model, effort)
    flag = "resume" if backend == "codex" else "--resume"
    assert flag not in commands[0]
    assert flag in commands[1] and flag in commands[2]
    if backend == "codex":
        assert commands[1].index("resume") < commands[1].index("--model")


def test_named_role_settings_round_trip_and_are_visible(tmp_path, monkeypatch, capsys):
    root, _home, workspace = _project(tmp_path, monkeypatch)
    checks = []

    def supported(backend, **kwargs):
        checks.append((backend, kwargs.get("settings")))
        return _supported(backend)

    monkeypatch.setattr("zippergen.assistant_backends.check_cli_assistant", supported)
    assert main(["assistant", "configure", "coding", "codex", "--model", "exact-sol", "--reasoning-effort", "high"]) == 0
    assert main(["assistant", "configure", "reviewer", "claude", "--model", "exact-opus", "--effort", "max"]) == 0
    assert main(["assistant", "assign", "Developer", "coding"]) == 0
    assert main(["assistant", "assign", "Reviewer", "reviewer"]) == 0
    capsys.readouterr()
    manifest = tomllib.loads((root / "zippergen.toml").read_text())
    assert manifest["assistants"]["configurations"] == workspace.assistant_configurations() == {
        "coding": {"backend": "codex", "model": "exact-sol", "reasoning_effort": "high"},
        "reviewer": {"backend": "claude", "model": "exact-opus", "reasoning_effort": "max"},
    }
    assert main(["config", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    rows = {row["target"]: row for row in report["assistants"]["resolved"]}
    assert rows["Developer.implement"]["model"] == "exact-sol"
    assert rows["Reviewer.review"]["reasoning_effort"] == "max"
    assert ("claude", {"model": "exact-opus", "reasoning_effort": "max"}) in checks
    assert main(["assistant"]) == 0
    output = capsys.readouterr().out
    assert "exact-sol" in output and "exact-opus" in output and "max" in output


def test_action_assignment_selects_its_whole_configuration(tmp_path, monkeypatch):
    root, _home, workspace = _project(tmp_path, monkeypatch)
    workspace.save_assistant_configuration("default", "claude", model="default-model", reasoning_effort="high")
    workspace.save_assistant_configuration("role", "codex", model="role-model", reasoning_effort="low")
    workspace.save_assistant_configuration("action", "claude", model="action-model", reasoning_effort="max")
    workspace.save_assistant_assignment_profile(
        "workflow.py:maintenance", default="default",
        lifelines={"Developer": "role"}, actions={"Developer.implement": "action"},
    )
    wf, module = load_workflow_spec(str(root / "workflow.py") + ":maintenance")
    routing = project_assistant_routing(workspace, "workflow.py:maintenance", wf, module=module)
    assert routing.default_backend == {"backend": "claude", "model": "default-model", "reasoning_effort": "high"}
    assert routing.overrides["Developer"] == {"backend": "codex", "model": "role-model", "reasoning_effort": "low"}
    assert routing.overrides["Developer.implement"] == {"backend": "claude", "model": "action-model", "reasoning_effort": "max"}
    overridden = apply_assistant_overrides(routing, workflow=wf, module=module, overrides={"Developer.implement": "codex"})
    assert overridden.overrides["Developer.implement"] == "codex"
    assert overridden.default_backend == routing.default_backend


@pytest.mark.parametrize("key,value", [("model", ""), ("model", 42), ("model", "--other-option"), ("reasoning_effort", "two words"), ("reasoning_effort", True)])
def test_invalid_settings_do_not_mutate_the_manifest(tmp_path, monkeypatch, key, value):
    root, _home, workspace = _project(tmp_path, monkeypatch)
    before = (root / "zippergen.toml").read_bytes()
    with pytest.raises(WorkspaceError, match=key):
        workspace.save_assistant_configuration("broken", "codex", **{key: value})
    assert (root / "zippergen.toml").read_bytes() == before


def test_unknown_settings_fail_instead_of_disappearing():
    with pytest.raises(ValueError, match="Unknown assistant setting"):
        assistant_settings({"backend": "codex", "reasoning_efort": "high"})
    assert normalize_assistant_selection("CODEX") == "codex"
    assert assistant_settings({"backend": "claude", "model": "future-model", "reasoning_effort": "HIGH"})["reasoning_effort"] == "high"


@pytest.mark.parametrize("spec", ASSISTANT_BACKEND_SPECS, ids=lambda spec: spec.name)
def test_check_requires_model_options_only_when_pinned(tmp_path, monkeypatch, spec):
    monkeypatch.setattr("zippergen.assistant_backends.shutil.which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(
        "zippergen.assistant_backends.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=" ".join(spec.required_options), stderr=""),
    )
    assert check_cli_assistant(spec.name).supported
    result = check_cli_assistant(spec.name, settings={"model": "pinned-model"})
    assert not result.supported and "--model" in result.detail


def test_plain_and_durable_runs_and_deployments_retain_pins(tmp_path, monkeypatch, capsys):
    root, home, workspace = _project(tmp_path, monkeypatch)
    default = {"backend": "codex", "model": "exact-sol", "reasoning_effort": "high"}
    reviewer = {"backend": "claude", "model": "exact-opus", "reasoning_effort": "max"}
    workspace.save_assistant_configuration("coding", **default)
    workspace.save_assistant_configuration("reviewer", **reviewer)
    workspace.save_assistant_assignment_profile("workflow.py:maintenance", default="coding", lifelines={"Reviewer": "reviewer"}, actions={})
    captured = []

    def factory(selected=None, *, project_root=None, routes=None):
        captured.append((selected, routes))
        return lambda action, inputs: {action.outputs[0][0]: "done"}

    monkeypatch.setattr("zippergen.assistant_backends.make_cli_assistant_backend", factory)
    for args in (["run", "--input", "request=change", "--yes"], ["run", "--durable", "--input", "request=change", "--yes"]):
        assert main(args) == 0
        assert captured[-1] == (default, {"Reviewer": reviewer})
    record = json.loads(next((home / "workspaces").glob("*/runs/*.json")).read_text())
    assert record["assistant"] == default and record["assistants"] == {"Reviewer": reviewer}
    monkeypatch.setattr("zippergen.serve._bundle_deployment", lambda *args: None)
    monkeypatch.setattr("zippergen.serve._prepare_deployment_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr("zippergen.serve._run_deployment_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr("zippergen.serve._doctor_checks", lambda *args, **kwargs: [])
    assert main(["deploy", "--set", "request=change", "--no-start", "--yes"]) == 0
    profile = json.loads(_one_deployment(home).read_text())
    assert profile["assistant"] == default and profile["assistants"] == {"Reviewer": reviewer}
    args = _run_args_from_deployment(profile)
    assert args.assistant == default and args.assistants == {"Reviewer": reviewer}


def test_durable_resume_uses_recorded_pins_after_project_configuration_changes(
    tmp_path, monkeypatch, capsys,
):
    from zippergen.durable_runs import run_durable

    _root, _home, workspace = _project(tmp_path, monkeypatch)
    workspace.save_assistant_configuration("coding", "codex", model="original-sol", reasoning_effort="high")
    workspace.save_assistant_configuration("reviewer", "claude", model="original-opus", reasoning_effort="max")
    workspace.save_assistant_assignment_profile("workflow.py:maintenance", default="coding", lifelines={"Reviewer": "reviewer"}, actions={})
    commands = []
    monkeypatch.setattr("zippergen.assistant_backends.shutil.which", lambda name: f"/tools/{name}")

    def cli(command, **kwargs):
        commands.append(command)
        if len(commands) == 2:
            return subprocess.CompletedProcess(command, 1, stderr="Interrupted reviewer", stdout="")
        return subprocess.CompletedProcess(command, 0, stdout=_successful_cli_stdout(command), stderr="")

    monkeypatch.setattr("zippergen.assistant_backends.subprocess.run", cli)
    with pytest.raises(RuntimeError, match="Interrupted reviewer"):
        run_durable(
            workspace, workflow_spec="workflow.py:maintenance",
            provided_inputs={"request": "change"}, timeout=5, interactive=False,
        )
    workspace.save_assistant_configuration("coding", "codex", model="changed-sol", reasoning_effort="low")
    workspace.save_assistant_configuration("reviewer", "claude", model="changed-opus", reasoning_effort="low")
    resumed = run_durable(workspace, resume=True, timeout=5)
    assert resumed["status"] == "done"
    assert len(commands) == 3
    _assert_pins(commands[0], "codex", "original-sol", "high")
    _assert_pins(commands[1], "claude", "original-opus", "max")
    _assert_pins(commands[2], "claude", "original-opus", "max")
