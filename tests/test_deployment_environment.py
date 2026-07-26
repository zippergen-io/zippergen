import subprocess
from pathlib import Path

import pytest

from zippergen.deployment import DeploymentSpec
from zippergen.serve import _prepare_deployment_environment


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
        "zippergen.serve.shutil.which",
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

    monkeypatch.setattr("zippergen.serve.subprocess.run", fake_run)
    profile: dict[str, object] = {"name": "reviewed-answer"}

    _prepare_deployment_environment(
        profile,
        DeploymentSpec(),
        skip_install=False,
    )

    assert calls[0][0:3] == ["/tools/uv", "venv", "--python"]
    assert calls[1][0:3] == ["/tools/uv", "pip", "install"]
    assert Path(str(profile["python"])) == environment / "bin" / "python"
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
    monkeypatch.setattr("zippergen.serve.shutil.which", lambda _name: None)

    class BrokenBuilder:
        def create(self, _target):
            raise subprocess.CalledProcessError(
                -6,
                ["python", "-m", "ensurepip"],
            )

    monkeypatch.setattr(
        "zippergen.serve.venv.EnvBuilder",
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
