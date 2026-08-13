import json
import os
import shutil
from io import StringIO
from pathlib import Path

import pytest

from zippergen.live_display import _screen_lines, watch_frames
from zippergen.locator import resolve_path, statement_node_paths
from zippergen.projection import project
from zippergen.serve import load_workflow_spec, main
from zippergen.control import encode_control
from zippergen.store import open_store, write_role_state
from zippergen.syntax import ActStmt, _ordered_workflow_lifelines
from zippergen.workspace import Workspace


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "email_approval.py"


def _observed_run(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    home = tmp_path / "home"
    workspace = Workspace(root, home=home)
    workspace.initialize_project(name="inspection")
    record = workspace.new_run(
        workflow_spec="workflow.py:email_approval",
        workflow_name="email_approval",
        fingerprint="test",
        inputs={"request": "hello"},
        llm="mock",
    )
    workflow, _module = load_workflow_spec(
        str(root / "workflow.py") + ":email_approval"
    )
    writer = next(
        participant
        for participant in _ordered_workflow_lifelines(workflow)
        if participant.name == "Writer"
    )
    local = project(workflow, writer)
    action_path = next(
        path
        for path in statement_node_paths(local).values()
        if isinstance(resolve_path(local, path), ActStmt)
    )
    node = resolve_path(local, action_path)
    connection = open_store(str(record["store"]))
    connection.execute("BEGIN IMMEDIATE")
    write_role_state(
        connection,
        "Writer",
        env={},
        control=encode_control(local, node),
        monitor=None,
        steps=1,
        status="running_model",
        detail={"action": "draft_reply", "kind": "model"},
    )
    connection.execute("COMMIT")
    connection.close()
    workspace.update_run(record["run_id"], status="interrupted")
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(home))
    return record


def test_inspect_shows_the_current_local_program_pointer(
    tmp_path, monkeypatch, capsys
):
    _observed_run(tmp_path, monkeypatch)

    assert main(["run", "inspect", "--agent", "Writer"]) == 0

    output = capsys.readouterr().out
    assert "Execution positions" in output
    assert "Writer" in output
    assert "running model action" in output
    assert "▶" in output
    assert "draft_reply" in output
    assert "request" not in output


def test_run_reset_archives_state_and_clears_the_current_run(
    tmp_path, monkeypatch, capsys
):
    record = _observed_run(tmp_path, monkeypatch)
    store = Path(str(record["store"]))
    assert store.is_file()

    assert main(["run", "reset", "--yes"]) == 0

    output = capsys.readouterr().out
    assert f"Reset durable run: {record['run_id']}" in output
    assert "Current durable run: none" in output
    assert not store.exists()
    workspace = Workspace()
    assert workspace.current_run() is None
    archives = list(
        (workspace.home / "trash" / "runs").glob(
            f"{record['run_id']}-*"
        )
    )
    assert len(archives) == 1
    assert (archives[0] / store.name).is_file()
    assert (archives[0] / f"{record['run_id']}.json").is_file()
    archived = open_store(str(archives[0] / store.name))
    assert archived.execute("SELECT COUNT(*) FROM role_state").fetchone() == (1,)
    archived.close()

    with pytest.raises(SystemExit, match="There is no current durable run"):
        main(["run", "inspect", "--agent", "Writer"])


def test_run_reset_requires_an_active_foreground_run_to_stop_first(
    tmp_path, monkeypatch
):
    record = _observed_run(tmp_path, monkeypatch)
    workspace = Workspace()
    workspace.update_run(record["run_id"], status="running")

    with pytest.raises(
        SystemExit,
        match="Stop its foreground process with Ctrl-C",
    ):
        main(["run", "reset", "--yes"])

    assert Path(str(record["store"])).is_file()

    assert main(["run", "reset", "--yes", "--force"]) == 0
    assert not Path(str(record["store"])).exists()
    assert Workspace().current_run() is None


def test_inspect_has_a_machine_readable_view(tmp_path, monkeypatch, capsys):
    record = _observed_run(tmp_path, monkeypatch)

    assert main(["run", "inspect", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["subject"] == f"run {record['run_id']}"
    assert payload["focus"] == "Writer"
    writer = next(
        item for item in payload["positions"]
        if item["participant"] == "Writer"
    )
    assert writer["state"] == "running_model"
    assert writer["detail"] == {
        "action": "draft_reply",
        "kind": "model",
    }


def test_inspect_selects_the_projects_unnamed_deployment_explicitly(
    tmp_path, monkeypatch, capsys
):
    record = _observed_run(tmp_path, monkeypatch)
    workspace = Workspace()
    name = workspace.directory.name
    deployments = workspace.home / "deployments"
    deployments.mkdir(parents=True)
    (deployments / f"{name}.json").write_text(json.dumps({
        "name": name,
        "project_id": workspace.project_manifest().get("project_id"),
        "source_cwd": str(workspace.root),
        "cwd": str(workspace.root),
        "workflow": "workflow.py:email_approval",
        "store": record["store"],
    }))

    assert main(["deploy", "inspect", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["subject"] == "project deployment"
    assert payload["store"] == record["store"]


def test_inspect_watch_refreshes_in_place_without_interrupting_execution(
    tmp_path, monkeypatch, capsys
):
    record = _observed_run(tmp_path, monkeypatch)
    frames: list[str] = []

    monkeypatch.setattr(
        "zippergen.live_display.live_display_available",
        lambda: True,
    )

    def capture(frame, *, interval):
        assert interval == 0.25
        frames.append(frame(100))
        return True

    monkeypatch.setattr("zippergen.live_display.watch_frames", capture)

    assert main(["run", "inspect", "--watch", "--interval", "0.25"]) == 0

    assert len(frames) == 1
    assert "Execution positions" in frames[0]
    assert "\n ▶" in frames[0]
    assert "draft_reply" in frames[0]
    output = capsys.readouterr().out
    assert f"Stopped watching run {record['run_id']}." in output
    assert "The execution was not interrupted." in output


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--watch", "--json"], "Use either --watch or --json"),
        (["--interval", "2"], "--interval requires --watch"),
        (["--watch", "--interval", "0"], "must be a positive number"),
    ],
)
def test_inspect_rejects_incoherent_watch_options(
    tmp_path, monkeypatch, arguments, message
):
    _observed_run(tmp_path, monkeypatch)

    with pytest.raises(SystemExit, match=message):
        main(["run", "inspect", *arguments])


def test_inspect_watch_requires_a_terminal(tmp_path, monkeypatch):
    _observed_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "zippergen.live_display.live_display_available",
        lambda: False,
    )

    with pytest.raises(SystemExit, match="requires an interactive terminal"):
        main(["run", "inspect", "--watch"])


def test_live_display_changes_only_rows_that_changed():
    stream = StringIO()
    frames = iter(["stable\npointer one", "stable\npointer two"])
    sleeps = 0

    def pause(_interval):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise KeyboardInterrupt

    assert watch_frames(
        lambda _columns: next(frames),
        interval=1,
        stream=stream,
        terminal_size=lambda: os.terminal_size((80, 24)),
        sleep=pause,
    )

    output = stream.getvalue()
    assert output.count("stable") == 1
    assert output.count("pointer one") == 1
    assert output.count("pointer two") == 1
    assert "\033[?1049h" in output
    assert output.endswith("\033[?25h\033[?1049l")


def test_live_display_keeps_the_program_pointer_in_a_short_terminal():
    frame = "\n".join(
        [
            "Execution positions",
            "Participants",
            "▶ Writer waiting",
            "Writer local projection",
            "=======================",
            *[f"  line {index}" for index in range(20)],
            " ▶ current line",
            *[f"  tail {index}" for index in range(20)],
        ]
    )

    visible = _screen_lines(frame, 80, 12)

    assert len(visible) <= 11
    assert any("▶ current line" in line for line in visible)
    assert "Writer local projection" in visible


def test_live_display_restores_the_terminal_after_an_error():
    stream = StringIO()

    def fail(_columns):
        raise RuntimeError("inspection failed")

    with pytest.raises(RuntimeError, match="inspection failed"):
        watch_frames(
            fail,
            stream=stream,
            terminal_size=lambda: os.terminal_size((80, 24)),
        )

    assert stream.getvalue().endswith("\033[?25h\033[?1049l")
