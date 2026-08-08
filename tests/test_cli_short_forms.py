"""The common path should be short enough not to undercut the story.

A first tutorial that says "just tell the coding agent what you want" is
undermined if every check then costs a workflow spec, an execution-mode flag
and an inline input. A single-workflow project already records its entry point,
so these commands infer it.

    zippergen validate workflow.py:email_approval --execution memory ...
    zg validate
"""

import shutil
from pathlib import Path

import pytest

from zippergen import serve
from zippergen.workspace import Workspace

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE = EXAMPLES / "email_approval.py"
# The tutorial workflow watches a mailbox and never ends on its own, so every
# run of it here carries a message budget.
BUDGET = ["--option", "max_messages=1"]


def _mailbox(root, *messages):
    box = root / "mailbox"
    box.mkdir(exist_ok=True)
    for index, text in enumerate(messages, start=1):
        (box / f"{index:02d}.txt").write_text(text, encoding="utf-8")
    return box


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "email-approval"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="email-approval")
    workspace.select_workflow("workflow.py:email_approval", cwd=root)
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    return root


@pytest.fixture
def input_project(tmp_path, monkeypatch):
    """A workflow that actually takes an input, for the prompting tests."""

    root = tmp_path / "hello"
    root.mkdir()
    shutil.copy(EXAMPLES / "hello.py", root / "workflow.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="hello")
    workspace.select_workflow("workflow.py:hello", cwd=root)
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    return root


def test_validate_infers_the_project_workflow(project, capsys):
    assert serve.main(["validate"]) == 0

    assert "email_approval: valid" in capsys.readouterr().out


def test_show_infers_the_project_workflow(project, capsys):
    assert serve.main(["show", "--agent", "Writer"]) == 0

    rendered = capsys.readouterr().out
    assert "email_approval__Writer" in rendered
    # The Writer takes no part in the User's decision, so it has no branch.
    assert "if " not in rendered


def test_run_infers_the_project_workflow(project, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    _mailbox(project, "Could we move our meeting?")

    rc = serve.main(["run", "--llm", "mock", "--yes", *BUDGET])

    assert rc == 0
    assert "result" in capsys.readouterr().out


def test_an_explicit_spec_still_wins(project, capsys):
    assert serve.main(["validate", "workflow.py:email_approval"]) == 0

    assert "email_approval: valid" in capsys.readouterr().out


def test_a_run_leaves_nothing_behind_by_default(project, monkeypatch, capsys):
    """`--execution memory` was an implementation detail in the reader's way."""

    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    _mailbox(project, "Could we move our meeting?")

    serve.main(["run", "--llm", "mock", "--yes", *BUDGET])

    assert "Store:" not in capsys.readouterr().err
    assert not list(project.glob("*.sqlite"))


def test_a_missing_input_is_asked_for_in_a_terminal(
    input_project, monkeypatch, capsys
):
    """Rather than refusing with a usage error."""

    monkeypatch.setattr(serve.sys.stdin, "isatty", lambda: True)
    answers = iter(["deployment", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    rc = serve.main(["run", "--llm", "mock"])

    assert rc == 0
    assert "result" in capsys.readouterr().out


def test_a_missing_input_outside_a_terminal_still_refuses_clearly(
    input_project, monkeypatch
):
    monkeypatch.setattr(serve.sys.stdin, "isatty", lambda: False)

    with pytest.raises(Exception) as error:
        serve.main(["run", "--llm", "mock"])

    assert "topic" in str(error.value)


def test_a_directory_without_a_project_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    with pytest.raises(SystemExit, match="none was found in this project"):
        serve.main(["validate"])


def test_diff_compares_a_baseline_against_the_project_workflow(
    project, tmp_path, capsys
):
    """One saved baseline plus one argument is the whole change-check ritual."""

    baseline = tmp_path / "before.json"
    assert serve.main(["snapshot", "-o", str(baseline)]) == 0
    capsys.readouterr()

    workflow = project / "workflow.py"
    workflow.write_text(
        workflow.read_text().replace(
            "Writer(draft) >> User(draft)",
            "Writer(draft) >> User(draft)\n        User(draft) >> Writer(draft)",
            1,
        )
    )

    assert serve.main(["diff", str(baseline)]) == 0

    assert "User(draft) >> Writer(draft)" in capsys.readouterr().out


def test_a_single_workflow_is_inferred_without_a_manifest_entry(
    tmp_path, monkeypatch, capsys
):
    """`zg init` runs before any workflow exists, so it cannot record one.

    A beginner should reach `zg validate` without first hand-writing
    workflow_entry into zippergen.toml.
    """

    root = tmp_path / "beginner"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="beginner")
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    assert serve.main(["validate"]) == 0

    assert "email_approval: valid" in capsys.readouterr().out
    # Inference is a convenience; it must not quietly rewrite the manifest.
    assert "workflow_entry" not in workspace.manifest_path.read_text()


def test_several_workflows_ask_for_an_explicit_choice(tmp_path, monkeypatch):
    root = tmp_path / "ambiguous"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    shutil.copy(EXAMPLE, root / "second.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="ambiguous")
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    with pytest.raises(SystemExit, match="several workflows"):
        serve.main(["validate"])


def test_a_manifest_entry_wins_over_discovery(tmp_path, monkeypatch, capsys):
    root = tmp_path / "explicit"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "workflow.py")
    shutil.copy(EXAMPLE, root / "second.py")
    workspace = Workspace(root, home=tmp_path / "home")
    workspace.initialize_project(name="explicit")
    workspace.select_workflow("second.py:email_approval", cwd=root)
    monkeypatch.chdir(root)
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))

    assert serve.main(["validate"]) == 0

    assert "email_approval: valid" in capsys.readouterr().out
