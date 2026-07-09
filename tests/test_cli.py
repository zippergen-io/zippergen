import json

from zippergen.serve import main


WORKFLOW_SOURCE = """
from zippergen import Lifeline, pure, workflow

User = Lifeline("User")

@pure
def add_suffix(topic: str) -> str:
    return topic + "!"

@workflow
def hello(topic: str @ User) -> str:
    User: reply = add_suffix(topic)
    return reply @ User
"""

SETUP_WORKFLOW_SOURCE = """
from zippergen import Lifeline, pure, workflow

User = Lifeline("User")
PREFIX = ""

def zippergen_setup(config):
    global PREFIX
    services = config.option("services", "fake")
    prefix = config.option("prefix", "")
    PREFIX = f"{services}:{prefix}:"

@pure
def add_prefix(topic: str) -> str:
    return PREFIX + topic

@workflow
def setup_hello(topic: str @ User) -> str:
    User: reply = add_prefix(topic)
    return reply @ User
"""


def test_run_command_loads_workflow_from_path(tmp_path, capsys):
    workflow_path = tmp_path / "sample_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    store_path = tmp_path / "run.sqlite"

    rc = main([
        "run",
        f"{workflow_path}:hello",
        "--store",
        str(store_path),
        "--input",
        "topic=deploy",
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert store_path.exists()
    assert f"Store: {store_path}" in captured.err
    assert json.loads(captured.out) == {"result": "deploy!"}


def test_run_command_loads_workflow_from_module(tmp_path, monkeypatch, capsys):
    workflow_path = tmp_path / "sample_module_workflow.py"
    workflow_path.write_text(WORKFLOW_SOURCE)
    store_path = tmp_path / "module-run.sqlite"
    monkeypatch.syspath_prepend(str(tmp_path))

    rc = main([
        "run",
        "sample_module_workflow:hello",
        "--store",
        str(store_path),
        "--input-json",
        '{"topic": "local"}',
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert store_path.exists()
    assert json.loads(captured.out) == {"result": "local!"}


def test_run_command_calls_setup_hook_with_options(tmp_path, capsys):
    workflow_path = tmp_path / "setup_workflow.py"
    workflow_path.write_text(SETUP_WORKFLOW_SOURCE)
    store_path = tmp_path / "setup-run.sqlite"

    rc = main([
        "run",
        f"{workflow_path}:setup_hello",
        "--store",
        str(store_path),
        "--input",
        "topic=deploy",
        "--option",
        "prefix=hook",
        "--services",
        "live",
        "--timeout",
        "10",
    ])

    captured = capsys.readouterr()
    assert rc == 0
    assert store_path.exists()
    assert json.loads(captured.out) == {"result": "live:hook:deploy"}
