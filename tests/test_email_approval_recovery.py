"""The tutorial inbox must survive both sides of its external I/O boundary."""

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from zippergen.store import load_role_state

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "email_approval.py"

RUNNER = """
import functools
import json
import os
from pathlib import Path
import sys

from zippergen.sqlite_runner import LocalSupervisor
from zippergen.workflow_io import load_workflow_spec

example, directory, phase, approved = sys.argv[1:]
root = Path(directory)
wf, module = load_workflow_spec(example + ':email_approval')
module._mailbox = root / 'mailbox'
module._max_messages = 1
module._poll_seconds = 0.01

if phase in {'read', 'complete'}:
    action = module.next_unread_message if phase == 'read' else module.complete_message
    original = action.fn

    @functools.wraps(original)
    def crash_after_effect(*args):
        original(*args)
        os._exit(99)

    object.__setattr__(action, 'fn', crash_after_effect)

if phase == 'after-complete':
    import zippergen.role_runner as role_runner
    original_commit = role_runner.RoleRunner._commit_state

    def crash_after_commit(self, *args, **kwargs):
        original_commit(self, *args, **kwargs)
        if self.role == 'Mailbox' and self.env.get('processed') == 1:
            os._exit(99)

    role_runner.RoleRunner._commit_state = crash_after_commit

def model(action, inputs):
    with (root / 'drafts.jsonl').open('a') as output:
        output.write(json.dumps(inputs['message']) + '\\n')
    return {'draft': 'A draft for ' + inputs['message']}

runner = LocalSupervisor(
    wf, None, None, store_path=str(root / 'run.sqlite'),
    llm_backend=model,
    human_backend=lambda action, inputs: {'approved': approved == 'yes'},
    timeout=5,
)
print(json.dumps({'result': runner.run()}))
"""


@pytest.mark.parametrize("phase", ["read", "complete", "after-complete"])
@pytest.mark.parametrize("approved", [True, False])
def test_tutorial_recovers_the_same_message_and_budget(tmp_path, phase, approved):
    mailbox = tmp_path / "mailbox"
    mailbox.mkdir()
    (mailbox / "01.txt").write_text("First request", encoding="utf-8")
    (mailbox / "02.txt").write_text("Second request", encoding="utf-8")
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

    def run(selected_phase):
        return subprocess.run(
            [sys.executable, "-c", RUNNER, str(EXAMPLE), str(tmp_path),
             selected_phase, "yes" if approved else "no"],
            cwd=tmp_path, env=environment, capture_output=True, text=True,
            timeout=15,
        )

    crashed = run(phase)
    assert crashed.returncode == 99, crashed.stdout + crashed.stderr
    assert (mailbox / "01.txt").exists() == (phase == "read")
    assert (mailbox / "02.txt").is_file()

    resumed = run("resume")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads(resumed.stdout.splitlines()[-1]) == {"result": int(approved)}
    assert (mailbox / "01.done").read_text(encoding="utf-8") == "First request"
    assert not (mailbox / "01.txt").exists()
    assert (mailbox / "02.txt").read_text(encoding="utf-8") == "Second request"
    assert (tmp_path / "drafts.jsonl").read_text().splitlines() == ['"First request"']
    with sqlite3.connect(tmp_path / "run.sqlite") as conn:
        state = load_role_state(conn, "Mailbox")
    assert state["env"]["processed"] == 1
