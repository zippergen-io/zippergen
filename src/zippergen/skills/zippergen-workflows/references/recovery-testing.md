# Testing recovery with fake services

Use this recipe when an external effect must survive a crash. It uses no
accounts, credentials, network calls or deployed services. Copy the two Python
blocks into the named files in a temporary test directory. Run them with an
environment that already has ZipperGen and pytest installed:

```bash
python -m pytest -q test_recovery.py
```

The fake service has its own SQLite database, separate from ZipperGen's store.
It supports a stable request ID and commits the result before the workflow
process is deliberately killed. On resume, the effect is called again. The
test checks that there were two attempts but only one external result, that
the original draft and approval were restored, and that rejection sends
nothing. Both outcomes are exercised.

This proves recovery for an **idempotent fake service**. Gmail sends and
Telegram announcements do not acquire this property just by being wrapped in
an effect. For those operations, test the application's uncertain-outcome
handling instead. See [connector helpers](connector-helpers.md).

## recovery_workflow.py

```python
from contextlib import closing
import os
from pathlib import Path
import sqlite3

from zippergen import Lifeline, effect, human, llm, pure, workflow

Mailbox = Lifeline("Mailbox")
Owner = Lifeline("Owner")


@llm(system="Draft a reply.", user="{request_id}", parse="text",
     outputs=[("body", str)])
def draft_reply(request_id: str) -> None: ...


@human(kind="confirm", context="{body}", outputs=["approved: bool"])
def approve_reply(body: str) -> None: ...


@effect
def send_once(directory: str, request_id: str, body: str) -> str:
    # A fake external service with server-side idempotency, not a local
    # ledger wrapped around an actual Gmail send.
    with closing(sqlite3.connect(Path(directory) / "external.sqlite")) as db:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS replies (id TEXT PRIMARY KEY, body TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS attempts (id TEXT)")
            db.execute("INSERT INTO attempts VALUES (?)", (request_id,))
            db.execute("INSERT OR IGNORE INTO replies VALUES (?, ?)", (request_id, body))
            saved = db.execute("SELECT body FROM replies WHERE id=?", (request_id,)).fetchone()
            if saved != (body,):
                raise ValueError("Request ID reused with different content")
    # The external transaction committed, but the action has not returned.
    if os.environ.get("RECOVERY_TEST_CRASH") == "yes":
        os._exit(99)
    return "sent"


@pure
def declined() -> str:
    return "rejected"


@workflow
def delivery(directory: str @ Mailbox, request_id: str @ Mailbox) -> str:
    Mailbox: body = draft_reply(request_id)
    Mailbox(body) >> Owner(body)
    Owner: approved = approve_reply(body)
    Owner(approved) >> Mailbox(approved)
    if approved @ Mailbox:
        Mailbox: outcome = send_once(directory, request_id, body)
    else:
        Mailbox: outcome = declined()
    return outcome @ Mailbox
```

## test_recovery.py

```python
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

DRIVER = r"""
import json
import os
from pathlib import Path
import sys

from zippergen.sqlite_runner import LocalSupervisor
from zippergen.workflow_io import load_workflow_spec

spec, directory, phase, answer = sys.argv[1:]
os.environ['RECOVERY_TEST_CRASH'] = 'yes' if phase == 'crash' else 'no'
wf, module = load_workflow_spec(spec)

def model(action, inputs):
    if phase == 'resume':
        raise AssertionError('The draft was already persisted')
    return {'body': 'Reply for ' + inputs['request_id']}

def human(action, inputs):
    if phase == 'resume':
        raise AssertionError('Approval was already persisted')
    # HumanAction has one .output, unlike LLMAction.outputs.
    return {action.output: answer == 'yes'}

runner = LocalSupervisor(
    wf, lifelines=None,
    initial_envs={'Mailbox': {'directory': directory, 'request_id': 'request-1'}},
    store_path=str(Path(directory) / 'run.sqlite'),
    llm_backend=model, human_backend=human, timeout=5,
)
print(json.dumps(runner.run()))
"""


@pytest.mark.parametrize("approved", [True, False])
def test_recovery(tmp_path, approved):
    spec = str(Path(__file__).with_name("recovery_workflow.py")) + ":delivery"

    def run(phase):
        return subprocess.run(
            [sys.executable, "-c", DRIVER, spec, str(tmp_path), phase,
             "yes" if approved else "no"],
            cwd=tmp_path, capture_output=True, text=True, timeout=15,
        )

    first = run("crash")
    assert first.returncode == (99 if approved else 0), first.stdout + first.stderr
    external = tmp_path / "external.sqlite"
    expected = "sent" if approved else "rejected"
    if approved:
        with closing(sqlite3.connect(external)) as db:
            assert db.execute("SELECT * FROM replies").fetchall() == [
                ("request-1", "Reply for request-1")
            ]
            assert db.execute("SELECT count(*) FROM attempts").fetchone() == (1,)
    else:
        assert json.loads(first.stdout) == expected
        assert not external.exists()

    # The first resume replays the interrupted effect. A second resume of
    # the completed run must return the saved result without repeating it.
    for _ in range(2):
        resumed = run("resume")
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert json.loads(resumed.stdout) == expected
    if approved:
        with closing(sqlite3.connect(external)) as db:
            assert db.execute("SELECT * FROM replies").fetchall() == [
                ("request-1", "Reply for request-1")
            ]
            assert db.execute("SELECT count(*) FROM attempts").fetchone() == (2,)
    else:
        assert not external.exists()
```

## Adapting the recipe

`LocalSupervisor(wf, lifelines, initial_envs, *, store_path, ...)` requires the
first three arguments. `lifelines=None` selects all workflow participants.
`initial_envs` maps participant names to input dictionaries. Use `None` when
there are no required inputs. Resume with the same workflow and store, rather
than deleting the store or changing the protocol between attempts.

Backends receive `(action, inputs)` and return a dictionary keyed by the
action's declared output names, not the workflow variable names at the call
site. `HumanAction` exposes `.output` and `.output_type` for its single output.
`LLMAction.outputs` is a tuple of `(name, type)` pairs. The fake human backend
above tests workflow decisions, not Telegram routing, authorization or durable
pending-task delivery. Test those separately only when your change affects them.

Use durable fake-service state that survives the subprocess. An in-memory
mock or a Python module global cannot establish recovery behavior. Inject
`os._exit` only in a disposable child process, with an expected exit code and
bounded subprocess timeout. Also test claim/finalize boundaries if your
application has them. For a non-idempotent send, assert that recovery reports
the uncertain operation instead of repeating it automatically.

For structural assertions, `workflow_semantics(wf)` from `zippergen.semantic`
returns `action_sites` as a **list of dictionaries**. Each site includes
`lifeline`, `action`, `kind`, `inputs`, `outputs`, `context` and `code`.
`action` is the action name string, and `outputs` lists workflow variable names.
Use these records to locate a site, then test behavior with fake backends.
Do not treat finding an approval site as proof that every send requires it.
