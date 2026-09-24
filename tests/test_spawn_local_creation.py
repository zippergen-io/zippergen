"""Retained endpoints create children without waiting for the initiator."""
from contextlib import closing
import subprocess
import sys
import threading

import pytest

from zippergen import Lifeline, Var, effect, fragment, pure, run, run_sqlite, spawn, workflow, skip
from zippergen.channels import InProcessChannel
from zippergen.control import program_fingerprint
from zippergen.projection import project, project_templates
from zippergen.sessions import fresh_endpoint
from zippergen.store import open_store, open_store_readonly
from zippergen.syntax import BoundForkStmt

Sender, Initiator, Worker, Auditor = map(Lifeline, ["Sender", "Initiator", "Worker", "Auditor"])
request = Var("request", int, default=7)
reply = Var("reply", int)
marker = Var("marker", int)
release_initiator = threading.Event()


@effect
def wait_for_early_send() -> int:
    if not release_initiator.wait(5):
        raise RuntimeError("retained role waited for the initiator")
    return 1


@pure
def increment(value: int) -> int:
    return value + 1


@fragment
def work():
    skip(Initiator)
    Sender(request) >> Worker(request)
    Worker: reply = increment(request)
    Worker(reply) >> Sender(reply)


@workflow
def early_retained():
    Initiator: marker = wait_for_early_send()
    spawn(work, at=Initiator, retain=(Sender, Initiator), fresh="all")
    Sender: request = increment(request)


@fragment
def nested_work():
    with spawn(fresh="all"):
        work()


@workflow
def nested_early_retained():
    Initiator: marker = wait_for_early_send()
    spawn(nested_work, at=Initiator, retain=(Sender, Initiator), fresh="all")
    Sender: request = increment(request)


@workflow
def repeated_early_retained():
    while (request < 10) @ Sender:
        Initiator: marker = wait_for_early_send()
        spawn(work, at=Initiator, retain=(Sender, Initiator), fresh="all")
        Sender: request = increment(request)


@fragment
def two_workers():
    skip(Initiator)
    Worker: reply = increment(1)
    Auditor: reply = increment(2)


@workflow
def allocate_two():
    spawn(two_workers, at=Initiator, retain=(Initiator,))


@workflow
def retained_only():
    with spawn(at=Initiator, retain=(Sender, Initiator), fresh="all"):
        skip(Initiator)
        Sender: reply = increment(request)


@workflow
def retained_initiator_only():
    with spawn(at=Sender, retain=(Sender,), fresh="all"):
        Sender: reply = increment(request)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("wf", [early_retained, nested_early_retained, repeated_early_retained])
def test_retained_send_precedes_initiator_and_allocation(backend, wf, tmp_path, monkeypatch):
    release_initiator.clear()
    events = []
    path = str(tmp_path / "early.sqlite")
    allocations = []
    original = InProcessChannel.register_spawn_allocation

    def record(self, child, role, endpoint):
        allocations.append((child, role, endpoint))
        return original(self, child, role, endpoint)

    monkeypatch.setattr(InProcessChannel, "register_spawn_allocation", record)

    def observe(event):
        events.append(event)
        if event["type"] == "send" and event.get("session") != "0":
            if event.get("sender") == "Worker" or event.get("lifeline") == "Worker":
                return
            if release_initiator.is_set():
                return
            assert not any(e["type"] == "init_send" for e in events)
            assert not allocations
            if backend == "sqlite":
                # Trace is emitted inside the sending transaction. Read only
                # already committed state: the retained fork must have committed.
                with closing(open_store_readonly(path)) as conn:
                    assert conn.execute("SELECT COUNT(*) FROM spawn_instances").fetchone()[0] >= 1
                    assert conn.execute("SELECT COUNT(*) FROM spawn_allocations").fetchone()[0] == 0
                    assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 1
            release_initiator.set()

    try:
        if backend == "memory":
            run(wf, [Sender, Initiator], {}, trace=observe, timeout=8)
        else:
            run_sqlite(wf, store_path=path, trace=observe, timeout=8)
    finally:
        release_initiator.set()
    child_send = next(i for i, e in enumerate(events) if e["type"] == "send" and e.get("session") != "0")
    init = next(i for i, e in enumerate(events) if e["type"] == "init_send")
    assert child_send < init
    assert not any(e["type"] == "init_recv" for e in events)
    replies = sorted(e["outputs"]["reply"] for e in events
                     if e["type"] == "act" and e.get("lifeline") == "Worker")
    assert replies == ([8, 9, 10] if wf is repeated_early_retained else [8])
    forks = [e for e in events if e["type"] == "fork"]
    assert len(forks) == (2 if wf is early_retained else 5 if wf is nested_early_retained else 6)
    assert forks[0]["child_session"] == events[init]["child_session"]
    if wf is repeated_early_retained:
        assert {e["child_session"] for e in forks} == {
            e["child_session"] for e in events if e["type"] == "init_send"
        }


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("wf", [retained_only, retained_initiator_only])
def test_creation_without_new_roles(wf, backend, tmp_path):
    events = []
    if backend == "memory":
        roots = [Sender] if wf is retained_initiator_only else [Sender, Initiator]
        run(wf, roots, {}, trace=events.append, timeout=5)
    else:
        run_sqlite(wf, store_path=str(tmp_path / "empty.sqlite"), trace=events.append, timeout=5)
    assert not any(e["type"] in {"init_send", "init_recv"} for e in events)
    owner = Sender if wf is retained_initiator_only else Initiator
    assert isinstance(project(wf, owner), BoundForkStmt)
    assert not any(e["type"] == "spawn_visit" for e in events)


# A real process exit tests transaction rollback and fresh-process recovery.
CRASH_DRIVER = r'''
import importlib.util, os, sys
from zippergen import run_sqlite
from zippergen.sessions import ScopedChannel
import zippergen.store as store
spec = importlib.util.spec_from_file_location('local_creation_fixture', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
path, phase = sys.argv[2:4]
if phase.startswith('init_') or phase == 'resume':
    m.release_initiator.set()
else:
    # Hold the initiator outside a transaction until the retained fork kills us.
    pass
orig_activate = ScopedChannel._activate
orig_allocate = store.record_spawn_allocation

def activate(self, descriptor, role):
    result = orig_activate(self, descriptor, role)
    if phase == 'fork_before' and role == 'Sender':
        os._exit(71)
    return result

def allocate(conn, child, role, endpoint):
    result = orig_allocate(conn, child, role, endpoint)
    if phase == 'init_before':
        os._exit(71)
    return result

ScopedChannel._activate = activate
store.record_spawn_allocation = allocate

def trace(e):
    if phase == 'fork_after' and e['type'] == 'send' and e.get('session') != '0':
        # A child's send can only run after its local fork transaction committed.
        os._exit(71)

if phase == 'init_after':
    # For two fresh roles, the second send starts after the first committed.
    def allocate(conn, child, role, endpoint):
        if conn.execute('SELECT COUNT(*) FROM spawn_allocations').fetchone()[0] == 1:
            os._exit(71)
        return orig_allocate(conn, child, role, endpoint)
    store.record_spawn_allocation = allocate
wf = m.allocate_two if phase == 'init_after' or (phase == 'resume' and sys.argv[4] == 'init_after') else m.early_retained
run_sqlite(wf, store_path=path, trace=trace, timeout=8)
'''


@pytest.mark.parametrize("phase", ["fork_before", "fork_after", "init_before", "init_after"])
def test_crash_windows_preserve_independent_creation(tmp_path, phase):
    path = str(tmp_path / "restart.sqlite")
    cmd = [sys.executable, "-c", CRASH_DRIVER, __file__, path]
    killed = subprocess.run([*cmd, phase, phase], capture_output=True, text=True, timeout=15)
    assert killed.returncode == 71, killed.stderr
    with closing(open_store(path)) as conn:
        allocated = conn.execute("SELECT child_id,role,endpoint FROM spawn_allocations").fetchall()
        assert len(allocated) == (1 if phase == "init_after" else 0)
        if phase == "fork_after":
            assert conn.execute("SELECT COUNT(*) FROM spawn_instances").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 1
    resumed = subprocess.run([*cmd, "resume", phase], capture_output=True, text=True, timeout=15)
    assert resumed.returncode == 0, resumed.stderr
    with closing(open_store(path)) as conn:
        rows = conn.execute("SELECT child_id,role,endpoint FROM spawn_allocations").fetchall()
        assert len(rows) == (2 if phase == "init_after" else 1)
        assert all(endpoint == fresh_endpoint(child, role) for child, role, endpoint in rows)
        assert all(row in rows for row in allocated)
        assert conn.execute("SELECT COUNT(*) FROM spawn_instances").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM outstanding_messages").fetchone()[0] == 0


def test_initiated_semantics_version_changes_resume_fingerprint():
    from dataclasses import replace
    programs = {r.name: project(early_retained, r) for r in (Sender, Initiator)}
    templates = project_templates(early_retained)
    old_templates = {key: replace(t, template=t.template.replace("fresh-v2:", "fresh-v1:"))
                     for key, t in templates.items()}
    assert program_fingerprint(programs, templates=templates) != program_fingerprint(programs, templates=old_templates)


def test_schema_five_upgrade_preserves_state_without_inventing_allocations(tmp_path):
    from zippergen.store import SCHEMA_VERSION
    path = str(tmp_path / "schema5.sqlite")
    with closing(open_store(path)) as conn:
        conn.execute("INSERT INTO adapter_state(key,value,updated_at) VALUES('kept','{}',0)")
        conn.execute("DROP TABLE spawn_allocations")
        conn.execute("UPDATE store_meta SET value='5' WHERE key='schema_version'")
    with closing(open_store_readonly(path)) as conn:
        assert conn.execute("SELECT value FROM store_meta WHERE key='schema_version'").fetchone()[0] == '5'
    with closing(open_store(path)) as conn:
        assert conn.execute("SELECT value FROM store_meta WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        assert conn.execute("SELECT value FROM adapter_state WHERE key='kept'").fetchone()[0] == '{}'
        assert conn.execute("SELECT COUNT(*) FROM spawn_allocations").fetchone()[0] == 0
