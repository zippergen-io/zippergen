"""A finished endpoint releases its worker without joining the whole workflow.

These test protocols use external barriers only to hold the parent/peer alive
while observing worker lifetimes. No application progress relies on the barriers
surviving restart.
"""
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from zippergen import Lifeline, Var, effect, fragment, pure, run, run_sqlite, spawn, workflow, skip
from zippergen.sessions import SessionState
from zippergen.store import open_store, open_store_readonly
from zippergen.syntax import EmptyStmt

Intake, Worker, Peer = map(Lifeline, ["Intake", "RetiringWorker", "RetiringPeer"])
counter = Var("counter", int, default=0)
marker = Var("marker", int)
shared = Var("shared", int)
reply = Var("reply", int)
parked = threading.Event()
release = threading.Event()


@pure
def one() -> int:
    return 1


@pure
def increment(value: int) -> int:
    return value + 1


@effect
def park() -> int:
    parked.set()
    if not release.wait(10):
        raise RuntimeError("test barrier timed out")
    return 1


@workflow
def intake(limit: int @ Intake) -> int:
    while (counter < limit) @ Intake:
        with spawn(at=Intake, retain=(Intake,)):
            skip(Intake)
            Worker: marker = one()
        Intake: counter = increment(counter)
    Intake: marker = park()
    return counter @ Intake


@fragment
def nested_body():
    skip(Intake)
    Worker: shared = one()
    with spawn():
        Worker(shared) >> Peer(shared)
        Peer: reply = park()
        Peer(reply) >> Worker(reply)
        Worker: reply = increment(shared)
    Worker: marker = one()


@workflow
def nested():
    spawn(nested_body, at=Intake, retain=(Intake,))


@pytest.fixture(autouse=True)
def barriers():
    parked.clear()
    release.clear()
    yield
    release.set()


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("worker did not reach the required lifecycle state")
        time.sleep(0.01)


def start_run(wf, backend, path, events, initial=None):
    outcomes = []
    def execute():
        try:
            if backend == "sqlite":
                outcomes.append(run_sqlite(wf, store_path=str(path), initial_envs=initial or {},
                                           trace=events, timeout=12))
            else:
                outcomes.append(run(wf, [Intake], initial or {}, trace=events, timeout=12))
        except BaseException as exc:
            outcomes.append(exc)
    supervisor = threading.Thread(target=execute, name="test-supervisor")
    supervisor.start()
    return supervisor, outcomes


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_finished_workers_retire_while_intake_remains_alive(backend, tmp_path, monkeypatch):
    import zippergen.sqlite_runner as runner
    worker_threads = []
    connections = []
    original = runner.open_store
    def record_connection(*args, **kwargs):
        conn = original(*args, **kwargs)
        if threading.current_thread().name == Worker.name:
            connections.append(conn)
        return conn
    monkeypatch.setattr(runner, "open_store", record_connection)
    def observe(event):
        if event["type"] == "act" and event.get("lifeline") == Worker.name:
            worker_threads.append(threading.current_thread())
    path = tmp_path / "retirement.sqlite"
    supervisor, outcomes = start_run(intake, backend, path, observe, {"Intake": {"limit": 40}})
    try:
        assert parked.wait(5)
        wait_for(lambda: len(worker_threads) == 40)
        for thread in worker_threads:
            thread.join(3)
            assert not thread.is_alive()
        assert supervisor.is_alive()  # No global completion/join was needed.
        assert not outcomes
        if backend == "sqlite":
            assert len(connections) == 40
            for conn in connections:
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    conn.execute("SELECT 1")
            with closing(open_store_readonly(path)) as conn:
                assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 80
                assert conn.execute("SELECT COUNT(*) FROM role_state WHERE status='done'").fetchone()[0] == 40
    finally:
        release.set()
        supervisor.join(5)
    assert not supervisor.is_alive()
    assert outcomes == [40]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_retained_descendant_keeps_endpoint_and_inherited_cells_alive(backend, tmp_path):
    outer_done = threading.Event()
    worker_threads = []
    events = []
    def observe(event):
        events.append(event)
        if event["type"] == "act" and event.get("lifeline") == Worker.name:
            worker_threads.append(threading.current_thread())
            if "marker" in event.get("outputs", {}):
                outer_done.set()
    supervisor, outcomes = start_run(nested, backend, tmp_path / "nested.sqlite", observe)
    try:
        assert parked.wait(5)
        assert outer_done.wait(5)
        assert worker_threads[-1].is_alive()
        assert supervisor.is_alive()
    finally:
        release.set()
        supervisor.join(5)
    assert not supervisor.is_alive()
    assert len(outcomes) == 1 and not isinstance(outcomes[0], BaseException), outcomes
    assert any(e.get("outputs", {}).get("reply") == 2 and e.get("lifeline") == Worker.name for e in events)
    assert all(not thread.is_alive() for thread in worker_threads)


def test_uninitialized_endpoint_is_not_retirable():
    state = SessionState.start(EmptyStmt(), endpoint="uninitialized", templates={}, keep_alive=True)
    assert state.done
    assert not state.can_retire


CRASH_DRIVER = r'''
import importlib.util, os, sys, threading
from zippergen import run_sqlite
spec = importlib.util.spec_from_file_location('retirement_fixture', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
workers = []
def trace(event):
    if sys.argv[3] == 'before_final' and event['type'] == 'act' and event.get('lifeline') == m.Worker.name:
        os._exit(72)
    if event['type'] == 'act' and event.get('lifeline') == m.Worker.name:
        workers.append(threading.current_thread())
def kill_after_retirement():
    assert m.parked.wait(8)
    # The parent is parked outside its transaction; all child workers must exit.
    import time
    deadline = time.monotonic() + 5
    while len(workers) < 4 and time.monotonic() < deadline:
        time.sleep(.01)
    assert len(workers) == 4
    for worker in workers:
        worker.join(3)
        assert not worker.is_alive()
    os._exit(71)
threading.Thread(target=kill_after_retirement, daemon=True).start()
run_sqlite(m.intake, store_path=sys.argv[2], initial_envs={'Intake': {'limit': 4}}, trace=trace, timeout=15)
'''


@pytest.mark.parametrize("phase", ["before_final", "after_retirement"])
def test_fresh_process_resume_skips_retired_workers(tmp_path, monkeypatch, phase):
    import zippergen.sqlite_runner as runner
    path = tmp_path / "resume.sqlite"
    killed = subprocess.run([sys.executable, "-c", CRASH_DRIVER, str(Path(__file__)), str(path), phase],
                            capture_output=True, text=True, timeout=20)
    assert killed.returncode == (72 if phase == "before_final" else 71), killed.stderr
    with closing(open_store(str(path))) as conn:
        retired = {row[0] for row in conn.execute("SELECT role FROM role_state WHERE status='done'")}
        assert len(retired) == (0 if phase == "before_final" else 4)
    started = []
    original = runner.RoleRunner
    def record(*args, **kwargs):
        started.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "RoleRunner", record)
    release.set()
    events = []
    assert run_sqlite(intake, store_path=str(path), initial_envs={"Intake": {"limit": 4}},
                      trace=events.append, timeout=5) == 4
    assert not retired.intersection(started)
    if phase == "after_retirement":
        assert started == ["Intake"]
        assert not any(e.get("lifeline") == Worker.name for e in events)
    else:
        assert any(e.get("lifeline") == Worker.name for e in events)
    with closing(open_store(str(path))) as conn:
        assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 8


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("conflict", [False, True])
def test_late_initialization_wakes_only_to_ack_or_reject(backend, conflict, tmp_path, monkeypatch):
    import json
    from zippergen.channels import InProcessChannel
    from zippergen.store import DurableChannel

    sent = []
    workers = []
    events = []
    acknowledged = threading.Event()
    original_put = InProcessChannel.put
    original_accept = SessionState.accept_invitation
    def record_put(self, sender, receiver, channel, values, *args, **kwargs):
        if channel.startswith("init:invite:"):
            sent.append((self, sender, receiver, channel, values))
        return original_put(self, sender, receiver, channel, values, *args, **kwargs)
    def accept(self, channel):
        retired = self.can_retire
        try:
            result = original_accept(self, channel)
        except BaseException:
            if retired:
                acknowledged.set()
            raise
        if retired and result:
            acknowledged.set()
        return result
    monkeypatch.setattr(InProcessChannel, "put", record_put)
    monkeypatch.setattr(SessionState, "accept_invitation", accept)
    def observe(event):
        events.append(event)
        if event["type"] == "act" and event.get("lifeline") == Worker.name:
            workers.append(threading.current_thread())
    path = tmp_path / "duplicate.sqlite"
    supervisor, outcomes = start_run(intake, backend, path, observe, {"Intake": {"limit": 1}})
    try:
        assert parked.wait(5)
        wait_for(lambda: bool(workers))
        workers[0].join(3)
        assert not workers[0].is_alive()
        if backend == "memory":
            channel, sender, endpoint, route, payload = sent[0]
            descriptor = {**payload[0]}
        else:
            with closing(open_store(str(path))) as conn:
                descriptor = json.loads(conn.execute("SELECT descriptor FROM spawn_instances").fetchone()[0])
            sender, endpoint = "Intake", descriptor["bindings"][Worker.name]
            route = f"init:invite:{descriptor['child']}:{Worker.name}"
        if conflict:
            descriptor["template"] = "incompatible-template"
        if backend == "memory":
            original_put(channel, sender, endpoint, route, (descriptor,))
        else:
            with closing(open_store(str(path))) as conn:
                DurableChannel(conn, sender).put(sender, endpoint, route, (descriptor,))
        assert acknowledged.wait(5)
    finally:
        release.set()
        supervisor.join(5)
    assert not supervisor.is_alive()
    assert len(outcomes) == 1
    if conflict:
        assert isinstance(outcomes[0], RuntimeError)
        assert "template" in str(outcomes[0])
    else:
        assert outcomes == [1]
    assert len(workers) == 1  # Duplicate initialization never reruns the action.
