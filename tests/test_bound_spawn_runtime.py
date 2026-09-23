"""Fresh endpoint allocation, activation, and durable recovery."""

import subprocess
import sys
import threading
import time

import pytest

from zippergen import Lifeline, Var, fragment, pure, run, run_sqlite, spawn, workflow
from zippergen.sessions import SessionState, fresh_endpoint
from zippergen.store import DurableChannel, load_spawn_instance, open_store


A, B = Lifeline("A"), Lifeline("B")
item = Var("item", int)
answer = Var("answer", int)


@pytest.mark.parametrize("child_fields", [{}, {"child": None}, {"child": 1}])
def test_descriptor_rejects_missing_or_nonstring_child(child_fields):
    state = SessionState(sessions={})
    with pytest.raises(ValueError, match="invalid child session"):
        state.validate_descriptor({"template": "review", **child_fields}, "review", "B")


@pure
def double(value: int) -> int:
    return value * 2


@pure
def one() -> int:
    return 1


@fragment
def child():
    A(item) >> B(item)
    B: answer = double(item)
    B(answer) >> A(answer)


@workflow
def flow(item: int @ A):
    spawn(child, at=A, retain=(A,), fresh="all")


@fragment
def inherited_grandchild():
    B: answer = one()
    with spawn(fresh="all"):
        B: answer = double(answer)


@workflow
def recruit_then_inherit():
    spawn(inherited_grandchild, at=A, retain=())


def _flow_args(runner, tmp_path, *, trace=None):
    kwargs = {"initial_envs": {"A": {"item": 7}}, "timeout": 5}
    if trace is not None:
        kwargs["trace"] = trace
    if runner is run:
        kwargs["lifelines"] = [A]
    else:
        kwargs["store_path"] = str(tmp_path / "state.sqlite")
    return kwargs


@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_fresh_child_runs_on_a_new_endpoint(runner, tmp_path):
    events = []
    runner(flow, **_flow_args(runner, tmp_path, trace=events.append))
    creation = next(event for event in events if event["type"] == "bind")
    child_id = creation["child_session"]
    assert creation["bindings"] == {"A": "A", "B": fresh_endpoint(child_id, "B")}
    assert any(event["type"] == "recv" and event["to"] == "A"
               and event.get("session") == child_id for event in events)
    assert any(event["type"] == "act" and event.get("lifeline") == "B"
               and event.get("endpoint") == fresh_endpoint(child_id, "B")
               for event in events)


@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_fresh_root_input_type_is_checked(runner, tmp_path):
    kwargs = _flow_args(runner, tmp_path)
    kwargs["initial_envs"] = {"A": {"item": "bad"}}
    with pytest.raises((TypeError, ValueError, RuntimeError), match="item|input"):
        runner(flow, **kwargs)


@pytest.mark.parametrize("phase", ["bind", "invite", "fork"])
def test_durable_allocation_resumes_from_committed_position(tmp_path, phase):
    from examples.dynamic_spawn.workflow import onboard_employees

    path = str(tmp_path / "state.sqlite")
    interrupted = False

    def interrupt(event):
        nonlocal interrupted
        if interrupted:
            return
        interrupted = (
            phase == "bind" and event["type"] == "bind"
            or phase == "invite" and event["type"] == "init_send"
            and event["kind"] == "invite"
            or phase == "fork" and event["type"] == "fork"
            and event["lifeline"] == "HR"
        )
        if interrupted:
            raise RuntimeError("simulated process loss before commit")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        run_sqlite(onboard_employees, store_path=path, trace=interrupt, timeout=5)
    assert interrupted
    assert run_sqlite(onboard_employees, store_path=path, timeout=5) == 2
    conn = open_store(path)
    try:
        rows = conn.execute("SELECT descriptor FROM spawn_instances").fetchall()
        assert len(rows) == 2
        assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 4
    finally:
        conn.close()


def test_in_memory_completion_waits_during_invitation_acceptance(monkeypatch):
    from zippergen.channels import InProcessChannel

    picked = threading.Event()
    release = threading.Event()
    original = InProcessChannel.try_get_invitation

    def hold_after_dequeue(self, endpoint):
        invitation = original(self, endpoint)
        if invitation is not None:
            picked.set()
            assert release.wait(2)
        return invitation

    monkeypatch.setattr(InProcessChannel, "try_get_invitation", hold_after_dequeue)
    outcome = []

    def execute():
        try:
            run(flow, [A], {"A": {"item": 7}}, timeout=4)
            outcome.append("done")
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert picked.wait(2)
        time.sleep(0.05)
        assert thread.is_alive() and not outcome
    finally:
        release.set()
        thread.join(timeout=3)
    assert outcome == ["done"]


def test_durable_allocation_resumes_in_fresh_process(tmp_path):
    path = str(tmp_path / "state.sqlite")
    code = """
import sys
from examples.dynamic_spawn.workflow import onboard_employees
from zippergen import run_sqlite

def interrupt(event):
    if event['type'] == 'init_send' and event['kind'] == 'invite':
        raise RuntimeError('injected crash')

result = run_sqlite(onboard_employees, store_path=sys.argv[1],
                    trace=interrupt if sys.argv[2] == 'crash' else None,
                    timeout=5)
assert result == 2
"""
    first = subprocess.run([sys.executable, "-c", code, path, "crash"],
                           capture_output=True, text=True)
    assert first.returncode != 0 and "injected crash" in first.stderr
    second = subprocess.run([sys.executable, "-c", code, path, "resume"],
                            capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    conn = open_store(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM spawn_instances").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 4
    finally:
        conn.close()


def test_active_inherited_grandchild_decodes_after_restart(tmp_path):
    from zippergen.store import load_role_state

    path = str(tmp_path / "grandchild.sqlite")

    def interrupt(event):
        if event["type"] == "act_start" and event.get("action") == "double":
            raise RuntimeError("stop with active grandchild")

    with pytest.raises(RuntimeError, match="active grandchild"):
        run_sqlite(recruit_then_inherit, store_path=path, trace=interrupt, timeout=5)
    conn = open_store(path)
    try:
        descriptor = load_spawn_instance(conn, "0/0:1")
        assert descriptor is not None
        endpoint = descriptor["bindings"]["B"]
        state = load_role_state(conn, endpoint)
        assert state is not None
        assert any(entry["name"].count("/") == 2 and entry["template"]
                   for entry in state["control"]["sessions"])
    finally:
        conn.close()
    events = []
    run_sqlite(recruit_then_inherit, store_path=path,
               trace=events.append, timeout=5)
    assert any(event["type"] == "act" and event.get("action") == "double"
               and event.get("outputs", {}).get("answer") == 2
               for event in events)


def _inject_invitation(path, descriptor):
    endpoint = descriptor["bindings"]["B"]
    conn = open_store(path)
    try:
        DurableChannel(conn, "A").put(
            "A", endpoint, f"init:invite:{descriptor['child']}:B", (descriptor,),
        )
    finally:
        conn.close()


def test_stale_identical_invitation_is_acked_without_reexecution(tmp_path):
    path = str(tmp_path / "duplicate.sqlite")
    run_sqlite(flow, store_path=path, initial_envs={"A": {"item": 7}}, timeout=5)
    conn = open_store(path)
    try:
        descriptor = load_spawn_instance(conn, "0/0:1")
        assert descriptor is not None
        before = conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0]
    finally:
        conn.close()
    _inject_invitation(path, descriptor)
    events = []
    run_sqlite(flow, store_path=path, initial_envs={"A": {"item": 7}},
               trace=events.append, timeout=5)
    assert not any(event["type"] in {"act", "recv", "fork"} for event in events)
    conn = open_store(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == before
        assert conn.execute("SELECT COUNT(*) FROM outstanding_messages").fetchone()[0] == 0
    finally:
        conn.close()


def test_conflicting_stale_invitation_is_rejected(tmp_path):
    path = str(tmp_path / "conflict.sqlite")
    run_sqlite(flow, store_path=path, initial_envs={"A": {"item": 7}}, timeout=5)
    conn = open_store(path)
    try:
        descriptor = load_spawn_instance(conn, "0/0:1")
        assert descriptor is not None
    finally:
        conn.close()
    _inject_invitation(path, {**descriptor, "template": "different-template"})
    with pytest.raises(RuntimeError, match="[Ii]ncompatible .*spawn template"):
        run_sqlite(flow, store_path=path, initial_envs={"A": {"item": 7}}, timeout=5)


@pytest.mark.parametrize("conflict", [False, True])
def test_in_memory_duplicate_invitation_is_idempotent_or_rejected(monkeypatch, conflict):
    from zippergen.channels import InProcessChannel

    original = InProcessChannel.put

    def duplicate(self, sender, receiver, channel, values, *args, **kwargs):
        sequence = original(self, sender, receiver, channel, values, *args, **kwargs)
        if channel.startswith("init:invite:"):
            repeated = values
            if conflict:
                repeated = ({**values[0], "template": "different-template"},)
            original(self, sender, receiver, channel, repeated, *args, **kwargs)
        return sequence

    monkeypatch.setattr(InProcessChannel, "put", duplicate)
    events = []
    execute = lambda: run(flow, [A], {"A": {"item": 7}},
                          trace=events.append, timeout=5)
    if conflict:
        with pytest.raises(RuntimeError, match="[Ii]ncompatible .*spawn template"):
            execute()
    else:
        execute()
        assert sum(event["type"] == "act" and event.get("action") == "double"
                   for event in events) == 1
