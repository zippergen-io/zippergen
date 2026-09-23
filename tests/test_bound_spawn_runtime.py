"""Managed endpoints execute bound child roles in memory and SQLite."""

import pytest
import threading
import time
import subprocess
import sys

from zippergen import Json, Lifeline, Var, fragment, pure, run, run_sqlite, spawn, workflow
from zippergen.store import DurableChannel, load_spawn_instance, open_store


A, B = Lifeline("A"), Lifeline("B")
binding = Var("binding", Json)
item = Var("item", int)
answer = Var("answer", int)
counter = Var("counter", int, default=0)


@pure
def double(value: int) -> int:
    return value * 2


@pure
def one() -> int:
    return 1


@pure
def increment(value: int) -> int:
    return value + 1


@fragment
def child():
    A(item) >> B(item)
    B: answer = double(item)
    B(answer) >> A(answer)


@workflow
def flow(binding: Json @ A, item: int @ A):
    spawn(child, bindings=binding @ A, retain=(A,), fresh="all")


@fragment
def self_child():
    A: answer = one()


@workflow
def busy_parent(binding: Json @ A):
    spawn(self_child, bindings=binding @ A, retain=())
    while (counter < 100) @ A:
        A: counter = increment(counter)


@fragment
def inherited_grandchild():
    B: answer = one()
    with spawn(fresh="all"):
        B: answer = double(answer)


@workflow
def recruit_then_inherit(binding: Json @ A):
    spawn(inherited_grandchild, bindings=binding @ A, retain=())


@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_recruited_child_runs_at_configured_endpoint(runner, tmp_path):
    events = []
    kwargs = {"endpoints": {"worker"}, "initial_envs": {
        "A": {"binding": {"A": "A", "B": "worker"}, "item": 7},
    }, "trace": events.append}
    if runner is run_sqlite:
        kwargs["store_path"] = str(tmp_path / "state.sqlite")
    else:
        kwargs["lifelines"] = [A]
    runner(flow, **kwargs)
    assert any(event["type"] == "recv" and event["to"] == "A"
               and event.get("session") != "0" for event in events)


@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_bound_root_input_type_is_checked(runner, tmp_path):
    kwargs = {"endpoints": {"worker"}, "initial_envs": {
        "A": {"binding": {"A": "A", "B": "worker"}, "item": "bad"},
    }}
    if runner is run:
        kwargs["lifelines"] = [A]
    else:
        kwargs["store_path"] = str(tmp_path / "invalid.sqlite")
    with pytest.raises((TypeError, ValueError, RuntimeError), match="item|input"):
        runner(flow, **kwargs)


@pytest.mark.parametrize("phase", ["bind", "desc", "invite", "fork"])
def test_durable_creation_resumes_from_committed_position(tmp_path, phase):
    from examples.dynamic_spawn.workflow import access_reviews

    path = str(tmp_path / "state.sqlite")
    interrupted = False

    def interrupt(event):
        nonlocal interrupted
        if interrupted:
            return
        if phase == "bind" and event["type"] == "bind":
            interrupted = True
        elif phase in {"desc", "invite"} and event["type"] == "init_send" \
                and event["kind"] == phase:
            interrupted = True
        elif phase == "fork" and event["type"] == "fork" \
                and event["lifeline"] == "IT":
            interrupted = True
        if interrupted:
            raise RuntimeError("simulated process loss before commit")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        run_sqlite(access_reviews, store_path=path,
                   endpoints={"reviewer-east", "reviewer-west"}, trace=interrupt,
                   timeout=5)
    assert interrupted
    events = []
    assert run_sqlite(access_reviews, store_path=path,
                      endpoints={"reviewer-east", "reviewer-west"},
                      trace=events.append, timeout=5) == 2
    conn = open_store(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM spawn_instances").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 6
    finally:
        conn.close()
    assert sum(event["type"] == "fork" and event["lifeline"] == "IT"
               for event in events) <= 2


def test_in_memory_supervisor_waits_while_invitation_is_being_accepted(monkeypatch):
    from zippergen.channels import InProcessChannel

    picked = threading.Event()
    release = threading.Event()
    original = InProcessChannel.try_get_invitation

    def hold_after_dequeue(self, endpoint):
        item = original(self, endpoint)
        if item is not None:
            picked.set()
            assert release.wait(2)
        return item

    monkeypatch.setattr(InProcessChannel, "try_get_invitation", hold_after_dequeue)
    outcome = []

    def execute():
        try:
            run(flow, [A], {"A": {"binding": {"A": "A", "B": "worker"},
                                  "item": 7}}, endpoints={"worker"}, timeout=4)
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


@pytest.mark.parametrize("runner", [run, run_sqlite])
def test_invitation_is_serviced_before_busy_parent_finishes(runner, tmp_path):
    events = []
    kwargs = {"initial_envs": {"A": {"binding": {"A": "A"}}},
              "trace": events.append, "timeout": 5}
    if runner is run:
        kwargs["lifelines"] = [A]
    else:
        kwargs["store_path"] = str(tmp_path / "busy.sqlite")
    runner(busy_parent, **kwargs)
    child = next(i for i, event in enumerate(events)
                 if event["type"] == "act" and event.get("action") == "one")
    final_parent = max(i for i, event in enumerate(events)
                       if event["type"] == "act" and event.get("action") == "increment")
    assert child < final_parent


def test_durable_bound_spawn_resumes_in_fresh_process(tmp_path):
    path = str(tmp_path / "state.sqlite")
    code = """
import sys
from examples.dynamic_spawn.workflow import access_reviews
from zippergen import run_sqlite

def interrupt(event):
    if event['type'] == 'init_send' and event['kind'] == 'invite':
        raise RuntimeError('injected crash')

result = run_sqlite(access_reviews, store_path=sys.argv[1],
                    endpoints={'reviewer-east', 'reviewer-west'},
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
        assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 6
    finally:
        conn.close()


def test_active_inherited_grandchild_decodes_after_restart(tmp_path):
    from zippergen.store import load_role_state

    path = str(tmp_path / "grandchild.sqlite")

    def interrupt(event):
        if event["type"] == "act_start" and event.get("action") == "double":
            raise RuntimeError("stop with active grandchild")

    with pytest.raises(RuntimeError, match="active grandchild"):
        run_sqlite(recruit_then_inherit, store_path=path,
                   initial_envs={"A": {"binding": {"B": "worker"}}},
                   endpoints={"worker"}, trace=interrupt, timeout=5)
    conn = open_store(path)
    try:
        state = load_role_state(conn, "worker")
        assert state is not None
        controls = state["control"]["sessions"]
        assert any(entry["name"].count("/") == 2 and entry["template"]
                   for entry in controls)
    finally:
        conn.close()

    events = []
    run_sqlite(recruit_then_inherit, store_path=path,
               initial_envs={"A": {"binding": {"B": "worker"}}},
               endpoints={"worker"}, trace=events.append, timeout=5)
    assert any(event["type"] == "act" and event.get("action") == "double"
               and event.get("outputs", {}).get("answer") == 2
               for event in events)


def test_stale_identical_invitation_is_acknowledged_without_reexecution(tmp_path):
    path = str(tmp_path / "duplicate.sqlite")
    supplied = {"A": {"binding": {"A": "A", "B": "worker"}, "item": 7}}
    run_sqlite(flow, store_path=path, initial_envs=supplied,
               endpoints={"worker"}, timeout=5)
    conn = open_store(path)
    try:
        descriptor = load_spawn_instance(conn, "0/0:1")
        assert descriptor is not None
        before = conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0]
        DurableChannel(conn, "A").put(
            "A", "worker", f"init:invite:{descriptor['child']}:B", (descriptor,),
        )
    finally:
        conn.close()

    events = []
    run_sqlite(flow, store_path=path, initial_envs=supplied,
               endpoints={"worker"}, trace=events.append, timeout=5)
    assert not any(event["type"] in {"act", "recv", "fork"} for event in events)
    conn = open_store(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == before
        assert conn.execute("SELECT COUNT(*) FROM outstanding_messages").fetchone()[0] == 0
    finally:
        conn.close()


def test_conflicting_stale_invitation_is_rejected(tmp_path):
    path = str(tmp_path / "conflict.sqlite")
    supplied = {"A": {"binding": {"A": "A", "B": "worker"}, "item": 7}}
    run_sqlite(flow, store_path=path, initial_envs=supplied,
               endpoints={"worker"}, timeout=5)
    conn = open_store(path)
    try:
        descriptor = load_spawn_instance(conn, "0/0:1")
        assert descriptor is not None
        conflicting = {**descriptor, "template": "different-template"}
        DurableChannel(conn, "A").put(
            "A", "worker", f"init:invite:{descriptor['child']}:B", (conflicting,),
        )
    finally:
        conn.close()
    with pytest.raises(RuntimeError, match="[Ii]ncompatible .*bound-spawn template"):
        run_sqlite(flow, store_path=path, initial_envs=supplied,
                   endpoints={"worker"}, timeout=5)


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
    execute = lambda: run(
        flow, [A], {"A": {"binding": {"A": "A", "B": "worker"}, "item": 7}},
        endpoints={"worker"}, trace=events.append, timeout=5,
    )
    if conflict:
        with pytest.raises(RuntimeError, match="[Ii]ncompatible .*bound-spawn template"):
            execute()
    else:
        execute()
        assert sum(event["type"] == "act" and event.get("action") == "double"
                   for event in events) == 1
