"""Owner-allocated child roles run on fresh managed endpoints."""

import subprocess
import sys
import threading
import time
from contextlib import closing

import pytest

from zippergen import Lifeline, Var, branch, fragment, human, llm, make_lifeline_router, parallel, pure, run, run_sqlite, spawn, workflow, skip
from zippergen.syntax import _ordered_workflow_lifelines

Manager, Reviewer, IT = (Lifeline(name) for name in ("Manager", "Reviewer", "IT"))
payload = Var("payload", str, default="first")
observed = Var("observed", str)
value = Var("value", int)
result = Var("result", int)
approved = Var("approved", bool)
marker = Var("marker", str)


@pure
def second() -> str:
    return "second"


@pure
def echo(payload: str) -> str:
    return payload


@pure
def five() -> int:
    return 5


@pure
def seven() -> int:
    return 7


@pure
def twice(value: int) -> int:
    return value * 2


@llm(system="Echo the request.", user="{payload}", parse="text",
     outputs=(("observed", str),))
def model_reply(payload: str) -> str: ...


@human(kind="confirm", outputs=["approved: bool"], context="Request {payload}")
def approve_request(payload: str) -> bool: ...


@fragment
def review_child():
    Manager(payload) >> Reviewer(payload)
    Reviewer: observed = echo(payload)


@workflow
def two_reviews():
    spawn(review_child, at=Manager, retain=(Manager,), fresh="all")
    Manager: payload = second()
    spawn(review_child, at=Manager, retain=(Manager,), fresh="all")


@workflow
def parallel_reviews():
    with parallel:
        with branch:
            spawn(review_child, at=Manager, retain=(Manager,), fresh="all")
        with branch:
            spawn(review_child, at=Manager, retain=(Manager,), fresh="all")


@fragment
def two_fresh_roles_child():
    Manager(payload) >> Reviewer(payload)
    Reviewer(payload) >> IT(payload)
    IT: observed = echo(payload)


@workflow
def two_fresh_roles():
    spawn(two_fresh_roles_child, at=Manager, retain=(Manager,), fresh="all")


@fragment
def inherited_grandchild():
    skip(Manager)
    Reviewer: value = five()
    with spawn(fresh="all"):
        Reviewer: result = twice(value)


@workflow
def allocate_then_inherit():
    spawn(inherited_grandchild, at=Manager, retain=(Manager,))


@fragment
def nested_allocated_child():
    skip(Manager)
    with spawn(at=Reviewer, retain=(Reviewer,)):
        skip(Reviewer)
        IT: value = five()


@workflow
def allocate_then_allocate():
    spawn(nested_allocated_child, at=Manager, retain=(Manager,))


@fragment
def self_child():
    Manager: value = five()


@workflow
def owner_retained_in_child():
    spawn(self_child, at=Manager, retain=(Manager,))


@fragment
def fresh_cell_child():
    skip(Manager)
    Reviewer: value = five()


@workflow
def fresh_cell_isolated():
    IT: value = seven()
    spawn(fresh_cell_child, at=Manager, retain=(Manager,))
    return value @ IT


@workflow
def parent_finishes_before_acceptance():
    spawn(review_child, at=Manager, retain=(Manager,), fresh="all")
    Manager: marker = second()


@fragment
def model_child():
    Manager(payload) >> Reviewer(payload)
    Reviewer: observed = model_reply(payload)


@workflow
def two_model_reviews():
    spawn(model_child, at=Manager, retain=(Manager,), fresh="all")
    Manager: payload = second()
    spawn(model_child, at=Manager, retain=(Manager,), fresh="all")


@fragment
def human_child():
    Manager(payload) >> Reviewer(payload)
    Reviewer: approved = approve_request(payload)
    Reviewer(approved) >> Manager(approved)


@workflow
def two_human_reviews():
    spawn(human_child, at=Manager, retain=(Manager,), fresh="all")
    Manager: payload = second()
    spawn(human_child, at=Manager, retain=(Manager,), fresh="all")


def _execute(wf, backend, tmp_path, *, trace):
    if backend == "sqlite":
        return run_sqlite(wf, trace=trace.append, timeout=5,
                          store_path=str(tmp_path / "state.sqlite"))
    return run(wf, list(_ordered_workflow_lifelines(wf)), {},
               trace=trace.append, timeout=5)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_repeated_allocations_get_distinct_endpoints_and_cells(backend, tmp_path):
    events = []
    _execute(two_reviews, backend, tmp_path, trace=events)
    bindings = [event["bindings"] for event in events if event["type"] == "init_send"]
    assert len(bindings) == 2
    assert bindings[0]["Manager"] == bindings[1]["Manager"] == "Manager"
    assert bindings[0]["Reviewer"] != bindings[1]["Reviewer"]
    assert all(item["Reviewer"].startswith("~spawn:") for item in bindings)
    outputs = [event["outputs"]["observed"] for event in events
               if event["type"] == "act" and event.get("lifeline") == "Reviewer"
               and "observed" in event.get("outputs", {})]
    assert sorted(outputs) == ["first", "second"]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_parallel_sites_allocate_disjoint_endpoints(backend, tmp_path):
    events = []
    _execute(parallel_reviews, backend, tmp_path, trace=events)
    allocations = [event for event in events if event["type"] == "init_send"]
    assert len(allocations) == 2
    assert len({event["site"] for event in allocations}) == 2
    assert len({event["bindings"]["Reviewer"] for event in allocations}) == 2
    assert all(event["bindings"]["Reviewer"].startswith("~spawn:")
               for event in allocations)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_multiple_fresh_roles_use_distinct_workers_and_route_messages(
    backend, tmp_path,
):
    events = []
    _execute(two_fresh_roles, backend, tmp_path, trace=events)
    descriptor = next(event["bindings"] for event in events if event["type"] == "init_send")
    assert descriptor["Manager"] == "Manager"
    assert len(set(descriptor.values())) == 3
    assert descriptor["Reviewer"].startswith("~spawn:")
    assert descriptor["IT"].startswith("~spawn:")
    assert any(event["type"] == "act" and event.get("lifeline") == "IT"
               and event.get("outputs", {}).get("observed") == "first" for event in events)
    if backend == "sqlite":
        from zippergen.store import open_store
        with closing(open_store(str(tmp_path / "state.sqlite"))) as conn:
            assert conn.execute("SELECT COUNT(*) FROM spawn_activations").fetchone()[0] == 3


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_fresh_role_can_fork_inherited_grandchild(backend, tmp_path):
    events = []
    _execute(allocate_then_inherit, backend, tmp_path, trace=events)
    assert any(event["type"] == "act" and event.get("outputs", {}).get("result") == 10
               for event in events)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_fresh_role_can_allocate_nested_role(backend, tmp_path):
    events = []
    _execute(allocate_then_allocate, backend, tmp_path, trace=events)
    assert any(event["type"] == "act" and event.get("lifeline") == "IT"
               and event.get("outputs", {}).get("value") == 5 for event in events)
    assert len([event for event in events if event["type"] == "init_send"]) == 2
    allocated = [event["bindings"] for event in events if event["type"] == "init_send"]
    addresses = {address for descriptor in allocated for address in descriptor.values()
                 if address.startswith("~spawn:")}
    assert len(addresses) == 2
    invitation = next(index for index, event in enumerate(events)
                      if event["type"] == "init_send" and event.get("to") == "IT")
    action = next(index for index, event in enumerate(events)
                  if event["type"] == "act" and event.get("lifeline") == "IT")
    assert invitation < action


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_all_retained_spawn_keeps_owner_endpoint(backend, tmp_path):
    events = []
    _execute(owner_retained_in_child, backend, tmp_path, trace=events)
    assert not any(event["type"] == "init_send" for event in events)
    assert any(event["type"] == "fork" and event["endpoint"] == "Manager" for event in events)
    assert any(event["type"] == "act" and event.get("outputs", {}).get("value") == 5
               and event["session"] != "0" for event in events)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_fresh_role_has_no_parent_protocol_cell_alias(backend, tmp_path):
    events = []
    assert _execute(fresh_cell_isolated, backend, tmp_path, trace=events) == 7
    assert any(event["type"] == "act" and event.get("outputs", {}).get("value") == 5
               and event["session"] != "0" for event in events)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_retained_fresh_cell_copies_at_local_fork(backend, tmp_path, monkeypatch):
    from zippergen.sessions import ScopedChannel

    original = ScopedChannel.send_init
    changed = False

    def update_between_allocation_and_fork(self, stmt):
        nonlocal changed
        descriptor = original(self, stmt)
        if not changed and stmt.recipient == Reviewer:
            changed = True
            self.root_env["payload"] = "late"
        return descriptor

    monkeypatch.setattr(ScopedChannel, "send_init", update_between_allocation_and_fork)
    events = []
    _execute(two_reviews, backend, tmp_path, trace=events)
    assert changed
    reviewer_outputs = [event["outputs"]["observed"] for event in events
                        if event["type"] == "act" and event.get("lifeline") == "Reviewer"
                        and "observed" in event.get("outputs", {})]
    assert "late" in reviewer_outputs


def test_parent_can_finish_before_fresh_role_accepts(monkeypatch):
    from zippergen.sessions import SessionState

    parent_finished = False
    events = []
    original = SessionState.accept_invitation

    def delay_child(self, channel):
        if self.endpoint and self.endpoint.startswith("~spawn:") and not parent_finished:
            return False
        return original(self, channel)

    def record(event):
        nonlocal parent_finished
        events.append(event)
        if event["type"] == "act" and event.get("outputs", {}).get("marker") == "second":
            parent_finished = True

    monkeypatch.setattr(SessionState, "accept_invitation", delay_child)
    run(parent_finishes_before_acceptance, [Manager], {}, trace=record, timeout=5)
    completed = next(index for index, event in enumerate(events)
                     if event["type"] == "act" and "marker" in event.get("outputs", {}))
    received = next(index for index, event in enumerate(events)
                    if event["type"] == "recv" and event.get("to") == "Reviewer")
    assert completed < received


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_allocated_model_role_uses_managed_fake_backend(backend, tmp_path):
    calls = []

    def fake_model(action, inputs):
        calls.append(inputs["payload"])
        return {"observed": "model:" + inputs["payload"]}

    events = []
    if backend == "sqlite":
        run_sqlite(two_model_reviews, store_path=str(tmp_path / "state.sqlite"),
                   llm_backend=fake_model, trace=events.append, timeout=5)
    else:
        run(two_model_reviews, [Manager], {}, llm_backend=fake_model,
            trace=events.append, timeout=5)
    assert sorted(calls) == ["first", "second"]
    assert {event.get("outputs", {}).get("observed") for event in events
            if event["type"] == "act" and event.get("action") == "model_reply"} == {
                "model:first", "model:second"
            }


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_allocated_model_role_routes_by_static_role_name(backend, tmp_path):
    calls = []

    def reviewer_backend(action, inputs):
        calls.append((action.name, inputs["payload"], threading.current_thread().name))
        return {"observed": "reviewed:" + inputs["payload"]}

    router = make_lifeline_router({"Reviewer": reviewer_backend})
    events = []
    if backend == "sqlite":
        run_sqlite(two_model_reviews, store_path=str(tmp_path / "router.sqlite"),
                   llm_backend=router, trace=events.append, timeout=5)
    else:
        run(two_model_reviews, [Manager], {}, llm_backend=router,
            trace=events.append, timeout=5)
    assert sorted(calls) == [
        ("model_reply", "first", "Reviewer"),
        ("model_reply", "second", "Reviewer"),
    ]
    assert {event.get("outputs", {}).get("observed") for event in events
            if event["type"] == "act" and event.get("action") == "model_reply"} == {
                "reviewed:first", "reviewed:second"
            }


def test_durable_allocation_resumes_recorded_descriptor(tmp_path):
    from zippergen.store import load_spawn_instance, open_store

    path = str(tmp_path / "state.sqlite")

    def crash_after_allocation(event):
        if event["type"] == "fork":
            raise RuntimeError("crash after allocation")

    with pytest.raises(RuntimeError, match="crash after allocation"):
        run_sqlite(two_reviews, store_path=path, trace=crash_after_allocation,
                   timeout=5)
    with closing(open_store(path)) as conn:
        first = load_spawn_instance(conn, "0/0.0:1")
        assert first is not None
        assert first["bindings"]["Reviewer"].startswith("~spawn:")
    events = []
    run_sqlite(two_reviews, store_path=path, trace=events.append, timeout=5)
    with closing(open_store(path)) as conn:
        assert load_spawn_instance(conn, "0/0.0:1") == first
        assert conn.execute("SELECT COUNT(*) FROM spawn_instances").fetchone()[0] == 2
    assert not any(event["type"] == "init_send" and event["child_session"] == "0/0.0:1"
                   for event in events)


def test_fresh_human_tasks_are_distinct_and_resume_in_new_process(tmp_path):
    from zippergen.sqlite_runner import LocalSupervisor
    from zippergen.store import complete_human_task, list_history, open_store

    path = str(tmp_path / "human.sqlite")
    supervisor = LocalSupervisor(two_human_reviews, None, {}, store_path=path, timeout=0)
    errors = []

    def execute():
        try:
            supervisor.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    rows = []
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with closing(open_store(path)) as conn:
                rows = conn.execute(
                    "SELECT task_id, role, locator FROM human_tasks ORDER BY task_id"
                ).fetchall()
            if len(rows) == 2:
                break
            time.sleep(0.02)
        assert len(rows) == 2, f"expected two durable tasks, got {rows!r}"
        assert len({row[0] for row in rows}) == 2
        assert {row[1] for row in rows} == {"Reviewer"}
        assert len({row[2] for row in rows}) == 1
    finally:
        supervisor.stop.set()
        thread.join(timeout=3)
    assert not thread.is_alive()

    with closing(open_store(path)) as conn:
        complete_human_task(conn, rows[0][0], {"approved": True})
        complete_human_task(conn, rows[1][0], {"approved": False})

    code = """
import importlib.util
import sys
from zippergen import run_sqlite
spec = importlib.util.spec_from_file_location('fresh_human_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
run_sqlite(module.two_human_reviews, store_path=sys.argv[2], timeout=5)
"""
    resumed = subprocess.run([sys.executable, "-c", code, __file__, path],
                             capture_output=True, text=True, timeout=10)
    assert resumed.returncode == 0, resumed.stderr
    with closing(open_store(path)) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM human_tasks WHERE status = 'done'"
        ).fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM outstanding_messages").fetchone()[0] == 0
        events = [row["event"] for row in list_history(conn)]
    received = [event for event in events if event["type"] == "recv"
                and event.get("to") == "Manager" and event.get("from") == "Reviewer"]
    assert {event["bindings"]["approved"] for event in received} == {True, False}
