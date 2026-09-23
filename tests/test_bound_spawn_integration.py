"""Bound child sessions execute on configured endpoint workers."""

import threading
import time
import subprocess
import sys
from contextlib import closing

import pytest

from zippergen import Json, Lifeline, Var, fragment, human, llm, pure, run, run_sqlite, spawn, workflow
from zippergen.syntax import _ordered_workflow_lifelines

Manager, Reviewer, IT = (Lifeline(name) for name in ("Manager", "Reviewer", "IT"))
binding = Var("binding", Json)
payload = Var("payload", str, default="first")
observed = Var("observed", str)
marker = Var("marker", str)
value = Var("value", int)
result = Var("result", int)
approved = Var("approved", bool)


@pure
def review_binding() -> Json:
    return {"Manager": "Manager", "Reviewer": "shared"}


@pure
def it_binding() -> Json:
    return {"Manager": "Manager", "IT": "shared"}


@pure
def reviewer_only_binding() -> Json:
    return {"Reviewer": "reviewer"}


@pure
def manager_self_binding() -> Json:
    return {"Manager": "Manager"}


@pure
def nested_it_binding() -> Json:
    return {"IT": "worker"}


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
def twice(value: int) -> int:
    return value * 2


@pure
def seven() -> int:
    return 7


@pure
def reviewer_on_it_endpoint() -> Json:
    return {"Reviewer": "IT"}


@pure
def east_binding() -> Json:
    return {"Manager": "Manager", "Reviewer": "east"}


@pure
def west_binding() -> Json:
    return {"Manager": "Manager", "Reviewer": "west"}


@llm(system="Return the endpoint marker.", user="{payload}", parse="text",
     outputs=(("observed", str),))
def model_reply(payload: str) -> str: ...


@human(kind="confirm", outputs=["approved: bool"], context="Request {payload}")
def approve_request(payload: str) -> bool: ...


@fragment
def review_child():
    Manager(payload) >> Reviewer(payload)
    Reviewer: observed = echo(payload)


@fragment
def it_child():
    Manager(payload) >> IT(payload)
    IT: observed = echo(payload)


@workflow
def shared_endpoint_roles():
    Manager: binding = review_binding()
    spawn(review_child, bindings=binding @ Manager,
          retain=(Manager,), fresh="all")
    Manager: payload = second()
    Manager: binding = it_binding()
    spawn(it_child, bindings=binding @ Manager,
          retain=(Manager,), fresh="all")


@fragment
def inherited_grandchild():
    Reviewer: value = five()
    with spawn(fresh="all"):
        Reviewer: result = twice(value)


@workflow
def recruit_then_inherit():
    Manager: binding = reviewer_only_binding()
    spawn(inherited_grandchild, bindings=binding @ Manager, retain=())


@fragment
def nested_bound_child():
    Reviewer: binding = nested_it_binding()
    with spawn(bindings=binding @ Reviewer, retain=()):
        IT: value = five()


@workflow
def recruit_then_recruit():
    Manager: binding = reviewer_only_binding()
    spawn(nested_bound_child, bindings=binding @ Manager, retain=())


@fragment
def self_child():
    Manager: value = five()


@workflow
def recruit_self():
    Manager: binding = manager_self_binding()
    spawn(self_child, bindings=binding @ Manager, retain=())


@fragment
def recruited_cell_child():
    Reviewer: value = five()


@workflow
def recruited_cell_isolated():
    IT: value = seven()
    Manager: binding = reviewer_on_it_endpoint()
    spawn(recruited_cell_child, bindings=binding @ Manager, retain=())
    return value @ IT


@workflow
def parent_finishes_before_acceptance():
    Manager: binding = review_binding()
    spawn(review_child, bindings=binding @ Manager,
          retain=(Manager,), fresh="all")
    Manager: marker = second()


@workflow
def runtime_selected_binding(binding: Json @ Manager):
    spawn(review_child, bindings=binding @ Manager, retain=(Manager,))


@fragment
def model_child():
    Manager(payload) >> Reviewer(payload)
    Reviewer: observed = model_reply(payload)


@workflow
def two_model_endpoints():
    Manager: binding = east_binding()
    spawn(model_child, bindings=binding @ Manager,
          retain=(Manager,), fresh="all")
    Manager: payload = second()
    Manager: binding = west_binding()
    spawn(model_child, bindings=binding @ Manager,
          retain=(Manager,), fresh="all")


@fragment
def human_child():
    Manager(payload) >> Reviewer(payload)
    Reviewer: approved = approve_request(payload)
    Reviewer(approved) >> Manager(approved)


@workflow
def two_human_sessions():
    Manager: binding = review_binding()
    spawn(human_child, bindings=binding @ Manager,
          retain=(Manager,), fresh="all")
    Manager: payload = second()
    spawn(human_child, bindings=binding @ Manager,
          retain=(Manager,), fresh="all")


def _execute(wf, backend, tmp_path, *, endpoints, trace):
    if backend == "sqlite":
        return run_sqlite(wf, endpoints=endpoints, trace=trace.append, timeout=5,
                          store_path=str(tmp_path / "state.sqlite"))
    return run(wf, list(_ordered_workflow_lifelines(wf)), {}, endpoints=endpoints,
               trace=trace.append, timeout=5)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_one_endpoint_executes_different_roles_in_overlapping_sessions(
    backend, tmp_path,
):
    events = []
    _execute(shared_endpoint_roles, backend, tmp_path,
             endpoints={"shared"}, trace=events)
    outputs = {(event.get("lifeline"), event.get("outputs", {}).get("observed"))
               for event in events if event["type"] == "act"}
    assert ("Reviewer", "first") in outputs
    assert ("IT", "second") in outputs
    sessions = {event["session"] for event in events
                if event["type"] == "act" and "observed" in event.get("outputs", {})}
    assert len(sessions) == 2


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recruit_can_fork_inherited_grandchild(backend, tmp_path):
    events = []
    _execute(recruit_then_inherit, backend, tmp_path,
             endpoints={"reviewer"}, trace=events)
    assert any(event["type"] == "act" and event.get("outputs", {}).get("result") == 10
               for event in events)
    assert any(event["type"] == "fork" and event.get("lifeline") == "Reviewer"
               for event in events)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recruit_can_create_nested_bound_child(backend, tmp_path):
    events = []
    _execute(recruit_then_recruit, backend, tmp_path,
             endpoints={"reviewer", "worker"}, trace=events)
    assert any(event["type"] == "act" and event.get("lifeline") == "IT"
               and event.get("outputs", {}).get("value") == 5 for event in events)
    assert len([event for event in events if event["type"] == "bind"]) == 2


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_owner_can_invite_its_own_endpoint_as_new_child_role(backend, tmp_path):
    events = []
    _execute(recruit_self, backend, tmp_path, endpoints=set(), trace=events)
    assert any(event["type"] == "init_send" and event["to"] == "Manager"
               and event["kind"] == "invite" for event in events)
    assert any(event["type"] == "act" and event.get("outputs", {}).get("value") == 5
               and event["session"] != "0" for event in events)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recruited_role_on_parent_endpoint_gets_new_cells(backend, tmp_path):
    events = []
    outcome = _execute(recruited_cell_isolated, backend, tmp_path,
                       endpoints=set(), trace=events)
    assert outcome == 7
    assert any(event["type"] == "act" and event.get("outputs", {}).get("value") == 5
               and event["session"] != "0" for event in events)


def test_application_message_waits_for_destination_activation(monkeypatch, tmp_path):
    from zippergen.sessions import SessionState

    sent = False
    events = []
    original = SessionState.accept_invitation

    def delay_reviewer(self, channel):
        if self.endpoint == "shared" and not sent:
            return False
        return original(self, channel)

    def record(event):
        nonlocal sent
        events.append(event)
        if event["type"] == "send" and event.get("to") == "Reviewer":
            sent = True

    monkeypatch.setattr(SessionState, "accept_invitation", delay_reviewer)
    run(shared_endpoint_roles, list(_ordered_workflow_lifelines(shared_endpoint_roles)),
        {}, endpoints={"shared"}, trace=record, timeout=5)
    first_send = next(index for index, event in enumerate(events)
                      if event["type"] == "send" and event.get("to") == "Reviewer")
    first_receive = next(index for index, event in enumerate(events)
                         if event["type"] == "recv" and event.get("to") == "Reviewer")
    assert first_send < first_receive


def test_parent_can_finish_before_recruit_accepts(monkeypatch):
    from zippergen.sessions import SessionState

    parent_finished = False
    events = []
    original = SessionState.accept_invitation

    def delay_reviewer(self, channel):
        if self.endpoint == "shared" and not parent_finished:
            return False
        return original(self, channel)

    def record(event):
        nonlocal parent_finished
        events.append(event)
        if event["type"] == "act" and event.get("outputs", {}).get("marker") == "second":
            parent_finished = True

    monkeypatch.setattr(SessionState, "accept_invitation", delay_reviewer)
    run(parent_finishes_before_acceptance,
        list(_ordered_workflow_lifelines(parent_finishes_before_acceptance)), {},
        endpoints={"shared"}, trace=record, timeout=5)
    completed = next(index for index, event in enumerate(events)
                     if event["type"] == "act" and "marker" in event.get("outputs", {}))
    received = next(index for index, event in enumerate(events)
                    if event["type"] == "recv" and event.get("to") == "Reviewer")
    assert completed < received


@pytest.mark.parametrize(
    "selected, message",
    [
        ({"Manager": "Manager"}, "exactly these role keys"),
        ({"Manager": "Manager", "Reviewer": "reviewer", "Extra": "extra"},
         "exactly these role keys"),
        ({"Manager": "Manager", "Reviewer": "Manager"}, "injective"),
        ({"Manager": "other", "Reviewer": "reviewer"}, "preserve its endpoint"),
        ({"Manager": "Manager", "Reviewer": "bad endpoint"}, "Invalid endpoint"),
        ({"Manager": "Manager", "Reviewer": "unregistered"}, "not registered"),
    ],
)
def test_invalid_runtime_binding_is_rejected(selected, message):
    with pytest.raises(RuntimeError, match=message):
        run(runtime_selected_binding, [Manager], {"Manager": {"binding": selected}},
            endpoints={"reviewer"}, timeout=3)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_retained_fresh_cell_copies_at_local_fork(backend, tmp_path, monkeypatch):
    from zippergen.sessions import ScopedChannel

    original = ScopedChannel.send_init
    changed = False

    def update_between_selection_and_fork(self, stmt):
        nonlocal changed
        descriptor = original(self, stmt)
        if not changed and stmt.recipient == Reviewer:
            changed = True
            self.root_env["payload"] = "late"
        return descriptor

    monkeypatch.setattr(ScopedChannel, "send_init", update_between_selection_and_fork)
    events = []
    _execute(shared_endpoint_roles, backend, tmp_path,
             endpoints={"shared"}, trace=events)
    assert changed
    reviewer_result = next(event["outputs"]["observed"] for event in events
                           if event["type"] == "act" and event.get("lifeline") == "Reviewer"
                           and "observed" in event.get("outputs", {}))
    assert reviewer_result == "late"


def test_resume_refuses_missing_registered_endpoint(tmp_path):
    from zippergen.store import WorkflowIdentityError
    from zippergen.store import load_spawn_instance, open_store

    path = str(tmp_path / "state.sqlite")
    selected = {"Manager": "Manager", "Reviewer": "reviewer"}

    def crash_after_selection(event):
        if event["type"] == "init_send":
            raise RuntimeError("crash after selection")

    with pytest.raises(RuntimeError, match="crash after selection"):
        run_sqlite(runtime_selected_binding, store_path=path,
                   initial_envs={"Manager": {"binding": selected}},
                   endpoints={"reviewer"}, trace=crash_after_selection,
                   timeout=3)
    conn = open_store(path)
    try:
        descriptor = load_spawn_instance(conn, "0/0:1")
        assert descriptor is not None and descriptor["bindings"] == selected
    finally:
        conn.close()

    with pytest.raises(WorkflowIdentityError, match="workflow changed|endpoint|registered"):
        run_sqlite(runtime_selected_binding, store_path=path,
                   initial_envs={"Manager": {"binding": selected}},
                   endpoints=set(), timeout=1)
    with pytest.raises(WorkflowIdentityError, match="workflow changed|endpoint|registered"):
        run_sqlite(runtime_selected_binding, store_path=path,
                   initial_envs={"Manager": {"binding": selected}},
                   endpoints={"reviewer", "extra"}, timeout=1)


def test_resume_uses_committed_binding_selection(tmp_path):
    from zippergen.store import load_spawn_instance, open_store

    path = str(tmp_path / "state.sqlite")
    original = {"Manager": "Manager", "Reviewer": "reviewer"}
    changed = {"Manager": "Manager", "Reviewer": "other"}
    configured = {"reviewer", "other"}

    def crash_after_selection(event):
        if event["type"] == "init_send":
            raise RuntimeError("crash after selection")

    with pytest.raises(RuntimeError, match="crash after selection"):
        run_sqlite(runtime_selected_binding, store_path=path,
                   initial_envs={"Manager": {"binding": original}},
                   endpoints=configured, trace=crash_after_selection,
                   timeout=3)
    events = []
    run_sqlite(runtime_selected_binding, store_path=path,
               initial_envs={"Manager": {"binding": changed}},
               endpoints=configured, trace=events.append, timeout=3)
    conn = open_store(path)
    try:
        descriptor = load_spawn_instance(conn, "0/0:1")
        assert descriptor is not None and descriptor["bindings"] == original
        assert conn.execute("SELECT COUNT(*) FROM spawn_instances").fetchone()[0] == 1
    finally:
        conn.close()
    assert not any(event["type"] == "bind" for event in events)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recruited_llm_uses_physical_endpoint_backend(backend, tmp_path):
    calls = []

    def selected(name):
        def answer(action, inputs):
            calls.append((name, inputs["payload"]))
            return {"observed": f"{name}:{inputs['payload']}"}
        return answer

    def wrong_endpoint(action, inputs):
        raise AssertionError("default model backend must not handle recruited work")

    overrides = {
        "east": {"llm": selected("east")},
        "west": {"llm": selected("west")},
    }
    events = []
    kwargs = dict(endpoints={"east", "west"}, endpoint_backends=overrides,
                  llm_backend=wrong_endpoint, trace=events.append, timeout=5)
    if backend == "sqlite":
        run_sqlite(two_model_endpoints, store_path=str(tmp_path / "state.sqlite"),
                   **kwargs)
    else:
        run(two_model_endpoints, [Manager], {}, **kwargs)
    assert sorted(calls) == [("east", "first"), ("west", "second")]
    assert {event.get("outputs", {}).get("observed") for event in events
            if event["type"] == "act" and event.get("action") == "model_reply"} == {
                "east:first", "west:second"
            }


def test_recruited_human_tasks_are_distinct_and_resume_at_shared_endpoint(tmp_path):
    from zippergen.sqlite_runner import LocalSupervisor
    from zippergen.store import complete_human_task, list_history, open_store

    path = str(tmp_path / "human.sqlite")
    supervisor = LocalSupervisor(two_human_sessions, None, {}, store_path=path,
                                 endpoints={"shared"}, timeout=0)
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
        # Both visits execute the same template location; the durable task ID
        # must still distinguish their child sessions.
        assert len({row[2] for row in rows}) == 1
        assert {row[1] for row in rows} == {"shared"}
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
spec = importlib.util.spec_from_file_location('bound_human_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
run_sqlite(module.two_human_sessions, store_path=sys.argv[2],
           endpoints={'shared'}, timeout=5)
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
