"""CPL follows concrete job handoffs, including the durable replay boundary."""
import json
import subprocess
import sys
import threading

import pytest

from zippergen.pool_store import execute_pool_operation
from zippergen.pools import PoolOperation, PoolError, PoolOperationConflict
from zippergen.projection import project
from zippergen.role_runner import RoleRunner
from zippergen.runtime import _build_formula_monitors
from zippergen.store import load_role_state, open_store
from zippergen.syntax import _ordered_workflow_lifelines
from zippergen.value_codec import decode_value
from zippergen.workflow_io import load_workflow_spec


SOURCE = '''
from zippergen import At, Here, Json, Lifeline, Pool, pure, workflow
A, B, C = Lifeline("A"), Lifeline("B"), Lifeline("C")
jobs = Pool("jobs")

@pure
def value(x: Json) -> Json:
    return x

@pure
def request_of(claim: Json) -> Json:
    return claim["payload"]["request"]

@pure
def id_of(claim: Json) -> str:
    return claim["job_id"]

approved_for_job = ((At[A].approved == True) &
                    (At[A].request == Here.request) &
                    (At[A].job == Here.job_id))
reviewed_by_b = At[B].reviewed == True

@workflow
def handoffs():
    A: approved = value(True)
    A: request = value("one")
    A: job = jobs.put({"request": "one"})
    A: approved = value(False)
    A: request = value("two")
    A: job = jobs.put({"request": "two"})
    A: request = value("future")

    B: first = jobs.try_claim()
    if (first is not None) @ B:
        B: request = request_of(first)
        B: job_id = id_of(first)
        if approved_for_job @ B:
            B: accepted_first = value(True)
        else:
            B: accepted_first = value(False)
        B: reviewed = value(True)
        B: released = jobs.release(first)
    B: second = jobs.try_claim()
    if (second is not None) @ B:
        B: request = request_of(second)
        B: job_id = id_of(second)
        if approved_for_job @ B:
            B: accepted_second = value(True)
        else:
            B: accepted_second = value(False)
        B: completed = jobs.ack(second)

    C: claim = jobs.try_claim()
    if (claim is not None) @ C:
        C: request = request_of(claim)
        C: job_id = id_of(claim)
        if (approved_for_job & reviewed_by_b) @ C:
            C: accepted = value(True)
        else:
            C: accepted = value(False)
        C: completed = jobs.ack(claim)
    C: empty = jobs.try_claim()
'''


def write_workflow(tmp_path, source=SOURCE):
    path = tmp_path / "workflow.py"
    path.write_text(source)
    return f"{path}:handoffs"


def run_roles(spec, store, roles=("A", "B", "C")):
    """Choose a legal schedule so handoff assertions never rely on timing.

    There are no FIFO messages in these workflows. Each role has only local
    actions and locally owned guards, and can finish without its peers.
    """
    wf, _ = load_workflow_spec(spec)
    lifelines = _ordered_workflow_lifelines(wf)
    monitors, conditions = _build_formula_monitors(wf, lifelines)
    result = {}
    thread = threading.current_thread()
    original_name = thread.name
    for role in roles:
        conn = open_store(str(store))
        thread.name = role
        try:
            result[role] = RoleRunner(
                conn, role, project(wf, next(ll for ll in lifelines if ll.name == role)),
                {}, wf.ns, monitor=monitors[role], formula_conditions=conditions,
            ).run()
        finally:
            conn.close()
            thread.name = original_name
    return result


def test_jobs_export_post_put_state_and_release_exports_worker_state(tmp_path):
    spec = write_workflow(tmp_path)
    store = tmp_path / "run.sqlite"
    result = run_roles(spec, store)
    assert result["B"]["accepted_first"] is True
    assert result["B"]["accepted_second"] is False
    assert result["C"]["accepted"] is True
    assert result["C"]["empty"] is None
    conn = open_store(str(store))
    try:
        events = [json.loads(row[0]) for row in conn.execute("SELECT payload FROM history")]
        acts = [e for e in events if e["type"] == "act"]
        put = next(e for e in acts if e["action"] == "jobs_put")
        release = next(e for e in acts if e["action"] == "jobs_release")
        claims = [e for e in acts if e["action"] == "jobs_try_claim"]
        assert claims[0]["causal_vc"] == put["vc"]
        assert claims[0]["vc"]["B"] == 1  # one act, no synthetic receive
        assert claims[2]["causal_vc"] == release["vc"]
        assert claims[2]["vc"]["C"] == 1
        assert "causal_vc" not in claims[3]  # empty doesn't import pool state
        a_state = load_role_state(conn, "A")["monitor"]
        c_state = load_role_state(conn, "C")["monitor"]
        assert decode_value(c_state["field_view"])["A"]["request"] == "one"
        assert c_state["vc"]["A"] < a_state["vc"]["A"]
    finally:
        conn.close()


def test_empty_claim_imports_nothing_even_when_the_pool_has_claimed_jobs(tmp_path):
    source = SOURCE.replace('B: released = jobs.release(first)', 'B: released = value(False)')
    spec = write_workflow(tmp_path, source)
    store = tmp_path / "run.sqlite"
    result = run_roles(spec, store)
    assert result["C"]["claim"] is None
    conn = open_store(str(store))
    try:
        state = load_role_state(conn, "C")["monitor"]
        assert state["vc"] == {"A": 0, "B": 0, "C": 3}
        assert not decode_value(state["field_view"])["A"]
        assert not decode_value(state["field_view"])["B"]
    finally:
        conn.close()


def test_expiry_preserves_published_context_without_importing_the_failed_worker(tmp_path, monkeypatch):
    source = SOURCE.replace('B: released = jobs.release(first)', 'B: released = value(False)')
    spec = write_workflow(tmp_path, source)
    store = tmp_path / "run.sqlite"
    monkeypatch.setattr("zippergen.pool_store.time.time", lambda: 1000)
    run_roles(spec, store, ("A", "B"))
    monkeypatch.setattr("zippergen.pool_store.time.time", lambda: 1400)
    result = run_roles(spec, store, ("C",))
    assert result["C"]["claim"]["payload"]["request"] == "one"
    assert result["C"]["accepted"] is False  # B never published its review
    conn = open_store(str(store))
    try:
        state = load_role_state(conn, "C")["monitor"]
        assert decode_value(state["field_view"])["A"]["approved"] is True
        assert state["vc"]["B"] == 0
    finally:
        conn.close()


def test_pool_receipts_freeze_context_and_roll_back_failed_observations(tmp_path):
    conn = open_store(str(tmp_path / "run.sqlite"))
    seen = []

    def call(op, op_id, inputs, stamp, *, key="before", owner="A"):
        def observe(result, incoming):
            seen.append(incoming)
            return stamp
        return execute_pool_operation(
            conn, PoolOperation("jobs", op, 300), operation_id=op_id,
            owner=owner, inputs=inputs, context_key=key, observe=observe,
        )

    try:
        call("put", "p", {"payload": "one"}, "producer")
        claim = call("try_claim", "c", {}, "claim", owner="B")
        assert seen[-1] == "producer"
        call("release", "r", {"claim": claim}, "release", owner="B")
        call("try_claim", "new", {}, "next", owner="C")
        assert seen[-1] == "release"
        assert call("try_claim", "c", {}, "claim", owner="B") == claim
        assert seen[-1] == "producer"  # retry didn't reread the released job
        with pytest.raises(PoolOperationConflict):
            call("try_claim", "c", {}, "claim", key="changed", owner="B")

        def fail_observation(*_):
            raise RuntimeError("failed CPL predicate")

        with pytest.raises(RuntimeError, match="failed CPL predicate"):
            execute_pool_operation(
                conn, PoolOperation("jobs", "put", 300), operation_id="failed",
                owner="A", inputs={"payload": "lost"},
                context_key="before", observe=fail_observation,
            )
        assert conn.execute("SELECT count(*) FROM pool_jobs").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM pool_job_context").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM pool_operations WHERE operation_id='failed'").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_monitored_claim_refuses_missing_context_instead_of_inventing_a_past(tmp_path):
    conn = open_store(str(tmp_path / "run.sqlite"))
    try:
        execute_pool_operation(conn, PoolOperation("jobs", "put", 300),
                               operation_id="p", owner="A", inputs={"payload": "one"})
        with pytest.raises(PoolError, match="missing CPL context"):
            execute_pool_operation(
                conn, PoolOperation("jobs", "try_claim", 300), operation_id="c",
                owner="B", inputs={}, context_key="before", observe=lambda *_: "stamp",
            )
        assert conn.execute("SELECT status FROM pool_jobs").fetchone()[0] == "ready"
    finally:
        conn.close()


def test_pool_causal_policy_refuses_pre_handoff_state_before_running(tmp_path, monkeypatch):
    from zippergen.control import program_fingerprint
    from zippergen.sqlite_runner import run_sqlite
    from zippergen.store import claim_workflow_identity, WorkflowIdentityError

    spec = write_workflow(tmp_path)
    wf, _ = load_workflow_spec(spec)
    locals_ = {ll.name: project(wf, ll) for ll in _ordered_workflow_lifelines(wf)}
    current_policy = PoolOperation.semantics
    with monkeypatch.context() as patch:
        patch.setattr(PoolOperation, "semantics", lambda self: {
            k: v for k, v in current_policy(self).items() if k != "causality"
        })
        old = program_fingerprint(locals_)
    assert old != program_fingerprint(locals_)
    store = tmp_path / "run.sqlite"
    conn = open_store(str(store))
    try:
        claim_workflow_identity(conn, wf.name, old)
        with pytest.raises(WorkflowIdentityError):
            run_sqlite(wf, store_path=str(store))
        assert conn.execute("SELECT count(*) FROM role_state").fetchone()[0] == 0
        assert not conn.execute("SELECT name FROM sqlite_master WHERE name='pool_jobs'").fetchall()
    finally:
        conn.close()


def test_causality_is_transitive_across_messages_and_a_pool(tmp_path):
    spec = write_workflow(tmp_path, '''
from zippergen import At, Lifeline, Pool, pure, workflow
A, B, C, D = (Lifeline(name) for name in ("A", "B", "C", "D"))
jobs = Pool("jobs")
approved = At[A].authorized == True
@pure
def flag(x: bool) -> bool:
    return x
@workflow
def handoffs():
    A: authorized = flag(True)
    A(authorized) >> B(authorized)
    B: job = jobs.put("request")
    C: claim = jobs.try_claim()
    C(claim) >> D(claim)
    if approved @ D:
        D: accepted = flag(True)
    else:
        D: accepted = flag(False)
''')
    result = run_roles(spec, tmp_path / "run.sqlite", ("A", "B", "C", "D"))
    assert result["D"]["accepted"] is True


@pytest.mark.parametrize("allow", [True, False])
def test_cpl_pool_example_runs_both_approval_branches(allow):
    from zippergen.sqlite_runner import run_sqlite
    wf, _ = load_workflow_spec("examples/work_pool_cpl/workflow.py:approved_work_pool")
    expected = "processed report-17" if allow else "rejected report-17"
    assert run_sqlite(wf, initial_envs={"Producer": {"allow": allow}}, timeout=5) == expected


DRIVER = '''
import json, os, sys
from tests.test_pool_causality import run_roles
from zippergen.role_runner import RoleRunner
from zippergen.pools import PoolOperation
original = RoleRunner._resolve_external
def crash(self, pending):
    result = original(self, pending)
    op = pending.node.action.fn
    target = sys.argv[3]
    if isinstance(op, PoolOperation) and (op.operation == target or (
        target == "empty" and op.operation == "try_claim"
        and next(iter(result.outputs.values())) is None
    )):
        os._exit(91)
    return result
if sys.argv[3] != "resume":
    RoleRunner._resolve_external = crash
print(json.dumps(run_roles(sys.argv[1], sys.argv[2])))
'''


@pytest.mark.parametrize("operation", ["put", "try_claim", "release", "ack", "empty"])
def test_fresh_process_resume_replays_the_same_causal_event_once(tmp_path, operation):
    spec = write_workflow(tmp_path)
    store = tmp_path / "run.sqlite"
    killed = subprocess.run(
        [sys.executable, "-c", DRIVER, spec, str(store), operation],
        text=True, capture_output=True, timeout=20,
    )
    assert killed.returncode == 91, killed.stderr
    conn = open_store(str(store))
    try:
        conn.execute("DELETE FROM history")
    finally:
        conn.close()
    resumed = subprocess.run(
        [sys.executable, "-c", DRIVER, spec, str(store), "resume"],
        text=True, capture_output=True, timeout=20,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(resumed.stdout)
    assert result["B"]["accepted_first"] is True
    assert result["B"]["accepted_second"] is False
    assert result["C"]["accepted"] is True

    clean_store = tmp_path / "clean.sqlite"
    run_roles(spec, clean_store)
    conn = open_store(str(store))
    clean = open_store(str(clean_store))
    try:
        for role in ("A", "B", "C"):
            restored = load_role_state(conn, role)["monitor"]
            expected = load_role_state(clean, role)["monitor"]
            assert restored["vc"] == expected["vc"]
            assert restored["view"] == expected["view"]
            assert restored["val"] == expected["val"]
        assert conn.execute("SELECT count(*) FROM pool_operations").fetchone()[0] == 9
    finally:
        conn.close()
        clean.close()
