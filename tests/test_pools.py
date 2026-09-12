"""Pool semantics at the transaction boundary and through real recovery."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import threading

import pytest

from zippergen import ClaimExpired, Pool, PoolError, PoolOperationConflict
from zippergen.pool_store import execute_pool_operation, pool_operation_id
from zippergen.pools import PoolOperation
from zippergen.store import open_store
from zippergen.sqlite_runner import run_sqlite
from zippergen.workflow_io import load_workflow_spec


@pytest.fixture
def pool_db(tmp_path):
    conn = open_store(str(tmp_path / "run.sqlite"))
    try:
        yield conn
    finally:
        conn.close()


def invoke(conn, op_id, operation, value=None, *, owner="Worker", pool="jobs", lease=300):
    inputs = {"payload": value} if operation == "put" else (
        {} if operation == "try_claim" else {"claim": value}
    )
    return execute_pool_operation(
        conn, PoolOperation(pool, operation, lease), operation_id=op_id,
        owner=owner, inputs=inputs,
    )


def test_fifo_claims_and_explicit_completion(pool_db):
    one = invoke(pool_db, "p1", "put", {"number": 1})
    two = invoke(pool_db, "p2", "put", {"number": 2})
    first = invoke(pool_db, "c1", "try_claim", owner="A")
    second = invoke(pool_db, "c2", "try_claim", owner="B")
    assert [first["job_id"], second["job_id"]] == [one, two]
    assert first["payload"] == {"number": 1}
    assert invoke(pool_db, "empty", "try_claim") is None
    # Empty does not mean finished: both jobs are still claimed.
    assert pool_db.execute("SELECT count(*) FROM pool_jobs WHERE status='claimed'").fetchone()[0] == 2
    assert invoke(pool_db, "ack2", "ack", second, owner="B") is True
    assert invoke(pool_db, "ack1", "ack", first, owner="A") is True
    assert invoke(pool_db, "still_empty", "try_claim") is None
    assert pool_db.execute("SELECT count(*) FROM pool_jobs WHERE status='completed'").fetchone()[0] == 2


def test_lost_responses_are_replayed_including_none(pool_db):
    assert invoke(pool_db, "empty", "try_claim") is None
    job = invoke(pool_db, "put", "put", "payload")
    assert invoke(pool_db, "put", "put", "payload") == job
    assert invoke(pool_db, "empty", "try_claim") is None
    claim = invoke(pool_db, "claim", "try_claim")
    assert invoke(pool_db, "claim", "try_claim") == claim
    assert invoke(pool_db, "ack", "ack", claim) is True
    assert invoke(pool_db, "ack", "ack", claim) is True
    assert invoke(pool_db, "claim", "try_claim") == claim
    assert pool_db.execute("SELECT count(*) FROM pool_jobs").fetchone()[0] == 1


def test_reusing_an_operation_id_with_other_arguments_is_rejected(pool_db):
    invoke(pool_db, "id", "put", "original")
    for operation, value, owner, pool in [
        ("put", "changed", "Worker", "jobs"),
        ("put", "original", "Other", "jobs"),
        ("put", "original", "Worker", "other_pool"),
        ("try_claim", None, "Worker", "jobs"),
    ]:
        with pytest.raises(PoolOperationConflict):
            invoke(pool_db, "id", operation, value, owner=owner, pool=pool)
    assert pool_db.execute("SELECT count(*) FROM pool_jobs").fetchone()[0] == 1


def test_release_rejoins_tail_and_its_replay_cannot_release_a_new_claim(pool_db):
    first = invoke(pool_db, "p1", "put", 1)
    second = invoke(pool_db, "p2", "put", 2)
    claim = invoke(pool_db, "c1", "try_claim")
    assert invoke(pool_db, "release", "release", claim) is True
    assert invoke(pool_db, "c2", "try_claim")["job_id"] == second
    new_claim = invoke(pool_db, "c3", "try_claim")
    assert new_claim["job_id"] == first
    assert new_claim["token"] != claim["token"]
    assert invoke(pool_db, "release", "release", claim) is True
    assert invoke(pool_db, "empty", "try_claim") is None
    with pytest.raises(ClaimExpired):
        invoke(pool_db, "old_ack", "ack", claim)
    assert invoke(pool_db, "new_ack", "ack", new_claim) is True


def test_expiry_rejoins_tail_and_replay_does_not_renew(pool_db, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("zippergen.pool_store.time.time", lambda: now[0])
    first = invoke(pool_db, "p1", "put", 1, lease=10)
    old = invoke(pool_db, "c1", "try_claim", lease=10)
    second = invoke(pool_db, "p2", "put", 2, lease=10)
    now[0] = 1010.0
    assert invoke(pool_db, "c1", "try_claim", lease=10) == old
    with pytest.raises(ClaimExpired):
        invoke(pool_db, "stale_ack", "ack", old, lease=10)
    assert invoke(pool_db, "c2", "try_claim", lease=10)["job_id"] == second
    replacement = invoke(pool_db, "c3", "try_claim", lease=10)
    assert replacement["job_id"] == first
    assert replacement["token"] != old["token"]
    with pytest.raises(ClaimExpired):
        invoke(pool_db, "stale_release", "release", old, lease=10)
    assert invoke(pool_db, "ack", "ack", replacement, lease=10) is True
    now[0] = 2000.0
    assert invoke(pool_db, "ack", "ack", replacement, lease=10) is True


def test_another_worker_or_pool_cannot_acknowledge_a_claim(pool_db):
    invoke(pool_db, "p", "put", None)
    claim = invoke(pool_db, "c", "try_claim", owner="A")
    with pytest.raises(ClaimExpired):
        invoke(pool_db, "wrong_worker", "ack", claim, owner="B")
    with pytest.raises(PoolError, match="this pool"):
        invoke(pool_db, "wrong_pool", "ack", claim, owner="A", pool="other")
    assert invoke(pool_db, "right", "ack", claim, owner="A") is True


def test_pool_declaration_is_immutable_and_policy_is_consistent(pool_db):
    from dataclasses import FrozenInstanceError
    pool = Pool("jobs")
    with pytest.raises(FrozenInstanceError):
        pool.name = "other"
    assert pool.put.name == "jobs_put"
    for name in ("", "two pools", "a.b", "class"):
        with pytest.raises(ValueError):
            Pool(name)
    for lease in (0, -1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            Pool("jobs", lease_seconds=lease)
    invoke(pool_db, "p", "put", 1)
    with pytest.raises(PoolError, match="configuration"):
        invoke(pool_db, "c", "try_claim", lease=50)
    with pytest.raises(PoolError, match="SQLite"):
        pool.try_claim.fn()


def test_pool_operation_and_receipt_rollback_together(pool_db):
    invoke(pool_db, "seed", "put", "seed")
    pool_db.execute("""CREATE TRIGGER fail_receipt BEFORE INSERT ON pool_operations
        BEGIN SELECT RAISE(ABORT, 'injected receipt failure'); END""")
    with pytest.raises(Exception, match="injected receipt failure"):
        invoke(pool_db, "put", "put", "must roll back")
    with pytest.raises(Exception, match="injected receipt failure"):
        invoke(pool_db, "claim", "try_claim")
    assert pool_db.execute("SELECT payload,status FROM pool_jobs").fetchall() == [('"seed"', "ready")]
    assert pool_db.execute("SELECT count(*) FROM pool_operations").fetchone()[0] == 1


def test_competing_connections_claim_distinct_jobs(tmp_path):
    path = str(tmp_path / "concurrent.sqlite")
    conn = open_store(path)
    jobs = {invoke(conn, f"p{i}", "put", i) for i in range(20)}
    conn.close()
    barrier = threading.Barrier(8)

    def consume(index):
        db = open_store(path)
        claims = []
        try:
            barrier.wait(timeout=10)
            for visit in range(30):
                claim = invoke(db, f"c{index}-{visit}", "try_claim", owner=f"W{index}")
                if claim is None:
                    return claims
                claims.append(claim["job_id"])
                invoke(db, f"a{index}-{visit}", "ack", claim, owner=f"W{index}")
        finally:
            db.close()
        pytest.fail("consumer did not drain the finite pool")

    with ThreadPoolExecutor(max_workers=8) as executor:
        claimed = [job for result in executor.map(consume, range(8)) for job in result]
    assert len(claimed) == len(set(claimed)) == 20
    assert set(claimed) == jobs


RECOVERY_SOURCE = '''
from zippergen import Lifeline, Pool, workflow
A = Lifeline("A")
jobs = Pool("jobs")

@workflow
def recovery():
    A: first = jobs.put({"n": 1})
    A: second = jobs.put({"n": 2})
    A: initial = jobs.try_claim()
    A: released = jobs.release(initial)
    A: second_claim = jobs.try_claim()
    A: second_done = jobs.ack(second_claim)
    A: first_claim = jobs.try_claim()
    A: first_done = jobs.ack(first_claim)
    A: empty = jobs.try_claim()
'''

CRASH_DRIVER = '''
import os
import sys
from zippergen.role_runner import RoleRunner
from zippergen.pools import PoolOperation
from zippergen.sqlite_runner import run_sqlite
from zippergen.workflow_io import load_workflow_spec

original = RoleRunner._resolve_external
def interrupted(self, pending):
    result = original(self, pending)
    operation = pending.node.action.fn
    if isinstance(operation, PoolOperation):
        target = sys.argv[3]
        if operation.operation == target or (
            target == "empty" and operation.operation == "try_claim"
            and next(iter(result.values())) is None
        ):
            os._exit(91)
    return result
RoleRunner._resolve_external = interrupted
wf, _ = load_workflow_spec(sys.argv[1])
run_sqlite(wf, store_path=sys.argv[2])
'''


@pytest.mark.parametrize("operation", ["put", "try_claim", "ack", "release", "empty"])
def test_fresh_process_recovers_an_effect_committed_before_role_progress(tmp_path, operation):
    source = tmp_path / "workflow.py"
    source.write_text(RECOVERY_SOURCE)
    store = tmp_path / "run.sqlite"
    spec = f"{source}:recovery"
    killed = subprocess.run(
        [sys.executable, "-c", CRASH_DRIVER, spec, str(store), operation],
        capture_output=True, text=True, timeout=20,
    )
    assert killed.returncode == 91, killed.stderr
    conn = open_store(str(store))
    try:
        conn.execute("DELETE FROM history")
    finally:
        conn.close()
    # A fresh interpreter imports the source again and resumes the saved run.
    resumed = subprocess.run(
        [sys.executable, "-c", """
import json, sys
from zippergen.sqlite_runner import run_sqlite
from zippergen.workflow_io import load_workflow_spec
wf, _ = load_workflow_spec(sys.argv[1])
print(json.dumps(run_sqlite(wf, store_path=sys.argv[2])))
""", spec, str(store)], capture_output=True, text=True, timeout=20,
    )
    assert resumed.returncode == 0, resumed.stderr
    env = json.loads(resumed.stdout)["A"]
    assert env["second_claim"]["payload"] == {"n": 2}
    assert env["first_claim"]["payload"] == {"n": 1}
    assert env["empty"] is None
    conn = open_store(str(store))
    try:
        assert conn.execute("SELECT status FROM pool_jobs").fetchall() == [("completed",), ("completed",)]
        assert conn.execute("SELECT count(*) FROM pool_operations").fetchone()[0] == 9
    finally:
        conn.close()


def test_logical_ids_distinguish_loop_visits_roles_and_parallel_sites():
    same = pool_operation_id("A", 2, [0, 1])
    assert same == pool_operation_id("A", 2, [0, 1])
    assert len({same, pool_operation_id("A", 3, [0, 1]), pool_operation_id("B", 2, [0, 1]), pool_operation_id("A", 2, [1, 0])}) == 4


def test_example_runs_with_two_competing_workers_and_fresh_pools():
    wf, _ = load_workflow_spec("examples/work_pool/workflow.py:work_pool")
    for _ in range(2):
        assert run_sqlite(wf) == "processed job 1; processed job 2"
    assert wf() == "processed job 1; processed job 2"


def test_plain_cli_run_selects_temporary_sqlite_for_pools(tmp_path, monkeypatch, capsys):
    from zippergen.serve import main
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    assert main(["run", "--workflow", "examples/work_pool/workflow.py:work_pool"]) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "processed job 1; processed job 2"


def test_plain_pool_run_keeps_human_input_in_the_terminal(tmp_path, monkeypatch, capsys):
    from zippergen.serve import main
    monkeypatch.setenv("ZIPPERGEN_HOME", str(tmp_path / "home"))
    source = tmp_path / "workflow.py"
    source.write_text('''
from zippergen import Lifeline, Pool, human, workflow
A = Lifeline("A")
jobs = Pool("jobs")
@human(kind="confirm", instruction="Finish the job?", outputs=["approved: bool"])
def approve(): ...
@workflow
def review():
    A: submitted = jobs.put("job")
    A: claim = jobs.try_claim()
    A: approved = approve()
    if approved @ A:
        A: completed = jobs.ack(claim)
    else:
        A: released = jobs.release(claim)
    return approved @ A
''')
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert main(["run", "--workflow", f"{source}:review", "--timeout", "5"]) == 0
    assert '"result": true' in capsys.readouterr().out


@pytest.mark.parametrize("count", [0, 1])
def test_example_empty_branches_when_workers_outnumber_jobs(tmp_path, count):
    from pathlib import Path
    source = Path("examples/work_pool/workflow.py").read_text()
    source = source.replace(
        "@workflow\ndef work_pool", "@pure\ndef skipped_submission() -> str:\n    return 'ready'\n\n@workflow\ndef work_pool",
    ).replace('jobs.put({"number": 1})', 'skipped_submission()')
    if count == 0:
        source = source.replace('jobs.put({"number": 2})', 'skipped_submission()')
    path = tmp_path / "workflow.py"
    path.write_text(source)
    wf, _ = load_workflow_spec(f"{path}:work_pool")
    expected = "no job available; " + ("processed job 2" if count else "no job available")
    assert run_sqlite(wf) == expected


def test_a_loop_visit_is_a_new_poll_after_a_committed_empty_result(tmp_path):
    source = tmp_path / "workflow.py"
    source.write_text('''
from zippergen import Json, Lifeline, Pool, Var, pure, workflow
A = Lifeline("A")
jobs = Pool("jobs")
attempt = Var("attempt", int, default=0)
@pure
def inc(value: int) -> int:
    return value + 1
@workflow
def polling():
    while (attempt < 2) @ A:
        A: claim = jobs.try_claim()
        if (claim is None) @ A:
            A: submitted = jobs.put({"n": 1})
        else:
            A: completed = jobs.ack(claim)
        A: attempt = inc(attempt)
''')
    wf, _ = load_workflow_spec(f"{source}:polling")
    result = run_sqlite(wf)
    assert result["A"]["claim"]["payload"] == {"n": 1}
    assert result["A"]["completed"] is True


def test_validation_rejects_conflicting_pool_policies(tmp_path):
    from zippergen.validation import validate_workflow
    source = tmp_path / "workflow.py"
    source.write_text(RECOVERY_SOURCE.replace(
        'jobs = Pool("jobs")', 'jobs = Pool("jobs")\nother = Pool("jobs", lease_seconds=600)',
    ).replace('A: empty = jobs.try_claim()', 'A: empty = other.try_claim()'))
    wf, module = load_workflow_spec(f"{source}:recovery")
    report = validate_workflow(wf, module)
    assert not report["valid"]
    assert any(c["name"] == "work pool jobs" and c["status"] == "fail" for c in report["checks"])


def test_pool_policy_is_visible_and_changes_the_durable_identity(tmp_path):
    from zippergen.control import program_fingerprint
    from zippergen.projection import project
    from zippergen.semantic import semantic_diff
    from zippergen.validation import validate_workflow
    from zippergen.view import ViewOptions, render_workflow

    source = tmp_path / "workflow.py"
    source.write_text(RECOVERY_SOURCE)
    wf, module = load_workflow_spec(f"{source}:recovery")
    fingerprint = program_fingerprint({"A": project(wf, module.A)})
    assert "lease_seconds=300.0" in render_workflow(wf, options=ViewOptions(detail="full"))
    assert validate_workflow(wf, module)["valid"]
    source.write_text(RECOVERY_SOURCE.replace('Pool("jobs")', 'Pool("jobs", lease_seconds=600)'))
    changed, changed_module = load_workflow_spec(f"{source}:recovery")
    assert program_fingerprint({"A": project(changed, changed_module.A)}) != fingerprint
    assert semantic_diff(wf, changed, module, changed_module)["changed"]
