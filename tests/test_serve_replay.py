"""run_role drives one role over the durable store, and a fresh run_role on the
same store resumes to the identical final env without duplicating events."""
from zippergen.syntax import Lifeline, Var, VarExpr, LitExpr, MsgStmt, IfStmt, SeqStmt, ActStmt
from zippergen.actions import pure
from zippergen.projection import project
from zippergen.store import open_store, DurableChannel
from zippergen.serve import run_role, seed_env
from zippergen.role_runner import RoleRunner
from tests.test_examples_regression import _two_role_branch_workflow, A, B

def _run_both(conn_a, conn_b, wf, seed):
    la = project(wf, A); lb = project(wf, B)
    import threading
    envs = {}
    def go(conn, role, local, seed_env):
        envs[role] = run_role(conn, role, local, dict(seed_env), wf.ns)
    ta = threading.Thread(target=go, args=(conn_a, "A", la, seed))
    tb = threading.Thread(target=go, args=(conn_b, "B", lb, {}))
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)
    return envs

def test_run_role_completes_two_role_branch(tmp_path):
    path = str(tmp_path / "s.sqlite")
    wf = _two_role_branch_workflow()
    envs = _run_both(open_store(path), open_store(path), wf, {"x": 7})
    assert envs["A"]["ok"] is True

def test_role_runner_direct_completes_two_role_branch(tmp_path):
    path = str(tmp_path / "s.sqlite")
    wf = _two_role_branch_workflow()
    la = project(wf, A); lb = project(wf, B)
    import threading
    envs = {}
    ta = threading.Thread(target=lambda: envs.__setitem__(
        "A", RoleRunner(open_store(path), "A", la, {"x": 7}, wf.ns).run()))
    tb = threading.Thread(target=lambda: envs.__setitem__(
        "B", run_role(open_store(path), "B", lb, {}, wf.ns)))
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)
    assert envs["A"]["ok"] is True

def test_run_role_replay_is_idempotent(tmp_path):
    path = str(tmp_path / "s.sqlite")
    wf = _two_role_branch_workflow()
    conn_a, conn_b = open_store(path), open_store(path)
    envs1 = _run_both(conn_a, conn_b, wf, {"x": 7})
    before = conn_a.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    # Re-run both roles on the same store: pure replay, no new events.
    la = project(wf, A); lb = project(wf, B)
    env_a = run_role(open_store(path), "A", la, {"x": 7}, wf.ns)
    env_b = run_role(open_store(path), "B", lb, {}, wf.ns)
    after = conn_a.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert env_a["ok"] is True
    assert after == before  # replay reserved every recorded send; nothing new inserted


def test_seed_env_persists_then_reads_back(tmp_path):
    path = str(tmp_path / "s.sqlite")
    wf = _two_role_branch_workflow()
    conn1 = open_store(path)
    got1 = seed_env(conn1, "A", wf, {"x": 42})
    assert got1 == {"x": 42}
    # Restart: different inputs are ignored; the recorded seed wins.
    conn2 = open_store(path)
    got2 = seed_env(conn2, "A", wf, {"x": -1})
    assert got2 == {"x": 42}
    rows = conn2.execute("SELECT COUNT(*) FROM events WHERE kind='seed' AND sender='A'").fetchone()[0]
    assert rows == 1


import threading


class _Crash(Exception):
    pass


def _run_role_crash_after_n_commits(path, role, local, seed, ns, n):
    """Run a role but raise _Crash right after the n-th live commit."""
    conn = open_store(path)
    ch = DurableChannel(conn, role)
    from zippergen.syntax import EmptyStmt
    from zippergen.runtime import _step, mock_llm
    from zippergen.human_backends import make_cli_human_backend
    hb = make_cli_human_backend()
    env = dict(seed); residual = local; commits = 0
    while ch.replaying() and not isinstance(residual, EmptyStmt):
        residual, prog = _step(residual, env, ch, ns, mock_llm, hb, None, None, {}, None)
        if not prog:
            break
    while not isinstance(residual, EmptyStmt):
        conn.execute("BEGIN")
        new_r, prog = _step(residual, env, ch, ns, mock_llm, hb, None, None, {}, None)
        if prog:
            ch.commit_txn(); residual = new_r; commits += 1
            if commits == n:
                raise _Crash()
        else:
            ch.rollback_txn()
    return env


def test_kill_and_resume_reaches_same_state(tmp_path):
    path = str(tmp_path / "s.sqlite")
    wf = _two_role_branch_workflow()
    la = project(wf, A); lb = project(wf, B)

    # B runs to completion in a background thread; A crashes after its 1st commit.
    envs = {}
    def run_b():
        envs["B"] = run_role(open_store(path), "B", lb, {}, wf.ns)
    tb = threading.Thread(target=run_b); tb.start()

    try:
        _run_role_crash_after_n_commits(path, "A", la, {"x": 7}, wf.ns, n=1)
    except _Crash:
        pass

    # Supervisor "restarts" A: fresh process, same store, no --input needed
    # because the seed was recorded. It must replay past the committed send.
    env_a = run_role(open_store(path), "A", la, {"x": 7}, wf.ns)
    tb.join(timeout=10)

    assert env_a["ok"] is True
    # Exactly one send from A survived (its first message), never duplicated.
    conn = open_store(path)
    a_sends = conn.execute(
        "SELECT COUNT(*) FROM events WHERE sender='A' AND kind='msg'").fetchone()[0]
    assert a_sends == 1
