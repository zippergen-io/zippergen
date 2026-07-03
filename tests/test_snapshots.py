from zippergen.runtime import run
from tests.loop_fixture import counter_loop, A, B


def test_fixture_runs_inprocess_to_known_result():
    # Sanity-check the fixture on the thread-based path before durable tests.
    assert run(counter_loop, [A, B], {"A": {"n": 0, "limit": 3}}, timeout=10) == 3


import threading
from zippergen.projection import project
from zippergen.store import open_store, load_snapshot
from zippergen.serve import run_role


def _run_both(path, seed):
    la = project(counter_loop, A); lb = project(counter_loop, B)
    envs = {}
    def go(role, local, s):
        envs[role] = run_role(open_store(path), role, local, dict(s), counter_loop.ns)
    ta = threading.Thread(target=go, args=("A", la, seed))
    tb = threading.Thread(target=go, args=("B", lb, {}))
    ta.start(); tb.start(); ta.join(timeout=15); tb.join(timeout=15)
    return envs


def test_durable_run_matches_and_writes_snapshot(tmp_path):
    path = str(tmp_path / "s.sqlite")
    envs = _run_both(path, {"n": 0, "limit": 3})
    assert envs["A"]["n"] == 3
    # A owns the loop, so A snapshots at boundaries.
    snap = load_snapshot(open_store(path), "A")
    assert snap is not None and isinstance(snap["locator"], list)
    assert "out" in snap["floor"] and "cursors" in snap["floor"]


def test_resume_from_snapshot_matches_full_run(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _run_both(path, {"n": 0, "limit": 3})
    # Re-run A from its snapshot on the same store: must reach the same final env
    # and (idempotent replay) insert no new events.
    conn = open_store(path)
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    la = project(counter_loop, A)
    env_a = run_role(open_store(path), "A", la, {"n": 0, "limit": 3}, counter_loop.ns)
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert env_a["n"] == 3
    assert after == before


# ---------------------------------------------------------------------------
# Crash-after-snapshot + stale-fallback (Task 5)
# ---------------------------------------------------------------------------
from zippergen.syntax import EmptyStmt
from zippergen.runtime import _step, mock_llm
from zippergen.store import DurableChannel, write_snapshot
from zippergen.locator import loop_node_paths
from zippergen.human_backends import make_cli_human_backend


class _Crash(Exception):
    pass


def _run_role_crash_after_k_snapshots(path, role, local, seed, ns, k):
    """Mirror run_role, but raise _Crash right after writing the k-th snapshot."""
    from zippergen.serve import _try_resume, _maybe_snapshot
    conn = open_store(path)
    loop_paths = loop_node_paths(local)
    env, residual, since = _try_resume(conn, role, local, dict(seed))
    ch = DurableChannel(conn, role, since=since)
    hb = make_cli_human_backend()
    while ch.replaying() and not isinstance(residual, EmptyStmt):
        residual, prog = _step(residual, env, ch, ns, mock_llm, hb, None, None, {}, None)
        if not prog:
            break
    snaps = 0
    while not isinstance(residual, EmptyStmt):
        conn.execute("BEGIN IMMEDIATE")
        new_r, prog = _step(residual, env, ch, ns, mock_llm, hb, None, None, {}, None)
        if prog:
            ch.commit_txn(); residual = new_r
            if id(residual) in loop_paths:
                _maybe_snapshot(conn, role, env, loop_paths[id(residual)], ch)
                snaps += 1
                if snaps == k:
                    raise _Crash()
        else:
            ch.rollback_txn()
    return env


def test_crash_after_snapshot_resumes_from_tail(tmp_path):
    path = str(tmp_path / "s.sqlite")
    la = project(counter_loop, A); lb = project(counter_loop, B)
    envs = {}
    def run_b():
        envs["B"] = run_role(open_store(path), "B", lb, {}, counter_loop.ns)
    tb = threading.Thread(target=run_b); tb.start()
    try:
        _run_role_crash_after_k_snapshots(path, "A", la, {"n": 0, "limit": 3}, counter_loop.ns, k=1)
    except _Crash:
        pass
    # Supervisor restarts A: resumes from the snapshot + tail, finishes the loop.
    env_a = run_role(open_store(path), "A", la, {"n": 0, "limit": 3}, counter_loop.ns)
    tb.join(timeout=15)
    assert env_a["n"] == 3
    # A's data sends are not duplicated by the snapshot-resume: exactly one per
    # iteration (3). Control-broadcast rows (κ_ctrl) also use kind='msg', so
    # exclude them — the point is that the pre-crash committed send is not re-sent.
    conn = open_store(path)
    a_data_sends = conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE sender='A' AND kind='msg' AND payload NOT LIKE '%ctrl%'"
    ).fetchone()[0]
    assert a_data_sends == 3


def test_stale_snapshot_falls_back_to_full_replay(tmp_path):
    path = str(tmp_path / "s.sqlite")
    _run_both(path, {"n": 0, "limit": 3})   # completes; writes a real snapshot
    # Corrupt the snapshot with an unresolvable locator; run_role must discard it
    # and replay from seed, still reaching the correct result.
    conn = open_store(path)
    write_snapshot(conn, "A", {"n": 999}, [7, 7, 7], {"out": 0, "cursors": {}})
    la = project(counter_loop, A)
    env_a = run_role(open_store(path), "A", la, {"n": 0, "limit": 3}, counter_loop.ns)
    assert env_a["n"] == 3   # seed replay wins, not the bogus env n=999
