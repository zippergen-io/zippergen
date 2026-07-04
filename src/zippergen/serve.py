"""Per-role durable runtime: project one role, replay its committed history,
then run live, persisting each step atomically."""
from __future__ import annotations

import sqlite3
import time

from zippergen.syntax import EmptyStmt, WhileStmt, WhileRecvStmt
from zippergen.runtime import _step, mock_llm
from zippergen.store import DurableChannel, load_snapshot, write_snapshot
from zippergen.locator import loop_node_paths, resolve_path


def _floor_coherent(conn, role: str, floor: dict) -> bool:
    """A floor is coherent only if it carries all three keys and none points past
    the committed log. A floor lacking "journal" predates journaling -> incoherent
    (forces full replay from seed)."""
    try:
        out = floor["out"]; cursors = floor["cursors"]; journal = floor["journal"]
    except (KeyError, TypeError):
        return False
    max_out = conn.execute(
        "SELECT MAX(rowid) FROM events WHERE sender=? AND kind IN ('msg','ctrl')",
        (role,),
    ).fetchone()[0] or 0
    if out > max_out:
        return False
    max_journal = conn.execute(
        "SELECT MAX(rowid) FROM events WHERE sender=? AND kind IN ('act','decision')",
        (role,),
    ).fetchone()[0] or 0
    if journal > max_journal:
        return False
    durable = {ck: c for ck, c in conn.execute(
        "SELECT chan_key, consumed FROM cursors WHERE role=?", (role,)).fetchall()}
    return all(v <= durable.get(k, 0) for k, v in cursors.items())


def _try_resume(conn, role: str, local_stmt, env: dict):
    """Return (env, residual, since) from a valid snapshot, else (env, local_stmt, None)."""
    snap = load_snapshot(conn, role)
    if snap is None:
        return env, local_stmt, None
    node = resolve_path(local_stmt, snap["locator"])
    if isinstance(node, (WhileStmt, WhileRecvStmt)) and _floor_coherent(conn, role, snap["floor"]):
        return dict(snap["env"]), node, snap["floor"]
    return env, local_stmt, None   # stale/invalid -> full replay from seed


def _maybe_snapshot(conn, role: str, env: dict, locator: list, ch) -> None:
    try:
        write_snapshot(conn, role, env, locator, ch.position())
    except (TypeError, ValueError, sqlite3.OperationalError):
        # Best-effort: a snapshot is a rebuildable cache and must never fail a
        # healthy role — skip on a non-serializable env OR a transient store
        # error (e.g. lock contention in the separate snapshot transaction).
        pass


def run_role(conn, role: str, local_stmt, env: dict, ns: dict, *,
             llm_backend=None, human_backend=None, trace=None) -> dict:
    if llm_backend is None:
        llm_backend = mock_llm
    if human_backend is None:
        from zippergen.human_backends import make_cli_human_backend
        human_backend = make_cli_human_backend()

    loop_paths = loop_node_paths(local_stmt)
    env, residual, since = _try_resume(conn, role, local_stmt, env)
    ch = DurableChannel(conn, role, since=since)

    # ---- replay: reconstruct (env, residual) from the tail (or full history) --
    # No transactions, no trace: put reserves recorded sends, try_get serves
    # recorded recvs. Determinism of local steps reproduces the exact boundary.
    while ch.replaying() and not isinstance(residual, EmptyStmt):
        residual, progressed = _step(
            residual, env, ch, ns, llm_backend, human_backend, None, None, {}, None)
        if not progressed:
            break  # blocked on live input at the replay boundary

    # ---- live: one transaction per progressing step ------------------------
    while not isinstance(residual, EmptyStmt):
        # BEGIN IMMEDIATE (not deferred): a recv step reads (SELECT) before it
        # writes (cursor INSERT in commit_txn); a deferred BEGIN would make that
        # a read->write upgrade, which SQLite fails immediately with
        # "database is locked" (bypassing busy_timeout) when the peer role holds
        # the write lock. Taking the write lock up front serializes cleanly, the
        # same pattern seed_env uses.
        conn.execute("BEGIN IMMEDIATE")
        new_residual, progressed = _step(
            residual, env, ch, ns, llm_backend, human_backend, None, trace, {}, None)
        if progressed:
            ch.commit_txn()
            residual = new_residual
            # At a loop-iteration boundary the residual is (by identity) a loop
            # node in the projected program; checkpoint env + position there.
            if id(residual) in loop_paths:
                _maybe_snapshot(conn, role, env, loop_paths[id(residual)], ch)
        else:
            ch.rollback_txn()
            time.sleep(0.02)
    return env


# ---------------------------------------------------------------------------
# CLI: `zippergen serve --workflow PATH --role NAME --store PATH [--input k=v]`
# ---------------------------------------------------------------------------
import argparse
import importlib.util
import json
import sys

from zippergen.syntax import Workflow, Lifeline
from zippergen.projection import project
from zippergen.store import open_store


def load_workflow(module_path: str, role_name: str) -> tuple[Workflow, Lifeline]:
    spec = importlib.util.spec_from_file_location("_zippergen_wf", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wf = next(v for v in vars(module).values() if isinstance(v, Workflow))
    lifelines = {ll.name: ll for ll in _workflow_lifelines(wf)}
    if role_name not in lifelines:
        raise SystemExit(f"role {role_name!r} not in workflow lifelines {sorted(lifelines)}")
    return wf, lifelines[role_name]


def _workflow_lifelines(wf: Workflow) -> tuple[Lifeline, ...]:
    from zippergen.syntax import _ordered_workflow_lifelines
    return _ordered_workflow_lifelines(wf)


def seed_env(conn, role: str, wf: Workflow, inputs: dict) -> dict:
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT payload FROM events WHERE kind='seed' AND sender=? ORDER BY rowid LIMIT 1",
            (role,),
        ).fetchone()
        if row is not None:
            conn.execute("ROLLBACK")
            return json.loads(row[0])
        conn.execute(
            "INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp) "
            "VALUES(?,?,?,?,?,?)",
            (role, None, None, "seed", json.dumps(inputs), None),
        )
        conn.execute("COMMIT")
        return dict(inputs)
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _parse_inputs(pairs: list[str]) -> dict:
    out: dict = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)      # 7 -> int, "true" via JSON, '"s"' -> str
        except json.JSONDecodeError:
            out[k] = v                  # bare string fallback
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zippergen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="run one role as a durable process")
    sv.add_argument("--workflow", required=True)
    sv.add_argument("--role", required=True)
    sv.add_argument("--store", required=True)
    sv.add_argument("--input", action="append", default=[], metavar="k=v")
    args = ap.parse_args(argv)

    wf, role_ll = load_workflow(args.workflow, args.role)
    conn = open_store(args.store)
    env = seed_env(conn, args.role, wf, _parse_inputs(args.input))
    local = project(wf, role_ll)
    final = run_role(conn, args.role, local, env, wf.ns)
    print(json.dumps({k: v for k, v in final.items()
                      if isinstance(v, (bool, int, float, str, type(None)))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
