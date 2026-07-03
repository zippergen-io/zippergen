"""Per-role durable runtime: project one role, replay its committed history,
then run live, persisting each step atomically."""
from __future__ import annotations

import time

from zippergen.syntax import EmptyStmt
from zippergen.runtime import _step, mock_llm
from zippergen.store import DurableChannel


def run_role(conn, role: str, local_stmt, env: dict, ns: dict, *,
             llm_backend=None, human_backend=None, trace=None) -> dict:
    if llm_backend is None:
        llm_backend = mock_llm
    if human_backend is None:
        from zippergen.human_backends import make_cli_human_backend
        human_backend = make_cli_human_backend()

    ch = DurableChannel(conn, role)
    residual = local_stmt

    # ---- replay: reconstruct (env, residual) from committed history --------
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
