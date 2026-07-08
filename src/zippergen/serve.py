"""Per-role durable runtime: project one role, replay its committed history,
then run live, persisting each step atomically.

Non-deterministic results are journaled so replay reconstructs them instead of
re-executing: external acts (LLM/Human/Planner/Effect) and owner if/while
decisions are recorded as ``kind='act'``/``'decision'`` rows and consumed on
replay; a ``@pure`` act is deterministic by contract and recomputed. A blocking external
call runs OUTSIDE the SQLite write transaction (the lock is released first), then
its result is journaled and committed before env/residual advance.

Limitations (v1):
- External effects are at-least-once: a crash between an external call returning
  and its journal row committing re-runs the act on restart. Harmless for an LLM
  (paid twice); visible human actions are first materialized as durable
  ``human_tasks`` rows and can be answered out-of-band. Irreversible effects
  (e.g. sending mail) are NOT made exactly-once here — that needs effect-level
  idempotency keys.
- No snapshot fires inside a parallel region (the rebuilt residual changes
  identity each step), so a crash replays the region from the enclosing loop
  boundary; replay-length bounding inside parallel is a later refinement.
- CPL Formula guards use full replay for now because role snapshots do not yet
  persist monitor state.
- A blocking external act in one parallel branch stalls that role's other
  branches until it returns (no intra-role concurrency of external calls).

See docs/superpowers/specs/2026-07-04-durable-deploy-hardening-design.md."""
from __future__ import annotations

from zippergen.role_runner import (
    JournalContext,
    RoleRunner,
    _floor_coherent,
    _maybe_snapshot,
    _try_resume,
    run_role,
)


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


def _seed_inputs(wf: Workflow, inputs: dict) -> dict:
    """Var defaults from the workflow namespace, overlaid by caller inputs —
    parity with the in-process run() seeding (runtime.py:1014)."""
    from zippergen.syntax import Var
    env = {k: v.default for k, v in wf.ns.items() if isinstance(v, Var)}
    env.update(inputs)
    return env


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
    lifelines = _workflow_lifelines(wf)
    from zippergen.runtime import _build_formula_monitors
    monitors, formula_conditions = _build_formula_monitors(wf, lifelines)
    conn = open_store(args.store)
    env = seed_env(conn, args.role, wf, _seed_inputs(wf, _parse_inputs(args.input)))
    local = project(wf, role_ll)
    final = run_role(
        conn,
        args.role,
        local,
        env,
        wf.ns,
        monitor=monitors.get(args.role),
        formula_conditions=formula_conditions,
    )
    print(json.dumps({k: v for k, v in final.items()
                      if isinstance(v, (bool, int, float, str, type(None)))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
