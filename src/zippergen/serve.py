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
# CLI:
#   `zippergen run MODULE_OR_PATH:WORKFLOW [--llm SPEC] [--store PATH] [--input k=v]`
#   `zippergen serve --workflow PATH --role NAME --store PATH [--input k=v]`
# ---------------------------------------------------------------------------
import argparse
import hashlib
import importlib
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from zippergen.syntax import Workflow, Lifeline
from zippergen.projection import project
from zippergen.store import list_workflow_results, open_store


@dataclass(frozen=True)
class RunConfig:
    """Configuration passed to an optional module-level ``zippergen_setup`` hook."""

    workflow_spec: str
    workflow: Workflow
    module: ModuleType
    llm: str | None
    llm_idle_timeout: float | None
    store_path: str | None
    inputs: dict[str, object]
    options: dict[str, object]
    ui: bool
    timeout: float
    execution: str
    show_decisions: bool

    def option(self, name: str, default: object = None) -> object:
        return self.options.get(name, default)


def _import_module_path(module_path: str) -> ModuleType:
    path = Path(module_path).expanduser()
    spec = importlib.util.spec_from_file_location(
        f"_zippergen_wf_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    return module


def _looks_like_path(name: str) -> bool:
    return name.endswith(".py") or "/" in name or "\\" in name or Path(name).exists()


def _import_workflow_module(module_ref: str) -> ModuleType:
    if _looks_like_path(module_ref):
        return _import_module_path(module_ref)
    return importlib.import_module(module_ref)


def load_workflow_spec(spec_text: str) -> tuple[Workflow, ModuleType]:
    """Load ``module:workflow`` or ``path.py:workflow``.

    If no workflow name is supplied, the module must define exactly one
    ``Workflow`` object.
    """

    module_ref, sep, workflow_name = spec_text.partition(":")
    if not module_ref:
        raise SystemExit("Workflow spec must be MODULE:WORKFLOW or PATH.py:WORKFLOW.")
    module = _import_workflow_module(module_ref)
    if sep:
        try:
            value = getattr(module, workflow_name)
        except AttributeError as exc:
            raise SystemExit(f"Workflow {workflow_name!r} not found in {module_ref!r}.") from exc
        if not isinstance(value, Workflow):
            raise SystemExit(f"{module_ref}:{workflow_name} is not a ZipperGen Workflow.")
        return value, module

    workflows = {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, Workflow)
    }
    if len(workflows) == 1:
        return next(iter(workflows.values())), module
    if not workflows:
        raise SystemExit(f"No ZipperGen Workflow found in {module_ref!r}.")
    names = ", ".join(sorted(workflows))
    raise SystemExit(f"Multiple workflows found in {module_ref!r}: {names}. Use MODULE:WORKFLOW.")


def load_workflow(module_path: str, role_name: str) -> tuple[Workflow, Lifeline]:
    module = _import_module_path(module_path)
    workflows = [value for value in vars(module).values() if isinstance(value, Workflow)]
    if not workflows:
        raise SystemExit(f"No ZipperGen Workflow found in {module_path!r}.")
    wf = workflows[0]
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
        if not k or not _:
            raise SystemExit(f"Invalid --input {p!r}; expected name=value.")
        try:
            out[k] = json.loads(v)      # 7 -> int, "true" via JSON, '"s"' -> str
        except json.JSONDecodeError:
            out[k] = v                  # bare string fallback
    return out


def _parse_input_json(text: str | None) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--input-json must be valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SystemExit("--input-json must be a JSON object.")
    return value


def _parse_options(pairs: list[str], *, services: str | None = None) -> dict:
    options = _parse_inputs(pairs)
    if services is not None:
        existing = options.get("services")
        if existing is not None and existing != services:
            raise SystemExit("Use either --services or --option services=..., not both.")
        options["services"] = services
    return options


def _seed_inputs(wf: Workflow, inputs: dict) -> dict:
    """Var defaults from the workflow namespace, overlaid by caller inputs —
    parity with the in-process run() seeding (runtime.py:1014)."""
    from zippergen.syntax import Var
    env = {k: v.default for k, v in wf.ns.items() if isinstance(v, Var)}
    env.update(inputs)
    return env


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-._")
    return text or "workflow"


def _default_store_path(workflow_spec: str, wf: Workflow) -> str:
    base = workflow_spec.split(":", 1)[0]
    if _looks_like_path(base):
        label = f"{Path(base).stem}.{wf.name}"
    else:
        label = f"{base}.{wf.name}"
    return str(Path.home() / ".zippergen" / "runs" / f"{_slug(label)}.sqlite")


def _ensure_store_parent(path: str) -> str:
    expanded = Path(path).expanduser()
    expanded.parent.mkdir(parents=True, exist_ok=True)
    return str(expanded)


def _call_setup_hook(module: ModuleType, config: RunConfig) -> None:
    setup = getattr(module, "zippergen_setup", None)
    if setup is None:
        return
    if not callable(setup):
        raise SystemExit("zippergen_setup exists but is not callable.")
    setup(config)


def _safe_json_loads(value):
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _fmt_time(ts: float | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _store_status(store_path: str) -> dict[str, object]:
    path = Path(store_path).expanduser()
    if not path.exists():
        return {
            "store": str(path),
            "exists": False,
            "state": "missing",
            "summary": "store does not exist",
        }

    conn = open_store(str(path))
    try:
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        last_event = conn.execute(
            "SELECT rowid, sender, receiver, channel, kind, payload "
            "FROM events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        human_rows = conn.execute(
            "SELECT task_id, role, action, status, created_at, updated_at "
            "FROM human_tasks ORDER BY updated_at DESC"
        ).fetchall()
        pending_tasks = [
            {
                "task_id": row[0],
                "role": row[1],
                "action": row[2],
                "created_at": row[4],
                "updated_at": row[5],
            }
            for row in human_rows
            if row[3] == "pending"
        ]
        done_task_count = sum(1 for row in human_rows if row[3] == "done")
        results = list_workflow_results(conn)
    finally:
        conn.close()

    if pending_tasks:
        state = "waiting"
        summary = f"waiting for {len(pending_tasks)} human task(s)"
    elif results:
        state = "done"
        summary = f"{len(results)} workflow result(s)"
    elif event_count:
        state = "active"
        summary = "events recorded; no result yet"
    else:
        state = "empty"
        summary = "store is initialized but empty"

    last_event_dict = None
    if last_event is not None:
        last_event_dict = {
            "rowid": last_event[0],
            "sender": last_event[1],
            "receiver": last_event[2],
            "channel": last_event[3],
            "kind": last_event[4],
            "payload": _safe_json_loads(last_event[5]),
        }

    return {
        "store": str(path),
        "exists": True,
        "state": state,
        "summary": summary,
        "event_count": event_count,
        "last_event": last_event_dict,
        "pending_human_tasks": pending_tasks,
        "done_human_task_count": done_task_count,
        "workflow_results": results,
    }


def _print_status(status: dict[str, object]) -> None:
    print(f"Store: {status['store']}")
    print(f"State: {status['state']} ({status['summary']})")
    if not status.get("exists"):
        return

    print(f"Events: {status['event_count']}")
    last_event = status.get("last_event")
    if isinstance(last_event, dict):
        sender = last_event.get("sender")
        receiver = last_event.get("receiver") or "-"
        kind = last_event.get("kind")
        rowid = last_event.get("rowid")
        print(f"Last event: #{rowid} {kind} {sender}->{receiver}")

    tasks = status.get("pending_human_tasks")
    if isinstance(tasks, list):
        print(f"Pending human tasks: {len(tasks)}")
        for task in tasks[:10]:
            print(
                f"  {task['task_id']} {task['role']}.{task['action']} "
                f"updated {_fmt_time(task['updated_at'])}"
            )

    results = status.get("workflow_results")
    if isinstance(results, list):
        print(f"Workflow results: {len(results)}")
        for result in results[:10]:
            print(
                f"  {result['workflow']} = {json.dumps(result['value'], default=str)} "
                f"updated {_fmt_time(result['updated_at'])}"
            )


def _run_workflow_command(args) -> int:
    wf, module = load_workflow_spec(args.workflow)
    inputs = _parse_input_json(args.input_json)
    inputs.update(_parse_inputs(args.input))
    options = _parse_options(args.option, services=args.services)

    store_path = args.store
    if args.execution == "sqlite":
        store_path = _ensure_store_parent(store_path or _default_store_path(args.workflow, wf))
        print(f"Store: {store_path}", file=sys.stderr)
    elif store_path:
        print("--store is ignored when --execution memory is used.", file=sys.stderr)

    config = RunConfig(
        workflow_spec=args.workflow,
        workflow=wf,
        module=module,
        llm=args.llm,
        llm_idle_timeout=args.llm_idle_timeout,
        store_path=store_path,
        inputs=inputs,
        options=options,
        ui=args.ui,
        timeout=args.timeout,
        execution=args.execution,
        show_decisions=args.show_decisions,
    )
    _call_setup_hook(module, config)

    configure_kwargs = {
        "ui": args.ui,
        "timeout": args.timeout,
        "llm_idle_timeout": args.llm_idle_timeout,
        "execution": args.execution,
        "store_path": store_path,
        "show_decisions": args.show_decisions,
    }
    if args.llm:
        wf.configure(args.llm, **configure_kwargs)
    else:
        wf.configure(**configure_kwargs)

    result = wf(**inputs)
    print(json.dumps({"result": result}, default=str))
    if args.ui and sys.stdin.isatty():
        input("ZipperChat running at http://localhost:8765. Press Enter to exit. ")
    return 0


def _status_command(args) -> int:
    status = _store_status(args.store)
    if args.json:
        print(json.dumps(status, default=str))
    else:
        _print_status(status)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zippergen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run", help="run a workflow locally through SQLite")
    rn.add_argument("workflow", help="Workflow spec: module:workflow or path.py:workflow")
    rn.add_argument("--llm", metavar="SPEC", help="LLM spec: mock, openai:gpt-4o, ollama:qwen2.5:7b, ...")
    rn.add_argument("--llm-idle-timeout", type=float, help="Release a managed local LLM after this many idle seconds.")
    rn.add_argument("--store", help="SQLite store path. Defaults to ~/.zippergen/runs/<workflow>.sqlite")
    rn.add_argument("--input", action="append", default=[], metavar="name=value", help="Workflow input value.")
    rn.add_argument("--input-json", help="Workflow inputs as a JSON object.")
    rn.add_argument("--option", action="append", default=[], metavar="name=value", help="Option passed to zippergen_setup(config).")
    rn.add_argument("--services", choices=("fake", "live"), help="Shortcut for --option services=<value>.")
    rn.add_argument("--ui", action="store_true", help="Start ZipperChat and attach it to the run store.")
    rn.add_argument("--timeout", type=float, default=60.0, help="Workflow timeout in seconds.")
    rn.add_argument("--execution", choices=("sqlite", "memory"), default="sqlite", help="Execution backend.")
    rn.add_argument("--show-decisions", action="store_true", help="Show branch/control events in ZipperChat.")

    st = sub.add_parser("status", help="show local SQLite deployment status")
    st.add_argument("--store", required=True, help="SQLite store path.")
    st.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    sv = sub.add_parser("serve", help="run one role as a durable process")
    sv.add_argument("--workflow", required=True)
    sv.add_argument("--role", required=True)
    sv.add_argument("--store", required=True)
    sv.add_argument("--input", action="append", default=[], metavar="k=v")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        return _run_workflow_command(args)
    if args.cmd == "status":
        return _status_command(args)

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
