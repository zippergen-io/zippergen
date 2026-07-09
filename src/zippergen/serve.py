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
from zippergen.store import (
    complete_human_task,
    ensure_human_task_token,
    list_workflow_results,
    load_human_task,
    load_human_task_token,
    mark_human_task_token_used,
    open_store,
)


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


def _load_trace_events(
    store_path: str,
    *,
    after_rowid: int = 0,
    limit: int = 50,
) -> list[dict]:
    if limit <= 0:
        raise SystemExit("--tail must be greater than 0.")
    path = Path(store_path).expanduser()
    if not path.exists():
        raise SystemExit(f"Store does not exist: {store_path}")
    conn = open_store(str(path))
    try:
        rows = conn.execute(
            "SELECT rowid, sender, payload FROM events "
            "WHERE kind='trace' AND rowid>? "
            "ORDER BY rowid DESC LIMIT ?",
            (after_rowid, limit),
        ).fetchall()
    finally:
        conn.close()
    rows = list(reversed(rows))
    return [
        {
            "rowid": row[0],
            "role": row[1],
            "event": _safe_json_loads(row[2]),
        }
        for row in rows
    ]


def _load_human_tasks(
    store_path: str,
    *,
    status: str | None = "pending",
    limit: int | None = None,
    with_tokens: bool = False,
    token_channel: str = "cli",
) -> list[dict]:
    conn = open_store(str(Path(store_path).expanduser()))
    try:
        query = (
            "SELECT task_id FROM human_tasks "
            + ("WHERE status=? " if status is not None else "")
            + "ORDER BY updated_at DESC, task_id"
        )
        params: tuple[object, ...] = (status,) if status is not None else ()
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        rows = conn.execute(query, params).fetchall()
        tasks = []
        for row in rows:
            task = load_human_task(conn, row[0])
            if task is not None:
                if with_tokens:
                    record = ensure_human_task_token(conn, task["task_id"], channel=token_channel)
                    task["token"] = record["token"]
                    task["token_channel"] = record["channel"]
                tasks.append(task)
        return tasks
    finally:
        conn.close()


def _short_text(value: object, *, limit: int = 120) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _short_json(value: object, *, limit: int = 160) -> str:
    text = json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _trace_summary(role: str, event: object) -> str:
    if not isinstance(event, dict):
        return f"{role} {_short_json(event)}"

    event_type = event.get("type", "event")
    if event_type == "send":
        source = event.get("from", role)
        target = event.get("to", "?")
        channel = event.get("channel") or "-"
        return f"{role} send {source}->{target} {channel} values={_short_json(event.get('values') or [])}"
    if event_type == "recv":
        source = event.get("from", "?")
        target = event.get("to", role)
        channel = event.get("channel") or "-"
        return f"{role} recv {source}->{target} {channel} bindings={_short_json(event.get('bindings') or {})}"
    if event_type in {"act_start", "act"}:
        action = event.get("action", "?")
        kind = event.get("action_kind") or "action"
        payload_name = "outputs" if event_type == "act" else "inputs"
        payload = event.get(payload_name) or {}
        seq = event.get("seq")
        seq_text = f" seq={seq}" if seq is not None else ""
        return f"{role} {event_type} {kind} {action}{seq_text} {payload_name}={_short_json(payload)}"
    if event_type == "decision":
        kind = event.get("kind", "if")
        return f"{role} decision {kind} value={_short_json(event.get('value'))}"
    return f"{role} {event_type} {_short_json(event)}"


def _print_trace_events(events: list[dict]) -> None:
    print(f"Trace events: {len(events)}")
    for item in events:
        print(f"#{item['rowid']} {_trace_summary(item.get('role') or '-', item.get('event'))}")


def _print_tasks(tasks: list[dict], *, heading: str) -> None:
    print(f"{heading}: {len(tasks)}")
    for task in tasks:
        spec = task.get("spec") or {}
        rendered = spec.get("rendered") or {}
        output = spec.get("output")
        output_type = spec.get("output_type")
        print(
            f"{task['task_id']} {task['role']}.{task['action']} "
            f"{spec.get('kind', 'human')} -> {output}: {output_type} "
            f"status={task['status']} updated={_fmt_time(task['updated_at'])}"
        )
        if task.get("token"):
            print(f"  token[{task.get('token_channel', 'default')}]: {task['token']}")
        instruction = rendered.get("instruction")
        context = rendered.get("context")
        prefill = rendered.get("prefill")
        if instruction:
            print(f"  instruction: {_short_text(instruction)}")
        if context:
            print(f"  context: {_short_text(context)}")
        if prefill:
            print(f"  prefill: {_short_text(prefill)}")


def _notify_stdout_task(task: dict, *, store_path: str) -> None:
    spec = task.get("spec") or {}
    rendered = spec.get("rendered") or {}
    token = task.get("token")
    print("=" * 72)
    print(f"Human task: {task['task_id']}")
    if token:
        print(f"Token: {token}")
    print(f"Action: {task['role']}.{task['action']} ({spec.get('kind', 'human')})")
    instruction = rendered.get("instruction")
    context = rendered.get("context")
    prefill = rendered.get("prefill")
    if instruction:
        print("\nInstruction:")
        print(instruction)
    if context:
        print("\nContext:")
        print(context)
    if prefill:
        print("\nPrefill:")
        print(prefill)
    if token:
        print("\nApprove:")
        print(f"  zippergen approve --store {store_path} --token {token}")
        if spec.get("output_type") == "bool":
            print("Decline:")
            print(f"  zippergen approve --store {store_path} --token {token} --no")
        else:
            print("Respond:")
            print(f"  zippergen approve --store {store_path} --token {token} --value '<value>'")


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


def _trace_command(args) -> int:
    events = _load_trace_events(args.store, after_rowid=args.after, limit=args.tail)
    if args.json:
        print(json.dumps(events, default=str))
    else:
        _print_trace_events(events)
    return 0


def _tasks_command(args) -> int:
    if not Path(args.store).expanduser().exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    status = None if args.all else "pending"
    tasks = _load_human_tasks(
        args.store,
        status=status,
        limit=args.limit,
        with_tokens=args.tokens,
        token_channel=args.channel,
    )
    if args.json:
        print(json.dumps(tasks, default=str))
    else:
        _print_tasks(tasks, heading="Human tasks" if args.all else "Pending human tasks")
    return 0


def _parse_bool_value(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"true", "yes", "1", "y", "approve", "approved", "ack"}:
        return True
    if text in {"false", "no", "0", "n", "decline", "declined", "reject", "rejected"}:
        return False
    raise SystemExit(f"Cannot parse boolean human response: {raw!r}")


def _approve_result_from_args(task: dict, args) -> dict:
    spec = task.get("spec") or {}
    output = spec.get("output")
    if not output:
        raise SystemExit(f"Task {task['task_id']} has no output field in its spec.")
    output_type = spec.get("output_type", "str")

    if args.result_json is not None:
        try:
            result = json.loads(args.result_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--result-json must be valid JSON: {exc.msg}") from exc
        if not isinstance(result, dict):
            raise SystemExit("--result-json must be a JSON object.")
        if output not in result:
            raise SystemExit(f"--result-json must include output key {output!r}.")
        result[output] = _parse_bool_value(result[output]) if output_type == "bool" else str(result[output])
        return result

    if args.yes and args.no:
        raise SystemExit("Use only one of --yes or --no.")
    if args.value is not None and (args.yes or args.no):
        raise SystemExit("Use either --value or --yes/--no, not both.")

    if output_type == "bool":
        if args.no:
            value = False
        elif args.value is not None:
            value = _parse_bool_value(args.value)
        else:
            value = True
    else:
        if args.yes or args.no:
            raise SystemExit("--yes/--no can only be used for boolean human tasks.")
        if args.value is None:
            raise SystemExit(f"Task {task['task_id']} requires --value for output {output!r}.")
        value = args.value
    return {output: value}


def _approve_command(args) -> int:
    store_path = str(Path(args.store).expanduser())
    if not Path(store_path).exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    conn = open_store(store_path)
    try:
        token_record = None
        task_id = args.task
        if args.token is not None:
            token_record = load_human_task_token(conn, args.token)
            if token_record is None:
                raise SystemExit(f"Human task token not found: {args.token}")
            task_id = token_record["task_id"]
        task = load_human_task(conn, task_id)
        if task is None:
            raise SystemExit(f"Human task not found: {task_id}")
        if task["status"] != "pending":
            raise SystemExit(f"Human task {task_id} is already {task['status']}.")
        result = _approve_result_from_args(task, args)
        conn.execute("BEGIN IMMEDIATE")
        try:
            task = complete_human_task(conn, task_id, result)
            if token_record is not None:
                mark_human_task_token_used(conn, token_record["token"])
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()

    if args.json:
        print(json.dumps(task, default=str))
    else:
        print(f"Completed human task {task['task_id']}: {json.dumps(task['result'], default=str)}")
    return 0


def _notify_stdout_command(args) -> int:
    store_path = str(Path(args.store).expanduser())
    if not Path(store_path).exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    seen: set[str] = set()
    while True:
        tasks = _load_human_tasks(
            store_path,
            status="pending",
            limit=args.limit,
            with_tokens=True,
            token_channel=args.channel,
        )
        emitted = 0
        for task in tasks:
            token = task.get("token") or task["task_id"]
            if token in seen:
                continue
            _notify_stdout_task(task, store_path=store_path)
            seen.add(token)
            emitted += 1
        if not args.watch:
            if emitted == 0 and not args.quiet:
                print("No pending human tasks.")
            return 0
        time.sleep(args.interval)


def _notify_telegram_command(args) -> int:
    from zippergen.telegram_notify import (
        TelegramBotClient,
        TelegramNotifier,
        load_telegram_chat_id,
        load_telegram_token,
    )

    store_path = str(Path(args.store).expanduser())
    if not Path(store_path).exists():
        raise SystemExit(f"Store does not exist: {args.store}")
    token = load_telegram_token(args.bot_token)
    chat_id = load_telegram_chat_id(args.chat_id)
    if not chat_id:
        raise SystemExit("Telegram chat id is required. Set ZIPPERGEN_TELEGRAM_CHAT_ID or pass --chat-id.")
    client = TelegramBotClient(token)
    notifier = TelegramNotifier(
        store_path=store_path,
        client=client,
        chat_id=chat_id,
        channel=args.channel,
        limit=args.limit,
    )

    if not args.watch:
        sent = notifier.send_pending_once(resend=args.resend)
        processed = notifier.poll_updates_once(timeout=0)
        if not args.quiet:
            print(f"Telegram: sent {sent} task notification(s), processed {processed} update(s).")
        return 0

    if not args.quiet:
        print(f"Watching Telegram chat {chat_id} for store {store_path}.")
    while True:
        sent = notifier.send_pending_once(resend=args.resend)
        processed = notifier.poll_updates_once(timeout=args.poll_timeout)
        if not args.quiet and (sent or processed):
            print(f"Telegram: sent {sent} task notification(s), processed {processed} update(s).")
        time.sleep(args.interval)


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
    rn.add_argument("--timeout", type=float, default=60.0, help="Workflow timeout in seconds; use 0 for no deadline.")
    rn.add_argument("--execution", choices=("sqlite", "memory"), default="sqlite", help="Execution backend.")
    rn.add_argument("--show-decisions", action="store_true", help="Show branch/control events in ZipperChat.")

    st = sub.add_parser("status", help="show local SQLite deployment status")
    st.add_argument("--store", required=True, help="SQLite store path.")
    st.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    tr = sub.add_parser("trace", help="show recent trace events from a local SQLite store")
    tr.add_argument("--store", required=True, help="SQLite store path.")
    tr.add_argument("--tail", type=int, default=50, help="Maximum number of trace events to show.")
    tr.add_argument("--after", type=int, default=0, help="Only show trace events after this event rowid.")
    tr.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    tk = sub.add_parser("tasks", help="list human tasks in a local SQLite store")
    tk.add_argument("--store", required=True, help="SQLite store path.")
    tk.add_argument("--all", action="store_true", help="Include completed tasks.")
    tk.add_argument("--limit", type=int, help="Maximum number of tasks to show.")
    tk.add_argument("--tokens", action="store_true", help="Generate/show durable approval tokens.")
    tk.add_argument("--channel", default="cli", help="Token channel name used with --tokens.")
    tk.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    apv = sub.add_parser("approve", help="complete a pending human task")
    apv.add_argument("--store", required=True, help="SQLite store path.")
    target = apv.add_mutually_exclusive_group(required=True)
    target.add_argument("--task", help="Human task id.")
    target.add_argument("--token", help="Durable approval token.")
    apv.add_argument("--yes", action="store_true", help="Complete a boolean task with true.")
    apv.add_argument("--no", action="store_true", help="Complete a boolean task with false.")
    apv.add_argument("--value", help="Value for string tasks, or explicit true/false for boolean tasks.")
    apv.add_argument("--result-json", help="Complete with an explicit JSON object result.")
    apv.add_argument("--json", action="store_true", help="Print the completed task as JSON.")

    nt = sub.add_parser("notify", help="run a notification adapter")
    notify_sub = nt.add_subparsers(dest="adapter", required=True)
    out = notify_sub.add_parser("stdout", help="print pending human tasks with approval tokens")
    out.add_argument("--store", required=True, help="SQLite store path.")
    out.add_argument("--channel", default="stdout", help="Approval token channel name.")
    out.add_argument("--watch", action="store_true", help="Keep polling for new pending tasks.")
    out.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds for --watch.")
    out.add_argument("--limit", type=int, help="Maximum number of tasks to notify per poll.")
    out.add_argument("--quiet", action="store_true", help="Suppress the no-pending-tasks message in one-shot mode.")

    tg = notify_sub.add_parser("telegram", help="send and receive human task approvals through Telegram")
    tg.add_argument("--store", required=True, help="SQLite store path.")
    tg.add_argument("--bot-token", help="Telegram bot token. Defaults to ZIPPERGEN_TELEGRAM_TOKEN.")
    tg.add_argument("--chat-id", help="Telegram chat id. Defaults to ZIPPERGEN_TELEGRAM_CHAT_ID.")
    tg.add_argument("--channel", default="telegram", help="Approval token channel name.")
    tg.add_argument("--watch", action="store_true", help="Keep polling Telegram and the local store.")
    tg.add_argument("--interval", type=float, default=2.0, help="Delay between store scans in --watch mode.")
    tg.add_argument("--poll-timeout", type=float, default=20.0, help="Telegram long-poll timeout in seconds.")
    tg.add_argument("--limit", type=int, help="Maximum number of tasks to notify per poll.")
    tg.add_argument("--resend", action="store_true", help="Resend already-notified pending tasks.")
    tg.add_argument("--quiet", action="store_true", help="Suppress progress messages.")

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
    if args.cmd == "trace":
        return _trace_command(args)
    if args.cmd == "tasks":
        return _tasks_command(args)
    if args.cmd == "approve":
        return _approve_command(args)
    if args.cmd == "notify" and args.adapter == "stdout":
        return _notify_stdout_command(args)
    if args.cmd == "notify" and args.adapter == "telegram":
        return _notify_telegram_command(args)

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
