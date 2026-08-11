"""Reusable durable role runner and its replay bookkeeping."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

from zippergen.locator import (
    action_node_paths,
    execution_frontier_paths,
    loop_node_paths,
    resolve_path,
)
from zippergen.runtime import (
    PendingExternal,
    _input_hash,
    _step,
    external_out_map,
    mock_llm,
)
from zippergen.store import (
    DurableChannel,
    load_snapshot,
    record_trace_event,
    recovery_high_water,
    write_execution_state,
    write_snapshot,
)
from zippergen.store import (
    complete_human_task,
    ensure_human_task,
    human_task_id,
    load_human_task,
)
from zippergen.syntax import (
    ActStmt,
    AssistantAction,
    EffectAction,
    EmptyStmt,
    HumanAction,
    IfRecvStmt,
    LLMAction,
    PlannerAction,
    ReceiveAnyStmt,
    RecvStmt,
    WhileRecvStmt,
    WhileStmt,
)


@dataclass
class JournalContext:
    channel: object          # DurableChannel
    act_paths: dict          # id(node) -> child-index path


def seed_env(conn, role: str, inputs: dict) -> dict:
    """Record initial role inputs once, then return that durable seed."""

    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT payload FROM events WHERE kind='seed' AND sender=? "
            "ORDER BY rowid LIMIT 1",
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


def _floor_coherent(conn, role: str, floor: dict) -> bool:
    """A floor is coherent only if it carries all three keys and none points past
    the committed log. A floor lacking "journal" predates journaling -> incoherent
    (forces full replay from seed)."""
    try:
        out = floor["out"]; cursors = floor["cursors"]; journal = floor["journal"]
    except (KeyError, TypeError):
        return False
    recorded = recovery_high_water(conn, role)
    max_out = conn.execute(
        "SELECT MAX(rowid) FROM events WHERE sender=? AND kind IN ('msg','ctrl')",
        (role,),
    ).fetchone()[0] or 0
    max_out = max(int(max_out), recorded["out"])
    if out > max_out:
        return False
    max_journal = conn.execute(
        "SELECT MAX(rowid) FROM events "
        "WHERE sender=? AND kind IN ('act','decision','effect')",
        (role,),
    ).fetchone()[0] or 0
    max_journal = max(int(max_journal), recorded["journal"])
    if journal > max_journal:
        return False
    durable = {ck: c for ck, c in conn.execute(
        "SELECT chan_key, consumed FROM cursors WHERE role=?", (role,)).fetchall()}
    return all(v <= durable.get(k, 0) for k, v in cursors.items())


def _try_resume(conn, role: str, local_stmt, env: dict, monitor=None):
    """Return (env, residual, since) from a valid snapshot, else (env, local_stmt, None)."""
    snap = load_snapshot(conn, role)
    if snap is None:
        return env, local_stmt, None
    node = resolve_path(local_stmt, snap["locator"])
    if (
        isinstance(node, (WhileStmt, WhileRecvStmt))
        and _floor_coherent(conn, role, snap["floor"])
    ):
        if monitor is not None:
            monitor_state = snap.get("monitor")
            if not isinstance(monitor_state, dict):
                return env, local_stmt, None
            try:
                monitor.restore_state(monitor_state)
            except (TypeError, ValueError):
                return env, local_stmt, None
        return dict(snap["env"]), node, snap["floor"]
    return env, local_stmt, None   # stale/invalid -> full replay from seed


def _maybe_snapshot(
    conn,
    role: str,
    env: dict,
    locator: list,
    ch,
    monitor=None,
) -> None:
    try:
        write_snapshot(
            conn,
            role,
            env,
            locator,
            ch.position(),
            monitor.snapshot_state() if monitor is not None else None,
        )
    except (TypeError, ValueError, sqlite3.OperationalError):
        # Best-effort: a snapshot is a rebuildable cache and must never fail a
        # healthy role — skip on a non-serializable env OR a transient store
        # error (e.g. lock contention in the separate snapshot transaction).
        pass


def _open_observation_connection(conn) -> sqlite3.Connection | None:
    """Open an isolated, short-timeout connection for best-effort telemetry."""

    observer: sqlite3.Connection | None = None
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
        path = next(
            (
                str(row[2])
                for row in rows
                if len(row) >= 3 and row[1] == "main" and row[2]
            ),
            "",
        )
        if not path or path == ":memory:":
            return None
        observer = sqlite3.connect(
            path,
            isolation_level=None,
            check_same_thread=False,
            timeout=0.05,
        )
        observer.execute("PRAGMA busy_timeout=50")
        return observer
    except Exception:
        if observer is not None:
            try:
                observer.close()
            except Exception:
                pass
        return None


def _begin_immediate(conn, stop: threading.Event | None = None) -> None:
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
            if stop is not None and stop.is_set():
                raise RuntimeError("Workflow cancelled") from exc
            time.sleep(0.05)


def _render_template(template: str | None, inputs: dict) -> str | None:
    return template.format(**inputs) if template else None


def _human_task_spec(action: HumanAction, inputs: dict) -> dict:
    return {
        "name": action.name,
        "kind": action.kind,
        "output": action.output,
        "output_type": action.output_type.__name__,
        "context": action.context,
        "instruction": action.instruction,
        "prefill": action.prefill,
        "submit_label": action.submit_label,
        "cancel_label": action.cancel_label,
        "rendered": {
            "context": _render_template(action.context, inputs),
            "instruction": _render_template(action.instruction, inputs),
            "prefill": _render_template(action.prefill, inputs),
        },
    }


class RoleRunner:
    """Run one projected local program against the durable SQLite store."""

    _IDLE_SLEEP_INITIAL = 0.02
    _IDLE_SLEEP_MAX = 1.0
    _IDLE_SLEEP_FACTOR = 2.0

    def __init__(
        self,
        conn,
        role: str,
        local_stmt,
        env: dict,
        ns: dict,
        *,
        llm_backend=None,
        human_backend=None,
        assistant_backend=None,
        trace=None,
        monitor=None,
        formula_conditions: dict | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        if llm_backend is None:
            llm_backend = mock_llm
        if human_backend is None:
            from zippergen.human_backends import make_cli_human_backend
            human_backend = make_cli_human_backend()
        if assistant_backend is None:
            from zippergen.assistant_backends import make_cli_assistant_backend
            assistant_backend = make_cli_assistant_backend()

        self.conn = conn
        self.role = role
        self.local_stmt = local_stmt
        self.ns = ns
        self.llm_backend = llm_backend
        self.human_backend = human_backend
        self.assistant_backend = assistant_backend
        self.trace = trace
        self.monitor = monitor
        self.formula_conditions = formula_conditions or {}
        self.stop = stop

        self.loop_paths = loop_node_paths(local_stmt)
        self.env, self.residual, since = _try_resume(
            conn,
            role,
            local_stmt,
            env,
            self.monitor,
        )
        self.channel = DurableChannel(conn, role, since=since)
        self.journal = JournalContext(self.channel, action_node_paths(local_stmt))
        self.trace = self._make_trace(trace)
        self._idle_sleep = self._IDLE_SLEEP_INITIAL
        self._execution_state_signature: str | None = None
        self._observation_conn = _open_observation_connection(conn)

    def _make_trace(self, trace):
        def durable_trace(event: dict) -> None:
            record_trace_event(self.conn, self.role, event)
            if trace is not None:
                trace(event)

        return durable_trace

    def _record_act_outputs(
        self,
        pending: PendingExternal,
        out_map: dict,
        *,
        human_task: str | None = None,
    ) -> None:
        node = cast(ActStmt, pending.node)
        loc = self.journal.act_paths[id(pending.node)]
        payload = {
            "status": "done",
            "locator": loc,
            "action": node.action.name,
            "input_hash": _input_hash(pending.inputs),
            "outputs": out_map,
        }
        if human_task is not None:
            payload["human_task"] = human_task
        _begin_immediate(self.conn, self.stop)
        try:
            self.channel.record_act(payload)
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def _set_execution_state(
        self,
        state: str,
        locators: list[list[int]],
        detail: dict | None = None,
    ) -> None:
        """Best-effort observation state; never make execution depend on it."""

        normalized_detail = detail or {}
        signature = repr((state, locators, sorted(normalized_detail.items())))
        if signature == self._execution_state_signature:
            return
        if self._observation_conn is None:
            return
        try:
            write_execution_state(
                self._observation_conn,
                self.role,
                state,
                locators,
                normalized_detail,
            )
        except Exception:
            return
        self._execution_state_signature = signature

    def _frontier(self) -> tuple[list[list[int]], list[object]]:
        locators = execution_frontier_paths(self.local_stmt, self.residual)
        nodes = [
            node
            for locator in locators
            if (node := resolve_path(self.local_stmt, locator)) is not None
        ]
        return locators, nodes

    def _set_residual_state(self, *, blocked: bool = False) -> None:
        locators, nodes = self._frontier()
        if not locators:
            self._set_execution_state("done", [])
            return
        receives = (RecvStmt, ReceiveAnyStmt, IfRecvStmt, WhileRecvStmt)
        if blocked and nodes and all(isinstance(node, receives) for node in nodes):
            state = "waiting_receive"
        elif blocked:
            state = "blocked"
        else:
            state = "running"
        self._set_execution_state(state, locators)

    def _set_external_state(self, pending: PendingExternal) -> None:
        node = cast(ActStmt, pending.node)
        locator = self.journal.act_paths.get(id(node))
        locators = [locator] if locator is not None else []
        action = node.action
        detail = {"action": action.name}
        if isinstance(action, HumanAction) and action.visible:
            state = "waiting_human"
            detail["kind"] = "human"
        elif isinstance(action, AssistantAction):
            state = "running_assistant"
            detail["kind"] = "assistant"
        elif isinstance(action, (LLMAction, PlannerAction)):
            state = "running_model"
            detail["kind"] = "model"
        elif isinstance(action, EffectAction):
            state = "running_effect"
            detail["kind"] = "effect"
        else:
            state = "running_action"
            detail["kind"] = type(action).__name__
        self._set_execution_state(state, locators, detail)

    def _wait_for_human_task(self, task_id: str) -> dict:
        while True:
            task = load_human_task(self.conn, task_id)
            if task is None:
                raise RuntimeError(f"Human task {task_id!r} disappeared")
            status = task["status"]
            if status == "done":
                return task
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"Human task {task_id!r} ended with status {status!r}")
            if self.stop is not None and self.stop.is_set():
                raise RuntimeError("Workflow cancelled")
            time.sleep(0.05)

    def _resolve_human_task(self, pending: PendingExternal) -> tuple[dict, str]:
        node = cast(ActStmt, pending.node)
        action = node.action
        assert isinstance(action, HumanAction)
        outs = node.outputs
        loc = self.journal.act_paths[id(node)]
        input_hash = _input_hash(pending.inputs)
        task_id = human_task_id(self.role, loc, input_hash, self.channel.journal_position())
        spec = _human_task_spec(action, pending.inputs)

        _begin_immediate(self.conn, self.stop)
        try:
            task, created = ensure_human_task(
                self.conn,
                task_id=task_id,
                role=self.role,
                locator=loc,
                action=action.name,
                input_hash=input_hash,
                inputs=pending.inputs,
                spec=spec,
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

        sqlite_owned = getattr(self.human_backend, "uses_sqlite_human_tasks", False)
        claims_pending = getattr(
            self.human_backend,
            "claims_pending_human_tasks",
            False,
        )
        should_prompt = created or claims_pending
        if should_prompt and task["status"] == "pending" and not sqlite_owned:
            named_outputs = self.human_backend(action, pending.inputs)
            result = {action.output: named_outputs[action.output]}
            _begin_immediate(self.conn, self.stop)
            try:
                task = complete_human_task(self.conn, task_id, result)
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

        task = task if task["status"] == "done" else self._wait_for_human_task(task_id)
        result = task["result"] or {}
        return {outs[0].name: result[action.output]}, task_id

    def _resolve_external(self, pending: PendingExternal) -> tuple[dict, str | None]:
        node = cast(ActStmt, pending.node)
        action = node.action
        if isinstance(action, HumanAction) and action.visible:
            return self._resolve_human_task(pending)
        outs = node.outputs
        return external_out_map(
            action,
            pending.inputs,
            outs,
            self.llm_backend,
            self.human_backend,
            self.assistant_backend,
        ), None

    def step(self, residual, trace):
        return _step(
            residual,
            self.env,
            cast(Any, self.channel),
            self.ns,
            self.llm_backend,
            self.human_backend,
            self.monitor,
            trace,
            self.formula_conditions,
            self.stop,
            journal=self.journal,
            assistant_backend=self.assistant_backend,
        )

    def replay_committed(self) -> None:
        # No transactions, no trace: put reserves recorded sends, try_get serves
        # recorded recvs. Determinism of local steps reproduces the exact boundary.
        while self.channel.replaying() and not isinstance(self.residual, EmptyStmt):
            out, progressed = self.step(self.residual, None)
            if isinstance(out, PendingExternal):
                break            # unreached during replay: committed acts consume, not pend
            if not progressed:
                break            # blocked on live input at the replay boundary
            self.residual = out

    def _reset_idle_backoff(self) -> None:
        self._idle_sleep = self._IDLE_SLEEP_INITIAL

    def _sleep_after_idle_step(self) -> None:
        delay = self._idle_sleep
        time.sleep(delay)
        self._idle_sleep = min(self._IDLE_SLEEP_MAX, delay * self._IDLE_SLEEP_FACTOR)

    def run_live(self) -> None:
        self._set_residual_state()
        while not isinstance(self.residual, EmptyStmt):
            if self.stop is not None and self.stop.is_set():
                raise RuntimeError("Workflow cancelled")
            # BEGIN IMMEDIATE (not deferred): a recv step reads (SELECT) before it
            # writes (cursor INSERT in commit_txn); a deferred BEGIN would make that
            # a read->write upgrade, which SQLite fails immediately with
            # "database is locked" (bypassing busy_timeout) when the peer role holds
            # the write lock. Taking the write lock up front serializes cleanly, the
            # same pattern seed_env uses.
            _begin_immediate(self.conn, self.stop)
            out, progressed = self.step(self.residual, self.trace)
            if isinstance(out, PendingExternal):
                self.conn.execute("ROLLBACK")                 # release the write lock first
                self._set_external_state(out)
                out_map, task_id = self._resolve_external(out)   # OUTSIDE any txn
                self._record_act_outputs(out, out_map, human_task=task_id)
                # pass 2 (no txn): consume the just-committed act row, apply env, advance
                self.residual, resolved = self.step(self.residual, self.trace)
                assert resolved, "durable resolve failed to consume the just-committed act row"
                if id(self.residual) in self.loop_paths:
                    _maybe_snapshot(
                        self.conn,
                        self.role,
                        self.env,
                        self.loop_paths[id(self.residual)],
                        self.channel,
                        self.monitor,
                    )
                self._set_residual_state()
                self._reset_idle_backoff()
                continue
            if progressed:
                self.channel.commit_txn()
                self.residual = out
                self._reset_idle_backoff()
                # At a loop-iteration boundary the residual is (by identity) a loop
                # node in the projected program; checkpoint env + position there.
                if id(self.residual) in self.loop_paths:
                    _maybe_snapshot(
                        self.conn,
                        self.role,
                        self.env,
                        self.loop_paths[id(self.residual)],
                        self.channel,
                        self.monitor,
                    )
                self._set_residual_state()
            else:
                self.channel.rollback_txn()
                self._set_residual_state(blocked=True)
                if self.stop is not None and self.stop.is_set():
                    raise RuntimeError("Workflow cancelled")
                self._sleep_after_idle_step()

    def run(self) -> dict:
        try:
            try:
                self.replay_committed()
                self.run_live()
            except BaseException as exc:
                if self.conn.in_transaction:
                    self.conn.execute("ROLLBACK")
                state = (
                    "cancelled"
                    if "Workflow cancelled" in str(exc)
                    else "failed"
                )
                self._set_execution_state(
                    state,
                    execution_frontier_paths(
                        self.local_stmt,
                        self.residual,
                    ),
                    {"error": type(exc).__name__},
                )
                raise
            self._set_execution_state("done", [])
            return self.env
        finally:
            if self._observation_conn is not None:
                try:
                    self._observation_conn.close()
                except Exception:
                    pass


def run_role(conn, role: str, local_stmt, env: dict, ns: dict, *,
             llm_backend=None, human_backend=None, assistant_backend=None, trace=None,
             monitor=None, formula_conditions: dict | None = None) -> dict:
    return RoleRunner(
        conn,
        role,
        local_stmt,
        env,
        ns,
        llm_backend=llm_backend,
        human_backend=human_backend,
        assistant_backend=assistant_backend,
        trace=trace,
        monitor=monitor,
        formula_conditions=formula_conditions,
    ).run()
