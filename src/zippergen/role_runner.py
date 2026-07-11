"""Reusable durable role runner.

This module owns the per-role replay/live loop used by the local SQLite
supervisor. `zippergen serve` still exposes the same machinery as a legacy
low-level entry point.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

from zippergen.locator import action_node_paths, loop_node_paths, resolve_path
from zippergen.runtime import (
    PendingExternal,
    _input_hash,
    _step,
    external_out_map,
    mock_llm,
)
from zippergen.store import DurableChannel, load_snapshot, record_trace_event, write_snapshot
from zippergen.store import (
    complete_human_task,
    ensure_human_task,
    human_task_id,
    load_human_task,
)
from zippergen.syntax import ActStmt, EmptyStmt, HumanAction, WhileRecvStmt, WhileStmt


@dataclass
class JournalContext:
    channel: object          # DurableChannel
    act_paths: dict          # id(node) -> child-index path


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

        self.conn = conn
        self.role = role
        self.local_stmt = local_stmt
        self.ns = ns
        self.llm_backend = llm_backend
        self.human_backend = human_backend
        self.trace = trace
        self.monitor = monitor
        self.formula_conditions = formula_conditions or {}
        self.stop = stop

        self.loop_paths = loop_node_paths(local_stmt)
        if self.monitor is None:
            self.env, self.residual, since = _try_resume(conn, role, local_stmt, env)
        else:
            # Snapshots do not persist monitor state yet. Monitored roles are
            # still correct by full replay; resume snapshots once monitor
            # snapshots are part of the store format.
            self.env, self.residual, since = env, local_stmt, None
        self.channel = DurableChannel(conn, role, since=since)
        self.journal = JournalContext(self.channel, action_node_paths(local_stmt))
        self.trace = self._make_trace(trace)
        self._idle_sleep = self._IDLE_SLEEP_INITIAL

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
        if created and task["status"] == "pending" and not sqlite_owned:
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
                out_map, task_id = self._resolve_external(out)   # OUTSIDE any txn
                self._record_act_outputs(out, out_map, human_task=task_id)
                # pass 2 (no txn): consume the just-committed act row, apply env, advance
                self.residual, resolved = self.step(self.residual, self.trace)
                assert resolved, "durable resolve failed to consume the just-committed act row"
                if self.monitor is None and id(self.residual) in self.loop_paths:
                    _maybe_snapshot(
                        self.conn,
                        self.role,
                        self.env,
                        self.loop_paths[id(self.residual)],
                        self.channel,
                    )
                self._reset_idle_backoff()
                continue
            if progressed:
                self.channel.commit_txn()
                self.residual = out
                self._reset_idle_backoff()
                # At a loop-iteration boundary the residual is (by identity) a loop
                # node in the projected program; checkpoint env + position there.
                if self.monitor is None and id(self.residual) in self.loop_paths:
                    _maybe_snapshot(
                        self.conn,
                        self.role,
                        self.env,
                        self.loop_paths[id(self.residual)],
                        self.channel,
                    )
            else:
                self.channel.rollback_txn()
                if self.stop is not None and self.stop.is_set():
                    raise RuntimeError("Workflow cancelled")
                self._sleep_after_idle_step()

    def run(self) -> dict:
        self.replay_committed()
        self.run_live()
        return self.env


def run_role(conn, role: str, local_stmt, env: dict, ns: dict, *,
             llm_backend=None, human_backend=None, trace=None,
             monitor=None, formula_conditions: dict | None = None) -> dict:
    return RoleRunner(
        conn,
        role,
        local_stmt,
        env,
        ns,
        llm_backend=llm_backend,
        human_backend=human_backend,
        trace=trace,
        monitor=monitor,
        formula_conditions=formula_conditions,
    ).run()
