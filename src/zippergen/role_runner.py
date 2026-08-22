"""Run one projected local program against durable SQLite state.

The loop is the whole correctness story, so it is meant to be read:

    while there is work left:
        BEGIN IMMEDIATE
        take one interpreter step
        if it needs the outside world:
            ROLLBACK, call out with no transaction open,
            then BEGIN and commit (result + next control state) together
        elif it progressed:
            delete the messages it consumed, write the new role state, COMMIT
        else:
            ROLLBACK and wait

Every commit writes one role's whole durable position: variables, control
state, and monitor. Nothing else is needed to resume.

The crash rule:

    The committed role state describes what is known to have completed. The
    step the control state points at may run again after a crash. For an LLM
    call or an @effect that means it may be performed a second time, including
    its outside-world side effect.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any, cast

from zippergen.errors import WorkflowCancelled
from zippergen.control import (
    PartialReceiveAny,
    decode_control,
    encode_control,
    frontier_paths,
)
from zippergen.locator import resolve_path, statement_node_paths
from zippergen.human_tasks import (
    build_human_task_spec,
    validate_human_task_result,
)
from zippergen.runtime import (
    PendingExternal,
    _ResolvedExternal,
    _input_hash,
    _step,
    external_out_map,
    mock_llm,
)
from zippergen.store import (
    DurableChannel,
    complete_human_task,
    ensure_human_task,
    human_task_id,
    load_human_task,
    load_role_state,
    record_last_failure,
    record_history,
    RoleStateConflict,
    set_role_status,
    write_role_state,
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
)

__all__ = ["RoleRunner", "run_role"]


def _begin_immediate(conn, stop: threading.Event | None = None) -> None:
    """Take the write lock up front.

    A receive reads before it writes, and a deferred BEGIN would turn that into
    a read-to-write upgrade, which SQLite fails immediately with "database is
    locked" instead of routing through busy_timeout.
    """

    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
            if stop is not None and stop.is_set():
                raise WorkflowCancelled("Workflow cancelled") from exc
            time.sleep(0.05)


class RoleRunner:
    """Run one lifeline's local program, keeping its durable state current."""

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
        self.monitor = monitor
        self.formula_conditions = formula_conditions or {}
        self.stop = stop
        self.node_paths = statement_node_paths(local_stmt)

        self.env, self.residual = self._load_or_seed(env)
        self.channel = DurableChannel(conn, role)
        self.trace = self._make_trace(trace)
        self._idle_sleep = self._IDLE_SLEEP_INITIAL
        self._status_signature: object = None

    # ---- startup ----------------------------------------------------------
    def _load_or_seed(self, env: dict):
        """Resume from durable state, or write the starting state once."""

        state = load_role_state(self.conn, self.role)
        if state is not None:
            return self._adopt(state)
        seeded = dict(env)
        _begin_immediate(self.conn, self.stop)
        try:
            # Another process may have raced us; the first writer wins and we
            # read its state back rather than overwriting it.
            existing = load_role_state(self.conn, self.role)
            if existing is not None:
                self.conn.execute("ROLLBACK")
                return self._adopt(existing)
            self.steps = 0
            write_role_state(
                self.conn,
                self.role,
                env=seeded,
                control=encode_control(self.local_stmt, self.local_stmt),
                monitor=self._monitor_state(),
                steps=0,
                status="running",
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return seeded, self.local_stmt

    def _adopt(self, state: dict):
        if self.monitor is not None and state["monitor"] is not None:
            self.monitor.restore_state(state["monitor"])
        self.steps = state["steps"]
        return dict(state["env"]), decode_control(self.local_stmt, state["control"])

    def _monitor_state(self) -> dict | None:
        return None if self.monitor is None else self.monitor.snapshot_state()

    def _make_trace(self, trace):
        def durable_trace(event: dict) -> None:
            # Progress events share the current state transaction. External
            # starts and retries happen with no transaction open and therefore
            # autocommit. Recovery never reads either kind of history row.
            record_history(self.conn, self.role, event)
            if trace is not None:
                trace(event)

        return durable_trace

    # ---- durable state ----------------------------------------------------
    def _commit_state(self, residual, status: str, detail: dict | None = None) -> None:
        """Persist this role's whole position and end the open transaction."""

        self.channel.delete_taken()
        write_role_state(
            self.conn,
            self.role,
            env=self.env,
            control=encode_control(self.local_stmt, residual),
            monitor=self._monitor_state(),
            steps=self.steps + 1,
            status=status,
            detail=detail,
            expected_steps=self.steps,
        )
        self.conn.execute("COMMIT")
        self.steps += 1
        self.channel.clear_taken()
        self._status_signature = (status, tuple(sorted((detail or {}).items())))

    def _rollback(self) -> None:
        self.conn.execute("ROLLBACK")
        self.channel.clear_taken()

    def _publish_status(self, status: str, detail: dict | None = None) -> None:
        """Update the diagnostic status, but only when it actually changed.

        A blocked role polls in a loop. Writing the same status every time would
        take the write lock over and over for no new information, and would
        contend with the peer that is trying to make progress.
        """

        signature = (status, tuple(sorted((detail or {}).items())))
        if signature == self._status_signature:
            return
        set_role_status(
            self.conn,
            self.role,
            status,
            detail,
            expected_steps=self.steps,
        )
        self._status_signature = signature

    def _status_for(self, residual, *, blocked: bool) -> tuple[str, dict]:
        paths = frontier_paths(self.local_stmt, residual)
        if not paths:
            return "done", {}
        if not blocked:
            return "running", {}
        receives = (RecvStmt, ReceiveAnyStmt, IfRecvStmt, WhileRecvStmt)
        nodes = [
            node
            for path in paths
            if (node := resolve_path(self.local_stmt, path)) is not None
        ]
        waiting = bool(nodes) and all(
            isinstance(node, receives) for node in nodes
        )
        return ("waiting_receive" if waiting else "blocked"), {}

    # ---- external actions -------------------------------------------------
    def _external_status(self, pending: PendingExternal) -> tuple[str, dict]:
        node = cast(ActStmt, pending.node)
        action = node.action
        detail: dict = {"action": action.name}
        if isinstance(action, HumanAction) and action.visible:
            detail["kind"] = "human"
            return "waiting_human", detail
        if isinstance(action, AssistantAction):
            detail["kind"] = "assistant"
            return "running_assistant", detail
        if isinstance(action, (LLMAction, PlannerAction)):
            detail["kind"] = "model"
            return "running_model", detail
        if isinstance(action, EffectAction):
            detail["kind"] = "effect"
            return "running_effect", detail
        detail["kind"] = type(action).__name__
        return "running_action", detail

    def _wait_for_human_task(self, task_id: str) -> dict:
        while True:
            task = load_human_task(self.conn, task_id)
            if task is None:
                raise RuntimeError(f"Human task {task_id!r} disappeared")
            status = task["status"]
            if status == "done":
                return task
            if status in {"failed", "cancelled"}:
                raise RuntimeError(
                    f"Human task {task_id!r} ended with status {status!r}"
                )
            if self.stop is not None and self.stop.is_set():
                raise WorkflowCancelled("Workflow cancelled")
            time.sleep(0.05)

    def _resolve_human_task(self, pending: PendingExternal) -> dict:
        """Human actions get a durable request that outlives this process.

        Unlike a model call, the question may sit unanswered for days while
        nothing is running. The task row is that durable request, and its id is
        derived from the position and inputs so a restart re-finds the same one
        instead of asking twice.
        """

        node = cast(ActStmt, pending.node)
        action = node.action
        assert isinstance(action, HumanAction)
        path = self.node_paths.get(id(node)) or []
        input_hash = _input_hash(pending.inputs)
        task_id = human_task_id(self.role, path, input_hash, self._human_task_nonce())
        spec = build_human_task_spec(action, pending.inputs)

        _begin_immediate(self.conn, self.stop)
        try:
            task, created = ensure_human_task(
                self.conn,
                task_id=task_id,
                role=self.role,
                locator=path,
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
            self.human_backend, "claims_pending_human_tasks", False
        )
        if (created or claims_pending) and task["status"] == "pending" and not sqlite_owned:
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
        answer = validate_human_task_result(
            task["spec"],
            task["result"] or {},
            context=f"Human task {task_id!r} result",
        )
        return {node.outputs[0].name: answer[action.output]}

    def _human_task_nonce(self) -> int:
        """Distinguish repeat visits to the same human action across a loop.

        Two iterations reach the same statement with the same inputs and must
        still ask twice, so position alone is not enough. ``steps`` is the
        count of committed steps: durable, and identical after a restart, so a
        crashed-and-resumed role re-finds its pending question instead of
        asking a second one.
        """

        return self.steps

    def _resolve_external(self, pending: PendingExternal) -> dict:
        node = cast(ActStmt, pending.node)
        action = node.action
        if isinstance(action, HumanAction) and action.visible:
            return self._resolve_human_task(pending)
        return external_out_map(
            action,
            pending.inputs,
            node.outputs,
            self.llm_backend,
            self.human_backend,
            self.assistant_backend,
            # The retry wait must end the moment the deployment is stopped,
            # and this call is made with no transaction open, so a long wait
            # here blocks nothing but itself.
            stop=self.stop,
            trace=self.trace,
        )

    # ---- the loop ---------------------------------------------------------
    def step(self, residual, *, resolved: dict | None = None):
        return _step(
            residual,
            self.env,
            cast(Any, self.channel),
            self.ns,
            self.llm_backend,
            self.human_backend,
            self.monitor,
            self.trace,
            self.formula_conditions,
            self.stop,
            durable=True,
            assistant_backend=self.assistant_backend,
            resolved=resolved,
        )

    def _reset_idle_backoff(self) -> None:
        self._idle_sleep = self._IDLE_SLEEP_INITIAL

    def _sleep_after_idle_step(self) -> None:
        time.sleep(self._idle_sleep)
        self._idle_sleep = min(
            self._IDLE_SLEEP_MAX, self._idle_sleep * self._IDLE_SLEEP_FACTOR
        )

    def run_live(self) -> None:
        while not isinstance(self.residual, EmptyStmt):
            if self.stop is not None and self.stop.is_set():
                raise WorkflowCancelled("Workflow cancelled")

            _begin_immediate(self.conn, self.stop)
            out, progressed = self.step(self.residual)

            if isinstance(out, PendingExternal):
                # Nothing durable happened yet, so drop the transaction before
                # touching the outside world.
                self._rollback()
                status, detail = self._external_status(out)
                self._publish_status(status, detail)
                if out.trace_start is not None:
                    self.trace(out.trace_start)
                started_at = time.monotonic()
                try:
                    out_map = self._resolve_external(out)
                except BaseException as exc:
                    if (
                        out.trace_start is not None
                        and not isinstance(exc, WorkflowCancelled)
                    ):
                        try:
                            self.trace({
                                "type": "act_failed",
                                "lifeline": self.role,
                                "action": out.trace_start.get("action"),
                                "action_kind": out.trace_start.get("action_kind"),
                                "seq": out.trace_seq,
                                "attempt_id": out.attempt_id,
                                "duration_ms": max(
                                    0,
                                    round(
                                        (time.monotonic() - started_at) * 1000
                                    ),
                                ),
                                "error": type(exc).__name__,
                                "message": " ".join(str(exc).split())[:500],
                            })
                        except Exception:
                            # Failure evidence is best-effort and must never
                            # replace the exception that actually killed the
                            # lifeline.
                            pass
                    raise
                duration_ms = max(
                    0, round((time.monotonic() - started_at) * 1000)
                )
                # The result and the next control state commit together. If the
                # process dies before this, the control state still points at
                # the action and it runs again.
                _begin_immediate(self.conn, self.stop)
                try:
                    advanced, moved = self.step(
                        self.residual,
                        resolved={
                            id(out.node): _ResolvedExternal(
                                out_map,
                                out.trace_seq,
                                out.attempt_id,
                                duration_ms,
                            )
                        },
                    )
                    assert moved and not isinstance(advanced, PendingExternal), (
                        "resolved external action failed to advance the role"
                    )
                    next_status, next_detail = self._status_for(
                        advanced, blocked=False
                    )
                    self._commit_state(advanced, next_status, next_detail)
                except BaseException:
                    self._rollback()
                    raise
                self.residual = advanced
                self._reset_idle_backoff()
                continue

            if progressed:
                status, detail = self._status_for(out, blocked=False)
                try:
                    self._commit_state(out, status, detail)
                except BaseException:
                    self._rollback()
                    raise
                self.residual = out
                self._reset_idle_backoff()
                continue

            self._rollback()
            status, detail = self._status_for(self.residual, blocked=True)
            self._publish_status(status, detail)
            if self.stop is not None and self.stop.is_set():
                raise WorkflowCancelled("Workflow cancelled")
            self._sleep_after_idle_step()

    def run(self) -> dict:
        try:
            self.run_live()
        except BaseException as exc:
            if self.conn.in_transaction:
                self._rollback()
            if isinstance(exc, RoleStateConflict):
                raise
            state = "cancelled" if isinstance(exc, WorkflowCancelled) else "failed"
            failure_detail = {
                "error": type(exc).__name__,
                "message": " ".join(str(exc).split())[:500],
            }
            try:
                set_role_status(
                    self.conn,
                    self.role,
                    state,
                    failure_detail,
                    expected_steps=self.steps,
                )
            except (sqlite3.Error, RoleStateConflict):
                pass
            if state == "failed":
                try:
                    record_last_failure(self.conn, self.role, exc)
                except sqlite3.Error:
                    pass
            raise
        set_role_status(
            self.conn,
            self.role,
            "done",
            {},
            expected_steps=self.steps,
        )
        return self.env


def run_role(conn, role: str, local_stmt, env: dict, ns: dict, *,
             llm_backend=None, human_backend=None, assistant_backend=None,
             trace=None, monitor=None,
             formula_conditions: dict | None = None) -> dict:
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
