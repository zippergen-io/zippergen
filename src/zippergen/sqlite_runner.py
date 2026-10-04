"""Local SQLite-backed workflow runner.

This is the default local workflow runner: it runs all roles in one Python
process, but every role communicates through the durable SQLite store via
``RoleRunner``.
"""
from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from zippergen.availability import require_workflow_availability
from zippergen.control import program_fingerprint
from zippergen.errors import WorkflowCancelled
from zippergen.projection import project, project_templates
from zippergen.role_runner import RoleRunner
from zippergen.runtime import _build_formula_monitors, console_trace, mock_llm
from zippergen.store import (
    claim_workflow_identity,
    load_workflow_result,
    list_role_states,
    list_spawn_allocations,
    open_store,
    write_workflow_result,
)
from zippergen.syntax import (
    contains_initiated_spawn,
    Lifeline,
    Var,
    Workflow,
    _clone_zvalue,
    _ordered_workflow_lifelines,
    validate_zvalue,
)

__all__ = ["LocalSupervisor", "run_sqlite"]

_NO_RESULT = object()


def _default_env(wf: Workflow) -> dict:
    return {
        v.name: _clone_zvalue(v.default, v.type)
        for v in wf.ns.values()
        if isinstance(v, Var) and v.has_default
    }


def _workflow_result(wf: Workflow, final_envs: dict[str, dict]) -> object:
    if len(wf.outputs) == 0:
        return final_envs
    if len(wf.outputs) == 1:
        var, lifeline = wf.outputs[0]
        return final_envs[lifeline.name][var.name]
    return tuple(final_envs[lifeline.name][var.name] for var, lifeline in wf.outputs)


def _restore_workflow_result(wf: Workflow, value: object) -> object:
    if len(wf.outputs) > 1 and isinstance(value, list):
        return tuple(value)
    return value


class LocalSupervisor:
    """Run all projected roles locally over one SQLite event store."""

    def __init__(
        self,
        wf: Workflow,
        lifelines: list[Lifeline] | tuple[Lifeline, ...] | None,
        initial_envs: dict[str, dict[str, object]] | None,
        *,
        store_path: str,
        llm_backend=None,
        human_backend=None,
        assistant_backend=None,
        verbose: bool = False,
        trace=None,
        timeout: float = 60.0,
    ) -> None:
        require_workflow_availability(wf, initial_envs)
        from zippergen.session_cpl import compile_session_cpl
        self.cpl_plans = compile_session_cpl(wf)
        self.wf = wf
        self.lifelines = tuple(lifelines) if lifelines is not None else _ordered_workflow_lifelines(wf)
        self.initial_envs = initial_envs or {}
        self.store_path = store_path
        self.llm_backend = llm_backend if llm_backend is not None else mock_llm
        if human_backend is None:
            from zippergen.human_backends import make_sqlite_human_backend
            human_backend = make_sqlite_human_backend()
        self.human_backend = human_backend
        if assistant_backend is None:
            from zippergen.assistant_backends import make_cli_assistant_backend
            assistant_backend = make_cli_assistant_backend()
        from zippergen.assistant_backends import _new_assistant_execution
        self.assistant_backend = _new_assistant_execution(assistant_backend)
        self.trace = trace if trace is not None else (console_trace if verbose else None)
        self.timeout = timeout
        self.stop = threading.Event()

    def _starting_envs(self) -> dict[str, dict]:
        """Build each role's starting variables.

        These are only used when a role has no durable state yet. A resuming
        role reads its own committed environment instead.
        """

        starting: dict[str, dict] = {}
        for lifeline in self.lifelines:
            env = _default_env(self.wf)
            supplied = self.initial_envs.get(lifeline.name, {})
            env.update(supplied)
            for name, ztype, owner in self.wf.inputs:
                if (
                    owner is not None
                    and owner.name == lifeline.name
                    and name in supplied
                ):
                    env[name] = _clone_zvalue(
                        validate_zvalue(
                            supplied[name],
                            ztype,
                            context=f"{self.wf.name} input {name!r}",
                        ),
                        ztype,
                    )
            starting[lifeline.name] = env
        return starting

    def _claim_identity(self, local_programs: Mapping[str, object], templates=None) -> None:
        """Refuse to resume durable state written by different code."""

        conn = open_store(self.store_path)
        try:
            claim_workflow_identity(
                conn,
                self.wf.name,
                program_fingerprint(local_programs, templates=templates,
                                    session_cpl=self.cpl_plans),
            )
        finally:
            conn.close()

    def _load_existing_result(self) -> object:
        conn = open_store(self.store_path)
        try:
            result = load_workflow_result(conn, self.wf.name)
            if result is None:
                return _NO_RESULT
            return _restore_workflow_result(self.wf, result)
        finally:
            conn.close()

    def run(self) -> object:
        templates = project_templates(self.wf)
        if contains_initiated_spawn(self.wf.body):
            return self._run_fresh(templates)
        local_programs = {
            lifeline.name: project(self.wf, lifeline) for lifeline in self.lifelines
        }
        self._claim_identity(local_programs)
        existing = self._load_existing_result()
        if existing is not _NO_RESULT:
            return existing

        starting_envs = self._starting_envs()
        monitors, formula_conditions = (({}, {}) if self.cpl_plans else
                                        _build_formula_monitors(self.wf, self.lifelines))
        result_boxes: dict[str, object] = {}
        threads: list[threading.Thread] = []
        human_backend = self.human_backend
        human_dispatcher = None
        if getattr(human_backend, "requires_main_thread", False):
            from zippergen.human_backends import _MainThreadHumanDispatcher

            human_dispatcher = _MainThreadHumanDispatcher(human_backend, self.stop)
            human_backend = human_dispatcher.worker_backend

        def make_target(lifeline: Lifeline, seed_env: dict):
            local_stmt = local_programs[lifeline.name]

            def target() -> None:
                conn = None
                try:
                    conn = open_store(self.store_path)
                    runner = RoleRunner(
                        conn,
                        lifeline.name,
                        local_stmt,
                        dict(seed_env),
                        self.wf.ns,
                        llm_backend=self.llm_backend,
                        human_backend=human_backend,
                        assistant_backend=self.assistant_backend,
                        trace=self.trace,
                        monitor=monitors.get(lifeline.name),
                        formula_conditions=formula_conditions, cpl_plans=self.cpl_plans,
                        stop=self.stop,
                    )
                    result_boxes[lifeline.name] = runner.run()
                except BaseException as exc:
                    result_boxes[lifeline.name] = exc
                    self.stop.set()
                finally:
                    if conn is not None:
                        conn.close()

            return target

        for lifeline in self.lifelines:
            t = threading.Thread(
                target=make_target(lifeline, starting_envs[lifeline.name]),
                name=lifeline.name,
                daemon=True,
            )
            threads.append(t)
            t.start()

        if human_dispatcher is not None:
            deadline = (
                None if self.timeout <= 0 else time.monotonic() + self.timeout
            )
            try:
                while any(t.is_alive() for t in threads):
                    if self.stop.is_set():
                        break
                    wait = 0.05
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self.stop.set()
                            human_dispatcher.cancel_pending()
                            for t in threads:
                                t.join(timeout=1.0)
                            alive = next((t for t in threads if t.is_alive()), None)
                            if alive is not None:
                                raise TimeoutError(
                                    f"Lifeline '{alive.name}' did not finish within "
                                    f"{self.timeout}s"
                                )
                            raise TimeoutError(
                                f"Workflow did not finish within {self.timeout}s"
                            )
                        wait = min(wait, remaining)
                    human_dispatcher.service_next(timeout=wait)
            except BaseException:
                self.stop.set()
                human_dispatcher.cancel_pending()
                for t in threads:
                    t.join(timeout=1.0)
                raise
            for t in threads:
                if t.is_alive():
                    t.join(timeout=1.0)
        elif self.timeout <= 0:
            try:
                while any(t.is_alive() for t in threads):
                    if self.stop.is_set():
                        break
                    time.sleep(0.05)
            except KeyboardInterrupt:
                self.stop.set()
                for t in threads:
                    t.join(timeout=1.0)
                raise
            for t in threads:
                if t.is_alive():
                    t.join(timeout=1.0)
        else:
            deadline = time.monotonic() + self.timeout
            for t in threads:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.stop.set()
                    raise TimeoutError(f"Workflow did not finish within {self.timeout}s")
                t.join(timeout=remaining)
                if t.is_alive():
                    self.stop.set()
                    t.join(timeout=1.0)
                    raise TimeoutError(f"Lifeline '{t.name}' did not finish within {self.timeout}s")

        root_cause: tuple[str, BaseException] | None = None
        cancelled: tuple[str, BaseException] | None = None
        final_envs: dict[str, dict] = {}
        missing: list[str] = []
        for lifeline in self.lifelines:
            result = result_boxes.get(lifeline.name)
            if result is None:
                missing.append(lifeline.name)
                continue
            if isinstance(result, BaseException):
                if isinstance(result, WorkflowCancelled):
                    if cancelled is None:
                        cancelled = (lifeline.name, result)
                elif root_cause is None:
                    root_cause = (lifeline.name, result)
            else:
                final_envs[lifeline.name] = cast(dict, result)

        error = root_cause or cancelled
        if error is not None:
            name, exc = error
            raise RuntimeError(f"Lifeline '{name}' raised: {exc}") from exc
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise RuntimeError(f"Lifeline(s) produced no result: {names}.")

        result = _workflow_result(self.wf, final_envs)
        conn = open_store(self.store_path)
        try:
            write_workflow_result(conn, self.wf.name, result)
        finally:
            conn.close()
        return result

    def _run_fresh(self, templates) -> object:
        from zippergen.control import decode_control
        from zippergen.sessions import SessionState, validate_root_endpoint
        from zippergen.syntax import EmptyStmt

        roots = {role.name: project(self.wf, role) for role in self.lifelines}
        root_bindings = {role.name: validate_root_endpoint(role.name)
                         for role in self.lifelines}
        configured = {endpoint: role for role, endpoint in root_bindings.items()}
        local_programs = dict(roots)
        self._claim_identity(local_programs, templates)
        existing = self._load_existing_result()
        if existing is not _NO_RESULT:
            conn = open_store(self.store_path)
            try:
                pending = conn.execute(
                    "SELECT 1 FROM outstanding_messages LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            if pending is None:
                return existing

        starting = self._starting_envs()
        results: dict[str, object] = {}
        threads: dict[str, threading.Thread] = {}
        completion = threading.Event()
        backend = self.human_backend
        dispatcher = None
        if getattr(backend, "requires_main_thread", False):
            from zippergen.human_backends import _MainThreadHumanDispatcher
            dispatcher = _MainThreadHumanDispatcher(backend, self.stop)
            backend = dispatcher.worker_backend

        def worker(endpoint, static_role):
            conn = None
            try:
                conn = open_store(self.store_path)
                runner = RoleRunner(
                    conn, endpoint, local_programs.get(endpoint, EmptyStmt()),
                    starting.get(endpoint, {}), self.wf.ns,
                    llm_backend=self.llm_backend,
                    human_backend=backend,
                    assistant_backend=self.assistant_backend,
                    trace=self.trace, stop=self.stop,
                    root_role=endpoint if endpoint in roots else None,
                    assistant_participant=static_role,
                    root_bindings=root_bindings,
                    templates=templates, completion=completion, cpl_plans=self.cpl_plans,
                )
                results[endpoint] = runner.run()
            except BaseException as exc:
                results[endpoint] = exc
                self.stop.set()
            finally:
                if conn is not None:
                    conn.close()

        def start_worker(endpoint, static_role):
            thread = threading.Thread(target=worker, args=(endpoint, static_role),
                                      name=static_role, daemon=True)
            threads[endpoint] = thread
            thread.start()

        for endpoint in sorted(configured):
            start_worker(endpoint, endpoint)

        observer = open_store(self.store_path)
        deadline = None if self.timeout <= 0 else time.monotonic() + self.timeout
        try:
            while not completion.is_set():
                if self.stop.is_set():
                    break
                for endpoint, thread in list(threads.items()):
                    if not thread.is_alive():
                        thread.join()
                        del threads[endpoint]
                # These facts must come from one SQLite snapshot. If an
                # invitation is accepted between separate reads, old "done"
                # role rows plus the new empty queue falsely look quiescent.
                observer.execute("BEGIN")
                try:
                    states = list_role_states(observer)
                    pending_messages = observer.execute(
                        "SELECT 1 FROM outstanding_messages LIMIT 1"
                    ).fetchone()
                    pending_init = {row[0] for row in observer.execute(
                        "SELECT DISTINCT receiver FROM outstanding_messages "
                        "WHERE channel LIKE 'init:invite:%'"
                    ).fetchall()}
                    allocations = list_spawn_allocations(observer)
                    observer.execute("COMMIT")
                except BaseException:
                    observer.execute("ROLLBACK")
                    raise
                state_by_endpoint = {row["role"]: row for row in states}
                for _child, role, endpoint in allocations:
                    if endpoint not in configured:
                        configured[endpoint] = role
                        saved = state_by_endpoint.get(endpoint)
                        if saved is not None and saved["status"] == "done" and endpoint not in pending_init:
                            control = decode_control(
                                EmptyStmt(), saved["control"], templates=templates,
                                endpoint=endpoint, root_bindings=root_bindings,
                                keep_alive=True,
                            )
                            if isinstance(control, SessionState) and control.can_retire:
                                # Resume only live endpoints; retain their durable
                                # completion/activation evidence without a thread.
                                results[endpoint] = {}
                                continue
                        start_worker(endpoint, role)
                for endpoint in pending_init:
                    if endpoint in configured and endpoint not in threads:
                        start_worker(endpoint, configured[endpoint])
                if (len(states) == len(configured)
                        and all(row["status"] == "done" for row in states)
                        and pending_messages is None):
                    completion.set()
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    self.stop.set()
                    raise TimeoutError(f"Workflow did not finish within {self.timeout}s")
                if dispatcher is not None:
                    dispatcher.service_next(timeout=0.02)
                else:
                    time.sleep(0.02)
        finally:
            observer.close()
            if self.stop.is_set():
                completion.set()
                if dispatcher is not None:
                    dispatcher.cancel_pending()
            for thread in threads.values():
                thread.join(timeout=1)

        for endpoint, value in results.items():
            if isinstance(value, BaseException):
                raise RuntimeError(f"Endpoint {endpoint!r} raised: {value}") from value
        if len(results) != len(configured):
            raise RuntimeError("An endpoint worker stopped without a result")
        final_envs = {role: cast(dict, results[endpoint])
                      for role, endpoint in root_bindings.items()}
        result = _workflow_result(self.wf, final_envs)
        conn = open_store(self.store_path)
        try:
            write_workflow_result(conn, self.wf.name, result)
        finally:
            conn.close()
        return result


def run_sqlite(
    wf: Workflow,
    lifelines: list[Lifeline] | tuple[Lifeline, ...] | None = None,
    initial_envs: dict[str, dict[str, object]] | None = None,
    *,
    store_path: str | None = None,
    llm_backend=None,
    human_backend=None,
    assistant_backend=None,
    verbose: bool = False,
    trace=None,
    timeout: float = 60.0,
) -> object:
    """Run ``wf`` locally through the durable SQLite role runner."""
    if store_path is None:
        with tempfile.TemporaryDirectory(prefix="zippergen-run-") as tmp:
            path = str(Path(tmp) / "run.sqlite")
            return LocalSupervisor(
                wf,
                lifelines,
                initial_envs,
                store_path=path,
                llm_backend=llm_backend,
                human_backend=human_backend,
                assistant_backend=assistant_backend,
                verbose=verbose,
                trace=trace,
                timeout=timeout,
            ).run()

    return LocalSupervisor(
        wf,
        lifelines,
        initial_envs,
        store_path=store_path,
        llm_backend=llm_backend,
        human_backend=human_backend,
        assistant_backend=assistant_backend,
        verbose=verbose,
        trace=trace,
        timeout=timeout,
    ).run()
