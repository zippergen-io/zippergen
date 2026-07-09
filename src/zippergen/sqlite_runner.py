"""Local SQLite-backed workflow runner.

This is the default local workflow runner: it runs all roles in one Python
process, but every role communicates through the durable SQLite store via
``RoleRunner``.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

from zippergen.projection import project
from zippergen.role_runner import RoleRunner
from zippergen.runtime import _build_formula_monitors, console_trace, mock_llm
from zippergen.store import open_store, write_workflow_result
from zippergen.syntax import Lifeline, Var, Workflow, _ordered_workflow_lifelines

__all__ = ["LocalSupervisor", "run_sqlite"]

_NO_RESULT = object()


def _default_env(wf: Workflow) -> dict:
    return {k: v.default for k, v in wf.ns.items() if isinstance(v, Var)}


def _seed_role_env(conn, role: str, env: dict) -> dict:
    """Persist one role's seed env, returning an existing seed on restart."""
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
            (role, None, None, "seed", json.dumps(env), None),
        )
        conn.execute("COMMIT")
        return dict(env)
    except BaseException:
        conn.execute("ROLLBACK")
        raise


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
        verbose: bool = False,
        trace=None,
        timeout: float = 60.0,
    ) -> None:
        self.wf = wf
        self.lifelines = tuple(lifelines) if lifelines is not None else _ordered_workflow_lifelines(wf)
        self.initial_envs = initial_envs or {}
        self.store_path = store_path
        self.llm_backend = llm_backend if llm_backend is not None else mock_llm
        if human_backend is None:
            from zippergen.human_backends import make_sqlite_human_backend
            human_backend = make_sqlite_human_backend()
        self.human_backend = human_backend
        self.trace = trace if trace is not None else (console_trace if verbose else None)
        self.timeout = timeout
        self.stop = threading.Event()

    def _seed_all(self) -> dict[str, dict]:
        seeded: dict[str, dict] = {}
        conn = open_store(self.store_path)
        try:
            for lifeline in self.lifelines:
                env = _default_env(self.wf)
                env.update(self.initial_envs.get(lifeline.name, {}))
                seeded[lifeline.name] = _seed_role_env(conn, lifeline.name, env)
        finally:
            conn.close()
        return seeded

    def _load_existing_result(self) -> object:
        conn = open_store(self.store_path)
        try:
            row = conn.execute(
                "SELECT value FROM workflow_results WHERE workflow=?",
                (self.wf.name,),
            ).fetchone()
            if row is None:
                return _NO_RESULT
            return _restore_workflow_result(self.wf, json.loads(row[0]))
        finally:
            conn.close()

    def run(self) -> object:
        existing = self._load_existing_result()
        if existing is not _NO_RESULT:
            return existing

        seeded_envs = self._seed_all()
        monitors, formula_conditions = _build_formula_monitors(self.wf, self.lifelines)
        result_boxes: dict[str, object] = {}
        threads: list[threading.Thread] = []

        def make_target(lifeline: Lifeline, seed_env: dict):
            local_stmt = project(self.wf, lifeline)

            def target() -> None:
                conn = open_store(self.store_path)
                try:
                    runner = RoleRunner(
                        conn,
                        lifeline.name,
                        local_stmt,
                        dict(seed_env),
                        self.wf.ns,
                        llm_backend=self.llm_backend,
                        human_backend=self.human_backend,
                        trace=self.trace,
                        monitor=monitors.get(lifeline.name),
                        formula_conditions=formula_conditions,
                        stop=self.stop,
                    )
                    result_boxes[lifeline.name] = runner.run()
                except BaseException as exc:
                    result_boxes[lifeline.name] = exc
                    self.stop.set()
                finally:
                    conn.close()

            return target

        for lifeline in self.lifelines:
            t = threading.Thread(
                target=make_target(lifeline, seeded_envs[lifeline.name]),
                name=lifeline.name,
                daemon=True,
            )
            threads.append(t)
            t.start()

        if self.timeout <= 0:
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
                if "Workflow cancelled" in str(result):
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


def run_sqlite(
    wf: Workflow,
    lifelines: list[Lifeline] | tuple[Lifeline, ...] | None = None,
    initial_envs: dict[str, dict[str, object]] | None = None,
    *,
    store_path: str | None = None,
    llm_backend=None,
    human_backend=None,
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
        verbose=verbose,
        trace=trace,
        timeout=timeout,
    ).run()
