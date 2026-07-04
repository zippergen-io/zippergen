"""End-to-end integration tests for the durable-deploy hardening feature:
at-least-once external-effect semantics on a crash before the journal commit,
that a blocking external act does not hold the SQLite write lock, and a real
two-process parallel `kill -9` resume.

Cross-restart memoization of an external act is covered separately in
tests/test_serve_journal.py::test_external_act_memoized_across_restart.
"""
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from zippergen import Lifeline, Var, workflow
from zippergen.actions import llm as llm_deco
from zippergen.projection import project
from zippergen.store import open_store
import zippergen.store as store_mod
from zippergen.serve import run_role

A = Lifeline("A")
n = Var("n", int, default=0)
label = Var("label", int, default=0)

# Single-role workflow with one external (LLM) act — self-contained, no peer —
# so the crash-injection and lock tests need no thread orchestration.
classify = llm_deco(system="s", user="{n}", parse="json", outputs=[("label", int)])


@classify
def classify_fn(n: int) -> int: ...


@workflow
def solo(n: int @ A):
    A: label = classify_fn(n)
    return label @ A


def test_at_least_once_replays_act_on_crash_before_commit(tmp_path, monkeypatch):
    """Crash after the external call returns but before the act row commits ->
    on restart the act re-executes (at-least-once); final state is correct and
    the log holds exactly one committed act row."""
    path = str(tmp_path / "s.sqlite")
    calls = {"n": 0}

    def backend(action, inputs):
        calls["n"] += 1
        return {"label": 42}                          # deterministic result

    la = project(solo, A)

    # First attempt: raise right after record_act's INSERT, before COMMIT persists.
    orig_record = store_mod.DurableChannel.record_act

    def crash_record(self, payload):
        orig_record(self, payload)                    # INSERT into the open txn
        raise sqlite3.OperationalError("simulated crash before act commit")

    monkeypatch.setattr(store_mod.DurableChannel, "record_act", crash_record)

    conn1 = open_store(path)
    with pytest.raises(sqlite3.OperationalError):
        run_role(conn1, "A", la, {"n": 1}, solo.ns, llm_backend=backend)
    conn1.close()                                     # release the uncommitted txn's lock
    assert calls["n"] == 1

    # Restart cleanly: the act row was never committed -> re-execute, then finish.
    monkeypatch.setattr(store_mod.DurableChannel, "record_act", orig_record)
    env = run_role(open_store(path), "A", la, {"n": 1}, solo.ns, llm_backend=backend)
    assert calls["n"] == 2 and env["label"] == 42
    acts = open_store(path).execute(
        "SELECT COUNT(*) FROM events WHERE sender='A' AND kind='act'").fetchone()[0]
    assert acts == 1                                  # exactly one committed act row


def test_blocking_external_act_does_not_hold_write_lock(tmp_path):
    """While role A is inside a slow external act, a second connection can still
    take the write lock — proof the lock is released across the blocking call."""
    path = str(tmp_path / "s.sqlite")
    started = threading.Event()

    def slow_backend(action, inputs):
        started.set()
        time.sleep(0.5)
        return {"label": 7}

    la = project(solo, A)
    t = threading.Thread(target=lambda: run_role(
        open_store(path), "A", la, {"n": 1}, solo.ns, llm_backend=slow_backend))
    t.start()
    assert started.wait(timeout=5)                    # A is now inside the slow act
    other = open_store(path)
    other.execute("BEGIN IMMEDIATE")                  # would block/raise if A held the lock
    other.execute("INSERT INTO events(sender,receiver,channel,kind,payload,causal_stamp)"
                  " VALUES('B',NULL,NULL,'msg','[1]',NULL)")
    other.execute("COMMIT")
    t.join(timeout=10)
    assert not t.is_alive()


def test_parallel_two_process_kill9(tmp_path):
    store = str(tmp_path / "par.sqlite")
    wf = os.path.join(os.path.dirname(__file__), "fixtures", "parallel_deploy.py")

    def serve(role, inputs):
        cmd = [sys.executable, "-m", "zippergen.serve", "serve",
               "--workflow", wf, "--role", role, "--store", store]
        for k, val in inputs.items():
            cmd += ["--input", f"{k}={val}"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

    b = serve("B", {})
    a = serve("A", {"x": 0, "y": 0})
    time.sleep(0.6)
    a.kill()                                          # SIGKILL mid-run
    a.wait()
    a2 = serve("A", {"x": 0, "y": 0})                 # supervisor restarts A
    out_a, _ = a2.communicate(timeout=40)
    out_b, _ = b.communicate(timeout=40)
    assert a2.returncode == 0, f"A(restarted) failed: {out_a}"
    assert b.returncode == 0, f"B failed: {out_b}"
    result = json.loads(out_a.strip().splitlines()[-1])
    assert result["x"] == 1                           # branch completed exactly once
