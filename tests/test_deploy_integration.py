"""End-to-end durable behaviour, including a real two-process `kill -9`.

Three properties: an external action whose result never committed runs again on
restart, a blocking external action does not hold the SQLite write lock, and a
SIGKILLed role resumes from its committed state without redoing committed work.
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
from zippergen.role_runner import run_role

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


def test_an_external_action_runs_again_when_its_result_did_not_commit(
    tmp_path, monkeypatch
):
    """The documented weaker guarantee, at the real commit boundary.

    The model call returns, then the process dies before the transaction that
    would record the result and advance the control state. On restart the
    control state still points at the action, so it runs a second time.
    """

    path = str(tmp_path / "s.sqlite")
    calls = {"n": 0}

    def backend(action, inputs):
        calls["n"] += 1
        return {"label": 42}

    la = project(solo, A)

    import zippergen.role_runner as role_runner_mod

    real_write = role_runner_mod.write_role_state
    crashed = {"done": False}

    def crash_once(conn, role, **kwargs):
        # Let the initial state write through, then die on the commit that
        # would record the model's answer.
        if not crashed["done"] and kwargs.get("seq", 0) > 0:
            crashed["done"] = True
            raise sqlite3.OperationalError("simulated crash before commit")
        return real_write(conn, role, **kwargs)

    monkeypatch.setattr(role_runner_mod, "write_role_state", crash_once)

    conn1 = open_store(path)
    with pytest.raises(sqlite3.OperationalError):
        run_role(conn1, "A", la, {"n": 1}, solo.ns, llm_backend=backend)
    conn1.close()
    assert calls["n"] == 1, "the model was called"

    state = open_store(path).execute(
        "SELECT env FROM role_state WHERE role='A'"
    ).fetchone()
    assert "label" not in json.loads(state[0]), "but nothing recorded its answer"

    monkeypatch.setattr(role_runner_mod, "write_role_state", real_write)
    env = run_role(open_store(path), "A", la, {"n": 1}, solo.ns, llm_backend=backend)

    assert calls["n"] == 2, "so it is called again; this is the documented rule"
    assert env["label"] == 42


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
    other.execute(
        "INSERT INTO outstanding_messages(sender,receiver,channel,payload)"
        " VALUES('B','C','main','[1]')"
    )
    other.execute("COMMIT")
    t.join(timeout=10)
    assert not t.is_alive()


def test_parallel_two_process_kill9(tmp_path):
    store = str(tmp_path / "par.sqlite")
    wf = os.path.join(os.path.dirname(__file__), "fixtures", "parallel_deploy.py")

    def serve(role, inputs):
        runner = os.path.join(
            os.path.dirname(__file__), "fixtures", "run_role_process.py"
        )
        cmd = [sys.executable, runner,
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
