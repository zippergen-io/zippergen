"""Deferred probes yield branches/sessions and do not commit empty results."""
from contextlib import closing
import threading
import time

import pytest

from zippergen import (EffectPending, Lifeline, Var, effect, pure, workflow, spawn,
                       parallel, branch, run_sqlite, project)
from zippergen.errors import WorkflowCancelled
from zippergen.role_runner import RoleRunner
from zippergen.store import open_store, load_role_state, list_history

A = Lifeline('A')
value = Var('value', int)
other = Var('other', int)
i = Var('i', int, default=0)
ready = set()
probes = []


@effect(visible=False)
def read_when_ready(number: int) -> int:
    probes.append(number)
    if number not in ready:
        raise EffectPending(retry_after=0.01)
    return number * 10


@effect
def make_ready() -> int:
    ready.update({1, 2})
    return 3


@pure
def increment(number: int) -> int:
    return number + 1


@workflow
def parallel_wait():
    with parallel:
        with branch:
            A: value = read_when_ready(1)
        with branch:
            A: other = make_ready()
    return value @ A


@workflow
def session_wait():
    while (i < 2) @ A:
        A: i = increment(i)
        with spawn(fresh='all'):
            A: value = read_when_ready(i)
    A: other = make_ready()


@workflow
def just_wait():
    A: value = read_when_ready(1)
    return value @ A


@pytest.fixture(autouse=True)
def clean_probes():
    probes.clear()
    ready.clear()


def test_wait_yields_to_parallel_branch(tmp_path):
    assert run_sqlite(parallel_wait, store_path=str(tmp_path/'run.sqlite'), timeout=3) == 10
    assert len(probes) >= 2


def test_same_action_template_in_sibling_sessions_does_not_stall_parent(tmp_path):
    run_sqlite(session_wait, store_path=str(tmp_path/'run.sqlite'), timeout=3)
    assert {1, 2} <= set(probes)
    with closing(open_store(str(tmp_path/'run.sqlite'))) as db:
        assert load_role_state(db, 'A')['status'] == 'done'


def test_quiet_probes_leave_control_and_history_unchanged_and_resume(tmp_path):
    path = str(tmp_path/'run.sqlite')
    stop = threading.Event()

    class StopAfterProbes(RoleRunner):
        def _sleep_after_idle_step(self):
            if len(probes) >= 4:
                stop.set()
            super()._sleep_after_idle_step()

    with closing(open_store(path)) as db:
        runner = StopAfterProbes(db, 'A', project(just_wait, A), {}, just_wait.ns, stop=stop)
        before = load_role_state(db, 'A')
        with pytest.raises(WorkflowCancelled):
            runner.run_live()
        after = load_role_state(db, 'A')
        assert after['steps'] == before['steps']
        assert after['control'] == before['control']
        assert after['env'] == before['env']
        assert list_history(db) == []
        ready.add(1)
        # A fresh runner has no volatile delay. The saved action still runs.
        resumed = RoleRunner(db, 'A', project(just_wait, A), {}, just_wait.ns)
        assert resumed.run()['value'] == 10


def test_long_retry_delay_does_not_delay_shutdown(tmp_path):
    @effect(visible=False)
    def later() -> int:
        raise EffectPending(retry_after=3600)

    # Use explicit IR so this local action is not a workflow module dependency.
    from zippergen import ActStmt
    stmt = ActStmt(A, later, (), (value,))
    stop = threading.Event()
    timer = threading.Timer(.1, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        with closing(open_store(str(tmp_path/'run.sqlite'))) as db:
            runner = RoleRunner(db, 'A', stmt, {}, {}, stop=stop)
            with pytest.raises(WorkflowCancelled):
                runner.run_live()
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2


@pytest.mark.parametrize('delay', [0, -1, float('inf'), float('nan'), True])
def test_invalid_retry_delay(delay):
    with pytest.raises(ValueError, match='finite and positive'):
        EffectPending(retry_after=delay)


def test_visible_wait_attempt_is_paired_in_history(tmp_path):
    from zippergen import ActStmt
    from zippergen.execution_inspection import _trace_rows

    attempts = []

    @effect
    def visible_probe() -> int:
        attempts.append(True)
        if len(attempts) == 1:
            raise EffectPending(retry_after=.01)
        return 7

    with closing(open_store(str(tmp_path/'run.sqlite'))) as db:
        runner = RoleRunner(db, 'A', ActStmt(A, visible_probe, (), (value,)), {}, {})
        assert runner.run()['value'] == 7
        rows = _trace_rows(list_history(db))
    assert any('waiting' in str(row) for row in rows)
    assert all('incomplete' not in str(row) for row in rows)


def test_pending_probe_recovers_in_a_fresh_process(tmp_path):
    import os
    import subprocess
    import sys

    script = tmp_path/'probe.py'
    script.write_text('''from pathlib import Path
import os
from zippergen import EffectPending, Lifeline, effect, workflow, run_sqlite
A = Lifeline("A")
@effect(visible=False)
def probe() -> int:
    if not Path("ready").exists():
        raise EffectPending(retry_after=.01)
    return 42
@workflow
def wait_for_value() -> int:
    A: result = probe()
    return result @ A
if os.environ.get("CRASH_WAIT"):
    from zippergen.role_runner import RoleRunner
    def stop_at_wait(self):
        os._exit(97)
    RoleRunner._sleep_after_idle_step = stop_at_wait
result = run_sqlite(wait_for_value, store_path="run.sqlite", timeout=5)
Path("result").write_text(str(result))
''')
    env = {**os.environ, 'CRASH_WAIT': '1'}
    first = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                           env=env, capture_output=True, text=True, timeout=10)
    assert first.returncode == 97, first.stderr
    (tmp_path/'ready').touch()
    env.pop('CRASH_WAIT')
    second = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=10)
    assert second.returncode == 0, second.stderr
    assert (tmp_path/'result').read_text() == '42'


def test_memory_runner_reports_unsupported_wait_clearly():
    from zippergen import run
    with pytest.raises(RuntimeError, match='SQLite runner'):
        run(just_wait, [A], {}, timeout=2)
