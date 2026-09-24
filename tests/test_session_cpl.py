"""Current-session CPL on shared endpoints and freshly allocated endpoints."""
from contextlib import closing
from pathlib import Path
import subprocess
import sys

import pytest

from zippergen import (Lifeline, Var, workflow, fragment, pure, effect, spawn,
                       run, run_sqlite, project, human, skip)
from zippergen.formula import At, Here, P, Y, on, since, atom, true
from zippergen.channels import InProcessChannel
from zippergen.projection import project_templates
from zippergen.session_cpl import compile_session_cpl
from zippergen.sessions import SessionState, Session, ScopedChannel
from zippergen.store import open_store
from zippergen.syntax import EmptyStmt

A, B, C = (Lifeline(name) for name in ('Sender', 'Reviewer', 'Auditor'))
n = Var('n', int, default=0)
flag = Var('flag', bool)
verdict = Var('verdict', int)
first = Var('first', int)
marker = Var('marker', int)

seen_action = P(atom(lambda env, event: event.kind == 'act', version='observed-act-v1'))
matching = (At[B].flag == True) & (At[B].n == Here.n) & (At[A].n == Here.n)


@pure
def inc(x: int) -> int:
    return x + 1


@pure
def number(x: int) -> int:
    return x


@pure
def even(x: int) -> bool:
    return x % 2 == 0


@fragment
def review():
    # The parent's increment is not part of this child's causal history.
    if seen_action @ A:
        A: first = number(-1)
    else:
        A: first = number(1)
    A(n) >> B(n)
    B: flag = even(n)
    B(flag, n) >> C(flag, n)
    if matching @ C:
        C: verdict = number(1)
    else:
        C: verdict = number(0)


@workflow
def inherited_reviews():
    while (n < 4) @ A:
        A: n = inc(n)
        if seen_action @ A:
            spawn(review, fresh='all')


@workflow
def fresh_reviews():
    while (n < 4) @ A:
        A: n = inc(n)
        if seen_action @ A:
            spawn(review, at=A, retain=(A,), fresh='all')


@fragment
def grandchild():
    if seen_action @ B:
        B: first = number(-1)
    else:
        B: first = number(1)


@fragment
def child():
    skip(A)
    B: marker = number(10)
    spawn(grandchild, fresh='all')
    # Fork is not an observed event: previous is still marker's action.
    if Y(on(B)) @ B:
        B: verdict = number(1)
    else:
        B: verdict = number(-1)


@workflow
def nested():
    spawn(child, at=A, retain=(A,))


@workflow
def local_temporal():
    with spawn(fresh='all'):
        if Y(true()) @ A:
            A: first = number(-1)
        else:
            A: first = number(1)
        if since(true(), Here.first == 1) @ A:
            A: verdict = number(1)
        else:
            A: verdict = number(-1)


def execute(wf, backend, tmp_path, events):
    options = dict(trace=events.append, timeout=10)
    if backend == 'sqlite':
        return run_sqlite(wf, store_path=str(tmp_path / 'run.sqlite'), **options)
    from zippergen.syntax import _ordered_workflow_lifelines
    return run(wf, _ordered_workflow_lifelines(wf), {}, **options)


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
@pytest.mark.parametrize('wf', [inherited_reviews, fresh_reviews])
def test_isolation_and_transitive_same_session_evidence(wf, backend, tmp_path):
    events = []
    execute(wf, backend, tmp_path, events)
    assert [e['outputs']['first'] for e in events if e.get('outputs', {}).get('first')] == [1]*4
    child_first = [e for e in events if e['type'] == 'decision'
                   and e.get('lifeline') == A.name and e['session'] != '0']
    assert len(child_first) == 4 and all(e['vc'][A.name] == 1 for e in child_first)
    decisions = [e for e in events if e['type'] == 'decision' and e.get('lifeline') == C.name]
    assert len(decisions) == 4
    for event in decisions:
        session = event['session']
        receive = next(e for e in events if e['type'] == 'recv' and e.get('session') == session
                       and e.get('to') == C.name and 'n' in e.get('bindings', {}))
        result = next(e['outputs']['verdict'] for e in events if e.get('session') == session
                      and 'verdict' in e.get('outputs', {}))
        assert result == int(receive['bindings']['n'] % 2 == 0)
        assert set(event['vc']) == {A.name, B.name, C.name}
    if backend == 'sqlite':
        # Finished fresh workers may retire; already-delivered evidence remains usable.
        with closing(open_store(str(tmp_path / 'run.sqlite'))) as conn:
            assert conn.execute('SELECT COUNT(*) FROM outstanding_messages').fetchone()[0] == 0


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
@pytest.mark.parametrize('wf', [nested, local_temporal])
def test_nested_empty_histories_and_local_temporal_operators(wf, backend, tmp_path):
    events = []
    execute(wf, backend, tmp_path, events)
    values = [value for e in events for key, value in e.get('outputs', {}).items()
              if key in ('first', 'verdict')]
    assert values == [1, 1]


@pytest.mark.parametrize('guard', [At[B](true()), on(B), At[B].flag == True,
                                  atom(lambda env, event: True, version='declared-v1', roles=[B.name])])
def test_out_of_session_references_rejected(guard):
    from zippergen.syntax import Workflow, SpawnStmt, IfStmt, SkipStmt
    body = SpawnStmt(IfStmt(guard, A, SkipStmt(A), SkipStmt(A)))
    wf = Workflow('bad_scope', (), None, (), body)
    with pytest.raises(ValueError, match='outside the current session'):
        project(wf, A)


def test_session_message_rejects_foreign_or_incompatible_summary():
    plans = compile_session_cpl(inherited_reviews)
    plan = next(plan for plan in plans.values() if plan.formulas)
    channel = InProcessChannel()
    state = SessionState.start(EmptyStmt())
    session = Session('s', EmptyStmt(), monitor=plan.monitor(A.name), cpl_layout=plan.layout)
    scoped = ScopedChannel(channel, state, session, {})
    for tag in [None, {'session': 'other', 'layout': plan.layout, 'fields': {}},
                {'session': 's', 'layout': 'other', 'fields': {}}]:
        channel.put(B.name, A.name, scoped.route('test'), (), {}, {}, tag)
        with pytest.raises(ValueError, match='different session or formula layout'):
            scoped.try_get(B.name, A.name, 'test')


@workflow
def invisible_reviewer():
    with spawn(fresh='all'):
        if At[B](true()) @ A:
            A: verdict = number(-1)
        else:
            A: verdict = number(1)
        B: marker = number(99)


@workflow
def copied_data():
    A: n = number(9)
    with spawn(fresh=(A.n,)):
        if ((Here.n == 9) & ~seen_action) @ A:
            A: verdict = number(1)
        else:
            A: verdict = number(-1)


@workflow
def shared_data():
    A: n = number(9)
    with spawn():
        if ((Here.n == 9) & ~seen_action) @ A:
            A: verdict = number(1)
        else:
            A: verdict = number(-1)


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
@pytest.mark.parametrize('wf', [invisible_reviewer, copied_data, shared_data])
def test_data_is_not_history_and_invisible_is_not_absent(wf, backend, tmp_path):
    events = []
    execute(wf, backend, tmp_path, events)
    assert [e['outputs']['verdict'] for e in events if 'verdict' in e.get('outputs', {})] == [1]


@fragment
def recovery_body():
    A: n = number(42)
    A(n) >> B(n)
    if (At[A].n == 42) @ B:
        B: verdict = number(1)
    else:
        B: verdict = number(-1)


@workflow
def recover_inherited():
    spawn(recovery_body, fresh='all')


@workflow
def recover_fresh():
    spawn(recovery_body, at=A, retain=(A,), fresh='all')


CRASH_DRIVER = r'''
import importlib.util, os, sys, threading
from zippergen import run_sqlite
from zippergen.role_runner import RoleRunner
spec = importlib.util.spec_from_file_location('cpl_fixture', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
wf = getattr(m, sys.argv[3])
phase, boundary = sys.argv[4:6]
kill = threading.local()
original = RoleRunner._commit_state
def commit(self, *args, **kwargs):
    if getattr(kill, 'now', False) and boundary == 'before':
        os._exit(73)
    result = original(self, *args, **kwargs)
    if getattr(kill, 'now', False):
        os._exit(73)
    return result
RoleRunner._commit_state = commit
def trace(event):
    role = event.get('lifeline') or event.get('to')
    kind = event['type']
    kill.now = ((phase == 'act' and kind == 'act' and role == m.A.name)
                or (phase == 'recv' and kind == 'recv' and role == m.B.name)
                or (phase == 'decision' and kind == 'decision')
                or (phase == 'fork' and kind == 'fork')
                or (phase == 'init' and kind == 'init_send'))
run_sqlite(wf, store_path=sys.argv[2], trace=trace, timeout=10)
'''


@pytest.mark.parametrize('boundary', ['before', 'after'])
@pytest.mark.parametrize('wf,phase', [(recover_inherited, 'act'), (recover_inherited, 'recv'),
                                     (recover_inherited, 'fork'), (recover_fresh, 'init'),
                                     (recover_fresh, 'recv'), (recover_fresh, 'decision')])
def test_fresh_process_restores_session_monitor(tmp_path, wf, phase, boundary):
    path = str(tmp_path / 'crash.sqlite')
    killed = subprocess.run([sys.executable, '-c', CRASH_DRIVER, str(Path(__file__)), path,
                             wf.name, phase, boundary], capture_output=True, text=True, timeout=15)
    assert killed.returncode == 73, killed.stderr
    run_sqlite(wf, store_path=path, timeout=5)
    from zippergen.store import list_history
    with closing(open_store(path)) as conn:
        history = list_history(conn)
        verdicts = [e['event']['outputs']['verdict'] for e in history
                    if 'verdict' in e['event'].get('outputs', {}) and e['event']['type'] == 'act']
        assert verdicts == [1]
        assert conn.execute('SELECT COUNT(*) FROM outstanding_messages').fetchone()[0] == 0


from zippergen import Pool
jobs = Pool('scope_test')


@workflow
def unsupported_pool():
    with spawn():
        A: claim = jobs.try_claim()
        if true() @ A:
            A: verdict = number(1)


def test_cpl_pool_scope_is_explicitly_unsupported():
    with pytest.raises(ValueError, match='pool causal handoffs'):
        project(unsupported_pool, A)


@human(kind='confirm', outputs=['flag: bool'], context='Review {n}')
def approve(n: int) -> bool: ...


human_seen = P(atom(lambda env, event: event.kind == 'act' and env.get('flag') is True,
                   version='human-true-act-v1'))


@fragment
def human_review():
    A(n) >> B(n)
    B: flag = approve(n)
    if human_seen @ B:
        B: verdict = number(1)
    else:
        B: verdict = number(0)


@workflow
def human_reviews():
    while (n < 3) @ A:
        A: n = inc(n)
        spawn(human_review, at=A, retain=(A,), fresh='all')


def test_pending_humans_observe_answer_once_per_session(tmp_path):
    import threading
    import time
    from zippergen.sqlite_runner import LocalSupervisor
    from zippergen.store import complete_human_task, list_history
    path = str(tmp_path / 'humans.sqlite')
    supervisor = LocalSupervisor(human_reviews, None, {}, store_path=path, timeout=10)
    failures = []
    def worker():
        try:
            supervisor.run()
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=worker)
    thread.start()
    rows = []
    try:
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            with closing(open_store(path)) as conn:
                rows = conn.execute('SELECT task_id,role FROM human_tasks ORDER BY created_at').fetchall()
            if len(rows) == 3:
                break
            time.sleep(.02)
        assert len(rows) == 3
        # Resume one approval at a time, including a rejection after an acceptance.
        with closing(open_store(path)) as conn:
            for index, (task, _) in enumerate(reversed(rows)):
                complete_human_task(conn, task, {'flag': index != 1})
        thread.join(6)
        assert not thread.is_alive()
        assert not failures, failures
    finally:
        supervisor.stop.set()
        thread.join(3)
    with closing(open_store(path)) as conn:
        events = [row['event'] for row in list_history(conn)]
    verdicts = [e['outputs']['verdict'] for e in events if 'verdict' in e.get('outputs', {})]
    assert sorted(verdicts) == [0, 1, 1]
    decisions = [e for e in events if e.get('formula') and e.get('lifeline') == B.name]
    assert len(decisions) == 3
    assert all(e['vc'][B.name] == 3 for e in decisions)  # recv, human completion, choice


def test_duplicate_activation_does_not_reset_cpl_history():
    from zippergen.runtime import mock_llm
    wf = recover_fresh
    templates = project_templates(wf)
    plans = compile_session_cpl(wf)
    channel = InProcessChannel()
    root = SessionState.start(project(wf, A), role=A.name, endpoint=A.name,
                              templates=templates, root_bindings={A.name: A.name}, keep_alive=True)
    # Advance parent until it has emitted initialization.
    for _ in range(8):
        root.step({}, channel, wf.ns, mock_llm, None, None, None, cpl_plans=plans)
        if channel.pending_initialization_endpoints():
            break
    endpoint = next(iter(channel.pending_initialization_endpoints()))
    child = SessionState.start(EmptyStmt(), endpoint=endpoint, templates=templates, keep_alive=True)
    assert child.accept_invitation(channel)
    # Parent's child must send its evidence to the freshly activated reviewer.
    env = {}
    for _ in range(8):
        root.step(env, channel, wf.ns, mock_llm, None, None, None, cpl_plans=plans)
    child.step({}, channel, wf.ns, mock_llm, None, None, None, cpl_plans=plans)
    session = next(iter(child.sessions.values()))
    assert session.monitor is not None
    before = session.monitor.snapshot_state()
    descriptor = next(iter(child.activations.values()))
    channel.put(A.name, endpoint, 'init:invite:retry', (descriptor,))
    assert child.accept_invitation(channel)
    assert session.monitor.snapshot_state() == before


@pytest.mark.parametrize('wf', [recover_inherited, recover_fresh])
def test_deployment_and_runtime_use_same_monitor_identity(wf, tmp_path):
    from dataclasses import replace
    from zippergen.deployment_checks import _workflow_state_check
    from zippergen.sqlite_runner import LocalSupervisor
    from zippergen.store import WorkflowIdentityError
    path = tmp_path / 'identity.sqlite'
    run_sqlite(wf, store_path=str(path), timeout=5)
    assert _workflow_state_check(wf, path)['status'] == 'ok'
    supervisor = LocalSupervisor(wf, None, {}, store_path=str(path))
    key = next(key for key, plan in supervisor.cpl_plans.items() if plan.formulas)
    supervisor.cpl_plans[key] = replace(supervisor.cpl_plans[key], layout='changed-policy')
    with pytest.raises(WorkflowIdentityError):
        supervisor.run()
