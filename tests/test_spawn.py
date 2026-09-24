"""Experimental spawn: behavior, projection, and durable session boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from contextlib import closing

import pytest

from zippergen import (
    Json, Lifeline, Var, workflow, fragment, pure, human, spawn, parallel, branch,
    run_sqlite, project, EmptyStmt, MsgStmt, VarExpr, SpawnStmt, ForkStmt, Workflow,
    seq, run, effect,
)
from zippergen.control import program_fingerprint
from zippergen.store import open_store, load_role_state, complete_human_task, list_history
from zippergen.sqlite_runner import LocalSupervisor
from zippergen.availability import AvailabilityViolation
from zippergen.locator import _children
from zippergen.formula import atom

A, B, C = Lifeline('A'), Lifeline('B'), Lifeline('C')
i = Var('i', int, default=0)
total = Var('total', int, default=0)
values = Var('values', Json, default=[])
item = Var('item', int)
answer = Var('answer', int)
approved = Var('approved', bool)
spawn_guard = atom(lambda env: True, version='spawn-test')


@pure
def increment(x: int) -> int:
    return x + 1


@pure
def double(x: int) -> int:
    return x * 2


@pure
def add(x: int, y: int) -> int:
    return x + y


@pure
def append(xs: Json, x: int) -> Json:
    return [*xs, x]


@human(kind='confirm', outputs=['approved: bool'], context='Request {item}')
def approve(item: int) -> bool: ...


@fragment
def exchange():
    A(item) >> B(item)
    B: answer = double(item)
    B(answer) >> A(answer)
    A: total = add(total, answer)


release_child = threading.Event()


@effect
def await_release() -> int:
    if not release_child.wait(3):
        raise RuntimeError('Parent did not continue past parallel')
    return 7


@effect
def release(value: int) -> int:
    release_child.set()
    return value


@workflow
def repeated(count: int @ A):
    while (i < count) @ A:
        A: item = increment(i)
        with spawn(fresh=(A.item, A.answer, B.item, B.answer)):
            exchange()
        A: i = increment(i)
    return total @ A


@workflow
def approvals(count: int @ A):
    while (i < count) @ A:
        A: item = increment(i)
        with spawn(fresh=(A.item, B.item, B.approved)):
            A(item) >> B(item)
            B: approved = approve(item)
            if approved @ B:
                B: values = append(values, item)
        A: i = increment(i)
    return values @ B


@workflow
def nested():
    A: item = double(2)
    with spawn(fresh=(A.item,)):
        A: item = increment(item)
        with spawn(fresh=(A.approved, A.answer)):
            A: approved = approve(item)
            if approved @ A:
                A: answer = double(item)
                A: total = add(total, answer)
    A: item = double(50)
    return total @ A


@workflow
def conditional(enabled: bool @ A):
    if enabled @ A:
        with spawn(fresh=(A.item, B.item)):
            A: item = double(2)
            A(item) >> B(item)
            B: total = add(total, item)
    return total @ B


@workflow
def unsupported_cpl():
    with spawn():
        if spawn_guard @ A:
            A: total = increment(total)


@workflow
def human_parallel():
    with spawn(fresh=(A.approved, A.item)):
        with parallel:
            with branch:
                A: approved = approve(1)
            with branch:
                A: item = release(7)
        if approved @ A:
            A: total = add(total, item)
    return total @ A


@workflow
def escape():
    with parallel:
        with branch:
            with spawn(fresh=(A.item, B.item)):
                B: item = await_release()
                B(item) >> A(item)
        with branch:
            A: total = increment(total)
    A: total = release(total)
    return total @ A


def nodes(stmt):
    yield stmt
    for c in _children(stmt):
        yield from nodes(c)


@pytest.mark.parametrize('runner', [run_sqlite, run])
def test_repeated_spawn_snapshots_inputs_and_shares_accumulator(runner):
    events = []
    assert runner(repeated, [A, B], {'A': {'count': 4}}, trace=events.append, timeout=5) == 20
    forks = [e for e in events if e['type'] == 'fork']
    assert len(forks) == 8
    a_names = [e['child_session'] for e in forks if e['lifeline'] == 'A']
    b_names = [e['child_session'] for e in forks if e['lifeline'] == 'B']
    assert a_names == b_names and len(set(a_names)) == 4
    assert sorted(e['inputs']['item'] for e in events if e.get('action') == 'double' and e['type'] == 'act') == [1, 2, 3, 4]


def test_projection_contains_forks_at_child_only_role():
    assert project(repeated, C) == EmptyStmt()
    for participant in (A, B):
        forks = [n for n in nodes(project(repeated, participant)) if isinstance(n, ForkStmt)]
        assert len(forks) == 1
        assert forks[0].fresh == ('answer', 'item')
    assert 'while recv_decision' in __import__('zippergen.view', fromlist=['render_workflow']).render_workflow(
        repeated, options=__import__('zippergen.view', fromlist=['ViewOptions']).ViewOptions(agent='B'))


def test_local_communication_view_omits_action_only_fork():
    import ast
    from zippergen.view import ViewOptions, render_workflow

    @workflow
    def local_child(item: int @ A):
        with spawn(fresh=(A.answer,)):
            A: answer = double(item)
        A(item) >> B(item)

    for communications_only in (False, True):
        code = render_workflow(local_child, options=ViewOptions(
            agent='A', communications_only=communications_only))
        ast.parse(code)
        assert "send('B', item)" in code
        assert ('with fork_session(' in code) is not communications_only


def test_local_communication_view_retains_fork_with_messages():
    import ast
    from zippergen.view import ViewOptions, render_workflow
    for participant in ('A', 'B'):
        code = render_workflow(repeated, options=ViewOptions(
            agent=participant, communications_only=True))
        ast.parse(code)
        assert 'with fork_session(' in code
        assert 'double(' not in code


@pytest.mark.parametrize('selection, expected', [
    ((B.item, B.answer), False),
    ((B.item,), True),
    ((B.item, A.answer), True),
])
def test_validation_warns_about_participant_qualified_shared_writes(selection, expected):
    from types import ModuleType
    from zippergen.syntax import ActStmt, LitExpr
    from zippergen.validation import validate_workflow
    body = seq(MsgStmt(A, (LitExpr(7, int),), B, (VarExpr(item),)),
               ActStmt(B, double, (VarExpr(item),), (answer,)))
    wf = Workflow('shared_writes', (), None, (), SpawnStmt(body, selection))
    report = validate_workflow(wf, ModuleType('shared_writes'))
    assert report['valid']
    warnings = [c for c in report['checks'] if c['status'] == 'warn']
    assert bool(warnings) is expected
    if expected:
        assert 'B.answer' in warnings[0]['detail']
        assert 'B.item' not in warnings[0]['detail']


def test_sharing_warnings_include_receives_but_respect_nested_freshening():
    from zippergen.syntax import ActStmt, LitExpr
    from zippergen.validation import _spawn_sharing_checks
    body = seq(MsgStmt(A, (LitExpr(7, int),), B, (VarExpr(item),)),
               ActStmt(B, double, (VarExpr(item),), (answer,)))
    private_child = SpawnStmt(body, (B.item, B.answer))
    wf = Workflow('nested_private', (), None, (), SpawnStmt(private_child))
    assert _spawn_sharing_checks(wf) == []
    inherited_child = SpawnStmt(body, (B.answer,))
    wf = Workflow('nested_inherited', (), None, (), SpawnStmt(inherited_child, (B.item,)))
    warnings = _spawn_sharing_checks(wf)
    assert len(warnings) == 1
    assert warnings[0]['name'] == 'spawn shared writes at 0.0'
    assert 'B.item' in warnings[0]['detail']


def test_intentional_shared_accumulator_is_a_warning_not_an_error():
    from types import ModuleType
    from zippergen.validation import validate_workflow
    report = validate_workflow(repeated, ModuleType('repeated'))
    assert report['valid']
    warnings = [c for c in report['checks'] if c['status'] == 'warn']
    assert len(warnings) == 1
    assert 'A.total' in warnings[0]['detail']


def test_absent_fresh_value_does_not_fall_back_to_parent():
    from zippergen.sessions import CellEnv
    env = CellEnv({'item': 7}, {'child': {}}, {'item': 'child'})
    assert 'item' not in env
    assert list(env) == []
    with pytest.raises(KeyError):
        env['item']


@pytest.mark.parametrize('operation', [
    lambda env: 'item' in env,
    lambda env: env.get('item'),
    lambda env: list(env),
    lambda env: env.__setitem__('item', 9),
])
def test_missing_cell_allocation_is_an_invariant_error(operation):
    from zippergen.sessions import CellEnv
    env = CellEnv({'item': 7}, {}, {'item': 'missing-child'})
    with pytest.raises(RuntimeError, match='Missing cell allocation'):
        operation(env)


@pytest.mark.parametrize('monitor_args', [
    {'monitor': object()}, {'formula_conditions': {1: spawn_guard}},
])
def test_session_scheduler_refuses_endpoint_wide_monitoring(monitor_args):
    from zippergen.sessions import SessionState
    state = SessionState.start(EmptyStmt())
    with pytest.raises(ValueError, match='Endpoint-wide CPL monitors'):
        state.step({}, None, {}, None, None, None, None, **monitor_args)


@pytest.mark.parametrize('runner', [run_sqlite, run])
def test_child_escapes_parallel_local_join(runner):
    # A waits in the child for a message that B cannot send before the parent
    # has escaped the enclosing parallel join. A joining implementation hangs.
    release_child.clear()
    events = []
    assert runner(escape, [A, B], {}, timeout=5, trace=events.append) == 1
    assert any(e['type'] == 'recv' and e.get('session') != '0' for e in events)


def test_identical_spawn_occurrences_have_distinct_static_ids():
    body = SpawnStmt(MsgStmt(A, (VarExpr(item),), B, (VarExpr(item),)), (A.item, B.item))
    wf = Workflow('twice', (), None, (), seq(body, body), {})
    for participant in (A, B):
        forks = [n for n in nodes(project(wf, participant)) if isinstance(n, ForkStmt)]
        assert len({n.site for n in forks}) == 2
    assert [n.site for n in nodes(project(wf,A)) if isinstance(n,ForkStmt)] == [n.site for n in nodes(project(wf,B)) if isinstance(n,ForkStmt)]


def test_fresh_fields_affect_durable_identity():
    body = MsgStmt(A, (VarExpr(item),), B, (VarExpr(item),))
    def fingerprint(fresh):
        wf = Workflow('identity', (), None, (), SpawnStmt(body, fresh), {})
        return program_fingerprint({a.name: project(wf,a) for a in (A,B)})
    assert fingerprint((A.item,)) != fingerprint((A.item,B.item))


def test_child_writes_do_not_initialize_parent_return():
    @workflow
    def unavailable():
        with spawn(fresh=(A.answer,)):
            A: answer = double(1)
        return answer @ A
    with pytest.raises(AvailabilityViolation):
        run_sqlite(unavailable, timeout=2)


def test_spawn_validation_rejects_bad_fresh_fields():
    body = MsgStmt(A, (VarExpr(item),), B, (VarExpr(item),))
    with pytest.raises(ValueError, match='absent'):
        SpawnStmt(body, (C.item,))
    bad = Workflow('bad_fresh', (), None, (), SpawnStmt(body, (A.typo,)))
    with pytest.raises(ValueError, match='undeclared'):
        project(bad, A)
    with pytest.raises(ValueError, match='duplicate'):
        SpawnStmt(body, (A.item,A.item))


@pytest.mark.parametrize('runner', [run_sqlite, run])
def test_zero_visits_and_conditional_activation(runner):
    assert runner(repeated, [A, B], {'A': {'count': 0}}, timeout=3) == 0
    for enabled in (True, False):
        events = []
        assert runner(conditional, [A, B], {'A': {'enabled': enabled}}, timeout=3,
                      trace=events.append) == (4 if enabled else 0)
        assert len([e for e in events if e['type'] == 'fork']) == (2 if enabled else 0)


def test_spawn_accepts_session_scoped_cpl():
    assert isinstance(project(unsupported_cpl, A), ForkStmt)


def _await(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError('Timed out waiting for workflow state')


def test_pending_humans_do_not_block_peers_and_resume_out_of_order(tmp_path):
    path = str(tmp_path/'run.sqlite')
    errors = []
    runner = LocalSupervisor(approvals, None, {'A': {'count': 3}}, store_path=path, timeout=0)
    def execute():
        try:
            runner.run()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    def tasks():
        with closing(open_store(path)) as db:
            rows = db.execute('SELECT task_id,spec FROM human_tasks ORDER BY created_at').fetchall()
            return rows if len(rows) == 3 else None
    try:
        rows = _await(tasks)
        with closing(open_store(path)) as db:
            complete_human_task(db, rows[2][0], {'approved': True})
        def third_finished():
            with closing(open_store(path)) as db:
                state = load_role_state(db, 'B')
                return state and state['env'].get('values') == [3]
        _await(third_finished)
        # Leave first pending, complete the second with rejection.
        with closing(open_store(path)) as db:
            complete_human_task(db, rows[1][0], {'approved': False})
        def rejection_consumed():
            with closing(open_store(path)) as db:
                state = load_role_state(db, 'B')
                return state and len(state['control'].get('sessions', [])) == 2
        _await(rejection_consumed)
    finally:
        runner.stop.set()
        thread.join(3)
    assert not thread.is_alive()
    with closing(open_store(path)) as db:
        complete_human_task(db, rows[0][0], {'approved': True})
    assert run_sqlite(approvals, store_path=path, initial_envs={'A': {'count': 999}}, timeout=3) == [3,1]
    with closing(open_store(path)) as db:
        assert db.execute('SELECT count(*) FROM human_tasks').fetchone()[0] == 3
        assert db.execute('SELECT count(*) FROM outstanding_messages').fetchone()[0] == 0


CRASH_RUNNER = '''
import json, os, sys
from zippergen.workflow_io import load_workflow_spec
from zippergen.role_runner import RoleRunner
from zippergen.sessions import SessionState
from zippergen import run_sqlite

source, database, phase = sys.argv[1:]
wf, _ = load_workflow_spec(source + ':repeated')
commit = RoleRunner._commit_state
def boundary(self, residual, *args, **kwargs):
    target = (self.role == 'A' and isinstance(residual, SessionState)
              and len(residual.sessions) > 1)
    if target and phase == 'before':
        os._exit(99)
    commit(self, residual, *args, **kwargs)
    if target and phase == 'after':
        os._exit(99)
RoleRunner._commit_state = boundary
print(json.dumps(run_sqlite(wf, initial_envs={'A': {'count': 4}},
                           store_path=database, timeout=5)))
'''


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_fork_is_atomic_across_process_death(tmp_path, phase):
    database = str(tmp_path / 'run.sqlite')
    def process(mode):
        return subprocess.run([sys.executable, '-c', CRASH_RUNNER, __file__, database, mode],
                              capture_output=True, text=True, timeout=12)
    crashed = process(phase)
    assert crashed.returncode == 99, crashed.stdout + crashed.stderr
    resumed = process('resume')
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads(resumed.stdout.strip()) == 20
    with closing(open_store(database)) as db:
        events = [row['event'] for row in list_history(db)]
        for role in ('A', 'B'):
            forks = [e for e in events if e['type'] == 'fork' and e['lifeline'] == role]
            assert len(forks) == 4
            assert len({e['child_session'] for e in forks}) == 4
        assert db.execute('SELECT count(*) FROM outstanding_messages').fetchone()[0] == 0


def test_descendant_keeps_inherited_cell_after_parent_finishes(tmp_path):
    database = str(tmp_path / 'nested.sqlite')
    script = '''
import os, sys
from zippergen.workflow_io import load_workflow_spec
from zippergen.role_runner import RoleRunner
from zippergen.sessions import SessionState
from zippergen import run_sqlite, EmptyStmt
wf, _ = load_workflow_spec(sys.argv[1] + ':nested')
commit = RoleRunner._commit_state
def boundary(self, residual, *args, **kwargs):
    commit(self, residual, *args, **kwargs)
    if (isinstance(residual, SessionState)
        and isinstance(residual.sessions['0'].residual, EmptyStmt)
        and len(residual.sessions) == 2
        and any(s.waiting for s in residual.sessions.values())):
        os._exit(99)
RoleRunner._commit_state = boundary
run_sqlite(wf, store_path=sys.argv[2], timeout=5)
'''
    result = subprocess.run([sys.executable, '-c', script, __file__, database],
                            capture_output=True, text=True, timeout=12)
    assert result.returncode == 99, result.stdout + result.stderr
    with closing(open_store(database)) as db:
        state = load_role_state(db, 'A')
        assert state['env']['item'] == 100
        tasks = db.execute('SELECT task_id FROM human_tasks').fetchall()
        assert len(tasks) == 1
        complete_human_task(db, tasks[0][0], {'approved': True})
    assert run_sqlite(nested, store_path=database, timeout=5) == 10


def test_pending_human_allows_an_external_parallel_branch(tmp_path):
    database = str(tmp_path / 'parallel.sqlite')
    runner = LocalSupervisor(human_parallel, None, None, store_path=database, timeout=4)
    results = []
    errors = []
    def execute():
        try:
            results.append(runner.run())
        except BaseException as exc:
            errors.append(exc)
    release_child.clear()
    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert release_child.wait(2)
        with closing(open_store(database)) as db:
            tasks = db.execute('SELECT task_id FROM human_tasks').fetchall()
            assert len(tasks) == 1
            complete_human_task(db, tasks[0][0], {'approved': True})
        thread.join(3)
        assert not thread.is_alive()
        assert errors == []
        assert results == [7]
    finally:
        runner.stop.set()
        thread.join(3)


@pytest.mark.parametrize('runner', [run_sqlite, run])
def test_example_keeps_request_and_approval_correlated(runner):
    from zippergen.workflow_io import load_workflow_spec
    source = Path(__file__).resolve().parents[1] / 'examples/spawn/workflow.py'
    wf, module = load_workflow_spec(str(source) + ':draft_sessions')
    seen = []
    def approve_example(action, inputs):
        seen.append(dict(inputs))
        return {'approved': inputs['request'] != 'Review the prototype'}
    result = runner(wf, [module.Mailbox, module.Writer, module.User], {},
                    human_backend=approve_example, timeout=4)
    assert len(seen) == 3
    assert {entry['request'] for entry in result} == {'Discuss the paper', 'Plan the next meeting'}
    for entry in result:
        assert entry['request'] in entry['draft']


def test_fresh_field_can_have_a_lifeline_attribute_name():
    @workflow
    def named():
        A: name = double(4)
        with spawn(fresh=(A.name,)):
            A: name = increment(name)
            A: total = add(total, name)
        return total @ A
    assert run_sqlite(named, timeout=3) == 9


def test_fresh_field_used_only_in_a_guard():
    @workflow
    def guarded(enabled: bool @ A):
        with spawn(fresh=(A.enabled,)):
            if enabled @ A:
                A: total = increment(total)
        return total @ A
    for enabled in (True, False):
        assert run_sqlite(guarded, initial_envs={'A': {'enabled': enabled}}, timeout=3) == int(enabled)


def test_functional_spawn_accepts_an_existing_fragment():
    @workflow
    def functional():
        A: item = double(2)
        spawn(exchange, fresh=(A.item, A.answer, B.item, B.answer))
        return total @ A
    assert run_sqlite(functional, timeout=3) == 8


def test_durable_cli_lists_all_approvals_without_prompting(tmp_path):
    import shutil
    repository = Path(__file__).resolve().parents[1]
    for filename in ('workflow.py', 'specification.md', 'zippergen.toml'):
        shutil.copyfile(repository / 'examples/spawn' / filename, tmp_path / filename)
    environment = dict(os.environ, ZIPPERGEN_HOME=str(tmp_path / 'site'),
                       PYTHONPATH=str(repository / 'src'))
    command = [sys.executable, '-m', 'zippergen.serve']
    process = subprocess.Popen(command + ['run', '--durable', '--yes', '--timeout', '15'],
                               cwd=tmp_path, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    def ready():
        databases = list((tmp_path / 'site').rglob('*.sqlite'))
        if not databases:
            return False
        with closing(open_store(str(databases[0]))) as db:
            return db.execute('SELECT count(*) FROM human_tasks').fetchone()[0] == 3
    try:
        _await(ready, timeout=8)
        def cli(*args):
            result = subprocess.run(command + list(args), cwd=tmp_path, env=environment,
                                    capture_output=True, text=True, timeout=5)
            assert result.returncode == 0, result.stdout + result.stderr
            return result.stdout
        tasks = json.loads(cli('run', 'tasks', '--json'))
        assert len(tasks) == 3
        positions = json.loads(cli('run', 'inspect', '--json'))['positions']
        user = next(position for position in positions if position['participant'] == 'User')
        assert len(user['locators']) == 3
        for task in tasks:
            option = '--no' if task['inputs']['request'] == 'Review the prototype' else '--yes'
            cli('run', 'approve', '--task', task['task_id'], option)
        output, error = process.communicate(timeout=5)
        assert process.returncode == 0, output + error
        record = json.loads(cli('run', 'status', '--json'))
        assert record['run_status'] == 'done'
        assert record['durable_state']['done_human_task_count'] == 3
        assert len(record['durable_state']['workflow_results'][0]['value']) == 2
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


@pytest.mark.parametrize('deployed', [False, True])
def test_pool_spawn_humans_use_durable_queue_only_with_accessible_store(
    tmp_path, monkeypatch, capsys, deployed,
):
    from zippergen.serve import main, _run_args_from_deployment, _run_workflow_command
    from zippergen.workspace import Workspace
    source = tmp_path / 'workflow.py'
    source.write_text('''
from zippergen import Lifeline, Pool, Var, human, pure, spawn, workflow
A = Lifeline('A')
jobs = Pool('jobs')
i = Var('i', int, default=0)
@human(kind='confirm', outputs=['approved: bool'], context='Item {item}')
def approve(item: int) -> bool: ...
@pure
def increment(value: int) -> int:
    return value + 1
@workflow
def review():
    while (i < 3) @ A:
        with spawn(fresh=(A.i, A.job, A.approved)):
            A: job = jobs.put(i)
            A: approved = approve(i)
        A: i = increment(i)
    return i @ A
''')
    workspace = Workspace(tmp_path, home=tmp_path / 'home')
    workspace.initialize_project(name='spawn-pool')
    workspace.select_workflow('workflow.py:review')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ZIPPERGEN_HOME', str(workspace.home))
    prompts = []
    def answer(prompt):
        assert not deployed, 'A deployed workflow must not prompt in the terminal'
        prompts.append(prompt)
        return 'y'
    monkeypatch.setattr('builtins.input', answer)
    if not deployed:
        assert main(['run', '--yes', '--timeout', '5']) == 0
        assert len(prompts) == 3
    else:
        store = str(tmp_path / 'deployment.sqlite')
        open_store(store).close()
        stop = threading.Event()
        errors = []
        answered = []
        def complete_all():
            try:
                while not stop.wait(0.01):
                    with closing(open_store(store)) as db:
                        tasks = db.execute("SELECT task_id FROM human_tasks WHERE status='pending'").fetchall()
                        if len(tasks) != 3:
                            continue
                        for (task_id,) in tasks:
                            complete_human_task(db, task_id, {'approved': True})
                            answered.append(task_id)
                        return
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=complete_all)
        thread.start()
        try:
            args = _run_args_from_deployment({
                'workflow': str(source) + ':review', 'store': store, 'timeout': 5,
            })
            assert _run_workflow_command(args) == 0
            assert errors == []
            assert len(set(answered)) == 3
            assert not prompts
        finally:
            stop.set()
            thread.join(2)
    assert '"result": 3' in capsys.readouterr().out
