"""The isolation shorthand expands to ordinary, inspectable fresh fields."""
import pytest

from zippergen import (Lifeline, Var, workflow, fragment, spawn, pure, run_sqlite,
                       project, SpawnStmt, all_except)
from zippergen.control import program_fingerprint

A, B = Lifeline('A'), Lifeline('B')
x = Var('x', int, default=7)
y = Var('y', int)
flag = Var('flag', bool, default=True)
unused = Var('unused', int, default=99)


@pure
def twice(x: int) -> int:
    return x * 2


@fragment
def work():
    A(x) >> B(x)
    if flag @ B:
        B: y = twice(x)


@workflow
def inferred():
    with spawn(fresh='all'):
        work()
    return x @ A


@workflow
def explicit():
    with spawn(fresh=(A.x, B.flag, B.x, B.y)):
        work()
    return x @ A


def test_all_expands_to_exact_fields_and_same_projected_program():
    assert inferred.body.fresh == explicit.body.fresh
    assert type(inferred.body) is SpawnStmt
    assert all(field.name != 'unused' for field in inferred.body.fresh)
    # Workflow names are not relevant to the projected control fingerprint.
    assert program_fingerprint({p.name: project(inferred, p) for p in (A, B)}) == \
           program_fingerprint({p.name: project(explicit, p) for p in (A, B)})
    events = []
    assert run_sqlite(inferred, timeout=3, trace=events.append) == 7
    assert any(e['type'] == 'act' and e.get('outputs') == {'y': 14} for e in events)


def test_nested_and_fragment_form_keep_fresh_bindings():
    @workflow
    def nested():
        with spawn(fresh='all'):
            A: x = twice(x)
            spawn(work, fresh='all')
        return x @ A
    events = []
    assert run_sqlite(nested, timeout=3, trace=events.append) == 7
    assert any(e['type'] == 'act' and e.get('outputs') == {'y': 28} for e in events)
    assert {f.lifeline.name + '.' + f.name for f in nested.body.fresh} == {
        'A.x', 'B.x', 'B.y', 'B.flag'}


def test_guard_only_input_is_copied():
    @workflow
    def guard_input(enabled: bool @ A):
        with spawn(fresh='all'):
            if enabled @ A:
                A: y = twice(3)
    assert {f.name for f in guard_input.body.fresh} == {'enabled', 'y'}
    run_sqlite(guard_input, initial_envs={'A': {'enabled': True}}, timeout=3)


def test_bad_string_refuses_to_guess():
    with pytest.raises(ValueError, match='"all"'):
        @workflow
        def broken():
            with spawn(fresh='everything'):
                A: y = twice(x)


def test_unsupported_literal_assignment_is_not_silently_omitted():
    with pytest.raises(SyntaxError, match='must call a declared action'):
        @workflow
        def broken():
            A: y = True


def test_all_except_matches_explicit_fields_and_fragment_form():
    @workflow
    def excluding():
        spawn(work, fresh=all_except(A.x, B.flag))

    @workflow
    def listing():
        with spawn(fresh=(B.x, B.y)):
            work()

    assert type(excluding.body) is SpawnStmt
    assert excluding.body.fresh == listing.body.fresh
    assert program_fingerprint({p.name: project(excluding, p) for p in (A, B)}) == \
           program_fingerprint({p.name: project(listing, p) for p in (A, B)})


def test_empty_exclusion_is_all():
    @workflow
    def excluding_nothing():
        with spawn(fresh=all_except()):
            work()
    assert excluding_nothing.body.fresh == inferred.body.fresh


def test_nested_exclusion_inherits_parent_binding_not_root():
    @workflow
    def nested():
        with spawn(fresh='all'):
            A: x = twice(x)
            with spawn(fresh=all_except(A.x)):
                A: x = twice(x)
                A(x) >> B(x)
                B: y = twice(x)
        return x @ A

    events = []
    assert run_sqlite(nested, timeout=3, trace=events.append) == 7
    # The grandchild sees and updates the outer child's x=14, not root x=7.
    assert any(e['type'] == 'act' and e.get('outputs') == {'y': 56} for e in events)


@pytest.mark.parametrize('field', [A.unused, B.unused, A.flag])
def test_unused_or_wrong_participant_exclusion_is_rejected(field, monkeypatch):
    # flag occurs only at B. A.flag must not silently leave B.flag fresh.
    monkeypatch.setitem(globals(), 'excluded_selector', all_except(field))
    with pytest.raises(ValueError, match='not used by the spawned body'):
        @workflow
        def broken():
            spawn(work, fresh=excluded_selector)


def test_bad_exclusion_type_and_duplicates_are_rejected():
    with pytest.raises(TypeError, match='participant fields'):
        all_except('A.x')
    with pytest.raises(TypeError, match='participant fields'):
        all_except(x)
    with pytest.raises(ValueError, match='duplicate'):
        all_except(A.x, A.x)


def test_excluded_writes_still_have_sharing_warning():
    from zippergen.validation import _spawn_sharing_checks

    @workflow
    def shared_write():
        with spawn(fresh=all_except(A.x)):
            A: x = twice(x)

    checks = _spawn_sharing_checks(shared_write)
    assert len(checks) == 1 and checks[0]['status'] == 'warn'
    assert 'A.x' in checks[0]['detail']
