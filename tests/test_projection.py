"""Tests for Layer 4: projection engine (π_A).

IR nodes are constructed directly — no builder — so these tests are
independent of Layer 3 and test projection semantics in isolation.
"""

import pytest

from zippergen.syntax import (
    occurring_variables,
    EmptyStmt, MsgStmt, CoregionStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt,
    ParallelStmt,
    SendStmt, RecvStmt, ReceiveAnyStmt, IfRecvStmt, WhileRecvStmt, ParallelLocalStmt,
    Lifeline, Var, VarExpr, LitExpr,
    Workflow, seq, is_kappa_ctrl,
)
from zippergen.actions import pure
from zippergen.projection import project
from zippergen.builder import workflow


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

A = Lifeline("A")
B = Lifeline("B")
C = Lifeline("C")
D = Lifeline("D")

x = Var("x", int)
y = Var("y", int)
z = Var("z", int)


def _make_workflow(body, name="test") -> Workflow:
    return Workflow(
        name=name,
        inputs=(),
        output_type=str,
        vars=(),
        body=body,
        ns={},
    )


# ---------------------------------------------------------------------------
# Base cases
# ---------------------------------------------------------------------------

def test_project_empty():
    wf = _make_workflow(EmptyStmt())
    assert project(wf, A) == EmptyStmt()


def test_project_msg_sender():
    stmt = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    wf = _make_workflow(stmt)
    result = project(wf, A)
    assert isinstance(result, SendStmt)
    assert result.lifeline == A
    assert result.receiver == B
    assert result.payload == (VarExpr(x),)


def test_project_msg_receiver():
    stmt = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    wf = _make_workflow(stmt)
    result = project(wf, B)
    assert isinstance(result, RecvStmt)
    assert result.lifeline == B
    assert result.sender == A
    assert result.bindings == (VarExpr(y),)


def test_project_msg_bystander():
    stmt = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    wf = _make_workflow(stmt)
    assert project(wf, C) == EmptyStmt()


def test_project_coregion_sender_gets_send():
    stmt = CoregionStmt((
        MsgStmt(A, (VarExpr(x),), C, (VarExpr(y),)),
        MsgStmt(B, (VarExpr(z),), C, (VarExpr(z),)),
    ))
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, SendStmt)
    assert result.lifeline == A
    assert result.receiver == C
    assert result.payload == (VarExpr(x),)


def test_project_coregion_receiver_gets_receive_any():
    stmt = CoregionStmt((
        MsgStmt(A, (VarExpr(x),), C, (VarExpr(y),)),
        MsgStmt(B, (VarExpr(z),), C, (VarExpr(z),)),
    ))
    wf = _make_workflow(stmt)

    result = project(wf, C)
    assert isinstance(result, ReceiveAnyStmt)
    assert result.lifeline == C
    assert result.receives == ((A, (VarExpr(y),)), (B, (VarExpr(z),)))


def test_project_coregion_bystander():
    D = Lifeline("D")
    stmt = CoregionStmt((
        MsgStmt(A, (VarExpr(x),), C, (VarExpr(y),)),
        MsgStmt(B, (VarExpr(z),), C, (VarExpr(z),)),
    ))
    wf = _make_workflow(stmt)

    assert project(wf, D) == EmptyStmt()


def test_project_act_owner():
    @pure
    def f(v: int) -> int:
        return v
    stmt = ActStmt(A, f, (VarExpr(x),), (y,))
    wf = _make_workflow(stmt)
    result = project(wf, A)
    assert result is stmt  # frozen node reused unchanged


def test_project_act_nonowner():
    @pure
    def f(v: int) -> int:
        return v
    stmt = ActStmt(A, f, (VarExpr(x),), (y,))
    wf = _make_workflow(stmt)
    assert project(wf, B) == EmptyStmt()


def test_project_skip_owner():
    stmt = SkipStmt(A)
    wf = _make_workflow(stmt)
    assert project(wf, A) is stmt


def test_project_skip_nonowner():
    stmt = SkipStmt(A)
    wf = _make_workflow(stmt)
    assert project(wf, B) == EmptyStmt()


# ---------------------------------------------------------------------------
# Sequential composition
# ---------------------------------------------------------------------------

def test_project_seq_distributes():
    s1 = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    s2 = MsgStmt(B, (VarExpr(y),), A, (VarExpr(z),))
    wf = _make_workflow(SeqStmt(s1, s2))

    # A: send then recv
    result_A = project(wf, A)
    assert isinstance(result_A, SeqStmt)
    assert isinstance(result_A.first, SendStmt)
    assert isinstance(result_A.second, RecvStmt)

    # B: recv then send
    result_B = project(wf, B)
    assert isinstance(result_B, SeqStmt)
    assert isinstance(result_B.first, RecvStmt)
    assert isinstance(result_B.second, SendStmt)


def test_project_seq_epsilon_elimination():
    """If one side projects to ε, seq collapses to the other side."""
    s1 = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    s2 = SkipStmt(C)
    wf = _make_workflow(SeqStmt(s1, s2))

    # A is only in s1 — s2 projects to ε for A, result should be just SendStmt
    result = project(wf, A)
    assert isinstance(result, SendStmt)


# ---------------------------------------------------------------------------
# Parallel composition
# ---------------------------------------------------------------------------

def test_project_parallel_uses_branch_channels():
    stmt = ParallelStmt((
        MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),)),
        MsgStmt(A, (VarExpr(x),), C, (VarExpr(z),)),
    ))
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, ParallelLocalStmt)
    assert len(result.branches) == 2
    assert result.branch_indices == (0, 1)
    sends = [branch for branch in result.branches if isinstance(branch, SendStmt)]
    assert len(sends) == 2
    assert sends[0].channel != sends[1].channel
    assert all(send.channel != "main" for send in sends)


def test_project_parallel_shared_receiver_gets_local_parallel():
    stmt = ParallelStmt((
        MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),)),
        MsgStmt(C, (VarExpr(z),), B, (VarExpr(z),)),
    ))
    wf = _make_workflow(stmt)

    result = project(wf, B)
    assert isinstance(result, ParallelLocalStmt)
    assert len(result.branches) == 2
    assert result.branch_indices == (0, 1)
    assert all(isinstance(branch, RecvStmt) for branch in result.branches)


def test_project_parallel_preserves_global_branch_index_for_single_branch_participant():
    stmt = ParallelStmt((
        MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),)),
        MsgStmt(C, (VarExpr(z),), B, (VarExpr(z),)),
    ))
    wf = _make_workflow(stmt)

    result = project(wf, C)
    assert isinstance(result, ParallelLocalStmt)
    assert result.branch_indices == (1,)
    assert len(result.branches) == 1


def test_project_parallel_accepts_shared_reachability_cycle():
    """Under the filtered shuffle semantics, programs with cyclic SRG are
    admissible: only the cyclic shuffled executions are filtered out at the
    semantic level, not the program itself."""
    stmt = ParallelStmt((
        MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),)),
        MsgStmt(B, (VarExpr(y),), A, (VarExpr(x),)),
    ))
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, ParallelLocalStmt)
    assert len(result.branches) == 2


def test_project_parallel_allows_private_intra_branch_request_response_cycle():
    branch_with_private_cycle = seq(
        seq(
            MsgStmt(A, (VarExpr(x),), C, (VarExpr(y),)),
            MsgStmt(C, (VarExpr(y),), A, (VarExpr(x),)),
        ),
        MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),)),
    )
    stmt = ParallelStmt((
        branch_with_private_cycle,
        MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),)),
    ))
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, ParallelLocalStmt)
    assert len(result.branches) == 2


# ---------------------------------------------------------------------------
# If — owner
# ---------------------------------------------------------------------------

def test_project_if_owner_gets_ifstmt():
    """The owner of an if receives an IfStmt with control broadcasts prepended."""
    cond = lambda _e: True
    body_true = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(A)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, IfStmt)
    assert result.owner == A
    assert result.condition is cond


def test_project_if_owner_broadcasts_to_receivers():
    """Owner's true/false branches start with a SendStmt(kappa_ctrl^P) to each receiver."""
    cond = lambda _e: True
    # B and C both appear in branches → both are receivers
    body_true  = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(C)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, IfStmt)

    # true branch: broadcast True to B and C (sorted), then projected body
    true_branch = result.branch_true
    # Unwrap the SeqStmt chain to find the first SendStmt
    def _collect_seq(s):
        stmts = []
        while isinstance(s, SeqStmt):
            stmts.append(s.first)
            s = s.second
        stmts.append(s)
        return stmts

    true_stmts = _collect_seq(true_branch)
    ctrl_sends = [s for s in true_stmts if isinstance(s, SendStmt) and any(is_kappa_ctrl(e) for e in s.payload)]
    assert len(ctrl_sends) == 2  # one for B, one for C
    receivers = {s.receiver for s in ctrl_sends}
    assert receivers == {B, C}


# ---------------------------------------------------------------------------
# If — receiver
# ---------------------------------------------------------------------------

def test_project_if_receiver_gets_ifrecvstmt():
    cond = lambda _e: True
    body_true  = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(B)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    result = project(wf, B)
    assert isinstance(result, IfRecvStmt)
    assert result.sender == A
    assert result.lifeline == B


def test_project_if_receiver_fresh_ctrl_var():
    cond = lambda _e: True
    body_true  = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(B)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    result = project(wf, B)
    assert isinstance(result, IfRecvStmt)
    ctrl_var_expr = result.bindings[0]
    assert isinstance(ctrl_var_expr, VarExpr)
    assert ctrl_var_expr.var.name.startswith("_ctrl")
    assert ctrl_var_expr.var.type is bool


def test_project_if_receiver_nested_fresh_ctrl_names_differ():
    """Two nested ifs produce two distinct _ctrl variables."""
    cond = lambda _e: True
    inner = IfStmt(
        condition=cond, owner=A,
        branch_true=SkipStmt(B),
        branch_false=SkipStmt(B),
    )
    outer = IfStmt(
        condition=cond, owner=A,
        branch_true=inner,
        branch_false=SkipStmt(B),
    )
    wf = _make_workflow(outer)
    result = project(wf, B)
    assert isinstance(result, IfRecvStmt)
    outer_ctrl = result.bindings[0].var.name
    inner_result = result.branch_true
    assert isinstance(inner_result, IfRecvStmt)
    inner_ctrl = inner_result.bindings[0].var.name
    assert outer_ctrl != inner_ctrl


# ---------------------------------------------------------------------------
# If — bystander
# ---------------------------------------------------------------------------

def test_project_if_bystander():
    cond = lambda _e: True
    body_true  = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    body_false = SkipStmt(A)
    stmt = IfStmt(condition=cond, owner=A, branch_true=body_true, branch_false=body_false)
    wf = _make_workflow(stmt)

    # C does not appear in any branch and is not the owner
    assert project(wf, C) == EmptyStmt()


# ---------------------------------------------------------------------------
# While — owner / receiver / bystander
# ---------------------------------------------------------------------------

def test_project_while_owner_gets_whilestmt():
    cond = lambda _e: False
    body = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    exit_body = SkipStmt(A)
    stmt = WhileStmt(condition=cond, owner=A, body=body, exit_body=exit_body)
    wf = _make_workflow(stmt)

    result = project(wf, A)
    assert isinstance(result, WhileStmt)
    assert result.owner == A


def test_project_while_receiver_gets_whilerecvstmt():
    cond = lambda _e: False
    body = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    exit_body = EmptyStmt()
    stmt = WhileStmt(condition=cond, owner=A, body=body, exit_body=exit_body)
    wf = _make_workflow(stmt)

    result = project(wf, B)
    assert isinstance(result, WhileRecvStmt)
    assert result.sender == A
    assert result.lifeline == B


def test_project_while_bystander():
    cond = lambda _e: False
    body = MsgStmt(A, (VarExpr(x),), B, (VarExpr(y),))
    exit_body = EmptyStmt()
    stmt = WhileStmt(condition=cond, owner=A, body=body, exit_body=exit_body)
    wf = _make_workflow(stmt)

    assert project(wf, C) == EmptyStmt()


# ---------------------------------------------------------------------------
# Freshness of generated control variables
# ---------------------------------------------------------------------------

def test_generated_control_variables_avoid_names_the_author_used():
    """The paper requires each generated variable not to occur in P.

    Counting alone is not freshness. An author may write ``_ctrl1``; binding
    the received branch decision over it changed the receiver's own value and
    branched on the wrong one, with no error raised anywhere.
    """

    from zippergen.syntax import occurring_variables
    from zippergen.projection import _Context

    occupied = frozenset({"_ctrl1", "_ctrl2", "_ctrl4"})
    allocator = _Context(occupied)
    issued = [allocator.fresh().name for _ in range(3)]

    assert issued == ["_ctrl3", "_ctrl5", "_ctrl6"]
    assert not set(issued) & occupied


# `@workflow` reads the defining source, so these live at module level --
# and each Python name must equal its Var name, because a condition is
# resolved by the name as written in the source.

_A = Lifeline("A")
_B = Lifeline("B")
_ctrl1 = Var("_ctrl1", bool)
go = Var("go", bool)
out = Var("out", str)


@pure
def _decide_true() -> bool:
    return True


@pure
def _carry_false() -> bool:
    return False


@pure
def _report_ctrl(_ctrl1: bool) -> str:
    return f"_ctrl1={_ctrl1}"


@workflow
def _collides() -> str:
    _A: go = _decide_true()
    _B: _ctrl1 = _carry_false()
    if go @ _A:
        _A(go) >> _B(go)
    _B: out = _report_ctrl(_ctrl1)
    return out @ _B


def test_a_user_variable_named_like_a_control_variable_survives_projection():
    """End to end: the receiver must read back the value it wrote."""

    assert "_ctrl1" in occurring_variables(_collides.body)
    assert _collides() == "_ctrl1=False"


sent = Var("sent", str)
got = Var("got", str)
decided = Var("decided", bool)
looped = Var("looped", str)


@pure
def _make_x() -> str:
    return "x"


@pure
def _decide_false() -> bool:
    return False


@pure
def _touch(got: str) -> str:
    return got


@workflow
def _many() -> str:
    _A: sent = _make_x()
    _A: decided = _decide_false()
    _A(sent) >> _B(got)
    while decided @ _A:
        _A: decided = _decide_false()
    else:
        _B: looped = _touch(got)
    return looped @ _B


def test_every_variable_the_author_wrote_is_counted_as_occupied():
    """A name missed by the collector is a name a control variable can land on."""

    names = occurring_variables(_many.body)
    for expected in ("sent", "got", "decided", "looped"):
        assert expected in names, f"{expected} was not counted as occupied"


# ---------------------------------------------------------------------------
# The reserved control namespace
# ---------------------------------------------------------------------------

def test_no_user_value_boundary_can_produce_a_control_value():
    """Disjointness is structural, not a spelling convention.

    The paper needs user payloads and control payloads to be distinguishable.
    A reserved string prefix could not deliver that: every boundary that
    admits a string admits the prefix too, so a workflow input or an action
    output could always forge one. `ControlTag` is a distinct type that no
    declarable coordination type matches.
    """

    from zippergen.builder import _to_expr
    from zippergen.syntax import is_control_value, validate_zvalue
    from zippergen.value_codec import ControlTag

    forgery = str(ControlTag("anything"))

    # The literal boundary: a string stays a string.
    assert not is_control_value(_to_expr(forgery).value)

    # The runtime boundary, for every declarable coordination type.
    for declared in (str, int, float, bool):
        with pytest.raises(TypeError):
            validate_zvalue(ControlTag("x"), declared)

    # And the forged string is admitted as the ordinary string it is.
    assert validate_zvalue(forgery, str) == forgery
    assert not is_control_value(forgery)


def test_a_control_value_survives_the_durable_boundary_as_itself():
    """A stored control message must not decode into something forgeable."""

    from zippergen.syntax import is_control_value
    from zippergen.value_codec import ControlTag, dumps_value, loads_value

    tag = ControlTag("construct-digest")
    assert loads_value(dumps_value(tag)) == tag
    assert is_control_value(loads_value(dumps_value(tag)))

    # The same text, sent by a workflow, comes back a plain string.
    text = str(tag)
    assert loads_value(dumps_value(text)) == text
    assert not is_control_value(loads_value(dumps_value(text)))


def test_no_module_classifies_a_control_value_by_its_spelling():
    """One question, asked of the type; never of the characters."""

    import pathlib
    import re

    source_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    # Any comparison against the display prefix, in either quoting style.
    spelling = re.compile(r"(?:startswith|==|in)\s*\(?\s*[\"']\u03ba_ctrl")
    offenders = [
        path.name
        for path in source_root.rglob("*.py")
        if spelling.search(path.read_text())
    ]
    assert not offenders, (
        "these modules classify a control value by its spelling instead of "
        f"asking is_control_value: {offenders}"
    )


# ---------------------------------------------------------------------------
# Completeness: every global construct must have a projection rule
# ---------------------------------------------------------------------------

def _one_of_each_global_statement():
    """A minimal instance of every member of the global `Stmt` union."""

    A = Lifeline("A")
    B = Lifeline("B")
    v = Var("v", int)
    msg = MsgStmt(A, (VarExpr(v),), B, (VarExpr(v),))
    return {
        EmptyStmt: EmptyStmt(),
        MsgStmt: msg,
        CoregionStmt: CoregionStmt((msg,)),
        ActStmt: ActStmt(A, _make_x, (), (Var("q", str),)),
        SkipStmt: SkipStmt(A),
        SeqStmt: SeqStmt(msg, EmptyStmt()),
        IfStmt: IfStmt(lambda _e: True, A, msg, EmptyStmt()),
        WhileStmt: WhileStmt(lambda _e: False, A, msg, EmptyStmt()),
        ParallelStmt: ParallelStmt((msg, EmptyStmt())),
    }


def test_every_global_construct_has_a_projection_rule():
    """A construct with no rule fails only when someone projects it.

    `_project` takes AnyStmt, because SeqStmt/IfStmt/WhileStmt are shared by
    global and local programs and carry AnyStmt children. That means pyright
    cannot force a new global member to gain a rule -- a contributor can add a
    construct, wire the builder and the runtime, and leave projection out with
    every check still green. This test is what forces it.
    """

    import typing

    from zippergen.projection import _Context, _project
    from zippergen.syntax import Stmt

    declared = {member for member in typing.get_args(Stmt)}
    covered = _one_of_each_global_statement()
    missing = declared - set(covered)
    assert not missing, (
        "the global Stmt union gained member(s) this test does not build, so "
        f"nothing checks that projection handles them: {sorted(m.__name__ for m in missing)}"
    )

    for member, instance in covered.items():
        for lifeline in (Lifeline("A"), Lifeline("B"), Lifeline("C")):
            try:
                _project(instance, lifeline, _Context(frozenset()))
            except TypeError as exc:
                raise AssertionError(
                    f"{member.__name__} has no projection rule: {exc}"
                ) from exc


def test_participation_is_analysed_once_per_node_not_once_per_query():
    """The paper's analysis is syntax-directed and linear; so is this one.

    Projection asks for the participants of both continuations at every
    `if`/`while`, and each ask used to re-walk the whole subtree. Nested
    control constructs therefore cost O(|P|^2) per lifeline. Counting node
    visits is what makes the difference visible -- wall-clock would not.
    """

    from zippergen import syntax
    from zippergen.projection import _Context, _project

    A = Lifeline("A")
    B = Lifeline("B")
    v = Var("v", int)
    msg = MsgStmt(A, (VarExpr(v),), B, (VarExpr(v),))

    # Twelve nested conditionals: deep enough that rescanning is unmistakable.
    body = msg
    for _ in range(12):
        body = IfStmt(lambda _e: True, A, body, msg)
    workflow_body = SeqStmt(body, EmptyStmt())

    visits = 0
    original = syntax._participation_set

    def counting(stmt, memo):
        nonlocal visits
        visits += 1
        return original(stmt, memo)

    syntax._participation_set = counting
    try:
        _project(workflow_body, B, _Context(frozenset()))
    finally:
        syntax._participation_set = original

    nodes = 2 * 12 + 2
    assert visits <= nodes * 2, (
        f"{visits} analyses for about {nodes} nodes: participation is being "
        "recomputed rather than consulted"
    )
