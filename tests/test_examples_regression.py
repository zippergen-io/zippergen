"""Characterization gate: the in-process run() path must not change behavior.

Uses a deterministic two-role, one-exchange, one-branch workflow built directly
from the IR (no builder, no randomness) so the assertion is exact. This fixture
is the canonical finite example reused by later tasks in the deployable-runtime
plan; it must keep passing, unmodified in observable behavior, through every
subsequent task.
"""
from zippergen.syntax import (
    ActStmt, IfStmt, LitExpr, MsgStmt, SeqStmt, Var, VarExpr, Workflow, Lifeline,
)
from zippergen.actions import pure
from zippergen.runtime import run

A = Lifeline("A")
B = Lifeline("B")
x = Var("x", int)
ok = Var("ok", bool)


@pure
def is_positive(x: int) -> bool:
    return x > 0


def _two_role_branch_workflow() -> Workflow:
    # A sends x to B; B decides ok = is_positive(x) and both learn the outcome.
    body = SeqStmt(
        MsgStmt(A, (VarExpr(x),), B, (VarExpr(x),)),
        SeqStmt(
            # act B: ok := is_positive(x)
            ActStmt(B, is_positive, (VarExpr(x),), (ok,)),
            IfStmt(
                condition=lambda _e: _e.ok,
                owner=B,
                branch_true=MsgStmt(B, (LitExpr(True, bool),), A, (VarExpr(ok),)),
                branch_false=MsgStmt(B, (LitExpr(False, bool),), A, (VarExpr(ok),)),
            ),
        ),
    )
    return Workflow(
        name="two_role_branch",
        inputs=(("x", int, A),),
        output_type=bool,
        vars=(),
        body=body,
        outputs=((ok, A),),
        ns={"x": x, "ok": ok},
    )


def test_inprocess_two_role_branch_golden():
    wf = _two_role_branch_workflow()
    result = run(wf, [A, B], {"A": {"x": 7}}, timeout=10)
    assert result is True


def test_inprocess_two_role_branch_false():
    wf = _two_role_branch_workflow()
    result = run(wf, [A, B], {"A": {"x": -3}}, timeout=10)
    assert result is False
