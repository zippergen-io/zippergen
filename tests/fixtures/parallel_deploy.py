"""Two-role parallel workflow for the durable-deploy integration test
(tests/test_deploy_integration.py::test_parallel_two_process_kill9).

A owns two inputs (x, y) and hands each off to B in its own parallel branch;
B bumps each value and echoes it back to A. Pure actions only — no LLM calls —
so the test isolates the crash/resume behavior of the durable runtime itself.

The two branches sleep for different durations before returning (mirroring
examples/parallel.py's staggered `run_tests`/`scan_security` timings) so that
a kill mid-run lands with one branch still in flight and the other already
settled — real partial-progress coverage instead of a race that only ever
hits "before anything happened" or "after everything happened".

`@workflow` functions must live at module top level (the builder reads this
file's source for AST rewriting), and this module is loaded via
`python -m zippergen.serve serve --workflow <this file>`, so it must define
exactly one `Workflow` object at module scope (see serve.load_workflow).
"""
import time

from zippergen import Lifeline, Var, workflow, branch, parallel
from zippergen.actions import pure

A = Lifeline("A"); B = Lifeline("B")
x = Var("x", int, default=0); y = Var("y", int, default=0)
u = Var("u", int, default=0); v = Var("v", int, default=0)


@pure
def bump_u(n: int) -> int:
    time.sleep(0.6)
    return n + 1


@pure
def bump_v(n: int) -> int:
    time.sleep(0.2)
    return n + 1


@workflow
def par_flow(x: int @ A, y: int @ A):
    with parallel:
        with branch:
            A(x) >> B(u)
            B: u = bump_u(u)
            B(u) >> A(x)
        with branch:
            A(y) >> B(v)
            B: v = bump_v(v)
            B(v) >> A(y)
    return x @ A
