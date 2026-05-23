# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Smoke test: two while loops inside a parallel block with a shared User lifeline.

Stream_A and Stream_B each drain their own FIFO queue.
Every item passes through User, who handles it.

Expected: all 5 items processed in some interleaved order; no deadlock.

Run with:
    python examples/smoke_parallel_while.py
"""

from zippergen import Lifeline, Var, branch, parallel, pure, workflow

Stream_A = Lifeline("Stream_A")
Stream_B = Lifeline("Stream_B")
User     = Lifeline("User")

# Parallel branches share User's env — use distinct variable names per stream
# to avoid one branch overwriting the other's in-flight value.
item_a   = Var("item_a",   str)
item_b   = Var("item_b",   str)
result_a = Var("result_a", str)
result_b = Var("result_b", str)
summary  = Var("summary",  str)

QUEUE_A = ["a1", "a2"]
QUEUE_B = ["b1", "b2", "b3"]


def has_a() -> bool:
    return bool(QUEUE_A)


def has_b() -> bool:
    return bool(QUEUE_B)


@pure
def pop_a() -> str:
    return QUEUE_A.pop(0)


@pure
def pop_b() -> str:
    return QUEUE_B.pop(0)


@pure
def handle_a(item_a: str) -> str:
    print(f"  [User] from A: {item_a}")
    return f"ok:{item_a}"


@pure
def handle_b(item_b: str) -> str:
    print(f"  [User] from B: {item_b}")
    return f"ok:{item_b}"


@pure
def done() -> str:
    return "all done"


@workflow
def parallel_while_smoke() -> str:
    with parallel:
        with branch:
            while has_a() @ Stream_A:
                Stream_A: item_a = pop_a()
                Stream_A(item_a) >> User(item_a)
                User: result_a = handle_a(item_a)
        with branch:
            while has_b() @ Stream_B:
                Stream_B: item_b = pop_b()
                Stream_B(item_b) >> User(item_b)
                User: result_b = handle_b(item_b)

    User: summary = done()
    return summary @ User


if __name__ == "__main__":
    parallel_while_smoke.configure(llms="mock", ui=False)
    result = parallel_while_smoke()
    print(f"\nResult: {result}")
