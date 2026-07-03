"""A deterministic two-role bounded loop: A counts up to `limit`, exchanging the
counter with B each iteration. Owner of the loop guard is A. No LLM, no kpar."""
from zippergen import Lifeline, Var, workflow
from zippergen.actions import pure

A = Lifeline("A")
B = Lifeline("B")

# All names used in the DSL body must be module-level Vars (the AST transform
# evaluates receive-side / message names against this namespace).
n     = Var("n",     int, default=0)
limit = Var("limit", int, default=0)
m     = Var("m",     int, default=0)
ack   = Var("ack",   int, default=0)
got   = Var("got",   int, default=0)


@pure
def add_one(n: int) -> int:
    return n + 1


@pure
def relay(m: int) -> int:
    return m


@workflow
def counter_loop(n: int @ A, limit: int @ A):
    while (n < limit) @ A:
        A(n) >> B(m)
        B: ack = relay(m)
        B(ack) >> A(got)
        A: n = add_one(n)
    return n @ A
