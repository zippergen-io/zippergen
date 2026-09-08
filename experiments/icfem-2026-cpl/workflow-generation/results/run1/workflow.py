"""Forward an opaque token and decide the causally prior coin result at L3."""

from zippergen import At, Lifeline, pure, workflow


L0 = Lifeline("L0")
L1 = Lifeline("L1")
L2 = Lifeline("L2")
L3 = Lifeline("L3")


@pure
def heads_result() -> str:
    return "heads"


@pure
def tails_result() -> str:
    return "tails"


@workflow
def coin_chain(outcome: str @ L0, token: str @ L0) -> str:
    L0(token) >> L1(token)
    L1(token) >> L2(token)
    L2(token) >> L3(token)
    if (At[L0].outcome == "heads") @ L3:
        L3: result = heads_result()
    else:
        L3: result = tails_result()
    return result @ L3
