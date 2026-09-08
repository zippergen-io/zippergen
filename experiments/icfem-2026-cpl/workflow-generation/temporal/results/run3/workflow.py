"""Preserve a pass through checking updates, until a failure cancels it."""

from zippergen import At, Here, Lifeline, since, pure, workflow

L0 = Lifeline("L0")
L1 = Lifeline("L1")
L2 = Lifeline("L2")
L3 = Lifeline("L3")


@pure
def record_status(value: str) -> str:
    if value not in ("passed", "checking", "failed"):
        raise ValueError("status must be passed, checking, or failed")
    return value


@pure
def accept() -> str:
    return "accept"


@pure
def reject() -> str:
    return "reject"


@workflow
def status_chain(
    first_status: str @ L0,
    second_status: str @ L0,
    third_status: str @ L0,
    token: str @ L0,
) -> str:
    L0: status = record_status(first_status)
    L0: status = record_status(second_status)
    L0: status = record_status(third_status)

    L0(token) >> L1(token)
    L1(token) >> L2(token)
    L2(token) >> L3(token)

    if At[L0](since(Here.status != "failed", Here.status == "passed")) @ L3:
        L3: result = accept()
    else:
        L3: result = reject()
    return result @ L3
