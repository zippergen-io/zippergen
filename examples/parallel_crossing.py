# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Parallel messages between the same two lifelines, received in reverse order.

Both branches send from Sender to Receiver. The sender sends P1 before P2, but
the receiver delays the P1 receive branch and therefore receives P2 first. In
an MSC-style view, the two message arrows cross.
"""

import time

from zippergen import Lifeline, Var, branch, parallel, pure, workflow


Sender = Lifeline("Sender")
Receiver = Lifeline("Receiver")

slow_msg = Var("slow_msg", str)
fast_msg = Var("fast_msg", str)
gate = Var("gate", str)
summary = Var("summary", str)


@pure
def wait_before_receiving(item: str) -> str:
    time.sleep(0.45)
    return item


@pure
def summarize_order(first: str, second: str) -> str:
    return f"received first: {first}; received second: {second}"


@workflow
def crossing_messages(item: str @ Sender) -> str:
    with parallel:
        with branch:
            Receiver: (gate,) = wait_before_receiving("first branch")
            Sender("first confirmation") >> Receiver(slow_msg)

        with branch:
            Sender("second confirmation") >> Receiver(fast_msg)

    Receiver: (summary,) = summarize_order(fast_msg, slow_msg)
    return summary @ Receiver


if __name__ == "__main__":
    crossing_messages.configure(llms="mock", ui=True, timeout=30)
    print(crossing_messages(item="batch-4"))
    input("\nZipperChat is running at http://localhost:8765\nPress Enter to stop.\n")
