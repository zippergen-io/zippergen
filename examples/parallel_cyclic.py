# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Cyclic dependency rejection example.

This file deliberately constructs an ill-formed parallel region where two
shared lifelines have a circular message dependency:

  Branch 1:  Analyst  →  Reviewer   (Analyst sends to Reviewer)
  Branch 2:  Reviewer →  Analyst    (Reviewer sends to Analyst)

The reachability graph induced by the shared lifelines {Analyst, Reviewer}
contains the cycle Analyst → Reviewer → Analyst, so the region is rejected at
projection time with a ValueError.

Run this file to see the error:

  python examples/parallel_cyclic.py
"""

from zippergen import Lifeline, Var, branch, parallel, workflow

Analyst = Lifeline("Analyst")
Reviewer = Lifeline("Reviewer")

draft = Var("draft", str)
feedback = Var("feedback", str)


@workflow
def cyclic_exchange(text: str @ Analyst) -> str:
    with parallel:
        with branch:
            # Analyst sends to Reviewer
            Analyst(text) >> Reviewer(draft)

        with branch:
            # Reviewer sends back to Analyst — creates cycle with branch 1
            Reviewer(draft) >> Analyst(feedback)

    return feedback @ Analyst


if __name__ == "__main__":
    try:
        cyclic_exchange.configure(llms="mock")
        cyclic_exchange(text="hello")
    except ValueError as e:
        print(f"Rejected (as expected):\n  {e}")
