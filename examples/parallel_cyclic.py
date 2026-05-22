# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Feedback example: parallel region with a shared-lifeline message cycle.

Two branches exchange messages between the same pair of shared lifelines:

  Branch 1:  Analyst   ->  Reviewer    (Analyst sends to Reviewer)
  Branch 2:  Reviewer  ->  Analyst     (Reviewer sends back to Analyst)

The dependency graph induced by the shared lifelines {Analyst, Reviewer} is
cyclic. Under the filtered shuffle semantics, the program is admissible:
the source semantics keeps shuffled tuples that are complete MSCs, and at
least one valid interleaving exists (each side sends before receiving).

Run this file to execute the workflow with the mock backend:

  python examples/parallel_cyclic.py
"""

from zippergen import Lifeline, Var, branch, parallel, workflow

Analyst = Lifeline("Analyst")
Reviewer = Lifeline("Reviewer")

draft = Var("draft", str)
feedback = Var("feedback", str)


@workflow
def feedback_exchange(text: str @ Analyst) -> str:
    with parallel:
        with branch:
            Analyst(text) >> Reviewer(draft)

        with branch:
            Reviewer(draft) >> Analyst(feedback)

    return feedback @ Analyst


if __name__ == "__main__":
    feedback_exchange.configure(llms="mock")
    result = feedback_exchange(text="hello")
    print(f"feedback: {result}")
