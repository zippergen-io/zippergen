# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""Human approval gate example.

A Planner drafts a task description; a human reviews and approves or rejects it.
Run with:
    python examples/human_approval.py
"""

from zippergen import Lifeline, Var, workflow
from zippergen.actions import llm, human, pure

Planner  = Lifeline("Planner")
Reviewer = Lifeline("Reviewer")

plan     = Var("plan",     str)
approved = Var("approved", bool)
summary  = Var("summary",  str)


@llm(
    system="You are a concise task planner.",
    user="Write a one-sentence plan for: {request}",
    parse="text",
    outputs=[("plan", str)],
)
def draft_plan(request: str): pass


@human(prompt="Approve this plan?\n\n  {plan}\n", outputs=["approved: bool"])
def review_plan(plan: str): pass


@pure
def summarise(approved: bool, plan: str) -> str:
    if approved:
        return f"Approved: {plan}"
    return "Rejected — no plan executed."


@workflow
def approval_workflow(request: str @ Planner) -> str:
    Planner: plan = draft_plan(request)
    Planner(plan) >> Reviewer(plan)
    Reviewer: approved = review_plan(plan)
    Reviewer(approved) >> Planner(approved)
    Planner: summary = summarise(approved, plan)
    return summary @ Planner


if __name__ == "__main__":
    approval_workflow.configure(llms="mock", ui=False)
    result = approval_workflow(request="organise a team offsite")
    print(result)
