# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""Human approval gate example — all three input types.

A Planner drafts a task description; a human Reviewer:
  1. sets a priority level  (choice)
  2. adds an execution note (free text)
  3. approves or rejects    (bool)

Run with:
    python examples/human_approval.py
"""

from zippergen import Lifeline, Var, workflow
from zippergen.actions import llm, human, pure

Planner  = Lifeline("Planner")
Reviewer = Lifeline("Reviewer")

plan     = Var("plan",     str)
priority = Var("priority", str)
note     = Var("note",     str)
approved = Var("approved", bool)
summary  = Var("summary",  str)


@llm(
    system="You are a concise task planner.",
    user="Write a one-sentence plan for: {request}",
    parse="text",
    outputs=[("plan", str)],
)
def draft_plan(request: str): pass


@human(
    kind="select",
    context="{plan}",
    instruction="Set priority for this plan.",
    prefill="high\nmedium\nlow",
    outputs=["priority: str"],
)
def set_priority(plan: str): pass


@human(
    kind="input",
    context="{plan}",
    instruction="Add an execution note for this plan.",
    outputs=["note: str"],
)
def add_note(plan: str): pass


@human(
    kind="confirm",
    context="{plan}",
    instruction="Approve this plan?",
    outputs=["approved: bool"],
)
def review_plan(plan: str): pass


@pure
def summarise(approved: bool, plan: str, priority: str, note: str) -> str:
    if approved:
        return f"[{priority.upper()}] Approved: {plan}\nNote: {note}"
    return "Rejected — no plan executed."


@workflow
def approval_workflow(request: str @ Planner) -> str:
    Planner: plan = draft_plan(request)
    Planner(plan) >> Reviewer(plan)
    Reviewer: priority = set_priority(plan)
    Reviewer: note = add_note(plan)
    Reviewer: approved = review_plan(plan)
    Reviewer(priority, note, approved) >> Planner(priority, note, approved)
    Planner: summary = summarise(approved, plan, priority, note)
    return summary @ Planner


if __name__ == "__main__":
    approval_workflow.configure(llms="mock", ui=True)
    result = approval_workflow(request="organise a team offsite")
    print(f"\nResult → {result}")
    try:
        input("\nZipperChat is running at http://localhost:8765\nPress Enter to stop, or click Run again in the browser.\n")
    except EOFError:
        pass
