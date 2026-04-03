# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Review-before-action workflow — motivating example from the paper.

A Planner decides whether its plan needs review.  If so, a Reviewer
critiques it and forwards the critique to the Orchestrator.  Either way,
the plan is sent to an Executor whose result, together with any critique,
is finalized by the Orchestrator.

Direct transcription of Listing 1 / Figure 1 from the paper
"Provable Coordination for LLM Agents via Message Sequence Charts".
"""

from zippergen.syntax import (
    Lifeline, Var,
    pp,
)
from zippergen.actions import llm, pure
from zippergen.builder import workflow, skip

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

Planner      = Lifeline("Planner")
Reviewer     = Lifeline("Reviewer")
Orchestrator = Lifeline("Orchestrator")
Executor     = Lifeline("Executor")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

task             = Var("task",             str)
plan             = Var("plan",             str)
plan_needs_review = Var("plan_needs_review", bool, default=False)
tR               = Var("tR",               str)
critique         = Var("critique",         str, default="")
tE               = Var("tE",               str)
result           = Var("result",           str)
final            = Var("final",            str)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@llm(
    system=(
        "You are a project planner. Given a task description, produce a concise "
        "action plan and decide whether it should be reviewed before execution."
    ),
    user="Task: {task}",
    parse="json",
    outputs=(("plan", str), ("plan_needs_review", bool)),
)
def make_plan(task: str) -> None: ...


@llm(
    system=(
        "You are a plan reviewer. Read the plan and provide a short, constructive critique."
    ),
    user="Plan: {plan}",
    parse="json",
    outputs=(("critique", str),),
)
def review_plan(plan: str) -> None: ...


@llm(
    system=(
        "You are an executor. Carry out the plan and report what was done."
    ),
    user="Plan: {plan}",
    parse="json",
    outputs=(("result", str),),
)
def execute_plan(plan: str) -> None: ...


@llm(
    system=(
        "You are an orchestrator. Summarize the outcome, taking any reviewer "
        "critique into account."
    ),
    user="Result: {result}\nCritique (may be empty): {critique}",
    parse="json",
    outputs=(("final", str),),
)
def finalize_with_review(result: str, critique: str) -> None: ...


# ---------------------------------------------------------------------------
# Proc — direct translation of the paper's reviewed_execution
# ---------------------------------------------------------------------------

@workflow
def reviewed_execution(task: str @ Planner) -> str:
    Planner: (plan, plan_needs_review) = make_plan(task)

    if plan_needs_review @ Planner:
        Planner(plan) >> Reviewer(tR)
        Reviewer: critique = review_plan(tR)
        Reviewer(critique) >> Orchestrator(critique)
    else:
        skip(Planner)

    Planner(plan) >> Executor(tE)
    Executor: result = execute_plan(tE)
    Executor(result) >> Orchestrator(result)
    Orchestrator: final = finalize_with_review(critique, result)
    return final @ Orchestrator


if __name__ == "__main__":
    USE_UI = True

    reviewed_execution.configure(
        llms="mock",
        ui=USE_UI,
        timeout=60,
        mock_delay=(5.0, 10.0),
    )
    final = reviewed_execution(task="Build a REST API for a to-do list application.")
    print(f"\nResult → {final}")
    if USE_UI:
        input("ZipperChat is running at http://localhost:8765 . Press Enter to close. ")
