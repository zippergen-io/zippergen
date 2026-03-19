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
    Program,
    pp,
)
from zippergen.actions import llm, pure
from zippergen.builder import proc, skip

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

task            = Var("task",            str)
plan            = Var("plan",            str)
planNeedsReview = Var("planNeedsReview", bool, default=False)
tR              = Var("tR",              str)
critique        = Var("critique",        str, default="")
tE              = Var("tE",              str)
result          = Var("result",          str)
final           = Var("final",           str)

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
    outputs=(("plan", str), ("planNeedsReview", bool)),
)
def makePlan(task: str) -> None: ...


@llm(
    system=(
        "You are a plan reviewer. Read the plan and provide a short, constructive critique."
    ),
    user="Plan: {plan}",
    parse="json",
    outputs=(("critique", str),),
)
def reviewPlan(plan: str) -> None: ...


@llm(
    system=(
        "You are an executor. Carry out the plan and report what was done."
    ),
    user="Plan: {plan}",
    parse="json",
    outputs=(("result", str),),
)
def executePlan(plan: str) -> None: ...


@llm(
    system=(
        "You are an orchestrator. Summarize the outcome, taking any reviewer "
        "critique into account."
    ),
    user="Result: {result}\nCritique (may be empty): {critique}",
    parse="json",
    outputs=(("final", str),),
)
def finalizeWithReview(result: str, critique: str) -> None: ...


# ---------------------------------------------------------------------------
# Proc — direct translation of the paper's reviewedExecution
# ---------------------------------------------------------------------------

@proc
def reviewedExecution(task: str @ Planner) -> str:
    Planner: (plan, planNeedsReview) = makePlan(task)

    if planNeedsReview @ Planner:
        Planner(plan) >> Reviewer(tR)
        Reviewer: critique = reviewPlan(tR)
        Reviewer(critique) >> Orchestrator(critique)
    else:
        skip(Planner)

    Planner(plan) >> Executor(tE)
    Executor: result = executePlan(tE)
    Executor(result) >> Orchestrator(result)
    Orchestrator: final = finalizeWithReview(critique, result)
    return final @ Orchestrator


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

program = Program(
    lifelines=(Planner, Reviewer, Orchestrator, Executor),
    actions=(makePlan, reviewPlan, executePlan, finalizeWithReview),
    procs=(reviewedExecution,),
)

# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from zippergen.runtime import mock_llm
    from zipperchat import WebTrace

    wt = WebTrace(program.lifelines).start()
    time.sleep(0.3)   # give the browser a moment to connect

    reviewedExecution.configure(
        backend=lambda a, i: mock_llm(a, i, min_delay=5, max_delay=10),
        trace=wt,
        timeout=60,
    )

    while True:
        wt.reset()
        print("Running reviewedExecution (mock LLM)…")
        final = reviewedExecution(task="Build a REST API for a to-do list application.")
        wt.done()
        print(f"\nResult → {final}")
        print("Click ▶ Run again in the browser, or Ctrl-C to quit.")
        wt.wait_for_replay()
