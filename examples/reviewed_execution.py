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
    Text, Bool,
    Lifeline, Var,
    Program,
    pp,
)
from zippergen.actions import llm, pure
from zippergen.builder import proc, msg, act, skip, if_

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

task            = Var("task",            Text)
plan            = Var("plan",            Text)
planNeedsReview = Var("planNeedsReview", Bool, default=False)
tR              = Var("tR",              Text)
critique        = Var("critique",        Text, default="")
tE              = Var("tE",              Text)
result          = Var("result",          Text)
final           = Var("final",           Text)

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
    outputs=(("plan", Text), ("planNeedsReview", Bool)),
)
def makePlan(task: Text) -> None: ...


@llm(
    system=(
        "You are a plan reviewer. Read the plan and provide a short, constructive critique."
    ),
    user="Plan: {plan}",
    parse="json",
    outputs=(("critique", Text),),
)
def reviewPlan(plan: Text) -> None: ...


@llm(
    system=(
        "You are an executor. Carry out the plan and report what was done."
    ),
    user="Plan: {plan}",
    parse="json",
    outputs=(("result", Text),),
)
def executePlan(plan: Text) -> None: ...


@llm(
    system=(
        "You are an orchestrator. Summarize the outcome, taking any reviewer "
        "critique into account."
    ),
    user="Result: {result}\nCritique (may be empty): {critique}",
    parse="json",
    outputs=(("final", Text),),
)
def finalizeWithReview(result: Text, critique: Text) -> None: ...


# ---------------------------------------------------------------------------
# Proc — direct translation of the paper's reviewedExecution
# ---------------------------------------------------------------------------

@proc
def reviewedExecution(task: Text) -> Text:
    act(Planner, makePlan, (task,), (plan, planNeedsReview))

    def if_body():
        msg(Planner,  (plan,),     Reviewer,     (tR,))
        act(Reviewer, reviewPlan,  (tR,),         (critique,))
        msg(Reviewer, (critique,), Orchestrator,  (critique,))

    def else_body():
        skip(Planner)

    if_(planNeedsReview, Planner, then=if_body, else_=else_body)

    msg(Planner,   (plan,),   Executor,     (tE,))
    act(Executor,  executePlan, (tE,),      (result,))
    msg(Executor,  (result,), Orchestrator, (result,))
    act(Orchestrator, finalizeWithReview, (critique, result), (final,))


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
    from zippergen.runtime import run, mock_llm
    from zipperchat import WebTrace

    initial = {
        "Planner": {
            "task": "Build a REST API for a to-do list application.",
        }
    }

    wt = WebTrace(program.lifelines).start()
    time.sleep(0.3)   # give the browser a moment to connect

    print("Running reviewedExecution (mock LLM)…")
    final_envs = run(
        reviewedExecution,
        list(program.lifelines),
        initial,
        llm_backend=lambda a, i: mock_llm(a, i, min_delay=5, max_delay=10),
        trace=wt,
        timeout=60,
    )
    wt.done()

    print(f"\nResult → {final_envs['Orchestrator'].get('final')}")
    input("\nPress Enter to stop the server…")
