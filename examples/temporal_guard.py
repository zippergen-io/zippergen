# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""Temporal guard example — approval before action.

An Orchestrator produces a plan and an approval decision.
The Executor may only run the plan if the latest causally visible
Orchestrator event shows approval=True.

Guard (evaluated by Executor):
    Y[Orchestrator](atom(lambda env: env.get("approval", False)))

Run with:
    python examples/temporal_guard.py
"""

from zippergen import Lifeline, Var, workflow
from zippergen.actions import llm, pure
from zippergen import Y, atom

Orchestrator = Lifeline("Orchestrator")
Executor     = Lifeline("Executor")

plan     = Var("plan",     str)
approval = Var("approval", bool)
result   = Var("result",   str)


@llm(
    system="You are a concise task planner.",
    user="Generate a one-sentence plan for: {request}",
    parse="text",
    outputs=[("plan", str)],
)
def make_plan(request: str): pass


@llm(
    system="You are a safety reviewer. Reply true to approve, false to reject.",
    user="Approve this plan? {plan}",
    parse="bool",
    outputs=[("approval", bool)],
)
def approve_plan(plan: str): pass


@llm(
    system="You are an executor. Carry out the given plan.",
    user="Execute: {plan}",
    parse="text",
    outputs=[("result", str)],
)
def execute_plan(plan: str): pass


@pure
def blocked(plan: str) -> str:
    return f"[BLOCKED] No causal approval visible for: {plan}"


# Guard evaluated by Executor: the latest causally visible Orchestrator
# event must record approval=True.
approved_by_orchestrator = Y[Orchestrator](
    atom(lambda env: env.get("approval", False), src="approval")
)


@workflow
def guarded_execution(request: str @ Orchestrator) -> str:
    Orchestrator: plan     = make_plan(request)
    Orchestrator: approval = approve_plan(plan)
    Orchestrator(plan, approval) >> Executor(plan, approval)
    if approved_by_orchestrator @ Executor:
        Executor: result = execute_plan(plan)
    else:
        Executor: result = blocked(plan)
    return result @ Executor


if __name__ == "__main__":
    guarded_execution.configure(llms="mock", ui=True)
    r = guarded_execution(request="organise a team offsite")
    print(f"\nResult → {r}")
    input("\nZipperChat is running at http://localhost:8765\nPress Enter to stop.\n")
