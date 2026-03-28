# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""
Open Planner — LLM-generated workflow with LLM-written actions.

Demonstrates @planner with allow=["llm"] and no pre-defined vocabulary.
The planner LLM invents all @llm actions from scratch, tailored to the
specific request — rather than composing a fixed library.

Task: professional cover letter drafting and critique.
"""

import json

from zippergen.syntax import Lifeline, Var
from zippergen.actions import planner
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User    = Lifeline("User")
Planner = Lifeline("Planner")
Worker1 = Lifeline("Worker1")
Worker2 = Lifeline("Worker2")

# ---------------------------------------------------------------------------
# Outer workflow variables
# ---------------------------------------------------------------------------

request_var = Var("request",     str)
inputs_json = Var("inputs_json", str)
result      = Var("result",      str)

# ---------------------------------------------------------------------------
# Open planner — no base vocabulary, LLM writes all actions
# ---------------------------------------------------------------------------

@planner(
    system=(
        "You are a workflow planner for professional writing tasks. "
        "Given a user request and available input data, design a multi-agent "
        "workflow and write all the LLM actions it needs from scratch. "
        "Tailor every system prompt and user prompt precisely to the task."
    ),
    actions=[],
    lifelines=[Worker1, Worker2],
    allow=["llm"],
)
def write_document(request: str, inputs_json: str) -> str: ...


# ---------------------------------------------------------------------------
# Outer coordination protocol
# ---------------------------------------------------------------------------

@workflow
def openPlannerAgent(request: str @ User, inputs_json: str @ User) -> str:
    User(request, inputs_json) >> Planner(request, inputs_json)
    Planner: result = write_document(request, inputs_json)
    Planner(result) >> User(result)
    return result @ User


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    openPlannerAgent.configure(
        llms={"Planner": "openai"},
        ui=True,
        timeout=300,
    )

    SAMPLE_REQUEST = (
        "Draft a cover letter. Worker1 should write the initial draft; "
        "Worker2 should critique it for tone, conciseness, and fit with the job; "
        "then Worker1 should revise the draft based on the critique."
    )
    SAMPLE_INPUTS = {
        "job_desc": (
            "Senior Software Engineer — distributed systems team at a fintech startup. "
            "We build high-throughput payment infrastructure. "
            "Looking for someone with strong Python/Go skills, experience with "
            "event-driven architecture, and a bias for operational simplicity."
        ),
        "cv_sketch": (
            "5 years backend engineering. "
            "Led migration of monolith to event-driven microservices at previous company (Python, Kafka). "
            "Open-source contributor to a distributed task queue library. "
            "Passionate about clean APIs and minimal abstractions."
        ),
    }

    result_val = openPlannerAgent(
        request=SAMPLE_REQUEST,
        inputs_json=json.dumps(SAMPLE_INPUTS),
    )
    print(f"\n{'='*60}")
    print("RESULT")
    print("=" * 60)
    print(result_val)
    input("\nZipperChat is running at http://localhost:8765 . Press Enter to close. ")
