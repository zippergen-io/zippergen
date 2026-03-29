# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""
Open Planner — LLM-generated workflow with LLM-written actions.

Demonstrates @planner with allow=["llm"] and no pre-defined vocabulary.
The planner LLM invents all @llm actions from scratch, tailored to the
specific request — rather than composing a fixed library.

Task: professional cover letter drafting and critique.
"""

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

request = Var("request", str)
job_desc  = Var("job_desc",  str)
cv_sketch = Var("cv_sketch", str)
result    = Var("result",    str)

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
def write_document(request: str, job_desc: str, cv_sketch: str) -> str: ...


# ---------------------------------------------------------------------------
# Outer coordination protocol
# ---------------------------------------------------------------------------

@workflow
def openPlannerAgent(request: str @ User, job_desc: str @ User, cv_sketch: str @ User) -> str:
    User(request, job_desc, cv_sketch) >> Planner(request, job_desc, cv_sketch)
    Planner: result = write_document(request, job_desc, cv_sketch)
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

    result_val = openPlannerAgent(
        request="Draft a polished cover letter using the available workers.",
        job_desc=(
            "Senior Software Engineer — distributed systems team at a fintech startup. "
            "We build high-throughput payment infrastructure. "
            "Looking for someone with strong Python/Go skills, experience with "
            "event-driven architecture, and a bias for operational simplicity."
        ),
        cv_sketch=(
            "5 years backend engineering. "
            "Led migration of monolith to event-driven microservices at previous company (Python, Kafka). "
            "Open-source contributor to a distributed task queue library. "
            "Passionate about clean APIs and minimal abstractions."
        ),
    )
    print(f"\n{'='*60}")
    print("RESULT")
    print("=" * 60)
    print(result_val)
    input("\nZipperChat is running at http://localhost:8765 . Press Enter to close. ")
