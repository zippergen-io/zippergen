# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Nested workflows — exercise multi-output and WorkflowAction.

Topology:

  inner workflow  draft_and_review(topic @ Writer) -> (draft @ Writer, score @ Reviewer)
      Writer  : drafts a response to the topic
      Reviewer: scores the draft (0-10)

  outer workflow  pipeline(topic @ User) -> str
      User        : provides the topic
      Coordinator : calls draft_and_review as a black-box WorkflowAction
                    receives draft + score, forwards to User

All actions are pure so this runs without any LLM backend.
"""

from zippergen.syntax import Lifeline, Var
from zippergen.actions import pure
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Lifelines — inner workflow has its own set
# ---------------------------------------------------------------------------

User        = Lifeline("User")
Coordinator = Lifeline("Coordinator")

Writer   = Lifeline("Writer")
Reviewer = Lifeline("Reviewer")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

topic  = Var("topic",  str)
draft  = Var("draft",  str)
score  = Var("score",  int)
result = Var("result", str)

# ---------------------------------------------------------------------------
# Pure actions
# ---------------------------------------------------------------------------

@pure
def write_draft(t: str) -> str:
    return f"Draft about '{t}': Lorem ipsum."

@pure
def review_draft(d: str) -> int:
    return min(10, len(d.split()))   # word count as a mock score

@pure
def format_result(d: str, s: int) -> str:
    return f"{d}  [score: {s}/10]"

# ---------------------------------------------------------------------------
# Inner workflow — multi-output: returns (draft @ Writer, score @ Reviewer)
# ---------------------------------------------------------------------------

@workflow
def draft_and_review(topic: str @ Writer) -> tuple:
    Writer: (draft,) = write_draft(topic)
    Writer(draft) >> Reviewer(draft)
    Reviewer: (score,) = review_draft(draft)
    return (draft @ Writer, score @ Reviewer)

# ---------------------------------------------------------------------------
# Outer workflow — uses draft_and_review as a WorkflowAction
# ---------------------------------------------------------------------------

@workflow
def pipeline(topic: str @ User) -> str:
    User(topic) >> Coordinator(topic)
    Coordinator: (draft, score) = draft_and_review(topic @ Writer)
    Coordinator: (result,) = format_result(draft, score)
    Coordinator(result) >> User(result)
    return result @ User


if __name__ == "__main__":
    USE_UI = True

    pipeline.configure(
        llms="mock",
        ui=USE_UI,
        timeout=120,
    )

    result_val = pipeline(topic="quantum computing")
    print(f"\nResult → {result_val}")
    if USE_UI:
        input("ZipperChat is running at http://localhost:8765 . Press Enter to close. ")
