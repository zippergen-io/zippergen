# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Nested workflows — 3 levels deep + two parallel sub-workflows.

Topology (depth-first):

  Level 3 (deepest)
    score_text(content @ Scorer) -> int
        Scorer: computes a word-count score

  Level 2a — calls score_text (→ 3 levels total)
    analyze_and_score(topic @ Analyst) -> (analysis, score)
        Analyst: writes an analysis, then calls score_text as a sub-workflow

  Level 2b — runs in parallel with 2a inside main_pipeline
    quick_summarize(topic @ Summarizer) -> summary
        Summarizer: makes a brief then expands it (two steps)

  Level 1 (outer)
    main_pipeline(topic @ User) -> str
        User        : broadcasts topic to Worker1 and Worker2
        Worker1     : calls analyze_and_score  ┐ parallel
        Worker2     : calls quick_summarize    ┘
        Orchestrator: receives both results, combines, returns to User

All actions are pure — no LLM backend needed.
"""

from zippergen.syntax import Lifeline, Var
from zippergen.actions import pure
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User         = Lifeline("User")
Worker1      = Lifeline("Worker1")
Worker2      = Lifeline("Worker2")
Orchestrator = Lifeline("Orchestrator")

Analyst    = Lifeline("Analyst")    # lives inside analyze_and_score
Scorer     = Lifeline("Scorer")     # lives inside score_text
Summarizer = Lifeline("Summarizer") # lives inside quick_summarize
Editor     = Lifeline("Editor")     # lives inside refine_and_finalize

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

topic    = Var("topic",    str)
analysis = Var("analysis", str)
score    = Var("score",    int)
brief    = Var("brief",    str)
refined  = Var("refined",  str)
summary  = Var("summary",  str)
result   = Var("result",   str)

# ---------------------------------------------------------------------------
# Pure actions
# ---------------------------------------------------------------------------

@pure
def write_analysis(t: str) -> str:
    return f"Analysis of '{t}': key themes are X, Y, and Z."

@pure
def compute_score(text: str) -> int:
    return min(10, len(text.split()))

@pure
def make_brief(t: str) -> str:
    return f"Brief: {t[:40]}…"

@pure
def apply_edits(b: str) -> str:
    return f"{b} (edited)"

@pure
def finalize_text(b: str) -> str:
    return f"Final: {b}"

@pure
def combine_results(a: str, s: int, summ: str) -> str:
    return f"[score {s}/10] {a} | Summary: {summ}"

# ---------------------------------------------------------------------------
# Level 3 — deepest leaf: just scores a text string
# ---------------------------------------------------------------------------

@workflow
def score_text(content: str @ Scorer) -> int:
    Scorer: (score,) = compute_score(content)
    return score @ Scorer

# ---------------------------------------------------------------------------
# Level 2a — analysis + nested scoring (calls score_text → 3 levels)
# ---------------------------------------------------------------------------

@workflow
def analyze_and_score(topic: str @ Analyst) -> tuple:
    Analyst: (analysis,) = write_analysis(topic)
    Analyst: (score,)    = score_text(analysis @ Scorer)
    return (analysis @ Analyst, score @ Analyst)

# ---------------------------------------------------------------------------
# Level 3 — refine a draft through an Editor (called from quick_summarize)
# ---------------------------------------------------------------------------

@workflow
def refine_and_finalize(draft: str @ Editor) -> str:
    Editor: (refined,) = apply_edits(draft)
    Editor: (summary,) = finalize_text(refined)
    return summary @ Editor

# ---------------------------------------------------------------------------
# Level 2b — quick summarise (parallel sibling of analyze_and_score)
#             calls refine_and_finalize → also 3 levels deep
# ---------------------------------------------------------------------------

@workflow
def quick_summarize(topic: str @ Summarizer) -> str:
    Summarizer: (brief,)   = make_brief(topic)
    Summarizer: (summary,) = refine_and_finalize(brief @ Editor)
    return summary @ Summarizer

# ---------------------------------------------------------------------------
# Level 1 — outer pipeline: spawns both sub-workflows in parallel
# ---------------------------------------------------------------------------

@workflow
def main_pipeline(topic: str @ User) -> str:
    User(topic) >> Worker1(topic)
    User(topic) >> Worker2(topic)
    # Worker1 and Worker2 act concurrently — no dependency between them
    Worker1: (analysis, score) = analyze_and_score(topic @ Analyst)
    Worker2: (summary,)        = quick_summarize(topic @ Summarizer)
    # Collect results at Orchestrator
    Worker1(analysis) >> Orchestrator(analysis)
    Worker1(score)    >> Orchestrator(score)
    Worker2(summary)  >> Orchestrator(summary)
    Orchestrator: (result,) = combine_results(analysis, score, summary)
    Orchestrator(result) >> User(result)
    return result @ User


if __name__ == "__main__":
    USE_UI = True

    main_pipeline.configure(
        llms="mock",
        ui=USE_UI,
        timeout=120,
    )

    result_val = main_pipeline(topic="quantum computing")
    print(f"\nResult → {result_val}")
    if USE_UI:
        input("ZipperChat running at http://localhost:8765 — press Enter to close.")
