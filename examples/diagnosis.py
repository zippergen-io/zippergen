# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Medical Diagnosis Consensus — running example.

Direct transcription of Listing 1 / Listing 2 from the paper
"Provable Coordination for LLM Agents via Message Sequence Charts".

Two independent LLMs analyze patient notes and iterate until they agree
on a diagnosis verdict, or until MAX_ROUNDS is reached. LLM1 owns the
consensus loop.
"""

from zippergen.syntax import (
    Lifeline, Var,
    Program,
)
from zippergen.actions import llm, pure
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ROUNDS = 5

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User = Lifeline("User")
LLM1 = Lifeline("LLM1")
LLM2 = Lifeline("LLM2")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

# Workflow inputs (each LLM gets its own copy via message binding)
notes           = Var("notes",         str)
diagnosis       = Var("diagnosis",     str)

# Own verdict and reasoning (local to each LLM)
verdict         = Var("verdict",       bool)
reason          = Var("reason",        str)

# Received verdict and reasoning (from the other LLM)
other_verdict   = Var("other_verdict", bool)
other_reason    = Var("other_reason",  str)

# Control and output
agreed = Var("agreed", bool, default=False)
trials = Var("trials", int,  default=0)
result = Var("result", str)

# ---------------------------------------------------------------------------
# Action definitions  (Listing 2 from the paper)
# ---------------------------------------------------------------------------

@llm(
    system=(
        "You are a medical expert. Analyze the notes and determine "
        "if the diagnosis applies. Return verdict (true/false/unknown) "
        "and your reasoning."
    ),
    user="Notes: {notes}\nDiagnosis: {diag}",
    parse="json",   # expects {"verdict": "...", "reason": "..."}
    outputs=(("verdict", bool), ("reason", str)),
)
def assess(notes: str, diag: str) -> None: ...


@llm(
    system=(
        "You are a medical expert. Given your previous assessment "
        "and a colleague's assessment, reconsider your verdict. "
        "You may change or maintain your position."
    ),
    user=(
        "Notes: {notes}\nDiagnosis: {diag}\n"
        "Your verdict: {myVerdict} because {myReason}\n"
        "Colleague: {otherVerdict} because {otherReason}"
    ),
    parse="json",   # expects {"verdict": true/false, "reason": "..."}
    outputs=(("verdict", bool), ("reason", str)),
)
def reconsider(notes: str, diag: str, 
               myVerdict: bool, myReason: str,
               otherVerdict: bool, otherReason: str) -> None: ...


@pure
def incTrials(t: int) -> int:
    return t + 1


@pure
def checkAgreement(v1: bool, v2: bool) -> bool:
    return v1 == v2


@pure
def chooseResult(v: bool, agreed: bool) -> str:
    return ("true" if v else "false") if agreed else "unknown"


# Aliases to match the paper's camelCase names used in ActStmt below
inc_trials      = incTrials
check_agreement = checkAgreement
choose_result   = chooseResult

# ---------------------------------------------------------------------------
# Workflow — direct translation of Listing 1
# ---------------------------------------------------------------------------

@workflow
def diagnosisConsensus(notes: str @ User, diagnosis: str @ User) -> str:
    # Distribute notes to both LLMs
    User(notes, diagnosis) >> LLM1(notes, diagnosis)
    User(notes, diagnosis) >> LLM2(notes, diagnosis)

    # Independent initial assessments
    LLM1: (verdict, reason) = assess(notes, diagnosis)
    LLM2: (verdict, reason) = assess(notes, diagnosis)

    # Consensus loop — owned by LLM1 (at most MAX_ROUNDS rounds)
    while (not agreed and trials < MAX_ROUNDS) @ LLM1:
        LLM1(verdict, reason) >> LLM2(other_verdict, other_reason)
        LLM2(verdict, reason) >> LLM1(other_verdict, other_reason)
        LLM1: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
        LLM2: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
        LLM2(verdict) >> LLM1(other_verdict)
        with LLM1:
            agreed = check_agreement(verdict, other_verdict)
            trials = inc_trials(trials)
    else:
        LLM1(verdict, reason) >> LLM2(other_verdict, other_reason)

    # Final result computed by LLM1, sent to User
    LLM1: result = choose_result(verdict, agreed)
    LLM1(result) >> User(result)
    return result @ User


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

program = Program(
    lifelines=(User, LLM1, LLM2),
    actions=(assess, reconsider, inc_trials, check_agreement, choose_result),
    procs=(diagnosisConsensus,),
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

    diagnosisConsensus.configure(
        backend=lambda a, i: mock_llm(a, i, min_delay=10.3, max_delay=20.2),
        trace=wt,
        timeout=600,
    )

    while True:
        wt.reset()
        print("Running diagnosis consensus (mock LLM)…")
        result = diagnosisConsensus(
            notes="Patient has fever, cough, and fatigue for 5 days.",
            diagnosis="Influenza",
        )
        wt.done()
        print(f"\nResult → {result}")
        print("Click ▶ Run again in the browser, or Ctrl-C to quit.")
        wt.wait_for_replay()
