# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false
"""
Medical Diagnosis Consensus — running example.

Direct transcription of Listing 1 / Listing 2 from the paper
"Provable Coordination for LLM Agents via Message Sequence Charts".

Two independent LLMs analyze patient notes and iterate until they agree
on a diagnosis verdict. LLM1 owns the consensus loop.
"""

from zippergen.syntax import (
    Lifeline, Var,
    Program,
    pp,
)
from zippergen.actions import llm, pure
from zippergen.builder import proc

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User = Lifeline("User")
LLM1 = Lifeline("LLM1")
LLM2 = Lifeline("LLM2")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

# Proc inputs
notes     = Var("notes",     str)
diagnosis = Var("diagnosis", str)

# Local copies of inputs
n1 = Var("n1", str)
n2 = Var("n2", str)
d1 = Var("d1", str)
d2 = Var("d2", str)

# Verdicts (bool) and reasoning (str), local to each LLM
verdict1 = Var("verdict1", bool)
reason1  = Var("reason1",  str)
verdict2 = Var("verdict2", bool)
reason2  = Var("reason2",  str)

# Received copies for reconsideration
v1 = Var("v1", bool)
r1 = Var("r1", str)
v2 = Var("v2", bool)
r2 = Var("r2", str)

# Control and output
agreed = Var("agreed", bool, default=False)
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
def checkAgreement(v1: bool, v2: bool) -> bool:
    return v1 == v2


@pure
def chooseResult(v: bool, agreed: bool) -> str:
    return ("true" if v else "false") if agreed else "unknown"


# Aliases to match the paper's camelCase names used in ActStmt below
check_agreement = checkAgreement
choose_result   = chooseResult

# ---------------------------------------------------------------------------
# Proc — direct translation of Listing 1
# ---------------------------------------------------------------------------

@proc
def diagnosisConsensus(notes: str @ User, diagnosis: str @ User) -> str:
    # Distribute notes to both LLMs
    User(notes, diagnosis) >> LLM1(n1, d1)
    User(notes, diagnosis) >> LLM2(n2, d2)

    # Independent initial assessments
    LLM1: (verdict1, reason1) = assess(n1, d1)
    LLM2: (verdict2, reason2) = assess(n2, d2)

    # Consensus loop — owned by LLM1
    while (not agreed) @ LLM1:
        LLM1(verdict1, reason1) >> LLM2(v1, r1)
        LLM2(verdict2, reason2) >> LLM1(v2, r2)
        LLM1: (verdict1, reason1) = reconsider(n1, d1, verdict1, reason1, v2, r2)
        LLM2: (verdict2, reason2) = reconsider(n2, d2, verdict2, reason2, v1, r1)
        LLM2(verdict2) >> LLM1(verdict2)
        LLM1: agreed = check_agreement(verdict1, verdict2)
    else:
        LLM1(verdict1, reason1) >> LLM2(v1, r1)

    # Final result computed by LLM1, sent to User
    LLM1: result = choose_result(verdict1, agreed)
    LLM1(result) >> User(result)
    return result @ User


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

program = Program(
    lifelines=(User, LLM1, LLM2),
    actions=(assess, reconsider, check_agreement, choose_result),
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
