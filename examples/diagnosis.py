# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Medical Diagnosis Consensus — running example.

Direct transcription of Listing 1 / Listing 2 from the paper
"Provable Coordination for LLM Agents via Message Sequence Charts".

Two independent LLMs analyze patient notes and iterate until they agree
on a diagnosis verdict, or until MAX_ROUNDS is reached. LLM1 owns the
consensus loop.

Set the provider choice in ``__main__`` via ``diagnosisConsensus.configure``.
"""

from zippergen.syntax import (
    Lifeline, Var,
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
    LLM2(verdict) >> LLM1(other_verdict)
    LLM1: agreed = checkAgreement(verdict, other_verdict)

    # Consensus loop — owned by LLM1 (at most MAX_ROUNDS rounds)
    while (not agreed and trials < MAX_ROUNDS) @ LLM1:
        LLM1(verdict, reason) >> LLM2(other_verdict, other_reason)
        LLM2(verdict, reason) >> LLM1(other_verdict, other_reason)
        LLM1: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
        LLM2: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
        LLM2(verdict) >> LLM1(other_verdict)
        with LLM1:
            agreed = checkAgreement(verdict, other_verdict)
            trials = incTrials(trials)
    else:
        LLM1(verdict, reason) >> LLM2(other_verdict, other_reason)

    # Final result computed by LLM1, sent to User
    LLM1: result = chooseResult(verdict, agreed)
    LLM1(result) >> User(result)
    return result @ User


if __name__ == "__main__":
    USE_UI = True

    diagnosisConsensus.configure(
        llms="mock",
        ui=USE_UI,
        timeout=600,
    )
    result = diagnosisConsensus(
        notes=(
            "56-year-old woman presents with sudden shortness of breath and right-sided "
            "pleuritic chest pain for 8 hours. Heart rate 112, blood pressure 128/76, "
            "oxygen saturation 93% on room air, temperature 37.4 C. She returned from "
            "a 10-hour flight 2 days ago. History of breast cancer in remission and "
            "hypertension. Mild swelling and tenderness of the left calf noted on exam. "
            "No productive cough. Troponin negative. Chest X-ray shows no focal infiltrate. "
            "D-dimer elevated at 1.8 mg/L FEU."
        ),
        diagnosis="Pulmonary embolism",
    )
    print(f"\nResult → {result}")
    if USE_UI:
        input("ZipperChat is running at http://localhost:8765 . Press Enter to close. ")
