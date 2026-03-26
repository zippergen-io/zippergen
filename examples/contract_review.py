# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Contract Review — multi-agent legal analysis.

Four agents collaborate to review a contract for legal risks:

  User         — submits the contract, receives the final report
  Jurisdiction — reviews governing law, venue, and dispute resolution
  Liability    — reviews indemnification, limitation of liability, and warranties
  Confidentiality — reviews intellectual property, confidentiality, and non-compete clauses
  Orchestrator — consolidates findings and decides whether to escalate

The global protocol is written once and projected to each agent automatically.
No agent can deadlock or deviate from the protocol by construction.

AI Act relevance
----------------
Under the EU AI Act, high-risk AI systems must be transparent, auditable,
and produce traceable outputs. The global workflow below is the formal
specification of how this system coordinates — it can be read, audited, and
submitted to regulators directly. The projection guarantees that no agent
acts outside its declared role, and the deadlock-freedom theorem (Corollary 3.1)
guarantees the review always terminates.
"""

import os

from zippergen.syntax import Lifeline, Var
from zippergen.actions import llm
from zippergen.backends import make_openai_backend
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User         = Lifeline("User")
Jurisdiction = Lifeline("Jurisdiction")
Liability    = Lifeline("Liability")
Confidentiality = Lifeline("Confidentiality")
Orchestrator = Lifeline("Orchestrator")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

contract    = Var("contract",    str)   # input contract text

j_issues    = Var("j_issues",    str)   # jurisdictional issues found
j_critical  = Var("j_critical",  bool)  # whether any are critical
l_issues    = Var("l_issues",    str)
l_critical  = Var("l_critical",  bool)
cf_issues   = Var("cf_issues",   str)
cf_critical = Var("cf_critical", bool)

critical_found = Var("critical_found", bool)  # Orchestrator's escalation decision
summary        = Var("summary",        str)   # initial consolidated summary

context = Var("context", str)   # escalation context sent to specialists
j_deep  = Var("j_deep",  str)   # deep review findings per specialist
l_deep  = Var("l_deep",  str)
cf_deep = Var("cf_deep", str)

report = Var("report", str)     # final report returned to User

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@llm(
    system=(
        "You are a legal expert specialising in jurisdictional analysis. "
        "Review the contract for issues related to: governing law, venue, "
        "dispute resolution mechanisms, and applicable regulations. "
        "Flag any clauses that are ambiguous, one-sided, or potentially "
        "unenforceable."
    ),
    user=(
        "Contract:\n\n{contract}\n\n"
        "Return a concise summary of jurisdictional issues found (issues) "
        "and whether any of them are critical and require escalation (critical)."
    ),
    parse="json",
    outputs=(("issues", str), ("critical", bool)),
)
def analyze_jurisdiction(contract: str) -> None: ...


@llm(
    system=(
        "You are a legal expert specialising in liability and indemnification. "
        "Review the contract for issues related to: indemnification clauses, "
        "limitation of liability, warranty disclaimers, and risk allocation. "
        "Flag any clauses that expose one party to disproportionate risk."
    ),
    user=(
        "Contract:\n\n{contract}\n\n"
        "Return a concise summary of liability issues found (issues) "
        "and whether any of them are critical and require escalation (critical)."
    ),
    parse="json",
    outputs=(("issues", str), ("critical", bool)),
)
def analyze_liability(contract: str) -> None: ...


@llm(
    system=(
        "You are a legal expert specialising in intellectual property and "
        "confidentiality. Review the contract for issues related to: IP "
        "ownership and assignment, confidentiality obligations, non-compete "
        "and non-solicitation clauses, and data protection. "
        "Flag any clauses that could result in unintended IP transfer or "
        "excessive confidentiality burdens."
    ),
    user=(
        "Contract:\n\n{contract}\n\n"
        "Return a concise summary of IP and confidentiality issues found (issues) "
        "and whether any of them are critical and require escalation (critical)."
    ),
    parse="json",
    outputs=(("issues", str), ("critical", bool)),
)
def analyze_confidentiality(contract: str) -> None: ...


@llm(
    system=(
        "You are a senior legal counsel coordinating a multi-specialist contract "
        "review. You have received preliminary findings from three specialist "
        "reviewers. Assess the combined risk profile and decide whether the "
        "findings warrant an escalated deep review."
    ),
    user=(
        "Jurisdiction issues: {j_issues} (critical: {j_critical})\n"
        "Liability issues: {l_issues} (critical: {l_critical})\n"
        "Confidentiality issues: {cf_issues} (critical: {cf_critical})\n\n"
        "Return: a brief consolidated summary (summary) and whether an "
        "escalated deep review is required (critical_found)."
    ),
    parse="json",
    outputs=(("critical_found", bool), ("summary", str)),
)
def consolidate(
    j_issues: str, j_critical: bool,
    l_issues: str, l_critical: bool,
    cf_issues: str, cf_critical: bool,
) -> None: ...


@llm(
    system=(
        "You are a legal specialist conducting a deep-dive review. "
        "You have been asked to revisit your preliminary analysis in light "
        "of a consolidated risk summary from the coordinating counsel. "
        "Provide a detailed, actionable assessment."
    ),
    user=(
        "Original contract:\n\n{contract}\n\n"
        "Your preliminary findings: {issues}\n\n"
        "Consolidated risk context from coordinating counsel: {context}\n\n"
        "Provide a detailed deep-review assessment with specific clause "
        "references and recommended actions."
    ),
    parse="text",
    outputs=(("deep_findings", str),),
)
def deep_review(contract: str, issues: str, context: str) -> None: ...


@llm(
    system=(
        "You are a senior legal counsel producing a final contract review report "
        "after an escalated multi-specialist deep review. "
        "Structure your report clearly for a non-legal executive audience."
    ),
    user=(
        "Consolidated risk summary: {summary}\n\n"
        "Deep review — Jurisdiction: {j_deep}\n"
        "Deep review — Liability: {l_deep}\n"
        "Deep review — Confidentiality: {cf_deep}\n\n"
        "Produce a structured final report with: Executive Summary, "
        "Critical Issues (with clause references), Recommended Actions, "
        "and an overall Risk Rating (Low / Medium / High / Critical)."
    ),
    parse="text",
    outputs=(("report", str),),
)
def final_report_critical(summary: str, j_deep: str, l_deep: str, cf_deep: str) -> None: ...


@llm(
    system=(
        "You are a senior legal counsel producing a final contract review report. "
        "Structure your report clearly for a non-legal executive audience."
    ),
    user=(
        "Consolidated findings: {summary}\n\n"
        "Produce a structured final report with: Executive Summary, "
        "Issues Found (with clause references), Recommended Actions, "
        "and an overall Risk Rating (Low / Medium / High / Critical)."
    ),
    parse="text",
    outputs=(("report", str),),
)
def standard_report(summary: str) -> None: ...


# ---------------------------------------------------------------------------
# Global coordination protocol
# ---------------------------------------------------------------------------

@workflow
def contractReview(contract: str @ User) -> str:
    # --- Phase 1: distribute contract to all specialists ---
    User(contract) >> Jurisdiction(contract)
    User(contract) >> Liability(contract)
    User(contract) >> Confidentiality(contract)

    # --- Phase 2: independent specialist analysis (concurrent) ---
    Jurisdiction:    (j_issues, j_critical)   = analyze_jurisdiction(contract)
    Liability:       (l_issues, l_critical)   = analyze_liability(contract)
    Confidentiality: (cf_issues, cf_critical) = analyze_confidentiality(contract)

    # --- Phase 3: specialists report to Orchestrator ---
    Jurisdiction(j_issues, j_critical)       >> Orchestrator(j_issues, j_critical)
    Liability(l_issues, l_critical)          >> Orchestrator(l_issues, l_critical)
    Confidentiality(cf_issues, cf_critical)  >> Orchestrator(cf_issues, cf_critical)

    # --- Phase 4: Orchestrator consolidates and decides whether to escalate ---
    Orchestrator: (critical_found, summary) = consolidate(
        j_issues, j_critical,
        l_issues, l_critical,
        cf_issues, cf_critical,
    )

    # --- Phase 5: conditional escalation (owned by Orchestrator) ---
    if critical_found @ Orchestrator:
        # Broadcast escalation context to all specialists
        Orchestrator(summary) >> Jurisdiction(context)
        Orchestrator(summary) >> Liability(context)
        Orchestrator(summary) >> Confidentiality(context)

        # Deep review by each specialist in parallel
        Jurisdiction:    j_deep  = deep_review(contract, j_issues, context)
        Liability:       l_deep  = deep_review(contract, l_issues, context)
        Confidentiality: cf_deep = deep_review(contract, cf_issues, context)

        # Specialists send deep findings to Orchestrator
        Jurisdiction(j_deep)       >> Orchestrator(j_deep)
        Liability(l_deep)          >> Orchestrator(l_deep)
        Confidentiality(cf_deep)   >> Orchestrator(cf_deep)

        # Final report incorporating deep review
        Orchestrator: report = final_report_critical(summary, j_deep, l_deep, cf_deep)
    else:
        # Standard report — no escalation needed
        Orchestrator: report = standard_report(summary)

    # --- Phase 6: deliver report to User ---
    Orchestrator(report) >> User(report)
    return report @ User


# ---------------------------------------------------------------------------
# Sample contract (short NDA excerpt for demo purposes)
# ---------------------------------------------------------------------------

SAMPLE_CONTRACT = """
MUTUAL NON-DISCLOSURE AGREEMENT

This Agreement is entered into as of the Effective Date between AlphaCorp Ltd,
a company incorporated in Delaware, USA ("Disclosing Party") and BetaVentures
GmbH, a company incorporated in Munich, Germany ("Receiving Party").

1. CONFIDENTIAL INFORMATION
   "Confidential Information" means any information disclosed by either party
   that is designated as confidential or that reasonably should be understood
   to be confidential. This includes, without limitation, trade secrets,
   business plans, financial data, customer lists, and technical specifications.

2. OBLIGATIONS
   The Receiving Party shall: (a) hold all Confidential Information in strict
   confidence; (b) not disclose Confidential Information to third parties
   without prior written consent; (c) use Confidential Information solely
   for evaluating a potential business relationship.

3. INTELLECTUAL PROPERTY
   Any inventions, improvements, or works created by the Receiving Party using
   Confidential Information shall be assigned in full to the Disclosing Party.
   The Receiving Party waives all moral rights to such works worldwide and
   in perpetuity.

4. TERM AND TERMINATION
   This Agreement shall remain in effect for a period of ten (10) years from
   the Effective Date. Upon termination, the Receiving Party shall destroy
   all Confidential Information within 7 days.

5. GOVERNING LAW AND DISPUTE RESOLUTION
   This Agreement shall be governed by the laws of the State of Delaware, USA,
   without regard to conflict of law principles. Any disputes shall be resolved
   by binding arbitration in New York, conducted in English, under AAA rules.
   Each party irrevocably waives the right to a jury trial.

6. LIMITATION OF LIABILITY
   In no event shall either party be liable for indirect, incidental, special,
   or consequential damages. The Disclosing Party's total liability shall not
   exceed USD 500. The Receiving Party's liability is unlimited.

7. NON-COMPETE
   During the term and for five (5) years thereafter, the Receiving Party shall
   not engage in any business activity that competes, directly or indirectly,
   with any current or future business of the Disclosing Party, anywhere in
   the world.
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    USE_UI = True

    contractReview.configure(
        # llms="mock",
        llms={
            "Jurisdiction":    make_openai_backend(api_key=os.environ["OPENAI_API_KEY_J"]),
            "Liability":       make_openai_backend(api_key=os.environ["OPENAI_API_KEY_L"]),
            "Confidentiality": make_openai_backend(api_key=os.environ["OPENAI_API_KEY_C"]),
            "Orchestrator":    "mistral",
        },
        ui=USE_UI,
        timeout=600,
    )

    result = contractReview(contract=SAMPLE_CONTRACT)
    print(f"\n{'='*60}")
    print("CONTRACT REVIEW REPORT")
    print('='*60)
    print(result)

    if USE_UI:
        input("\nZipperChat is running at http://localhost:8765 . Press Enter to close. ")
