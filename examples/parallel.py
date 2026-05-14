# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Minimal parallel-region example.

The orchestrator prepares a merge candidate. The test runner and security
checker work independently, and the committer waits for both results before
making the final decision. In ZipperChat, the two parallel branches are shown
with their branch labels while the global event order is still preserved.
"""

import time

from zippergen import Lifeline, Var, branch, parallel, pure, workflow


Orchestrator = Lifeline("Orchestrator")
TestRunner = Lifeline("TestRunner")
Security = Lifeline("Security")
Committer = Lifeline("Committer")

test_status = Var("test_status", str)
security_status = Var("security_status", str)
decision = Var("decision", str)


@pure
def run_tests(candidate: str) -> str:
    time.sleep(0.6)
    return f"tests passed for {candidate}"


@pure
def scan_security(candidate: str) -> str:
    time.sleep(0.3)
    return f"security cleared for {candidate}"


@pure
def decide_merge(candidate: str, tests: str, security: str) -> str:
    if "passed" in tests and "cleared" in security:
        return f"merge {candidate}"
    return f"rerun checks for {candidate}"


@workflow
def merge_candidate(candidate: str @ Orchestrator) -> str:
    Orchestrator(candidate) >> Committer(candidate)

    with parallel:
        with branch:
            Orchestrator(candidate) >> TestRunner(candidate)
            TestRunner: (test_status,) = run_tests(candidate)
            TestRunner(test_status) >> Committer(test_status)

        with branch:
            Orchestrator(candidate) >> Security(candidate)
            Security: (security_status,) = scan_security(candidate)
            Security(security_status) >> Committer(security_status)

    Committer: (decision,) = decide_merge(candidate, test_status, security_status)
    return decision @ Committer


if __name__ == "__main__":
    merge_candidate.configure(llms="mock", ui=True, timeout=30)
    print(merge_candidate(candidate="patch-17 on main@8fd2"))
    input("\nZipperChat is running at http://localhost:8765\nPress Enter to stop.\n")
