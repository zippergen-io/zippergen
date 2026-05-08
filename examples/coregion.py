# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Minimal co-region example with nondeterministic arrival order.

The collector needs one report from each analyst, but it does not care which
report arrives first. Each analyst first spends a random amount of time preparing
the report, so the concrete receive order at the collector can vary between
runs. The co-region projects to ordinary sends for the analysts and one
receive-any block for the collector.
"""

import random
import time

from zippergen import Lifeline, Var, coregion, pure, workflow


Analyst_A = Lifeline("Analyst_A")
Analyst_B = Lifeline("Analyst_B")
Collector = Lifeline("Collector")

a_report = Var("a_report", str)
b_report = Var("b_report", str)


@pure
def prepare(report: str) -> str:
    time.sleep(random.uniform(0.0, 0.5))
    return report


@workflow
def collect_reports(a: str @ Analyst_A, b: str @ Analyst_B) -> tuple:
    with Analyst_A:
        a = prepare(a)
    with Analyst_B:
        b = prepare(b)
    with coregion:
        Analyst_A(a) >> Collector(a_report)
        Analyst_B(b) >> Collector(b_report)
    return (a_report @ Collector, b_report @ Collector)


if __name__ == "__main__":
    collect_reports.configure(llms="mock", ui=True)
    print(collect_reports(a="ready from A", b="ready from B"))
    input("\nZipperChat is running at http://localhost:8765\nPress Enter to stop.\n")
