# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Nested dashboard mode - several top-level workflow runs, each with subworkflows.

Run this file and open ZipperChat. The dashboard shows two independent
top-level workflow runs. Each top-level run contains a nested workflow, so this
also exercises nested paths below separate dashboard roots.
"""

from threading import Thread
import time

from zipperchat import WebTrace
from zippergen.actions import pure
from zippergen.builder import workflow
from zippergen.syntax import Lifeline, Var


# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

Client = Lifeline("Client")
Writer = Lifeline("Writer")
Reviewer = Lifeline("Reviewer")
Editor = Lifeline("Editor")

Operator = Lifeline("Operator")
Sensor = Lifeline("Sensor")
Monitor = Lifeline("Monitor")
Inspector = Lifeline("Inspector")


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

topic = Var("topic", str)
draft = Var("draft", str)
report = Var("report", str)
approved = Var("approved", bool)

device = Var("device", str)
reading = Var("reading", str)
classification = Var("classification", str)
status = Var("status", str)


# ---------------------------------------------------------------------------
# Pure actions
# ---------------------------------------------------------------------------

@pure
def draft_report(t: str) -> str:
    time.sleep(0.4)
    return f"Draft report about {t}"


@pure
def polish_report(text: str) -> str:
    time.sleep(0.5)
    return f"{text} (polished)"


@pure
def approve_report(text: str) -> bool:
    time.sleep(0.2)
    return "polished" in text


@pure
def read_device(name: str) -> str:
    time.sleep(0.3)
    return f"{name}: temperature normal"


@pure
def classify_reading(value: str) -> str:
    time.sleep(0.5)
    return "normal" if "normal" in value else "needs attention"


@pure
def summarize_status(value: str) -> str:
    time.sleep(0.2)
    return f"Device status: {value}"


# ---------------------------------------------------------------------------
# Subworkflows
# ---------------------------------------------------------------------------

@workflow
def edit_report(draft: str @ Editor) -> str:
    Editor: (report,) = polish_report(draft)
    return report @ Editor


@workflow
def inspect_reading(reading: str @ Inspector) -> str:
    Inspector: (classification,) = classify_reading(reading)
    return classification @ Inspector


# ---------------------------------------------------------------------------
# Top-level workflows
# ---------------------------------------------------------------------------

@workflow
def report_review(topic: str @ Client) -> str:
    Client(topic) >> Writer(topic)
    Writer: (draft,) = draft_report(topic)
    Writer: (report,) = edit_report(draft @ Editor)
    Writer(report) >> Reviewer(report)
    Reviewer: (approved,) = approve_report(report)
    if approved @ Reviewer:
        Reviewer(report) >> Client(report)
    else:
        Reviewer(draft) >> Client(report)
    return report @ Client


@workflow
def device_check(device: str @ Operator) -> str:
    Operator(device) >> Sensor(device)
    Sensor: (reading,) = read_device(device)
    Sensor: (classification,) = inspect_reading(reading @ Inspector)
    Sensor(classification) >> Monitor(classification)
    Monitor: (status,) = summarize_status(classification)
    Monitor(status) >> Operator(status)
    return status @ Operator


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dashboard = WebTrace.dashboard().start()

    report_review.configure(ui=True, trace=dashboard, timeout=30)
    device_check.configure(ui=True, trace=dashboard, timeout=30)

    def run_report() -> None:
        result = report_review(topic="incident response")
        print(f"Report result -> {result}")

    def run_device_check() -> None:
        result = device_check(device="pump-7")
        print(f"Device result -> {result}")

    t1 = Thread(target=run_report)
    t2 = Thread(target=run_device_check)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    input("\nZipperChat is running at http://localhost:8765\nPress Enter to stop.\n")
