# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Dashboard mode - several independent top-level workflow runs in one ZipperChat.

The two workflows below are ordinary ZipperGen workflows. They are configured
with the same WebTrace.dashboard(), then launched from Python threads. ZipperChat
shows each run as its own top-level group.
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

Operator = Lifeline("Operator")
Sensor = Lifeline("Sensor")
Monitor = Lifeline("Monitor")


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

topic = Var("topic", str)
draft = Var("draft", str)
accepted = Var("accepted", bool)
report = Var("report", str)

device = Var("device", str)
reading = Var("reading", str)
status = Var("status", str)


# ---------------------------------------------------------------------------
# Pure actions
# ---------------------------------------------------------------------------

@pure
def draft_report(t: str) -> str:
    time.sleep(0.6)
    return f"Draft report about {t}"


@pure
def approve_report(text: str) -> bool:
    time.sleep(0.3)
    return "Draft" in text


@pure
def revise_report(text: str) -> str:
    time.sleep(0.4)
    return f"{text} (revised)"


@pure
def read_device(name: str) -> str:
    time.sleep(0.2)
    return f"{name}: green"


@pure
def summarize_reading(value: str) -> str:
    time.sleep(0.5)
    return f"Status update: {value}"


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------

@workflow
def report_review(topic: str @ Client) -> str:
    Client(topic) >> Writer(topic)
    Writer: (draft,) = draft_report(topic)
    Writer(draft) >> Reviewer(draft)
    Reviewer: (accepted,) = approve_report(draft)
    if accepted @ Reviewer:
        Reviewer(draft) >> Client(report)
    else:
        Reviewer(draft) >> Writer(draft)
        Writer: (report,) = revise_report(draft)
        Writer(report) >> Client(report)
    return report @ Client


@workflow
def device_check(device: str @ Operator) -> str:
    Operator(device) >> Sensor(device)
    Sensor: (reading,) = read_device(device)
    Sensor(reading) >> Monitor(reading)
    Monitor: (status,) = summarize_reading(reading)
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
