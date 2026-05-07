# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Causal-past guard example: stale status through two relays.

A device reports whether it is on through two independent relay processes.
The indicator receives a newer ``on=True`` status through Relay_2 before an
older delayed ``on=False`` status through Relay_1.  A local guard on the
indicator's ``on`` variable would be stale after the delayed message; the
causal guard below keeps reading the latest causally visible Device status.
"""

from zippergen import Lifeline, Var, workflow
from zippergen.actions import pure
from zippergen import Y, atom

Device = Lifeline("Device")
Relay_1 = Lifeline("Relay_1")
Relay_2 = Lifeline("Relay_2")
Indicator = Lifeline("Indicator")

on = Var("on",  bool)

latest_device_on = Y[Device](
    atom(lambda env: env.get("on", False), src="on")
)

@pure
def set_status(status: bool) -> bool:
    return status

@workflow
def status_indicator() -> tuple:
    Device: on = set_status(True)
    Device(on) >> Relay_1(on)
    Relay_1(on) >> Indicator(on)
    if latest_device_on @ Indicator:
        pass
    else:
        pass

    Device: on = set_status(False)
    Device(on) >> Relay_1(on)
    Device: on = set_status(True)
    Device(on) >> Relay_2(on)
    Relay_2(on) >> Indicator(on)
    if latest_device_on @ Indicator:
        pass
    else:
        pass

    Relay_1(on) >> Indicator(on)
    if latest_device_on @ Indicator:
        pass
    else:
        pass


if __name__ == "__main__":
    status_indicator.configure(llms="mock", ui=True)
    status_indicator()
    input("\nZipperChat is running at http://localhost:8765\nPress Enter to stop.\n")
