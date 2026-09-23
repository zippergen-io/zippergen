"""Onboard Alice and Bob using fresh Employee endpoints.

The arrival and confirmation actions are local fakes. Run directly for a
SQLite-backed demonstration without credentials or external side effects.
"""
from zippergen import Lifeline, Var, fragment, pure, run_sqlite, spawn, workflow

HR, Employee = Lifeline("HR"), Lifeline("Employee")
index = Var("index", int, default=0)
hire = Var("hire", str)
details = Var("details", str)
response = Var("response", str)
confirmation = Var("confirmation", str)


@pure
def next_hire(index: int) -> str:
    return ("Alice", "Bob")[index]


@pure
def confirm(details: str) -> str:
    return "Confirmed: " + details


@pure
def advance(index: int) -> int:
    return index + 1


@fragment
def Onboarding():
    HR(hire) >> Employee(details)
    Employee: response = confirm(details)
    Employee(response) >> HR(confirmation)


@workflow
def onboard_employees() -> int:
    while (index < 2) @ HR:
        HR: hire = next_hire(index)
        spawn(
            Onboarding,
            at=HR,
            retain=(HR,),
            fresh=(HR.hire, HR.confirmation),
        )
        HR: index = advance(index)
    return index @ HR


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="zippergen-onboarding-") as directory:
        result = run_sqlite(
            onboard_employees,
            store_path=str(Path(directory) / "run.sqlite"),
            verbose=True,
        )
        print(f"Dispatched {result} hires; all onboarding sessions finished.")
