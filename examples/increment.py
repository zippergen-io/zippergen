# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Increment — minimal tutorial example.

User sends a number to Compute, Compute increments it by one and then doubles it,
and sends the result back.
"""

from zippergen.syntax import Lifeline, Var, Program
from zippergen.actions import pure
from zippergen.builder import proc

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User  = Lifeline("User")
Compute = Lifeline("Compute")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

number = Var("number", int)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@pure
def inc(x: int) -> int:
    return x + 1

@pure
def double(x: int) -> int:
    return x * 2

# ---------------------------------------------------------------------------
# Proc
# ---------------------------------------------------------------------------

@proc
def increment(number: int @ User) -> int:
    User(number) >> Compute(number)
    with Compute:
        number = inc(number)
        number = double(number)
    Compute(number) >> User(number)
    return number @ User

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

result = increment(number=1)
print(f"\nResult: {result}")
