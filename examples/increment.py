# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false
"""
Increment — minimal tutorial example.

User sends a number to Adder, Adder increments it by one,
and sends the result back.
"""

from zippergen.syntax import Lifeline, Var, Program
from zippergen.actions import pure
from zippergen.builder import proc

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User  = Lifeline("User")
Adder = Lifeline("Adder")

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

# ---------------------------------------------------------------------------
# Proc
# ---------------------------------------------------------------------------

@proc
def increment(number: int @ User) -> int:
    User(number) >> Adder(number)
    Adder: number = inc(number)
    Adder(number) >> User(number)
    return number @ User

# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

program = Program(
    lifelines=(User, Adder),
    actions=(inc,),
    procs=(increment,),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

result = increment(number=1)
print(f"\nResult: {result}")
