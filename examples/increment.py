# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false
"""
Increment — minimal tutorial example.

User sends a number to Adder, Adder increments it by one,
and sends the result back.
"""

from zippergen.syntax import Int, Lifeline, Var, Program
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

number = Var("number", Int)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@pure()
def inc(x: Int) -> Int:
    return x + 1

# ---------------------------------------------------------------------------
# Proc
# ---------------------------------------------------------------------------

@proc
def increment(number: Int @ User) -> Int:
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

if __name__ == "__main__":
    import time
    from zipperchat import WebTrace

    wt = WebTrace(program.lifelines).start()
    time.sleep(0.3)

    increment.configure(trace=wt, timeout=10)

    while True:
        wt.reset()
        print("Running increment…")
        result = increment(number=1)
        wt.done()
        print(f"\nResult → {result}")
        print("Click ▶ Run again in the browser, or Ctrl-C to quit.")
        wt.wait_for_replay()
