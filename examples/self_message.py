# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Self-message — minimal example of the self-send feature.

A sends a number to B.  B doubles it, then uses a self-send to rename the
result variable before sending it back to A.
"""

from zippergen.syntax import Lifeline, Var
from zippergen.actions import pure
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

A = Lifeline("A")
B = Lifeline("B")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

n      = Var("n",      int)
doubled = Var("doubled", int)
result = Var("result", int)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@pure
def double(x: int) -> int:
    return x * 2

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow
def self_message(n: int @ A) -> int:
    A(n) >> B(n)
    with B:
        doubled = double(n)
    B(doubled) >> B(result)   # self-send: rename `doubled` to `result`
    B(result) >> A(result)
    return result @ A

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    self_message.configure(llms="mock", ui=True)
    r = self_message(n=3)
    print(f"\nResult: {r}")   # expected: 6
    input("\nPress Enter to exit...")
