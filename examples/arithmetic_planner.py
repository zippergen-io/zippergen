# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Arithmetic Planner — LLM-generated expression evaluation.

The planner receives an arithmetic expression as text and generates a
workflow that evaluates it step by step using a Calculator lifeline.

The planner must figure out on its own:
  - how to decompose the expression into atomic operations
  - in what order to apply them, respecting data dependencies

ZipperGen validates the generated workflow structurally before running it.
"""

from zippergen.syntax import Lifeline, Var
from zippergen.actions import planner, pure
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User        = Lifeline("User")
Planner     = Lifeline("Planner")
Calculator1 = Lifeline("Calculator1")
Calculator2 = Lifeline("Calculator2")
Calculator3 = Lifeline("Calculator3")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

expression = Var("expression", str)
result     = Var("result",     str)

# ---------------------------------------------------------------------------
# Calculator actions — pure arithmetic on floats
# ---------------------------------------------------------------------------

@pure
def add(a: float, b: float) -> float:
    return a + b


@pure
def subtract(a: float, b: float) -> float:
    return a - b


@pure
def multiply(a: float, b: float) -> float:
    return a * b


@pure
def divide(a: float, b: float) -> float:
    return a / b


@pure
def identity(x: float) -> float:
    return x


@pure
def is_zero(x: float) -> bool:
    return x == 0.0



# ---------------------------------------------------------------------------
# Planner action
# ---------------------------------------------------------------------------

@planner(
    description=(
        "An arithmetic planner that evaluates an expression with maximum parallelism. "
        "It identifies independent subexpressions and evaluates them concurrently on "
        "separate Calculator lifelines, then combines the results. "
        "If the expression is not defined (division by 0 somewhere), return 0 as result."
    ),
    actions=[add, subtract, multiply, divide, identity, is_zero],
    lifelines=[Calculator1, Calculator2, Calculator3],
    allow=["if"],
    max_retries=8,
)
def evaluate(expression: str) -> str: ...


# ---------------------------------------------------------------------------
# Outer workflow
# ---------------------------------------------------------------------------

@workflow
def arithmetic_planner(expression: str @ User) -> str:
    User(expression) >> Planner(expression)
    Planner: result = evaluate(expression)
    Planner(result) >> User(result)
    return result @ User


# ---------------------------------------------------------------------------
# Entry point — two test cases
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    USE_UI = True

    import os
    from zippergen.backends import make_openai_backend
    arithmetic_planner.configure(
        llms={"Planner": make_openai_backend(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o")},
        ui=USE_UI,
        timeout=120,
    )


    expr = sys.argv[1] if len(sys.argv) > 1 else "(2 - 4) * (2 + 3) + (3 / (3 - 2))"
    print(f"Expression: {expr}")
    result = arithmetic_planner(expression=expr)
    print(f"\nResult: {result}")
    if USE_UI:
        input("\nZipperChat is running at http://localhost:8765 . Press Enter to close. ")
