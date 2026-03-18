"""
Hello World — minimal tutorial example.

User sends a name to Greeter, Greeter builds a personalised greeting,
and sends it back. Demonstrates the core DSL in five lines:
one input, one message each way, one local action, one return value.
"""

from zippergen.syntax import Text, Lifeline, Var, Program
from zippergen.actions import pure
from zippergen.builder import proc

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User    = Lifeline("User")
Greeter = Lifeline("Greeter")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

name     = Var("name",     Text)
greeting = Var("greeting", Text)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@pure()
def greet(name: Text) -> Text:
    return f"Hello, {name}!"

# ---------------------------------------------------------------------------
# Proc
# ---------------------------------------------------------------------------

@proc
def hello(name: Text @ User) -> Text:
    User(name) >> Greeter(name)
    Greeter: greeting = greet(name)
    Greeter(greeting) >> User(greeting)
    return greeting @ User

# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

program = Program(
    lifelines=(User, Greeter),
    actions=(greet,),
    procs=(hello,),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from zipperchat import WebTrace

    wt = WebTrace(program.lifelines).start()
    time.sleep(0.3)   # give the browser a moment to connect

    hello.configure(trace=wt, timeout=10)

    while True:
        wt.reset()
        print("Running hello world…")
        result = hello(name="World")
        wt.done()
        print(f"\nResult → {result}")
        print("Click ▶ Run again in the browser, or Ctrl-C to quit.")
        wt.wait_for_replay()
