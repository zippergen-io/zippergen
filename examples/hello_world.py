from zippergen.syntax import Int, Lifeline, Var
from zippergen.actions import pure
from zippergen.builder import proc

User  = Lifeline("User")
Adder = Lifeline("Adder")

number = Var("number", Int)

@pure
def inc(x: Int) -> Int:
    return x + 1

@proc
def increment(number: Int @ User) -> Int:
    User(number) >> Adder(number)
    Adder: number = inc(number)
    Adder(number) >> User(number)
    return number @ User

result = increment(number=1)   # → 2