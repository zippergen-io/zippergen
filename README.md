# ZipperGen

**Zip** (our mascot) literally keeps your agents in line. ZipperGen is a Python DSL and runtime for structured multi-agent LLM coordination, grounded in the theory of Message Sequence Charts. You write a single global protocol; ZipperGen projects it onto each agent and runs them concurrently, with deadlock-freedom guaranteed by construction.

## Quick start

```bash
git clone https://github.com/zippergen-io/zippergen.git
cd zippergen
pip install -e .
```

Python 3.11 or later required.

## Hello, World!

Here is the smallest possible ZipperGen program. `User` sends a number to `Adder`, `Adder` increments it, and the result is returned to the caller:

```python
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
```

- `User(number) >> Adder(number)` — `User` sends `number` to `Adder`.
- `Adder: number = inc(number)` — `Adder` runs a local action.
- `return number @ User` — declares `User` as the lifeline that owns the result.

ZipperGen projects this global protocol onto each agent and runs them in parallel threads. Deadlock-freedom is guaranteed by construction.

Run it with ZipperChat to see the live MSC diagram in your browser:

```bash
python examples/increment.py
```

## See it in action

The `diagnosis` example runs two LLMs through a medical consensus protocol until they agree:

```bash
python examples/diagnosis.py
```

Then visit **http://localhost:8765** — ZipperChat will show the agents exchanging messages in real time as a message sequence chart. The example uses a mock LLM backend by default (no API key needed), with simulated latency so you can watch Zip do its thing.

![ZipperChat screenshot](assets/zipperchat-screenshot.jpg)
*ZipperChat — live MSC diagram for the reviewed-execution example (Planner, Reviewer, Orchestrator, Executor)*

## How it works

ZipperGen programs are *global coordination protocols* — you describe what messages flow between which agents and who owns each decision. ZipperGen automatically projects the global protocol onto per-agent local programs and executes them in parallel threads with FIFO message queues.

Control structures use native `if`/`while` with `@ Lifeline` to name the agent that owns the decision:

```python
while (not agreed) @ LLM1:   # LLM1 owns the loop condition
    LLM1(verdict1, reason1) >> LLM2(v1, r1)
    LLM2(verdict2, reason2) >> LLM1(v2, r2)
    LLM1: (verdict1, reason1) = reconsider(n1, d1, verdict1, reason1, v2, r2)
    LLM2: (verdict2, reason2) = reconsider(n2, d2, verdict2, reason2, v1, r1)
    LLM2(verdict2) >> LLM1(verdict2)
    LLM1: agreed = check_agreement(verdict1, verdict2)
else:                          # else = exit body (runs once on loop exit)
    LLM1(verdict1, reason1) >> LLM2(v1, r1)
```

The `@ LLM1` annotation mirrors the paper's notation `c@B` — it tells ZipperGen which agent evaluates the condition and broadcasts control messages to the others.

The formal foundation is in the paper *"Provable Coordination for LLM Agents via Message Sequence Charts"*.

## Wiring a real LLM

Pass any callable as `backend` to `configure()`:

```python
def my_backend(action, inputs):
    # call OpenAI / Anthropic / etc.
    return {"verdict": True, "reason": "..."}

my_proc.configure(backend=my_backend, timeout=60)
result = my_proc(input="...")
```
