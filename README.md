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

Here is the smallest possible ZipperGen program. `User` sends a number to `Compute`, `Compute` increments it and doubles it, and the result is returned to the caller:

```python
from zippergen.syntax import Lifeline, Var
from zippergen.actions import pure
from zippergen.builder import workflow

User  = Lifeline("User")
Compute = Lifeline("Compute")

number = Var("number", int)

@pure
def inc(x: int) -> int:
    return x + 1

@pure
def double(x: int) -> int:
    return x * 2

@workflow
def increment(number: int @ User) -> int:
    User(number) >> Compute(number)
    with Compute:
        number = inc(number)
        number = double(number)
    Compute(number) >> User(number)
    return number @ User

result = increment(number=1)   # → 4
```

- `User(number) >> Compute(number)` — `User` sends `number` to `Compute`.
- `with Compute:` — a block of consecutive local actions on `Compute`.
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
while (not agreed and trials < MAX_ROUNDS) @ LLM1:   # LLM1 owns the loop condition
    LLM1(verdict, reason) >> LLM2(other_verdict, other_reason)
    LLM2(verdict, reason) >> LLM1(other_verdict, other_reason)
    LLM1: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
    LLM2: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
    LLM2(verdict) >> LLM1(other_verdict)
    with LLM1:
        agreed = checkAgreement(verdict, other_verdict)
        trials = incTrials(trials)
else:                                                  # else = exit body (runs once on loop exit)
    LLM1(verdict, reason) >> LLM2(other_verdict, other_reason)
```

The `@ LLM1` annotation mirrors the paper's notation `c@B` — it tells ZipperGen which agent evaluates the condition and broadcasts control messages to the others.

The formal foundation is in the forthcoming paper *"Provable Coordination for LLM Agents via Message Sequence Charts"*.

## Wiring a real LLM

Pass any callable as `backend` to `configure()`:

```python
def my_backend(action, inputs):
    # call OpenAI / Anthropic / etc.
    return {"verdict": True, "reason": "..."}

my_workflow.configure(backend=my_backend, timeout=60)
result = my_workflow(input="...")
```
