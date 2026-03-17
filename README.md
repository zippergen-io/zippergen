# ZipperGen

**Zip** (our mascot) keeps your agents in line — literally. ZipperGen is a Python DSL and runtime for structured multi-agent LLM coordination, grounded in the theory of Message Sequence Charts. You write a single global protocol; ZipperGen projects it onto each agent and runs them concurrently, with deadlock-freedom guaranteed by construction.

No orchestrator polling. No spaghetti callbacks. Just Zip, zipping things together.

![ZipperChat screenshot](assets/zipperchat-screenshot.jpg)
*ZipperChat — live MSC diagram for the reviewed-execution example (Planner, Reviewer, Orchestrator, Executor)*

## Quick start

```bash
git clone https://github.com/zippergen-io/zippergen.git
cd zippergen
pip install -e .
```

Python 3.11 or later required.

## See it in action

The `diagnosis` example runs two LLMs through a medical consensus protocol until they agree. It opens a live diagram in your browser:

```bash
python examples/diagnosis.py
```

Then visit **http://localhost:8765** — ZipperChat will show the agents exchanging messages in real time as a message sequence chart. The example uses a mock LLM backend by default (no API key needed), with simulated latency so you can watch Zip do its thing.

## Wiring a real LLM

Pass any callable as `llm_backend` to `run()`:

```python
from zippergen.runtime import run

def my_backend(action, inputs):
    # call OpenAI / Anthropic / etc.
    return {"verdict": True, "reason": "..."}

run(proc, lifelines, initial_envs, llm_backend=my_backend)
```

## How it works

ZipperGen programs are *global coordination protocols* — you describe what messages flow between which agents and who owns each decision. ZipperGen automatically projects the global protocol onto per-agent local programs and executes them in parallel threads with FIFO message queues.

Programs are written as plain Python functions decorated with `@proc`. Control structures use native `if`/`while` with `@ Lifeline` to name the agent that owns the decision:

```python
@proc
def diagnosisConsensus(notes: Text, diagnosis: Text) -> Text:
    msg(User, (notes, diagnosis), LLM1, (n1, d1))
    msg(User, (notes, diagnosis), LLM2, (n2, d2))
    act(LLM1, assess, (n1, d1), (verdict1, reason1))
    act(LLM2, assess, (n2, d2), (verdict2, reason2))

    while (not agreed) @ LLM1:          # LLM1 owns the loop condition
        msg(LLM1, (verdict1,), LLM2, (v1,))
        msg(LLM2, (verdict2,), LLM1, (v2,))
        act(LLM1, check_agreement, (verdict1, verdict2), (agreed,))
    else:                                # else = exit body (runs once on exit)
        msg(LLM1, (verdict1,), User, (result,))
```

The `@ LLM1` annotation mirrors the paper's notation `c@B` — it tells ZipperGen which agent evaluates the condition and broadcasts control messages to the others. Deadlock-freedom is guaranteed by construction.

The formal foundation is in the paper *"Provable Coordination for LLM Agents via Message Sequence Charts"*.
