# ZipperGen

**Zip** (our mascot) keeps your agents in line — literally. ZipperGen is a Python DSL and runtime for structured multi-agent LLM coordination, grounded in the theory of Message Sequence Charts. You write a single global protocol; ZipperGen projects it onto each agent and runs them concurrently, with deadlock-freedom guaranteed by construction.

No orchestrator polling. No spaghetti callbacks. Just Zip, zipping things together.

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

The formal foundation is in the paper *"Provable Coordination for LLM Agents via Message Sequence Charts"*.
