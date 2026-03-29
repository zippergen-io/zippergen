# ZipperGen

**Zip** (our mascot) literally keeps your agents in line.

ZipperGen is a Python DSL and runtime for structured multi-agent LLM coordination. You write a single **global protocol** — who sends what to whom, who runs which LLM, who owns each decision. ZipperGen projects it onto each agent automatically and runs them concurrently.

Because coordination is derived by projection rather than left to each agent, the global protocol is also a complete audit trail — and **deadlock is impossible by construction**.

## Quick start

```bash
git clone https://github.com/zippergen-io/zippergen.git
cd zippergen
pip install -e .
```

Python 3.11 or later required. No external dependencies — stdlib only.

## Hello, World!

`User` sends a number to `Compute`, which increments and doubles it:

```python
from zippergen.syntax import Lifeline, Var
from zippergen.actions import pure
from zippergen.builder import workflow

User    = Lifeline("User")
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

ZipperGen projects this global protocol onto each agent and runs them in parallel threads.

## See it in action

Examples ship with the repo. The first two work out of the box with the built-in mock backend — no API key needed.

```bash
python examples/diagnosis.py       # two LLMs reach consensus iteratively
python examples/contract_review.py # four agents review a contract in parallel
python examples/planner.py         # LLM designs and runs its own sub-workflow (needs OPENAI_API_KEY)
```

Open **http://localhost:8765** to watch the agents exchange messages in real time as a message sequence chart.

![ZipperChat screenshot](assets/zipperchat-screenshot.png)

## How it works

ZipperGen programs are *global coordination protocols*: you describe what messages flow between which agents and who owns each decision. ZipperGen projects the global protocol onto per-agent local programs and executes them in parallel threads with FIFO message queues.

### Diagnosis consensus

Two LLMs independently assess a case, then iterate until they agree or a round limit is reached:

```python
@workflow
def diagnosisConsensus(notes: str @ User, diagnosis: str @ User) -> str:
    # Distribute inputs to both LLMs
    User(notes, diagnosis) >> LLM1(notes, diagnosis)
    User(notes, diagnosis) >> LLM2(notes, diagnosis)

    # Independent initial assessments
    LLM1: (verdict, reason) = assess(notes, diagnosis)
    LLM2: (verdict, reason) = assess(notes, diagnosis)

    # Consensus loop — owned by LLM1 (at most MAX_ROUNDS rounds)
    while (not agreed and trials < MAX_ROUNDS) @ LLM1:
        LLM1(verdict, reason) >> LLM2(other_verdict, other_reason)
        LLM2(verdict, reason) >> LLM1(other_verdict, other_reason)
        LLM1: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
        LLM2: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
        LLM2(verdict) >> LLM1(other_verdict)
        with LLM1:
            agreed = checkAgreement(verdict, other_verdict)
            trials = incTrials(trials)

    # Final result computed by LLM1, returned to User
    LLM1: result = chooseResult(verdict, agreed)
    LLM1(result) >> User(result)
    return result @ User
```

`while cond @ LLM1` means LLM1 owns the loop guard and broadcasts the decision each iteration. `if cond @ Owner` works the same way for conditionals. ZipperGen figures out which other agents need to receive the decision and generates the control messages automatically.

## Dynamic planning

For tasks where the coordination structure itself isn't known in advance, `@planner` delegates workflow design to an LLM at runtime. The planner LLM receives the task description and available lifelines, generates a complete sub-workflow, and ZipperGen validates and executes it.

```python
from zippergen.actions import planner

@planner(
    description="A workflow planner for professional writing tasks.",
    actions=[],          # pre-defined action vocabulary (empty = LLM writes everything)
    lifelines=[Worker1, Worker2],
    allow=["llm"],       # permit the LLM to define new @llm actions
)
def write_document(request: str, job_desc: str, cv_sketch: str) -> str: ...
```

The decorated function slots into a `@workflow` exactly like any other action:

```python
@workflow
def openPlannerAgent(request: str @ User, job_desc: str @ User, cv_sketch: str @ User) -> str:
    User(request, job_desc, cv_sketch) >> Planner(request, job_desc, cv_sketch)
    Planner: result = write_document(request, job_desc, cv_sketch)
    Planner(result) >> User(result)
    return result @ User
```

The `description` is all the user provides. The runtime builds the full hidden system prompt automatically: it describes the available workers, injects the DSL rules, and lists the allow extensions. At runtime, the planner LLM synthesises a sub-workflow tailored to the request — for example, assigning Worker1 to write a first draft, Worker2 to critique it, and Worker1 to revise. The generated sub-workflow is structurally validated before execution: it must start with the Planner sending inputs to its workers and end with a worker returning the result to the Planner. ZipperChat lets you drill into the sub-workflow to see its MSC alongside the outer one.

**`allow` controls what the planner LLM may define:**

| `allow` value | Effect |
|---|---|
| `[]` (default) | LLM may only use the pre-defined `actions` vocabulary |
| `["pure"]` | LLM may also define `@pure` Python helper functions |
| `["llm"]` | LLM may also define new `@llm` actions with custom prompts |
| `["pure", "llm"]` | Both |

## Defining LLM actions

Prompts are defined directly on Python functions with `@llm`:

```python
@llm(
    system=(
        "You are a medical expert. Analyze the notes and determine "
        "if the diagnosis applies."
    ),
    user="Notes: {notes}\nDiagnosis: {diag}",
    parse="json",
    outputs=(("verdict", bool), ("reason", str)),
)
def assess(notes: str, diag: str) -> None: ...
```

- `parse="json"` — ZipperGen appends a type-annotated JSON instruction to the prompt and validates the response against the declared output types.
- `parse="text"` — single `str` output, returned as plain text.
- `parse="bool"` — single `bool` output, model replies `true` or `false`.

## Using real LLMs

The simplest way is to export your API key and set `llms=` in `configure()`.

**All agents on the same provider:**

```bash
export OPENAI_API_KEY=...
```

```python
diagnosisConsensus.configure(llms="openai", ui=True, timeout=600)
```

**Different providers per agent:**

```bash
export MISTRAL_API_KEY=...
export OPENAI_API_KEY=...
```

```python
diagnosisConsensus.configure(
    llms={"LLM1": "mistral", "LLM2": "openai"},
    ui=True,
    timeout=600,
)
```

**Different API keys per agent** (useful for parallel rate limits):

```python
from zippergen.backends import make_openai_backend

contractReview.configure(
    llms={
        "Jurisdiction":    make_openai_backend(api_key="sk-..."),
        "Liability":       make_openai_backend(api_key="sk-..."),
        "Confidentiality": make_openai_backend(api_key="sk-..."),
        "Orchestrator":    "mistral",
    },
    timeout=600,
)
```

Built-in provider names: `"openai"`, `"mistral"`, `"claude"` (alias: `"anthropic"`).

The built-in backends read these environment variables:

| Variable | Default |
|---|---|
| `OPENAI_API_KEY` | — |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `MISTRAL_API_KEY` | — |
| `MISTRAL_MODEL` | `mistral-small-latest` |
| `ANTHROPIC_API_KEY` | — |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` |

**Custom backend:**

```python
def my_backend(action, inputs):
    # action.system_prompt, action.user_prompt, action.outputs available
    return {"verdict": True, "reason": "..."}

my_workflow.configure(backend=my_backend, timeout=60)
```

## Formal foundation

The implementation is grounded in the theory of Message Sequence Charts. The key properties:

- **Correctness** — The distributed projected programs produce exactly the same behaviors as the global program.
- **Deadlock-freedom** — Follows by structural induction; no runtime checking required.
