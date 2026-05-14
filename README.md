<p align="center">
  <img src="assets/zippergen-lockup-ink.svg" alt="ZipperGen" width="420">
</p>

<p align="center">
  <a href="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml"><img src="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://arxiv.org/abs/2604.17612"><img src="https://img.shields.io/badge/arXiv-2604.17612-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/zippergen-io/paper-isola/tree/main/Lean"><img src="assets/lean-formalized.svg" alt="Lean formalized"></a>
  <a href="https://github.com/zippergen-io/paper-isola/tree/main/Lean"><img src="assets/lean.svg" alt="Lean verified"></a>
</p>

ZipperGen is a Python framework for multi-agent LLM coordination. You write a single global protocol (who sends what to whom, who runs which LLM, who owns each decision), and ZipperGen projects it onto each agent automatically. If the protocol compiles, it cannot deadlock. This is not a runtime check; it follows from how the projection works.

ZipperGen separates **what agents do** (LLM calls and pure functions) from **how they coordinate** (the protocol). Unlike tool-calling frameworks, ZipperGen provides formal guarantees: coordination is provably deadlock-free by construction, whether the protocol is written by hand or generated at runtime.

Each participant in a workflow is called a **lifeline**, which is the standard term from Message Sequence Charts (MSCs), the formalism ZipperGen is based on. In practice a lifeline is simply an agent: one sequential thread of execution that sends and receives messages.

ZipperChat visualizes a run as a message sequence chart, including actions, messages, decisions, and human control points.

![ZipperChat screenshot](assets/zipperchat-screenshot.png)

## Quick start

```bash
git clone https://github.com/zippergen-io/zippergen.git
cd zippergen
pip install -e .
```

Python 3.11 or later required. No external dependencies: stdlib only (LLM backends optional).

## Hello, World

Three agents collaborate: `Writer` drafts a tweet, `Editor` decides whether it's good enough, and ZipperGen handles the coordination. No API key needed; the built-in mock backend returns placeholder model outputs.

```python
from zippergen.syntax import Lifeline, Var
from zippergen.actions import llm
from zippergen.builder import workflow

User   = Lifeline("User")
Writer = Lifeline("Writer")
Editor = Lifeline("Editor")

tweet    = Var("tweet",    str)
approved = Var("approved", bool)

@llm(system="Write a one-sentence tweet about the topic.",
     user="{topic}", parse="text", outputs=(("tweet", str),))
def draft(topic: str) -> None: ...

@llm(system="Is this tweet engaging and under 280 chars? Reply true or false.",
     user="{tweet}", parse="bool", outputs=(("approved", bool),))
def approve(tweet: str) -> None: ...

@llm(system="Improve this tweet: shorter and punchier.",
     user="{tweet}", parse="text", outputs=(("tweet", str),))
def revise(tweet: str) -> None: ...

@workflow
def write_tweet(topic: str @ User) -> str:
    User(topic) >> Writer(topic)
    Writer: tweet = draft(topic)
    Writer(tweet) >> Editor(tweet)
    Editor: approved = approve(tweet)
    if approved @ Editor:
        Editor(tweet) >> User(tweet)
    else:
        Editor(tweet) >> Writer(tweet)
        Writer: tweet = revise(tweet)
        Writer(tweet) >> User(tweet)
    return tweet @ User

# No API key needed; the built-in mock backend returns placeholder model outputs.
# Switch to a real LLM: write_tweet.configure(llms="openai")
result = write_tweet(topic="a git commit message that tells the truth")
print(result)
```

`if approved @ Editor` is the key line. `Editor` owns the branching decision; ZipperGen automatically determines which agents need to receive that decision and generates the coordination messages. You don't write any routing code.

Human control points can be inserted directly into a lifeline, too. A workflow may pause for approval, review, correction, or data entry exactly where the global protocol requires it; the interaction remains explicit in the projected program and visible in ZipperChat.

The mock backend produces placeholder output (`[draft:tweet]`, `[revise:tweet]`). Add one line to switch to a real LLM:

```python
write_tweet.configure(llms="openai")   # or "mistral", "claude"
result = write_tweet(topic="a git commit message that tells the truth")
```

The full example is at `examples/write_tweet.py`.

## Why it can't deadlock

In most multi-agent frameworks, control flow lives inside each agent. Agents call tools, decide what to do next, and rely on the other agents being ready to receive. This works until a subtle ordering problem causes two agents to wait on each other indefinitely.

ZipperGen works differently. You write the control flow once, as a global protocol. ZipperGen then *projects* that protocol onto each agent: each agent receives exactly the local view of the global plan that it needs. Because every send has a corresponding receive by construction, deadlock cannot occur. This is a structural property, not something checked at runtime.

The formal statement is in [our paper](https://arxiv.org/abs/2604.17612): the projected programs produce exactly the same behaviors as the global program, and deadlock-freedom follows by structural induction.

The practical consequence: the global protocol is also a complete audit trail of what your agents are allowed to do. You can read it, reason about it, and submit it to anyone who needs to understand how the system works.

## Causal runtime guards

ZipperGen also supports causal-past guards for workflows where the latest locally received value may be stale. In `examples/cpl_test.py`, a device reports its status through two relays. The indicator receives a newer `on=True` update before an older delayed `on=False` update. A local guard on `on` would follow the delayed stale message, but a causal guard reads the latest causally visible device state:

```python
latest_device_on = At[Device](
    atom(lambda env: env.on, src="on")
)

if latest_device_on @ Indicator:
    ...
```

The `env` argument is the lifeline's local variable store; attribute access (`env.on`) is equivalent to a dict lookup.

**Field terms** let a guard compare its local variables with another lifeline's latest causally visible variables. Every message automatically piggybacks the sender's latest variable snapshot, so the receiving lifeline's monitor has it for free:

```python
version_matches = At[Reviewer].rev_version == Here.version
```

Here `At[Reviewer].rev_version` is the Reviewer's `rev_version` at its latest causally visible event, while `Here.version` is the deciding lifeline's current `version`. The Reviewer's version was never explicitly sent to the Gatekeeper; it arrives implicitly on the verdict message.

The result is determined entirely from the asynchronous communication structure: vector clocks record which events are causally visible, and message-carried views provide the latest guard values at those visible events.

## Parallel regions

A `parallel` region runs several full sub-programs concurrently. Branches are structurally independent: each send/receive action carries a channel name derived from its syntactic position, so FIFO order is maintained per branch without any programmer-visible bookkeeping.

```python
@workflow
def merge_candidate(candidate: str @ Orchestrator) -> str:
    Orchestrator(candidate) >> Committer(candidate)

    with parallel:
        with branch:
            Orchestrator(candidate) >> TestRunner(candidate)
            TestRunner: (test_status,) = run_tests(candidate)
            TestRunner(test_status) >> Committer(test_status)

        with branch:
            Orchestrator(candidate) >> Security(candidate)
            Security: (security_status,) = scan_security(candidate)
            Security(security_status) >> Committer(security_status)

    Committer: (decision,) = decide_merge(candidate, test_status, security_status)
    return decision @ Committer
```

`Orchestrator` and `Committer` are *shared lifelines*: each appears in both branches. ZipperGen statically checks that shared lifelines do not form a dependency cycle. If they did, projection would be rejected before the workflow ever runs. The projection of a shared lifeline interleaves its branch-local programs while preserving their internal order.

See `examples/parallel.py` for the full example and `examples/parallel_cyclic.py` for a rejected cyclic case.

## See it in action

Examples ship with the repo. The first two run without an API key.

```bash
python examples/parallel.py           # fan-out/fan-in with static acyclicity check (no key needed)
python examples/parallel_cyclic.py    # rejected cyclic dependency; shows the error (no key needed)
python examples/cpl_test.py           # causal guard ignores stale relay status (no key needed)
python examples/field_terms.py        # field-term guard: cross-lifeline version check (no key needed)
python examples/coregion.py           # unordered receives from independent analysts (no key needed)
python examples/dashboard.py          # several top-level workflow runs in one ZipperChat page
python examples/nested_dashboard.py   # several dashboard runs, each with nested subworkflows
python examples/write_tweet.py        # draft-and-approve with mock LLM (no key needed)
python examples/write_tweet_local.py  # same workflow through a local OpenAI-compatible server
python examples/human_approval.py     # human priority, notes, and approval in ZipperChat (no key needed)
python examples/diagnosis.py          # two LLMs reach consensus iteratively (no key needed with mock)
python examples/contract_review.py    # four agents review a contract in parallel (needs MISTRAL_API_KEY)
python examples/morning_digest.py     # inbox triage: parallel analysis, owned branching (needs MISTRAL_API_KEY)
python examples/arithmetic_planner.py # LLM decomposes and evaluates an arithmetic expression in parallel (needs OPENAI_API_KEY)
python examples/planner.py            # LLM designs and runs its own sub-workflow (needs OPENAI_API_KEY)
```

Open **http://localhost:8765** to watch the agents exchange messages in real time as a message sequence chart.

For applications that call several workflows from ordinary Python code, ZipperChat
can show multiple independent runs on the same page:

```python
from zipperchat import WebTrace

dashboard = WebTrace.dashboard().start()
first_workflow.configure(ui=True, trace=dashboard)
second_workflow.configure(ui=True, trace=dashboard)
```

## How it works

The code is organized as a pipeline of layers that mirror the paper almost literally. A user writes a global workflow in Python DSL syntax; `@workflow` rewrites it into an immutable IR; the projection layer turns that global IR into one local program per lifeline; the runtime starts one thread per lifeline and connects them with FIFO queues. The optional `@planner` primitive asks an LLM to generate a new global workflow at runtime, then runs it through the exact same pipeline, so the guarantee holds there too.

### Diagnosis consensus

Two LLMs independently assess a case, then iterate until they agree or a round limit is reached:

```python
@workflow
def diagnosis_consensus(notes: str @ User, diagnosis: str @ User) -> str:
    # Distribute inputs to both LLMs
    User(notes, diagnosis) >> LLM1(notes, diagnosis)
    User(notes, diagnosis) >> LLM2(notes, diagnosis)

    # Independent initial assessments
    LLM1: (verdict, reason) = assess(notes, diagnosis)
    LLM2: (verdict, reason) = assess(notes, diagnosis)

    # Consensus loop, owned by LLM1 (at most MAX_ROUNDS rounds)
    while (not agreed and trials < MAX_ROUNDS) @ LLM1:
        LLM1(verdict, reason) >> LLM2(other_verdict, other_reason)
        LLM2(verdict, reason) >> LLM1(other_verdict, other_reason)
        LLM1: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
        LLM2: (verdict, reason) = reconsider(notes, diagnosis, verdict, reason, other_verdict, other_reason)
        LLM2(verdict) >> LLM1(other_verdict)
        with LLM1:
            agreed = check_agreement(verdict, other_verdict)
            trials = inc_trials(trials)

    # Final result computed by LLM1, returned to User
    LLM1: result = choose_result(verdict, agreed)
    LLM1(result) >> User(result)
    return result @ User
```

`while cond @ LLM1` means LLM1 owns the loop guard and broadcasts the decision each iteration. `if cond @ Owner` works the same way for conditionals. ZipperGen figures out which other agents need to receive the decision and generates the control messages automatically.

Workflows that return a value end with `return var @ Lifeline`. This declares which lifeline owns the result once all agents have finished: it is a declaration, not a control flow statement. No matter which branches executed, the result always lands in the same place. Output-free workflows can use `-> tuple` and omit the return.

## Defining LLM actions

Prompts are defined directly on Python functions with `@llm`. The `parse` parameter controls how the response is interpreted, and determines what `outputs` must look like:

**`parse="json"`**: multiple typed outputs; ZipperGen appends a JSON instruction and validates the response:

```python
@llm(
    system="You are a medical expert. Analyze the notes and determine if the diagnosis applies.",
    user="Notes: {notes}\nDiagnosis: {diag}",
    parse="json",
    outputs=(("verdict", str), ("reason", str)),    # one or more (name, type) pairs
)
def assess(notes: str, diag: str) -> None: ...
```

**`parse="text"`**: exactly one `str` output; the model's raw response is returned as-is:

```python
@llm(
    system="You are a medical writer. Summarise the following notes in one paragraph.",
    user="{notes}",
    parse="text",
    outputs=(("summary", str),),                    # exactly one str entry
)
def summarise(notes: str) -> None: ...
```

**`parse="bool"`**: exactly one `bool` output; the model is asked to reply `true` or `false`:

```python
@llm(
    system="Is this tweet engaging, original, and under 180 chars? Reply true or false.",
    user="{tweet}",
    parse="bool",
    outputs=(("approved", bool),),                  # exactly one bool entry
)
def approve(tweet: str) -> None: ...
```

## Dynamic planning

For tasks where the coordination structure isn't known in advance, `@planner` lets an LLM design the workflow at runtime. Give it a description, an action vocabulary, and a set of lifelines; it generates a complete sub-workflow, which ZipperGen validates and executes.

Actions in the vocabulary can be atomic tools (`@pure` functions or `@llm` calls) or full skills: entire `@workflow`s that appear to the planner as a single typed action but internally run their own verified coordination protocol.

Here the planner receives an arithmetic expression and must evaluate it using three Calculator lifelines with maximum parallelism, guarding against division by zero:

```python
@planner(
    description="Evaluate an arithmetic expression with maximum parallelism. "
                "Identify independent subexpressions and evaluate them concurrently. "
                "If the expression is undefined (division by zero), return 0.",
    actions=[add, subtract, multiply, divide, identity, is_zero],
    lifelines=[Calculator1, Calculator2, Calculator3],
    allow=["if"],
)
def evaluate(expression: str) -> str: ...
```

The decorated function slots into a `@workflow` like any other action. Given `(2 - 4) * (2 + 3) + (3 / (3 - 2))`, GPT-4o generates:

```python
@workflow
def generated_workflow(expression: str @ Planner) -> str:
    Planner() >> Calculator1()
    Planner() >> Calculator2()
    Planner() >> Calculator3()

    Calculator1: subtract1 = subtract(2.0, 4.0)          # (2 - 4) = -2  ─┐
    Calculator2: add1      = add(2.0, 3.0)               # (2 + 3) =  5  ─┤ parallel
    Calculator3: subtract2 = subtract(3.0, 2.0)          # (3 - 2) =  1  ─┘
    Calculator3: zero      = is_zero(subtract2)          # check before dividing

    if zero @ Calculator3:
        Calculator3(0.0) >> Planner(result)              # guard: return 0
    else:
        Calculator3: divide1   = divide(3.0, subtract2)  # 3 / 1 = 3

        Calculator2(add1)    >> Calculator1(add1)
        Calculator3(divide1) >> Calculator1(divide1)

        Calculator1: multiply1 = multiply(subtract1, add1)   # -2 * 5 = -10
        Calculator1: result    = add(multiply1, divide1)     # -10 + 3 = -7
        Calculator1(result) >> Planner(result)

    return result @ Planner
```

The LLM parsed the expression, identified that `(2 - 4)`, `(2 + 3)`, and `(3 - 2)` are all independent, evaluated them in parallel across three calculators, checked the denominator before dividing, and wired the join correctly, all from the description and action vocabulary alone. ZipperGen validates the generated workflow structurally before running it.

**`allow`** controls extensions: `"pure"` (define helper functions), `"llm"` (define new LLM actions), `"if"` (conditional branching), `"while"` (loops). Default is `[]` (pre-defined vocabulary only, linear workflows).

## Using real LLMs

The simplest way is to export your API key and set `llms=` in `configure()`.

**All agents on the same provider:**

```bash
export OPENAI_API_KEY=...
```

```python
diagnosis_consensus.configure(llms="openai", ui=True, timeout=600)
```

**Different providers per agent:**

```bash
export MISTRAL_API_KEY=...
export OPENAI_API_KEY=...
```

```python
diagnosis_consensus.configure(
    llms={"LLM1": "mistral", "LLM2": "openai"},
    ui=True,
    timeout=600,
)
```

**Different API keys per agent** (useful for parallel rate limits):

```python
from zippergen.backends import make_openai_backend

contract_review.configure(
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

OpenAI-compatible local servers, such as vLLM, can be used through the same backend:

```python
from zippergen.backends import make_openai_backend

local_llm = make_openai_backend(
    api_key="EMPTY",
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
)

write_tweet.configure(backend=local_llm, ui=True)
```

For a remote GPU server, run the model server on the remote machine with a local
bind address and use SSH port forwarding, for example
`ssh -L 8000:127.0.0.1:8000 lmf-gpu`. See `examples/write_tweet_local.py`
for a complete local-model variant of the hello-world workflow.

The built-in backends read these environment variables:

| Variable | Default |
|---|---|
| `OPENAI_API_KEY` | (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `MISTRAL_API_KEY` | (required) |
| `MISTRAL_MODEL` | `mistral-small-latest` |
| `ANTHROPIC_API_KEY` | (required) |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` |

**Custom backend:**

```python
def my_backend(action, inputs):
    # action.system_prompt, action.user_prompt, action.outputs available
    return {"verdict": True, "reason": "..."}

my_workflow.configure(backend=my_backend, timeout=60)
```

## Why not LangGraph / CrewAI / AutoGen?

The short answer: those frameworks leave coordination up to the agents or the graph structure. ZipperGen makes coordination explicit and proves it correct.

**LangGraph** uses a graph of nodes and edges. Conditional branching requires a router function that returns the name of the next node. The graph structure implies an execution order, but protocol-level deadlock-freedom and projection correctness are not the focus of the framework. It's a good fit when you need fine-grained control over an irregular flow and are comfortable reasoning about the graph yourself.

**CrewAI and AutoGen** are conversation-based: agents exchange messages and decide what to do next. The coordination is mostly emergent from the agent prompts. This works well for open-ended tasks where you can't or don't want to specify the coordination structure in advance. The tradeoff is that protocol behavior is harder to audit and outside the scope of those frameworks to prove correct.

**ZipperGen** requires you to write the coordination structure explicitly. That's a constraint. In return, you get a protocol that can be read by a person, checked by a tool, and submitted to anyone who needs to understand how the system behaves, along with a proof that it terminates without deadlock. If your use case involves a fixed or semi-fixed coordination structure (which most production systems do), the explicitness is an asset.

If the structure genuinely isn't known in advance, use `@planner`; the LLM generates the sub-workflow, ZipperGen validates it structurally, and the guarantee still holds.

## Formal foundation

The implementation is based on the theory of Message Sequence Charts. The key properties:

- **Correctness**: The distributed projected programs produce exactly the same behaviors as the global program.
- **Deadlock-freedom**: Follows by structural induction; no runtime checking required.

The formal proofs are in [our paper](https://arxiv.org/abs/2604.17612). The main theorems (Theorem 3.1 and Corollary 3.1) have been machine-checked in Lean 4; see the [formalization](https://github.com/zippergen-io/paper-isola/tree/main/Lean).

## License

ZipperGen is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full terms.
