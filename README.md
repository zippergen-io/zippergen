<p align="center">
  <img src="assets/zippergen-lockup-ink.svg" alt="ZipperGen" width="420">
</p>

<p align="center">
  <a href="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml"><img src="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://arxiv.org/abs/2604.17612"><img src="https://img.shields.io/badge/arXiv-2604.17612-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/zippergen-io/paper-isola/tree/main/Lean"><img src="assets/lean-formalized.svg" alt="Lean formalized"></a>
  <a href="https://github.com/zippergen-io/paper-isola/tree/main/Lean"><img src="assets/lean.svg" alt="Lean verified"></a>
</p>

ZipperGen is a Python framework for AI workflows where several agents, tools, and humans must coordinate without ad-hoc message routing.

You write the workflow once as a global protocol: who sends what to whom, who runs which LLM, and who owns each decision. ZipperGen projects it to local agent programs automatically.

For well-formed workflows, the generated coordination is deadlock-free by construction. This follows from the projection discipline, not from runtime checking.

ZipperGen separates **what agents do** (LLM calls, tool use, human input) from **how they coordinate** (the protocol). The protocol is readable, auditable, and can be shared with anyone who needs to understand how the system works.

Each participant is called a **lifeline**, which is the standard term from Message Sequence Charts (MSCs), the formalism ZipperGen is based on. In practice a lifeline is simply an agent: one sequential thread of execution that sends and receives messages.

ZipperChat visualizes a run as a message sequence chart, including actions, messages, decisions, and human control points.

![ZipperChat screenshot](assets/zipperchat-screenshot.png)

## Quick start

```bash
git clone https://github.com/zippergen-io/zippergen.git
cd zippergen
pip install -e .
python examples/hello.py
```

Python 3.11 or later required. No external dependencies: stdlib only (LLM backends optional).

## Hello, ZipperGen

Two lifelines, one LLM call, one message back.

```python
from zippergen.syntax import Lifeline
from zippergen.actions import llm
from zippergen.builder import workflow

User   = Lifeline("User")
Writer = Lifeline("Writer")

@llm(system="Write a concise reply.",
     user="{topic}", parse="text", outputs=(("draft", str),))
def write_reply(topic: str) -> None: ...

@workflow
def hello(topic: str @ User) -> str:
    User(topic) >> Writer(topic)
    Writer: draft = write_reply(topic)
    Writer(draft) >> User(draft)
    return draft @ User

hello.configure(llms="mock", ui=True)
result = hello(topic="Say hello to ZipperGen")
print(result)
```

`User` sends a value to `Writer`, `Writer` runs an LLM action, and the result comes back. The workflow says explicitly who owns each step: which lifeline sends, which receives, and which runs each action. Open **http://localhost:8765** to watch the exchange in ZipperChat.

Switch to a real LLM with one line:

```python
hello.configure(llms="openai", ui=True)   # or "mistral", "claude"
```

The full example is at `examples/hello.py`.

## Owned decisions

The previous example has no coordination choice. Here is the first place where ZipperGen matters more: one lifeline owns a decision, and ZipperGen generates the required coordination messages automatically.

Three agents collaborate: `Writer` drafts a tweet, `Editor` decides whether it's good enough, and `Writer` revises if needed.

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

result = write_tweet(topic="a git commit message that tells the truth")
print(result)
```

`if approved @ Editor` is the key line. `Editor` owns the branching decision; ZipperGen automatically determines which agents need to receive that decision and generates the coordination messages. You don't write any routing code.

The full example is at `examples/write_tweet.py`.

## Why protocols?

In most multi-agent frameworks, control flow lives inside each agent. Agents call tools, decide what to do next, and rely on the other agents being ready to receive. This works until a subtle ordering problem causes two agents to wait on each other indefinitely.

ZipperGen works differently. You write the control flow once, as a global protocol. ZipperGen then *projects* that protocol onto each agent: each agent receives exactly the local view of the global plan that it needs. Because every send has a corresponding receive by construction, deadlock cannot occur for well-formed protocols. This is a structural property, not something checked at runtime.

The formal statement is in [our paper](https://arxiv.org/abs/2604.17612): the projected programs produce exactly the same behaviors as the global program, and deadlock-freedom follows by structural induction.

The practical consequence: the global protocol is also a complete audit trail of what your agents are allowed to do. You can read it, reason about it, and show it to anyone who needs to understand how the system works.

## Command center

`examples/command_center.py` is a larger example that runs two independent event loops in parallel, sharing the same lifelines. One branch handles incoming email: `Mailbox` polls the inbox, `Dispatcher` classifies each message, and named fragments handle the rest (`scheduling_branch`, `cancellation_branch`, `task_branch`, `reply_branch`). The other branch handles Telegram commands from the owner: `Chat` receives a command, `Dispatcher` classifies it, and a matching fragment runs (`schedule_meeting_from_chat`, `create_task_from_chat`, and so on). Telegram is only the command interface: `Calendar` owns calendar changes, `Mailbox` owns email drafts, `TasksTool` owns tasks, and `User` owns approvals.

```python
@workflow
def command_center():
    with parallel:
        with branch:
            while True @ Mailbox:
                if mail_present() @ Mailbox:
                    Mailbox: email = pop_pending_email()
                    Mailbox(email) >> Dispatcher(email)
                    Dispatcher: route = classify_email(email)
                    Dispatcher: route = normalize_route(route)

                    if (route == "spam") @ Dispatcher:
                        ...
                    elif (route == "scheduling") @ Dispatcher:
                        scheduling_branch(email)
                    elif (route == "task") @ Dispatcher:
                        task_branch(email)
                        reply_branch(email)
                    else:
                        reply_branch(email)
                else:
                    Mailbox: _ = wait_briefly()

        with branch:
            while True @ Chat:
                if chat_present() @ Chat:
                    Chat: chat_msg = pop_pending_chat()
                    Chat(chat_msg) >> Dispatcher(chat_msg)
                    Dispatcher: chat_route = classify_chat(chat_msg)
                    Dispatcher: chat_route = normalize_route(chat_route)

                    if (chat_route == "schedule_meeting") @ Dispatcher:
                        Dispatcher(chat_msg) >> Writer(chat_msg)
                        schedule_meeting_from_chat(chat_msg)
                    elif (chat_route == "create_task") @ Dispatcher:
                        Dispatcher(chat_msg) >> Writer(chat_msg)
                        create_task_from_chat(chat_msg)
                    ...
                else:
                    Chat: _ = wait_briefly()
```

`Calendar`, `Writer`, `Mailbox`, and `User` are shared between the two branches. ZipperGen's projection ensures each one receives exactly the messages it needs from whichever stream generated them, in the order the global protocol requires. The two loops interleave freely at runtime without any programmer-visible synchronization.

Run it with mock data to see both streams in ZipperChat without any API keys:

```bash
python examples/command_center.py --mock
```

For a live setup with Gmail, Google Calendar, Google Tasks, and Telegram, follow the one-time setup steps in the file's docstring.

## Parallel regions

A `parallel` region runs several full sub-programs concurrently. Branches are structurally independent: each send/receive action carries a channel name derived from its syntactic position, so FIFO order is maintained per branch without any programmer-visible bookkeeping. A lifeline that appears in multiple branches is a *shared lifeline*; its projection interleaves the branch-local programs while preserving their internal order. At the semantic level, ZipperGen keeps only complete message-sequence-chart executions, which lets you write realistic feedback patterns between shared lifelines without manually managing branch-local channels.

See `examples/parallel.py` for fan-out/fan-in and `examples/parallel_cyclic.py` for a feedback pattern between shared lifelines.

## Human control points

Workflows can pause for human input anywhere in the protocol. A `@human` action specifies the interaction shape and appears in ZipperChat as a card waiting for a response. When the human responds, the workflow continues from exactly that point.

```python
@human(
    kind="confirm",
    context="{draft}",
    instruction="Send this reply?",
    outputs=["approved: bool"],
    submit_label="Send",
    cancel_label="Discard",
)
def approve_reply(draft: str): pass
```

Supported kinds: `confirm` (yes/no), `edit` (review and edit text), `input` (free-form entry), `select` (choose from options), and `ack` (acknowledge a completed event). Human actions are visible in ZipperChat by default and can be hidden with `visible=False`.

See `examples/human_approval.py` for approval, notes, and priority selection patterns.

## Examples

**Start here** (no API key needed):

```bash
python examples/write_tweet.py        # draft-and-approve loop with mock LLM
python examples/parallel.py           # fan-out/fan-in across parallel branches
python examples/command_center.py --mock  # email triage + Telegram commands, two parallel streams
```

**Core coordination patterns:**

```bash
python examples/diagnosis.py          # two LLMs reach consensus iteratively
python examples/contract_review.py    # four agents review a contract in parallel (needs MISTRAL_API_KEY)
python examples/morning_digest.py     # inbox triage: parallel analysis, owned branching (needs MISTRAL_API_KEY)
python examples/human_approval.py     # human priority, notes, and approval in ZipperChat
```

**Advanced features:**

```bash
python examples/cpl_test.py           # causal guard ignores stale relay status
python examples/field_terms.py        # cross-lifeline version check via field-term guard
python examples/arithmetic_planner.py # LLM decomposes an expression and evaluates it in parallel (needs OPENAI_API_KEY)
python examples/planner.py            # LLM designs and runs its own sub-workflow (needs OPENAI_API_KEY)
python examples/parallel_cyclic.py    # feedback pattern between shared lifelines
python examples/coregion.py           # unordered receives from independent analysts
python examples/dashboard.py          # several workflow runs in one ZipperChat page
python examples/nested_dashboard.py   # dashboard runs with nested subworkflows
python examples/write_tweet_local.py  # hello-world through a local OpenAI-compatible server
```

Open **http://localhost:8765** to watch the agents exchange messages in real time as a message sequence chart.

For applications that call several workflows from ordinary Python code, ZipperChat can show multiple independent runs on the same page:

```python
from zipperchat import WebTrace

dashboard = WebTrace.dashboard().start()
first_workflow.configure(ui=True, trace=dashboard)
second_workflow.configure(ui=True, trace=dashboard)
```

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

## Advanced

### Causal runtime guards

ZipperGen supports causal-past guards for workflows where the latest locally received value may be stale. In `examples/cpl_test.py`, a device reports its status through two relays. The indicator receives a newer `on=True` update before an older delayed `on=False` update. A local guard on `on` would follow the delayed stale message, but a causal guard reads the latest causally visible device state:

```python
latest_device_on = At[Device].on == True

if latest_device_on @ Indicator:
    ...
```

**Field terms** let a guard compare its local variables with another lifeline's latest causally visible variables. Every message automatically piggybacks the sender's latest variable snapshot, so the receiving lifeline's monitor has it for free:

```python
version_matches = At[Reviewer].rev_version == Here.version
```

Here `At[Reviewer].rev_version` is the Reviewer's `rev_version` at its latest causally visible event, while `Here.version` is the deciding lifeline's current `version`. The Reviewer's version was never explicitly sent to the Gatekeeper; it arrives implicitly on the verdict message.

The result is determined entirely from the asynchronous communication structure: vector clocks record which events are causally visible, and message-carried views provide the latest guard values at those visible events.

### Dynamic planning

For tasks where the coordination structure is not known in advance, `@planner` lets an LLM design the workflow at runtime. Give it a description, an action vocabulary, and a set of lifelines; it generates a complete sub-workflow, which ZipperGen validates structurally and then executes. The same coordination guarantee applies to the generated workflow once it validates.

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

The LLM parsed the expression, identified that `(2 - 4)`, `(2 + 3)`, and `(3 - 2)` are all independent, evaluated them in parallel across three calculators, checked the denominator before dividing, and wired the join correctly, all from the description and action vocabulary alone.

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

For a remote GPU server, run the model server on the remote machine with a local bind address and use SSH port forwarding, for example `ssh -L 8000:127.0.0.1:8000 lmf-gpu`. See `examples/write_tweet_local.py` for a complete local-model variant of the hello-world workflow.

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

Three different answers to the same coordination problem:

**LangGraph** is state-centric orchestration. Control flow is a graph of nodes and edges; conditional branching requires a router function that returns the name of the next node. It is a good fit when you need fine-grained control over an irregular flow and are comfortable reasoning about the graph yourself.

**CrewAI and AutoGen** are conversation-centric orchestration. Agents exchange messages and decide what to do next. Coordination is mostly emergent from the agent prompts. This works well for open-ended tasks where you cannot or do not want to specify the structure in advance. The tradeoff is that protocol behavior is harder to audit.

**ZipperGen** is protocol-centric orchestration. You write the coordination structure explicitly as a global protocol, then ZipperGen projects it to each agent with a structural correctness guarantee. That is a constraint. In return, you get a protocol that can be read by a person, checked by a tool, and submitted to anyone who needs to understand how the system behaves, along with a structural guarantee against coordination deadlocks.

If the structure genuinely is not known in advance, use `@planner`: the LLM generates the sub-workflow, ZipperGen validates it structurally, and the coordination guarantee applies to the generated workflow once it passes validation.

## Formal foundation

The implementation is based on the theory of Message Sequence Charts. The key properties:

- **Correctness**: The distributed projected programs produce exactly the same behaviors as the global program.
- **Deadlock-freedom**: Follows by structural induction; no runtime checking required.

The formal proofs are in [our paper](https://arxiv.org/abs/2604.17612). The main theorems (Theorem 3.1 and Corollary 3.1) have been machine-checked in Lean 4; see the [formalization](https://github.com/zippergen-io/paper-isola/tree/main/Lean).

The causal runtime guards (`At[Lifeline].var`, `Here.var`) are described in a companion paper: Bollig. [*Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows*](https://arxiv.org/abs/2605.20923). arXiv:2605.20923 [cs.LO].

## License

ZipperGen is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full terms.
