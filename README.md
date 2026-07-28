<p align="center">
  <img src="assets/zippergen-lockup-ink.svg" alt="ZipperGen" width="420">
</p>

<p align="center">
  <a href="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml"><img src="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://arxiv.org/abs/2604.17612"><img src="https://img.shields.io/badge/arXiv-2604.17612-b31b1b.svg" alt="arXiv"></a>
</p>

Multi-agent systems scatter coordination across agents and callbacks. The
complete message order becomes hard to read, and a blocked run is hard to
diagnose.

ZipperGen puts that coordination in one readable Python protocol. It projects
the protocol into a local program for each participant, runs them durably,
shows exactly where each participant is waiting, and deploys the result as a
real service.

Coordination deadlocks are ruled out by construction, not by runtime checks.
This guarantee applies to well-formed workflows in ZipperGen's supported
language.

## Contents

- [See the current program position](#see-the-current-program-position)
- [Quick start](#quick-start)
- [What Studio looks like](#what-studio-looks-like)
- [Why protocols](#why-protocols)
- [Hello, ZipperGen](#hello-zippergen)
- [Inspect workflows as code](#inspect-workflows-as-code)
- [Assistant safety](#assistant-safety)
- [Durable runs and deployment](#durable-runs-and-deployment)
- [Examples and documentation](#examples-and-documentation)
- [Formal foundation](#formal-foundation)

## See the current program position

Studio can show the local program for one participant and mark its current
statement. The view stays bounded even when a workflow contains a long loop.
It also shows why the other participants are waiting.

![ZipperGen Studio showing a live local projection](assets/studio-run-inspect.svg)

This is real output from a mock run that is waiting for human approval. The
view does not display workflow values, prompts, credentials, or action inputs.

## Quick start

The published PyPI alpha does not yet contain the current Studio preview. For
now, install and run the current source:

```bash
git clone https://github.com/zippergen-io/zippergen.git
cd zippergen
uv sync
uv run zippergen run examples/hello.py:hello --llm mock --input 'topic=Say hello to ZipperGen'
```

The last command runs a complete two-participant workflow without an API key.
It returns a mock result and writes a durable SQLite store.

Requirements:

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- macOS or Linux for the current managed deployment path

Open Studio from the same directory:

```bash
uv run zippergen
```

Codex and Claude Code are optional. You need one of them only when you ask
Studio to generate or change workflow source from a natural-language
specification.

## What Studio looks like

Studio keeps the main path short:

```text
workflow create
workflow implement codex
workflow review
run
deploy
```

`workflow create` opens one versioned `specification.md`. A coding assistant
may implement that specification as ordinary Python code and focused tests.
The assistant never deploys the result. Studio keeps validation, inspection,
human acceptance, and deployment as explicit steps.

Commands are grouped by purpose:

| Area | What it covers |
|---|---|
| `workflow` | Specification, implementation, views, validation, review |
| `model` | Providers, named configurations, checks, assignments |
| `connector` | Provider credentials, reusable resources, human-action routes |
| `run`, `resume`, and `runs` | Durable development execution, decisions, traces, recovery |
| `deploy` | Bundles, services, durable state, decisions, and logs |

Named local Ollama configurations can release their model after an idle
period. The workflow stays active, and the model loads again automatically at
the next LLM action.

Tab completion shows valid commands, workflows, participants, model
configurations, and deployments. `current` gives one project summary. Every
structured result ends with a useful `Next` section.

### Read the communication protocol

The communication-only view hides local action details. It keeps messages,
loops, decisions, and ownership visible.

![ZipperGen Studio communication-only workflow view](assets/studio-workflow-communications.svg)

Other views show an overview, the full protocol, actions and prompts, complete
source, selected participants, or one exact local projection.

### Understand a deployment

`deploy show` separates the installed bundle, service process, workflow
run, SQLite store, model routing, connectors, and likely failure cause.

![ZipperGen Studio deployment state](assets/studio-deployment-show.svg)

This real capture shows a deployment running in the foreground. Its immutable
bundle is installed. The workflow is waiting for human action in its durable
store. The user service is not loaded because this capture used the foreground
path.

## Why protocols

Many agent frameworks spread control flow across agents and callbacks. That
makes the complete communication order hard to read. It can also make a blocked
system hard to diagnose.

ZipperGen starts from one global protocol. A participant is called a
**lifeline**. A lifeline can represent an agent, a person, a mailbox, or an
external-system role.

Projection gives each lifeline only the sends, receives, decisions, and local
actions it needs. Within ZipperGen's supported workflow language, a well-formed
global protocol projects to compatible local programs. The formal result gives
behavior preservation and deadlock freedom for the generated coordination.

The global workflow is useful before and after execution:

- It is readable design documentation.
- It is executable Python code.
- It defines every communication and decision owner.
- It produces exact local programs.
- It gives runtime inspection a precise program position.

ZipperGen is based on [Message Sequence
Charts](https://en.wikipedia.org/wiki/Message_sequence_chart) and
[choreographic programming](https://en.wikipedia.org/wiki/Choreographic_programming).

Message Sequence Charts are useful for explaining a short protocol. Local
projections are better for inspecting long-running loops.

## Hello, ZipperGen

This workflow has two lifelines, one LLM action, and two messages:

```python
from zippergen import Lifeline, llm, workflow

User = Lifeline("User")
Writer = Lifeline("Writer")


@llm(
    system="Write a concise reply.",
    user="{topic}",
    parse="text",
    outputs=(("draft", str),),
)
def write_reply(topic: str) -> None: ...


@workflow
def hello(topic: str @ User) -> str:
    User(topic) >> Writer(topic)
    Writer: draft = write_reply(topic)
    Writer(draft) >> User(draft)
    return draft @ User
```

Run the complete file with the built-in mock backend:

```bash
uv run zippergen run examples/hello.py:hello \
  --llm mock \
  --input 'topic=Say hello to ZipperGen'
```

The `@ User` annotation says that `User` owns the input and final result. The
message expressions state exactly when ownership moves between lifelines.

Owned decisions use the same notation:

```python
if approved @ Reviewer:
    Reviewer(draft) >> Requester(draft)
else:
    Reviewer(concerns) >> Writer(concerns)
```

`Reviewer` owns this decision. ZipperGen projects the needed control messages
to the other lifelines.

A large workflow can use `@fragment` helpers. Fragments keep the global
protocol readable. They are expanded before validation and projection.

## Inspect workflows as code

The scriptable CLI exposes the same views as Studio:

```bash
uv run zippergen validate examples/tutorial_review.py:tutorial_review

uv run zippergen show \
  examples/tutorial_review.py:tutorial_review \
  --communications

uv run zippergen show \
  examples/tutorial_review.py:tutorial_review \
  --agent Reviewer

uv run zippergen show \
  examples/tutorial_review.py:tutorial_review \
  --detail full
```

Useful Studio forms:

```text
workflow list
workflow import /path/to/existing_workflow.py
workflow select
workflow show
workflow show communications
workflow show agent Reviewer
workflow validate
workflow diff
workflow review
workflow accept
```

`workflow import` copies an existing workflow into the current project. It
also copies statically imported local Python modules and literal resource files
declared by the workflow. Existing project files are never overwritten
silently. If the imported file contains one workflow, Studio selects it. If it
contains several, Studio displays the entry points for selection.

Validation and acceptance answer different questions:

- `workflow validate` is a machine check. It checks structure, projection,
  metadata, and canonical rendering.
- `workflow accept` records a human decision that the reviewed intent and code
  are ready.

Running a candidate locally is useful. Deploying code that changed after human
acceptance is blocked until the user reviews the difference or records an
explicit override.

## Assistant safety

Studio can call Codex or Claude Code during development. A workflow can also
declare a first-class `@assistant` action that runs inside the protocol.

Assistant actions have three visible capability axes:

| Axis | Default | Meaning |
|---|---|---|
| `access` | `read-only` | Repository read or write access |
| `external_tools` | `none` | MCP, dedicated web tools, and configured integrations |
| `shell` | `restricted` | Strongest practical backend-specific shell boundary |

Security-relevant examples should state all three axes:

```python
@assistant(
    instructions_file="prompts/update_release_notes.md",
    access="write",
    external_tools="none",
    shell="restricted",
    workspace=".",
)
def update_release_notes(change: str) -> str: ...
```

These fields always appear in semantic snapshots and diffs.

The effective restricted shell differs by backend:

- Codex keeps command execution inside its read-only or workspace-write
  sandbox. Network access is disabled when external tools are disabled.
  Strict configuration parsing makes unsupported isolation settings fail.
- Claude receives no Bash tool. `shell="enabled"` gives Claude Bash access but
  does not give the same structural network isolation. Validation warns about
  this choice.

For a stronger provider-independent boundary, let the assistant edit without a
shell. Run a fixed verification command in a separate visible `@effect`.
Never execute a command string produced by an assistant.

Write access does not imply permission to deploy, restart services, commit,
push, or modify unrelated systems. The static instruction should state those
boundaries. Validation warns when a write workspace contains the executing
workflow itself.

## Durable runs and deployment

Development runs use SQLite by default. Messages, completed actions, human
tasks, traces, results, and participant positions survive terminal closure and
computer restart.

Inside Studio:

```text
run
run inspect
run inspect Reviewer
run tasks
run approve
run trace
resume
runs
```

Every new development run gets its own managed durable state. Users work with
the run and do not need to manage SQLite files.

A named deployment owns one stable logical store and an immutable source
bundle. The store is initialized when the deployment is prepared, even when
`--no-start` is used. Redeploying, starting, and restarting keep the same
store:

```text
deploy review-demo --no-start
deploy doctor review-demo
deploy show review-demo
deploy start review-demo
deploy inspect review-demo
deploy tasks review-demo
deploy approve review-demo
deploy trace review-demo
deploy logs review-demo
deploy restart review-demo
deploy stop review-demo
deploy remove review-demo
```

The service policy restarts failed workflows. It does not restart a finite
workflow after successful completion. `deploy stop` preserves the deployment.
`deploy remove` stops it, unregisters the user service, and moves its profile,
private secrets, bundles, environment, store, and logs into a private archive.
Use `deploy remove review-demo --purge` only when those files must be deleted
permanently. A permanent purge requires the explicit deployment name and a
second confirmation.

Studio discovers human actions from workflow code. Their delivery uses the
same provider, configuration, and assignment lifecycle as models. Private
credentials remain outside Git:

```text
connector provider configure telegram
connector provider check telegram
connector config create telegram-approvals
connector config check telegram-approvals
connector assign HumanApprover telegram-approvals
connector assignments
connector assignments check
```

Telegram uses task-specific buttons and starts automatically with the
deployment. Participant assignments cover all human actions on that lifeline.
`connector assign HumanApprover.approve_contract legal-approvals` shows the
action-level override form, where `legal-approvals` is another saved Telegram
configuration. One configuration may be assigned to several participants.
Their model and connector calls can still run in parallel.

Gmail and Google Sheets use the same provider and configuration pattern. The
workflow declares logical mailbox and table operations. Studio stores Google
authorization, the Gmail query, and the concrete spreadsheet privately:

For a source checkout, install the optional Google support once:

```bash
uv sync --extra google
```

For an installed package, use `pip install "zippergen[google]"`. Managed
deployments detect Gmail and Sheets requirements and install this support
automatically.

```text
connector setup
```

For the call-intake workflow, this one command authorizes Gmail and Sheets,
asks for the mailbox query, spreadsheet, and tab, checks both resources, and
binds the saved configurations:

```text
workflow select examples/call_intake.py:call_intake
connector setup
connector assignments check
deploy call-intake
```

The default reply mode creates Gmail drafts. Change it to `send` only after
testing with a dedicated account. Before setup, enable the Gmail and Google
Sheets APIs in one Google Cloud project and download a Desktop app OAuth
client. Studio imports that client into private storage. If it is on another
computer, Studio prints one `scp` command and removes the temporary upload
after import. On an SSH-only server, it also prints exact loopback-forwarding
commands for completing authorization in the local browser. Read-only
connector requirements request read-only Google scopes.
Writing to an existing spreadsheet requires Google's broader spreadsheets
scope. See
[`google_sheets_records.py`](examples/google_sheets_records.py) for a complete
stable-key table example and [`call_intake.py`](examples/call_intake.py) for
the complete Gmail and Sheets service.

## Examples and documentation

Start with these examples:

| Example | What it shows |
|---|---|
| [`hello.py`](examples/hello.py) | Two lifelines and one LLM action |
| [`tutorial_review.py`](examples/tutorial_review.py) | Retry loop and human approval |
| [`google_sheets_records.py`](examples/google_sheets_records.py) | Configured Google Sheets reads and retry-safe writes |
| [`call_intake.py`](examples/call_intake.py) | Gmail intake, controlled replies, and a Google Sheets register |
| [`parallel.py`](examples/parallel.py) | Parallel branches |
| [`command_center.py`](examples/command_center.py) | Long-running application loops |
| [`codex_claude_review.py`](examples/codex_claude_review.py) | Bounded coding-assistant review loop |
| [`assistant_maintenance.py`](examples/assistant_maintenance.py) | First-class assistant action |

Documentation:

- [Your First ZipperGen Workflow](docs/first-workflow.pdf)
- [Development and Deployment Manual](docs/workflow-development-deployment-guide.pdf)
- [Call Intake end-to-end TeX source](docs/call-intake-end-to-end.tex)
- [Tutorial TeX source](docs/first-workflow.tex)
- [Manual TeX source](docs/workflow-development-deployment-guide.tex)
- [Workflow authoring skill](.agents/skills/zippergen-workflows/SKILL.md)

The first document is the shortest complete path. The manual contains the
concepts, full command reference, model setup, durable execution, connectors,
deployment, recovery, and troubleshooting.

The PDFs are built from the committed TeX sources:

```bash
make docs
```

## Formal foundation

The main result states that the projected local programs produce the same
behaviors as the global workflow. Deadlock freedom follows for well-formed
workflows in the supported formal model.

The main theorems are machine-checked in Lean 4. The formal results are
described in these papers:

- [Provable Coordination for LLM Agents via Message Sequence
  Charts](https://arxiv.org/abs/2604.17612)
- [Causal Past Logic for Runtime Verification of Distributed LLM Agent
  Workflows](https://arxiv.org/abs/2605.20923)

Causal Past Logic supports runtime conditions over causally visible distributed
state. It complements the design-time protocol and projection guarantees.

## License

ZipperGen is released under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the full terms.
