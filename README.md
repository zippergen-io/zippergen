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
workflow edit-spec
workflow implement codex
workflow validate
run
deploy
```

`workflow edit-spec` opens one versioned `specification.md`. A coding assistant
may implement that specification as ordinary Python code and focused tests.
The assistant never deploys the result. Studio summarizes the files changed,
assistant checks, and assistant report when implementation finishes. Validation,
inspection, running, and deployment remain explicit steps.

Commands are grouped by purpose:

| Area | What it covers |
|---|---|
| `workflow` | Specification, implementation, views, validation, and differences |
| `model` | Providers, named configurations, checks, assignments |
| `connector` | Provider credentials, reusable resources, human-action routes |
| `run`, `resume`, and `runs` | Durable development execution, decisions, traces, recovery |
| `deploy` | Bundles, services, durable state, decisions, and logs |

Named local Ollama configurations can release their model after an idle
period. The workflow stays active, and the model loads again automatically at
the next LLM action.

Tab completion shows valid commands, workflows, participants, model
configurations, and deployments. `project` shows the whole inventory as one
tree. The workflow, specification, models, connectors, runs, and deployments
are visible in one place. `current` answers a different question. It shows
what you were most recently working on. Every structured result ends with a
useful `Next` section.

### Read the communication protocol

The communication-only view hides local action details. It keeps messages,
loops, decisions, and ownership visible.

![ZipperGen Studio communication-only workflow view](assets/studio-workflow-communications.svg)

Other views show an overview, the full protocol, actions and prompts, complete
source, selected participants, or one exact local projection.

### Understand a deployment

`deploy show` separates the installed bundle, service process, workflow
run, SQLite store, model routing, connectors, boot policy, and likely failure
cause.

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

### Structured workflow data

Workflow variables can carry structured data directly with the `Json` type:

```python
from zippergen import Json, Lifeline, workflow

Requester = Lifeline("Requester")
Worker = Lifeline("Worker")


@workflow
def process_record(record: Json @ Requester) -> Json:
    Requester(record) >> Worker(record)
    Worker(record) >> Requester(record)
    return record @ Requester
```

A `Json` value is normal Python data made from `None`, booleans, numbers,
strings, lists, and dictionaries with string keys. ZipperGen validates nested
values before durable execution. It preserves them across messages, crashes,
and resume. Arbitrary Python objects and pickle data are deliberately not
supported. Lists and dictionaries must be ordinary built-in containers.

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
workflow show
workflow show communications
workflow show agent Reviewer
workflow validate
workflow diff
```

`workflow import` copies an existing workflow into the current project. It
also copies statically imported local Python modules and literal resource files
declared by the workflow. If the project already has a workflow, Studio shows
what will be replaced and asks before overwriting files or changing the entry
point. If the imported file contains several workflows, name the intended one
as `PATH.py:WORKFLOW`.

One project has one configured workflow. `workflow import` writes its entry
point to the visible, versioned `zippergen.toml` manifest. Pointing it at a
file already inside the project only adopts that entry point and copies
nothing. `workflow list` remains a read-only source scan. A clone containing
the manifest therefore knows its specification and workflow without private
Studio state.

`workflow validate` is a machine check. It checks structure, projection,
metadata, and canonical rendering. `workflow diff` shows the available
specification and semantic baseline again in detail.

Running never requires an implementation record. `workflow implement` writes
the portable `zippergen.lock` record that relates the generated source to the
current specification. Studio derives four states from committed project files:
`absent`, `stale`, `current`, or `external`. A fresh clone derives the same
state without private Studio data. Deployment blocks `absent` and `stale`.
It warns and proceeds for `external`, because provenance is unknown rather
than known to disagree. Two commands produce `current`, from opposite
directions: `workflow implement` makes the code follow the specification, and
`workflow adopt` writes a specification for code Studio did not generate,
leaving every implementation file untouched. `current` means the two
correspond, not that one produced the other.

In an interactive Git project, `workflow implement` then offers to commit the
specification, implementation files, `zippergen.lock`, and a manifest that it
changed. The suggested message is editable. Studio commits only that unit and
never pushes. `workflow status` and `deploy` warn when the unit has uncommitted
changes, because a fresh clone could derive a different state. The warning
never blocks deployment. Outside Git, and in non-interactive use, Studio stays
silent and behaves as before.

Project configuration travels with the workflow. Named model and connector
configurations, participant and action assignments, and connector bindings are
written to `zippergen.toml`. Machine-specific facts stay private on each site.
These include local model endpoints, idle-release policy, API keys, bot tokens,
and Google authorization. Health checks are always live and are never stored,
so there is nothing about them to keep in step between machines. The effective
value is one lookup: a private site override wins when present, otherwise
Studio uses the project value. For example, keep the real model assignment in
the project and use `model assign Writer mock --site` on a laptop. The `project` view lists only
the site facts and secrets still missing after a fresh clone.

For a described specification change, use `workflow edit-refinement` and then
`workflow refine-spec`. The first command writes an ignored scratch buffer.
The second lets an isolated assistant see only the specification and requested
change, rewrites `specification.md`, and consumes the buffer. It never exposes
the implementation to the specification assistant.

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

Runtime `@planner` actions use a separate fail-closed boundary. Generated text
may contain only the supported workflow statements and schema-checked `@llm`
declarations. Imports, module-level code, Python function bodies, and generated
`@pure` helpers are rejected before the text is imported. A generated workflow
can call only the reviewed actions passed through `actions=` and the generated
`@llm` actions explicitly enabled by `allow=`.

## Durable runs and deployment

Development runs use SQLite by default. Messages, completed actions, human
tasks, traces, results, and participant positions survive terminal closure and
computer restart.

Inside Studio:

```text
run
run inspect
run inspect Reviewer
run inspect Reviewer --watch
run tasks
run approve
run trace
resume
runs
```

Every new development run gets its own managed durable state. Users work with
the run and do not need to manage SQLite files. The run records its model,
assistant, and connector routing when it starts. `resume` uses that recorded
routing. Later project configuration changes do not silently redirect an
incomplete run.

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
deploy inspect review-demo --watch
deploy tasks review-demo
deploy approve review-demo
deploy trace review-demo
deploy storage review-demo
deploy logs review-demo
deploy logs reset review-demo
deploy restart review-demo
deploy stop review-demo
deploy storage compact review-demo
deploy remove review-demo
```

`--watch` updates the participant positions and focused local projection once
per second in one fixed terminal view. Only changed terminal cells are redrawn,
so pointer movement remains stable instead of blinking. Press Ctrl-C to
restore the Studio screen. The run or deployment keeps running.

`deploy show` also compares the installed deployment with the current project.
It reports changes to the specification, workflow, models, idle
policy, assistant, connectors, and local model endpoint. A changed project does
not alter a running deployment. Redeploy to apply those changes.

`deploy storage` shows the size of the durable store, WAL, active log, and log
archives. It runs SQLite's structural quick check, then shows event counts,
snapshot coverage, and how much history can be removed safely. `deploy doctor`
runs the same integrity check when requested explicitly. Diagnostic traces are
pruned online in batches. The target is 10,000 traces and the store never keeps
more than 10,999. This does not stop the service. Completed human tasks, tokens,
and notifications remain as audit records.

After stopping the deployment, `deploy storage compact NAME` removes completed
events that are covered by recovery snapshots, rotates the active log, and
keeps the three newest log archives. Seed inputs, pending work, and events
still needed for recovery remain.
Studio then compacts the database file and truncates the WAL while preserving
the stable event identifiers used for recovery.
Deployments created before this feature must be redeployed once before safe
compaction is enabled.

Durable stores must live on a local filesystem with reliable SQLite locking and
`fsync`. Do not place them on NFS, a synchronized folder, or an unverified
remote container volume. ZipperGen creates store files with owner-only
permissions and explicitly uses WAL with `synchronous=FULL`. Do not edit the
database directly. Compaction legitimately removes recovery-safe rows, so an
ever-increasing row count is not a validity check.

External effects have a different boundary. A crash or restoration from an
older backup can repeat an effect whose remote result was not present in the
restored journal. Gmail sends, payments, appends, and similar operations should
use stable idempotency keys or an idempotent API.

The service policy restarts failed workflows. It does not restart a finite
workflow after successful completion. `deploy stop` preserves the deployment.
`deploy remove` stops it, unregisters the user service, and moves its profile,
private secrets, bundles, environment, store, and logs into a private archive.
Use `deploy remove review-demo --purge` only when those files must be deleted
permanently. A permanent purge requires the explicit deployment name and a
second confirmation.

Studio discovers human actions from workflow code. Their delivery uses the
same provider, configuration, and assignment lifecycle as models. Named
configurations and assignments are versioned. Private credentials remain
outside Git:

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
workflow declares logical mailbox and table operations. Studio versions the
Gmail query and concrete spreadsheet choice in `zippergen.toml`, while Google
authorization remains private on each site:

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

For the call-intake workflow, create a separate project directory first. This
setup then authorizes Gmail and Sheets, asks for the mailbox query,
spreadsheet, and tab, checks both resources, and binds the saved
configurations:

```text
project init call-intake
workflow import /path/to/zippergen/examples/call_intake.py:call_intake
connector setup
connector assignments check
deploy call-intake
```

The default reply mode creates Gmail drafts. Change it to `send` only after
testing with a dedicated account. Before setup, enable the Gmail and Google
Sheets APIs in one Google Cloud project and download a Desktop app OAuth
client. If Studio runs on a server without a browser, it prints one command to
run on your own computer, not on the server. `uvx` fetches the helper without
a local ZipperGen installation. Copy the complete `zg-google-v1...` line it
produces into Studio's hidden prompt. No client file, `scp` command, or SSH
tunnel is needed on the server. Studio checks that Google actually granted
every permission required by the workflow. Read-only connector requirements
request read-only Google scopes.
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
- [Call Intake End to End](docs/call-intake-end-to-end.pdf)
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

The main theorems are [machine-checked in Lean
4](https://github.com/zippergen-io/zippergen-lean/tree/main/isola). The formal
results are described in these papers:

- Bollig, Függer, and Nowak. [Provable Coordination for LLM Agents via
  Message Sequence Charts](https://arxiv.org/abs/2604.17612). Accepted at
  ISoLA 2026.
- Bollig. *Deadlock-Free Parallel Regions for Projected Workflows*. Accepted
  at EXPRESS/SOS 2026. Preprint forthcoming.
- Bollig. [Causal Past Logic for Runtime Verification of Distributed LLM Agent
  Workflows](https://arxiv.org/abs/2605.20923). Under submission.

Causal Past Logic supports runtime conditions over causally visible distributed
state. It complements the design-time protocol and projection guarantees.

## License

ZipperGen is released under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the full terms.
