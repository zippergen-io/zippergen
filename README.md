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

ZipperGen separates **what agents do** (LLM calls, tool use, human input) from **how they coordinate** (the protocol). The protocol is readable and auditable. It gives a compact description of the coordination logic.

Each participant is called a **lifeline**, which is the standard term from [Message Sequence Charts](https://en.wikipedia.org/wiki/Message_sequence_chart) (MSCs), the formalism ZipperGen is based on. In practice a lifeline is simply an agent: one sequential thread of execution that sends and receives messages.

Executions can be inspected as message sequence charts in ZipperChat.

<p align="center">
  <a href="https://zippergen.io/demo"><strong>Try the demo →</strong></a>
</p>

![ZipperChat MSC view](assets/zipperchat-msc.png)

Clicking a human action opens a detail view with the full context and a form to respond.

![ZipperChat dialog view](assets/zipperchat-dialog.png)

## Quick start

```bash
git clone https://github.com/zippergen-io/zippergen.git
cd zippergen
uv sync
uv run zippergen
```

Python 3.11 or later. `prompt-toolkit` provides Studio's interactive terminal
experience; LLM backends remain optional. `pip install -e .` is an alternative
to `uv sync`.

## ZipperGen Studio

Running `zippergen` with no subcommand opens the project-aware development
workspace. It discovers the Git/project root, remembers the current workflow,
and makes the main path visible through `help` and numbered selectors.

Studio supports project-aware Tab completion. Command and subcommand menus are
supplemented with the workflows, participants, LLM-active participants,
providers, remembered deployment, and project files that are valid at the
cursor. When only one match exists, the bottom toolbar still displays its
description; multiple matches show their descriptions in the completion menu.
Up/down arrows navigate private per-project command history; a faint
history suggestion can be accepted with the right arrow. Piped commands and
programmatic callers retain the ordinary non-interactive input path.

For workflow development, the application project may be separate from the
framework checkout. This is especially useful while developing ZipperGen from
source:

```text
zippergen-tutorial/          # project, Git root, and coding-assistant root
├── .zippergen/              # generated current task; ignored by Git
├── zippergen.toml           # visible project contract
├── specification.md         # canonical, versionable design intent
├── workflows/
├── tests/
└── zippergen/               # optional local framework checkout
```

Pass the parent with `zippergen studio --project PATH` for the first session.
Inside Studio, `project init [NAME]` creates `zippergen.toml` and safe Git
ignores; `create` adds `specification.md`. A manifest takes precedence during
later project discovery; an explicit `--project` path is always used exactly.

When using a nested editable checkout, expose its CLI once and initialize the
parent from the parent directory:

```bash
uv tool install --force --editable ./zippergen
zippergen studio --project .
```

Then enter `project init NAME` inside Studio. Subsequent sessions need only the
short `zippergen` command from the parent project root.

```text
$ uv run zippergen
ZipperGen Studio
Project: /path/to/zippergen
No workflow selected.

zippergen [no workflow]> use

── Output: use ─────────────────────────────────────────
Workflows
  1. examples/tutorial_review.py:tutorial_review
  ...

zippergen [tutorial_review]> show

── Output: show ────────────────────────────────────────
  1. Overview
  2. Protocol
  3. Communications only
  4. Actions and prompts
  5. Complete workflow
  6. One participant
  7. Selected participants

zippergen [tutorial_review]> run
```

Studio gives every outcome the same compact visual language: green `✓` means
an operation or check succeeded, yellow `⚠` means attention or incomplete
optional setup, and red `✗` means the command or check failed. The symbols
remain in plain output, while ANSI color is enabled only on an interactive
terminal. Redirected output, `NO_COLOR`, and the scriptable CLI's JSON modes
remain color-free.

Interactive commands also begin with a consistent output boundary such as
`── Output: current ──`. The blank line and labelled rule separate the echoed
command from its prompts, tables, warnings, or errors. Only the command family
is shown; prompt text, paths, model specifications, and secret values are never
repeated in the boundary. Empty input and `exit` produce no boundary.

`current` is the concise project dashboard: project and manifest, canonical
specification and pending-refinement state, workflow name, all participants,
the explicit subset containing `@llm` actions, human actions, external effects,
validation state, effective per-lifeline model assignments, provider
readiness, connector bindings, run, and deployment. It remains useful before a
workflow exists; unknown fields are shown as `none` rather than guessed.

Connector bindings are intentionally reported but not configured in this
slice. Future Gmail, Google Sheets, Telegram, email, and human-channel adapters
will bind to declared lifeline capabilities without conflating the lifeline's
authority with credentials or transport configuration.

`run` validates first, guides you through every workflow input, creates a
unique durable SQLite run automatically, and presents human decisions in the
same terminal. There are no store paths, task IDs, or environment exports to
manage. If the terminal closes during an incomplete run, return to the project
and enter `resume`. Use `current` to see the remembered workflow, run, and
deployment context. Terminal human actions are always presented by the
supervisor's main thread, so `Ctrl-C` leaves the durable task pending, stops the
role threads before Studio accepts another command, and allows an immediate
`resume` without competing readers on stdin.

Use `models` to choose an inherited default and optional models for individual
LLM-active lifelines. The profile is remembered with the workflow and carried
into both development runs and guided deployment:

```text
zippergen [tutorial_review]> models default mock
zippergen [tutorial_review]> models set Writer openai:gpt-4o-mini
zippergen [tutorial_review]> models set Reviewer claude:claude-sonnet-4-6
zippergen [tutorial_review]> models show
```

Provider configuration is separate from model routing:

```text
zippergen [tutorial_review]> providers
zippergen [tutorial_review]> providers set openai
zippergen [tutorial_review]> providers set local http://127.0.0.1:11434/v1
zippergen [tutorial_review]> providers check local
```

API keys are entered without echo and remain in owner-only Studio secret
storage. Local endpoint settings and non-secret routing are remembered, while
`providers` displays readiness without ever displaying a key.
When `models default` or `models set` selects a configured API provider, Studio
queries that provider's model endpoint with the saved key before changing the
routing profile. A green check confirms that the exact model or alias is
available to that key; an unavailable model is rejected without changing the
profile. Local model identifiers are checked against the endpoint's live model
list. A temporarily unreachable provider produces an explicit yellow
“saved but unchecked” warning so offline configuration remains possible.
`providers set local` calls the endpoint's OpenAI-compatible `/models` route
with a short timeout and saves the URL only after a successful response. The
saved status includes the check time and model count. Use `providers check
local` after reconnecting an SSH tunnel or restarting the model server; a
failed recheck is displayed as unreachable instead of leaving an old green
status.

`run openai:gpt-4o-mini` remains a one-run override of the default; explicit
lifeline overrides remain in effect. If any selected provider needs a declared
API key, Studio asks for it without echo and saves it once in an owner-only
development secret file. Later runs and post-crash resumes reuse it; the value
is never copied into workspace, run, or request JSON.

Development and deployment secrets remain separate, but Studio does not make
you paste the same provider key twice. On the first deployment of a selected
real provider, Studio identifies a matching configured key by environment
variable name and asks whether to reuse it. Press Enter to accept the default.
The value is copied directly between private stores, is never displayed, and
becomes scoped to that named deployment. Answer `n` to enter a different
deployment credential. Later deployments with the same name retain their
existing deployment key without prompting again.

To begin from natural language, let Studio maintain one readable, versioned
`specification.md`. Studio owns the filename and opens it in a terminal editor.
Choose a project-specific editor preference once:

```text
zippergen [no workflow]> editor set micro
zippergen [no workflow]> editor show
zippergen [no workflow]> create
```

The remembered preference survives Studio restarts and computer crashes. A
one-off choice does not change it: use `--editor nano` on the `create`
command, or enter `edit workflow --editor micro`. Without a preference,
Studio tries `$VISUAL`, `$EDITOR`, then `micro`, `nano`, `vim`, and `vi`.
`editor reset` restores that automatic discovery. Studio runs the editor
directly in the existing terminal; this uses neither an LLM nor MCP. Commands
with arguments must be quoted for one-off use, for example
`--editor "code --wait"`.

After saving the specification and leaving the editor, Studio prepares the
coding-assistant handoff:

```text
zippergen [no workflow]> create

── Output: create ──────────────────────────────────────
Creation
────────
  Specification  ✓ specification.md
  Task           ✓ .zippergen/current-task.md
  Next             assistant codex · assistant claude
  Inspect          task · task show · task history

zippergen [no workflow]> assistant codex
```

`create` creates or reopens the fixed canonical file and waits for a successful
editor exit. A new file starts with a comment-only writing guide covering
durable intent while excluding filenames, tests, commands, and coding-assistant
instructions. Studio removes that guide after real requirements are saved and
will not turn an untouched guide into a task. No prompt filename or ID is
required. `spec show`, `spec edit`, and `spec path` inspect, edit, or locate the
same document. For a genuinely short experiment, `create DESCRIPTION` writes
it without opening an editor. The advanced `create --file PATH` form imports
an existing UTF-8 document into `specification.md`; its original filename does
not become project state.

For an existing selected workflow, `spec refine` creates or reopens exactly one
automatically named `.zippergen/pending-refinement.md`:

```text
zippergen [reviewed_answer]> spec refine
zippergen [reviewed_answer]> spec pending
```

Running `spec refine` again opens that same pending document. The short
`refine` command is an alias; `refine CHANGE` appends a small addition rather
than creating another permanent prompt file. Studio records a semantic
pre-change baseline and builds the handoff from the canonical specification,
the pending change, and the selected workflow.

The assistant must integrate the change coherently into `specification.md`
alongside code and tests, while leaving the pending document for human review.
This can also be done manually with `spec edit`. `spec reconcile` does not
perform a merge: after inspection it verifies that the canonical specification
changed, asks whether to accept that existing integration, archives the pending
text privately, and clears it. `refine CHANGE` appends only to the pending
document, never to the canonical specification. `spec discard` safely archives
an unwanted change; `spec history` lists both outcomes. Accepted specification
history belongs in Git.

The handoff also includes required source, tests, validation, semantic views,
and the no-deployment boundary. Studio writes the complete current handoff to
the fixed, generated `.zippergen/current-task.md` file and keeps timestamped
private copies in the project workspace. `task` summarizes it, `task show`
prints it, `task path` gives its absolute path for integrations, and `task
history` lists the private archive. A later `create` or `spec refine`
deliberately replaces the current task; `specification.md` remains the durable
design record.

The task cannot silently lag behind that record. Studio fingerprints the
canonical specification and pending refinement. While a task is still
`ready for assistant`, Studio compares the fingerprint before `assistant`,
`task`, `task show`, `task path`, or `current`. If either input document
changed, Studio generates one synchronized replacement and records which
request it refreshes. Once an assistant has run, expected edits no longer look
like stale task input: the same request moves to `awaiting human review` and is
preserved until it is reconciled, discarded, deliberately rerun, or closed.

`assistant codex` or plain `assistant` opens the locally installed Codex CLI;
`assistant claude` runs Claude Code's one-shot agent mode with project-local
edits accepted. Studio starts either tool in the project root and asks it to
execute the synchronized fixed task; Claude prints its report and returns to
Studio rather than opening an empty prompt. Thus
there is no separate prompt-copying step: the assistant receives the complete
specification context through `.zippergen/current-task.md`. Studio does not call
an assistant through a ZipperGen workflow provider and needs no ZipperGen API
key or MCP configuration. Install and authenticate the chosen tool once:
[`codex login`](https://learn.chatgpt.com/docs/developer-commands?surface=cli#cli-codex-login)
for Codex, or follow Anthropic's
[`claude` setup](https://docs.anthropic.com/en/docs/claude-code/getting-started).
Each assistant retains its own model settings, approvals, and independently
configured tools. MCP is optional, not part of the ZipperGen handoff. Another
repository-aware coding assistant can consume `task show` or the file path.

Assistant commands execute immediately and synchronously; Studio has no hidden
task queue or scheduled assistant job. Before launch, `task` reports `ready for
assistant` and `Execution: not started; nothing is scheduled`. A successful
return records the assistant and time, then reports `awaiting human review`
with the actual review commands. A failed or interrupted session remains
visible and retryable; after a Studio or computer crash, an orphaned `running`
record is recovered as `assistant interrupted` on the next inspection. Studio
blocks an accidental second execution while review is pending; use `assistant
codex --rerun` or `assistant claude --rerun` only when another pass is
intentional.

After the assistant creates visible Python source, `use` selects it. For an
existing workflow, `spec refine` additionally saves a semantic baseline for a
meaningful before/after diff. A refinement task closes through `spec reconcile`
or `spec discard`; after reviewing an initial creation task, `task close`
clears its stable mirror while retaining private task history.

Every later source/design change uses the same visible loop:

```text
zippergen [reviewed_answer]> spec refine
zippergen [reviewed_answer]> spec pending
zippergen [reviewed_answer]> task       # optional summary
zippergen [reviewed_answer]> assistant claude  # or: assistant codex
zippergen [reviewed_answer]> current
zippergen [reviewed_answer]> validate
zippergen [reviewed_answer]> show communications
zippergen [reviewed_answer]> run
zippergen [reviewed_answer]> spec show
zippergen [reviewed_answer]> spec reconcile
```

A model change has two forms. Use `models set Writer SPEC` or `models default
SPEC` when only the remembered run/deployment routing changes; no assistant is
needed. Use `refine` followed by an assistant when the choice belongs in
versioned design intent or requires source, action prompts, deployment
metadata, or tests to change. For example:

```text
zippergen [reviewed_answer]> refine Use openai:gpt-4o-mini for Writer and preserve all protocol behavior.
zippergen [reviewed_answer]> assistant claude
```

When the mock/fake development run is satisfactory, `deploy` enters the
existing guided, secret-aware deployment path explicitly. Use `--no-start` for
a reviewable prepare-first transition; Studio remembers the deployment name,
so normal operation stays short:

```text
zippergen [tutorial_review]> deploy tutorial-review --no-start
zippergen [tutorial_review]> doctor
zippergen [tutorial_review]> status
zippergen [tutorial_review]> start
zippergen [tutorial_review]> status
zippergen [tutorial_review]> logs
zippergen [tutorial_review]> restart
zippergen [tutorial_review]> stop
```

The same durable development flow is scriptable outside Studio:

```bash
uv run zippergen dev examples/tutorial_review.py:tutorial_review
uv run zippergen dev examples/tutorial_review.py:tutorial_review \
  --llm mock \
  --llm-for Writer=openai:gpt-4o-mini \
  --llm-for Reviewer=claude:claude-sonnet-4-6
uv run zippergen dev --resume
```

Workspace state and managed development stores live below
`~/.zippergen/workspaces/` by default. `ZIPPERGEN_HOME` is an optional advanced
override, not a required setup step.

`project reset` never has an implicit scope. It opens a three-choice menu:

1. **Fresh design cycle** archives `zippergen.toml`, `specification.md`, any
   legacy prompt directory, and all private Studio state. Workflow source,
   tests, Git history, the framework checkout, and deployments remain in
   place. `project init` then genuinely creates a new manifest, and `create`
   opens a new guided specification.
2. **Studio state only** archives managed runs, assistant-task and command
   history, model/provider preferences, development secrets, generated tasks,
   and pending refinements while keeping every visible project file.
3. **Cancel** changes nothing.

Every archive is owner-only and recoverable below `$ZIPPERGEN_HOME/resets/`.
The unambiguous noninteractive forms are `project reset fresh --yes` and
`project reset state --yes`; plain `project reset --yes` is intentionally not
accepted. Neither reset mode stops or removes deployments.

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

hello.configure("mock", ui=True)
result = hello(topic="Say hello to ZipperGen")
print(result)
```

`User` sends a value to `Writer`, `Writer` runs an LLM action, and the result comes back. The workflow says explicitly who owns each step. Open **http://localhost:8765** to watch the exchange in ZipperChat.

Switch to a real LLM with one line:

```python
hello.configure("openai:gpt-4o", ui=True)   # or "mistral", "claude"
```

The full example is at `examples/hello.py`.

## Owned decisions

The previous example has no coordination choice. Here is the first place where ZipperGen matters more: one lifeline owns a decision, and ZipperGen generates the required coordination messages automatically.

Three agents collaborate: `Writer` drafts a reply to an incoming email, `Editor` decides whether it is ready to send, and `Writer` revises if needed.

```python
from zippergen.syntax import Lifeline
from zippergen.actions import llm
from zippergen.builder import workflow

User   = Lifeline("User")
Writer = Lifeline("Writer")
Editor = Lifeline("Editor")

@llm(system="Draft a concise email reply.",
     user="{email}", parse="text", outputs=(("draft", str),))
def draft_reply(email: str) -> None: ...

@llm(system="Is this reply accurate and appropriate? Reply true or false.",
     user="{draft}", parse="bool", outputs=(("approved", bool),))
def approve_reply(draft: str) -> None: ...

@llm(system="Revise the reply to be clearer and more direct.",
     user="{draft}", parse="text", outputs=(("draft", str),))
def revise_reply(draft: str) -> None: ...

@workflow
def review_draft(email: str @ User) -> str:
    User(email) >> Writer(email)
    Writer: draft = draft_reply(email)
    Writer(draft) >> Editor(draft)
    Editor: approved = approve_reply(draft)
    if approved @ Editor:
        Editor(draft) >> User(draft)
    else:
        Editor(draft) >> Writer(draft)
        Writer: draft = revise_reply(draft)
        Writer(draft) >> User(draft)
    return draft @ User
```

`if approved @ Editor` is the key line. `Editor` owns the branching decision; ZipperGen automatically determines which agents need to receive that decision and generates the coordination messages. You don't write any routing code.

The same coordination pattern is at `examples/write_tweet.py`.

## Why protocols?

In most multi-agent frameworks, control flow lives inside each agent. Agents call tools, decide what to do next, and rely on the other agents being ready to receive. This works until a subtle ordering problem causes two agents to wait on each other indefinitely.

ZipperGen works differently. You write the control flow once, as a global protocol. ZipperGen then *projects* that protocol onto each agent: each agent receives exactly the local view of the global plan that it needs. Because every send has a corresponding receive by construction, deadlock cannot occur for well-formed protocols. This is a structural property, not something checked at runtime.

This protocol-first style is close to [choreographic programming](https://en.wikipedia.org/wiki/Choreographic_programming): the distributed behavior is written globally and then projected to local participants. ZipperGen uses an MSC-based formal model and adapts this idea to LLM actions, tool calls, human control points, and runtime inspection.

The formal statement is in [our paper](https://arxiv.org/abs/2604.17612): the projected programs produce exactly the same behaviors as the global program, and deadlock-freedom follows by structural induction.

The practical consequence: the global protocol is also a complete audit trail of what your agents are allowed to do. You can read it, reason about it, and show it to anyone who needs to understand how the system works.

## ZipperChat

Each lifeline gets its own column. Actions, messages, and human task events appear as cards as they happen. ZipperChat is now treated as a legacy visualization surface, not the primary deployment approval channel. For deployed systems, human approvals should go through SQLite-backed adapters such as `zippergen approve`, `zippergen notify telegram`, email, or Slack.

For local visualization, start a workflow with `ui=True` and open **http://localhost:8765**. Pass `show_decisions=True` to also show branch decisions and control broadcasts.

For applications that run several workflows from ordinary Python code, ZipperChat can show multiple independent runs on the same page:

```python
from zipperchat import WebTrace

dashboard = WebTrace.dashboard().start()
first_workflow.configure(ui=True, trace=dashboard)
second_workflow.configure(ui=True, trace=dashboard)
```

## Examples

Start without API keys:

```bash
python examples/hello.py                        # two lifelines, one LLM call
python examples/write_tweet.py                  # owned-decision loop
python examples/parallel.py                     # fan-out / fan-in across branches
python examples/human_approval.py               # legacy browser approval demo
python examples/command_center.py --llm mock    # long-running dashboard with two event loops
```

Coordination patterns (requires an API key):

```bash
python examples/diagnosis.py                    # two LLMs reach consensus iteratively
python examples/contract_review.py              # parallel review with owned branching
python examples/morning_digest.py               # inbox triage
```

Advanced:

```bash
python examples/planner.py                      # LLM generates a sub-workflow at runtime
python examples/cpl_test.py                     # causal runtime guard
python examples/dashboard.py                    # multi-run ZipperChat page
python examples/write_tweet_local.py            # local OpenAI-compatible model server
```

## Using real LLMs

Export your API key and pass the LLM spec to `configure()`:

```bash
export OPENAI_API_KEY=...
```

```python
workflow.configure("openai:gpt-4o", ui=True, timeout=600)
```

Supported specs: `"mock"`, `"openai:<model>"`, `"ollama:<model>"`, `"mistral:<model>"`, `"claude:<model>"`. You can omit the model and use env defaults, for example `"openai"`. For per-agent routing: `llm={"Writer": "openai:gpt-4o", "Editor": "mistral"}`.

## Inspecting Workflows As Code

ZipperGen can render semantic views directly in the terminal. These views are
generated from the workflow IR, so they do not require a diagramming tool and
are suitable for both human review and coding assistants.

In Studio, enter `show` for the selectable views below, or `show agent` for a
participant selector. The scriptable equivalents are:

```bash
# Complete global protocol
zippergen show examples/call_intake.py:call_intake

# Messages and control flow only
zippergen show examples/call_intake.py:call_intake --communications

# Exact local program produced by formal projection
zippergen show examples/call_intake.py:call_intake --agent Extractor

# Focus on selected agents; hidden peers remain explicit boundaries
zippergen show examples/call_intake.py:call_intake \
  --agents Mailbox,Extractor

# Include action declarations, or everything including prompts and deployment
zippergen show examples/call_intake.py:call_intake --detail actions
zippergen show examples/call_intake.py:call_intake --detail full
```

Detail levels are `overview`, `protocol`, `actions`, and `full`. Add
`--format json` for structured metadata plus the canonical code view.

Validate loading, every local projection, canonical rendering, and deployment
metadata before deploying:

```bash
zippergen validate examples/call_intake.py:call_intake
zippergen validate examples/call_intake.py:call_intake --json
```

Compare workflow meaning instead of relying only on source-line diffs. For two
separate workflow modules:

```bash
zippergen diff before.py:workflow after.py:workflow
zippergen diff before.py:workflow after.py:workflow --format json
```

When modifying a workflow in place, save a stable semantic baseline first:

```bash
zippergen snapshot path/to/workflow.py:workflow -o /tmp/workflow-before.json
# edit path/to/workflow.py using a coding assistant or editor
zippergen validate path/to/workflow.py:workflow
zippergen diff /tmp/workflow-before.json path/to/workflow.py:workflow
```

The diff reports changes to participants, owned inputs and outputs, messages,
control context, action kinds and implementations, parallel regions, and
deployment requirements while ignoring irrelevant source layout.

## Creating And Refining Workflows From Prompts

The intended authoring loop is prompt → Python workflow → validated semantic
views. The coding assistant performs the open-ended translation; ZipperGen
provides the deterministic protocol validation, projections, views, and diffs.
This keeps generated workflows as ordinary reviewable code instead of hiding
them behind a separate visual builder or opaque generation service.

Studio exposes this handoff as `create` and `spec refine`. Multiline accepted
requirements remain in one normal, versioned specification:

```text
zippergen [no workflow]> create
zippergen [reviewed_answer]> spec refine
```

These are not disposable chat messages. Studio gives the coding assistant the
canonical `specification.md`, one pending refinement, and the current workflow.
The assistant integrates an accepted change back into the canonical document;
Git preserves its history. The Python workflow remains executable truth while
the specification remains durable intent.

Studio stores timestamped assistant requests and semantic baselines outside
the Git checkout under the project workspace. It mirrors only the current
generated task at `.zippergen/current-task.md`, which `project init` adds to
`.gitignore`. The canonical specification is ordinary project input; never put
API keys or other secrets in it, the pending refinement, or generated tasks.

This repository includes a reusable coding-assistant skill at
`.agents/skills/zippergen-workflows/`. Codex discovers it automatically, and
`AGENTS.md` directs repository-aware assistants to it. Give the assistant one
or several prompts such as:

> Create a workflow that watches a support inbox. A triage agent classifies each
> request, billing and technical specialists work independently when both are
> needed, and a human approves any refund over €100. Include guided deployment.

For an existing workflow, describe the change and the behavior to preserve:

> Extend `support.py:support` so enterprise refunds also require the account
> owner's approval. Preserve the current routing for all other requests. Show
> me the communication-only view and the local projections for Triage and the
> account owner, then report the semantic diff.

The bundled assistant workflow extracts participants, ownership, messages,
actions, decisions, concurrency, human authority, and deployment requirements;
edits the Python module and tests; runs `validate`; renders the requested global
and local code views; and verifies refinements against a pre-edit semantic
snapshot. Deployment is still a separate explicit action, so generating code
does not silently start services or perform live effects.

For a complete beginner-oriented walkthrough—from installation and a mock
workflow through prompt-driven refinement, semantic diff, durable approval,
guided deployment, and supervised operation—see
[`docs/workflow-development-deployment-guide.tex`](docs/workflow-development-deployment-guide.tex).

## Local Deployment

The guided path configures, validates, and starts a workflow in one command:

```bash
zippergen deploy examples/call_intake.py:call_intake
```

Inside Studio, select the workflow once and enter `deploy NAME --no-start` to
prepare it without starting a service. Inspect it with `doctor`, then use
`start` when authorized. Subsequent `status`, `logs`, `doctor`, `restart`, and
`stop` commands use the remembered deployment name.

When a workflow declares deployment requirements, ZipperGen asks for its
settings and secrets, creates a managed Python environment, installs declared
packages, runs one-time setup such as OAuth, checks readiness, snapshots the
workflow files, and starts a user service. It uses launchd on macOS and systemd
on Linux.

Normal configuration is stored in the deployment profile. Secrets are kept in
a separate mode-0600 file and loaded before the workflow module is imported;
they do not appear in the profile or generated service definition.

Day-to-day operation uses the deployment name:

```bash
zippergen status call-intake
zippergen logs call-intake --follow
zippergen doctor call-intake
zippergen restart call-intake
zippergen configure call-intake --restart
```

Run `zippergen deploy call-intake` again to snapshot and deploy updated source.
The stable SQLite store is retained, so committed workflow work is replayed
instead of repeated.

Workflow modules describe the guided experience with data-only declarations
that are also straightforward for workflow-generating LLMs to emit:

```python
from zippergen import DeploymentField, DeploymentPackage, DeploymentSpec

zippergen_deployment = DeploymentSpec(
    name="my-workflow",
    fields=(
        DeploymentField("llm", "Model", target="llm", default="openai:gpt-4o"),
        DeploymentField(
            "openai_key", "OpenAI API key",
            target="env", env="OPENAI_API_KEY", secret=True, required=True,
        ),
    ),
    packages=(DeploymentPackage("some-client", "some_client"),),
    files=("workflows/my_workflow.py",),
)
```

For quick experiments, `zippergen run` remains available:

```bash
zippergen run examples/hello.py:hello \
  --llm openai:gpt-4o \
  --input topic="Say hello to ZipperGen"
```

The workflow spec can be `module:workflow` or `path.py:workflow`. Runs and named
deployments use persistent SQLite stores under `~/.zippergen/runs/` by default.
Use `--ui` only for the legacy ZipperChat visualization; deployment approvals
remain in SQLite-backed tasks and notification adapters.

Inspect and complete human approvals without a browser:

```bash
zippergen tasks --store ~/.zippergen/runs/command-center.sqlite
zippergen approve --store ~/.zippergen/runs/command-center.sqlite --task <task-id>
zippergen approve --store ~/.zippergen/runs/command-center.sqlite --task <task-id> --no
zippergen approve --store ~/.zippergen/runs/command-center.sqlite --task <task-id> --value "edited reply"
```

External adapters can use durable approval tokens instead of raw task ids:

```bash
zippergen tasks --store ~/.zippergen/runs/command-center.sqlite --tokens --channel telegram
zippergen approve --store ~/.zippergen/runs/command-center.sqlite --token <token>
```

The first notification adapter prints pending tasks with approval commands:

```bash
zippergen notify stdout --store ~/.zippergen/runs/command-center.sqlite --channel telegram
zippergen notify stdout --store ~/.zippergen/runs/command-center.sqlite --channel telegram --watch
```

Telegram approvals are available as a real notification adapter:

```bash
export ZIPPERGEN_TELEGRAM_TOKEN=<bot-token>
export ZIPPERGEN_TELEGRAM_CHAT_ID=<chat-id>
zippergen notify telegram --store ~/.zippergen/runs/command-center.sqlite --watch
```

For deeper setup details, see the beginner deployment booklet in
[`docs/local-deployment.md`](docs/local-deployment.md).

## Formal foundation

The implementation is based on the theory of [Message Sequence Charts](https://en.wikipedia.org/wiki/Message_sequence_chart) and [choreographic programming](https://en.wikipedia.org/wiki/Choreographic_programming). A workflow is written from a global point of view and projected to local participants; ZipperGen adapts this to LLM actions, tool calls, human control points, and runtime inspection.

The key properties:

- **Correctness**: The distributed projected programs produce exactly the same behaviors as the global program.
- **Deadlock-freedom**: Follows by structural induction; no runtime checking required.

The main theorems (Theorem 3.1 and Corollary 3.1) have been machine-checked in Lean 4; see the [formalization](https://github.com/zippergen-io/paper-isola/tree/main/Lean).

Bollig, Függer, Nowak. [*Provable Coordination for LLM Agents via Message Sequence Charts.*](https://arxiv.org/abs/2604.17612) arXiv:2604.17612 [cs.PL]

Bollig. [*Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows.*](https://arxiv.org/abs/2605.20923) arXiv:2605.20923 [cs.LO]

## License

ZipperGen is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full terms.
