<p align="center">
  <img src="https://raw.githubusercontent.com/zippergen-io/zippergen/main/assets/zippergen-lockup-ink.svg" alt="ZipperGen" width="420">
</p>

<p align="center">
  <a href="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml"><img src="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://arxiv.org/abs/2604.17612"><img src="https://img.shields.io/badge/arXiv-2604.17612-b31b1b.svg" alt="arXiv"></a>
</p>

## Write one workflow. Run it as a service.

ZipperGen is a Python framework for workflows with LLM agents, people, and
services. You write who does what and who makes each decision. ZipperGen
derives the program for each participant and runs them.

It also saves workflow state, keeps track of human approvals, and runs the
workflow as a service on your machine or server. You use the same CLI to
configure models and services, check the project, deploy it, and see what is
happening.

---

## From code to a running service

Take an approval workflow. It waits for a message, asks a model to draft a
reply, and asks you before sending it. After handling that message, it waits
for the next one.

You write the workflow and the actions it calls. Before deployment, configure
the model and any external services it needs. Choose how you want to receive
approval requests, for example through Telegram. Then check and deploy the
project:

```bash
zg validate
zg check --strict
zg deploy
```

ZipperGen starts the service and saves its progress. Pending approvals stay
available across a restart. From the same project directory, you can see what
needs attention and how the service is running:

```bash
zg deploy tasks
zg deploy status
zg deploy logs
```

You can answer in the configured chat or use `zg deploy approve`. The
[quick start](#quick-start) runs the approval workflow in your terminal. The
[deployment guide](https://github.com/zippergen-io/zippergen/blob/main/docs/workflow-development-deployment-guide.pdf)
covers setup and running it as a service on macOS or Linux.

## How you work with it

A ZipperGen project is an ordinary directory. It contains a Python workflow, a
plain-text specification, and a small TOML file. You can edit it directly or
work with a coding agent such as Claude Code or Codex.

No special editor or hosted environment is required. `zippergen skill` gives a
coding agent its project instructions. You and the agent use the same CLI.
The workflow stays in Python files that you can read, review, and change.

## Install

```bash
uv tool install zippergen
```

Or with pipx, or into a virtual environment:

```bash
pipx install zippergen
python3 -m venv .venv && .venv/bin/pip install zippergen
```

To update an installation managed by uv:

```bash
uv tool upgrade zippergen
```

Gmail, Google Sheets and Google Calendar need one extra:

```bash
uv tool install "zippergen[google]"
```

ZipperGen needs Python 3.11 or newer. It has no other dependencies. It
installs two commands: `zippergen`, and `zg` for short.

### Running this checkout

The repository can contain features that are not yet on PyPI. From its root,
install the checkout and run its CLI explicitly:

```bash
uv sync
.venv/bin/python -m zippergen.serve --help
```

When running commands from the repository root, use
`.venv/bin/python -m zippergen.serve` in place of `zg`. A globally installed
`zg` continues to use its own installed package.

## Quick start

```bash
mkdir email-approval && cd email-approval
zippergen init
```

This creates a manifest, an empty specification, shared agent instructions,
and a small pointer that makes Claude Code read them:

```
ZipperGen project: email-approval
  zippergen.toml     created
  specification.md   created
  AGENTS.md          created
  CLAUDE.md          created
```

Now say what you want. You can write the workflow yourself, or open a coding
agent in that directory:

```bash
claude  # or: codex
```

Then ask:

> Build a ZipperGen workflow that watches plain `.txt` files in `mailbox/`,
> asks an LLM to draft a reply, and asks me to approve it before sending. It
> should keep waiting for new messages.

The agent follows the instructions from `zippergen skill`, writes
`specification.md` and `workflow.py`, and validates the result. The workflow
remains ordinary Python:

```python
message = Var("message", str)
item = Var("item", Json)
draft = Var("draft", str)
approved = Var("approved", bool)
handled = Var("handled", int, default=0)
processed = Var("processed", int, default=0)


@workflow
def email_approval() -> int:
    Mailbox: item = next_unread_message(processed)
    while (item is not None) @ Mailbox:
        Mailbox: message = message_text(item)
        Mailbox(message) >> Writer(message)
        Writer: draft = draft_reply(message)
        Writer(draft) >> Mailbox(draft)
        Mailbox: approved = approve_reply(draft)
        if approved @ Mailbox:
            Mailbox: handled = send_reply(draft, handled)
        else:
            Mailbox: handled = discard(handled)
        Mailbox: processed = complete_message(item, processed)
        Mailbox: item = next_unread_message(processed)
    return handled @ Mailbox
```

The item keeps the filename and message together. The workflow marks that
file as done after handling it. Reading an input leaves it available if the
process stops before saving the read result.

Check it and run it:

```bash
zg validate

mkdir -p mailbox
echo "Can we meet on Thursday" > mailbox/01.txt
zg run --llm mock
```

Validation states the workflow inputs explicitly. For this workflow it must
include:

```
OK   workflow inputs: none, the run starts without setup questions
```

```
No real model is in use: every participant answers with the mock. Assign one with 'zg model assign TARGET NAME'.

REQUEST · Mailbox

Proposed reply:

[draft_reply:draft]

Send this reply? [y/n]: y
✓ Mailbox · reply sent
```

The reply is a placeholder, because `mock` does not call a model. Use
`--llm openai:gpt-4o-mini`, with a key in your environment, to get a real one.

Then it waits for the next message. Press Ctrl-C to stop it.

The commands need no workflow name because the project already identifies it.

The tutorial goes through all of this step by step, including approval on your
phone and a real deployment:
[**Your first ZipperGen workflow**](https://github.com/zippergen-io/zippergen/blob/main/docs/first-workflow.pdf).

## What you get from writing one protocol

The workflow above has one decision, and `Mailbox` owns it. Its `@human`
action pauses that local program and asks a person. You can ask ZipperGen what
each participant really runs:

```bash
zg show --agent Mailbox
```

```python
@role('Mailbox')
def email_approval__Mailbox() -> int:
    item = next_unread_message(processed)
    while item is not None:
        send_decision('Writer', True)
        message = message_text(item)
        send('Writer', message)
        draft = recv('Writer')
        approved = approve_reply(draft)
        if approved:
            handled = send_reply(draft, handled)
        else:
            handled = discard(handled)
        processed = complete_message(item, processed)
        item = next_unread_message(processed)
    else:
        send_decision('Writer', False)
    return handled
```

```bash
zg show --agent Writer
```

```python
@role('Writer')
def email_approval__Writer() -> None:
    while recv_decision('Mailbox'):
        message = recv('Mailbox')
        draft = draft_reply(message)
        send('Mailbox', draft)
```

**The Writer has no approval branch.** ZipperGen generated both local programs
from the workflow.

At each loop iteration, the Writer learns whether another iteration follows.
It receives no approval result because neither approval branch contains Writer
work. Its projected program therefore cannot wait for that decision. Each
participant receives only the coordination it needs.

## Where next

The quick start covers creating, inspecting, validating, and running a local
workflow. The sections below summarize configuration, repeatable tests, and
deployment. For step-by-step instructions, use these guides:

- [Your first ZipperGen workflow](https://github.com/zippergen-io/zippergen/blob/main/docs/first-workflow.pdf)
- [Development and deployment guide](https://github.com/zippergen-io/zippergen/blob/main/docs/workflow-development-deployment-guide.pdf)
- [Durable storage](https://github.com/zippergen-io/zippergen/blob/main/docs/durable-storage.md)

## Configuration

A connector links a workflow to an external service or human channel, such as
a Telegram chat, Gmail mailbox, Google Sheet or Google Calendar.

Provider connections, models, coding assistants, and connectors follow the
same configuration pattern:

```text
zg provider configure NAME PROVIDER_KIND
zg model configure NAME CONNECTION MODEL
zg assistant configure NAME BACKEND
zg connector configure NAME CONNECTION [CONNECTOR_KIND]

zg model assign TARGET NAME
zg assistant assign TARGET NAME
zg connector assign TARGET NAME

zg FAMILY check [NAME]
zg FAMILY remove NAME
```

`FAMILY` is `provider`, `model`, `assistant`, or `connector`. Square brackets
mark an optional value. The connector kind is inferred when the selected
connection supports only one.

A provider connection stores access to one external provider, including its
private credential and any machine-specific endpoint. Model and connector
configurations reuse that connection. `connector assign` accepts a service
requirement or a human-action target. The workflow tells ZipperGen which kind
of target it is.

When you work in a terminal, you may leave out required values. ZipperGen asks
for them and shows available targets and saved configurations. For example,
`zg model configure`, `zg assistant configure`, and `zg connector configure`
are all guided. Reusing a name updates that configuration and presents its
current values as defaults. Scripts and coding agents should pass every value
explicitly.

For an `@assistant` action, choose Codex or Claude with a named configuration:

```bash
zg assistant configure coding-agent codex
zg assistant assign Maintainer coding-agent
zg assistant check
```

Assign `Maintainer.action_name` when only one action needs a different
backend. The `@assistant` declaration still controls filesystem access,
external tools, and shell access. Codex and Claude use their own login.
ZipperGen does not pass workflow model keys or connector credentials to them.

## Models and repeatable tests

Give the Writer a named model configuration, then assign it:

```bash
zg provider configure openai-main openai
zg provider set-credential openai-main
zg model configure writer openai-main gpt-4o-mini
zg model assign Writer writer
zg model
```

The credential command prompts without echo. The key is saved in the owner-only
`$ZIPPERGEN_HOME/workspaces/<project>/development.secrets.json` file on this
computer. It is not written to `zippergen.toml`. You may use the standard
`OPENAI_API_KEY` environment variable instead.

`zg run`, `zg run --durable`, and `zg deploy` all use that assignment.
`--llm mock` temporarily replaces all project assignments. Use
`--llm-for Writer=SPEC` only for a narrower one-command override.

For repeatable tests, put fixed model answers in a file:

```json
{
  "draft_reply": {"draft": "Thursday afternoon works for me. How about 3pm?"}
}
```

```bash
zg run --llm scripted:replies.json
```

Answers are used in order, per action. A single object answers every call the
same way. A list is consumed once. The run fails if it asks for more answers
than the file provides.

Scripted answers cover model actions only. A `@human` action still asks a
person. You can answer it at the terminal or pipe the answer in:

```bash
printf 'n\n' | zg run --llm scripted:replies.json
```

## Work pools (unreleased)

Use a pool when several workers should compete for available jobs. Define it
once, then use its ordinary local actions inside a workflow:

```python
from zippergen import Pool

jobs = Pool("jobs")
```

| Action inside a workflow | Meaning |
|---|---|
| `Producer: job_id = jobs.put(payload)` | Submit a JSON job. |
| `Worker: claim = jobs.try_claim()` | Claim the oldest ready job, or return `None` immediately. |
| `Worker: done = jobs.ack(claim)` | Mark a claimed job as handled. |
| `Worker: released = jobs.release(claim)` | Return unfinished work to the back of the ready queue. |

Branch on whether the claim is `None`, process `claim["payload"]`, and
acknowledge only after handling it. An empty result means no job is available
now. It does not mean the workflow is finished. Use explicit messages when
submission must precede another participant's claim attempt.

Each execution owns its pool in the managed SQLite store. No separate queue
service is needed. Durable runs retain jobs and operation receipts across
restart. Claims expire after 300 seconds by default, configurable with
`Pool("jobs", lease_seconds=...)`, and are not renewed automatically. Released
and expired jobs rejoin the back of the queue. Jobs are selected in FIFO order.
Worker fairness and completion order are unspecified. Processing may repeat,
so external work still needs its own idempotency protection.

See the [two-worker example](https://github.com/zippergen-io/zippergen/blob/main/examples/work_pool/workflow.py)
and its [specification and run commands](https://github.com/zippergen-io/zippergen/blob/main/examples/work_pool/specification.md).
Jobs automatically carry CPL context from their completed `put` or latest
explicit `release` to a successful claim. See the
[CPL approval example](https://github.com/zippergen-io/zippergen/blob/main/examples/work_pool_cpl/workflow.py),
which checks approval and job identity without a producer-to-worker message. The
[durable storage guide](https://github.com/zippergen-io/zippergen/blob/main/docs/durable-storage.md)
explains their recovery guarantees.

## Durable runs and deployment

Add `--durable` when you want to stop and resume a run:

```bash
zg run --durable --llm mock   # Ctrl-C part way through
zg run inspect --agent Writer # see where each participant is waiting
zg run --resume               # carry on where it stopped
```

A plain `zg run` leaves no resumable state. A durable run records coordination
state and completed external-action results. An external effect can still
repeat if the process crashes after the effect succeeds but before its result
is recorded. Use idempotency keys for irreversible operations.

A project can have only one active execution, either a foreground run or its
deployment. Status, inspection, trace, and task commands remain available
while it runs.

For a live view, keep the run open in one terminal and use another terminal:

```bash
zg run inspect --watch --agent Writer
```

Ctrl-C closes the view without interrupting the workflow. Use
`zg deploy inspect --watch` for a deployment.

`zg deploy` builds an immutable release, checks its models and connectors,
installs it as a supervised systemd or launchd user service, and starts it:

```bash
zg deploy
zg deploy status
zg deploy logs
zg deploy inspect --watch
zg deploy trace --follow
```

After changing the workflow, stop and redeploy it:

```bash
zg deploy stop
zg deploy       # rebuild and start the updated deployment
```

Before applying an update, ZipperGen checks the saved workflow state against
the new protocol. If their structure no longer matches, it keeps the previous
deployment and saved state and explains how to proceed.

The [deployment walkthrough](docs/reviews/2026-09-12-deployment-walkthrough.md)
records a fresh installation, approvals, process recovery, and workflow updates
on macOS with a simulated model provider. It also describes the limits of
those checks.

A workflow can ask a person on Telegram, read Gmail, write to Google
Sheets or create a Google Calendar event after approval. The destination belongs
in project configuration. Credentials stay in private state on the machine that runs ZipperGen. For Telegram:

```bash
zg provider configure approval-bot telegram
zg provider set-credential approval-bot       # hidden bot-token prompt
zg connector configure approval-chat approval-bot  # Telegram is inferred
zg connector assign Mailbox approval-chat
zg check
zg deploy
```

`zg config` shows effective routing and local credential readiness without
contacting providers. `zg check` performs readiness checks and may make a small
model request. Add `--strict` when a script should fail on anything that is not
ready.

The [development and deployment guide](https://github.com/zippergen-io/zippergen/blob/main/docs/workflow-development-deployment-guide.pdf)
covers Google authorization, Linux services, resets, removal, and recovery.
The [durable storage guide](https://github.com/zippergen-io/zippergen/blob/main/docs/durable-storage.md)
explains crash behavior, identity checks, and trace retention.

## The CLI

The public command surface fits in one tree:

```text
zg
├── init · skill · validate · show · snapshot · diff · check
├── config
├── workflow
│   └── select
├── provider
│   └── configure · set-credential · check · rename · remove · authorize · accept
├── model
│   └── configure · assign · unassign · check · rename · remove
├── assistant
│   └── configure · assign · unassign · check · rename · remove
├── connector
│   └── configure · assign · unassign · check · rename · remove
├── run
│   └── status · reset · inspect · trace · tasks · approve
├── deploy
│   └── list · prune · start · stop · status · logs · check
│       · inspect · trace · tasks · approve · compact · reset · remove
└── completion
```

`zg --help` renders this tree from the real command parser, so it cannot drift
from the implementation. Run `zg <command> --help` for arguments and examples.

Commands for durable state begin with its owner. To list human tasks, run
`zg run tasks` for a durable run or `zg deploy tasks` for a deployment. The
[development and deployment guide](https://github.com/zippergen-io/zippergen/blob/main/docs/workflow-development-deployment-guide.pdf)
documents reset, removal, compaction, and recovery.

Enable completion in the current shell with one command:

```bash
eval "$(zg completion zsh)"       # zsh
eval "$(zg completion bash)"      # bash
zg completion fish | source       # fish
```

Completion includes deployment actions, model, assistant, and connector
configurations, participants, actions, and connector requirements.

## Examples and documentation

| | |
|---|---|
| [`examples/email_approval.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/email_approval.py) | the tutorial workflow: watch a mailbox, draft, approve, send |
| [`examples/diagnosis.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/diagnosis.py) | two reviewers loop until they agree, the paper's example |
| [`examples/pair_programming.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/pair_programming.py) | two coding assistants and a person: one answer decides whether both continue |
| [`examples/parallel.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/parallel.py) | a parallel region, and what each participant runs inside it |
| [`examples/work_pool/workflow.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/work_pool/workflow.py) | two workers claim and acknowledge jobs from a durable FIFO pool |
| [`examples/work_pool_cpl/workflow.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/work_pool_cpl/workflow.py) | a CPL approval guard gets its causal evidence through a claimed job |
| [`examples/human_approval.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/human_approval.py) | every shape a `@human` question can take |
| [`examples/calendar_approval/`](https://github.com/zippergen-io/zippergen/tree/main/examples/calendar_approval) | Review a meeting proposal and create it on Google Calendar after approval |
| [`examples/inbox_triage.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/inbox_triage.py) | Gmail in, Sheets out, deployed as a supervised service |
| [Your first ZipperGen workflow](https://github.com/zippergen-io/zippergen/blob/main/docs/first-workflow.pdf) | the tutorial |
| [Development and deployment guide](https://github.com/zippergen-io/zippergen/blob/main/docs/workflow-development-deployment-guide.pdf) | the long reference |
| [Architecture](https://github.com/zippergen-io/zippergen/blob/main/docs/architecture.md) | layers, module boundaries, and which constructs each theorem covers |
| [Durable storage](https://github.com/zippergen-io/zippergen/blob/main/docs/durable-storage.md) | current-state recovery, crash guarantees, identity, and history retention |
| [Workflow authoring skill](https://github.com/zippergen-io/zippergen/blob/main/.agents/skills/zippergen-workflows/SKILL.md) | what a coding agent follows, also printed by `zippergen skill` |
| [Changelog](https://github.com/zippergen-io/zippergen/blob/main/CHANGELOG.md) | release notes and upgrade-visible changes |
| [Contributing](https://github.com/zippergen-io/zippergen/blob/main/CONTRIBUTING.md) | working agreements and the gate a change has to pass |

## Formal foundation

The formal results establish two properties for well-formed workflows in the
supported model. Complete executions of the projected local programs match
executions of the global workflow, up to generated control messages. Every
finite projected execution can also be extended to a complete one. The second
property is the paper's precise sense of freedom from deadlock.

The proved constructs are message, action, skip, sequence, `if`, and `while`
(ISoLA paper), plus the parallel operator (EXPRESS/SOS paper). These constructs
make up the current language. A construct is not added to the grammar until a
result covers it.

The core projection theorems are
[machine-checked in Lean 4](https://github.com/zippergen-io/zippergen-lean/tree/main/isola).
The parallel extension is established separately in the EXPRESS/SOS paper.
The formal results are described in these papers:

- Bollig, Függer, and Nowak. [Provable Coordination for LLM Agents via
  Message Sequence Charts](https://arxiv.org/abs/2604.17612). Accepted at
  ISoLA 2026.
- Bollig. *Deadlock-Free Parallel Regions for Projected Workflows*. Accepted
  at EXPRESS/SOS 2026. Preprint forthcoming.
- Bollig. [Causal Past Logic for Runtime Verification of Distributed LLM Agent
  Workflows](https://arxiv.org/abs/2605.20923). Accepted at ICFEM 2026.

Causal Past Logic lets a condition read distributed state that is causally
visible at that point in the run. It works alongside what projection already
guarantees before the run starts.

## License

ZipperGen is released under the Apache License 2.0. See
[`LICENSE`](https://github.com/zippergen-io/zippergen/blob/main/LICENSE)
for the full terms.
