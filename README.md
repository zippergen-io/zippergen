<p align="center">
  <img src="https://raw.githubusercontent.com/zippergen-io/zippergen/main/assets/zippergen-lockup-ink.svg" alt="ZipperGen" width="420">
</p>

<p align="center">
  <a href="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml"><img src="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://arxiv.org/abs/2604.17612"><img src="https://img.shields.io/badge/arXiv-2604.17612-b31b1b.svg" alt="arXiv"></a>
</p>

ZipperGen is a Python library for coordinating LLM agents, humans, and
services.

You write one protocol. It says who sends what to whom, who calls a model, and
who owns each decision. ZipperGen works out the program each participant runs,
and runs them. For well-formed workflows covered by ZipperGen's formal model,
those programs cannot deadlock. This is [proved formally](#formal-foundation).

---

## How you work with it

A ZipperGen project is an ordinary directory. It contains a Python workflow, a
plain-text specification, and a small TOML file. You can edit it directly or
work with a coding agent such as Claude Code or Codex.

No special editor or hosted environment is required. `zippergen skill` gives a
coding agent its project instructions. You and the agent use the same CLI.

```
you ────────────────────────────┐
                                │
Claude Code / Codex ─ skill ────┤
                                │
                          zippergen CLI
                    init · validate · run · deploy
                                │
                            ZipperGen
                 protocol · projection · runtime
```

## Install

Not on PyPI yet, so install it from this repository:

```bash
uv tool install "git+https://github.com/zippergen-io/zippergen.git"
```

Or with pipx, or into a virtual environment:

```bash
pipx install "git+https://github.com/zippergen-io/zippergen.git"
python3 -m venv .venv && .venv/bin/pip install "git+https://github.com/zippergen-io/zippergen.git"
```

The installation resolves to the repository's current commit. To update,
rerun the same command with `--refresh`, since Git sources are cached by
resolved commit:

```bash
uv tool install --force --refresh "git+https://github.com/zippergen-io/zippergen.git"
```

Gmail and Google Sheets need one extra:

```bash
uv tool install "zippergen[google] @ git+https://github.com/zippergen-io/zippergen.git"
```

ZipperGen needs Python 3.11 or newer. It has no other dependencies. It
installs two commands: `zippergen`, and `zg` for short.

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
draft = Var("draft", str)
approved = Var("approved", bool)
handled = Var("handled", int, default=0)


@workflow
def email_approval() -> int:
    Mailbox: message = next_unread_message()
    while message @ Mailbox:
        Mailbox(message) >> Writer(message)
        Writer: draft = draft_reply(message)
        Writer(draft) >> Mailbox(draft)
        Mailbox: approved = approve_reply(draft)
        if approved @ Mailbox:
            Mailbox: handled = send_reply(draft, handled)
        else:
            Mailbox: handled = discard(handled)
        Mailbox: message = next_unread_message()
    return handled @ Mailbox
```

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
    message = next_unread_message()
    while message:
        send_decision('Writer', True)
        send('Writer', message)
        draft = recv('Writer')
        approved = approve_reply(draft)
        if approved:
            handled = send_reply(draft, handled)
        else:
            handled = discard(handled)
        message = next_unread_message()
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

**The Writer has no approval branch.** Nobody wrote those two programs by
hand.

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

A workflow can ask a person on Telegram, read Gmail, or write to Google
Sheets. The destination belongs in project configuration. Credentials stay in
private state on the machine that runs ZipperGen. For Telegram:

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

Commands for durable state begin with its owner. Use `zg run tasks` for a
durable run and `zg deploy tasks` for a deployment. The
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
| [`examples/human_approval.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/human_approval.py) | every shape a `@human` question can take |
| [`examples/inbox_triage.py`](https://github.com/zippergen-io/zippergen/blob/main/examples/inbox_triage.py) | Gmail in, Sheets out, deployed as a supervised service |
| [Your first ZipperGen workflow](https://github.com/zippergen-io/zippergen/blob/main/docs/first-workflow.pdf) | the tutorial |
| [Development and deployment guide](https://github.com/zippergen-io/zippergen/blob/main/docs/workflow-development-deployment-guide.pdf) | the long reference |
| [Architecture](https://github.com/zippergen-io/zippergen/blob/main/docs/architecture.md) | layers, module boundaries, and which constructs each theorem covers |
| [Durable storage](https://github.com/zippergen-io/zippergen/blob/main/docs/durable-storage.md) | current-state recovery, crash guarantees, identity, and history retention |
| [Workflow authoring skill](https://github.com/zippergen-io/zippergen/blob/main/.agents/skills/zippergen-workflows/SKILL.md) | what a coding agent follows, also printed by `zippergen skill` |
| [Changelog](https://github.com/zippergen-io/zippergen/blob/main/CHANGELOG.md) | release notes and upgrade-visible changes |

## Formal foundation

The main result says that the projected local programs behave exactly like the
global workflow. Freedom from deadlock follows from this, for well-formed
workflows in the supported model.

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
