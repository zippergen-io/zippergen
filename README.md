<p align="center">
  <img src="assets/zippergen-lockup-ink.svg" alt="ZipperGen" width="420">
</p>

<p align="center">
  <a href="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml"><img src="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://arxiv.org/abs/2604.17612"><img src="https://img.shields.io/badge/arXiv-2604.17612-b31b1b.svg" alt="arXiv"></a>
</p>

ZipperGen is a Python library for coordinating LLM agents, humans, and
services.

You write one protocol. It says who sends what to whom, who calls a model, and
who owns each decision. ZipperGen works out the program each participant runs,
and runs them.

For well-formed workflows, those programs cannot deadlock. This is proved, not
just tested. The proof is checked by machine, in Lean 4.

---

## How you work with it

A ZipperGen project is a normal directory. It holds a workflow in Python, a
specification in plain text, and a small TOML file. You work on it like any
other code: in an editor, or by talking to a coding agent such as Claude Code
or Codex.

There is no special ZipperGen environment to learn. ZipperGen gives you two
things. A **skill**, which tells a coding agent how to work on a project. And a
**CLI**, which you and the agent both use.

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

The skill is for the agent. The CLI is shared: you and the agent run the same
commands, and you never have to go through the agent to use them.

## Install

```bash
pip install zippergen
```

ZipperGen needs Python 3.11 or newer. It has no other dependencies. It
installs two commands: `zippergen`, and `zg` for short.

> The version on PyPI is older than the CLI shown here. Until the next
> release, install from a clone:
>
> ```bash
> git clone https://github.com/zippergen-io/zippergen.git
> cd zippergen && pip install -e .
> ```

## Quick start

```bash
mkdir email-approval && cd email-approval
zippergen init
```

That writes four files and stops. A manifest, an empty specification, shared
agent instructions, and a small pointer that makes Claude Code read them:

```
ZipperGen project: email-approval
  zippergen.toml     created
  specification.md   created
  AGENTS.md          created
  CLAUDE.md           created
```

Now say what you want. You can write the workflow yourself, or open a coding
agent in that directory and ask for it:

> Build a ZipperGen workflow that watches a mailbox directory, asks an LLM to
> draft a short reply to each new message, and asks me to approve it before it
> is sent. It should keep running and wait for the next message.
>
> Use plain `.txt` files directly inside `mailbox/`. A file containing only
> `Can we meet on Thursday` is a complete message. Do not require `From` or
> `Subject` headers, and do not add an `inbox/` subdirectory. Use `mailbox/` as
> the local default rather than a required workflow input. The exact command
> `zg run --llm mock` must reach the approval without asking a setup question.
> Before reporting success, test that exact file, layout, and command.

The agent runs `zippergen skill`, reads the instructions it prints, writes
`specification.md` and `workflow.py`, then checks its own work. What comes out
is normal Python that you can read:

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
REQUEST · Mailbox

Proposed reply:

[draft_reply:draft]

Send this reply? [y/n]: y
✓ Mailbox · reply sent
```

The reply is a placeholder, because `mock` does not call a model. Use
`--llm openai:gpt-4o-mini`, with a key in your environment, to get a real one.

Then it waits for the next message. Press Ctrl-C to stop it.

You did not have to name the workflow in those commands. The project already
knows which one it holds.

The tutorial goes through all of this step by step, including approval on your
phone and a real deployment:
[**Your first ZipperGen workflow**](docs/first-workflow.pdf).

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

The Writer is told in every round whether to go on, because it has work to do
inside the loop. It is never told whether approval was granted at Mailbox,
because it does nothing either way. So that decision is simply not in its
program. It cannot wait for it, and it cannot block on it.

This is what projection means, and it is what the Lean proof is about.

## One configuration pattern

Provider connections, models, coding assistants, and connectors use one small
grammar:

```text
zg provider configure NAME KIND
zg TYPE configure NAME ...
zg TYPE assign TARGET NAME
zg TYPE check [NAME]
zg TYPE remove NAME
```

A connector for an external service uses `bind REQUIREMENT NAME` instead of
`assign`. A provider connection is the named access path to one external
provider: it owns the private credential and any machine-specific endpoint.
A model configuration then chooses a model through one connection; a connector
configuration chooses a chat, mailbox, spreadsheet, or other destination
through one connection. In the examples below, `approval-bot` is a Telegram
provider connection and `approval-chat` is a connector configuration using it.
`zg model`, `zg assistant`, `zg provider`, `zg connector`, and `zg config` show
the result.

When you work in a terminal, you may leave out required values. ZipperGen asks
for them and shows available targets and saved configurations. For example,
`zg model configure`, `zg assistant configure`, and `zg connector configure`
are all guided. Reusing a name updates that configuration and presents its
current values as defaults. Provider connections, models, and connector
destinations are asked for separately. Scripts and coding agents should pass
those values explicitly.

For an `@assistant` action, choose Codex or Claude with a named configuration:

```bash
zg assistant configure coding-agent codex
zg assistant assign Maintainer coding-agent
zg assistant check
```

Assign `Maintainer.action_name` when only one action needs a different
backend. `zg config` shows the effective backend together with the action's
`access`, `external_tools`, and `shell` policy. Those permissions remain part
of the reviewed `@assistant` declaration. Codex and Claude continue to manage
their own login and credentials.

## Deterministic testing

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
computer. It is not written to `zippergen.toml`. You may instead set the
normal `OPENAI_API_KEY` environment variable. Several model configurations can
reuse `openai-main`; define another OpenAI connection when they need a
different key.

For local Ollama configurations, `Idle release` means how long ZipperGen keeps
the model loaded after the last active call. `0` unloads it after every call;
an unset value leaves the provider policy unchanged.

The commands write the portable routing to `zippergen.toml`:

```toml
[providers.connections."openai-main"]
kind = "openai"

[models.configurations."writer"]
connection = "openai-main"
model = "gpt-4o-mini"

[models.assignments.lifelines]
Writer = "writer"
```

`zg run`, `zg run --durable`, and `zg deploy` all use that assignment.
`--llm mock` temporarily replaces all project assignments. Use
`--llm-for Writer=SPEC` only for a narrower one-command override.

`--llm mock` gives every action the same placeholder answer, so a run always
takes the same path. To give model actions fixed answers instead, write them
down in a file:

```json
{
  "draft_reply": {"draft": "Thursday afternoon works for me. How about 3pm?"}
}
```

```bash
zg run --llm scripted:replies.json
```

Answers are used in order, per action. A single object answers every call the
same way. A list is used once through. If the workflow asks for one more answer
than the list holds, the run fails instead of quietly repeating the last one.
So a change that calls an action more often than you expected shows up as an
error.

This covers model actions only. A `@human` action still asks a person, so to
drive an approval one way or the other you answer it, or pipe the answer in:

```bash
printf 'n\n' | zg run --llm scripted:replies.json
```

## Durable runs and deployment

There is one command for running a workflow. Add `--durable` to record it, so
you can continue it later:

```bash
zg run --durable --llm mock   # Ctrl-C part way through
zg run inspect --agent Writer # see where each participant is waiting
zg run --resume               # carry on where it stopped
```

```
Run email_approval-20260808-135754-015850000
```

A plain `run` just runs once and has no deadline unless `--timeout SECONDS` is
given. Both forms honor the project's connector routing; a disposable run may
use temporary private coordination while Telegram is active, but leaves no
resumable run behind. A durable run records coordination state and
completed external-action results. After an ordinary interruption it continues
instead of starting again. A crash in the narrow interval after an external
effect succeeds but before its result is recorded can repeat that effect, so
irreversible connectors should use idempotency keys.

`zg run status` shows the currently selected durable run. Starting another
`zg run --durable` selects a new run and discards any older selected
development run. To abandon the selected run explicitly, stop its foreground
process with Ctrl-C and use `zg run reset`; ZipperGen deletes its record and
SQLite state and clears the selection. Add `--archive` only when a recoverable
private copy is wanted. Afterwards `zg run status` reports no current run, and
the next execution begins explicitly with `zg run --durable`. If the project's
deployment is already running, stop it before starting a foreground run.
ZipperGen permits one active execution per project: one disposable or durable
foreground run, or the deployment. Inspection, trace, status and task commands
remain available while that execution runs.

For a live view, keep the run open in one terminal and use another terminal:

```bash
zg run inspect --watch --agent Writer
```

The view updates in place once per second. Ctrl-C closes only the view. It does
not interrupt the run. For the project's deployment,
`zg deploy inspect --watch` reads the same position information without
changing the running service.

The ordinary deployment command prepares, checks, and starts the service. It
stops before starting when a model, assistant CLI, or connector is not ready:

```bash
zg deploy
zg deploy status
zg deploy logs
zg deploy remove              # the durable store is kept
zg deploy reset --yes         # archive durable state and start fresh
```

Use `zg deploy --no-start` only when you deliberately want to prepare and
review a stopped deployment before a later `zg deploy start`. It is not a
required preliminary step. After a code change, stop a running deployment
before running `zg deploy` again, so its bundle and managed environment are
not replaced underneath it. `restart` reuses the existing bundle, while
`reset` discards durable execution state and `remove` uninstalls the service:

```bash
zg deploy stop
zg deploy       # rebuild and start the updated deployment
```

A workflow can ask a person on Telegram, read Gmail, or write to Google
Sheets. Which chat, which spreadsheet, which Gmail query: that is project
configuration, and it goes in `zippergen.toml`. Credentials never go there:

```bash
zg provider configure approval-bot telegram
zg provider set-credential approval-bot       # hidden bot-token prompt
zg connector configure approval-chat approval-bot  # Telegram is inferred
zg connector assign Mailbox approval-chat      # whose human action is routed

zg provider configure google-work google
zg provider authorize google-work --scopes gmail.readonly,spreadsheets
zg provider accept google-work                 # paste the private result
zg check                                       # routing and live providers
```

Use `zg config` at any time to see provider connections; effective model,
assistant, and connector configurations; assignments and bindings; private-state
location, and which local credentials or tools are available. It does not
contact providers and never prints credential values. `zg validate` is also
offline; `zg check` is the project-wide readiness operation and may send a
small model request. `zg config --json` and `zg check --json` provide the same
views for CI and coding agents. Family checks such as `zg model check` narrow
the diagnosis; without a name they check every saved configuration.

`zg check` exits zero once it has run, because a report is not a failure. Add
`--strict` when a script should stop on anything that is not ready. `zg deploy
check` works the same way. `zg validate` is the exception and exits non-zero on
a broken workflow, because that is a real error rather than news.

## The CLI

The whole public surface fits in one tree:

```text
zg
├── init · skill · validate · show · snapshot · diff · check
├── config
├── workflow
│   └── select
├── provider
│   └── configure · set-credential · check · remove · authorize · accept
├── model
│   └── configure · assign · unassign · check · remove
├── assistant
│   └── configure · assign · unassign · check · remove
├── connector
│   └── configure · assign · unassign · bind · unbind · check · remove
├── run
│   └── status · reset · inspect · trace · tasks · approve
├── deploy
│   └── list · prune · start · stop · restart · status · logs · check
│       · inspect · trace · tasks · approve · compact · reset · remove
└── completion
```

`zg --help` renders this tree from the real command parser, so it cannot drift
from the implementation. Run `zg <command> --help` for arguments and examples.

Stores are not part of the ordinary command surface. Each durable run owns its
managed store; the project deployment owns another. Their commands always
begin with that owner, for example `zg run tasks` and `zg deploy tasks`. To
discard a deployment's durable history,
`zg deploy reset --yes` stops it, archives its SQLite files under
`$ZIPPERGEN_HOME/trash/deployment-stores/`, creates an empty store, and starts
the service again if it was running. The archive is never silently deleted.
`zg deploy compact` prunes optional inspection history and rotates logs; stop
the deployment first, because the combined command refuses before changing
either resource while its service is running.
`zg deploy list` also works outside a project and shows every deployment on
the computer. If a project directory was deleted or reinitialized, use
`zg deploy prune`; it unregisters orphaned services and archives their durable
stores and logs rather than deleting them.

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
| [`examples/email_approval.py`](examples/email_approval.py) | the tutorial workflow: watch a mailbox, draft, approve, send |
| [`examples/diagnosis.py`](examples/diagnosis.py) | two reviewers loop until they agree, the paper's example |
| [`examples/call_intake.py`](examples/call_intake.py) | Gmail in, Sheets out, deployed as a service |
| [Your first ZipperGen workflow](docs/first-workflow.pdf) | the tutorial |
| [Development and deployment guide](docs/workflow-development-deployment-guide.pdf) | the long reference |
| [Durable storage](docs/durable-storage.md) | current-state recovery, crash guarantees, identity, and history retention |
| [Workflow authoring skill](.agents/skills/zippergen-workflows/SKILL.md) | what a coding agent follows, also printed by `zippergen skill` |

## Formal foundation

The main result says that the projected local programs behave exactly like the
global workflow. Freedom from deadlock follows from this, for well-formed
workflows in the supported model.

The main theorems are
[machine-checked in Lean 4](https://github.com/zippergen-io/zippergen-lean/tree/main/isola).
The formal results are described in these papers:

- Bollig, Függer, and Nowak. [Provable Coordination for LLM Agents via
  Message Sequence Charts](https://arxiv.org/abs/2604.17612). Accepted at
  ISoLA 2026.
- Bollig. *Deadlock-Free Parallel Regions for Projected Workflows*. Accepted
  at EXPRESS/SOS 2026. Preprint forthcoming.
- Bollig. [Causal Past Logic for Runtime Verification of Distributed LLM Agent
  Workflows](https://arxiv.org/abs/2605.20923). Under submission.

Causal Past Logic lets a condition read distributed state that is causally
visible at that point in the run. It works alongside what projection already
guarantees before the run starts.

## License

ZipperGen is released under the Apache License 2.0. See [`LICENSE`](LICENSE)
for the full terms.
