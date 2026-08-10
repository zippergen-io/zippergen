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
    User: message = next_unread_message()
    while message @ User:
        User(message) >> Writer(message)
        Writer: draft = draft_reply(message)
        Writer(draft) >> User(draft)
        User: approved = approve_reply(draft)
        if approved @ User:
            User: handled = send_reply(draft, handled)
        else:
            User: handled = discard(handled)
        User: message = next_unread_message()
    return handled @ User
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
Proposed reply:

[draft_reply:draft]

Send this reply? [y/n]: y
    sent: [draft_reply:draft]
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

The workflow above has one decision, and the `User` makes it. You can ask
ZipperGen what each participant really runs:

```bash
zg show --agent User
```

```python
@role('User')
def email_approval__User() -> int:
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
    while recv_decision('User'):
        message = recv('User')
        draft = draft_reply(message)
        send('User', draft)
```

**The Writer has no approval branch.** Nobody wrote those two programs by
hand.

The Writer is told in every round whether to go on, because it has work to do
inside the loop. It is never told what the User approved, because it does
nothing either way. So that decision is simply not in its program. It cannot
wait for it, and it cannot block on it.

This is what projection means, and it is what the Lean proof is about.

## One configuration pattern

Models, coding assistants, and connectors use the same small grammar:

```text
zg TYPE configure NAME PROVIDER_OR_SPEC
zg TYPE assign TARGET NAME
zg TYPE check [NAME]
zg TYPE remove NAME
```

A connector for an external service uses `bind REQUIREMENT NAME` instead of
`assign`. The provider or backend is an attribute of the named configuration,
not another object to manage. In the examples below, `approval-chat` is a name
chosen by the user and `telegram` is the provider. `zg model`, `zg assistant`,
`zg connector`, and `zg config` show the result.

When you work in a terminal, you may leave out required values. ZipperGen asks
for them and shows available targets and saved configurations. For example,
`zg model configure`, `zg assistant configure`, and `zg connector configure`
are all guided. The model command asks for the provider and model separately.
Scripts and coding agents should pass the compact `PROVIDER:MODEL` value
explicitly.

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
zg model configure writer openai:gpt-4o-mini
zg model assign Writer writer
zg model
```

If `OPENAI_API_KEY` is not already available, the first command offers a
hidden prompt. The key is saved in private storage on this computer. It is not
written to `zippergen.toml`. You may instead set the normal environment
variable before running the command.

The commands write the portable routing to `zippergen.toml`:

```toml
[models.configurations."writer"]
provider = "openai"
model = "gpt-4o-mini"
spec = "openai:gpt-4o-mini"

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
zg inspect --agent Writer     # see where each participant is waiting
zg run --resume               # carry on where it stopped
```

```
Run email_approval-20260808-135754-015850000
```

A plain `run` just runs once. A durable run records coordination state and
completed external-action results. After an ordinary interruption it continues
instead of starting again. A crash in the narrow interval after an external
effect succeeds but before its result is recorded can repeat that effect, so
irreversible connectors should use idempotency keys.

For a live view, keep the run open in one terminal and use another terminal:

```bash
zg inspect --watch --agent Writer
```

The view updates in place once per second. Ctrl-C closes only the view. It does
not interrupt the run. For a deployment, `zg inspect production --watch` reads
the same position information without changing the running service.

Preparing a deployment and starting one are two different steps. `--no-start`
writes the files and starts nothing. Without it, ZipperGen first checks every
model, assistant CLI, and connector, and stops if a required dependency is not
ready:

```bash
zg deploy create --name production --no-start   # first time: names it
zg deploy start production
zg deploy logs production
zg deploy remove production          # the durable store is kept
```

A workflow can ask a person on Telegram, read Gmail, or write to Google
Sheets. Which chat, which spreadsheet, which Gmail query: that is project
configuration, and it goes in `zippergen.toml`. Credentials never go there:

```bash
zg connector configure approval-chat telegram  # prompts for chat id and hidden token
zg connector assign User approval-chat        # who gets asked, and where
zg connector authorize google --scopes gmail.readonly,spreadsheets
zg config check                         # check the whole project
zg config check --live                  # contact each configured provider
```

Use `zg config` at any time to see the effective model, assistant, and
connector configurations, assignments, bindings, and missing site facts. It
never prints credential values. `zg config --json` provides the same view for
CI and coding agents.

## The CLI

```
init        create a project
skill       print the coding-agent skill
validate    load, project, and check a workflow
show        render the protocol, or one participant's local program
config      show or check all project configuration
model       configure models and assign participants or actions
assistant   configure Codex or Claude and assign participants or actions
connector   configure connectors, assignments, bindings, and authorization
run         run a workflow; --durable records it, --resume continues one
deploy      prepare and start a deployment
start · stop · restart · logs · status · trace · tasks · approve
remove · compact
completion  print shell completion for zsh, bash, or fish
```

Run `zippergen <command> --help` for any of them.

Enable completion in the current shell with one command:

```bash
eval "$(zg completion zsh)"       # zsh
eval "$(zg completion bash)"      # bash
zg completion fish | source       # fish
```

Completion includes current deployment names, model, assistant, and connector
configurations, participants, actions, and connector requirements.

## Examples and documentation

| | |
|---|---|
| [`examples/email_approval.py`](examples/email_approval.py) | the tutorial workflow: watch a mailbox, draft, approve, send |
| [`examples/diagnosis.py`](examples/diagnosis.py) | two reviewers loop until they agree, the paper's example |
| [`examples/call_intake.py`](examples/call_intake.py) | Gmail in, Sheets out, deployed as a service |
| [Your first ZipperGen workflow](docs/first-workflow.pdf) | the tutorial |
| [Development and deployment guide](docs/workflow-development-deployment-guide.pdf) | the long reference |
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
