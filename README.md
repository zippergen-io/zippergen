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

---

## How you work with it

A ZipperGen project is an ordinary directory: a workflow in Python, a
specification in prose, and a small TOML file. You develop it however you
develop anything else, in an editor or by talking to a coding agent such as
Claude Code or Codex.

There is no separate ZipperGen development environment to learn. ZipperGen
ships a **skill** that
teaches a coding agent how to work on a project, and a **CLI** that you and the
agent both use.

```
      you, or Claude Code / Codex
                  │
            zippergen skill
                  │
             zippergen CLI
      init · validate · run · deploy
                  │
              ZipperGen
   protocol · projection · runtime
```

## Install

```bash
pip install zippergen
```

ZipperGen has no runtime dependencies and needs Python 3.11 or newer. It
installs two commands, `zippergen` and the short form `zg`.

> The published PyPI alpha predates the current CLI. Until the next release,
> work from a clone:
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

That writes three files and stops: a manifest, an empty specification, and an
`AGENTS.md` that points a coding agent at ZipperGen's instructions:

```
ZipperGen project: email-approval
  zippergen.toml     created
  specification.md   created
  AGENTS.md          created
```

Now describe what you want. Either write the workflow yourself, or open a
coding agent in that directory and say so:

> Build a ZipperGen workflow that takes an email message, asks an LLM to draft
> a short reply, and sends the draft to me for approval before it is sent.

The agent reads `zippergen skill`, writes `specification.md` and `workflow.py`,
and checks its work. What comes out is ordinary, readable Python:

```python
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
echo "Could we move our meeting to Thursday?" > mailbox/01.txt
zg run --llm mock
```

The project already records which workflow it contains, so you rarely name it.

```
Message (str): Could we move our meeting to Thursday afternoon?

Proposed reply:

Thursday afternoon works for me. How about 3pm?

Send this reply? [y/n]: y
{"result": "Sent: Thursday afternoon works for me. How about 3pm?"}
```

A full walkthrough, including approval over Telegram and a real deployment, is
in [**Your first ZipperGen workflow**](docs/first-workflow.pdf).

## What the protocol buys you

The workflow above has one decision, and the `User` makes it. Ask ZipperGen
what each participant actually runs:

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

**The Writer has no branch.** Nobody wrote either line. The Writer is told
each round whether to continue, because it has work inside the loop. It is
never told what was approved, because it does nothing in either branch. The
decision is erased from its program, and it cannot wait on it, disagree
with it, or deadlock against it. That is the projection, and it is the same construction the
correctness proof is about.

## Deterministic testing

`--llm mock` answers every action with a placeholder, so a workflow driven by
it takes one path and no other. To reach a specific branch, script the
responses:

```json
{
  "draft_reply": {"draft": "Thursday afternoon works for me. How about 3pm?"}
}
```

```bash
zg run --llm scripted:replies.json
```

Responses are consumed in order per action. A bare object answers every call
the same way. A list is a finite sequence, and running past its end is an
error rather than a silent repeat, so a change that calls an action more often
than expected fails instead of passing quietly.

## Durable runs and deployment

There is one verb for running a workflow. `--durable` records the run so it can
be resumed:

```bash
zg run --durable --llm mock   # Ctrl-C part way through
zg run --resume               # carry on where it stopped
```

```
Run email_approval-20260808-135754-015850000
```

A plain `run` executes once. A durable run records every step before taking it,
so an interrupted one continues rather than starting over, and a model call
already made is not paid for twice.

Deployment is separate from preparation. `--no-start` writes the bundle,
environment and service files without starting anything. Without it, every
model and connector is probed live and a failure stops the deployment:

```bash
zg deploy --name production --no-start   # first time: names it
zg start production
zg logs production
zg remove production          # the durable store is kept
```

Human approvals can go to Telegram, and workflows can read Gmail or write
Google Sheets. Which chat, which spreadsheet, which mailbox query: those are
project configuration and live in `zippergen.toml`. Credentials never do:

```bash
zg connector configure telegram approvals --chat-id 12345678
zg connector assign User approvals        # who gets asked, and where
zg connector authorize google --scopes gmail.readonly,spreadsheets
```

## The CLI

```
init        create a project
skill       print the coding-agent skill
validate    load, project, and check a workflow
show        render the protocol, or one participant's local program
run         run a workflow; --durable records it, --resume continues one
deploy      prepare and start a deployment
start · stop · restart · logs · status · trace · tasks · approve
remove · compact
connector   configure Telegram, Gmail or Sheets; authorize Google
```

Run `zippergen <command> --help` for any of them.

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
