<p align="center">
  <img src="assets/zippergen-lockup-ink.svg" alt="ZipperGen" width="420">
</p>

<p align="center">
  <a href="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml"><img src="https://github.com/zippergen-io/zippergen/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://arxiv.org/abs/2604.17612"><img src="https://img.shields.io/badge/arXiv-2604.17612-b31b1b.svg" alt="arXiv"></a>
</p>

ZipperGen is a Python library for making several LLM agents work together.

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

That writes three files and stops. A manifest, an empty specification, and an
`AGENTS.md` file that tells a coding agent where ZipperGen's instructions are:

```
ZipperGen project: email-approval
  zippergen.toml     created
  specification.md   created
  AGENTS.md          created
```

Now say what you want. You can write the workflow yourself, or open a coding
agent in that directory and ask for it:

> Build a ZipperGen workflow that watches a mailbox directory, asks an LLM to
> draft a short reply to each new message, and asks me to approve it before it
> is sent. It should keep running and wait for the next message.

The agent reads `zippergen skill`, writes `specification.md` and
`workflow.py`, then checks its own work. What comes out is normal Python that
you can read:

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

**The Writer has no branch.** Nobody wrote those two programs by hand.

The Writer is told in every round whether to go on, because it has work to do
inside the loop. It is never told what the User approved, because it does
nothing either way. So that decision is simply not in its program. It cannot
wait for it, and it cannot block on it.

This is what projection means, and it is what the Lean proof is about.

## Deterministic testing

`--llm mock` gives every action the same placeholder answer, so a run always
takes the same path. To test another path, write the answers down in a file:

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

## Durable runs and deployment

There is one command for running a workflow. Add `--durable` to record it, so
you can continue it later:

```bash
zg run --durable --llm mock   # Ctrl-C part way through
zg run --resume               # carry on where it stopped
```

```
Run email_approval-20260808-135754-015850000
```

A plain `run` just runs once. A durable run writes down every step before it
takes it. If you stop it, it continues from there instead of starting again,
and you do not pay twice for a model call that already happened.

Preparing a deployment and starting one are two different steps. `--no-start`
writes the files and starts nothing. Without it, ZipperGen first checks every
model and connector for real, and stops if one of them does not answer:

```bash
zg deploy --name production --no-start   # first time: names it
zg start production
zg logs production
zg remove production          # the durable store is kept
```

A workflow can ask a person on Telegram, read Gmail, or write to Google
Sheets. Which chat, which spreadsheet, which search: that is project
configuration, and it goes in `zippergen.toml`. Credentials never go there:

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

The main result says that the projected local programs behave exactly like the
global workflow. Freedom from deadlock follows from this, for well-formed
workflows in the supported model.

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

Causal Past Logic lets a condition read distributed state that is causally
visible at that point in the run. It works alongside what projection already
guarantees before the run starts.

## License

ZipperGen is released under the Apache License 2.0. See [`LICENSE`](LICENSE)
for the full terms.
