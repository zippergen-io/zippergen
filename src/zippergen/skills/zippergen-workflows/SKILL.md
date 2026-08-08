---
name: zippergen-workflows
description: Create, extend, refactor, inspect, validate, compare, and prepare deployment-ready ZipperGen Python workflows from one or more natural-language prompts. Use when a coding agent needs to translate a coordination description into ZipperGen lifelines, messages, actions, owned decisions, parallel regions, human approvals, deployment declarations, or tests; explain an existing workflow at global, communication-only, selected-agent, or single-agent detail; or modify an existing workflow while proving the intended semantic change with ZipperGen CLI views and diffs.
---

# ZipperGen Workflows

Turn workflow intent into reviewable Python protocol code. Keep the global
workflow as the source of truth, make participant boundaries explicit, and use
ZipperGen's semantic tools to verify every generated or modified workflow.

There is no ZipperGen shell. A project is an ordinary directory, and every
operation below is a CLI command you run like any other. `zg` is the short
alias for `zippergen`; both are the same program.

Read [references/dsl-and-cli.md](references/dsl-and-cli.md) before authoring or
editing a workflow. Also inspect the repository's current `README.md`, nearby
workflow modules, and tests when available; prefer the installed version's API
over remembered syntax.

## Choose the operation

- For a new workflow, follow **Create from prompts**.
- For a change to an existing workflow, follow **Change an existing workflow**.
- For an explanation or review, follow **Inspect as code** without editing.
- Prepare or start a deployment only when the user asks for deployment work.

Commands resolve the workflow in this order:

1. an explicit `path.py:workflow` argument;
2. `workflow_entry` in `zippergen.toml`;
3. the project's only workflow, when there is exactly one;
4. otherwise they stop and ask.

So `zg validate` and `zg show --agent Writer` normally need no argument. A
project with one workflow needs no `workflow_entry` at all — do not add one
just to satisfy a command. Inference never writes to the manifest.

Set `workflow_entry` when the project has several workflows, or when you want
the choice recorded rather than inferred:

```toml
workflow_entry = "workflow.py:email_approval"
```

Treat portability separately from secrecy. What every machine shares — the
workflow entry point, named model and connector configurations, assignments,
and bindings — belongs in `zippergen.toml` and is committed. What belongs to
one machine — credentials, local endpoints, authorizations — lives in
`ZIPPERGEN_HOME` and is never committed. Resolve configuration with one rule
only: a site value wins when present, otherwise use the project value.

## Keep the specification current

You maintain `specification.md`: what the workflow is for, who takes part,
what each participant decides, and what must stay true. It is prose for the
user, not generated output, and no command reads it.

Update it in the same change as the code whenever participants, decisions,
external effects, or human control points change. When you are asked to work
on a workflow that has no specification, read the code and write one first —
it is how you and the user agree on intent before you edit.

## Model the intent first

Extract these facts from all supplied prompts before writing code:

1. Participants and their responsibilities.
2. Inputs, outputs, and the participant that initially or finally owns each.
3. Messages crossing participant boundaries.
4. Deterministic computation, LLM work, external effects, and human actions.
5. Decisions and loops, including the one participant that owns each guard.
6. Parallel work and the data needed to join it.
7. Logical connector requirements, external services, configuration, secrets,
   packages, setup, and source files.
8. Safety constraints, retry/idempotency expectations, and success examples.

Resolve contradictions between prompts explicitly. Ask only when a missing
choice would materially change the protocol or authorize an external effect.
Otherwise choose the smallest reasonable workflow and state the assumption.

## Create from prompts

1. Inspect analogous examples and public APIs in the target repository. Use
   `zg init` when the directory is not yet a project; it never overwrites an
   existing file.
2. Write a top-level Python module containing lifelines, variables, action
   declarations, one global `@workflow`, and deployment metadata when needed.
   Keep that global protocol readable: extract named `@fragment` helpers for
   coherent stages when leaving them inline would make the workflow difficult
   to understand, review, or maintain.
3. Keep external calls in `@effect`; keep deterministic transforms in `@pure`;
   use `@llm` only for model judgment or generation; use `@human` for explicit
   human control points.
4. Send values explicitly when ownership crosses a lifeline. Place every guard
   at the lifeline that actually knows and owns the decision.
5. Add focused tests that run with mock LLMs or fake services. Test protocol
   structure and safety behavior separately from live integrations.
6. Human delivery is inferred from `@human` action sites and routed with
   `zg connector assign`. Do not add a redundant connector requirement merely
   to reach a person through Telegram. For a non-human external service,
   declare an exact connector capability and test that it appears in workflow
   semantics. For Google Sheets, keep columns and a stable key in code, label
   each `@effect` with its connector and operation, and use the built-in JSON
   row helpers instead of embedding spreadsheet credentials or identifiers.
   For Gmail, declare the exact mailbox operations and keep the account,
   search query, and OAuth credential in the connector configuration. Declare
   connector access explicitly. Use `read-only` whenever the workflow does not
   modify the external service.
7. Run the validation and inspection gate below.

Do not invent a generic agent for every function. A lifeline represents a
sequential participant or trust/ownership boundary, not merely a code module.
Likewise, do not wait for literal duplication before using `@fragment`: a
single long protocol may be decomposed into meaningful coordination
subprograms. Keep participant transfers and owned control flow explicit, and
avoid tiny fragments that merely scatter a short protocol across files.

## Change an existing workflow

1. Read the module, its tests, its specification, and its deployment
   declaration before editing.
2. Save a semantic baseline to a unique temporary path outside the project:

   ```bash
   zg snapshot -o /tmp/<unique>-before.json
   ```

3. Translate the requested change into expected additions, removals, and
   preserved behavior. Prefer a focused edit over a rewrite.
4. Update the workflow code, action declarations, deployment metadata,
   specification, and tests together when the request crosses those
   boundaries.
5. Run the validation gate, then compare the baseline to the edited workflow:

   ```bash
   zg diff /tmp/<unique>-before.json
   ```

6. Confirm every reported change is intended and that every intended change is
   reported. Investigate unexpected implementation, message, control,
   participant, output, or deployment changes before handing off.
7. Report the semantic outcome, assumptions, tests, and any deliberately
   unchanged behavior. Do not present a source-line diff as proof of protocol
   equivalence.

The semantic diff compares protocol structure and action fingerprints. It does
not read inside a helper an action calls, and it cannot tell you a changed
constant was wrong. Where behavior matters, prove it with a test.

## Run the validation gate

Run these commands for every created or modified workflow:

```bash
zg validate
zg show --communications
zg show --detail full
```

Then inspect every changed or newly introduced participant using exact local
projection:

```bash
zg show --agent AgentName
```

Use `--format json` when programmatic checking helps. Run focused tests first,
then the repository's broader suite and static checks in proportion to risk.
When a specification names logical connectors, confirm every exact name in the
full view or semantic JSON before reporting success.

Treat load, projection, rendering, test, or type-check failures as blockers.

Respect project boundaries when the application root contains a nested
framework checkout. If `zippergen.toml` declares `framework_directory`, use
that nested project's environment for ZipperGen commands, run application
tests by their explicit path, and do not let a bare recursive pytest invocation
mistake the framework's own tests for the application suite. Run the framework
suite separately only when framework source changed. Do not use transient
dependency flags such as `--with pytest` during a restricted assistant run;
use the project's declared, initially synchronized development dependencies
and prefer the package runner's offline mode during verification.

## Exercise both sides of a decision

`--llm mock` returns one placeholder for every action, so a workflow with a
branch runs only one path under it. That is not enough to claim a decision
works. Two different things drive the two kinds of answer.

**Model answers** come from a scripted file:

```bash
zg run --llm scripted:answers.json --input message=hello
```

Each key is `Participant.action`, or a bare `action` name as a fallback. A bare
value repeats for every call; a list is a finite sequence, and running past its
end is an error rather than a silent repeat:

```json
{"Writer.draft_reply": {"draft": "How about Thursday?"}}
```

**Human answers** are not scripted. A `@human` action asks a person on the
terminal, or reaches them through an assigned connector. To drive one without a
person, feed standard input:

```bash
printf 'y\n' | zg run --llm scripted:answers.json --input message=hello
printf 'n\n' | zg run --llm scripted:answers.json --input message=hello
```

Run the workflow once per branch and assert the outcome. Prefer this over a
live model whenever you are testing the protocol rather than the prompt.

## Inspect as code

Choose the narrowest view that answers the question:

- Use the default `show` view for the global protocol.
- Add `--communications` for only messages and control flow.
- Add `--agent NAME` for the exact formally projected local program.
- Add `--agents A,B` for a selected group with hidden peers shown as explicit
  external boundaries.
- Add `--detail actions` for action signatures.
- Add `--detail full` for prompts, action bodies, and deployment requirements.

Quote or summarize the rendered code rather than inventing a diagram. Preserve
the distinction between a selected-agent focus view and exact single-agent
projection.

A participant that takes no part in a decision has no branch for it in its
projection. That holds even when it works inside a loop around that decision
and is told each round whether to continue. When you explain a workflow, say
so. It is the property the projection gives you, and one command checks it.

## Prepare deployment

Keep deployment declarations data-only and colocated with the workflow module.
Declare required fields, secrets, packages, setup steps, and bundled files; do
not embed credentials or copy secret values into ordinary profiles or tests.

Connectors are configured once per machine and routed per workflow:

```bash
zg connector configure telegram --bind approvals
zg connector assign User approvals
```

`configure` stores the credential in `ZIPPERGEN_HOME`; the routing it produces
is committed with the project and holds no secret. Deployment reads both and
refuses to start when a required connector is unbound, unauthorized, or
assigned to a participant that has no `@human` action.

Before starting or restarting anything, run:

```bash
zg validate
zg show --detail full
```

Use the guided path when deployment is explicitly authorized:

```bash
zg deploy --name production
```

Afterward use the deployment name with `doctor`, `status`, `logs`, `restart`,
`stop`, `compact`, and `configure --restart`.

Never assume permission to send live messages, modify production data,
complete OAuth, or restart a live service merely because the user requested
workflow code.

## Tell the user what to do next

End every piece of work by saying what state the project is in and what the
obvious next command is — validate, run, assign a connector, deploy. The user
moves between you and the shell, so a command they can paste is worth more than
a description of it.
