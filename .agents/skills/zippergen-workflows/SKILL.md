---
name: zippergen-workflows
description: Create, extend, refactor, inspect, validate, compare, and prepare deployment-ready ZipperGen Python workflows from one or more natural-language prompts. Use when Codex needs to translate a coordination description into ZipperGen lifelines, messages, actions, owned decisions, parallel regions, human approvals, deployment declarations, or tests; explain an existing workflow at global, communication-only, selected-agent, or single-agent detail; or modify an existing workflow while proving the intended semantic change with ZipperGen CLI views and diffs.
---

# ZipperGen Workflows

Turn workflow intent into reviewable Python protocol code. Keep the global
workflow as the source of truth, make participant boundaries explicit, and use
ZipperGen's semantic tools to verify every generated or modified workflow.

Invoke this repository skill explicitly as `$zippergen-workflows` when useful.

Read [references/dsl-and-cli.md](references/dsl-and-cli.md) before authoring or
editing a workflow. Also inspect the repository's current `README.md`, nearby
workflow modules, and tests when available; prefer the installed version's API
over remembered syntax.

## Choose the operation

- For a new workflow, follow **Create from prompts**.
- For a change to an existing workflow, follow **Refine from prompts**.
- For an explanation or review, follow **Inspect as code** without editing.
- Prepare or start a deployment only when the user asks for deployment work.

## Model the intent first

Extract these facts from all supplied prompts before writing code:

1. Participants and their responsibilities.
2. Inputs, outputs, and the participant that initially or finally owns each.
3. Messages crossing participant boundaries.
4. Deterministic computation, LLM work, external effects, and human actions.
5. Decisions and loops, including the one participant that owns each guard.
6. Parallel work and the data needed to join it.
7. External services, configuration, secrets, packages, setup, and source files.
8. Safety constraints, retry/idempotency expectations, and acceptance examples.

Resolve contradictions between prompts explicitly. Ask only when a missing
choice would materially change the protocol or authorize an external effect.
Otherwise choose the smallest reasonable workflow and state the assumption.

## Create from prompts

1. Inspect analogous examples and public APIs in the target repository.
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
6. Run the validation and inspection gate below.

Do not invent a generic agent for every function. A lifeline represents a
sequential participant or trust/ownership boundary, not merely a code module.
Likewise, do not wait for literal duplication before using `@fragment`: a
single long protocol may be decomposed into meaningful coordination
subprograms. Keep participant transfers and owned control flow explicit, and
avoid tiny fragments that merely scatter a short protocol across files.

## Refine from prompts

1. Identify the exact workflow spec (`path.py:workflow` or
   `module:workflow`) and read its module, tests, and deployment declaration.
2. Before editing, write a semantic baseline to a unique temporary JSON path:

   ```bash
   uv run zippergen snapshot path/to/workflow.py:workflow -o /tmp/<unique>-before.json
   ```

3. Translate the requested change into expected additions, removals, and
   preserved behavior. Prefer a focused edit over a rewrite.
4. Update the workflow code, action declarations, deployment metadata, and
   tests together when the request crosses those boundaries.
5. Run the validation gate, then compare the saved baseline to the edited
   workflow:

   ```bash
   uv run zippergen diff /tmp/<unique>-before.json path/to/workflow.py:workflow
   ```

6. Inspect the semantic diff. Confirm every reported change is intended and
   that expected changes are present. Investigate unexpected implementation,
   message, control, participant, output, or deployment changes before handing
   off the result.
7. Report the semantic outcome, assumptions, tests, and any intentionally
   unchanged behavior. Do not present a source-line diff as proof of protocol
   equivalence.

Keep the temporary snapshot outside the project and remove it only by an exact,
validated path when cleanup is useful.

## Run the validation gate

Run these commands for every created or modified workflow:

```bash
uv run zippergen validate path/to/workflow.py:workflow
uv run zippergen show path/to/workflow.py:workflow --communications
uv run zippergen show path/to/workflow.py:workflow --detail full
```

Then inspect every changed or newly introduced participant using exact local
projection:

```bash
uv run zippergen show path/to/workflow.py:workflow --agent AgentName
```

Use `--format json` when programmatic checking helps. Run focused tests first,
then the repository's broader suite and static checks in proportion to risk.
Treat load, projection, rendering, test, or type-check failures as blockers.

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

## Prepare deployment

Keep deployment declarations data-only and colocated with the workflow module.
Declare required fields, secrets, packages, setup steps, and bundled files; do
not embed credentials or copy secret values into ordinary profiles or tests.

Before starting or restarting anything, run:

```bash
uv run zippergen validate path/to/workflow.py:workflow
uv run zippergen show path/to/workflow.py:workflow --detail full
```

Use the guided path when deployment is explicitly authorized:

```bash
uv run zippergen deploy path/to/workflow.py:workflow
```

Afterward use the deployment name with `doctor`, `status`, `logs`, `restart`,
and `configure --restart`. Never assume permission to send live messages,
modify production data, complete OAuth, or restart a live service merely
because the user requested workflow code.
