# Prompt-First ZipperGen Studio - Design Spec

**Date:** 2026-07-18
**Status:** First vertical slice implemented

## Product promise

A new user should be able to describe a workflow, inspect the generated Python
at several semantic scopes, run it durably with human decisions in the same
terminal, refine it, and prepare a deployment without managing environment
variables, SQLite filenames, task IDs, or service-manager commands.

The complete manual remains the reference. The eventual beginner tutorial is a
short transcript of this product path, not a list of shell workarounds.

## Five-minute acceptance transcript

```text
$ uv run zippergen studio

ZipperGen Studio
Project: /path/to/project
No workflow selected.

zippergen [no workflow]> create
Describe the workflow:
> Draft an answer. A reviewer may request at most three revisions.

# Multiline requirements use a project-relative UTF-8 file instead:
zippergen [no workflow]> create --file prompts/review-reply.md

Creation brief: ~/.zippergen/workspaces/.../requests/...-create.md
Pass this brief to a repository-aware coding assistant.

# After the assistant creates and verifies the visible Python source:
zippergen [no workflow]> use workflows/review_reply.py:review_reply
Current workflow: workflows/review_reply.py:review_reply

zippergen [review_reply]> show
  1. Overview
  2. Protocol
  3. Communications only
  4. Actions and prompts
  5. Complete workflow
  6. One participant
  7. Selected participants

zippergen [review_reply]> run
Request [Explain the sky]: Explain durable execution.
Maximum retries [2]: 3

Draft: ...
Approve this draft? [y/n]: n
Draft: ...
Approve this draft? [y/n]: y

Result: ...
Run: review-reply-20260718-120000

zippergen [review_reply]> deploy review-reply
Deployment: review-reply

zippergen [review_reply]> status
```

The first implementation saves a structured assistant handoff instead of
invoking a provider. Native assistant adapters are a later vertical slice, but
the prompt and generated source contract must be stable from the beginning.

## Non-negotiable boundaries

- Generated and refined workflows are ordinary visible Python files.
- ZipperGen, not the coding assistant, performs deterministic loading,
  validation, projection, semantic rendering, diffing, execution, and
  deployment.
- Human authority remains a `@human` action and a lifeline.
- Durable development uses SQLite, but store paths are managed automatically.
- Scriptable CLI commands remain first-class for CI and automation.
- The studio is a project-aware control surface, not a replacement for the
  operating-system shell or editor.
- No secret is written to workspace state, a prompt brief, or shell history.

## Project workspace

Studio discovers the project root by walking upward from the current directory
and preferring a directory containing `.git`, then `pyproject.toml`. If neither
exists, the current directory is the project root.

State lives outside the checkout by default:

```text
$ZIPPERGEN_HOME/workspaces/<project-name>-<path-hash>/
|-- workspace.json
|-- development.secrets.json  # optional, mode 0600
|-- requests/
|   `-- <request-id>.md
`-- runs/
    |-- <run-id>.json
    `-- <run-id>.sqlite
```

`ZIPPERGEN_HOME` remains an advanced override. With no override, the existing
`$HOME/.zippergen` default is used. The user never needs to export it.

`workspace.json` contains only non-secret context. Declared development API
keys may be saved separately in `development.secrets.json` with owner-only
permissions; they never appear in workspace, run, or request JSON.

The non-secret context is:

- schema version;
- absolute project root;
- current workflow spec;
- current run ID;
- last named deployment; and
- last selected semantic view.

Each run record contains:

- run ID and timestamps;
- workflow spec and semantic workflow name;
- SQLite store path;
- public workflow inputs;
- LLM specification, options, and service mode;
- status (`created`, `running`, `waiting`, `done`, `failed`, `interrupted`);
- result or error summary; and
- the workflow semantic fingerprint used when the run began.

A run created for one semantic fingerprint is never silently reused as a new
run after source meaning changes. `run` creates a fresh run; `resume` explicitly
reuses an incomplete run.

## Command grammar

The same concepts appear in interactive and noninteractive forms.

### Context and selection

```text
current
use
use path/to/workflow.py:workflow
runs
```

`use` without an argument discovers top-level `@workflow` functions with AST
inspection, avoiding imports and their possible side effects, and opens a
numbered selector.

### Code-first inspection

```text
show
show overview
show protocol
show communications
show actions
show full
show agent Reviewer
show agents Writer Reviewer
```

`show` without arguments opens a selector. All results are rendered semantic
code views. The menu is a discovery surface, not an alternate visual model.

The corresponding scriptable commands remain available:

```bash
zippergen show path.py:workflow --detail overview
zippergen show path.py:workflow --communications
zippergen show path.py:workflow --agent Reviewer
```

### Development execution

```text
run [LLM]
resume
```

The scriptable equivalents are:

```bash
zippergen dev [path.py:workflow]
zippergen dev --resume
```

`dev` performs this sequence:

1. Resolve the explicit or current workflow.
2. Validate it before execution.
3. Collect missing workflow inputs from deployment-field defaults or guided
   terminal prompts.
4. Default to the declared LLM setting, otherwise `mock`.
5. Collect an enabled, declared environment secret once when required and
   retain it in the private development secret file.
6. Create a unique managed run and SQLite store.
7. Run all lifelines locally through the durable SQLite supervisor.
8. Present newly created or resumed human tasks inline in the same terminal.
9. Persist status and print the result plus discoverable next commands.

`--resume` reloads the current incomplete run, checks its workflow fingerprint,
and continues with the same store and recorded inputs. A mismatch stops with a
clear choice: return to the matching source or start a new run. It never deletes
or mutates the old run to force progress.

### Prompt creation and refinement

```text
create [PROMPT]
create --file PATH
refine "Add a compliance review after retry exhaustion"
refine --file PATH
```

`--file` reads the complete UTF-8 prompt without forcing multiline
requirements through a single-line terminal input. Relative paths resolve from
the discovered project root; absolute paths and `~` are accepted. Prompt files
are ordinary reviewable project inputs and may be versioned, but must not
contain secrets. The generated assistant handoff and semantic baseline remain
outside the checkout.

The assistant handoff contains:

- combined natural-language requirements;
- project root and target path/spec;
- the ZipperGen skill invocation;
- participant/input/output/message/action/control/deployment checklist;
- required validation and code views;
- for refinement, a unique semantic baseline and expected preserved behavior;
- an explicit no-deployment boundary unless deployment was requested.

The first slice writes this handoff to `requests/` and prints the exact content.
An assistant adapter may later consume the same request and report generated
files. The adapter boundary must not be entangled with workflow execution.

### Guided deployment and operation

```text
deploy [NAME]
status [NAME]
doctor [NAME]
logs [NAME]
start [NAME]
restart [NAME]
stop [NAME]
```

`deploy` is an explicit transition from development to the existing guided,
secret-aware named deployment path. Studio remembers the successful deployment
name, so the operational commands need no argument afterward. They run from the
discovered project root and retain the same behavior as their scriptable CLI
equivalents.

## Inline durable human tasks

The existing SQLite role runner materializes a `human_tasks` row before asking
the human backend. Studio supplies the terminal human backend while retaining
SQLite execution. Therefore every inline decision is still durable and
auditable.

On resume, the terminal backend may claim an existing pending task and prompt
again. Notification-oriented backends do not claim tasks this way; they keep
waiting for the external adapter. This distinction is an explicit backend
capability, not a check for a particular function name.

## Implementation structure

```text
zippergen.workspace
  project discovery
  workspace/run/request paths
  atomic JSON state
  AST workflow discovery

zippergen.dev
  input collection
  managed run lifecycle
  inline durable execution
  resume compatibility checks

zippergen.studio
  line-oriented command loop
  help/current/use/show/run/resume/create/refine/deploy/operations
  thin calls into workspace, dev, semantic, and deployment application APIs

zippergen.serve
  argparse wiring and backward-compatible scriptable commands
```

Core operations must be callable as Python functions with injected input and
output functions. The studio and tests use those functions directly; they do
not recursively parse their own CLI output.

## First vertical slice

The first slice is complete when all of the following work:

- `zippergen studio [WORKFLOW]` starts with visible project/current context;
- `current`, `use`, `show`, `help`, and tab-free numbered selection are
  discoverable;
- `zippergen dev WORKFLOW` collects two or more typed inputs and creates a
  unique managed store;
- a durable human task is answered in the same terminal;
- rejecting a tutorial draft presents the next task inline;
- `zippergen dev --resume` can claim a pending task from the current run;
- no environment export, explicit store path, task ID, or cleanup command is
  required;
- guided deployment and later status/log/doctor/restart/stop operations are
  available without re-entering the deployment name;
- the ordinary `run`, `tasks`, `approve`, and deployment commands retain their
  existing behavior;
- focused tests and the complete repository suite pass.

Native prompt execution, full-screen TUI widgets, shell completion scripts,
and secret-manager integration are subsequent slices built on the same state
and application APIs.
