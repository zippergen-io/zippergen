# Prompt-First ZipperGen Studio - Design Spec

**Date:** 2026-07-18
**Status:** Design/inspection layer implemented; execution slice retained

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

Creation
────────
  Prompt   ✓ P001 registered — prompts/review-reply.md
  Task     ✓ .zippergen/current-task.md
  Next       assistant codex / assistant claude
  Inspect    task · task show · task history

zippergen [no workflow]> assistant codex

# After the assistant creates and verifies the visible Python source:
zippergen [no workflow]> use workflows/review_reply.py:review_reply
✓ Current workflow: workflows/review_reply.py:review_reply

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

Studio saves a structured assistant handoff and can open either the local Codex
CLI or Claude Code on it. These are thin, interactive launchers rather than
ZipperGen model providers. Each tool retains its own authentication, model
settings, approvals, tools, and optional MCP configuration. The task and
generated source contract remain usable by other repository-aware assistants
through `task show` and `task path`.

## Outcome feedback

All Studio commands use one semantic status renderer:

- green `✓` for a completed operation, a valid workflow, or a ready provider;
- yellow `⚠` for incomplete optional setup or a condition requiring attention;
- red `✗` for an actual command, validation, provider, run, or deployment
  failure;
- neutral `•` for noteworthy information without a success verdict.

Meaning never depends on color alone. ANSI color is automatic only when Studio
is writing to an interactive terminal, is suppressed when `NO_COLOR` is set,
and is absent from redirected output. Scriptable JSON output is unchanged.
Optional providers that are not selected are warnings, not failures.

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
and preferring the nearest `zippergen.toml`, then a directory containing
`.git`, then `pyproject.toml`. If `--project PATH` is supplied, that exact
directory is used. If no marker exists, the current directory is the root.

The visible project layer is deliberately separate from private runtime state:

```text
zippergen-tutorial/
|-- zippergen.toml
|-- .zippergen/            # generated current task, ignored by Git
|-- prompts/
|   |-- index.toml
|   `-- *.md
|-- workflows/
|-- tests/
`-- zippergen/             # optional source checkout, ignored by outer Git
```

`project init [NAME]` creates the manifest and empty prompt index. If a nested
`zippergen/` checkout exists, the manifest records it as the framework
directory, workflow discovery excludes it, and coding-assistant briefs point
to its repository guidance and workflow skill. Project initialization also
keeps the generated task, nested checkout, and optional transparent tutorial
runtime out of the outer project's Git index.

State lives outside the checkout by default:

```text
$ZIPPERGEN_HOME/workspaces/<project-name>-<path-hash>/
|-- workspace.json
|-- development.secrets.json  # optional, mode 0600
|-- requests/
|   |-- <request-id>.json
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
- current coding-assistant request ID;
- last named deployment; and
- last selected semantic view; and
- optional terminal-editor command preference.

The visible `zippergen.toml` contains the project name, prompt directory, and
optional framework-checkout directory. It contains no current run, secret, or
machine-specific deployment state.

## Ordered prompt ledger

Natural-language requirements are durable project inputs rather than terminal
or assistant-chat history. Each ledger entry has:

- a stable ID (`P001`, `P002`, ...);
- an `initial` or `refinement` provenance label;
- an ordered Markdown source file;
- active or archived state;
- optional workflow association and replacement provenance; and
- a title derived from its first Markdown heading or nonempty line.

The two provenance labels use the same lifecycle. Removing a prompt archives
it; replacing one creates a new entry and preserves the old entry. Reordering
is explicit and versionable. A coding-assistant handoff always includes every
active entry in ledger order and identifies the immediate request. Later
prompts take precedence only where they explicitly change or contradict an
earlier requirement; all unaffected earlier requirements remain in force.

The ledger is design intent. Visible Python, tests, and semantic validation are
the executable truth. A prompt never becomes an automatically deployed opaque
workflow.

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
project init [NAME]
project show
current
use
use path/to/workflow.py:workflow
runs
```

`current` reports the project and manifest, prompt counts, current workflow and
semantic name, participants, human actions, effects, validation state,
effective model routes, provider readiness, current run, and deployment.

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
create --edit [PATH] [--editor COMMAND]
refine "Add a compliance review after retry exhaustion"
refine --file PATH
refine --edit [PATH] [--editor COMMAND]
task
task show
task path
task history
assistant [codex|claude]
editor [show|set COMMAND|reset]
edit [workflow|file PATH] [--editor COMMAND]
prompts
prompts add [PROMPT]
prompts add --file PATH
prompts add --edit [PATH] [--editor COMMAND]
prompts show|inspect ID
prompts path ID
prompts edit ID [--editor COMMAND]
prompts context
prompts archive|restore ID
prompts enable|disable|remove ID
prompts replace ID [PROMPT|--file PATH|--edit [PATH] [--editor COMMAND]]
prompts move ID before|after ID
```

`--file` reads the complete UTF-8 prompt without forcing multiline
requirements through a single-line terminal input. Relative paths resolve from
the discovered project root; absolute paths and `~` are accepted. A file under
the canonical prompt directory is registered in place; an external file is
imported. Inline requirements are written to numbered Markdown files. Prompt
files and `index.toml` are ordinary reviewable project inputs and may be
versioned, but must not contain secrets. Timestamped assistant handoffs and
semantic baselines remain outside the checkout. The current handoff is also
mirrored atomically at the fixed, ignored `.zippergen/current-task.md` path.

`task` summarizes the current handoff. `task show` prints it verbatim, `task
path` prints only its stable absolute path, and `task history` lists the private
immutable archive. `assistant codex` (also plain `assistant`) launches the local
Codex CLI; `assistant claude` launches Claude Code. Both use the project as the
working directory and the current task as the initial instruction. The chosen
tool requires its own one-time installation and authentication, but no
ZipperGen provider or MCP setup. It may still use tools or MCP servers from its
own independent configuration.

`editor set COMMAND` remembers a machine-specific preference in the private
project workspace. `editor reset` restores discovery through `$VISUAL`,
`$EDITOR`, then `micro`, `nano`, `vim`, and `vi`. Every edit command accepts a
one-off `--editor COMMAND` override without changing the preference. Studio
launches the parsed executable directly in the project root without a shell,
model call, or MCP dependency. The path for `create/refine/prompts ... --edit`
is optional. Without one, Studio opens a private project-local draft, derives a
title from the first heading or nonempty line, assigns the stable ID, and
creates `prompts/NNN-title-slug.md`. It removes the draft only after successful
registration; failures leave the user's text recoverable. `edit file` only
changes a file; `edit workflow` targets the selected file-backed Python
workflow and suggests validation afterward.

`prompts` is a columnar ledger view with position, stable ID, provenance kind,
active/archive status, title, and file. `inspect`, `path`, `edit`, `archive`,
`restore`, `replace`, and `move` all target the ID. Direct edit uses a validated
staging copy and preserves the ID; replacement creates a new ID, archives the
old entry, and records provenance. A path-free replacement draft is prefilled
with the old text. Archive is the routine deletion semantics; it removes an
entry from active assistant context without destroying history.
`initial` and `refinement` remain useful provenance labels, while both kinds
share the same lifecycle and explicit ordered-precedence rules.

### Model and provider configuration

```text
models show
models default SPEC
models set LIFELINE SPEC
models reset LIFELINE|all
providers
providers set openai|anthropic|mistral
providers set local [BASE_URL]
providers reset NAME
```

Model routing and provider credentials are separate concepts. Routes use
compact explicit specs and may be inherited or overridden per LLM-active
lifeline. Provider readiness is visible without revealing secrets. API keys
are entered without echo and stored only in the owner-readable development
secret file; a local provider stores only its non-secret endpoint. Development
runs temporarily inject the selected configured providers and restore the
process environment afterward.

A routing-only model change uses `models default` or `models set` and requires
no source edit or coding assistant. A model change that belongs in versioned
intent, workflow/deployment declarations, action prompts, or tests uses the
normal source-change loop: `refine`, optionally inspect `task`, launch either
assistant, then `current`, `validate`, semantic views, and a fresh `run`.

### Future connector and human-channel bindings

The project and dashboard contracts reserve connector bindings as a separate
layer, but this slice does not pretend to configure services it cannot yet
validate. A lifeline remains a role, authority, or sequential participant; it
is not itself a Gmail token, Google Sheet, or Telegram chat. A later slice will
bind declared effect/human action capabilities to named adapters, for example:

```text
connectors bind Mailer gmail:work-account
connectors bind Records google-sheets:review-log
humans set Reviewer telegram:review-team
```

Those bindings must keep OAuth/API credentials private and make permissions,
idempotency, durable task identity, delivery, authenticated response, retries,
and audit behavior inspectable before execution. `current` already reports
`Connector bindings: none` so the future layer has an explicit place rather
than being confused with model routing or hidden inside prompt text.

The assistant handoff contains:

- combined natural-language requirements;
- project root and target path/spec;
- the ZipperGen skill invocation;
- participant/input/output/message/action/control/deployment checklist;
- required validation and code views;
- for refinement, a unique semantic baseline and expected preserved behavior;
- an explicit no-deployment boundary unless deployment was requested.

Studio archives the handoff under `requests/`, mirrors it at the fixed project
task path, and prints a concise result table instead of flooding the terminal
with the full content. The selected Codex or Claude Code launcher consumes the
same task and reports its exit status. The adapter boundary is not entangled
with workflow execution.

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

Role runners never read the shared terminal directly. A terminal backend marks
itself as requiring the main thread; the local supervisor bridges the worker's
request to that thread and returns its answer. Interrupting the prompt leaves
the SQLite task pending, releases the requesting role, stops the other roles,
and returns control to Studio only after no worker can consume stdin.

## Implementation structure

```text
zippergen.workspace
  project discovery
  visible manifest and ordered prompt ledger
  workspace/run/request paths
  atomic JSON state
  atomic TOML project state
  private provider profiles and secrets
  AST workflow discovery

zippergen.dev
  input collection
  managed run lifecycle
  inline durable execution
  resume compatibility checks

zippergen.studio
  line-oriented command loop
  project/prompts/current/use/show/models/providers/run/resume/create/refine
  deploy/operations
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
