# ZipperGen authoring reference

Use this reference as a compact syntax and review guide. Confirm details against
the target checkout because the API may evolve.

## Minimal module

Keep decorated workflow functions at module top level. The builder reads their
source and rewrites ZipperGen's Python-shaped protocol syntax.

```python
from zippergen import Json, Lifeline, llm, workflow

User = Lifeline("User")
Writer = Lifeline("Writer")


@llm(
    system="Write a concise, factual reply.",
    user="{topic}",
    parse="text",
    outputs=(("draft", str),),
)
def draft_reply(topic: str) -> None: ...


@workflow
def answer(topic: str @ User) -> str:
    User(topic) >> Writer(topic)
    Writer: draft = draft_reply(topic)
    Writer(draft) >> User(draft)
    return draft @ User
```

An annotated input such as `topic: str @ User` states initial ownership. A
return such as `draft @ User` states final ownership. A message binds values at
the receiver. Use supported coordination types (`str`, `bool`, `int`, `float`,
`Json`, and supported tuples) at protocol boundaries. `Json` means an ordinary
portable JSON value made from `None`, booleans, numbers, strings, lists, and
dictionaries with string keys. Use it for structured records and tool results.
ZipperGen validates the whole value before durable execution. Do not pass
arbitrary Python objects through workflow variables.

Use workflow inputs for application data that belongs to a participant, not
for every operational setting used by an effect. If the requested run command
does not pass a directory or endpoint, give that setting a useful project
default or expose it as an option. Preserve literal examples exactly. A sample
file directly in `mailbox/` does not authorize an extra `inbox/` directory, and
a plain text sample must not silently acquire mandatory headers. Run the exact
documented command and exact sample before reporting success.

## Model routing

Keep model choice out of action declarations. Put named, portable model
configurations and their participant or action assignments in
`zippergen.toml`:

```bash
zg model configure writer openai:gpt-4o-mini
zg model assign Writer writer
```

```toml
[models.configurations."writer"]
provider = "openai"
model = "gpt-4o-mini"
spec = "openai:gpt-4o-mini"

[models.assignments]
default = "mock"

[models.assignments.lifelines]
Writer = "writer"

[models.assignments.actions]
"Reviewer.check_draft" = "writer"
```

The action form is more specific than the participant form. Plain runs,
durable runs, and deployments resolve the same assignments. A deployment
stores the concrete result as its operational snapshot, so redeploy after
changing project routing. `--llm SPEC` replaces all assignments for one
command. `--llm-for PARTICIPANT_OR_ACTION=SPEC` is a narrower temporary
override. API keys remain in the environment or private site storage, never
in this file.

## Action selection

Choose an action by semantics, not convenience:

- `@pure`: deterministic, local computation with no external I/O or durable
  side effects.
- `@effect`: external I/O or mutation. Design retry-safe or idempotent behavior
  because durable execution may replay around failures.
- `@assistant`: repository-aware work performed by a local coding-assistant
  CLI. Keep instructions visible, declare dynamic inputs explicitly, and make
  the requested file changes safe to resume.
- `@llm`: model generation or judgment. Declare prompts, parse format, and all
  typed outputs explicitly.
- `@human`: a durable human input, confirmation, edit, selection, or
  acknowledgement.
- `@planner`: runtime generation of an allowed sub-workflow. Generated output
  is restricted to validated workflow statements and schema-checked `@llm`
  declarations. Define `@pure` helpers in reviewed source and pass them through
  `actions=`. Do not use a planner for ordinary prompt-to-source authoring.

Typical deterministic and effect actions:

```python
from zippergen import effect, pure


@pure
def normalize(value: str) -> str:
    return value.strip()


@effect
def send_reply(address: str, body: str) -> str:
    # Call the external service here; make retries safe.
    return "sent"
```

Use `@effect(visible=False)` only for intentionally hidden operational work,
not to conceal meaningful protocol behavior.

Coding-assistant work is a distinct, inspectable external action:

```python
from zippergen import assistant


@assistant(
    instructions_file="prompts/update_release_notes.md",
    access="write",
    external_tools="none",
    shell="restricted",
    workspace=".",
)
def update_release_notes(change: str) -> str: ...
```

Exactly one of `instructions=` and `instructions_file=` is required. Markdown
paths are project-relative, fingerprinted in semantic snapshots, validated,
and included automatically in guided deployment bundles. The function
parameters are typed workflow data passed separately from the static
instructions. Select the runtime CLI with a named project configuration:

```bash
zg assistant configure coding-agent codex
zg assistant assign Developer coding-agent
zg assistant check
```

A participant assignment covers all of that participant's assistant actions.
Use an exact target such as `Developer.update_release_notes` for one action.
The same routing applies to plain runs, durable runs, and deployments. The
low-level Python API may still pass `assistant_backend=` directly. The old
static `backend=` field remains a compatibility fallback, but project routing
takes precedence. Do not use it for new CLI projects.

`zg assistant check` verifies that the selected executable exists and supports
the required safety options. It does not inspect or manage authentication.
Codex and Claude keep their own login systems.
Assistant actions default to `access="read-only"`. Declare `access="write"`
explicitly for actions that may change the repository. ZipperGen maps this
policy to the selected CLI's non-interactive sandbox or permission mode; do not
rely on prompt wording alone to make a reviewer read-only.

Filesystem access is separate from external-tool access. The default
`external_tools="none"` disables configured MCP servers, dedicated web tools,
and assistant subagents. Use `external_tools="configured"` only when those
configured capabilities are an intentional part of the action. Both policies
are always visible in semantic snapshots and diffs.

Shell capability is a third explicit policy. The default
`shell="restricted"` gives each backend its strongest practical boundary:
Codex keeps sandboxed command execution with network disabled when external
tools are disabled, while Claude receives no Bash tool. Codex also runs with
strict configuration parsing so an unknown isolation key fails closed. Use
`shell="enabled"` only when the action genuinely requires it. Validation warns
when Claude may receive Bash because that backend does not provide the same
structural network boundary.

Prefer a separate visible `@effect` containing a fixed command and arguments
for verification after shell-free assistant edits. Never execute a command
string returned by the assistant. Truly provider-independent arbitrary-shell
isolation requires an external OS or container sandbox.

Validation also warns when a write workspace contains the executing workflow,
because that workspace can permit self-modification. In that case, make the
static instruction explicitly protect the workflow and prohibit deployment,
service control, commits, pushes, and unrelated external mutations unless the
reviewed protocol deliberately requires them.

An assistant action is journaled like other external actions in durable mode:
a recorded result is replayed without launching the assistant again. The
requested repository operation should nevertheless be restart-safe because a
process can fail after the CLI changes files but before its result is recorded.

Human delivery needs no separate connector declaration. The participant is
discovered from each `@human` action, and `zg connector assign` routes it to a
saved configuration. Assign a participant to cover all of its human actions,
or `Participant.action` to override a single one.

Non-human services remain explicit, credential-free requirements:

```python
from zippergen import (
    ConnectorRequirement,
    effect,
    read_json_rows,
    upsert_json_row,
)

zippergen_connectors = (
    ConnectorRequirement(
        name="review-log",
        kind="google-sheets",
        participant="Records",
        capabilities=("read-rows", "upsert-row"),
        access="read-write",
    ),
)

REVIEW_COLUMNS = ("review_id", "status", "notes")


@effect(connector="review-log", operation="upsert-json-row")
def save_review(review_json: str) -> str:
    return upsert_json_row(
        "review-log",
        review_json,
        columns=REVIEW_COLUMNS,
        key_field="review_id",
    )


@effect(connector="review-log", operation="read-json-rows")
def read_reviews() -> str:
    return read_json_rows("review-log", columns=REVIEW_COLUMNS)
```

Use connector declarations when workflow behavior requires a non-human
external capability independently of deployment configuration. Named
configurations, assignments, and bindings are stored in the committed
`zippergen.toml` manifest. Secrets and machine-specific state remain
private in `ZIPPERGEN_HOME`, while semantic snapshots and full
views retain the logical kind, participant, access, capabilities, and each
effect's logical connector operation. For Google Sheets writes, prefer a
stable-key upsert to a blind append. This makes a retry after a crash safe.
Never put a spreadsheet ID, OAuth token, or credentials path in workflow code.
The spreadsheet ID belongs in a named project connector configuration. The
OAuth token remains private site state.
Use `zg connector configure NAME PROVIDER` to save the concrete resource, then
`zg connector bind REQUIREMENT NAME` to connect the logical requirement to it.

In a human terminal, required values may be omitted and ZipperGen asks for
them. This applies to model, assistant, and connector configuration,
assignment, and binding. Scripts and coding agents should pass the values
explicitly. Model setup asks a person for provider and model separately, while
explicit commands use `PROVIDER:MODEL`. API keys and connector credentials are
prompted without echo and saved only in private site storage.

Gmail follows the same pattern:

```python
zippergen_connectors = (
    ConnectorRequirement(
        name="mailbox",
        kind="gmail",
        participant="Mailbox",
        capabilities=("read-messages", "mark-processed", "create-draft"),
        access="read-write",
    ),
)

@effect(connector="mailbox", operation="read-messages")
def read_mail() -> str:
    ...
```

Keep the account, Gmail search query, and OAuth token outside workflow source.
The account and query are project configuration. The token is private site
state.
`zg connector authorize google` can authorize Gmail and Google Sheets together
when the workflow requires both. Declare `access="read-only"` for readers. Use
`read-write` only when an action modifies Gmail or Sheets. That declaration
selects the narrowest supported Google OAuth scope, and deployment refuses to
start if the granted scopes do not cover it.

## Owned control flow

Every choice has one owning lifeline. That owner must possess the guard data.
ZipperGen projects the global choice and inserts required coordination.

```python
@workflow
def review(topic: str @ User) -> str:
    User(topic) >> Writer(topic)
    Writer: draft = draft_reply(topic)
    Writer(draft) >> Editor(draft)
    Editor: approved = approve_reply(draft)
    if approved @ Editor:
        Editor(draft) >> User(draft)
    else:
        Editor(draft) >> Writer(draft)
        Writer: draft = revise_reply(draft)
        Writer(draft) >> User(draft)
    return draft @ User
```

Use the same ownership form for loops:

```python
while (attempts < limit and not approved) @ Reviewer:
    ...
else:
    ...  # optional exit protocol
```

Keep guards free of external effects. The `else` of a `while` represents the
exit protocol, not an error handler.

## Reusable protocol fragments

Use `@fragment` for a reusable or conceptually coherent coordination
subsequence that belongs inside a larger global workflow. A fragment may be
worth naming even when called only once if it keeps a long protocol at a
reviewable size:

```python
from zippergen import fragment


@fragment
def request_review(draft):
    Writer(draft) >> Reviewer(draft)
    Reviewer: approved = approve_reply(draft)
    Reviewer(approved) >> Writer(approved)


@workflow
def answer(topic: str @ User):
    User(topic) >> Writer(topic)
    Writer: draft = draft_reply(topic)
    request_review(draft)
```

Calling the fragment inside `@workflow` records its statements directly in the
surrounding protocol, as if they had been written inline. Fragment parameters
are the DSL values already in scope at the call site; lifelines and other
module-level DSL values may be referenced as globals. A fragment is not a
separately loaded, run, deployed, or durable sub-workflow. Use a top-level
`@workflow` when independent execution and deployment are required.

## Parallel work

Use a parallel region only for independent branches. Each branch must contain
complete ordinary protocol statements, and the continuation runs after all
branches complete.

```python
from zippergen import branch, parallel

with parallel:
    with branch:
        Researcher: facts = research(topic)
    with branch:
        Writer: outline = outline_reply(topic)
```

Do not create parallel branches that race on the same logical value or depend
on one another's intermediate results.

## Human actions

Model human authority as a participant and a `@human` action. Prefer durable
CLI or notification-backed approvals for deployments; browser UI is a legacy
visualization surface.

```python
from zippergen import human


@human(
    kind="confirm",
    context="{draft}",
    instruction="Approve this reply?",
    outputs=["approved: bool"],
)
def approve(draft: str) -> None: ...
```

Do not replace a required human approval with an LLM judgment unless the user
explicitly changes the authority model.

## Deployment declaration

Add a data-only `zippergen_deployment` declaration when the workflow has
runtime requirements:

```python
from zippergen import (
    DeploymentField,
    DeploymentPackage,
    DeploymentSetup,
    DeploymentSpec,
)

zippergen_deployment = DeploymentSpec(
    name="answer-workflow",
    description="Generate reviewed answers.",
    fields=(
        DeploymentField(
            "service_token",
            "External service token",
            target="env",
            env="SERVICE_TOKEN",
            secret=True,
            required=True,
        ),
    ),
    packages=(DeploymentPackage("external-client", "external_client"),),
    setup=(
        DeploymentSetup(
            "authorize",
            "Authorize the external service",
            ("{python}", "path/to/setup_client.py", "--setup"),
            creates_env="SERVICE_TOKEN_PATH",
        ),
    ),
    files=("path/to/workflow.py", "path/to/setup_client.py"),
)
```

Field targets are `option`, `env`, `llm`, `services`, and `input`. Mark a field
as `secret=True` only with the `env` target. Use `when` and `when_values` for
conditional requirements. Use `path_exists=True` for required local paths.
List every source/support file needed by the deployment bundle.

## Semantic CLI contract

`zg` is the short alias for `zippergen`. Inside a project the workflow may be
omitted: commands take an explicit argument first, then `workflow_entry` from
`zippergen.toml`, then the project's only workflow when there is exactly one.
An explicit spec is given in either `module:workflow` or `path.py:workflow`
form.

```bash
# Global code view
zg show

# Communication-only view
zg show --communications

# Exact single-participant projection
zg show --agent Writer

# Selected participants with explicit external boundaries
zg show --agents Writer,Editor

# Action implementations, prompts, and deployment declaration
zg show --detail full

# Machine-readable forms
zg show --format json
zg validate --json

# Stable before/after change contract
zg snapshot -o /tmp/before.json
zg diff /tmp/before.json
zg diff /tmp/before.json --format json

# An explicit spec always wins over the project entry
zg show path/to/workflow.py:workflow --agent Writer
```

The semantic diff compares meaning-bearing IR facts: participants, owned
inputs/outputs, messages and their control context, action kinds and
implementations, action sites, control constructs, parallel regions, and
deployment requirements. It deliberately ignores irrelevant source layout.

## Running

```bash
# A plain run: nothing is written down
zg run --llm mock --input message=hello

# Scripted answers, so both sides of a decision can be exercised
zg run --llm scripted:answers.json --input message=hello

# Override one participant or one exact action
zg run --llm openai:gpt-4o --llm-for User.approve_reply=mock

# Record the run so it survives a stop, then continue it
zg run --durable --input message=hello
zg inspect --agent Writer
zg run --resume
```

A plain run keeps everything in memory and leaves no store behind. `--durable`
records the run to SQLite and collects missing inputs interactively; `--resume`
continues the project's most recent unfinished run, and `--run-id` picks a
different one. Passing `--store` implies a durable run.

While a durable run is active in one terminal, another terminal can follow its
program position with `zg inspect --watch`. Add `--agent NAME` to keep one
participant's local projection in focus. Ctrl-C closes the view without
interrupting the run.

In a scripted file each key is `Participant.action`, falling back to a bare
`action` name. A bare object repeats for every call; a list is a finite
sequence, and a call past its end is an error rather than a silent repeat.

A scripted file answers model actions only. A `@human` action asks a person on
the terminal, or reaches them through an assigned connector; to drive one
without a person, pipe the answer on standard input.

## Deployment operation

```bash
zg deploy --name production
zg status production
zg inspect production --watch
zg logs production
zg doctor production
zg restart production
zg stop production
zg compact production
zg remove production
```

`status`, `inspect`, and `doctor` read durable state without changing it.
`inspect` shows each participant's current local program position. Its
`--watch` mode refreshes that view in place, once per second by default. Use
`--interval SECONDS` to change the rate. `trace` and
`tasks` show recent events and pending human tasks. `compact` requires a
stopped deployment and removes only events covered by durable recovery
snapshots. Completed human tasks and connector notifications remain as audit
records. `remove` deletes a deployment but keeps its durable store unless you
purge it.

Keep stores owner-private on a local filesystem with reliable SQLite locking
and `fsync`. Never edit durable rows directly. External effects must be
idempotent, because crash recovery or a restore from an older backup can repeat
unjournaled remote work. The events table has an explicit integer primary key,
and compaction preserves those identifiers because recovery floors refer to
them.

## Connectors

```bash
# Hand Telegram setup to the user's terminal. It prompts for chat id and token
zg connector configure approval-chat telegram

# Save external-service connectors and bind their declared requirements
zg connector configure records google-sheets --spreadsheet-id SHEET_ID --tab Calls
zg connector bind review-log records
zg connector configure inbox gmail --query 'is:unread in:inbox'
zg connector bind mailbox inbox

# Route a participant's human actions to a saved connector
zg connector assign User approval-chat

# Inspect and check all model, assistant, and connector routing
zg config
zg config check

# Authorize Google on this computer, or accept an authorization made elsewhere
zg connector authorize google --scopes gmail.readonly
zg connector accept google
```

`configure` writes the credential into `ZIPPERGEN_HOME`, outside the project.
The routing it produces — which chat, which spreadsheet, which query — is
committed with the project and contains no secret; the deployment refers to
each credential by environment-variable name only.

Deployment refuses to start when a required connector is unbound, when Google
is unauthorized or its granted scopes do not cover the workflow, or when a
connector is assigned to a participant that has no `@human` action.

The credential file is written owner-readable and is not encrypted. Anyone who
can read the account's files can read it. Treat `ZIPPERGEN_HOME` as you would
an SSH key.

## Review checklist

Before handoff, verify:

- Every cross-participant value transfer is an explicit message.
- Every guard has one correct owner that possesses its data.
- Effects are retry-safe and testable with fake services.
- LLM output parsing and types match downstream use.
- Human authority remains explicit.
- Parallel branches are independent and their results join before use.
- Workflow outputs exist at the declared lifelines on every completing path.
- Secrets occur only in environment-backed deployment fields.
- `validate` succeeds and relevant local projections are readable.
- The semantic diff contains exactly the requested changes.
- Every branch of every decision has actually been run, not just one.
- `specification.md` describes the workflow as it now stands.
