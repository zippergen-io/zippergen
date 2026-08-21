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

### When the model fails

An `@llm` action fails in three ways, and they need different answers. A
refused connection, a timeout, a 429 or a transient 5xx is worth repeating. So
is an answer that arrives but does not parse, or does not match the declared
outputs -- the next sample may well be valid. A rejected key, an unknown model
or a malformed request is not: it will fail identically forever.

```python
@llm(
    system="Draft concise replies.",
    user="{message}",
    parse="text",
    outputs=[("draft", str)],
    retries="forever",          # keep trying until the run is stopped
)
def draft_reply(message: str): ...


@llm(
    system="Classify the request.",
    user="{message}",
    parse="bool",
    outputs=[("accepted", bool)],
    temperature=0,              # action policy overrides model configuration
    retries=3,                  # three attempts after the first
    fallback=False,             # then this, instead of failing
)
def classify(message: str): ...
```

`retries` is a non-negative integer or `"forever"`; omitted it is `3`.
`fallback` is the output value itself for one output, or a mapping naming
exactly the declared outputs for several. Its names, types and JSON validity
are checked when the action is declared, not when a model first misbehaves.

Omitting `fallback` keeps the failure loud: the last error is raised. Declaring
one changes what happens twice over -- when the retries run out, and when a
permanent failure occurs, which is not retried at all. Waiting is 2, 4, 8
seconds and so on, capped at 30, and a provider's `Retry-After` wins over that.
A stopped run interrupts the wait immediately and does *not* take the fallback:
cancellation is not failure.

Temperature is optional and ranges from 0 to 1. Put the ordinary default on a
named model configuration; use `@llm(temperature=...)` only when one action
needs to pin its sampling policy independently of routing. The action wins.
Zero lowers sampling variation but does not guarantee identical hosted-model
answers. Current model families that removed sampling controls reject an
explicit temperature clearly; ZipperGen never drops one silently.

Nothing else is caught. A bug in ZipperGen or in a workflow's own code
propagates, because a retry loop that swallows every exception turns a typo
into an unbounded wait.

A declared `tuple` output keeps its type. ZipperGen's tagged value encoding
distinguishes tuples from lists, including when they are nested, before the
fallback is checked. Fallback and real answers pass the same check, in the
action's own output namespace, before being bound to workflow variables.

Retries are counted within one invocation, so two lifelines calling the same
model do not share a budget. The count does not survive a crash: a process that
dies mid-action starts its budget again on resume, which is the same
at-least-once boundary every external action has.

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
zg provider configure openai-main openai
zg model configure writer openai-main gpt-4o-mini
zg model assign Writer writer

# A model on this machine: kind `local` (or `ollama`), an endpoint instead of
# a credential, so no set-credential step.
zg provider configure my-ollama local --base-url http://127.0.0.1:11434/v1
zg model configure extractor my-ollama qwen2.5:7b --idle-timeout 300
```

```toml
[providers.connections."openai-main"]
kind = "openai"

[models.configurations."writer"]
connection = "openai-main"
model = "gpt-4o-mini"

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
  because a crash before the successor state commits may repeat the operation.
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

### Durable state between actions

A durable run persists workflow variables and control state, not Python module
globals. Never use a mutable global to remember the item an earlier action
claimed, the path it renamed, or the cursor a later action needs. Resume starts
in a fresh process, so that memory is gone even though the workflow continues
at the next statement.

Return a stable identity and payload from the claiming effect, derive ordinary
workflow variables from that result, and pass the identity to later effects:

```python
from zippergen import Json, effect, pure, workflow


@effect
def claim_next() -> Json:
    existing = find_existing_claim()
    if existing is not None:
        return existing
    return create_claim()


@pure
def claim_id(claim: Json) -> str:
    return str(claim["id"])


@pure
def claim_payload(claim: Json) -> str:
    return str(claim["payload"])


@effect
def finalize(item_id: str, payload: str) -> str:
    # Return the same success if this item was already finalized.
    return finalize_idempotently(item_id, payload)


@workflow
def process_next() -> str:
    Mailbox: claim = claim_next()
    Mailbox: item_id = claim_id(claim)
    Mailbox: payload = claim_payload(claim)
    Mailbox: status = finalize(item_id, payload)
    return status @ Mailbox
```

Make both external boundaries retry-safe. If claiming succeeded but its return
did not commit, another call to `claim_next` must recover the same claim rather
than select another item. If finalization succeeded but its successor state did
not commit, another call to `finalize` must recognize the completed item and
return the same outcome. Test both cases using a fresh process; an in-process
test cannot reveal reliance on a module global.

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
low-level Python API may still pass `assistant_backend=` directly. Project
routing is the only static way to select a backend for an `@assistant` action.

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

An assistant action runs outside the SQLite transaction in durable mode. Its
result and successor control state commit together afterward. The requested
repository operation must therefore be restart-safe: a process can fail after
the CLI changes files but before that commit, causing the assistant to launch
again.

Human delivery needs no separate connector declaration. The participant is
discovered from each `@human` action, and `zg connector assign` routes it to a
saved configuration. Three levels, most specific first: `Participant.action`
for one action, `Participant` for all of that participant's human actions, and
`default` for every participant the other two do not name. Models and
assistants use the same three levels.

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
Use `zg provider configure CONNECTION KIND` to name the provider identity,
`zg connector configure NAME CONNECTION [KIND]` to save the concrete resource,
then
`zg connector assign REQUIREMENT NAME` to connect the logical requirement to it.

In a human terminal, required values may be omitted and ZipperGen asks for
them. This applies to model, assistant, and connector configuration,
assignment, and binding. Scripts and coding agents should pass the values
explicitly. Model setup asks for a provider connection and model separately.
`zg provider set-credential CONNECTION` prompts without echo and saves API keys or
bot tokens only in private site storage.

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
`GmailMailbox.fetch_one_unread()` returns one canonical `gmail_id`, the RFC
`message_id`, Gmail's integer `internal_date_ms`, and the raw sender-supplied
`date` header. Use `internal_date_ms` for Gmail inbox ordering; do not treat
the `date` header as trusted arrival time.
Google connectors need the optional extra, `zippergen[google]`, in the
environment that runs `zg`; `zg provider check` reports it as *google support
installed*. `zg provider authorize CONNECTION` then authorizes Gmail and Google
Sheets together when the workflow requires both, saving the credential on this
machine. Scopes are not typed: they follow from what the requirements declare. Declare `access="read-only"` for readers. Use
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

For Causal Past Logic guards, prefer structural field terms such as
`At[Sensor].version == Here.version`; these receive a durable identity
automatically. Python cannot reliably infer whether a lower-level predicate's
helpers, globals, or external dependencies still mean the same thing, so an
`atom()` used durably must declare its semantic identity explicitly:

```python
ready = atom(is_ready, src="ready", version="policy-v2")
```

`src` is only a display label. `version` is part of durable semantics, and must
change whenever the predicate's meaning changes. A live run refuses monitor
state written under a different version.

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

Model human authority as a participant and a `@human` action. Use the durable
CLI directly, or assign a connector for deployment delivery.

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

Terminal human prompts are labelled `REQUEST · Participant` (`NOTICE` for an
acknowledgement). If a local effect also prints a user-facing outcome, begin it
on a fresh line and use a compact participant label such as
`✓ Mailbox · reply sent`; concurrent output may interleave, so do not repeat a
large generated value in that notification.

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
`zippergen.toml` (set it with `zg workflow select SPEC`), then the project's
only workflow when there is exactly one.
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
zg snapshot /tmp/before.json
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
zg run inspect --agent Writer
zg run --resume
```

A plain run leaves no resumable state behind. It may use temporary private
SQLite coordination while an asynchronous connector is active. `--durable`
retains the run and collects missing inputs interactively; `--resume`
continues the project's most recent unfinished run. All modes honor the
project's model, assistant, and connector routing. ZipperGen owns the store;
ordinary commands never need its path.

`zg run status` reports the currently selected durable run. Starting a new
`zg run --durable` creates fresh state and discards any older selected
development run. `zg run reset` permanently discards the selected run record
and SQLite state, then clears the selection; add `--archive` only when a
recoverable private copy is wanted. It does not start another run; use `zg run
--durable` when one is wanted. Stop the foreground run with Ctrl-C before
resetting it. One project has one active execution: a disposable or durable
foreground run, or its deployment. Stop the current execution before starting
another. Observation commands such as `status`, `inspect`, `trace`, and `tasks`
remain available while it runs.

While a durable run is active in one terminal, another terminal can follow its
program position with `zg run inspect --watch`. Add `--agent NAME` to keep one
participant's local projection in focus. Ctrl-C closes the view without
interrupting the run.

In a scripted file each key is `Participant.action`, falling back to a bare
`action` name. A bare object repeats for every call; a list is a finite
sequence, and a call past its end is an error rather than a silent repeat.

A scripted file answers model actions only. A `@human` action asks a person on
the terminal, or reaches them through an assigned connector; to drive one
without a person, pipe the answer on standard input.

## Deployment operation

A project has one workflow, so it has one deployment. `zg deploy` creates and
starts it; every verb acts on that same one, so its name is never typed.

```bash
zg deploy
zg deploy status
zg deploy logs
zg deploy check
zg deploy stop
zg deploy compact
zg deploy reset --yes
zg deploy remove
```

Stop a running deployment before invoking bare `zg deploy` to update its code
or configuration. `start` on a deployment that is already running does
nothing, so it is safe to repeat.

`zg deploy status` and `zg deploy check` report workflow-bundle freshness and
ZipperGen-runtime freshness separately. Stale is a warning: the immutable
service keeps running its deployed snapshot until it is deliberately
redeployed.

`zg deploy inspect --watch` shows that deployment's live position.

There is no workflow or deployment name to pass: the project identifies both.

`status`, `inspect`, and `check` read durable state without changing it.
`inspect` shows each participant's current local program position. Its
`--watch` mode refreshes that view in place, once per second by default. Use
`--interval SECONDS` to change the rate. `trace` and
`tasks` show recent events and pending human tasks. Trace output is a timestamped
table; its event number remains the authoritative stored order, while the
wall-clock time and paired action duration are for operational diagnosis.
`compact` trims optional
inspection history and rotates logs; it refuses while the deployment is
running, before changing either resource. Recovery never reads that history,
but the stopped-service precondition also makes log rotation lossless.

How much history a store keeps is the operator's choice, recorded per store.
The default is the newest 10,000 rows. That is a row count, not a size, so a
workflow whose events carry large values holds far more bytes at the same
budget than one passing short strings:

```bash
zg deploy --history-keep 50000              # a wider window
zg deploy --history-keep 0                  # record no trace at all
zg deploy compact --set-history-keep 2000   # change it later, and apply it now
zg run --durable --history-keep 500         # the same choice for one run
```

`zg deploy status` and `zg run status` report the budget and how much of it is
in use. Bare `zg deploy compact` trims to that budget; it does not empty the
store. Use `--set-history-keep 0` to ask for that deliberately. Event numbers
keep climbing across a budget change, so `trace --after N` stays correct.
Completed human tasks, answer tokens, and connector notifications remain as
audit records. They are not needed for recovery, currently have no automatic
retention policy, and therefore grow with the number of human interactions.
`remove` deletes a deployment but keeps its durable store unless you
purge it. `reset` is the recoverable way to start over: it stops the service,
archives the store and its SQLite sidecars under ZipperGen's trash directory,
creates an empty store, and leaves the service stopped. Start it again with
`zg deploy start` once you have looked: a reset clears connector progress too,
so the next start may re-read a mailbox that was already handled.

Several deployments may share one Telegram bot; this is expected. Telegram's
inbound queue is one queue per bot with one cursor, and reading it confirms and
destroys, so exactly one process may read it at a time. ZipperGen coordinates
that with a shared per-bot store and lock file under
`ZIPPERGEN_HOME/connectors/`: whichever deployment takes the lock reads on
everyone's behalf, and each deployment then absorbs the updates its own task
tokens identify. Deployments sharing a bot must run on the same machine. Never
delete the lock file; an advisory lock lives on the inode, so unlinking the
path would break mutual exclusion.

Keep stores owner-private on a local filesystem with reliable SQLite locking
and `fsync`; do not place a live store on NFS or in a cloud-synchronised
directory. Managed execution currently targets macOS and Linux, not Windows.
Never edit durable rows directly. The store and its backups may contain
workflow inputs, model outputs, human-task context, and approval tokens, so
protect backups like the live `0600` file. External effects must be
idempotent, because crash recovery or a restore from an older backup can repeat
remote work whose successor state did not commit. Recovery reads only current
role state, outstanding messages, and durable human tasks; optional history can
be deleted without changing resumption.

## Connectors

```bash
# Hand credentials to the user's terminal; destinations remain portable
zg provider configure approval-bot telegram
zg provider set-credential approval-bot
zg connector configure approval-chat approval-bot

# Save external-service connectors and bind their declared requirements
zg provider configure google-work google
zg connector configure records google-work google-sheets --spreadsheet-id SHEET_ID --tab Calls
zg connector assign review-log records
zg connector configure inbox google-work gmail --query 'is:unread in:inbox'
zg connector assign mailbox inbox

# Route a participant's human actions to a saved connector
zg connector assign Mailbox approval-chat

# Inspect and check all model, assistant, and connector routing
zg config
zg check

# Authorize Google on this computer, or accept an authorization made elsewhere.
# Scopes come from what the workflow's requirements declare; pass --scopes only
# outside a project.
zg provider authorize google-work            # saves here
zg provider authorize google-work --handoff  # or hand it to another computer
zg provider accept google-work
```

Every configuration family offers the same verbs, so learning one teaches all
four:

```bash
zg model unassign Writer            # drop one assignment, keep the configuration
zg model rename old-name new-name   # move the name and everything naming it
zg model remove old-name            # delete an unused configuration
zg assistant unassign Developer
zg assistant rename old-name new-name
zg assistant remove old-name
zg connector unassign Mailbox
zg connector rename old-name new-name
zg connector remove old-name
zg provider rename old-name new-name
zg provider remove old-name
```

A selected durable run and a supervised deployment share the same inspection
verbs, one aimed at each:

```bash
zg run trace                        # recent protocol events
zg run tasks                        # decisions waiting for a person
zg run approve --task 1 --yes       # answer one
zg deploy trace
zg deploy tasks
zg deploy approve --task 1 --yes
zg deploy list                      # every deployment on this computer
zg deploy prune                     # clear orphans and stale archives
zg completion zsh                   # shell completion; also bash and fish
```

`rename OLD NEW` exists for every family. One command moves the name and
everything that referred to it: assignments, requirement bindings, and for a
provider connection its credential and site endpoint, which are keyed by that
name. Doing it by hand leaves the project inconsistent in between.

It touches up to three files, which cannot be written atomically, so the order
is the guarantee: private values are copied under the new name, the committed
manifest is switched, and only then are the old private values removed. An
interruption at any point leaves a project that still works -- under the old
name before the switch, the new one after it. If it stopped during the final
cleanup, a duplicate credential is left behind under the old name; running the
same rename again finishes the cleanup and removes it.

That rerun is recognised by a private marker recording the rename, written
before anything is copied and removed after the cleanup. Nothing else
authorises it: matching values would be ambiguous, because one API key shared
by two connections looks exactly like a half-finished rename. An unrelated
leftover is refused rather than deleted.

`configure` creates or updates the named configuration. In an interactive
terminal, an existing configuration's current values become the defaults.

`configure` writes the credential into the owner-only
`ZIPPERGEN_HOME/workspaces/<project>/development.secrets.json`, outside the project.
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
- No mutable module global carries per-run state between actions.
- Claim/finalize effects survive fresh-process resume and repeated execution.
- LLM output parsing and types match downstream use.
- Human authority remains explicit.
- Parallel branches are independent and their results join before use.
- Workflow outputs exist at the declared lifelines on every completing path.
- Secrets occur only in environment-backed deployment fields.
- `validate` succeeds and relevant local projections are readable.
- The semantic diff contains exactly the requested changes.
- Every branch of every decision has actually been run, not just one.
- `specification.md` describes the workflow as it now stands.
