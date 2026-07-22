# ZipperGen authoring reference

Use this reference as a compact syntax and review guide. Confirm details against
the target checkout because the API may evolve.

## Minimal module

Keep decorated workflow functions at module top level. The builder reads their
source and rewrites ZipperGen's Python-shaped protocol syntax.

```python
from zippergen import Lifeline, llm, workflow

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
and supported tuples) at protocol boundaries.

## Action selection

Choose an action by semantics, not convenience:

- `@pure`: deterministic, local computation with no external I/O or durable
  side effects.
- `@effect`: external I/O or mutation. Design retry-safe or idempotent behavior
  because durable execution may replay around failures.
- `@llm`: model generation or judgment. Declare prompts, parse format, and all
  typed outputs explicitly.
- `@human`: a durable human input, confirmation, edit, selection, or
  acknowledgement.
- `@planner`: runtime generation of an allowed sub-workflow; do not use it for
  ordinary prompt-to-source authoring by a coding assistant.

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
            "llm",
            "LLM provider and model",
            target="llm",
            default="openai:gpt-4o",
            required=True,
        ),
        DeploymentField(
            "openai_api_key",
            "OpenAI API key",
            target="env",
            env="OPENAI_API_KEY",
            secret=True,
            required=True,
            when="llm",
            when_values=("openai*",),
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

Use a workflow spec in either `module:workflow` or `path.py:workflow` form.

```bash
# Global code view
uv run zippergen show path/to/workflow.py:workflow

# Communication-only view
uv run zippergen show path/to/workflow.py:workflow --communications

# Exact single-participant projection
uv run zippergen show path/to/workflow.py:workflow --agent Writer

# Selected participants with explicit external boundaries
uv run zippergen show path/to/workflow.py:workflow --agents Writer,Editor

# Action implementations, prompts, and deployment declaration
uv run zippergen show path/to/workflow.py:workflow --detail full

# Machine-readable forms
uv run zippergen show path/to/workflow.py:workflow --format json
uv run zippergen validate path/to/workflow.py:workflow --json

# Stable before/after refinement contract
uv run zippergen snapshot path/to/workflow.py:workflow -o /tmp/before.json
uv run zippergen diff /tmp/before.json path/to/workflow.py:workflow
uv run zippergen diff /tmp/before.json path/to/workflow.py:workflow --format json
```

The semantic diff compares meaning-bearing IR facts: participants, owned
inputs/outputs, messages and their control context, action kinds and
implementations, action sites, control constructs, parallel regions, and
deployment requirements. It deliberately ignores irrelevant source layout.

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
