# ZipperGen architecture

The contributor-facing description of how this codebase is put together: its
layers, its module boundaries, and exactly which constructs each published
theorem covers.

This document is tracked and ships with the repository. `CLAUDE.md` is a local
convenience for coding agents and is not in version control, so anything a
contributor needs belongs here rather than there.

## Project Overview

ZipperGen is a Python DSL and runtime for structured multi-agent LLM coordination. It implements a formal system for writing coordination programs based on message sequence charts (MSCs), where multiple LLM agents collaborate according to provably correct protocols.

The **zippergen** package in `src/` contains the coordination language, IR,
projection engine, durable runtime, CLI, and deployment support.

## Commands

```bash
# Install in development mode
pip install -e .

# Run an example
python examples/diagnosis.py

# Run all tests
python -m pytest tests/

# Run a single test
python -m pytest tests/test_projection.py::test_if_bystander
```

ZipperGen's core has no required runtime dependencies. The optional
``zippergen[google]`` extra provides Google OAuth support. Python 3.11 or newer
is required.

## Architecture

The codebase is organized as six layers that transform a global coordination program into concurrent execution:

### Layer 1 — IR (`syntax.py`)
Frozen dataclasses representing the abstract syntax: `ZTypes`, `Var`, `Expr`, `Lifeline`, statements (`MsgStmt`, `ActStmt`, `IfStmt`, `WhileStmt`, `SeqStmt`, …), and local statements post-projection (`SendStmt`, `RecvStmt`, `SelfAssignStmt`, `IfRecvStmt`, `WhileRecvStmt`). Also contains the `Workflow` class (the result of `@workflow`) with its mutable `_WorkflowRuntime` side-car. Union type aliases are used instead of inheritance to match the formal grammar and enable exhaustive pattern matching.

### Layer 2 — Action Decorators (`actions.py`)
Six decorators that turn Python functions into IR nodes:
- `@llm(system=..., user=..., parse=..., outputs=...)` — produces `LLMAction`
- `@pure` — produces `PureAction`; output name and type inferred from the function return annotation
- `@effect` — ordinary Python with an outside-world effect; may run again after a crash
- `@human` — a typed human confirmation, choice, acknowledgement, or text input
- `@assistant` — a capability-declared Codex or Claude Code action
- `@planner(description=..., actions=..., lifelines=..., allow=..., instructions=...)` — produces `PlannerAction`; see Planner subsystem below

`@llm` and `@pure` return IR nodes, not callables. To call the underlying Python of a `PureAction` directly, use `action.fn(...)`.

### Layer 3 — Program Builder (`builder.py`)
High-level API: the `@workflow` decorator records statement builders (`msg()`, `act()`, `if_()`, `while_()`, `skip()`) into a global recording stack as the decorated function body executes. An AST transformer rewrites native DSL syntax before execution:

| Source syntax | Rewrites to |
|---|---|
| `A(x) >> B(y)` | `msg(A, (x,), B, (y,))` |
| `A: out = f(inp)` | `act(A, f, (inp,), (out,))` |
| `with A:\n    y = f(x)\n    z = g(y)` | one `act(...)` call per body line |
| `if cond @ Owner: ... else: ...` | `if_(cond, Owner, then=..., else_=...)` |
| `while cond @ Owner: ... else: ...` | `while_(cond, Owner, body=..., exit_body=...)` |
| `return var @ Lifeline` | captured as output spec; statement removed |

Conditions in `if`/`while` are captured as `lambda _e: <expr>` closures evaluated against a `_CondEnv` proxy. Because `@` has higher precedence than `not`/`and`/`or`, wrap compound conditions in parentheses: `if (not agreed) @ LLM1:`.

Self-sends (`A(x) >> A(y)`) are desugared to local variable renames (`SelfAssignStmt`) — no message is sent.

The `@workflow` decorator requires the function to be defined in a `.py` file (not interactively), because it reads the source for AST rewriting.

### Layer 4 — Projection Engine (`projection.py`)
Implements the syntax-directed projection π_A(P) that transforms a global `Program` into per-lifeline local programs. For each conditional/loop, lifelines are classified as **Owner** (broadcasts decision), **Receiver** (waits for decision), or **Bystander** (skips). Fresh control variables are generated automatically.

### Layer 5 — Runtime (`runtime.py`)
`run(proc, lifelines, initial_envs, ...)` spawns one thread per lifeline. Threads communicate via FIFO queues (one per directed pair). Messages carry sequence stamps for visualization pairing. Supports pluggable LLM backends and trace callbacks.

The low-level Python API is `Workflow.configure(...)` followed by
`workflow(**inputs)`:

```python
diagnosis_consensus.configure(
    llm={"LLM1": "openai", "LLM2": "mistral"},
    timeout=600,
)
result = diagnosis_consensus(notes=..., diagnosis=...)
```

`configure` accepts:
- `backend` — raw `(action, inputs_dict) → outputs_dict` callable
- `llm` — a compact model spec or a participant/action mapping
- `timeout` — per-thread timeout in seconds

### Layer 6 — LLM Backends (`backends.py`)
Ready-to-use backend factories: `make_openai_backend`, `make_mistral_backend`,
`make_anthropic_backend`. Each makes one HTTP call using only `urllib` (no
SDK). Retry classification, backoff, cancellation, and fallback are declared
on `@llm` actions and handled centrally by `llm_policy.py`; provider factories
do not keep a second retry budget.

`make_lifeline_router(backends_dict)` routes calls to the backend registered
for the current participant or exact action. `router_from_specs(routes)` builds
those routes from compact specs and reads provider credentials from the normal
environment variables.

### Planner Subsystem (`planner.py`)
`PlannerAction` enables LLM-driven dynamic workflow generation. When a `@planner`-decorated action is executed at runtime, `_exec_planner` in `planner.py`:
1. Builds a system prompt from the action vocabulary, DSL rules, and `allow` extensions
2. Calls the LLM to generate a ZipperGen workflow spec
3. Validates the spec with `_validate_planner_spec` (10 structural checks)
4. Writes the spec to a temp file, imports it, and runs the resulting `Workflow`

The `allow` parameter controls what the LLM can generate: `"llm"` (define a
strictly validated `@llm` action), `"if"` (conditional branching), and
`"while"` (loops). Generated `@pure` helpers are rejected. Reviewed helpers
must be supplied explicitly through `actions=`.

### Code Views And Validation (`view.py`, `validation.py`)
`zippergen show` renders deterministic source-like views from the IR: global,
communication-only, boundary-aware selected-agent, or exact local projection.
Detail levels are `overview`, `protocol`, `actions`, and `full`; JSON output
includes structured metadata and canonical code. `zippergen validate` loads a
workflow, projects every lifeline, validates deployment metadata, and exercises
the renderer. These commands are the deterministic substrate for prompt-driven
workflow creation and refinement.

### CLI And Operations

`serve.py` owns argument parsing and dispatch, not deployment implementation.
The supporting modules are deliberately split by responsibility:

- `project_configuration.py` — named model, assistant, and connector setup;
- `durable_runs.py` — managed resumable project runs;
- `deployment_environment.py` — source bundles and managed Python environments;
- `deployment_profiles.py` — deployment profile interpretation;
- `deployment_checks.py` — readiness and health checks;
- `deployment_platform.py` — paths plus launchd/systemd operations;
- `deployments.py` — removal, reset, and storage/log maintenance;
- `deployment_publication.py` — managed home, artifacts, unit, store setup,
  and the declared setup steps a deployment runs;
- `execution_inspection.py` and `live_display.py` — durable observation,
  including trace interpretation;
- `connector_wiring.py` — connector records, credentials, and worker lifecycle;
- `process_environment.py` — the one environment overlay used by runs, checks,
  and foreground commands.

Workflow loading and setup hooks live in `workflow_io.py`. Keep domain logic
in the focused modules above rather than adding it back to the CLI dispatcher.

### Coding-agent integration (`skill.py`, `skills/`)
There is no interactive shell. A ZipperGen project is an ordinary directory
driven from the CLI, by a person or by a coding agent such as Claude Code or
Codex. `zippergen skill` prints the packaged skill that teaches an agent how
to work on a project; `zippergen init` creates one. The skill is prose under
version control, not generated code.

## Key Design Patterns

- **Immutable IR**: All syntax nodes are frozen dataclasses — never mutate them, always construct new ones. `Workflow` is an exception: its `_WorkflowRuntime` side-car holds mutable runtime state (backend, trace, timeout, UI handle).
- **Global recording stack**: `@workflow` relies on a module-level list (single-threaded; not safe for concurrent `@workflow` definitions).
- **Union types for exhaustive matching**: Use `isinstance` chains or `match` statements covering all Union members when pattern-matching IR nodes.
- **Small dependency surface**: Keep the core dependency-free. Integrations
  with third-party SDK requirements belong in explicit optional extras.

## Theoretical Foundation

The paper is at `../paper-isola/paper/msc-agents.tex`. The implementation
corresponds directly to the formal definitions there, with one exception noted
below.

**What each result covers.** The ISoLA paper proves completeness, soundness,
and deadlock-freeness for the grammar given below: message, action, skip,
sequence, `if`, and `while`. The parallel operator is established separately in
the EXPRESS/SOS paper and `ParallelStmt` implements that result.

`CoregionStmt` is outside both. The ISoLA paper lists coregions as future work
(`paper-isola/paper/msc-agents.tex`, "Future work"), and no published result
covers them yet. The implementation projects and runs coregions, and the
projection follows the same shape as the proved constructs, but that is a
design decision rather than a theorem. Do not describe a coregion workflow as
carrying the paper's guarantee.

### Formal Grammar

**Global programs** (what `@workflow` in builder.py constructs):
```
P ::= ε | msg A(x⃗) → B(y⃗) | act A(y⃗) := f(x⃗) | skip_A
    | if c@B then P_T else P_F      # B = decider (owns the guard)
    | while c@B do P_body exit P_exit
    | P₁ ; P₂
```

**Local programs** (what projection.py produces per lifeline):
```
S ::= send A(x⃗) → B | recv A(y⃗) ← B | act | skip | S₁ ; S₂
    | if c@A then S_T else S_F          # decider-side: evaluates guard locally
    | if A(y⃗) ← B then S_T else S_F    # recipient-side: receives control from B
    | while variants of the above
```

### Projection Rules (projection.py)

For each control construct `if c@B` / `while c@B`, lifelines are classified:

| Role | Condition | What happens |
|---|---|---|
| **Owner** (Decider) | A = B | Evaluates guard; broadcasts `top`/`bot` to all recipients |
| **Recipient** | A ∈ L(P_T) ∪ L(P_F), A ≠ B | Receives control message from B; branches on it |
| **Bystander** | A ∉ L(P_T) ∪ L(P_F) | Projects to ε (not involved) |

`L(P)` is the **participation set** — lifelines that structurally appear in P. It determines who must receive control broadcasts.

**Control messages** use a reserved tag (`κ_ctrl`) so they can't collide with user messages on the same FIFO channel.

### Key Correctness Properties

- **Completeness + Soundness** (Theorem 3.1): The distributed projected programs produce exactly the same MSC behaviors as the global program (up to erasure of control messages).
- **Deadlock-freeness** (Corollary 3.1): Guaranteed by structural induction, not by runtime checking.
- **Sequential locality**: A lifeline can't begin phase k+1 until it finishes phase k — enforced by the thread-per-lifeline + FIFO queue model.

### Terminology Map

| Paper term | Code / concept |
|---|---|
| Lifeline | Agent name string; key in `initial_envs` |
| Decider / Owner | Lifeline that owns a control construct's condition |
| Recipient | Non-decider lifeline that appears in a branch |
| Participation set L(P) | Computed in projection to find control broadcast targets |
| Control message κ_ctrl | Reserved message tag for branch decisions |
| Erasure | Dropping control-message events to compare with global semantics |
| MSC | One possible global execution trace (send/recv event sequence) |

## Canonical Example

`examples/diagnosis.py` — a medical consensus protocol with 3 lifelines (User, LLM1, LLM2) that loops until the two LLMs agree on a diagnosis verdict. This is the reference implementation matching the formal paper.
