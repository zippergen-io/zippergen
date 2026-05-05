# Human-in-the-Loop Design

**Date:** 2026-05-05
**Status:** Approved

## Overview

Add a `@human` action decorator that makes a human a first-class lifeline in ZipperGen workflows. When the runtime reaches a human action, it blocks and waits for real input — from the terminal (CLI mode) or from ZipperChat (UI mode). The human lifeline participates in the workflow like any LLM lifeline: it receives messages, performs a local action, and sends its output forward. The projection engine and correctness guarantees are unchanged.

---

## 1. Workflow Author API

A human lifeline is a plain `Lifeline`. Human actions are declared with a new `@human` decorator, parallel to `@llm` and `@pure`:

```python
from zippergen import Lifeline, human

reviewer = Lifeline("Human")

@human(prompt="Approve this plan: {plan}?", outputs=["approved: bool"])
def review_plan(plan: str): pass

@human(prompt="Leave a comment:", outputs=["comment: str"])
def add_comment(plan: str): pass

@human(prompt="Choose next action:", options=["approve", "reject", "escalate"],
       outputs=["decision: str"])
def choose_action(plan: str): pass
```

**Input types** (determined by the output type annotation):
- `bool` — yes/no decision; stores `True`/`False`
- `str` without `options` — free-text input; stores a string
- `str` with `options` — selection from a fixed list; stores the chosen string

**Prompt interpolation:** `{var}` references must match declared function parameters exactly. The decorator validates this at definition time.

**Single output per action.** Multiple outputs are out of scope for this version (see Section 7).

---

## 2. `HumanAction` IR Node (`syntax.py`)

New frozen dataclass added to the `Action` union alongside `LLMAction`, `PureAction`, `PlannerAction`:

```python
@dataclass(frozen=True)
class HumanAction:
    name: str
    inputs: tuple[str, ...]          # from function parameter names
    output: str                      # single output variable name
    output_type: type                # bool or str
    prompt: str                      # template with {var} placeholders
    options: tuple[str, ...] | None  # None → bool or text; tuple → choice
```

**`@human` decorator** (in `actions.py`):
1. Reads `inputs` from the function's parameter names
2. Parses the single `"name: type"` string in `outputs` into `(name, type)`
3. Validates that every `{var}` in `prompt` is a declared input
4. Validates that `options` is only set when the output type is `str`
5. Returns the `HumanAction` node (not a callable), consistent with `@llm`/`@pure`

**Projection:** `HumanAction` belongs to exactly one lifeline and projects to itself — no cross-lifeline communication, identical to `PureAction` in projection treatment.

---

## 3. Runtime Dispatch (`runtime.py`)

New case in the `ActStmt` handler:

```python
case HumanAction():
    result = human_backend(action, named_inputs)
```

`human_backend` has the same signature as `llm_backend`:

```python
(action: HumanAction, inputs: dict[str, object]) -> dict[str, object]
```

`run()` gains one new parameter: `human_backend=None`. `Workflow.configure()` sets it automatically based on the `ui=` flag — no new parameter exposed to the workflow author:

```python
# Inside configure():
if ui:
    human_backend = web_trace.make_human_backend()
else:
    human_backend = make_cli_human_backend()
```

---

## 4. Backend Implementations

### CLI backend (`make_cli_human_backend()`)

Blocks on stdin. Behaviour by input type:

| Type | Widget | Stored value |
|------|--------|-------------|
| `bool` | Prints `[y/n]`, loops until valid | `True` / `False` |
| `str`, no options | Prints prompt, reads one line | `str` |
| `str` + options | Prints numbered list, loops until valid | chosen `str` |

### Web backend (`WebTrace.make_human_backend()`)

Created by `WebTrace` when it starts:

1. Generates a unique request id (UUID)
2. Registers a `threading.Event` + result slot keyed by id
3. Emits SSE event `human_input_required`
4. Blocks on the event
5. Returns `{name: value}` once the event is set

A new `POST /human-input` endpoint in `web.py`:
- Reads `{"id": "...", "value": "..."}` from the request body
- Looks up the pending request by id
- Stores the value and sets the `threading.Event`
- Emits SSE event `human_input` so the browser can update the card

**No combined backend.** Input source is determined entirely by `ui=`:
- `ui=True` → web backend
- `ui=False` → CLI backend

---

## 5. ZipperChat UI (`web.py` + `_HTML`)

### New SSE events

```json
{"type": "human_input_required", "id": "uuid", "lifeline": "Human",
 "prompt": "Approve this plan?", "input_type": "bool", "options": null}

{"type": "human_input", "id": "uuid", "lifeline": "Human", "value": "true"}
```

### Browser behaviour

When `human_input_required` arrives, the Human lifeline column shows a **pending card** with a distinct style (soft highlight, pulsing border) to indicate it is waiting:

- `bool` → **Yes** / **No** buttons
- `choice` → one button per option
- `text` → text field + **Submit** button

On submit, the browser `POST`s `{"id": "...", "value": "..."}` to `/human-input`. The pending card updates to display the submitted value, matching how other act cards show their output. No page reload or SSE reconnect needed.

The Human lifeline header renders identically to any other lifeline in both the diagram and the sidebar tree.

---

## 6. Files Changed

| File | Change |
|------|--------|
| `src/zippergen/syntax.py` | Add `HumanAction` dataclass; add to `Action` union |
| `src/zippergen/actions.py` | Add `@human` decorator |
| `src/zippergen/runtime.py` | Add `HumanAction` case in `ActStmt`; add `human_backend` param to `run()`; wire in `configure()` |
| `src/zippergen/__init__.py` | Export `human`, `HumanAction` |
| `src/zipperchat/web.py` | Add `POST /human-input` endpoint; add `make_human_backend()`; add pending card rendering and SSE events |

---

## 7. Out of Scope

- Combined CLI + web "first responder" mode (can be added later if needed)
- Timeout on human input (can be added later)
- Human actions in the planner's vocabulary
- Multiple outputs per `@human` action (add later if needed; CLI is trivial, ZipperChat requires a multi-widget card and a dict payload in `POST /human-input`)
- Multi-turn human dialogue within a single action
