# Human-in-the-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `@human` action decorator so a human lifeline can participate in ZipperGen workflows as a first-class agent, blocking for real input (CLI or ZipperChat) at runtime.

**Architecture:** `HumanAction` is a new frozen dataclass in `syntax.py` (parallel to `LLMAction`). A `human_backend` callable dispatches it at runtime — CLI backend for terminal use, web backend for ZipperChat. The web backend emits SSE events and fulfills requests via a new `POST /human-input` endpoint; input source is determined automatically by the `ui=` flag in `configure()`.

**Tech Stack:** Python 3.11+ stdlib only (threading, uuid, re, json, http.server). No new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `src/zippergen/syntax.py` | Add `HumanAction` dataclass; add to `ActStmt.action` union; add to `__all__` |
| `src/zippergen/actions.py` | Add `@human` decorator and `_parse_human_output` helper; add to `__all__` |
| `src/zippergen/human_backends.py` | New file: `make_cli_human_backend()` |
| `src/zippergen/runtime.py` | Add `HumanAction` case in `ActStmt`; add `human_backend` param to `run()`, `_thread_body()`, `_exec()`; wire into `_workflow_configure()` and `_workflow_run_once()`; add `_human_backend` to `_WorkflowRuntime` |
| `src/zippergen/__init__.py` | Export `HumanAction`, `human`, `make_cli_human_backend` |
| `src/zipperchat/web.py` | Add `_pending_human_inputs` dict; add `make_human_backend()` to `WebTrace`; add `POST /human-input` to handler; add `human_input_required`/`human_input` event rendering to `_HTML` JS |
| `tests/test_human_action.py` | New test file covering all tasks |
| `examples/human_approval.py` | New example workflow with human approval gate |

---

## Task 1: `HumanAction` IR Node

**Files:**
- Modify: `src/zippergen/syntax.py`
- Create: `tests/test_human_action.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_human_action.py
from zippergen.syntax import HumanAction

def test_human_action_fields():
    action = HumanAction(
        name="review_plan",
        inputs=(("plan", str),),
        output="approved",
        output_type=bool,
        prompt="Approve this plan?\n\n{plan}",
        options=None,
    )
    assert action.name == "review_plan"
    assert action.inputs == (("plan", str),)
    assert action.output == "approved"
    assert action.output_type is bool
    assert action.prompt == "Approve this plan?\n\n{plan}"
    assert action.options is None

def test_human_action_choice():
    action = HumanAction(
        name="choose",
        inputs=(("plan", str),),
        output="decision",
        output_type=str,
        prompt="Choose: {plan}",
        options=("approve", "reject", "escalate"),
    )
    assert action.options == ("approve", "reject", "escalate")
    assert action.output_type is str
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/bollig/zippergen-io/zippergen
python -m pytest tests/test_human_action.py::test_human_action_fields -v
```

Expected: `ImportError: cannot import name 'HumanAction'`

- [ ] **Step 3: Add `HumanAction` to `syntax.py`**

After `WorkflowAction` (line 240), add:

```python
@dataclass(frozen=True)
class HumanAction:
    name: str
    inputs: tuple[tuple[str, ZType], ...]   # (param_name, type) pairs
    output: str                             # single output variable name
    output_type: type                       # bool or str
    prompt: str                             # template with {var} placeholders
    options: tuple[str, ...] | None         # None → bool/text; tuple → choice

    def __repr__(self) -> str:
        ins = ", ".join(f"{n}: {t.__name__}" for n, t in self.inputs)
        return (
            f"HumanAction({self.name!r}, ({ins}) -> "
            f"{self.output}: {self.output_type.__name__})"
        )
```

- [ ] **Step 4: Add `HumanAction` to `ActStmt.action` union type**

In `syntax.py` at the `ActStmt` dataclass (around line 278), update the `action` field type:

```python
action: Union[LLMAction, PureAction, "PlannerAction", "WorkflowAction", "HumanAction"]
```

- [ ] **Step 5: Add `HumanAction` to `__all__` in `syntax.py`**

In the `__all__` list (around line 29), add `"HumanAction"` to the Actions section:

```python
# Actions
"LLMAction", "PureAction", "PlannerAction", "WorkflowAction", "HumanAction",
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_human_action.py -v
```

Expected: 2 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/zippergen/syntax.py tests/test_human_action.py
git commit -m "feat: add HumanAction IR node to syntax"
```

---

## Task 2: `@human` Decorator

**Files:**
- Modify: `src/zippergen/actions.py`
- Modify: `tests/test_human_action.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_human_action.py`:

```python
from zippergen.actions import human

def test_human_decorator_bool():
    @human(prompt="Approve this plan?\n\n{plan}", outputs=["approved: bool"])
    def review_plan(plan: str): pass

    assert isinstance(review_plan, HumanAction)
    assert review_plan.name == "review_plan"
    assert review_plan.inputs == (("plan", str),)
    assert review_plan.output == "approved"
    assert review_plan.output_type is bool
    assert review_plan.prompt == "Approve this plan?\n\n{plan}"
    assert review_plan.options is None

def test_human_decorator_text():
    @human(prompt="Add a comment about {plan}:", outputs=["comment: str"])
    def add_comment(plan: str): pass

    assert add_comment.output == "comment"
    assert add_comment.output_type is str
    assert add_comment.options is None

def test_human_decorator_choice():
    @human(
        prompt="Choose an action for {plan}:",
        options=["approve", "reject", "escalate"],
        outputs=["decision: str"],
    )
    def choose_action(plan: str): pass

    assert choose_action.output == "decision"
    assert choose_action.output_type is str
    assert choose_action.options == ("approve", "reject", "escalate")

def test_human_decorator_bad_placeholder():
    import pytest
    with pytest.raises(TypeError, match="unknown variables"):
        @human(prompt="Approve {typo}?", outputs=["approved: bool"])
        def bad_action(plan: str): pass

def test_human_decorator_options_requires_str():
    import pytest
    with pytest.raises(TypeError, match="options.*str"):
        @human(
            prompt="Choose: {plan}",
            options=["a", "b"],
            outputs=["decision: bool"],
        )
        def bad_choice(plan: str): pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_human_action.py::test_human_decorator_bool -v
```

Expected: `ImportError: cannot import name 'human'`

- [ ] **Step 3: Add `_parse_human_output` helper and `@human` decorator to `actions.py`**

At the top of `actions.py`, add `import re` to the existing imports.

After the `pure` decorator (end of file), append:

```python
# ---------------------------------------------------------------------------
# @human decorator
# ---------------------------------------------------------------------------

def _parse_human_output(spec: str, fn_name: str) -> tuple[str, type]:
    """Parse "name: type" output spec string into (name, type) pair."""
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) != 2:
        raise TypeError(
            f"@human '{fn_name}': output spec must be 'name: type', got {spec!r}"
        )
    name, type_str = parts
    type_map: dict[str, type] = {"bool": bool, "str": str}
    if type_str not in type_map:
        raise TypeError(
            f"@human '{fn_name}': output type must be 'bool' or 'str', "
            f"got {type_str!r}"
        )
    return name, type_map[type_str]


def human(
    *,
    prompt: str,
    outputs: list[str],
    options: list[str] | None = None,
):
    """
    Decorator that produces a HumanAction node.

    Parameters
    ----------
    prompt : str
        Prompt shown to the human. May contain ``{var_name}`` placeholders
        matching the function's parameter names.
    outputs : list of str
        Single-element list with ``"name: type"`` spec, e.g. ``["approved: bool"]``.
        Supported types: ``bool``, ``str``.
    options : list of str, optional
        Fixed choices for the human to select from. Only valid when output
        type is ``str``. Renders as buttons in ZipperChat.
    """
    from zippergen.syntax import HumanAction

    def decorator(fn: Callable) -> HumanAction:
        import re
        fn_name = fn.__name__
        inputs = _extract_inputs(fn)

        if len(outputs) != 1:
            raise TypeError(
                f"@human '{fn_name}': exactly one output required, "
                f"got {len(outputs)}"
            )
        output_name, output_type = _parse_human_output(outputs[0], fn_name)

        # Validate prompt placeholders
        placeholders = set(re.findall(r'\{(\w+)\}', prompt))
        input_names = {name for name, _ in inputs}
        unknown = placeholders - input_names
        if unknown:
            raise TypeError(
                f"@human '{fn_name}': prompt references unknown variables "
                f"{unknown}. Declared inputs: {input_names}"
            )

        # options only valid for str output
        if options is not None and output_type is not str:
            raise TypeError(
                f"@human '{fn_name}': options are only valid when output "
                f"type is 'str', got '{output_type.__name__}'"
            )

        _options = tuple(options) if options is not None else None

        return HumanAction(
            name=fn_name,
            inputs=inputs,
            output=output_name,
            output_type=output_type,
            prompt=prompt,
            options=_options,
        )

    return decorator
```

- [ ] **Step 4: Add `human` to `__all__` in `actions.py`**

```python
__all__ = ["llm", "pure", "planner", "human"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_human_action.py -v
```

Expected: 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/zippergen/actions.py tests/test_human_action.py
git commit -m "feat: add @human decorator producing HumanAction"
```

---

## Task 3: CLI Human Backend

**Files:**
- Create: `src/zippergen/human_backends.py`
- Modify: `tests/test_human_action.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_human_action.py`:

```python
from unittest.mock import patch
from zippergen.human_backends import make_cli_human_backend
from zippergen.syntax import HumanAction

def _make_action(output_type, options=None):
    return HumanAction(
        name="ask",
        inputs=(("plan", str),),
        output="result",
        output_type=output_type,
        prompt="Question: {plan}",
        options=options,
    )

def test_cli_backend_bool_yes():
    backend = make_cli_human_backend()
    action = _make_action(bool)
    with patch("builtins.input", return_value="y"):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": True}

def test_cli_backend_bool_no():
    backend = make_cli_human_backend()
    action = _make_action(bool)
    with patch("builtins.input", return_value="n"):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": False}

def test_cli_backend_text():
    backend = make_cli_human_backend()
    action = _make_action(str)
    with patch("builtins.input", return_value="looks good"):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": "looks good"}

def test_cli_backend_choice():
    backend = make_cli_human_backend()
    action = _make_action(str, options=("approve", "reject", "escalate"))
    with patch("builtins.input", side_effect=["99", "2"]):
        result = backend(action, {"plan": "do something"})
    assert result == {"result": "reject"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_human_action.py::test_cli_backend_bool_yes -v
```

Expected: `ModuleNotFoundError: No module named 'zippergen.human_backends'`

- [ ] **Step 3: Create `src/zippergen/human_backends.py`**

```python
"""
CLI human backend for HumanAction.

make_cli_human_backend() returns a callable with the same signature as
llm_backend: (action: HumanAction, inputs: dict) -> dict.
"""

from __future__ import annotations

__all__ = ["make_cli_human_backend"]


def make_cli_human_backend():
    """
    Return a human backend that blocks on stdin.

    - bool output: prompts [y/n], loops until valid, returns True/False.
    - str output without options: reads one line.
    - str output with options: prints numbered list, loops until valid selection.
    """
    def backend(action, inputs: dict) -> dict:
        from zippergen.syntax import HumanAction
        assert isinstance(action, HumanAction)

        prompt = action.prompt.format(**inputs)

        if action.output_type is bool:
            while True:
                raw = input(f"{prompt} [y/n]: ").strip().lower()
                if raw in ("y", "yes"):
                    value: object = True
                    break
                if raw in ("n", "no"):
                    value = False
                    break
                print("Please enter 'y' or 'n'.")

        elif action.options is not None:
            print(prompt)
            for i, opt in enumerate(action.options, 1):
                print(f"  {i}. {opt}")
            while True:
                raw = input("Enter number: ").strip()
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(action.options):
                        value = action.options[idx]
                        break
                print(f"Please enter a number between 1 and {len(action.options)}.")

        else:
            value = input(f"{prompt}: ").strip()

        return {action.output: value}

    return backend
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_human_action.py -v
```

Expected: 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/human_backends.py tests/test_human_action.py
git commit -m "feat: add CLI human backend"
```

---

## Task 4: Runtime Dispatch

**Files:**
- Modify: `src/zippergen/runtime.py`
- Modify: `src/zippergen/__init__.py`
- Modify: `tests/test_human_action.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_human_action.py`:

```python
from zippergen.syntax import Lifeline
from zippergen.actions import human, pure
from zippergen.builder import workflow
from zippergen.runtime import run

Human = Lifeline("Human")
Planner = Lifeline("Planner")

@pure
def make_task(n: int) -> str:
    return f"task-{n}"

@human(prompt="Approve: {plan}?", outputs=["approved: bool"])
def review_plan(plan: str): pass

@workflow
def approval_flow(n: int @ Planner) -> bool:
    Planner: plan = make_task(n)
    Planner(plan) >> Human(plan)
    Human: approved = review_plan(plan)
    return approved @ Human

def test_runtime_human_action():
    def mock_human_backend(action, inputs):
        # Always approve
        return {action.output: True}

    result = run(
        approval_flow,
        [Planner, Human],
        {"Planner": {"n": 42}},
        human_backend=mock_human_backend,
    )
    assert result is True

def test_runtime_human_action_reject():
    def mock_human_backend(action, inputs):
        return {action.output: False}

    result = run(
        approval_flow,
        [Planner, Human],
        {"Planner": {"n": 1}},
        human_backend=mock_human_backend,
    )
    assert result is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_human_action.py::test_runtime_human_action -v
```

Expected: `TypeError` or `match` failure — `HumanAction` case not handled.

- [ ] **Step 3: Add `_human_backend` to `_WorkflowRuntime` in `syntax.py`**

In `_WorkflowRuntime` (around line 476), add:

```python
_human_backend: object = field(default=None, repr=False)
```

Also add a property shim on `Workflow` after the existing `_last_kwargs` shim:

```python
@property
def _human_backend(self): return self._rt._human_backend
@_human_backend.setter
def _human_backend(self, v): self._rt._human_backend = v
```

- [ ] **Step 4: Add `human_backend` parameter to `run()` in `runtime.py`**

Find the `run()` signature (around line 540) and add the new parameter:

```python
def run(
    wf: Workflow,
    lifelines: list[Lifeline],
    initial_envs: dict[str, dict[str, object]],
    *,
    llm_backend=None,
    human_backend=None,
    verbose: bool = False,
    trace=None,
    timeout: float = 60.0,
) -> object:
```

At the start of `run()` body, add default after `llm_backend` defaulting:

```python
    if human_backend is None:
        from zippergen.human_backends import make_cli_human_backend
        human_backend = make_cli_human_backend()
```

- [ ] **Step 5: Thread `human_backend` through `_thread_body` and `_exec`**

Find `_thread_body` (search for `def _thread_body`) and add `human_backend` to its signature and internal `_exec` call:

```python
def _thread_body(stmt, env, ch, ns, box, llm_backend, human_backend, trace, stop):
    try:
        _exec(stmt, env, ch, ns, llm_backend, human_backend, trace, stop)
        box.append(dict(env))
    except Exception as e:
        box.append(e)
```

Find `_exec` signature and add `human_backend`:

```python
def _exec(stmt: LocalStmt, env: dict, ch: Channels, ns: dict,
          llm_backend, human_backend, trace, stop: threading.Event) -> None:
```

Update all recursive calls inside `_exec` to pass `human_backend`:
- `_exec(cast(LocalStmt, p1), env, ch, ns, llm_backend, human_backend, trace, stop)`
- `_exec(cast(LocalStmt, p2), env, ch, ns, llm_backend, human_backend, trace, stop)`
- `_exec(cast(LocalStmt, t if flag else f), env, ch, ns, llm_backend, human_backend, trace, stop)`
- `_exec(cast(LocalStmt, body), env, ch, ns, llm_backend, human_backend, trace, stop)`
- `_exec(cast(LocalStmt, exit_b), env, ch, ns, llm_backend, human_backend, trace, stop)`

Update the `make_target` closure inside `run()` to pass `human_backend`:

```python
def make_target(stmt, e, b):
    def target():
        _thread_body(stmt, e, channels, wf.ns, b, llm_backend, human_backend, trace, stop)
    return target
```

- [ ] **Step 6: Add `HumanAction` case in `ActStmt` handler inside `_exec`**

In the `ActStmt` block, after the `PlannerAction` branch and before the `else` (LLM) branch:

```python
            elif isinstance(action, HumanAction):
                named_outputs = human_backend(action, named_inputs)
                out_map = {outs[0].name: named_outputs[action.output]}
```

The full `ActStmt` block now reads:

```python
        case ActStmt(lifeline=A, action=action, inputs=ins, outputs=outs):
            ...
            if isinstance(action, PureAction):
                raw = action.fn(*in_vals)
                out_map = {outs[0].name: raw} if len(outs) == 1 else {
                    var.name: val for var, val in zip(outs, cast(tuple, raw))
                }
            elif isinstance(action, PlannerAction):
                out_map = {outs[0].name: _exec_planner(action, named_inputs, llm_backend, trace, seq)}
            elif isinstance(action, HumanAction):
                named_outputs = human_backend(action, named_inputs)
                out_map = {outs[0].name: named_outputs[action.output]}
            else:
                named_outputs = llm_backend(action, named_inputs)
                out_map = {
                    var.name: named_outputs.get(aname)
                    for (aname, _), var in zip(action.outputs, outs)
                }
```

- [ ] **Step 7: Wire `human_backend` into `_workflow_configure` and `_workflow_run_once`**

In `_workflow_configure` (around line 659), after the `if wf._rt._ui_enabled:` block, add:

```python
    # Human backend: web if UI is enabled, CLI otherwise.
    if wf._rt._ui_enabled and wf._rt._webtrace is not None:
        wf._rt._human_backend = wf._rt._webtrace.make_human_backend()
    else:
        from zippergen.human_backends import make_cli_human_backend
        wf._rt._human_backend = make_cli_human_backend()
```

In `_workflow_run_once` (around line 701), update the `run()` call:

```python
        return run(wf, list(lifelines), initial_envs,
                   llm_backend=backend,
                   human_backend=wf._rt._human_backend,
                   trace=wf._rt._trace,
                   timeout=wf._rt._timeout)
```

- [ ] **Step 8: Export `HumanAction`, `human`, `make_cli_human_backend` from `__init__.py`**

Add to `src/zippergen/__init__.py`:

```python
from zippergen.human_backends import *  # noqa: F401, F403
from zippergen import human_backends
```

And in `__all__`:

```python
__all__: list[str] = (syntax.__all__ + actions.__all__ + backends.__all__ +
                      demo.__all__ + builder.__all__ + projection.__all__ +
                      runtime.__all__ + human_backends.__all__)
```

- [ ] **Step 9: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all existing tests PASS, plus the 2 new runtime tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/zippergen/syntax.py src/zippergen/runtime.py src/zippergen/__init__.py tests/test_human_action.py
git commit -m "feat: dispatch HumanAction in runtime with pluggable human_backend"
```

---

## Task 5: Web Backend + ZipperChat UI

**Files:**
- Modify: `src/zipperchat/web.py`

- [ ] **Step 1: Add `import uuid` to `web.py` imports**

At the top of `src/zipperchat/web.py`, add `uuid` to the existing stdlib imports:

```python
import json
import pathlib
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
```

- [ ] **Step 2: Add `_pending_human_inputs` and `make_human_backend()` to `WebTrace`**

In `WebTrace.__init__`, add:

```python
        self._pending_human_inputs: dict[str, tuple[threading.Event, list]] = {}
```

After `WebTrace.stop()`, add:

```python
    def make_human_backend(self):
        """Return a human backend callable that blocks until ZipperChat provides input."""
        pending = self._pending_human_inputs

        def backend(action, inputs: dict) -> dict:
            req_id = str(uuid.uuid4())
            evt = threading.Event()
            result_box: list = []
            pending[req_id] = (evt, result_box)

            prompt = action.prompt.format(**inputs)
            if action.output_type is bool:
                input_type = "bool"
            elif action.options is not None:
                input_type = "choice"
            else:
                input_type = "text"

            lifeline_name = threading.current_thread().name
            self._bus.publish({
                "type": "human_input_required",
                "id": req_id,
                "lifeline": lifeline_name,
                "prompt": prompt,
                "input_type": input_type,
                "options": list(action.options) if action.options else None,
            })

            evt.wait()
            del pending[req_id]
            raw = result_box[0]

            if action.output_type is bool:
                value: object = str(raw).lower() in ("true", "yes", "1", "y")
            else:
                value = str(raw)

            self._bus.publish({
                "type": "human_input",
                "id": req_id,
                "lifeline": lifeline_name,
                "value": str(value),
            })

            return {action.output: value}

        return backend
```

- [ ] **Step 2: Pass `pending_human_inputs` into `_make_handler`**

Update the `_make_handler` signature:

```python
def _make_handler(bus: _EventBus, lifelines: list[str],
                  replay_event: threading.Event,
                  init_event: dict,
                  pending_human_inputs: dict):
```

Update `WebTrace.start()` to pass it:

```python
        handler = _make_handler(
            self._bus, self._lifelines, self._replay_event, init_ev,
            self._pending_human_inputs,
        )
```

- [ ] **Step 3: Add `POST /human-input` endpoint to `_make_handler`**

Inside `do_POST`, after the `/replay` branch:

```python
            elif self.path == "/human-input":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode())
                req_id = body.get("id")
                raw_value = str(body.get("value", ""))
                if req_id and req_id in pending_human_inputs:
                    evt, result_box = pending_human_inputs[req_id]
                    result_box.append(raw_value)
                    evt.set()
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()
```

- [ ] **Step 4: Add `human_input_required` and `human_input` event handling to the browser JS in `_HTML`**

Find the JavaScript section of `_HTML` that handles SSE events (the `evtSource.onmessage` or similar handler). Add cases for the two new event types.

When `human_input_required` arrives, append a pending card to the correct lifeline column. The card contains:
- For `bool`: two buttons "Yes" and "No"
- For `choice`: one button per option
- For `text`: a `<textarea>` and a Submit button

All buttons/submit call:

```javascript
function submitHumanInput(id, value) {
  fetch('/human-input', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id, value}),
  });
}
```

When `human_input` arrives, find the card by id and replace the widget with the submitted value text.

The pending card should have a CSS class `human-pending` with a pulsing left-border animation to indicate waiting, distinct from regular act cards.

- [ ] **Step 5: Smoke-test with the example from Task 6**

Run the example from Task 6 (after writing it) with `ui=True` and verify:
- Human lifeline column appears
- Pending card shows the correct widget
- Submitting input unblocks the workflow
- Card updates to show submitted value

- [ ] **Step 6: Commit**

```bash
git add src/zipperchat/web.py
git commit -m "feat: add web human backend and ZipperChat pending card UI"
```

---

## Task 6: Example + Integration

**Files:**
- Create: `examples/human_approval.py`

- [ ] **Step 1: Write the example**

```python
"""Human approval gate example.

A Planner drafts a task description; a human reviews and approves or rejects it.
Run with:
    python examples/human_approval.py
"""

from zippergen import Lifeline, workflow
from zippergen.actions import llm, human, pure

Planner  = Lifeline("Planner")
Reviewer = Lifeline("Reviewer")   # human lifeline

@llm(
    system="You are a concise task planner.",
    user="Write a one-sentence plan for: {request}",
    parse="text",
    outputs=[("plan", str)],
)
def draft_plan(request: str): pass


@human(prompt="Approve this plan?\n\n  {plan}\n", outputs=["approved: bool"])
def review_plan(plan: str): pass


@pure
def summarise(approved: bool, plan: str) -> str:
    if approved:
        return f"Approved: {plan}"
    return "Rejected — no plan executed."


@workflow
def approval_workflow(request: str @ Planner) -> str:
    Planner: plan = draft_plan(request)
    Planner(plan) >> Reviewer(plan)
    Reviewer: approved = review_plan(plan)
    Reviewer(approved) >> Planner(approved)
    Planner: summary = summarise(approved, plan)
    return summary @ Planner


if __name__ == "__main__":
    approval_workflow.configure(llms="mock", ui=False)
    result = approval_workflow(request="organise a team offsite")
    print(result)
```

- [ ] **Step 2: Run the example (CLI mode)**

```bash
cd /Users/bollig/zippergen-io/zippergen
python examples/human_approval.py
```

Expected: the terminal shows the plan and prompts `[y/n]`. Enter `y` or `n` and verify the correct summary is printed.

- [ ] **Step 3: Run with `ui=True` (ZipperChat mode)**

Edit `__main__` block temporarily to `ui=True`, run, open `http://localhost:8765`, verify:
- Human lifeline column appears
- Pending card shows Yes/No buttons
- Clicking a button resolves the workflow

- [ ] **Step 4: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add examples/human_approval.py
git commit -m "feat: add human approval gate example"
```
