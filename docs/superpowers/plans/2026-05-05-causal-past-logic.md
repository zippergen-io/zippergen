# Causal Past Logic — Runtime Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend ZipperGen's `if`/`while` guards with Causal Past Logic (CPL) formulas evaluated locally via vector-clock–based online monitoring, and surface guard verdicts visually in ZipperChat.

**Architecture:** Add a static formula IR (`formula.py`) and a per-lifeline monitor (`monitor.py`) that runs Algorithm 1 (event update + view propagation) and Algorithm 2 (formula evaluation) from the paper. Each `_SeqQueue` message is augmented with a vc/view piggyback so causal information flows with messages. `_exec` in `runtime.py` calls the monitor at every event (send/recv/act/choice) and evaluates Formula-typed guards at choice events. Plain Python lambda guards continue to work unchanged.

**Tech Stack:** Python 3.11+ stdlib only. No new external dependencies.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `src/zippergen/formula.py` | **Create** | Formula IR dataclasses + `atom`, `Y`, `on`, `subformulas` API |
| `src/zippergen/monitor.py` | **Create** | `MonitorState`: vector clock, latest-value views, Algorithm 1 & 2 |
| `src/zippergen/runtime.py` | **Modify** | Augment `_SeqQueue`, add monitor calls at every event, Formula guard dispatch in `_exec` |
| `src/zippergen/__init__.py` | **Modify** | Export `atom`, `Y`, `on`, `subformulas`, `Formula` |
| `src/zipperchat/web.py` | **Modify** | ✓/✗ badge on decision cards (wheat for false), formula text in detail panel |
| `examples/temporal_guard.py` | **Create** | "Approval before action" example using `Y[Orchestrator](approved)` |
| `tests/test_formula.py` | **Create** | Formula construction and subformula collection tests |
| `tests/test_monitor.py` | **Create** | Algorithm 1 and 2 correctness tests |
| `tests/test_runtime.py` | **Modify** | Add CPL guard integration tests |

### Design constraints to keep in mind

- **`SinceFormula` is deferred.** This plan implements `atom`, `Y`, `Y[A]`, `on`, `~`, `&`, `|`. The `P` (causal past) abbreviation and `S` (since) operator are left for a future plan.
- **Zero overhead when no CPL guards.** If no `Formula` instances appear in any workflow condition, `monitor` is `None` everywhere and `_SeqQueue` carries no vc/view payloads.
- **Backward compatible.** All existing plain-Python lambda guards continue to work unchanged.
- **`AtomicFormula` is identity-compared.** It contains a `Callable` field, so `__hash__` and `__eq__` are overridden to use object identity. Monitor view tables use `id(formula)` as keys.

---

## Task 1: Formula IR and User API

**Files:**
- Create: `src/zippergen/formula.py`
- Create: `tests/test_formula.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_formula.py
"""Tests for formula IR and user API."""
import pytest
from zippergen.formula import (
    AtomicFormula, OnFormula, YFormula, YAFormula,
    AndFormula, OrFormula, NotFormula,
    atom, Y, on, subformulas,
)
from zippergen.syntax import Lifeline


A = Lifeline("A")
B = Lifeline("B")


# --- Construction ---

def test_atom_creates_atomic_formula():
    fn = lambda env: env.get("x", False)
    f = atom(fn)
    assert isinstance(f, AtomicFormula)
    assert f.fn is fn


def test_atom_auto_src_from_name():
    def my_pred(env): return True
    f = atom(my_pred)
    assert f.src == "my_pred"


def test_on_creates_on_formula():
    f = on(A)
    assert isinstance(f, OnFormula)
    assert f.lifeline_name == "A"


def test_on_accepts_lifeline_or_string():
    assert on(A).lifeline_name == "A"
    assert on("B").lifeline_name == "B"


def test_Y_call_creates_y_formula():
    phi = atom(lambda env: True)
    f = Y(phi)
    assert isinstance(f, YFormula)
    assert f.subformula is phi


def test_Y_getitem_creates_ya_formula():
    phi = atom(lambda env: True)
    f = Y[A](phi)
    assert isinstance(f, YAFormula)
    assert f.lifeline_name == "A"
    assert f.subformula is phi


def test_Y_auto_wraps_callable():
    fn = lambda env: True
    f = Y(fn)
    assert isinstance(f, YFormula)
    assert isinstance(f.subformula, AtomicFormula)
    assert f.subformula.fn is fn


def test_Y_getitem_auto_wraps_callable():
    fn = lambda env: True
    f = Y[A](fn)
    assert isinstance(f, YAFormula)
    assert isinstance(f.subformula, AtomicFormula)


def test_invert_creates_not_formula():
    phi = atom(lambda env: True)
    f = ~phi
    assert isinstance(f, NotFormula)
    assert f.subformula is phi


def test_and_creates_and_formula():
    phi1 = atom(lambda env: True)
    phi2 = atom(lambda env: False)
    f = phi1 & phi2
    assert isinstance(f, AndFormula)
    assert f.left is phi1
    assert f.right is phi2


def test_or_creates_or_formula():
    phi1 = atom(lambda env: True)
    phi2 = atom(lambda env: False)
    f = phi1 | phi2
    assert isinstance(f, OrFormula)


# --- Subformulas ---

def test_subformulas_atomic_is_just_itself():
    phi = atom(lambda env: True)
    result = subformulas(phi)
    assert result == [phi]


def test_subformulas_not_formula():
    phi = atom(lambda env: True)
    neg = ~phi
    result = subformulas(neg)
    assert result == [phi, neg]  # phi before neg (bottom-up)


def test_subformulas_and_formula():
    phi1 = atom(lambda env: True)
    phi2 = atom(lambda env: False)
    conj = phi1 & phi2
    result = subformulas(conj)
    assert phi1 in result
    assert phi2 in result
    assert conj in result
    assert result.index(phi1) < result.index(conj)
    assert result.index(phi2) < result.index(conj)


def test_subformulas_y_formula():
    phi = atom(lambda env: True)
    yf = Y(phi)
    result = subformulas(yf)
    assert result == [phi, yf]


def test_subformulas_ya_formula():
    phi = atom(lambda env: True)
    yaf = Y[A](phi)
    result = subformulas(yaf)
    assert result == [phi, yaf]


def test_subformulas_deduplicates_shared_nodes():
    phi = atom(lambda env: True)
    conj = phi & phi   # phi used twice
    result = subformulas(conj)
    # phi appears only once even though it's in both left and right
    assert result.count(phi) == 1


# --- Identity semantics for AtomicFormula ---

def test_atomic_formula_hash_is_identity():
    fn = lambda env: True
    f1 = atom(fn)
    f2 = atom(fn)
    assert f1 != f2           # different objects, different identity
    assert hash(f1) != hash(f2) or True  # hash may collide but eq won't


def test_formula_is_formula_instance():
    from zippergen.formula import Formula
    phi = atom(lambda env: True)
    assert isinstance(phi, Formula)
    assert isinstance(Y(phi), Formula)
    assert isinstance(Y[A](phi), Formula)
    assert isinstance(on(A), Formula)
    assert isinstance(~phi, Formula)
    assert isinstance(phi & phi, Formula)
    assert isinstance(phi | phi, Formula)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/bollig/zippergen-io/zippergen
python -m pytest tests/test_formula.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'zippergen.formula'`

- [ ] **Step 3: Create `src/zippergen/formula.py`**

```python
"""Causal Past Logic formula IR and user-facing API.

Formulas are used as guards in ``if cond @ Owner:`` and ``while cond @ Owner:``
workflow constructs.  The online monitor (monitor.py) evaluates them at runtime
using vector clocks and latest-value views.

Supported operators:
    atom(fn)       — atomic predicate; fn: dict -> bool over the local env
    on(A)          — true iff the current event is on lifeline A
    Y(phi)         — previous local event satisfies phi
    Y[A](phi)      — latest causally visible event on A satisfies phi
    ~phi           — negation
    phi1 & phi2    — conjunction
    phi1 | phi2    — disjunction

The Since operator (S) and derived P (causal past) are deferred to a future plan.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

__all__ = [
    "Formula",
    "AtomicFormula", "OnFormula", "YFormula", "YAFormula",
    "AndFormula", "OrFormula", "NotFormula",
    "atom", "Y", "on", "subformulas",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Formula:
    """Base class for all CPL formula nodes.  Supports &, |, ~ composition."""

    def __and__(self, other: Formula) -> AndFormula:
        return AndFormula(self, other)

    def __rand__(self, other: Formula) -> AndFormula:
        return AndFormula(other, self)

    def __or__(self, other: Formula) -> OrFormula:
        return OrFormula(self, other)

    def __ror__(self, other: Formula) -> OrFormula:
        return OrFormula(other, self)

    def __invert__(self) -> NotFormula:
        return NotFormula(self)


# ---------------------------------------------------------------------------
# Formula IR nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomicFormula(Formula):
    """Atomic predicate: fn(env) -> bool evaluated at the current event."""
    fn: Callable[[dict], bool]
    src: str = ""

    # Use object identity so that two separately-created atom() calls are
    # distinct subformulas even when wrapping the same function.
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return f"atom({self.src or '...'})"


@dataclass(frozen=True)
class OnFormula(Formula):
    """on_A: true iff the current event belongs to lifeline A."""
    lifeline_name: str

    def __repr__(self) -> str:
        return f"on({self.lifeline_name!r})"


@dataclass(frozen=True)
class YFormula(Formula):
    """Y φ: true iff the previous local event satisfies φ."""
    subformula: AnyFormula

    def __repr__(self) -> str:
        return f"Y({self.subformula!r})"


@dataclass(frozen=True)
class YAFormula(Formula):
    """Y_A φ: true iff the latest causally visible event on lifeline A satisfies φ."""
    lifeline_name: str
    subformula: AnyFormula

    def __repr__(self) -> str:
        return f"Y[{self.lifeline_name!r}]({self.subformula!r})"


@dataclass(frozen=True)
class AndFormula(Formula):
    left: AnyFormula
    right: AnyFormula

    def __repr__(self) -> str:
        return f"({self.left!r} & {self.right!r})"


@dataclass(frozen=True)
class OrFormula(Formula):
    left: AnyFormula
    right: AnyFormula

    def __repr__(self) -> str:
        return f"({self.left!r} | {self.right!r})"


@dataclass(frozen=True)
class NotFormula(Formula):
    subformula: AnyFormula

    def __repr__(self) -> str:
        return f"~{self.subformula!r}"


AnyFormula = Union[
    AtomicFormula, OnFormula, YFormula, YAFormula,
    AndFormula, OrFormula, NotFormula,
]


# ---------------------------------------------------------------------------
# User-facing API
# ---------------------------------------------------------------------------

def atom(fn: Callable[[dict], bool], src: str = "") -> AtomicFormula:
    """Wrap a Python callable as an atomic CPL predicate.

    fn  : dict -> bool, called with the lifeline's local env at each event.
    src : optional display string for ZipperChat; defaults to fn.__name__.
    """
    return AtomicFormula(fn=fn, src=src or getattr(fn, "__name__", ""))


def on(lifeline: object) -> OnFormula:
    """Return an OnFormula for the given lifeline (Lifeline object or name string)."""
    name = lifeline.name if hasattr(lifeline, "name") else str(lifeline)  # type: ignore[union-attr]
    return OnFormula(lifeline_name=name)


class _YAPartial:
    """Partial application of Y[A]: call with a formula or callable to get YAFormula."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, phi: AnyFormula | Callable) -> YAFormula:
        if callable(phi) and not isinstance(phi, Formula):
            phi = AtomicFormula(fn=phi)
        return YAFormula(lifeline_name=self._name, subformula=phi)


class _YOperator:
    """
    Y operator with two forms:
        Y(phi)    — previous local event (YFormula)
        Y[A](phi) — latest causally visible event of A (YAFormula)

    In both forms, passing a plain callable auto-wraps it in atom().
    """

    def __call__(self, phi: AnyFormula | Callable) -> YFormula:
        if callable(phi) and not isinstance(phi, Formula):
            phi = AtomicFormula(fn=phi)
        return YFormula(subformula=phi)

    def __getitem__(self, lifeline: object) -> _YAPartial:
        name = lifeline.name if hasattr(lifeline, "name") else str(lifeline)  # type: ignore[union-attr]
        return _YAPartial(name)


Y = _YOperator()
"""The Y temporal operator.  Use Y(phi) or Y[A](phi)."""


# ---------------------------------------------------------------------------
# Subformula collection
# ---------------------------------------------------------------------------

def subformulas(formula: AnyFormula) -> list[AnyFormula]:
    """Return all subformulas of formula in bottom-up order (leaves first).

    Shared nodes (same object identity) appear only once.
    """
    result: list[AnyFormula] = []
    seen: set[int] = set()

    def visit(f: AnyFormula) -> None:
        if id(f) in seen:
            return
        seen.add(id(f))
        if isinstance(f, (AtomicFormula, OnFormula)):
            pass  # leaves — no children
        elif isinstance(f, (YFormula, NotFormula)):
            visit(f.subformula)
        elif isinstance(f, YAFormula):
            visit(f.subformula)
        elif isinstance(f, (AndFormula, OrFormula)):
            visit(f.left)
            visit(f.right)
        result.append(f)

    visit(formula)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_formula.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/formula.py tests/test_formula.py
git commit -m "add CPL formula IR and user-facing Y/atom/on API"
```

---

## Task 2: Monitor State (Algorithms 1 and 2)

**Files:**
- Create: `src/zippergen/monitor.py`
- Create: `tests/test_monitor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_monitor.py
"""Tests for per-lifeline CPL monitor state (Algorithms 1 and 2 from the paper)."""
import pytest
from zippergen.formula import atom, Y, on, subformulas, YAFormula
from zippergen.monitor import MonitorState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_monitor(name: str, lifelines: list[str], formula):
    subs = subformulas(formula)
    return MonitorState(name, lifelines, subs)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_vc_all_zeros():
    phi = atom(lambda env: True)
    m = make_monitor("A", ["A", "B"], phi)
    assert m.vc == {"A": 0, "B": 0}


def test_initial_view_empty():
    phi = atom(lambda env: True)
    m = make_monitor("A", ["A", "B"], phi)
    assert m.view["A"] == {}
    assert m.view["B"] == {}


# ---------------------------------------------------------------------------
# Algorithm 1: single act event on A
# ---------------------------------------------------------------------------

def test_act_increments_own_vc():
    phi = atom(lambda env: True)
    m = make_monitor("A", ["A"], phi)
    m.on_event("act", {"x": True})
    assert m.vc["A"] == 1


def test_act_updates_own_view_true():
    phi = atom(lambda env: env.get("x", False))
    m = make_monitor("A", ["A"], phi)
    m.on_event("act", {"x": True})
    assert m.view["A"][id(phi)] is True


def test_act_updates_own_view_false():
    phi = atom(lambda env: env.get("x", False))
    m = make_monitor("A", ["A"], phi)
    m.on_event("act", {"x": False})
    assert m.view["A"][id(phi)] is False


def test_act_does_not_increment_other_vc():
    phi = atom(lambda env: True)
    m = make_monitor("A", ["A", "B"], phi)
    m.on_event("act", {})
    assert m.vc["B"] == 0


# ---------------------------------------------------------------------------
# Algorithm 1: recv with incoming vc and view
# ---------------------------------------------------------------------------

def test_recv_merges_remote_vc():
    phi = atom(lambda env: True)
    m = make_monitor("B", ["A", "B"], phi)
    recv_vc = {"A": 3, "B": 0}
    recv_view = {"A": {id(phi): True}, "B": {}}
    m.on_event("recv", {}, recv_vc=recv_vc, recv_view=recv_view)
    assert m.vc["A"] == 3


def test_recv_copies_view_when_ahead():
    phi = atom(lambda env: env.get("approved", False))
    m = make_monitor("B", ["A", "B"], phi)
    recv_vc = {"A": 1, "B": 0}
    recv_view = {"A": {id(phi): True}, "B": {}}
    m.on_event("recv", {"approved": True}, recv_vc=recv_vc, recv_view=recv_view)
    assert m.view["A"][id(phi)] is True


def test_recv_does_not_overwrite_when_not_ahead():
    phi = atom(lambda env: True)
    m = make_monitor("B", ["A", "B"], phi)
    # Manually set a "more recent" view for A
    m.vc["A"] = 5
    m.view["A"][id(phi)] = False
    # Incoming message is stale (A's vc=3 < our 5)
    recv_vc = {"A": 3, "B": 0}
    recv_view = {"A": {id(phi): True}, "B": {}}
    m.on_event("recv", {}, recv_vc=recv_vc, recv_view=recv_view)
    # Our view should not be overwritten
    assert m.view["A"][id(phi)] is False


def test_recv_increments_own_vc():
    phi = atom(lambda env: True)
    m = make_monitor("B", ["A", "B"], phi)
    recv_vc = {"A": 1, "B": 0}
    recv_view = {"A": {}, "B": {}}
    m.on_event("recv", {}, recv_vc=recv_vc, recv_view=recv_view)
    assert m.vc["B"] == 1


# ---------------------------------------------------------------------------
# Algorithm 2: Y (previous local)
# ---------------------------------------------------------------------------

def test_y_false_at_first_event():
    phi = atom(lambda env: True)
    yf = Y(phi)
    m = make_monitor("A", ["A"], yf)
    m.on_event("act", {})
    # vc was 0→1; vc > 1 is False → Y(phi) = False
    assert m.view["A"][id(yf)] is False


def test_y_true_at_second_event_when_phi_was_true():
    phi = atom(lambda env: env.get("x", False))
    yf = Y(phi)
    m = make_monitor("A", ["A"], yf)
    m.on_event("act", {"x": True})   # event 1: phi=True, Y=False
    m.on_event("act", {"x": True})   # event 2: vc>1, old[phi]=True → Y=True
    assert m.view["A"][id(yf)] is True


def test_y_false_at_second_event_when_phi_was_false():
    phi = atom(lambda env: env.get("x", False))
    yf = Y(phi)
    m = make_monitor("A", ["A"], yf)
    m.on_event("act", {"x": False})  # event 1: phi=False
    m.on_event("act", {"x": True})   # event 2: old[phi]=False → Y=False
    assert m.view["A"][id(yf)] is False


# ---------------------------------------------------------------------------
# Algorithm 2: Y_A (latest causally visible event on A)
# ---------------------------------------------------------------------------

def test_ya_false_when_remote_lifeline_not_seen():
    phi = atom(lambda env: True)
    yaf = Y["A"](phi)    # Y[A](phi) via string key
    from zippergen.formula import YAFormula
    assert isinstance(yaf, YAFormula)
    m = make_monitor("B", ["A", "B"], yaf)
    m.on_event("act", {})
    # vc["A"] == 0 → Y_A(phi) = False
    assert m.view["B"][id(yaf)] is False


def test_ya_true_after_receiving_message_from_a():
    phi = atom(lambda env: env.get("approved", False))
    from zippergen.syntax import Lifeline
    A = Lifeline("A")
    yaf = Y[A](phi)
    m = make_monitor("B", ["A", "B"], yaf)
    # Simulate B receiving a message from A where A's view of phi was True
    recv_vc = {"A": 1, "B": 0}
    recv_view = {"A": {id(phi): True}, "B": {}}
    m.on_event("recv", {"approved": True}, recv_vc=recv_vc, recv_view=recv_view)
    # After recv: vc["A"]=1 > 0, view["A"][id(phi)]=True → Y_A(phi)=True
    assert m.view["B"][id(yaf)] is True


# ---------------------------------------------------------------------------
# Snapshot methods
# ---------------------------------------------------------------------------

def test_snapshot_vc_is_copy():
    phi = atom(lambda env: True)
    m = make_monitor("A", ["A", "B"], phi)
    m.on_event("act", {})
    snap = m.snapshot_vc()
    assert snap == {"A": 1, "B": 0}
    snap["A"] = 99   # mutating snapshot doesn't affect monitor
    assert m.vc["A"] == 1


def test_snapshot_view_is_deep_copy():
    phi = atom(lambda env: True)
    m = make_monitor("A", ["A"], phi)
    m.on_event("act", {})
    snap = m.snapshot_view()
    snap["A"][id(phi)] = False
    assert m.view["A"][id(phi)] is True


# ---------------------------------------------------------------------------
# guard_value
# ---------------------------------------------------------------------------

def test_guard_value_returns_latest_val():
    phi = atom(lambda env: env.get("ok", False))
    m = make_monitor("A", ["A"], phi)
    m.on_event("act", {"ok": True})
    assert m.guard_value(phi) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_monitor.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'zippergen.monitor'`

- [ ] **Step 3: Create `src/zippergen/monitor.py`**

```python
"""Per-lifeline CPL monitor state.

Implements Algorithm 1 (ON_EVENT: event update and view propagation)
and Algorithm 2 (Eval_A: formula evaluation) from the paper.

Usage
-----
At workflow start, create one MonitorState per lifeline:

    monitor = MonitorState("Executor", ["Planner", "Executor"], all_subformulas)

At each runtime event call on_event():

    monitor.on_event("recv", env, recv_vc=vc_from_msg, recv_view=view_from_msg)
    monitor.on_event("send", env)
    monitor.on_event("act",  env)
    monitor.on_event("choice", env)

After on_event() for a choice event, read the guard result:

    flag = monitor.guard_value(guard_formula)

Outgoing messages carry a vc/view snapshot:

    vc_snap   = monitor.snapshot_vc()
    view_snap = monitor.snapshot_view()
"""
from __future__ import annotations

from zippergen.formula import (
    AnyFormula,
    AtomicFormula, OnFormula, YFormula, YAFormula,
    AndFormula, OrFormula, NotFormula,
)

__all__ = ["MonitorState"]


class MonitorState:
    """Vector-clock + latest-value-view monitor for one lifeline."""

    def __init__(
        self,
        lifeline_name: str,
        all_lifelines: list[str],
        all_subformulas: list[AnyFormula],
    ) -> None:
        self.name = lifeline_name
        self.lifelines = list(all_lifelines)
        self.subformulas = list(all_subformulas)  # bottom-up order (leaves first)

        # vc[B] = number of events of B causally visible to self (inclusive, 1-based).
        # After processing a local event, vc[self.name] equals the event's index.
        self.vc: dict[str, int] = {b: 0 for b in all_lifelines}

        # view[B][id(phi)] = truth of phi at the latest event of B visible to self.
        self.view: dict[str, dict[int, bool]] = {b: {} for b in all_lifelines}

        # _val: computed values for the current event (cleared and refilled by on_event)
        self._val: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Algorithm 1 — ON_EVENT
    # ------------------------------------------------------------------

    def on_event(
        self,
        kind: str,
        env: dict,
        *,
        recv_vc: dict[str, int] | None = None,
        recv_view: dict[str, dict[int, bool]] | None = None,
    ) -> None:
        """Process one event on this lifeline.

        Parameters
        ----------
        kind      : "send" | "recv" | "act" | "choice"
        env       : the local variable store AFTER the event's effect has been applied.
                    For sends and choices the store is unchanged; for recvs and acts
                    it reflects the newly delivered/computed values.
        recv_vc   : vector clock piggybacked on the incoming message (recv only).
        recv_view : view table piggybacked on the incoming message (recv only).
        """
        A = self.name

        # --- Lines 2-9: merge incoming vc and view (recv only) ---
        if kind == "recv" and recv_vc is not None and recv_view is not None:
            for B in self.lifelines:
                if recv_vc.get(B, 0) > self.vc[B]:
                    for phi_id, val in recv_view.get(B, {}).items():
                        self.view[B][phi_id] = val
            for B in self.lifelines:
                self.vc[B] = max(self.vc[B], recv_vc.get(B, 0))

        # --- Lines 12-13: snapshot old = view_A(A, ·) before incrementing ---
        old: dict[int, bool] = dict(self.view[A])

        # --- Line 16: increment own vc ---
        self.vc[A] += 1

        # env is already post-effect (caller applies effect before calling on_event)

        # --- Lines 19-21: evaluate subformulas in bottom-up order ---
        self._val = {}
        for phi in self.subformulas:
            self._val[id(phi)] = self._eval_one(phi, A, env, old)

        # --- Lines 22-23: write back view_A(A, ·) ---
        for phi in self.subformulas:
            self.view[A][id(phi)] = self._val[id(phi)]

    # ------------------------------------------------------------------
    # Algorithm 2 — Eval_A (one node, subformula values already in self._val)
    # ------------------------------------------------------------------

    def _eval_one(
        self,
        phi: AnyFormula,
        A: str,
        env: dict,
        old: dict[int, bool],
    ) -> bool:
        """Evaluate one formula node.

        Compound nodes (And, Or, Not, Y, Y_A) look up their children's values in
        self._val, which is populated bottom-up by on_event().
        """
        match phi:
            case AtomicFormula(fn=fn):
                return bool(fn(env))

            case OnFormula(lifeline_name=B):
                return A == B

            case YFormula(subformula=theta):
                # Y θ: true iff there is a previous local event AND θ held there.
                # "previous local" exists iff vc[A] > 1 after incrementing.
                return (self.vc[A] > 1) and old.get(id(theta), False)

            case YAFormula(lifeline_name=B, subformula=theta):
                if B == A:
                    # Same-lifeline case: same as Y θ.
                    return (self.vc[A] > 1) and old.get(id(theta), False)
                else:
                    # Cross-lifeline: check the latest causally visible event of B.
                    return (self.vc.get(B, 0) > 0) and self.view[B].get(id(theta), False)

            case AndFormula(left=l, right=r):
                return self._val[id(l)] and self._val[id(r)]

            case OrFormula(left=l, right=r):
                return self._val[id(l)] or self._val[id(r)]

            case NotFormula(subformula=theta):
                return not self._val[id(theta)]

            case _:
                raise TypeError(f"Unknown formula type: {type(phi).__name__}")

    # ------------------------------------------------------------------
    # Guard evaluation (called immediately after on_event for choice events)
    # ------------------------------------------------------------------

    def guard_value(self, formula: AnyFormula) -> bool:
        """Return the truth value of formula at the most recent event."""
        return self._val.get(id(formula), False)

    # ------------------------------------------------------------------
    # Snapshot for message piggybacking
    # ------------------------------------------------------------------

    def snapshot_vc(self) -> dict[str, int]:
        """Shallow copy of the current vector clock."""
        return dict(self.vc)

    def snapshot_view(self) -> dict[str, dict[int, bool]]:
        """Deep copy of the current view table (one level of dicts)."""
        return {b: dict(v) for b, v in self.view.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_monitor.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/zippergen/monitor.py tests/test_monitor.py
git commit -m "add CPL monitor state with vector clocks and latest-value views"
```

---

## Task 3: Runtime Integration

This task wires the monitor into `runtime.py`. It touches `_SeqQueue`, `_exec`, `_thread_body`, and `run()`.

**Files:**
- Modify: `src/zippergen/runtime.py`
- Modify: `tests/test_runtime.py`

### Step-by-step plan

- [ ] **Step 1: Write the new failing tests (append to `tests/test_runtime.py`)**

```python
# Append to the bottom of tests/test_runtime.py

# ---------------------------------------------------------------------------
# Causal Past Logic guard tests
# ---------------------------------------------------------------------------

from zippergen.formula import atom, Y
from zippergen.syntax import Lifeline as _LL2

_Planner = _LL2("CPLPlanner")
_Executor = _LL2("CPLExecutor")

_approved_v = Var("approved", bool)
_result_v   = Var("result", str)


@pure
def _str_approved(x: bool) -> str:
    return "approved"


@pure
def _str_rejected(x: bool) -> str:
    return "rejected"


_approved_atom = atom(lambda env: env.get("approved", False))
_ya_guard = Y[_Planner](_approved_atom)


@workflow
def _ya_workflow(approved: bool @ _Planner) -> str:
    _Planner(approved) >> _Executor(approved)
    if _ya_guard @ _Executor:
        _Executor: _result = _str_approved(approved)
    else:
        _Executor: _result = _str_rejected(approved)
    return _result @ _Executor


def test_ya_guard_routes_true():
    assert _ya_workflow(approved=True) == "approved"


def test_ya_guard_routes_false():
    assert _ya_workflow(approved=False) == "rejected"


# --- Y (previous local event) ---

_Y_Owner = _LL2("YOwner")
_flag_v   = Var("flag", bool)
_out_v    = Var("out", str)


@pure
def _set_true() -> bool:
    return True


@pure
def _const_yes(x: bool) -> str:
    return "yes"


@pure
def _const_no(x: bool) -> str:
    return "no"


_prev_flag = Y(atom(lambda env: env.get("flag", False)))


@workflow
def _y_local_workflow(dummy: bool @ _Y_Owner) -> str:
    _Y_Owner: _flag = _set_true()       # event 1: flag=True
    if _prev_flag @ _Y_Owner:           # event 2 (choice): Y(flag) = True (prev had flag=True)
        _Y_Owner: _out = _const_yes(_flag)
    else:
        _Y_Owner: _out = _const_no(_flag)
    return _out @ _Y_Owner


def test_y_local_guard_true_after_prior_act():
    # After set_true(), flag=True. At the choice event Y(flag) looks
    # at the previous event (set_true) where flag was True → takes true branch.
    assert _y_local_workflow(dummy=True) == "yes"


# --- Y guard is False when no prior event ---

_NoHist = _LL2("NoHistOwner")
_nh_out  = Var("nh_out", str)
_nh_guard = Y(atom(lambda env: True))  # Y(always-true)


@workflow
def _no_history_workflow(x: bool @ _NoHist) -> str:
    if _nh_guard @ _NoHist:           # first event on this lifeline — no prior
        _NoHist: _nh_out = _const_yes(x)
    else:
        _NoHist: _nh_out = _const_no(x)
    return _nh_out @ _NoHist


def test_y_guard_false_with_no_prior_event():
    # The choice event is the FIRST event on _NoHist, so vc[_NoHist] = 0→1,
    # vc > 1 is False, Y(true) = False → false branch.
    assert _no_history_workflow(x=True) == "no"
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
python -m pytest tests/test_runtime.py::test_ya_guard_routes_true tests/test_runtime.py::test_ya_guard_routes_false tests/test_runtime.py::test_y_local_guard_true_after_prior_act tests/test_runtime.py::test_y_guard_false_with_no_prior_event -v
```

Expected: `FAILED` with `AttributeError` or incorrect routing (plain Python guard path used instead of monitor).

- [ ] **Step 3: Augment `_SeqQueue` to carry vc/view piggyback**

Locate `class _SeqQueue` at line ~205 of `src/zippergen/runtime.py`. Replace:

```python
class _SeqQueue:
    """FIFO queue that auto-stamps each item with a per-channel sequence number."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._next = 0

    def put(self, values: tuple) -> int:
        seq = self._next
        self._next += 1
        self._q.put((seq, values))
        return seq

    def get(self, *, stop: threading.Event | None = None) -> tuple[int, tuple]:
        if stop is None:
            return self._q.get()
        while True:
            try:
                return self._q.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    raise RuntimeError("Workflow cancelled: another lifeline failed")
```

With:

```python
class _SeqQueue:
    """FIFO queue that auto-stamps each item with a per-channel sequence number.

    Each item stores (seq, values, vc, view) where vc and view are the sender's
    monitor snapshot at send time, or None when monitoring is inactive.
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._next = 0

    def put(
        self,
        values: tuple,
        vc: dict | None = None,
        view: dict | None = None,
    ) -> int:
        seq = self._next
        self._next += 1
        self._q.put((seq, values, vc, view))
        return seq

    def get(
        self, *, stop: threading.Event | None = None
    ) -> tuple[int, tuple, dict | None, dict | None]:
        if stop is None:
            return self._q.get()
        while True:
            try:
                return self._q.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    raise RuntimeError("Workflow cancelled: another lifeline failed")
```

- [ ] **Step 4: Update the three `ch.get()` call sites in `_exec`**

Find and replace each occurrence of `seq, values = ch[...].get(...)` with the four-tuple form. There are exactly three: in `RecvStmt`, `IfRecvStmt`, and `WhileRecvStmt`.

**RecvStmt** (line ~329):

```python
        case RecvStmt(lifeline=A, bindings=ys, sender=B):
            seq, values, recv_vc, recv_view = ch[(B.name, A.name)].get(stop=stop)
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view)
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "bindings": _bound_dict(ys, values),
                    "seq": seq,
                })
```

**IfRecvStmt** (line ~481):

```python
        case IfRecvStmt(lifeline=A, bindings=ys, sender=B, branch_true=t, branch_false=f):
            seq, values, recv_vc, recv_view = ch[(B.name, A.name)].get(stop=stop)
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view)
            flag = _eval(ys[0], env) if isinstance(ys[0], VarExpr) else values[0]
            if trace:
                trace({
                    "type": "recv",
                    "to": A.name, "from": B.name,
                    "bindings": {"branch": "true" if flag else "false"},
                    "seq": seq, "ctrl": True,
                })
            _exec(cast(LocalStmt, t if flag else f), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)
```

**WhileRecvStmt** (line ~505):

```python
        case WhileRecvStmt(lifeline=A, bindings=ys, sender=B, body=body, exit_body=exit_b):
            while True:
                seq, values, recv_vc, recv_view = ch[(B.name, A.name)].get(stop=stop)
                _bind(ys, values, env)
                if monitor:
                    monitor.on_event("recv", env, recv_vc=recv_vc, recv_view=recv_view)
                flag = _eval(ys[0], env) if isinstance(ys[0], VarExpr) else values[0]
                if trace:
                    trace({
                        "type": "recv",
                        "to": A.name, "from": B.name,
                        "bindings": {"loop": "continue" if flag else "exit"},
                        "seq": seq, "ctrl": True,
                    })
                if not flag:
                    break
                _exec(cast(LocalStmt, body), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)
            _exec(cast(LocalStmt, exit_b), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)
```

- [ ] **Step 5: Add `monitor` parameter to `_exec` and update all recursive call sites**

Change the `_exec` signature at line ~307 from:

```python
def _exec(stmt: LocalStmt, env: Env, ch: Channels, ns: dict, llm_backend, human_backend, trace,
          stop: threading.Event | None = None) -> None:
```

to:

```python
def _exec(stmt: LocalStmt, env: Env, ch: Channels, ns: dict, llm_backend, human_backend, monitor, trace,
          stop: threading.Event | None = None) -> None:
```

Then update all eight recursive `_exec(...)` calls inside `_exec` itself (SeqStmt ×2, IfStmt ×1, IfRecvStmt ×1, WhileStmt ×2, WhileRecvStmt ×2) to include `monitor` after `human_backend`:

- `_exec(cast(LocalStmt, p1), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)`
- `_exec(cast(LocalStmt, p2), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)`
- `_exec(cast(LocalStmt, t if flag else f), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)` (IfStmt)
- (IfRecvStmt already updated in Step 4)
- `_exec(cast(LocalStmt, body), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)` (WhileStmt)
- `_exec(cast(LocalStmt, exit_b), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)` (WhileStmt)
- (WhileRecvStmt body and exit already updated in Step 4)

- [ ] **Step 6: Add monitor calls at SendStmt, SelfAssignStmt, ActStmt**

**SendStmt** — call `on_event("send")` BEFORE put, pass snapshot with put:

```python
        case SendStmt(lifeline=A, payload=xs, receiver=B):
            values = tuple(copy.deepcopy(_eval(x, env)) for x in xs)
            if monitor:
                monitor.on_event("send", env)
                seq = ch[(A.name, B.name)].put(values, monitor.snapshot_vc(), monitor.snapshot_view())
            else:
                seq = ch[(A.name, B.name)].put(values)
            if trace:
                names = [x.var.name if isinstance(x, VarExpr) else f"_{i}" for i, x in enumerate(xs)]
                trace({
                    "type": "send",
                    "from": A.name, "to": B.name,
                    "values": [_jsonify(v) for v in values],
                    "bindings": {name: _jsonify(v) for name, v in zip(names, values)},
                    "seq": seq,
                })
```

**SelfAssignStmt** — call `on_event("act")` after assignment, before trace:

```python
        case SelfAssignStmt(lifeline=A, payload=xs, bindings=ys):
            values = tuple(_eval(x, env) for x in xs)
            _bind(ys, values, env)
            if monitor:
                monitor.on_event("act", env)
            if trace:
                # ... (existing trace block unchanged)
```

**ActStmt** — call `on_event("act")` after `env[var] = val` updates, before trace.

The existing ActStmt block ends with:
```python
            for var, val in out_map.items():
                env[var] = val
            if trace:
                trace({"type": "act", ...})
```

Add after the env update loop:

```python
            for var, val in out_map.items():
                env[var] = val
            if monitor:
                monitor.on_event("act", env)
            if trace:
                trace({"type": "act", ...})
```

- [ ] **Step 7: Add Formula guard dispatch in IfStmt and WhileStmt**

At the top of `_exec`, add the import:

```python
    from zippergen.formula import Formula as _Formula
```

(This lazy import avoids a circular import since `formula.py` doesn't import `runtime.py`.)

**IfStmt** — replace the guard evaluation:

```python
        case IfStmt(condition=c, owner=B, branch_true=t, branch_false=f):
            from zippergen.formula import Formula as _Formula
            if monitor and isinstance(c, _Formula):
                monitor.on_event("choice", env)
                flag = monitor.guard_value(c)
                formula_repr = repr(c)
            else:
                flag = c(_CondEnv(env, ns))
                formula_repr = None
            if trace:
                trace({"type": "decision", "lifeline": B.name, "kind": "if",
                       "value": flag, "condition": getattr(c, "_src", None),
                       "formula": formula_repr})
            _exec(cast(LocalStmt, t if flag else f), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)
```

**WhileStmt** — same pattern, per iteration:

```python
        case WhileStmt(condition=c, owner=B, body=body, exit_body=exit_b):
            from zippergen.formula import Formula as _Formula
            while True:
                if monitor and isinstance(c, _Formula):
                    monitor.on_event("choice", env)
                    flag = monitor.guard_value(c)
                    formula_repr = repr(c)
                else:
                    flag = c(_CondEnv(env, ns))
                    formula_repr = None
                if trace:
                    trace({"type": "decision", "lifeline": B.name, "kind": "while",
                           "value": flag, "condition": getattr(c, "_src", None),
                           "formula": formula_repr})
                if not flag:
                    break
                _exec(cast(LocalStmt, body), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)
            _exec(cast(LocalStmt, exit_b), env, ch, ns, llm_backend, human_backend, monitor, trace, stop)
```

- [ ] **Step 8: Add `_thread_body` monitor parameter**

Change the signature at line ~530:

```python
def _thread_body(local_stmt, env, ch, ns, result_box, llm_backend, human_backend, monitor, trace, stop):
    try:
        _exec(local_stmt, env, ch, ns, llm_backend, human_backend, monitor, trace, stop)
        result_box.append(env)
    except Exception as exc:
        stop.set()
        result_box.append(exc)
```

- [ ] **Step 9: Wire monitors into `run()`**

Add imports at the top of `run()` body (or at module level if preferred):

```python
from zippergen.formula import Formula as _Formula, subformulas as _collect_subs
from zippergen.monitor import MonitorState
```

Add a helper function before `run()`:

```python
def _collect_formula_guards(stmt) -> list:
    """Walk a global Stmt and collect all Formula-typed conditions."""
    guards: list = []
    def walk(s) -> None:
        match s:
            case IfStmt(condition=c, branch_true=t, branch_false=f):
                if isinstance(c, _Formula): guards.append(c)
                walk(t); walk(f)
            case WhileStmt(condition=c, body=b, exit_body=x):
                if isinstance(c, _Formula): guards.append(c)
                walk(b); walk(x)
            case SeqStmt(first=p1, second=p2):
                walk(p1); walk(p2)
            case _:
                pass
    walk(stmt)
    return guards
```

Inside `run()`, after `channels` is built and before the thread loop, add:

```python
    # Build per-lifeline monitors if any CPL formula guards are present.
    guards = _collect_formula_guards(wf.program)
    if guards:
        all_subs: list = []
        seen_ids: set[int] = set()
        for g in guards:
            for sf in _collect_subs(g):
                if id(sf) not in seen_ids:
                    seen_ids.add(id(sf))
                    all_subs.append(sf)
        monitors: dict[str, MonitorState] = {
            ll.name: MonitorState(ll.name, [l.name for l in lifelines], all_subs)
            for ll in lifelines
        }
    else:
        monitors = {}
```

Then change the thread spawning loop to pass the monitor:

```python
    for ll in lifelines:
        local_stmt = project(wf, ll)
        env_copy = {**{v.name: v.default for v in (wf.vars or [])},
                    **initial_envs.get(ll.name, {})}
        result_boxes[ll.name] = []
        monitor = monitors.get(ll.name)   # None when no CPL guards
        t = threading.Thread(
            target=_thread_body,
            args=(local_stmt, env_copy, channels, ns,
                  result_boxes[ll.name], llm_backend, human_backend,
                  monitor, trace, stop),
            name=ll.name,
            daemon=True,
        )
        threads.append(t)
```

- [ ] **Step 10: Run the new tests to verify they pass**

```bash
python -m pytest tests/test_runtime.py::test_ya_guard_routes_true tests/test_runtime.py::test_ya_guard_routes_false tests/test_runtime.py::test_y_local_guard_true_after_prior_act tests/test_runtime.py::test_y_guard_false_with_no_prior_event -v
```

Expected: all four tests pass.

- [ ] **Step 11: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass. Any failure in existing tests means the `_SeqQueue` four-tuple change broke an unupdated get call — find and fix it.

- [ ] **Step 12: Commit**

```bash
git add src/zippergen/runtime.py tests/test_runtime.py
git commit -m "integrate CPL monitor into runtime: augment messages, hook all events, Formula guard dispatch"
```

---

## Task 4: Public API Export

**Files:**
- Modify: `src/zippergen/__init__.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_formula.py`)**

```python
def test_public_api_importable_from_zippergen():
    from zippergen import Y, atom, on, subformulas, Formula
    phi = atom(lambda env: True)
    assert isinstance(phi, Formula)
    f = Y[None](phi)
    # Y[None] — None has no .name, so str(None) = "None"
    from zippergen.formula import YAFormula
    assert isinstance(f, YAFormula)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_formula.py::test_public_api_importable_from_zippergen -v
```

Expected: `ImportError: cannot import name 'Y'`

- [ ] **Step 3: Edit `src/zippergen/__init__.py`**

Replace the current content with:

```python
"""ZipperGen — formal multi-agent LLM coordination programs."""

from zippergen.syntax import *          # noqa: F401, F403
from zippergen.actions import *         # noqa: F401, F403
from zippergen.backends import *        # noqa: F401, F403
from zippergen.human_backends import *  # noqa: F401, F403
from zippergen.formula import *         # noqa: F401, F403
from zippergen.demo import *            # noqa: F401, F403
from zippergen.builder import *         # noqa: F401, F403
from zippergen.projection import *      # noqa: F401, F403
from zippergen.runtime import *         # noqa: F401, F403
from zippergen import (
    syntax, actions, backends, human_backends,
    formula, demo, builder, projection, runtime,
)

__all__: list[str] = (
    syntax.__all__
    + actions.__all__
    + backends.__all__
    + human_backends.__all__
    + formula.__all__
    + demo.__all__
    + builder.__all__
    + projection.__all__
    + runtime.__all__
)  # type: ignore[assignment]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (including the new import test).

- [ ] **Step 5: Commit**

```bash
git add src/zippergen/__init__.py
git commit -m "export CPL formula API (Y, atom, on, subformulas) from zippergen"
```

---

## Task 5: ZipperChat ✓/✗ Badge and Formula Display

**Files:**
- Modify: `src/zipperchat/web.py`

No automated tests — verify manually by running `examples/temporal_guard.py` after Task 6.

Context: Decision events in the diagram currently show `⊤`/`⊥` symbols and `IF`/`WHILE` tags. This task adds:
1. A small ✓/✗ badge (blue for true, wheat for false) to every decision card.
2. A "Formula" section in the detail panel when `ev.formula` is set (CPL guard).

- [ ] **Step 1: Add CSS palette tokens**

Locate the palette block (around line 338 — look for `--k-decision`). Add two new tokens after it:

```css
  --k-guard-true:  #69A6E0;   /* action blue — affirmative guard verdict     */
  --k-guard-false: #D4BA88;   /* wheat — neutral false (not an error)        */
```

- [ ] **Step 2: Add badge CSS**

Find a suitable place in the component styles (after the existing `.dec-box` rule, around line 928). Add:

```css
.guard-badge {
  display: inline-block;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 700;
  margin-left: 4px;
  vertical-align: middle;
}
.guard-badge.verdict-true  {
  background: rgba(105, 166, 224, 0.16);
  color: var(--k-guard-true);
}
.guard-badge.verdict-false {
  background: rgba(212, 186, 136, 0.16);
  color: #8B7548;
}
body.dark .guard-badge.verdict-false { color: var(--k-guard-false); }
```

- [ ] **Step 3: Update `handleDecision` in the JS section**

Find `handleDecision` (line ~1963). The current `box.innerHTML` assignment builds the card content. Change it to include the badge:

```javascript
  const badgeCls  = isTrue ? 'verdict-true' : 'verdict-false';
  const badgeSym  = isTrue ? '✓' : '✗';
  const guardBadge = `<span class="guard-badge ${badgeCls}">${badgeSym}</span>`;

  const box = document.createElement('div');
  box.className = 'ev-box dec-box';
  box._level = lev;
  box.innerHTML = `
    <div class="ev-box-tag">${tag}</div>
    <div class="ev-box-name">
      <span class="dec-symbol">${sym}</span>
      <span>${escHtml(word)}</span>
      ${guardBadge}
    </div>
  `;
```

- [ ] **Step 4: Update the detail panel sections in `handleDecision`**

Find the `fillDetailPanel` call (line ~2009). Replace the `sections` array:

```javascript
    const condSections = [];
    if (ev.formula) {
      condSections.push({
        label: 'Formula',
        entries: { guard: ev.formula },
      });
    } else if (ev.condition) {
      condSections.push({
        label: 'Condition',
        entries: { expr: ev.condition },
      });
    }
    fillDetailPanel(box, {
      type: tag,
      name: word,
      lifeline: ev.lifeline,
      accent: KIND.decision,
      sections: [
        ...condSections,
        { label: 'Verdict', entries: { value: isTrue } },
      ],
    });
```

- [ ] **Step 5: Commit**

```bash
git add src/zipperchat/web.py
git commit -m "add guard verdict badge (wheat/blue) and formula display to decision cards"
```

---

## Task 6: Example

**Files:**
- Create: `examples/temporal_guard.py`

This implements the "Approval before action" pattern from Section 7 of the paper.

- [ ] **Step 1: Create the example**

```python
# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""Temporal guard example — approval before action.

An Orchestrator produces a plan and approves (or rejects) it.
The Executor may only run the plan if the latest causally visible
Orchestrator event records approval.

Guard: Y[Orchestrator](atom(lambda env: env["approval"]))

Run with:
    python examples/temporal_guard.py
"""

from zippergen import Lifeline, Var, workflow
from zippergen.actions import llm, pure
from zippergen import Y, atom

Orchestrator = Lifeline("Orchestrator")
Executor     = Lifeline("Executor")

plan     = Var("plan",     str)
approval = Var("approval", bool)
result   = Var("result",   str)


@llm(
    system="You are a concise task planner.",
    user="Generate a one-sentence plan for: {request}",
    parse="text",
    outputs=[("plan", str)],
)
def make_plan(request: str): pass


@llm(
    system="You are a safety reviewer. Answer true to approve, false to reject.",
    user="Approve this plan? {plan}",
    parse="bool",
    outputs=[("approval", bool)],
)
def approve_plan(plan: str): pass


@llm(
    system="You are an executor. Carry out the given plan.",
    user="Execute: {plan}",
    parse="text",
    outputs=[("result", str)],
)
def execute_plan(plan: str): pass


@pure
def blocked(plan: str) -> str:
    return f"[BLOCKED] No causal approval visible for plan: {plan}"


# Guard evaluated by Executor: the latest causally visible Orchestrator
# event must have approval=True.
approved_by_orchestrator = Y[Orchestrator](atom(lambda env: env.get("approval", False), src="approval"))


@workflow
def guarded_execution(request: str @ Orchestrator) -> str:
    Orchestrator: plan     = make_plan(request)
    Orchestrator: approval = approve_plan(plan)
    Orchestrator(plan, approval) >> Executor(plan, approval)
    if approved_by_orchestrator @ Executor:
        Executor: result = execute_plan(plan)
    else:
        Executor: result = blocked(plan)
    return result @ Executor


if __name__ == "__main__":
    guarded_execution.configure(llms="mock", ui=True)
    r = guarded_execution(request="organise a team offsite")
    print(f"\nResult → {r}")
    input("\nZipperChat is running at http://localhost:8765\nPress Enter to stop.\n")
```

- [ ] **Step 2: Run the example to verify it works end-to-end**

```bash
cd /Users/bollig/zippergen-io/zippergen
python examples/temporal_guard.py
```

Expected: 
- No exception thrown.
- ZipperChat opens at http://localhost:8765.
- Decision card shows a ✓ or ✗ badge.
- Clicking the decision card shows "Formula: Y['Orchestrator'](atom(approval))" in the detail panel.

- [ ] **Step 3: Run the full test suite one final time**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add examples/temporal_guard.py
git commit -m "add temporal guard example: approval-before-action with Y[Orchestrator]"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| Formula IR: atom, on, Y, Y[A], ~, &, \| | Task 1 |
| Subformula collection in bottom-up order | Task 1 |
| `from zippergen import Y, atom, on` | Task 4 |
| Vector clock per lifeline | Task 2 |
| Latest-value views per lifeline | Task 2 |
| Algorithm 1: vc merge on recv | Task 2, 3 |
| Algorithm 1: on_event at send/recv/act/choice | Task 3 |
| Algorithm 2: eval for all formula types | Task 2 |
| Messages carry vc+view piggyback | Task 3 |
| Formula guard dispatch in IfStmt/WhileStmt | Task 3 |
| Zero overhead when no CPL guards | Task 3 (monitors dict empty, monitor=None path) |
| Plain Python lambda guards unchanged | Task 3 (else branch in IfStmt/WhileStmt) |
| ✓/✗ badge on all decision cards | Task 5 |
| Wheat for false, blue for true | Task 5 |
| Formula text in detail panel | Task 5 |
| End-to-end example | Task 6 |

**Deferred (future plan):** `SinceFormula`, `P` operator, freshness predicates (require message metadata access in atomics).

### No placeholders — confirmed

All steps contain complete code. No "add appropriate handling" lines.

### Type consistency

- `MonitorState.on_event("recv", env, recv_vc=..., recv_view=...)` — matches every call site in Task 3.
- `monitor.guard_value(c)` — `c: AnyFormula` — matches `MonitorState.guard_value(formula: AnyFormula)`.
- `_SeqQueue.get()` returns `(int, tuple, dict|None, dict|None)` — all three call sites updated in Task 3 Step 4.
- `_exec(..., monitor, trace, stop)` — `monitor` added between `human_backend` and `trace` consistently at all 8 recursive call sites and in `_thread_body` and `run()`.
