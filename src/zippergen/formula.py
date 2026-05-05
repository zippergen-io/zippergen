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
    "AnyFormula",
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
    """Y phi: true iff the previous local event satisfies phi."""
    subformula: AnyFormula

    def __repr__(self) -> str:
        return f"Y({self.subformula!r})"


@dataclass(frozen=True)
class YAFormula(Formula):
    """Y_A phi: true iff the latest causally visible event on lifeline A satisfies phi."""
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
            phi = atom(phi)
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
            phi = atom(phi)
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
