"""Causal Past Logic formula IR and user-facing API.

Formulas are used as guards in ``if cond @ Owner:`` and ``while cond @ Owner:``
workflow constructs.  The online monitor (monitor.py) evaluates them at runtime
using vector clocks and latest-value views.

Supported operators:
    atom(fn)       — atomic predicate; fn: dict -> bool, or fn: (dict, EventContext) -> bool
    on(A)          — true iff the current event is on lifeline A
    At[A](phi)     — @A(phi): latest causally visible event on A satisfies phi (non-strict)
    Y(phi)         — Prev phi: previous local event satisfies phi (strict)
    since(a, b)    — local non-strict since: a S b
    P(phi)         — strict causal-past modality
    ~phi           — negation
    phi1 & phi2    — conjunction
    phi1 | phi2    — disjunction

Field terms — read a remote lifeline's latest accessible variable value inside atom():
    atom(lambda env, ctx: ctx.field_view["A"]["x"] == env["y"])
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Union

__all__ = [
    "EventContext",
    "Formula",
    "AtomicFormula", "OnFormula", "YFormula", "AtFormula", "YAFormula",
    "SinceFormula", "PastFormula", "ConstFormula",
    "AndFormula", "OrFormula", "NotFormula",
    "AnyFormula",
    "atom", "At", "Y", "Prev", "on", "since", "P", "true", "false", "subformulas",
]


@dataclass(frozen=True)
class EventContext:
    """Metadata for the event currently being monitored.

    Atomic predicates may accept this as an optional second argument.  For
    receive events, message_vc/message_view are the metadata carried by the
    incoming message; vc is the monitor's vector clock after merging and
    counting the current event.

    field_view[B][x] is the value of variable x at the latest event of
    lifeline B causally visible to the current event.  Use this inside
    atom() to implement field terms: @B.x in the paper notation.
    """

    kind: str
    lifeline: str
    vc: Mapping[str, int]
    message_vc: Mapping[str, int] | None = None
    message_view: Mapping[str, Mapping[int, bool]] | None = None
    field_view: Mapping[str, Mapping[str, object]] | None = None


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

    def since(self, other: AnyFormula | Callable) -> SinceFormula:
        """Return ``self S other``."""
        return since(self, other)


# ---------------------------------------------------------------------------
# Formula IR nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstFormula(Formula):
    value: bool

    def __repr__(self) -> str:
        return "true" if self.value else "false"


@dataclass(frozen=True)
class AtomicFormula(Formula):
    """Atomic predicate evaluated at the current event.

    fn may be either ``fn(env)`` or ``fn(env, event_context)``.
    """
    fn: Callable[..., bool]
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
class AtFormula(Formula):
    """@A(phi): latest causally visible event on lifeline A satisfies phi.

    Non-strict: when the current event is on A, last_A = current event,
    so phi is evaluated there (not at the strictly previous A-event).
    Use Y(phi) for the strict local previous.
    """
    lifeline_name: str
    subformula: AnyFormula

    def __repr__(self) -> str:
        return f"@{self.lifeline_name!r}({self.subformula!r})"


# Backward-compatible alias.
YAFormula = AtFormula


@dataclass(frozen=True)
class SinceFormula(Formula):
    """left S right: right held now, or left has held since a previous right."""
    left: AnyFormula
    right: AnyFormula

    def __repr__(self) -> str:
        return f"({self.left!r} S {self.right!r})"


@dataclass(frozen=True)
class PastFormula(Formula):
    """Strict causal past, implemented as an internal ``true S phi`` witness."""
    subformula: AnyFormula
    witness: SinceFormula

    def __repr__(self) -> str:
        return f"P({self.subformula!r})"


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
    ConstFormula, AtomicFormula, OnFormula, YFormula, AtFormula,
    SinceFormula, PastFormula,
    AndFormula, OrFormula, NotFormula,
]


# ---------------------------------------------------------------------------
# User-facing API
# ---------------------------------------------------------------------------

def atom(fn: Callable[..., bool], src: str = "") -> AtomicFormula:
    """Wrap a Python callable as an atomic CPL predicate.

    fn  : dict -> bool, or (dict, EventContext) -> bool.
    src : optional display string for ZipperChat; defaults to fn.__name__.
    """
    return AtomicFormula(fn=fn, src=src or getattr(fn, "__name__", ""))


def on(lifeline: object) -> OnFormula:
    """Return an OnFormula for the given lifeline (Lifeline object or name string)."""
    name = lifeline.name if hasattr(lifeline, "name") else str(lifeline)  # type: ignore[union-attr]
    return OnFormula(lifeline_name=name)


def true() -> ConstFormula:
    """Formula that is true at every event."""
    return ConstFormula(True)


def false() -> ConstFormula:
    """Formula that is false at every event."""
    return ConstFormula(False)


def _as_formula(phi: AnyFormula | Callable) -> AnyFormula:
    if callable(phi) and not isinstance(phi, Formula):
        return atom(phi)
    return phi


def since(left: AnyFormula | Callable, right: AnyFormula | Callable) -> SinceFormula:
    """Return the local non-strict since formula ``left S right``."""
    return SinceFormula(_as_formula(left), _as_formula(right))


def P(phi: AnyFormula | Callable) -> PastFormula:
    """Return the strict causal-past formula for ``phi``."""
    sub = _as_formula(phi)
    return PastFormula(subformula=sub, witness=SinceFormula(true(), sub))


class _AtPartial:
    """Partial application of At[A]: call with a formula or callable to get AtFormula."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, phi: AnyFormula | Callable) -> AtFormula:
        return AtFormula(lifeline_name=self._name, subformula=_as_formula(phi))


class _AtOperator:
    """At[A](phi) — @A(phi) in the paper: latest causally visible event of A satisfies phi.

    Non-strict: when the current event is itself on A, phi is evaluated at
    the current event.  For the strict previous use Y(phi).
    """

    def __getitem__(self, lifeline: object) -> _AtPartial:
        name = lifeline.name if hasattr(lifeline, "name") else str(lifeline)  # type: ignore[union-attr]
        return _AtPartial(name)


At = _AtOperator()
"""The @A causal operator.  Use At[A](phi)."""


class _YOperator:
    """
    Y / Prev operator: Y(phi) — previous local event satisfies phi (strict).

    Y[A](phi) is kept for backward compatibility and produces AtFormula
    with non-strict semantics (same as At[A](phi)).
    """

    def __call__(self, phi: AnyFormula | Callable) -> YFormula:
        return YFormula(subformula=_as_formula(phi))

    def __getitem__(self, lifeline: object) -> _AtPartial:
        name = lifeline.name if hasattr(lifeline, "name") else str(lifeline)  # type: ignore[union-attr]
        return _AtPartial(name)


Y = _YOperator()
"""The Y / Prev temporal operator.  Use Y(phi) for the strict local previous."""

Prev = Y
"""Alias for Y: Prev(phi) is the strict local-previous operator."""


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
        if isinstance(f, (ConstFormula, AtomicFormula, OnFormula)):
            pass  # leaves — no children
        elif isinstance(f, (YFormula, NotFormula)):
            visit(f.subformula)
        elif isinstance(f, AtFormula):
            visit(f.subformula)
        elif isinstance(f, SinceFormula):
            visit(f.left)
            visit(f.right)
        elif isinstance(f, PastFormula):
            visit(f.witness)
        elif isinstance(f, (AndFormula, OrFormula)):
            visit(f.left)
            visit(f.right)
        result.append(f)

    visit(formula)
    return result
