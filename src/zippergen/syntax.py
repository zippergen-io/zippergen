"""
ZipperGen — Layer 1: Abstract syntax (IR).

Design notes
------------
**What this module is.**
This module defines the internal representation (IR) of ZipperGen coordination
programs as plain Python dataclasses. Every class here corresponds one-to-one
to a construct in the formal grammar of the paper. Nothing in this module
does anything — it only *describes* programs as data.

**Why frozen dataclasses?**
All IR nodes use ``frozen=True``, which makes them immutable and hashable.
This means two nodes with identical contents are considered equal and can be
used as dictionary keys or set members. It also prevents accidental mutation,
which matters because the same node (e.g. a shared Var or Lifeline) can appear
in many places in a program tree.

**Why tuples instead of lists?**
Tuples are immutable, consistent with the frozen philosophy. A ``MsgStmt``
that looks the same always *is* the same.

**Why is Stmt a type alias rather than a base class?**
The formal grammar defines statements by cases, not by inheritance. Using a
``Union`` type alias keeps the structure flat and matches the paper directly.
The ``match``/``case`` pattern (used in ``participation_set`` and ``pp``) then
mirrors a mathematical case analysis, which makes it easy to check against the
paper's definitions.

**Why is ZType a type alias too?**
Same reason: the paper defines a small fixed set of types. There is no need
for an open hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

__all__ = [
    # Types
    "ZType",
    "TText", "TBool", "TInt", "TFloat", "TTuple",
    "Text", "Bool", "Int", "Float",
    # Lifeline
    "Lifeline",
    # Var
    "Var",
    # Expressions
    "Expr",
    "VarExpr", "LitExpr", "NotExpr", "AndExpr", "OrExpr", "TupleExpr",
    # Actions
    "LLMAction", "PureAction",
    # Type helpers
    "is_ztype",
    # Statements
    "Stmt",
    "EmptyStmt", "MsgStmt", "ActStmt", "SkipStmt",
    "SeqStmt", "IfStmt", "WhileStmt",
    # Local-only statements (produced by projection)
    "LocalStmt",
    "SendStmt", "RecvStmt", "IfRecvStmt", "WhileRecvStmt",
    # Proc and Program
    "Proc", "Program",
    # Reserved literals
    "kappa_ctrl",
    # Helpers
    "seq", "participation_set", "pp",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TText:
    def __repr__(self) -> str:
        return "Text"


@dataclass(frozen=True)
class TBool:
    def __repr__(self) -> str:
        return "Bool"


@dataclass(frozen=True)
class TInt:
    def __repr__(self) -> str:
        return "Int"


@dataclass(frozen=True)
class TFloat:
    def __repr__(self) -> str:
        return "Float"


@dataclass(frozen=True)
class TTuple:
    elements: tuple[ZType, ...]

    def __repr__(self) -> str:
        return f"TTuple({', '.join(repr(e) for e in self.elements)})"


ZType = Union[TText, TBool, TInt, TFloat, TTuple]

# Convenient singletons
Text: TText = TText()
Bool: TBool = TBool()
Int: TInt = TInt()
Float: TFloat = TFloat()



# ---------------------------------------------------------------------------
# Lifeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lifeline:
    name: str

    def __repr__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Var
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Var:
    name: str
    type: ZType
    default: object = None  # optional Python literal default

    def __repr__(self) -> str:
        if self.default is not None:
            return f"Var({self.name!r}: {self.type!r} = {self.default!r})"
        return f"Var({self.name!r}: {self.type!r})"


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VarExpr:
    var: Var

    def __repr__(self) -> str:
        return self.var.name


@dataclass(frozen=True)
class LitExpr:
    value: object  # Python literal: str, int, bool, float
    type: ZType

    def __repr__(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class NotExpr:
    operand: Expr

    def __repr__(self) -> str:
        return f"not {self.operand!r}"


@dataclass(frozen=True)
class AndExpr:
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left!r} and {self.right!r})"


@dataclass(frozen=True)
class OrExpr:
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left!r} or {self.right!r})"


@dataclass(frozen=True)
class TupleExpr:
    elements: tuple[Expr, ...]

    def __repr__(self) -> str:
        return f"({', '.join(repr(e) for e in self.elements)})"


Expr = Union[VarExpr, LitExpr, NotExpr, AndExpr, OrExpr, TupleExpr]

# Reserved control tag — used only by the projection engine in control-broadcast
# messages (send B(⊤, κ_ctrl) → C).  Must not appear in user-written programs.
kappa_ctrl: LitExpr = LitExpr("κ_ctrl", Text)


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def is_ztype(x: object) -> bool:
    """Return True iff x is a ZipperGen type instance."""
    return isinstance(x, (TText, TBool, TInt, TFloat, TTuple))


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMAction:
    name: str
    inputs: tuple[tuple[str, ZType], ...]   # (param_name, type) pairs
    outputs: tuple[tuple[str, ZType], ...]  # (param_name, type) pairs
    system_prompt: str
    user_prompt: str        # may contain {var_name} placeholders
    parse_format: str       # "json" | "text" | "bool"

    def __repr__(self) -> str:
        ins = ", ".join(f"{n}: {t!r}" for n, t in self.inputs)
        outs = ", ".join(f"{n}: {t!r}" for n, t in self.outputs)
        return f"LLMAction({self.name!r}, ({ins}) -> ({outs}))"


@dataclass(frozen=True)
class PureAction:
    name: str
    inputs: tuple[tuple[str, ZType], ...]
    outputs: tuple[tuple[str, ZType], ...]
    fn: object  # the actual Python callable

    def __repr__(self) -> str:
        ins = ", ".join(f"{n}: {t!r}" for n, t in self.inputs)
        outs = ", ".join(f"{n}: {t!r}" for n, t in self.outputs)
        return f"PureAction({self.name!r}, ({ins}) -> ({outs}))"


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmptyStmt:
    """ε — the empty program."""

    def __repr__(self) -> str:
        return "ε"


@dataclass(frozen=True)
class MsgStmt:
    """msg sender(payload) → receiver(bindings)

    Both payload and bindings may contain variables (VarExpr) or concrete
    values (LitExpr).  In projection-generated control messages the reserved
    literal kappa_ctrl appears in both positions.
    """
    sender: Lifeline
    payload: tuple[Expr, ...]   # variables or literals sent by sender
    receiver: Lifeline
    bindings: tuple[Expr, ...]  # variables or literals bound by receiver

    def __repr__(self) -> str:
        xs = ", ".join(repr(e) for e in self.payload)
        ys = ", ".join(repr(e) for e in self.bindings)
        return f"msg {self.sender.name}({xs}) → {self.receiver.name}({ys})"


@dataclass(frozen=True)
class ActStmt:
    """act lifeline: outputs := action(inputs)"""
    lifeline: Lifeline
    action: Union[LLMAction, PureAction]
    inputs: tuple[Expr, ...]
    outputs: tuple[Var, ...]

    def __repr__(self) -> str:
        ins = ", ".join(repr(e) for e in self.inputs)
        outs = ", ".join(v.name for v in self.outputs)
        return f"act {self.lifeline.name}: {outs} := {self.action.name}({ins})"


@dataclass(frozen=True)
class SkipStmt:
    """skip lifeline — local no-op."""
    lifeline: Lifeline

    def __repr__(self) -> str:
        return f"skip {self.lifeline.name}"


@dataclass(frozen=True)
class SeqStmt:
    """P1 ; P2 — sequential composition."""
    first: Stmt
    second: Stmt

    def __repr__(self) -> str:
        return f"({self.first!r} ; {self.second!r})"


@dataclass(frozen=True)
class IfStmt:
    """if condition@owner then branch_true else branch_false"""
    condition: Expr
    owner: Lifeline
    branch_true: Stmt
    branch_false: Stmt

    def __repr__(self) -> str:
        return (
            f"if {self.condition!r}@{self.owner.name} "
            f"then {self.branch_true!r} "
            f"else {self.branch_false!r}"
        )


@dataclass(frozen=True)
class WhileStmt:
    """while condition@owner do { body } exit { exit_body }"""
    condition: Expr
    owner: Lifeline
    body: Stmt
    exit_body: Stmt

    def __repr__(self) -> str:
        return (
            f"while {self.condition!r}@{self.owner.name} "
            f"{{ {self.body!r} }} "
            f"exit {{ {self.exit_body!r} }}"
        )


Stmt = Union[EmptyStmt, MsgStmt, ActStmt, SkipStmt, SeqStmt, IfStmt, WhileStmt]


# ---------------------------------------------------------------------------
# Local-only statements  (produced by the projection engine, Layer 4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SendStmt:
    """send A(x⃗) → B — sender's local view of a message."""
    lifeline: Lifeline
    payload: tuple[Expr, ...]
    receiver: Lifeline

    def __repr__(self) -> str:
        xs = ", ".join(repr(e) for e in self.payload)
        return f"send {self.lifeline.name}({xs}) → {self.receiver.name}"


@dataclass(frozen=True)
class RecvStmt:
    """recv A(y⃗) ← B — receiver's local view of a message."""
    lifeline: Lifeline
    bindings: tuple[Expr, ...]
    sender: Lifeline

    def __repr__(self) -> str:
        ys = ", ".join(repr(e) for e in self.bindings)
        return f"recv {self.lifeline.name}({ys}) ← {self.sender.name}"


@dataclass(frozen=True)
class IfRecvStmt:
    """if A(y⃗) ← B then branch_true else branch_false

    A receives y⃗ from B.  bindings[0] is a Bool variable; its runtime value
    determines which branch is taken.
    """
    lifeline: Lifeline
    bindings: tuple[Expr, ...]   # bindings[0] is the Bool control variable
    sender: Lifeline
    branch_true: LocalStmt
    branch_false: LocalStmt

    def __repr__(self) -> str:
        ys = ", ".join(repr(e) for e in self.bindings)
        return (
            f"if {self.lifeline.name}({ys}) ← {self.sender.name} "
            f"then {self.branch_true!r} "
            f"else {self.branch_false!r}"
        )


@dataclass(frozen=True)
class WhileRecvStmt:
    """while A(y⃗) ← B do body exit exit_body

    Each iteration A receives y⃗ from B.  bindings[0] is a Bool variable;
    True means continue the body, False means take the exit.
    """
    lifeline: Lifeline
    bindings: tuple[Expr, ...]   # bindings[0] is the Bool control variable
    sender: Lifeline
    body: LocalStmt
    exit_body: LocalStmt

    def __repr__(self) -> str:
        ys = ", ".join(repr(e) for e in self.bindings)
        return (
            f"while {self.lifeline.name}({ys}) ← {self.sender.name} "
            f"{{ {self.body!r} }} "
            f"exit {{ {self.exit_body!r} }}"
        )


LocalStmt = Union[
    EmptyStmt, SendStmt, RecvStmt, ActStmt, SkipStmt,
    SeqStmt, IfStmt, WhileStmt, IfRecvStmt, WhileRecvStmt,
]


# ---------------------------------------------------------------------------
# Proc and Program
# ---------------------------------------------------------------------------

@dataclass
class Proc:
    name: str
    inputs: tuple[tuple[str, ZType], ...]
    output_type: ZType
    vars: tuple[Var, ...]       # locally declared variables
    body: Stmt

    def __repr__(self) -> str:
        ins = ", ".join(f"{n}: {t!r}" for n, t in self.inputs)
        return f"Proc({self.name!r}, ({ins}) -> {self.output_type!r})"


@dataclass
class Program:
    lifelines: tuple[Lifeline, ...]
    actions: tuple[Union[LLMAction, PureAction], ...]
    procs: tuple[Proc, ...]

    def __repr__(self) -> str:
        ls = ", ".join(repr(l) for l in self.lifelines)
        ps = ", ".join(p.name for p in self.procs)
        return f"Program(lifelines=[{ls}], procs=[{ps}])"


# ---------------------------------------------------------------------------
# seq — right-associative fold with EmptyStmt identity
# ---------------------------------------------------------------------------

def seq(*stmts: Stmt) -> Stmt:
    """
    Right-associative sequential composition.

    seq()           → EmptyStmt()
    seq(s)          → s
    seq(s1, s2, s3) → SeqStmt(s1, SeqStmt(s2, s3))

    EmptyStmt() operands are dropped: seq(ε, s) = seq(s, ε) = s.
    """
    non_empty = [s for s in stmts if not isinstance(s, EmptyStmt)]
    if not non_empty:
        return EmptyStmt()
    if len(non_empty) == 1:
        return non_empty[0]
    return SeqStmt(non_empty[0], seq(*non_empty[1:]))


# ---------------------------------------------------------------------------
# participation_set — L(P) from the paper (Definition 8)
# ---------------------------------------------------------------------------

def participation_set(stmt: Stmt) -> frozenset[Lifeline]:
    """
    Compute the set of lifelines that appear in a statement.

    L(ε)                        = ∅
    L(msg A(x) → B(y))          = {A, B}
    L(act A: y := f(x))         = {A}
    L(skip A)                   = {A}
    L(P1 ; P2)                  = L(P1) ∪ L(P2)
    L(if c@B then P1 else P2)   = {B} ∪ L(P1) ∪ L(P2)
    L(while c@B do P exit P')   = {B} ∪ L(P) ∪ L(P')
    """
    match stmt:
        case EmptyStmt():
            return frozenset()
        case MsgStmt(sender=a, receiver=b):
            return frozenset({a, b})
        case ActStmt(lifeline=a):
            return frozenset({a})
        case SkipStmt(lifeline=a):
            return frozenset({a})
        case SeqStmt(first=p1, second=p2):
            return participation_set(p1) | participation_set(p2)
        case IfStmt(owner=b, branch_true=p1, branch_false=p2):
            return frozenset({b}) | participation_set(p1) | participation_set(p2)
        case WhileStmt(owner=b, body=p, exit_body=q):
            return frozenset({b}) | participation_set(p) | participation_set(q)
        case SendStmt(lifeline=a):
            return frozenset({a})
        case RecvStmt(lifeline=a):
            return frozenset({a})
        case IfRecvStmt(lifeline=a, sender=b, branch_true=p1, branch_false=p2):
            return frozenset({a, b}) | participation_set(p1) | participation_set(p2)
        case WhileRecvStmt(lifeline=a, sender=b, body=p, exit_body=q):
            return frozenset({a, b}) | participation_set(p) | participation_set(q)
        case _:
            raise TypeError(f"Unknown statement type: {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# pp — indented pretty-printer
# ---------------------------------------------------------------------------

def pp(node: Stmt, indent: int = 0) -> str:
    """
    Return an indented string representation of a statement tree.
    More readable than __repr__ for deeply nested programs.
    """
    pad = "  " * indent
    match node:
        case EmptyStmt():
            return f"{pad}ε"
        case MsgStmt(sender=s, payload=xs, receiver=r, bindings=ys):
            x_str = ", ".join(repr(e) for e in xs)
            y_str = ", ".join(repr(e) for e in ys)
            return f"{pad}msg {s.name}({x_str}) → {r.name}({y_str})"
        case ActStmt(lifeline=a, action=act, inputs=ins, outputs=outs):
            i_str = ", ".join(repr(e) for e in ins)
            o_str = ", ".join(v.name for v in outs)
            return f"{pad}act {a.name}: {o_str} := {act.name}({i_str})"
        case SkipStmt(lifeline=a):
            return f"{pad}skip {a.name}"
        case SeqStmt(first=p1, second=p2):
            return pp(p1, indent) + "\n" + pp(p2, indent)
        case IfStmt(condition=c, owner=b, branch_true=t, branch_false=f):
            return "\n".join([
                f"{pad}if {c!r} @{b.name}:",
                pp(t, indent + 1),
                f"{pad}else:",
                pp(f, indent + 1),
            ])
        case WhileStmt(condition=c, owner=b, body=body, exit_body=exit_b):
            return "\n".join([
                f"{pad}while {c!r} @{b.name}:",
                pp(body, indent + 1),
                f"{pad}exit:",
                pp(exit_b, indent + 1),
            ])
        case SendStmt(lifeline=a, payload=xs, receiver=b):
            x_str = ", ".join(repr(e) for e in xs)
            return f"{pad}send {a.name}({x_str}) → {b.name}"
        case RecvStmt(lifeline=a, bindings=ys, sender=b):
            y_str = ", ".join(repr(e) for e in ys)
            return f"{pad}recv {a.name}({y_str}) ← {b.name}"
        case IfRecvStmt(lifeline=a, bindings=ys, sender=b, branch_true=t, branch_false=f):
            y_str = ", ".join(repr(e) for e in ys)
            return "\n".join([
                f"{pad}if {a.name}({y_str}) ← {b.name}:",
                pp(t, indent + 1),
                f"{pad}else:",
                pp(f, indent + 1),
            ])
        case WhileRecvStmt(lifeline=a, bindings=ys, sender=b, body=body, exit_body=exit_b):
            y_str = ", ".join(repr(e) for e in ys)
            return "\n".join([
                f"{pad}while {a.name}({y_str}) ← {b.name}:",
                pp(body, indent + 1),
                f"{pad}exit:",
                pp(exit_b, indent + 1),
            ])
        case _:
            raise TypeError(f"Unknown statement type: {type(node).__name__}")
