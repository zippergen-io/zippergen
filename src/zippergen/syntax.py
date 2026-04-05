"""
Layer 1: IR dataclasses. One class per grammar construct from the paper.
Frozen so nodes are immutable and hashable.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from zipperchat import WebTrace as _WebTrace

__all__ = [
    # Types
    "ZType", "is_ztype",
    # Lifeline
    "Lifeline",
    # Var
    "Var",
    # Expressions
    "Expr",
    "VarExpr", "LitExpr",
    # Actions
    "LLMAction", "PureAction", "PlannerAction",
    # Type + lifeline annotation helper
    "ZTypeAtLifeline",
    # Statements
    "Stmt",
    "EmptyStmt", "MsgStmt", "ActStmt", "SkipStmt",
    "SeqStmt", "IfStmt", "WhileStmt",
    # Local-only statements (produced by projection)
    "LocalStmt", "AnyStmt",
    "SendStmt", "RecvStmt", "IfRecvStmt", "WhileRecvStmt",
    # Workflow and Program
    "Workflow", "Program",
    # Reserved literals
    "kappa_ctrl",
    # Helpers
    "seq", "participation_set", "pp",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# ZType is a Python built-in type used as a coordination type annotation.
# Supported: str, int, bool, float, tuple.
ZType = type

_BUILTIN_ZTYPES: frozenset[type] = frozenset({str, int, bool, float, tuple})


def is_ztype(x: object) -> bool:
    """Return True iff x is a supported ZipperGen coordination type."""
    return x in _BUILTIN_ZTYPES


# ---------------------------------------------------------------------------
# Lifeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lifeline:
    name: str

    def __repr__(self) -> str:
        return self.name

    def __rmatmul__(self, ztype: object) -> ZTypeAtLifeline:
        """Support ``str @ Planner`` as a parameter annotation."""
        return ZTypeAtLifeline(ztype, self)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ZTypeAtLifeline:
    """Result of ``ZType @ Lifeline`` in a ``@proc`` parameter annotation.

    Declares both the type of an input variable and the lifeline that holds it
    at the start of execution::

        def myProc(task: str @ Planner) -> str: ...
    """
    type: ZType
    lifeline: Lifeline


# ---------------------------------------------------------------------------
# Var
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Var:
    name: str
    type: ZType
    default: object = None  # optional Python literal default

    def __repr__(self) -> str:
        t = self.type.__name__
        if self.default is not None:
            return f"Var({self.name!r}: {t} = {self.default!r})"
        return f"Var({self.name!r}: {t})"


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


Expr = Union[VarExpr, LitExpr]

# Reserved control tag — used only by the projection engine in control-broadcast
# messages (send B(⊤, κ_ctrl) → C).  Must not appear in user-written programs.
kappa_ctrl: LitExpr = LitExpr("κ_ctrl", str)


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
        ins = ", ".join(f"{n}: {t.__name__}" for n, t in self.inputs)
        outs = ", ".join(f"{n}: {t.__name__}" for n, t in self.outputs)
        return f"LLMAction({self.name!r}, ({ins}) -> ({outs}))"


@dataclass(frozen=True)
class PureAction:
    name: str
    inputs: tuple[tuple[str, ZType], ...]
    outputs: tuple[tuple[str, ZType], ...]
    fn: Callable[..., object]

    def __repr__(self) -> str:
        ins = ", ".join(f"{n}: {t.__name__}" for n, t in self.inputs)
        outs = ", ".join(f"{n}: {t.__name__}" for n, t in self.outputs)
        return f"PureAction({self.name!r}, ({ins}) -> ({outs}))"


@dataclass(frozen=True)
class PlannerAction:
    """IR node for an LLM-generated sub-workflow action.

    The runtime generates a ZipperGen workflow spec via LLM (using
    ``system_prompt`` and the declared ``actions``/``lifelines`` vocabulary),
    writes it to a temp file, imports it, and runs it.  The single ``str``
    output is the result returned by the generated workflow.
    """
    name: str
    inputs: tuple[tuple[str, ZType], ...]   # always includes "request" and "inputs_json"
    outputs: tuple[tuple[str, ZType], ...]  # always single (name, str)
    system_prompt: str
    actions: tuple        # tuple of LLMAction | PureAction  (base vocabulary)
    lifelines: tuple      # tuple of Lifeline used in inner workflows
    allow: tuple[str, ...] = ()          # action kinds the LLM may define: "pure", "llm"
    instructions: str | None = None      # optional user guidance on worker roles
    max_retries: int = 3                 # max correction attempts on invalid generated spec

    def __repr__(self) -> str:
        ins = ", ".join(f"{n}: {t.__name__}" for n, t in self.inputs)
        outs = ", ".join(f"{n}: {t.__name__}" for n, t in self.outputs)
        return f"PlannerAction({self.name!r}, ({ins}) -> ({outs}))"


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
    action: Union[LLMAction, PureAction, "PlannerAction"]
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
    first: AnyStmt
    second: AnyStmt

    def __repr__(self) -> str:
        return f"({self.first!r} ; {self.second!r})"


@dataclass(frozen=True)
class IfStmt:
    """if condition@owner then branch_true else branch_false"""
    condition: Callable[..., bool]
    owner: Lifeline
    branch_true: AnyStmt
    branch_false: AnyStmt

    def __repr__(self) -> str:
        return (
            f"if {self.condition!r}@{self.owner.name} "
            f"then {self.branch_true!r} "
            f"else {self.branch_false!r}"
        )


@dataclass(frozen=True)
class WhileStmt:
    """while condition@owner do { body } exit { exit_body }"""
    condition: Callable[..., bool]
    owner: Lifeline
    body: AnyStmt
    exit_body: AnyStmt

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
    """send A(xs) → B — sender's local view of a message."""
    lifeline: Lifeline
    payload: tuple[Expr, ...]
    receiver: Lifeline

    def __repr__(self) -> str:
        xs = ", ".join(repr(e) for e in self.payload)
        return f"send {self.lifeline.name}({xs}) → {self.receiver.name}"


@dataclass(frozen=True)
class RecvStmt:
    """recv A(ys) ← B — receiver's local view of a message."""
    lifeline: Lifeline
    bindings: tuple[Expr, ...]
    sender: Lifeline

    def __repr__(self) -> str:
        ys = ", ".join(repr(e) for e in self.bindings)
        return f"recv {self.lifeline.name}({ys}) ← {self.sender.name}"


@dataclass(frozen=True)
class IfRecvStmt:
    """if A(ys) ← B then branch_true else branch_false

    A receives ys from B.  bindings[0] is a bool variable; its runtime value
    determines which branch is taken.
    """
    lifeline: Lifeline
    bindings: tuple[Expr, ...]   # bindings[0] is the bool control variable
    sender: Lifeline
    branch_true: AnyStmt
    branch_false: AnyStmt

    def __repr__(self) -> str:
        ys = ", ".join(repr(e) for e in self.bindings)
        return (
            f"if {self.lifeline.name}({ys}) ← {self.sender.name} "
            f"then {self.branch_true!r} "
            f"else {self.branch_false!r}"
        )


@dataclass(frozen=True)
class WhileRecvStmt:
    """while A(ys) ← B do body exit exit_body

    Each iteration A receives ys from B.  bindings[0] is a bool variable;
    True means continue the body, False means take the exit.
    """
    lifeline: Lifeline
    bindings: tuple[Expr, ...]   # bindings[0] is the bool control variable
    sender: Lifeline
    body: AnyStmt
    exit_body: AnyStmt

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

# Union covering both global (Stmt) and local (LocalStmt) programs.
# Used for recursive child positions in shared nodes (SeqStmt, IfStmt, WhileStmt)
# and for functions that operate on either kind of program.
AnyStmt = Union[
    EmptyStmt, MsgStmt, SendStmt, RecvStmt, ActStmt, SkipStmt,
    SeqStmt, IfStmt, WhileStmt, IfRecvStmt, WhileRecvStmt,
]


def _ordered_workflow_lifelines(wf: "Workflow") -> tuple[Lifeline, ...]:
    # Ordering policy: lifelines appear in the order they were declared in the
    # workflow's namespace (e.g. module-level Lifeline(...) variables), then any
    # that appear only in input annotations, then any remaining participants sorted
    # alphabetically. This gives a stable, predictable display order.
    participants = participation_set(wf.body)
    ordered: list[Lifeline] = []
    seen: set[Lifeline] = set()

    for value in wf.ns.values():
        if isinstance(value, Lifeline) and value in participants and value not in seen:
            ordered.append(value)
            seen.add(value)

    for _name, _ztype, lifeline in wf.inputs:
        if lifeline is not None and lifeline in participants and lifeline not in seen:
            ordered.append(lifeline)
            seen.add(lifeline)

    remaining = sorted((lifeline for lifeline in participants if lifeline not in seen), key=lambda l: l.name)
    ordered.extend(remaining)
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Workflow and Program
# ---------------------------------------------------------------------------

@dataclass
class _WorkflowRuntime:
    """Mutable runtime state for a Workflow, kept separate from the IR fields."""
    _backend: object = field(default=None, repr=False)
    _trace: object = field(default=None, repr=False)
    _timeout: float = field(default=60.0, repr=False)
    _webtrace: _WebTrace | None = field(default=None, repr=False)
    _ui_enabled: bool = field(default=False, repr=False)
    _run_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _replay_thread: object = field(default=None, repr=False)
    _last_kwargs: dict[str, object] = field(default_factory=dict, repr=False)


@dataclass
class Workflow:
    name: str
    inputs: tuple[tuple[str, ZType, Lifeline | None], ...]  # (name, type, lifeline)
    output_type: ZType
    vars: tuple[Var, ...]       # locally declared variables
    body: Stmt
    output_var: Var | None = None           # declared via ``return var @ Lifeline``
    output_lifeline: Lifeline | None = None
    ns: dict = field(default_factory=dict)  # workflow's global namespace (for condition lambdas)
    _rt: _WorkflowRuntime = field(default_factory=_WorkflowRuntime, init=False, repr=False)

    # Convenience property shims so planner.py and tests can still write
    # wf._backend = x, wf._trace = x, wf._timeout = x without changes.
    @property
    def _backend(self): return self._rt._backend
    @_backend.setter
    def _backend(self, v): self._rt._backend = v

    @property
    def _trace(self): return self._rt._trace
    @_trace.setter
    def _trace(self, v): self._rt._trace = v

    @property
    def _timeout(self): return self._rt._timeout
    @_timeout.setter
    def _timeout(self, v): self._rt._timeout = v

    @property
    def _webtrace(self): return self._rt._webtrace
    @_webtrace.setter
    def _webtrace(self, v): self._rt._webtrace = v

    @property
    def _ui_enabled(self): return self._rt._ui_enabled
    @_ui_enabled.setter
    def _ui_enabled(self, v): self._rt._ui_enabled = v

    @property
    def _run_lock(self): return self._rt._run_lock

    @property
    def _replay_thread(self): return self._rt._replay_thread
    @_replay_thread.setter
    def _replay_thread(self, v): self._rt._replay_thread = v

    @property
    def _last_kwargs(self): return self._rt._last_kwargs
    @_last_kwargs.setter
    def _last_kwargs(self, v): self._rt._last_kwargs = v

    def configure(self, *,
                  backend: object = None,
                  trace:   object = None,
                  timeout: float  = 60.0,
                  llms: str | Mapping[str, str | Callable] | None = None,
                  ui: bool | None = None,
                  mock_delay: tuple[float, float] = (1.0, 2.0)) -> Workflow:
        """Configure runtime parameters and return self for chaining.

        Parameters
        ----------
        backend : LLM backend callable ``(action, inputs_dict) → outputs_dict``.
                  Defaults to the built-in mock backend.
        trace   : trace callable passed to ``run()``.
        timeout : per-thread timeout in seconds (default 60).
        llms    : ``"mock"``, a provider name like ``"openai"`` / ``"mistral"``,
                  or a mapping ``lifeline_name -> provider``. This is a simple
                  convenience layer for examples and demos.
        ui      : if true, start ZipperChat and mirror the execution there.
        mock_delay : delay range used by the mock backend when ``llms="mock"``
                     or when no provider route is configured.
        """
        lifelines = _ordered_workflow_lifelines(self)

        if llms is not None:
            from zippergen.backends import router_from_env
            from zippergen.runtime import mock_llm

            if llms == "mock":
                routes: dict[str, str | Callable[..., object]] = {}
            elif isinstance(llms, str):
                routes = {lifeline.name: llms for lifeline in lifelines}
            else:
                # Values may be provider name strings OR pre-built backend callables.
                routes = {str(k): v for k, v in llms.items()}

            built_backend, _label = router_from_env(
                routes,
                fallback=lambda a, i: mock_llm(a, i, min_delay=mock_delay[0], max_delay=mock_delay[1]),
            )
            self._rt._backend = built_backend
        if backend is not None:
            self._rt._backend = backend

        if ui is not None:
            self._rt._ui_enabled = ui
        if self._rt._ui_enabled:
            from zippergen.runtime import console_trace, tee_traces
            from zipperchat import WebTrace

            if self._rt._webtrace is None:
                self._rt._webtrace = WebTrace(lifelines).start()
            base_trace = trace if trace is not None else console_trace
            self._rt._trace = tee_traces(self._rt._webtrace, base_trace)
        elif trace is not None:
            self._rt._trace = trace

        self._rt._timeout = timeout
        return self

    def _run_once(self, kwargs: dict[str, object]) -> object:
        from zippergen.runtime import run, mock_llm  # lazy to avoid circular import

        initial_envs: dict[str, dict[str, object]] = {}
        for name, _ztype, lifeline in self.inputs:
            if lifeline is None:
                raise TypeError(
                    f"{self.name}(): input '{name}' has no lifeline declared. "
                    f"Use 'name: type @ Lifeline' in the @workflow signature."
                )
            if name not in kwargs:
                raise TypeError(f"{self.name}() missing argument: '{name}'")
            initial_envs.setdefault(lifeline.name, {})[name] = kwargs[name]

        lifelines = _ordered_workflow_lifelines(self)
        backend = self._rt._backend if self._rt._backend is not None else mock_llm
        with self._rt._run_lock:
            if self._rt._webtrace is not None and self._rt._ui_enabled:
                self._rt._webtrace.reset()
            try:
                return run(
                    self,
                    list(lifelines),
                    initial_envs,
                    llm_backend=backend,
                    trace=self._rt._trace,
                    timeout=self._rt._timeout,
                )
            finally:
                if self._rt._webtrace is not None and self._rt._ui_enabled:
                    self._rt._webtrace.done()

    def _ensure_replay_loop(self) -> None:
        if not self._rt._ui_enabled or self._rt._webtrace is None or self._rt._replay_thread is not None:
            return

        def _worker() -> None:
            assert self._rt._webtrace is not None
            while True:
                self._rt._webtrace.wait_for_replay()
                if not self._rt._last_kwargs:
                    continue
                try:
                    result = self._run_once(dict(self._rt._last_kwargs))
                    print(f"\nResult → {result}")
                except Exception as exc:
                    print(f"\nReplay failed: {exc}")

        self._rt._replay_thread = threading.Thread(target=_worker, daemon=True)
        self._rt._replay_thread.start()

    def __call__(self, **kwargs: object) -> object:
        """Run this workflow like a regular Python function.

        Keyword arguments must match the workflow's declared inputs (one per
        parameter in the ``@workflow`` signature).  Call ``configure()`` first to
        set the LLM backend, trace, and timeout.
        """
        self._rt._last_kwargs = dict(kwargs)
        self._ensure_replay_loop()
        return self._run_once(dict(kwargs))

    def __repr__(self) -> str:
        parts = []
        for n, t, ll in self.inputs:
            parts.append(f"{n}: {t.__name__} @ {ll.name}" if ll else f"{n}: {t.__name__}")
        ins = ", ".join(parts)
        out = (f" → {self.output_var.name}@{self.output_lifeline.name}"
               if self.output_var and self.output_lifeline else "")
        return f"Workflow({self.name!r}, ({ins}) -> {self.output_type.__name__}{out})"


@dataclass
class Program:
    lifelines: tuple[Lifeline, ...]
    actions: tuple[Union[LLMAction, PureAction, PlannerAction], ...]
    procs: tuple[Workflow, ...]

    def __repr__(self) -> str:
        ls = ", ".join(repr(l) for l in self.lifelines)
        ps = ", ".join(p.name for p in self.procs)
        return f"Program(lifelines=[{ls}], procs=[{ps}])"


# ---------------------------------------------------------------------------
# seq — right-associative fold with EmptyStmt identity
# ---------------------------------------------------------------------------

def seq(*stmts: AnyStmt) -> AnyStmt:
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

def participation_set(stmt: AnyStmt) -> frozenset[Lifeline]:
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

def pp(node: AnyStmt, indent: int = 0) -> str:
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
