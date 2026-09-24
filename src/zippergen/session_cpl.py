"""Static CPL layouts for session bodies, independent of endpoint bindings."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from zippergen.formula import (
    AnyFormula, AtomicFormula, AtFormula, OnFormula, formula_fingerprint, subformulas,
)
from zippergen.monitor import MonitorState
from zippergen.syntax import (
    SpawnStmt, SeqStmt, IfStmt, WhileStmt, ParallelStmt,
    contains_spawn, participation_set,
)


@dataclass
class SessionCPL:
    roles: list[str]
    formulas: list[AnyFormula]
    conditions: dict[int, AnyFormula]
    layout: str

    def monitor(self, role: str) -> MonitorState | None:
        if role not in self.roles:
            raise ValueError(f"Role {role!r} is outside the CPL session interface")
        return MonitorState(role, self.roles, self.formulas) if self.formulas else None


def compile_session_cpl(wf) -> dict[str, SessionCPL]:
    """Keys are root or the static spawn site that creates the session.

    All endpoints use the same formula closure. A nested body is compiled
    separately, so its history and remote-role requirements do not leak upward.
    Versioned atoms give the transmitted layout a stable semantic identity.
    """
    from zippergen.runtime import _collect_formula_guards
    if not contains_spawn(wf.body):
        return {}
    plans: dict[str, SessionCPL] = {}

    def body(node, key):
        guards, conditions = _collect_formula_guards(node, wf.ns, descend_spawn=False)
        roles = sorted(role.name for role in participation_set(node))
        formulas: list[AnyFormula] = []
        seen = set()
        for guard in guards:
            for phi in subformulas(guard):
                if id(phi) in seen:
                    continue
                seen.add(id(phi))
                references = ({phi.lifeline_name} if isinstance(phi, (AtFormula, OnFormula))
                              else phi.roles if isinstance(phi, AtomicFormula) else set())
                absent = set(references) - set(roles)
                if absent:
                    raise ValueError(f"CPL at session body {key!r} references roles outside "
                                     f"the current session: {', '.join(sorted(absent))}")
                formulas.append(phi)
        shape = ["session-cpl-v1", roles, [formula_fingerprint(phi) for phi in formulas]]
        layout = hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest()
        plans[key] = SessionCPL(roles, formulas, conditions, layout)

    def walk(node, site):
        if isinstance(node, SpawnStmt):
            body(node.body, site)
            walk(node.body, site + ".0")
        elif isinstance(node, SeqStmt):
            walk(node.first, site + ".0")
            walk(node.second, site + ".1")
        elif isinstance(node, IfStmt):
            walk(node.branch_true, site + ".0")
            walk(node.branch_false, site + ".1")
        elif isinstance(node, WhileStmt):
            walk(node.body, site + ".0")
            walk(node.exit_body, site + ".1")
        elif isinstance(node, ParallelStmt):
            for index, branch in enumerate(node.branches):
                walk(branch, site + f".{index}")

    body(wf.body, "root")
    walk(wf.body, "0")
    if not any(plan.formulas for plan in plans.values()):
        return {}
    from zippergen.pools import pool_operations
    if pool_operations(wf):
        raise ValueError("CPL with spawn does not yet support pool causal handoffs")
    return plans
