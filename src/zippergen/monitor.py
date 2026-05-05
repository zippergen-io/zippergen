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
