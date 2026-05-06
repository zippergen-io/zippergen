"""Tests for per-lifeline CPL monitor state (Algorithms 1 and 2 from the paper)."""
import pytest
from zippergen.formula import atom, Y, P, since, on, subformulas, YAFormula
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


def test_guard_value_rejects_unregistered_formula():
    registered = atom(lambda env: True)
    missing = atom(lambda env: True)
    m = make_monitor("A", ["A"], registered)
    m.on_event("act", {})
    with pytest.raises(RuntimeError, match="not registered"):
        m.guard_value(missing)


# ---------------------------------------------------------------------------
# Compound formula tests
# ---------------------------------------------------------------------------

def test_compound_formula_and_both_true():
    phi1 = atom(lambda env: env.get("a", False))
    phi2 = atom(lambda env: env.get("b", False))
    conj = phi1 & phi2
    m = make_monitor("A", ["A"], conj)
    m.on_event("act", {"a": True, "b": True})
    assert m.view["A"][id(conj)] is True


def test_compound_formula_and_one_false():
    phi1 = atom(lambda env: env.get("a", False))
    phi2 = atom(lambda env: env.get("b", False))
    conj = phi1 & phi2
    m = make_monitor("A", ["A"], conj)
    m.on_event("act", {"a": True, "b": False})
    assert m.view["A"][id(conj)] is False


def test_compound_formula_not():
    phi = atom(lambda env: env.get("x", False))
    neg = ~phi
    m = make_monitor("A", ["A"], neg)
    m.on_event("act", {"x": True})
    assert m.view["A"][id(neg)] is False
    m.on_event("act", {"x": False})
    assert m.view["A"][id(neg)] is True


# ---------------------------------------------------------------------------
# Since and strict causal past
# ---------------------------------------------------------------------------

def test_since_is_non_strict_and_local():
    keep = atom(lambda env: env.get("keep", False))
    start = atom(lambda env: env.get("start", False))
    f = since(keep, start)
    m = make_monitor("A", ["A"], f)

    m.on_event("act", {"keep": False, "start": True})
    assert m.view["A"][id(f)] is True

    m.on_event("act", {"keep": True, "start": False})
    assert m.view["A"][id(f)] is True

    m.on_event("act", {"keep": False, "start": False})
    assert m.view["A"][id(f)] is False


def test_past_is_strict_on_same_lifeline():
    phi = atom(lambda env: env.get("x", False))
    f = P(phi)
    m = make_monitor("A", ["A"], f)

    m.on_event("act", {"x": True})
    assert m.view["A"][id(f)] is False

    m.on_event("act", {"x": False})
    assert m.view["A"][id(f)] is True


def test_past_sees_remote_causal_history_after_receive():
    phi = atom(lambda env: env.get("x", False))
    f = P(phi)
    m = make_monitor("B", ["A", "B"], f)

    recv_vc = {"A": 1, "B": 0}
    recv_view = {
        "A": {
            id(phi): True,
            id(f.witness.left): True,
            id(f.witness): True,
        },
        "B": {},
    }
    m.on_event("recv", {}, recv_vc=recv_vc, recv_view=recv_view)
    assert m.view["B"][id(f)] is True


def test_atom_can_read_event_context_metadata():
    phi = atom(lambda env, event: event.kind == "recv" and event.message_vc["A"] == 1)
    m = make_monitor("B", ["A", "B"], phi)
    m.on_event("recv", {}, recv_vc={"A": 1, "B": 0}, recv_view={"A": {}, "B": {}})
    assert m.view["B"][id(phi)] is True
