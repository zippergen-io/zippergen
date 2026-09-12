"""Tests for per-lifeline CPL monitor state (Algorithms 1 and 2 from the paper)."""
import pytest
from zippergen.formula import atom, At, Here, Y, P, since, on, subformulas, AtFormula
from zippergen.monitor import MonitorState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_monitor(name: str, lifelines: list[str], formula):
    subs = subformulas(formula)
    return MonitorState(name, lifelines, subs)


def test_pool_handoffs_preserve_event_kind_previous_and_monotone_causal_past():
    seen = []

    def inspect(env, event):
        seen.append(event)
        return env.get("approved", False)

    approval = atom(inspect, src="approval", version="v1", fields=("approved",))
    latest = At["A"](approval)
    previous = Y(approval)
    ever = P(approval)
    formula = latest | previous | ever
    a = make_monitor("A", ["A", "B"], formula)
    b = make_monitor("B", ["A", "B"], formula)

    a.on_event("act", {"approved": True})
    old = a.snapshot_vc(), a.snapshot_view(), a.snapshot_field_view()
    a.on_event("act", {"approved": False})
    new = a.snapshot_vc(), a.snapshot_view(), a.snapshot_field_view()
    b.on_event("act", {"approved": False})
    b.on_event("act", {"approved": True},
               recv_vc=new[0], recv_view=new[1], recv_field_view=new[2])
    assert not b.guard_value(previous)  # preceding local action had False
    assert not b.guard_value(latest)
    assert b.guard_value(ever)
    assert seen[-1].kind == "act"
    assert seen[-1].message_vc is None and seen[-1].message_view is None
    assert seen[-1].causal_vc == new[0]

    b.on_event("act", {"approved": False},
               recv_vc=old[0], recv_view=old[1], recv_field_view=old[2])
    assert b.guard_value(previous)
    assert not b.guard_value(latest)  # late delivery cannot revive old approval
    assert b.guard_value(ever)
    assert b.snapshot_vc() == {"A": 2, "B": 3}


def wire_view(monitor, entries):
    """Encode formula objects with the stable indexes used on messages."""
    indexes = {
        id(formula): index
        for index, formula in enumerate(monitor.subformulas)
    }
    return {
        lifeline: {
            indexes[id(formula)]: value
            for formula, value in values.items()
        }
        for lifeline, values in entries.items()
    }


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
    recv_view = wire_view(m, {"A": {phi: True}, "B": {}})
    m.on_event("recv", {}, recv_vc=recv_vc, recv_view=recv_view)
    assert m.vc["A"] == 3


def test_recv_copies_view_when_ahead():
    phi = atom(lambda env: env.get("approved", False))
    m = make_monitor("B", ["A", "B"], phi)
    recv_vc = {"A": 1, "B": 0}
    recv_view = wire_view(m, {"A": {phi: True}, "B": {}})
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
    recv_view = wire_view(m, {"A": {phi: True}, "B": {}})
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
    yaf = At["A"](phi)    # At[A](phi) via string key
    assert isinstance(yaf, AtFormula)
    m = make_monitor("B", ["A", "B"], yaf)
    m.on_event("act", {})
    # vc["A"] == 0 → Y_A(phi) = False
    assert m.view["B"][id(yaf)] is False


def test_ya_true_after_receiving_message_from_a():
    phi = atom(lambda env: env.get("approved", False))
    from zippergen.syntax import Lifeline
    A = Lifeline("A")
    yaf = At[A](phi)
    m = make_monitor("B", ["A", "B"], yaf)
    # Simulate B receiving a message from A where A's view of phi was True
    recv_vc = {"A": 1, "B": 0}
    recv_view = wire_view(m, {"A": {phi: True}, "B": {}})
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
    assert snap == {"A": {0: True}}
    snap["A"][0] = False
    assert m.view["A"][id(phi)] is True


def test_transmitted_view_uses_indexes_across_distinct_formula_objects():
    sent_atom = atom(lambda env: env.get("ok", False))
    sent_guard = At["A"](sent_atom)
    received_atom = atom(lambda env: env.get("ok", False))
    received_guard = At["A"](received_atom)
    sender = make_monitor("A", ["A", "B"], sent_guard)
    receiver = make_monitor("B", ["A", "B"], received_guard)

    sender.on_event("act", {"ok": True})
    receiver.on_event(
        "recv",
        {},
        recv_vc=sender.snapshot_vc(),
        recv_view=sender.snapshot_view(),
    )

    assert receiver.guard_value(received_guard) is True


def test_receive_rejects_invalid_formula_index():
    phi = atom(lambda env: True)
    monitor = make_monitor("B", ["A", "B"], phi)

    with pytest.raises(RuntimeError, match="invalid formula index"):
        monitor.on_event(
            "recv",
            {},
            recv_vc={"A": 1, "B": 0},
            recv_view={"A": {1: True}, "B": {}},
        )


def test_monitor_state_roundtrip_uses_stable_formula_indexes():
    phi = atom(
        lambda env: env.get("approved", False),
        src="approved",
        version="approved-v1",
    )
    guard = Y(phi)
    first = make_monitor("A", ["A", "B"], guard)
    first.on_event("act", {"approved": True})

    state = first.snapshot_state()
    second = make_monitor("A", ["A", "B"], guard)
    second.restore_state(state)

    assert second.snapshot_state() == state
    second.on_event("act", {"approved": True})
    assert second.guard_value(guard) is True


def test_monitor_state_refuses_changed_atom_semantics_with_same_label():
    old = Y(atom(lambda env: False, src="ready", version="ready-v1"))
    new = Y(atom(lambda env: True, src="ready", version="ready-v2"))
    first = make_monitor("A", ["A"], old)
    first.on_event("act", {})

    with pytest.raises(ValueError, match="formulas changed"):
        make_monitor("A", ["A"], new).restore_state(first.snapshot_state())


def test_monitor_state_preserves_tuple_field_values():
    guard = atom(lambda env: True, version="always-v1", fields={"coordinates"})
    first = make_monitor("A", ["A"], guard)
    first.on_event("act", {"coordinates": (1, [2, 3])})

    second = make_monitor("A", ["A"], guard)
    second.restore_state(first.snapshot_state())

    assert second.field_view["A"]["coordinates"] == (1, [2, 3])
    assert type(second.field_view["A"]["coordinates"]) is tuple


def test_field_view_snapshots_mutable_local_values():
    phi = atom(lambda env: True, fields={"items"})
    m = make_monitor("A", ["A"], phi)
    env = {"items": ["old"]}

    m.on_event("act", env)
    snap = m.snapshot_field_view()

    env["items"].append("env")
    snap["A"]["items"].append("snap")

    assert m.field_view["A"]["items"] == ["old"]


def test_recv_deep_copies_incoming_field_view_when_ahead():
    phi = atom(lambda env: True, fields={"items"})
    m = make_monitor("B", ["A", "B"], phi)
    recv_field_view = {"A": {"items": ["remote"]}, "B": {}}

    m.on_event(
        "recv",
        {},
        recv_vc={"A": 1, "B": 0},
        recv_view={"A": {}, "B": {}},
        recv_field_view=recv_field_view,
    )

    recv_field_view["A"]["items"].append("mutated")

    assert m.field_view["A"]["items"] == ["remote"]


def test_field_view_keeps_only_fields_used_by_cross_lifeline_terms():
    guard = At["A"].version == Here.version
    m = make_monitor("A", ["A", "B"], guard)

    m.on_event("act", {"version": "v1", "private_notes": "large"})

    assert m.snapshot_field_view()["A"] == {"version": "v1"}


def test_low_level_context_atom_without_field_declaration_keeps_all_fields():
    guard = atom(lambda env, ctx: bool(ctx.field_view["A"]["needed"]))
    m = make_monitor("A", ["A"], guard)

    m.on_event("act", {"needed": True, "other": "preserved"})

    assert m.snapshot_field_view()["A"] == {
        "needed": True,
        "other": "preserved",
    }


def test_field_term_formula_compares_latest_visible_values():
    guard = At["A"].version == Here.version
    a = make_monitor("A", ["A", "B"], guard)
    b = make_monitor("B", ["A", "B"], guard)

    a.on_event("act", {"version": "v1"})
    b.on_event(
        "recv",
        {"version": "v1"},
        recv_vc=a.snapshot_vc(),
        recv_view=a.snapshot_view(),
        recv_field_view=a.snapshot_field_view(),
    )

    assert b.view["B"][id(guard)] is True


def test_field_term_formula_rejects_mismatched_latest_visible_value():
    guard = At["A"].version == Here.version
    a = make_monitor("A", ["A", "B"], guard)
    b = make_monitor("B", ["A", "B"], guard)

    a.on_event("act", {"version": "v1"})
    b.on_event(
        "recv",
        {"version": "v2"},
        recv_vc=a.snapshot_vc(),
        recv_view=a.snapshot_view(),
        recv_field_view=a.snapshot_field_view(),
    )

    assert b.view["B"][id(guard)] is False


def test_field_term_formula_rejects_absent_latest_value_for_disequality():
    guard = At["A"].version != Here.version
    a = make_monitor("A", ["A", "B"], guard)
    b = make_monitor("B", ["A", "B"], guard)

    a.on_event("act", {"other": "v1"})
    b.on_event(
        "recv",
        {"version": "v2"},
        recv_vc=a.snapshot_vc(),
        recv_view=a.snapshot_view(),
        recv_field_view=a.snapshot_field_view(),
    )

    assert b.view["B"][id(guard)] is False


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
# Since and causal past
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


def test_pending_delimits_sequential_candidate_episodes():
    not_failed = atom(lambda env: env.get("status") != "failed")
    not_pending = atom(lambda env: env.get("status") != "pending")
    passed = atom(lambda env: env.get("status") == "passed")
    guard = since(not_failed & not_pending, passed)
    monitor = make_monitor("TestRunner", ["TestRunner"], guard)

    history = [
        ("A", "pending", False),
        ("A", "checking", False),
        ("A", "passed", True),
        ("A", "checking", True),
        ("B", "pending", False),
        ("B", "checking", False),
        ("B", "passed", True),
        ("B", "failed", False),
        ("B", "checking", False),
    ]
    for candidate, status, expected in history:
        monitor.on_event("act", {"candidate": candidate, "status": status})
        assert monitor.guard_value(guard) is expected


def test_past_is_non_strict_on_same_lifeline():
    phi = atom(lambda env: env.get("x", False))
    f = P(phi)
    m = make_monitor("A", ["A"], f)

    m.on_event("act", {"x": True})
    assert m.view["A"][id(f)] is True

    m.on_event("act", {"x": False})
    assert m.view["A"][id(f)] is True


def test_past_sees_remote_causal_history_after_receive():
    phi = atom(lambda env: env.get("x", False))
    f = P(phi)
    m = make_monitor("B", ["A", "B"], f)

    recv_vc = {"A": 1, "B": 0}
    recv_view = wire_view(m, {
        "A": {
            phi: True,
            f.witness.left: True,
            f.witness: True,
        },
        "B": {},
    })
    m.on_event("recv", {}, recv_vc=recv_vc, recv_view=recv_view)
    assert m.view["B"][id(f)] is True


def test_atom_can_read_event_context_metadata():
    phi = atom(lambda env, event: event.kind == "recv" and event.message_vc["A"] == 1)
    m = make_monitor("B", ["A", "B"], phi)
    m.on_event("recv", {}, recv_vc={"A": 1, "B": 0}, recv_view={"A": {}, "B": {}})
    assert m.view["B"][id(phi)] is True
