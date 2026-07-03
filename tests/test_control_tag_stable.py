"""Control tags must be process-stable.

Across two OS processes the same workflow is imported into distinct object
graphs, so a tag keyed on ``id(stmt)`` differs between owner and receiver and
coordination deadlocks/errors. These tests pin the tag to program *content*
(paper's κ_ctrl^P), so an owner's broadcast tag equals the receiver's expected
tag even when the two are projected from separately-constructed graphs — which
``copy.deepcopy`` faithfully simulates (fresh ids, identical structure).
"""
import copy
import dataclasses

from zippergen.projection import project
from zippergen.syntax import (
    SendStmt, WhileStmt, WhileRecvStmt, SeqStmt, is_kappa_ctrl,
)
from tests.loop_fixture import counter_loop, A, B


def _regraph(wf):
    """Return a structurally-identical workflow whose IR tree has fresh object
    ids — the deep-copied body simulates re-importing the module in a separate
    process. (The Workflow itself holds a threading.Lock and can't be deep-copied
    wholesale; only the body drives tag derivation.)"""
    return dataclasses.replace(wf, body=copy.deepcopy(wf.body))


def _first_ctrl_tag_in_sends(node):
    """Return the kappa tag of the first control SendStmt reachable in a body."""
    stack = [node]
    while stack:
        s = stack.pop()
        if isinstance(s, SendStmt):
            for e in s.payload:
                if is_kappa_ctrl(e):
                    return e
        if isinstance(s, SeqStmt):
            stack.append(s.second)
            stack.append(s.first)
    raise AssertionError("no control SendStmt found")


def _recv_tag(node):
    """Return the kappa tag a WhileRecvStmt binds from its sender."""
    assert isinstance(node, WhileRecvStmt)
    for e in node.bindings:
        if is_kappa_ctrl(e):
            return e
    raise AssertionError("no control tag in WhileRecvStmt bindings")


def test_owner_and_receiver_tags_match_across_object_graphs():
    # Owner A from the original graph; receiver B from a deep copy (distinct ids,
    # same syntax) — as if projected in a separate process.
    la = project(counter_loop, A)
    lb = project(_regraph(counter_loop), B)

    assert isinstance(la, WhileStmt)
    owner_tag = _first_ctrl_tag_in_sends(la.body)
    recv_tag = _recv_tag(lb)

    assert owner_tag == recv_tag, (
        f"owner broadcast tag {owner_tag!r} != receiver expected tag {recv_tag!r}; "
        "control tag is not process-stable"
    )


def test_tag_is_independent_of_object_identity():
    # Two independent deep copies must yield the same tag for the same construct.
    lb1 = project(_regraph(counter_loop), B)
    lb2 = project(_regraph(counter_loop), B)
    assert _recv_tag(lb1) == _recv_tag(lb2)
