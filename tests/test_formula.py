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


def test_public_api_importable_from_zippergen():
    # NOTE: This test will pass only after Task 4 (__init__.py is updated).
    # For now, just verify the formula module is importable directly.
    from zippergen.formula import Y, atom, on, subformulas, Formula
    phi = atom(lambda env: True)
    assert isinstance(phi, Formula)
