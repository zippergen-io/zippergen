/-
  MSCAgents/PayloadMatching.lean
  ==============================
  Formalization of Definition 4 (def:payload-matching) from:
  "Provable Coordination for LLM Agents via Message Sequence Charts"

  Paper section: sec:semantics

  From the paper (def:payload-matching):
    Let x⃗ be a concrete sender payload tuple and let y⃗ be a receiver
    payload pattern tuple (whose components may be variables or constants).

    match(x⃗, y⃗) holds iff:
      1. |x⃗| = |y⃗|
      2. For each position i:
         - if y_i is a constant, then x_i = y_i
         - if y_i is a variable of type τ, then x_i has type τ

  Design:
    We model payload components abstractly. A `PayloadComp V` is either
    a concrete value (`val v`) or a variable placeholder (`var`).
    A payload tuple is a `List (PayloadComp V)`.
    Matching checks pointwise: concrete positions must agree by equality;
    variable positions always match (we elide the type-tag check for
    simplicity, treating all values as having the same type).
-/

------------------------------------------------------------------------
-- Payload components
------------------------------------------------------------------------

/-- A payload component is either a concrete value or a variable placeholder.

    In the paper, y_i in a receiver pattern can be a constant or a variable.
    We model this distinction directly.

    The type parameter `V` is the type of concrete values. -/
inductive PayloadComp (V : Type) : Type where
  /-- A concrete (constant) value. -/
  | val (v : V) : PayloadComp V
  /-- A variable placeholder (matches any value). -/
  | var : PayloadComp V
  deriving DecidableEq, Repr

------------------------------------------------------------------------
-- Payload tuples
------------------------------------------------------------------------

/-- A payload tuple is a list of payload components. -/
abbrev PayloadTuple (V : Type) := List (PayloadComp V)

/-- Abstract interface for payload compatibility used by MSC label matching.
    Concrete payload domains can instantiate this with their own relation. -/
class PayloadCompatiblePred (Payload : Type) where
  compat : Payload → Payload → Prop

/-- Shorthand for the payload-compatibility relation. -/
abbrev PayloadCompatible (Payload : Type) [PayloadCompatiblePred Payload]
    (xs ys : Payload) : Prop :=
  PayloadCompatiblePred.compat xs ys

------------------------------------------------------------------------
-- List helper lemmas for zip membership (used in Matching section below)
------------------------------------------------------------------------

namespace List

/-- If `i < xs.length` and `i < ys.length`, then `(xs[i], ys[i]) ∈ zip xs ys`. -/
theorem get_mem_zip {α β : Type} {xs : List α} {ys : List β} {i : Nat}
    (hi : i < xs.length) (hj : i < ys.length) :
    (xs.get ⟨i, hi⟩, ys.get ⟨i, hj⟩) ∈ List.zip xs ys := by
  induction xs generalizing ys i with
  | nil => exact absurd hi (Nat.not_lt_zero _)
  | cons x xt ih =>
    cases ys with
    | nil => exact absurd hj (Nat.not_lt_zero _)
    | cons y yt =>
      simp only [List.zip_cons_cons]
      cases i with
      | zero =>
        simp only [List.get]
        exact List.Mem.head _
      | succ n =>
        simp only [List.get]
        exact List.Mem.tail _ (ih (Nat.lt_of_succ_lt_succ hi) (Nat.lt_of_succ_lt_succ hj))

/-- Membership in zip implies there exists an index with get equality. -/
theorem mem_zip_get {α β : Type} {xs : List α} {ys : List β} {p : α × β}
    (hmem : p ∈ List.zip xs ys) :
    ∃ i, ∃ hi : i < xs.length, ∃ hj : i < ys.length,
      xs.get ⟨i, hi⟩ = p.1 ∧ ys.get ⟨i, hj⟩ = p.2 := by
  induction xs generalizing ys with
  | nil => simp at hmem
  | cons x xt ih =>
    cases ys with
    | nil => simp at hmem
    | cons y yt =>
      simp only [List.zip_cons_cons, List.mem_cons] at hmem
      rcases hmem with rfl | hmem
      · exact ⟨0, Nat.zero_lt_succ _, Nat.zero_lt_succ _, rfl, rfl⟩
      · obtain ⟨i, hi, hj, hx, hy⟩ := ih hmem
        exact ⟨i + 1, Nat.succ_lt_succ hi, Nat.succ_lt_succ hj, hx, hy⟩

end List

------------------------------------------------------------------------
-- Pointwise matching
------------------------------------------------------------------------

section Matching

variable {V : Type} [DecidableEq V]

/-- Pointwise match of a single component:
    - If the pattern is a concrete value, the sender value must equal it.
    - If the pattern is a variable, any sender value matches. -/
def compMatch (x y : PayloadComp V) : Bool :=
  match y with
  | .var   => true
  | .val v =>
    match x with
    | .val w => decide (w = v)
    | .var   => false  -- a variable sender cannot match a concrete receiver

/-- `payloadMatch xs ys` implements match(x⃗, y⃗) from def:payload-matching.

    Returns true iff:
    1. |x⃗| = |y⃗|
    2. For each position i: compMatch x_i y_i holds.

    The paper: "match(x⃗, y⃗) holds iff |x⃗| = |y⃗| and for each position i,
    if y_i is a constant then x_i = y_i; if y_i is a variable then x_i has
    type τ." We elide the type check (single-sorted values). -/
def payloadMatch (xs ys : PayloadTuple V) : Bool :=
  xs.length == ys.length &&
  (List.zip xs ys).all (fun ⟨x, y⟩ => compMatch x y)

/-- Propositional version of payload matching. -/
def PayloadMatches (xs ys : PayloadTuple V) : Prop :=
  xs.length = ys.length ∧
  ∀ i : Fin xs.length,
    ∀ h : i.val < ys.length,
      match ys.get ⟨i.val, h⟩ with
      | .var   => True
      | .val v => xs.get i = .val v

/-- The boolean `payloadMatch` agrees with the propositional `PayloadMatches`
    when both sender entries are concrete values. -/
theorem payloadMatch_iff_matches (xs ys : PayloadTuple V) :
    payloadMatch xs ys = true ↔
    (xs.length = ys.length ∧
     ∀ (i : Nat) (hi : i < xs.length) (hj : i < ys.length),
       compMatch (xs.get ⟨i, hi⟩) (ys.get ⟨i, hj⟩) = true) := by
  simp only [payloadMatch, Bool.and_eq_true, beq_iff_eq]
  constructor
  · rintro ⟨hlen, hall⟩
    refine ⟨hlen, fun i hi hj => ?_⟩
    have hmem : (xs.get ⟨i, hi⟩, ys.get ⟨i, hj⟩) ∈ List.zip xs ys :=
      List.get_mem_zip hi hj
    exact List.all_eq_true.mp hall _ hmem
  · rintro ⟨hlen, hcomp⟩
    refine ⟨hlen, List.all_eq_true.mpr ?_⟩
    intro ⟨a, b⟩ hmem
    obtain ⟨i, hi, hj, ha, hb⟩ := List.mem_zip_get hmem
    exact ha ▸ hb ▸ hcomp i hi hj

/-- Empty payload tuples match trivially. -/
@[simp]
theorem payloadMatch_nil : payloadMatch (V := V) [] [] = true := by
  simp [payloadMatch]

end Matching

/-- Concrete payload tuples use Definition `def:payload-matching`
    as their compatibility relation. -/
instance payloadTupleCompatiblePred (V : Type) [DecidableEq V] :
    PayloadCompatiblePred (PayloadTuple V) where
  compat xs ys := PayloadMatches xs ys
