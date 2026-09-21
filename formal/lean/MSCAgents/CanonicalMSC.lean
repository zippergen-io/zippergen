/-
  MSCAgents/CanonicalMSC.lean
  ===========================
  Formalization of Definition 7 (def:base-msc) from:
  "Provable Coordination for LLM Agents via Message Sequence Charts"

  This file defines the canonical (atomic) MSCs from which the
  inductive semantics is built.

  Paper section: sec:semantics

  From the paper (def:base-msc):
    Fix A, B ∈ 𝓛, tuples x⃗, y⃗, action f, and choice letter γ ∈ Σ_B.
    • M_ε               : empty MSC (all w_X = ε)
    • M_γ^B             : w_B = γ,      w_X = ε for X ≠ B
    • M_{x⃗,y⃗}^{A,f}   : w_A = act A(y⃗) := f(x⃗), w_X = ε for X ≠ A
    • M_{x⃗,y⃗}^{A→B}   : w_A = send A(x⃗) → B,
                           w_B = recv B(y⃗) ← A,
                           w_X = ε for X ∉ {A,B}
-/

import MSCAgents.Alphabets

------------------------------------------------------------------------
-- Helper: singleton local word
------------------------------------------------------------------------

/-- A local word consisting of a single letter. -/
def singletonWord {L C F Payload : Type} {A : L}
    (ℓ : AlphabetOf (C := C) (F := F) (Payload := Payload) A) :
    LocalWord (C := C) (F := F) (Payload := Payload) A :=
  [ℓ]

------------------------------------------------------------------------
-- Canonical MSCs
------------------------------------------------------------------------

section CanonicalMSCs

variable {L C F Payload : Type} [DecidableEq L]

------------------------------------------------------------------------
-- M_ε : the empty MSC (def:base-msc, first bullet)
------------------------------------------------------------------------

/-- `mscEmpty` is the empty MSC M_ε.
    All local words are empty: w_X = ε for every X ∈ 𝓛.
    This is the same as `WordTuple.empty`. -/
def mscEmpty : WordTuple L C F Payload :=
  WordTuple.empty

-- M_γ^B : the canonical choice MSC (def:base-msc, third bullet)
------------------------------------------------------------------------

/-- `mscChoice γ B` is the canonical choice MSC M_γ^B for a choice letter γ
    already bundled as an `AlphabetOf B`.
    • w_B = [γ]
    • w_X = []  for X ≠ B -/
def mscChoice (B : L)
    (γ : AlphabetOf (C := C) (F := F) (Payload := Payload) B) :
    WordTuple L C F Payload :=
  fun X =>
    if h : X = B then
      h ▸ singletonWord γ
    else
      []

/-- The B-component of mscChoice γ B is [γ]. -/
@[simp]
theorem mscChoice_owner (B : L)
    (γ : AlphabetOf (C := C) (F := F) (Payload := Payload) B) :
    mscChoice B γ B = [γ] := by
  simp [mscChoice, singletonWord]

/-- For X ≠ B, the X-component of mscChoice γ B is empty. -/
@[simp]
theorem mscChoice_other (B X : L)
    (γ : AlphabetOf (C := C) (F := F) (Payload := Payload) B)
    (h : X ≠ B) :
    mscChoice B γ X = [] := by
  simp [mscChoice, h]

------------------------------------------------------------------------
-- Choice letters as AlphabetOf values
-- (convenience wrappers for use in mscChoice)
------------------------------------------------------------------------

/-- if_⊤(c@B) as an AlphabetOf B letter. -/
def choiceIfTrue (c : C) (B : L) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) B :=
  AlphabetOf.mkIfTrue B c

/-- if_⊥(c@B) as an AlphabetOf B letter. -/
def choiceIfFalse (c : C) (B : L) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) B :=
  AlphabetOf.mkIfFalse B c

/-- while_⊤(c@B) as an AlphabetOf B letter. -/
def choiceWhileTrue (c : C) (B : L) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) B :=
  AlphabetOf.mkWhileTrue B c

/-- while_⊥(c@B) as an AlphabetOf B letter. -/
def choiceWhileFalse (c : C) (B : L) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) B :=
  AlphabetOf.mkWhileFalse B c

------------------------------------------------------------------------
-- Shorthand canonical MSCs for choice letters
------------------------------------------------------------------------

/-- M_{if_⊤(c@B)}^B : the if-true canonical MSC. -/
def mscIfTrue (c : C) (B : L) : WordTuple L C F Payload :=
  mscChoice B (choiceIfTrue c B)

/-- M_{if_⊥(c@B)}^B : the if-false canonical MSC. -/
def mscIfFalse (c : C) (B : L) : WordTuple L C F Payload :=
  mscChoice B (choiceIfFalse c B)

/-- M_{while_⊤(c@B)}^B : the while-true canonical MSC. -/
def mscWhileTrue (c : C) (B : L) : WordTuple L C F Payload :=
  mscChoice B (choiceWhileTrue c B)

/-- M_{while_⊥(c@B)}^B : the while-false canonical MSC. -/
def mscWhileFalse (c : C) (B : L) : WordTuple L C F Payload :=
  mscChoice B (choiceWhileFalse c B)

------------------------------------------------------------------------
-- M_{x⃗,y⃗}^{A,f} : the canonical action MSC (def:base-msc, fourth bullet)
------------------------------------------------------------------------

/-- `mscAct A ys f xs` is the canonical action MSC M_{x⃗,y⃗}^{A,f}.
    • w_A = [act A(y⃗) := f(x⃗)]
    • w_X = []  for X ≠ A -/
def mscAct (A : L) (ys : Payload) (f : F) (xs : Payload) :
    WordTuple L C F Payload :=
  fun X =>
    if h : X = A then
      h ▸ singletonWord (AlphabetOf.mkAct (C := C) A ys f xs)
    else
      []

/-- The A-component of mscAct is [act A(ys) := f(xs)]. -/
@[simp]
theorem mscAct_owner (A : L) (ys : Payload) (f : F) (xs : Payload) :
    mscAct (C := C) A ys f xs A = [AlphabetOf.mkAct A ys f xs] := by
  simp [mscAct, singletonWord]

/-- For X ≠ A, the X-component of mscAct is empty. -/
@[simp]
theorem mscAct_other (A X : L) (ys : Payload) (f : F) (xs : Payload)
    (h : X ≠ A) :
    mscAct (C := C) A ys f xs X = [] := by
  simp [mscAct, h]

------------------------------------------------------------------------
-- M_{x⃗,y⃗}^{A→B} : the canonical message MSC (def:base-msc, fifth bullet)
------------------------------------------------------------------------

/-- `mscMsg A xs B ys h` is the canonical message MSC M_{x⃗,y⃗}^{A→B}.
    • w_A = [send A(x⃗) → B]
    • w_B = [recv B(y⃗) ← A]
    • w_X = []  for X ∉ {A, B}
    The proof `h : A ≠ B` is required (no self-channels). -/
def mscMsg (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B) : WordTuple L C F Payload :=
  fun X =>
    if hA : X = A then
      hA ▸ singletonWord (AlphabetOf.mkSend (C := C) (F := F) A xs B h)
    else if hB : X = B then
      hB ▸ singletonWord (AlphabetOf.mkRecv (C := C) (F := F) B ys A)
    else
      []

/-- The A-component of mscMsg is [send A(xs) → B]. -/
@[simp]
theorem mscMsg_sender (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B) :
    mscMsg (C := C) (F := F) A xs B ys h A =
    [AlphabetOf.mkSend A xs B h] := by
  simp [mscMsg, singletonWord]

/-- The B-component of mscMsg is [recv B(ys) ← A]. -/
@[simp]
theorem mscMsg_receiver (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B) :
    mscMsg (C := C) (F := F) A xs B ys h B =
    [AlphabetOf.mkRecv B ys A] := by
  simp [mscMsg, singletonWord, h.symm]

/-- For X ∉ {A, B}, the X-component of mscMsg is empty. -/
@[simp]
theorem mscMsg_other (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B) (X : L) (hA : X ≠ A) (hB : X ≠ B) :
    mscMsg (C := C) (F := F) A xs B ys h X = [] := by
  simp [mscMsg, hA, hB]

end CanonicalMSCs

------------------------------------------------------------------------
-- Sanity checks
------------------------------------------------------------------------

section SanityChecks

-- ExLifeline, ExPayload come from MSCAgents.Syntax (imported via Alphabets).
-- ExCond, ExFun come from MSCAgents.Alphabets.
open ExLifeline

-- mscEmpty: all components are empty
example : @mscEmpty ExLifeline ExCond ExFun ExPayload A = [] := rfl
example : @mscEmpty ExLifeline ExCond ExFun ExPayload B = [] := rfl

-- mscMsg A→B: A gets send, B gets recv, C gets empty
example : mscMsg (C := ExCond) (F := ExFun) A ["hello"] B ["x"] (by decide) A =
          [AlphabetOf.mkSend A ["hello"] B (by decide)] := by simp
example : mscMsg (C := ExCond) (F := ExFun) A ["hello"] B ["x"] (by decide) B =
          [AlphabetOf.mkRecv B ["x"] A] := by simp
example : mscMsg (C := ExCond) (F := ExFun) A ["hello"] B ["x"] (by decide) C = [] := by
          simp [show (C : ExLifeline) ≠ A from by decide,
                show (C : ExLifeline) ≠ B from by decide]

end SanityChecks
