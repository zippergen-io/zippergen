/-
  MSCAgents/Alphabets.lean
  ========================
  Formalization of Definition 2 (def:alphabets) from:
  "Provable Coordination for LLM Agents via Message Sequence Charts"

  This file defines the local-word alphabets Σ_A for each lifeline A.

  Paper section: sec:semantics (Semantics / Inductive MSC Model)

  From the paper (def:alphabets):
    For each lifeline A ∈ 𝓛, the local-word alphabet Σ_A contains:
      • send letters:    send A(x⃗) → B
      • receive letters: recv A(y⃗) ← B
      • local action letters: act A(y⃗) := f(x⃗)
      • choice letters: if_⊤(c@A), if_⊥(c@A), while_⊤(c@A), while_⊥(c@A)

  Design:
    We model the alphabet as a single inductive type `Letter L C F Payload`,
    each constructor carrying the owner lifeline.
    The alphabet Σ_A for a specific lifeline A is then the subtype
    `AlphabetOf A = { ℓ : Letter | ℓ.owner = A }`.
    Local words are lists of such letters: `LocalWord A = List (AlphabetOf A)`.
-/

import MSCAgents.Syntax

------------------------------------------------------------------------
-- Letter type — elements of ⋃_A Σ_A
------------------------------------------------------------------------

/-- A single letter in a local-word alphabet (def:alphabets).

    Every letter carries the lifeline that "owns" the event.
    Parameters:
      L       : lifeline type
      C       : condition type
      F       : action function type
      Payload : payload tuple type -/
inductive Letter (L : Type) (C : Type) (F : Type) (Payload : Type) : Type where

  /-- `sendLetter A xs B h`  models  `send A(x⃗) → B`
      Lifeline A sends payload xs to lifeline B. The proof `h` enforces
      the no-self-channel constraint. -/
  | sendLetter (owner : L) (xs : Payload) (target : L) (h : owner ≠ target) :
      Letter L C F Payload

  /-- `recvLetter A ys B`  models  `recv A(y⃗) ← B`
      Lifeline A receives payload ys from lifeline B. -/
  | recvLetter (owner : L) (ys : Payload) (source : L) : Letter L C F Payload

  /-- `actLetter A ys f xs`  models  `act A(y⃗) := f(x⃗)`
      Lifeline A executes action f with inputs xs and outputs ys. -/
  | actLetter (owner : L) (ys : Payload) (f : F) (xs : Payload)
      : Letter L C F Payload

  /-- `ifTrueLetter c A`  models  `if_⊤(c@A)`
      Choice letter: lifeline A evaluated condition c to ⊤. -/
  | ifTrueLetter (cond : C) (owner : L) : Letter L C F Payload

  /-- `ifFalseLetter c A`  models  `if_⊥(c@A)`
      Choice letter: lifeline A evaluated condition c to ⊥. -/
  | ifFalseLetter (cond : C) (owner : L) : Letter L C F Payload

  /-- `whileTrueLetter c A`  models  `while_⊤(c@A)`
      Choice letter: loop iteration (condition c true at A). -/
  | whileTrueLetter (cond : C) (owner : L) : Letter L C F Payload

  /-- `whileFalseLetter c A`  models  `while_⊥(c@A)`
      Choice letter: loop exit (condition c false at A). -/
  | whileFalseLetter (cond : C) (owner : L) : Letter L C F Payload

------------------------------------------------------------------------
-- Owner projection
------------------------------------------------------------------------

/-- Every letter carries exactly one owner lifeline.
    This function extracts it. -/
def Letter.owner {L C F Payload : Type} :
    Letter L C F Payload → L
  | .sendLetter  A _ _ _ => A
  | .recvLetter  A _ _   => A
  | .actLetter   A _ _ _ => A
  | .ifTrueLetter  _ A   => A
  | .ifFalseLetter _ A   => A
  | .whileTrueLetter  _ A => A
  | .whileFalseLetter _ A => A

------------------------------------------------------------------------
-- The alphabet Σ_A for a fixed lifeline A (def:alphabets)
------------------------------------------------------------------------

/-- `AlphabetOf A` is the local-word alphabet Σ_A:
    the set of all letters whose owner is lifeline A.

    This is the subtype `{ ℓ : Letter L C F Payload // ℓ.owner = A }`,
    directly matching the paper's definition of Σ_A. -/
abbrev AlphabetOf {L C F Payload : Type} (A : L) : Type :=
  { ℓ : Letter L C F Payload // ℓ.owner = A }

/-- A local word for lifeline A is a finite sequence of Σ_A-letters.
    This corresponds to `w_A ∈ (Σ_A)^*` in the paper. -/
abbrev LocalWord {L C F Payload : Type} (A : L) : Type :=
  List (AlphabetOf (C := C) (F := F) (Payload := Payload) A)

------------------------------------------------------------------------
-- Smart constructors for AlphabetOf
------------------------------------------------------------------------

namespace AlphabetOf

variable {L C F Payload : Type}

/-- `send A(xs) → B` as an element of Σ_A. -/
def mkSend (A : L) (xs : Payload) (B : L) (h : A ≠ B) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) A :=
  ⟨Letter.sendLetter A xs B h, rfl⟩

/-- `recv A(ys) ← B` as an element of Σ_A. -/
def mkRecv (A : L) (ys : Payload) (B : L) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) A :=
  ⟨Letter.recvLetter A ys B, rfl⟩

/-- `act A(ys) := f(xs)` as an element of Σ_A. -/
def mkAct (A : L) (ys : Payload) (f : F) (xs : Payload) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) A :=
  ⟨Letter.actLetter A ys f xs, rfl⟩

/-- `if_⊤(c@A)` as an element of Σ_A. -/
def mkIfTrue (A : L) (c : C) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) A :=
  ⟨Letter.ifTrueLetter c A, rfl⟩

/-- `if_⊥(c@A)` as an element of Σ_A. -/
def mkIfFalse (A : L) (c : C) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) A :=
  ⟨Letter.ifFalseLetter c A, rfl⟩

/-- `while_⊤(c@A)` as an element of Σ_A. -/
def mkWhileTrue (A : L) (c : C) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) A :=
  ⟨Letter.whileTrueLetter c A, rfl⟩

/-- `while_⊥(c@A)` as an element of Σ_A. -/
def mkWhileFalse (A : L) (c : C) :
    AlphabetOf (C := C) (F := F) (Payload := Payload) A :=
  ⟨Letter.whileFalseLetter c A, rfl⟩

end AlphabetOf

------------------------------------------------------------------------
-- Word tuples M = (w_A)_{A ∈ 𝓛}
------------------------------------------------------------------------

/-- A word tuple: a family of local words indexed by lifelines.
    Represents M = (w_A)_{A ∈ 𝓛} with w_A ∈ (Σ_A)^* for each A.
    This is the ambient structure in which MSCs are defined. -/
abbrev WordTuple (L C F Payload : Type) : Type :=
  ∀ (A : L), LocalWord (C := C) (F := F) (Payload := Payload) A

/-- The empty word tuple: all components are the empty word ε.
    Used for M_ε and as the identity for concatenation. -/
def WordTuple.empty {L C F Payload : Type} : WordTuple L C F Payload :=
  fun _ => []

------------------------------------------------------------------------
-- Sanity checks
------------------------------------------------------------------------

section SanityChecks

-- ExLifeline and ExPayload come from the imported MSCAgents.Syntax.
-- We add ExCond and ExFun here for the alphabet-specific examples.
abbrev ExCond := Bool
abbrev ExFun  := String

open ExLifeline

-- Send letter for lifeline A
#check (AlphabetOf.mkSend (C := ExCond) (F := ExFun)
          A [PayloadComp.val "hello"] B (by decide) :
        AlphabetOf (C := ExCond) (F := ExFun) (Payload := ExPayload) A)

-- Local word for lifeline A: [send]
#check ([ AlphabetOf.mkSend (C := ExCond) (F := ExFun) A [PayloadComp.val "hello"] B (by decide)
        ] : LocalWord (C := ExCond) (F := ExFun) (Payload := ExPayload) A)

-- Empty word tuple
#check (WordTuple.empty : WordTuple ExLifeline ExCond ExFun ExPayload)

end SanityChecks
