/-
  MSCAgents/MSCConcat.lean
  ========================
  Formalization of:
    • Definition 5  (def:msc-concat)  — MSC Concatenation
    • Remark 1      (rem:concat-msc)  — Concatenation with a Complete Prefix

  Paper: "Provable Coordination for LLM Agents via Message Sequence Charts"
  Section: sec:semantics

  From the paper:

  Definition (def:msc-concat):
    Let M₁ = (u_A)_A and M₂ = (v_A)_A.
    M₁ ∘ M₂ := (u_A v_A)_{A ∈ 𝓛}

  Remark (rem:concat-msc):
    The concatenation of two MSCs is not necessarily an MSC.
    If M₁ is a complete MSC and M₂ is an MSC,
      then M₁ ∘ M₂ is an MSC.
    If moreover M₂ is complete,
      then M₁ ∘ M₂ is complete.

  Note on the MSC predicate:
    The full definition of MSC well-typedness (def:msc) requires
    the FIFO relation and acyclicity infrastructure (def:tuple-fifo, def:msc).
    Here we axiomatize `IsMSCPredicate` capturing the properties needed
    for the remark. The concrete predicate is given in MSC.lean.
-/

import MSCAgents.Alphabets

------------------------------------------------------------------------
-- MSC Concatenation (def:msc-concat)
------------------------------------------------------------------------

/-- MSC concatenation: M₁ ∘ M₂ := (u_A ++ v_A)_{A ∈ 𝓛}.

    For each lifeline A, the local word of the concatenation is the
    concatenation of the two local words. This is pointwise list append. -/
def WordTuple.concat {L C F Payload : Type}
    (M1 M2 : WordTuple L C F Payload) : WordTuple L C F Payload :=
  fun A => M1 A ++ M2 A

/-- Infix notation for MSC concatenation (mirrors the paper's ∘). -/
infixl:60 " ∘ₘ " => WordTuple.concat

------------------------------------------------------------------------
-- Algebraic laws for concat
------------------------------------------------------------------------

section ConcatAlgebra

variable {L C F Payload : Type}
variable (M M1 M2 M3 : WordTuple L C F Payload)

/-- The empty word tuple is a left identity for ∘ₘ. -/
@[simp]
theorem WordTuple.concat_eps_left :
    WordTuple.empty ∘ₘ M = M := by
  funext A
  simp [WordTuple.concat, WordTuple.empty]

/-- The empty word tuple is a right identity for ∘ₘ. -/
@[simp]
theorem WordTuple.concat_eps_right :
    M ∘ₘ WordTuple.empty = M := by
  funext A
  simp [WordTuple.concat, WordTuple.empty]

/-- MSC concatenation is associative. -/
@[simp]
theorem WordTuple.concat_assoc :
    (M1 ∘ₘ M2) ∘ₘ M3 = M1 ∘ₘ (M2 ∘ₘ M3) := by
  funext A
  simp [WordTuple.concat, List.append_assoc]

end ConcatAlgebra

------------------------------------------------------------------------
-- Abstract MSC predicate
--
-- We axiomatize the properties of "being an MSC" and "being complete"
-- needed for rem:concat-msc. The concrete instance is in MSC.lean,
-- where the FIFO relation and acyclicity are constructed.
------------------------------------------------------------------------

/-- Abstract predicate capturing MSC well-typedness and completeness.

    The paper's full MSC definition (def:msc) requires:
      1. No unmatched receives (in the FIFO relation ⊲_M)
      2. Label-compatible matched pairs
      3. Acyclic causality
    And completeness means every send is matched.

    We axiomatize these as a typeclass so the remark (rem:concat-msc) can
    be stated and proved in full generality, to be instantiated concretely
    in MSC.lean. -/
class IsMSCPredicate (L C F Payload : Type) where
  /-- The MSC well-typedness predicate. -/
  isMSC : WordTuple L C F Payload → Prop
  /-- Completeness: every send event is matched in ⊲_M. -/
  isComplete : WordTuple L C F Payload → Prop
  /-- A complete MSC is an MSC. -/
  complete_implies_msc : ∀ M, isComplete M → isMSC M
  /-- The empty word tuple is a complete MSC. -/
  empty_is_complete : isComplete (WordTuple.empty (L := L) (C := C) (F := F) (Payload := Payload))
  /-- Concatenation with a complete prefix: M₁ complete ∧ M₂ MSC → M₁ ∘ M₂ MSC. -/
  concat_msc : ∀ M1 M2,
      isComplete M1 → isMSC M2 → isMSC (M1 ∘ₘ M2)
  /-- Completeness under concatenation: both complete → concat complete. -/
  concat_complete : ∀ M1 M2,
      isComplete M1 → isComplete M2 → isComplete (M1 ∘ₘ M2)

------------------------------------------------------------------------
-- Remark 1 (rem:concat-msc): Concatenation with a Complete Prefix
------------------------------------------------------------------------

section RemConcatMSC

variable {L C F Payload : Type}
variable [P : IsMSCPredicate L C F Payload]

/-- **Remark rem:concat-msc** (part 1):
    If M₁ is a complete MSC and M₂ is an MSC, then M₁ ∘ M₂ is an MSC.

    Paper: "If M₁ is a complete MSC and M₂ is an MSC, then M₁ ∘ M₂ is an MSC." -/
theorem concat_complete_msc_is_msc
    (M1 M2 : WordTuple L C F Payload)
    (h1 : P.isComplete M1)
    (h2 : P.isMSC M2) :
    P.isMSC (M1 ∘ₘ M2) :=
  P.concat_msc M1 M2 h1 h2

/-- **Remark rem:concat-msc** (part 2):
    If M₁ is a complete MSC and M₂ is also complete, then M₁ ∘ M₂ is complete.

    Paper: "If moreover M₂ is complete, then M₁ ∘ M₂ is complete." -/
theorem concat_complete_complete_is_complete
    (M1 M2 : WordTuple L C F Payload)
    (h1 : P.isComplete M1)
    (h2 : P.isComplete M2) :
    P.isComplete (M1 ∘ₘ M2) :=
  P.concat_complete M1 M2 h1 h2

/-- Corollary: concatenation of two complete MSCs is an MSC. -/
theorem concat_two_complete_is_msc
    (M1 M2 : WordTuple L C F Payload)
    (h1 : P.isComplete M1)
    (h2 : P.isComplete M2) :
    P.isMSC (M1 ∘ₘ M2) :=
  P.complete_implies_msc _ (P.concat_complete M1 M2 h1 h2)

end RemConcatMSC

------------------------------------------------------------------------
-- Iterated concatenation — used for the while-loop semantics
------------------------------------------------------------------------

/-- Concatenate a list of word-tuples left-to-right.
    `concatList [M₁, …, Mₙ] = M₁ ∘ₘ … ∘ₘ Mₙ`  (empty list gives M_ε). -/
def WordTuple.concatList {L C F Payload : Type}
    (Ms : List (WordTuple L C F Payload)) : WordTuple L C F Payload :=
  Ms.foldl (· ∘ₘ ·) WordTuple.empty

@[simp]
theorem WordTuple.concatList_nil {L C F Payload : Type} :
    @WordTuple.concatList L C F Payload [] = WordTuple.empty := rfl

-- Helper: foldl with shifted accumulator equals acc ∘ₘ foldl with empty.
private theorem foldl_concat_shift {L C F Payload : Type}
    (acc : WordTuple L C F Payload)
    (Ms : List (WordTuple L C F Payload)) :
    Ms.foldl (· ∘ₘ ·) acc = acc ∘ₘ Ms.foldl (· ∘ₘ ·) WordTuple.empty := by
  induction Ms generalizing acc with
  | nil => simp [WordTuple.concat_eps_right]
  | cons hd tl ih =>
    simp only [List.foldl]
    -- After unfolding: goal is tl.foldl ... (acc ∘ₘ hd) = acc ∘ₘ tl.foldl ... (empty ∘ₘ hd)
    rw [ih (acc ∘ₘ hd), ih (WordTuple.empty ∘ₘ hd)]
    -- Goal: (acc ∘ₘ hd) ∘ₘ tl.foldl ... empty = acc ∘ₘ ((empty ∘ₘ hd) ∘ₘ tl.foldl ... empty)
    simp [WordTuple.concat_assoc, WordTuple.concat_eps_left]

/-- Unfolding lemma: prepending M to a list corresponds to M ∘ₘ concatList rest. -/
@[simp]
theorem WordTuple.concatList_cons {L C F Payload : Type}
    (M : WordTuple L C F Payload)
    (Ms : List (WordTuple L C F Payload)) :
    WordTuple.concatList (M :: Ms) = M ∘ₘ WordTuple.concatList Ms := by
  simp only [WordTuple.concatList, List.foldl]
  -- Goal: Ms.foldl ... (empty ∘ₘ M) = M ∘ₘ Ms.foldl ... empty
  rw [foldl_concat_shift (WordTuple.empty ∘ₘ M) Ms, WordTuple.concat_eps_left]
