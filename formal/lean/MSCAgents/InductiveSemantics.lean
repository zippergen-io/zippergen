/-
  MSCAgents/InductiveSemantics.lean
  =================================
  Formalization of:
    • Definition 8  (def:inductive-msc)  — Inductive MSC Semantics ⟦P⟧

  Paper: "Provable Coordination for LLM Agents via Message Sequence Charts"
  Section: sec:semantics

  From the paper (def:inductive-msc):
    ⟦ε⟧                           = {M_ε}
    ⟦msg A(x⃗) → B(y⃗)⟧           = {M_{x⃗,y⃗}^{A→B}}
    ⟦act A(y⃗) := f(x⃗)⟧           = {M_{x⃗,y⃗}^{A,f}}
    ⟦P₁ ; P₂⟧                     = {M₁ ∘ M₂ | M₁ ∈ ⟦P₁⟧, M₂ ∈ ⟦P₂⟧}
    ⟦if c@B then P_⊤ else P_⊥⟧   = {M_{if_⊤(c@B)}^B ∘ M | M ∈ ⟦P_⊤⟧}
                                  ∪ {M_{if_⊥(c@B)}^B ∘ M | M ∈ ⟦P_⊥⟧}
    ⟦while c@B do P_body exit P_exit⟧
      = ⋃_{k≥0} { (M_{while_⊤}^B ∘ M₁) ∘ ⋯ ∘ (M_{while_⊤}^B ∘ Mₖ) ∘ (M_{while_⊥}^B ∘ M_exit)
                 | Mᵢ ∈ ⟦P_body⟧, M_exit ∈ ⟦P_exit⟧ }

-/

import MSCAgents.Syntax
import MSCAgents.Alphabets
import MSCAgents.MSCConcat
import MSCAgents.CanonicalMSC
import MSCAgents.MSC
import MSCAgents.WellTyped

------------------------------------------------------------------------
-- Semantic domain: sets of word-tuples
------------------------------------------------------------------------

/-- The semantic domain: a set of word-tuples (i.e., a set of potential MSCs).

    Since `Set` is not in scope without Mathlib, we represent sets as predicates
    `WordTuple L C F Payload → Prop`. -/
abbrev MSCSet (L C F Payload : Type) := WordTuple L C F Payload → Prop

------------------------------------------------------------------------
-- Inductive MSC semantics (def:inductive-msc)
------------------------------------------------------------------------

section InductiveSemantics

variable {L C F Payload : Type} [DecidableEq L]

/-- `mscSemantics P` computes the set of word-tuples (MSC semantics) ⟦P⟧.

    This is the central definition of the paper (def:inductive-msc),
    defined by structural recursion on the program P. -/
def mscSemantics :
    Prog L C F Payload → MSCSet L C F Payload
  -- ⟦ε⟧ = {M_ε}
  | .eps =>
      fun M => M = mscEmpty

  -- ⟦msg A(xs) → B(ys)⟧ = {M_{xs,ys}^{A→B}}
  | .msg A xs B ys h =>
      fun M => M = mscMsg A xs B ys h

  -- ⟦act A(ys) := f(xs)⟧ = {M_{xs,ys}^{A,f}}
  | .act A ys f xs =>
      fun M => M = mscAct A ys f xs

  -- ⟦P₁ ; P₂⟧ = {M₁ ∘ M₂ | M₁ ∈ ⟦P₁⟧, M₂ ∈ ⟦P₂⟧}
  | .seq P1 P2 =>
      fun M => ∃ M1, mscSemantics P1 M1 ∧ ∃ M2, mscSemantics P2 M2 ∧ M = M1 ∘ₘ M2

  -- ⟦if c@B then P_⊤ else P_⊥⟧
  --   = {M_{if_⊤(c@B)}^B ∘ M | M ∈ ⟦P_⊤⟧}
  --   ∪ {M_{if_⊥(c@B)}^B ∘ M | M ∈ ⟦P_⊥⟧}
  | .ite c B pTrue pFalse =>
      fun M => (∃ M', mscSemantics pTrue  M' ∧ M = mscIfTrue c B ∘ₘ M') ∨
               (∃ M', mscSemantics pFalse M' ∧ M = mscIfFalse c B ∘ₘ M')

  -- ⟦while c@B do P_body exit P_exit⟧
  --   = ⋃_{k≥0} { (whileTrue ∘ M₁) ∘ ⋯ ∘ (whileTrue ∘ Mₖ) ∘ (whileFalse ∘ M_exit)
  --               | Mᵢ ∈ ⟦P_body⟧, M_exit ∈ ⟦P_exit⟧ }
  | .whileLoop c B pBody pExit =>
      fun M =>
        ∃ k : Nat,
        ∃ bodies : Fin k → WordTuple L C F Payload,
        (∀ i, mscSemantics pBody (bodies i)) ∧
        ∃ exitMSC, mscSemantics pExit exitMSC ∧
          M =
            -- k iterations: (whileTrue^B ∘ M_body_i) for i = 0..k-1
            (WordTuple.concatList
              (List.ofFn (fun i => mscWhileTrue c B ∘ₘ bodies i)))
            -- followed by: whileFalse^B ∘ M_exit
            ∘ₘ (mscWhileFalse c B ∘ₘ exitMSC)

/-- Notation: ⟦P⟧ for the MSC semantics of program P. -/
notation "⟦" P "⟧" => mscSemantics P

end InductiveSemantics

------------------------------------------------------------------------
-- Basic existence: well-typed programs have some MSC semantics
------------------------------------------------------------------------

section SemanticsNonempty

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload]

/-- Every well-typed global program denotes at least one MSC in the inductive
    semantics. This is the semantic existence fact used by the zipper/control
    proofs to pick a concrete continuation when no branch decision has yet been
    fixed by the current prefix. -/
theorem mscSemantics_nonempty
    (prog : Prog L C F Payload)
    (hProg : WellTypedProgram prog) :
    ∃ M : WordTuple L C F Payload, ⟦prog⟧ M := by
  induction prog with
  | eps =>
      exact ⟨mscEmpty (L := L) (C := C) (F := F) (Payload := Payload), rfl⟩
  | msg A xs B ys h =>
      exact ⟨mscMsg (C := C) (F := F) A xs B ys h, rfl⟩
  | act A ys f xs =>
      exact ⟨mscAct (C := C) (F := F) A ys f xs, rfl⟩
  | seq P1 P2 ih1 ih2 =>
      rcases hProg with ⟨hP1, hP2⟩
      rcases ih1 hP1 with ⟨M1, hM1⟩
      rcases ih2 hP2 with ⟨M2, hM2⟩
      exact ⟨M1 ∘ₘ M2, ⟨M1, hM1, M2, hM2, rfl⟩⟩
  | ite c B pTrue pFalse ihTrue ihFalse =>
      rcases hProg with ⟨hTrue, hFalse⟩
      rcases ihTrue hTrue with ⟨MTrue, hMTrue⟩
      exact ⟨mscIfTrue (C := C) (F := F) (Payload := Payload) c B ∘ₘ MTrue,
        Or.inl ⟨MTrue, hMTrue, rfl⟩⟩
  | whileLoop c B pBody pExit ihBody ihExit =>
      rcases hProg with ⟨_hBody, hExit⟩
      rcases ihExit hExit with ⟨MExit, hMExit⟩
      exact ⟨mscWhileFalse (C := C) (F := F) (Payload := Payload) c B ∘ₘ MExit,
        ⟨0, Fin.elim0, by intro i; exact False.elim (Fin.elim0 i), MExit, hMExit, by
          simp [WordTuple.concatList, WordTuple.empty, WordTuple.concat_eps_left]⟩⟩

end SemanticsNonempty

------------------------------------------------------------------------
-- Remark 2 (rem:sem-complete): Completeness of Inductive Semantics
------------------------------------------------------------------------

/-
  The remark states:
    "For every program P and every M ∈ ⟦P⟧, the MSC M is complete."

  This is proved by structural induction on P.
  The proof now uses the concrete complete-MSC theorems from `MSC.lean`
  directly, rather than staging through placeholder axioms.
-/

section SemanticsCompleteness

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload] [Fintype L]

------------------------------------------------------------------------
-- Auxiliary: iterated concat of complete MSCs is complete
------------------------------------------------------------------------

/-- A list of complete MSCs, when concatenated (left fold), is complete.
    This follows by repeated application of `IsMSCPredicate.concat_complete`
    starting from the empty MSC (which is complete). -/
theorem concatList_complete
    (Ms : List (WordTuple L C F Payload))
    (hMs : ∀ M ∈ Ms, IsCompleteMSC M) :
    IsCompleteMSC (WordTuple.concatList Ms) := by
  induction Ms with
  | nil =>
    simp [WordTuple.concatList]
    exact mscEmpty_isCompleteMSC
  | cons hd tl ih =>
    simp [WordTuple.concatList_cons]
    apply concat_complete_complete
    · exact hMs hd List.mem_cons_self
    · exact ih (fun M hM => hMs M (List.mem_cons_of_mem hd hM))

------------------------------------------------------------------------
-- Main theorem: rem:sem-complete
------------------------------------------------------------------------

/-- **Remark rem:sem-complete** (Completeness of Inductive Semantics):
    For every program P and every M ∈ ⟦P⟧, the MSC M is complete.

    Proof: by structural induction on P. -/
theorem mscSemantics_complete
    (prog : Prog L C F Payload)
    (hProg : WellTypedProgram prog)
    (M : WordTuple L C F Payload)
    (hM : ⟦prog⟧ M) :
    IsCompleteMSC M := by
  induction prog generalizing M with

  -- Base case: ε
  -- ⟦ε⟧ = {M_ε}, and M_ε is complete.
  | eps =>
    -- hM : M ∈ {mscEmpty}, so M = mscEmpty
    simp [mscSemantics] at hM
    subst hM
    exact mscEmpty_isCompleteMSC

  -- Base case: msg A(xs) → B(ys)
  -- ⟦msg A→B⟧ = {mscMsg A xs B ys h}, which is complete.
  | msg A xs B ys h =>
    simp [mscSemantics] at hM
    subst hM
    exact mscMsg_isCompleteMSC A xs B ys h (by simpa [WellTypedProgram] using hProg)

  -- Base case: act A(ys) := f(xs)
  -- ⟦act⟧ = {mscAct ...}, which is complete.
  | act A ys f xs =>
    simp [mscSemantics] at hM
    subst hM
    exact mscAct_isCompleteMSC A ys f xs

  -- Inductive case: P₁ ; P₂
  -- ⟦P₁;P₂⟧ = {M₁ ∘ M₂ | M₁ ∈ ⟦P₁⟧, M₂ ∈ ⟦P₂⟧}
  -- By IH: M₁ complete, M₂ complete ⟹ M₁ ∘ M₂ complete.
  | seq P1 P2 ih1 ih2 =>
    rcases hProg with ⟨hP1, hP2⟩
    simp [mscSemantics] at hM
    obtain ⟨M1, hM1, M2, hM2, rfl⟩ := hM
    exact concat_complete_complete M1 M2 (ih1 hP1 M1 hM1) (ih2 hP2 M2 hM2)

  -- Inductive case: if c@B then P_⊤ else P_⊥
  -- Either M = ifTrue ∘ M' with M' ∈ ⟦P_⊤⟧, or M = ifFalse ∘ M' with M' ∈ ⟦P_⊥⟧.
  -- In both cases: canonical choice MSC is complete + IH gives M' complete
  --               ⟹ concatenation is complete.
  | ite c B pTrue pFalse ihTrue ihFalse =>
    rcases hProg with ⟨hTrue, hFalse⟩
    simp [mscSemantics] at hM
    rcases hM with ⟨M', hM'_in, rfl⟩ | ⟨M', hM'_in, rfl⟩
    · -- true branch
      exact concat_complete_complete _ _ (mscIfTrue_isCompleteMSC c B) (ihTrue hTrue M' hM'_in)
    · -- false branch
      exact concat_complete_complete _ _ (mscIfFalse_isCompleteMSC c B) (ihFalse hFalse M' hM'_in)

  -- Inductive case: while c@B do P_body exit P_exit
  -- M = (whileTrue ∘ M₁) ∘ ⋯ ∘ (whileTrue ∘ Mₖ) ∘ (whileFalse ∘ M_exit)
  -- Each (whileTrue ∘ Mᵢ) is complete by IH and concat_complete.
  -- The whole iterated concat is complete by concatList_complete.
  -- Finally concat with (whileFalse ∘ M_exit) preserves completeness.
  | whileLoop c B pBody pExit ihBody ihExit =>
    rcases hProg with ⟨hBody, hExitProg⟩
    simp [mscSemantics] at hM
    obtain ⟨k, bodies, hBodies, exitMSC, hExit, rfl⟩ := hM
    -- Each iteration step is complete
    have step_complete : ∀ i : Fin k,
        IsCompleteMSC (mscWhileTrue c B ∘ₘ bodies i) := by
      intro i
      exact concat_complete_complete _ _ (mscWhileTrue_isCompleteMSC c B)
            (ihBody hBody (bodies i) (hBodies i))
    -- The list of iteration steps is complete
    have list_complete :
        IsCompleteMSC (WordTuple.concatList
          (List.ofFn (fun i => mscWhileTrue c B ∘ₘ bodies i))) := by
      apply concatList_complete
      intro M hM
      rw [List.mem_ofFn] at hM
      obtain ⟨i, rfl⟩ := hM
      exact step_complete i
    -- The exit step is complete
    have exit_complete : IsCompleteMSC (mscWhileFalse c B ∘ₘ exitMSC) :=
      concat_complete_complete _ _ (mscWhileFalse_isCompleteMSC c B) (ihExit hExitProg exitMSC hExit)
    -- Combine: iterations ∘ exit is complete
    exact concat_complete_complete _ _ list_complete exit_complete

end SemanticsCompleteness

------------------------------------------------------------------------
-- Sanity checks on the semantics
------------------------------------------------------------------------

section SemanticsChecks

-- ExLifeline, ExPayload come from MSCAgents.Syntax; ExCond, ExFun from MSCAgents.Alphabets.
open ExLifeline

-- ⟦ε⟧ = fun M => M = mscEmpty
example : @mscSemantics ExLifeline ExCond ExFun ExPayload _
            Prog.eps = (fun M => M = mscEmpty) := rfl

-- ⟦msg A → B⟧ = fun M => M = mscMsg A xs B ys h
example : @mscSemantics ExLifeline ExCond ExFun ExPayload _
            (Prog.msg A [.val "hello"] B [.var] (by decide)) =
          (fun M => M = mscMsg A [.val "hello"] B [.var] (by decide)) := rfl

end SemanticsChecks
