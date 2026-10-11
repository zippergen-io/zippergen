/-
  MSCAgents/ParticipationSets.lean
  ================================
  Formalization of Definition 12 (def:participation-sets) from sec:projection of:
  "Provable Coordination for LLM Agents via Message Sequence Charts"
-/

import MSCAgents.Syntax

/-- Structural may-participation set `𝓛(P)` from Definition
    `def:participation-sets`, represented as a predicate on lifelines. -/
def participationSet {L C F Payload : Type} :
    Prog L C F Payload → L → Prop
  | .eps => fun _ => False
  | .msg A _ B _ _ => fun X => X = A ∨ X = B
  | .act A _ _ _ => fun X => X = A
  | .seq P1 P2 => fun X => participationSet P1 X ∨ participationSet P2 X
  | .ite _ B PTrue PFalse =>
      fun X => X = B ∨ participationSet PTrue X ∨ participationSet PFalse X
  | .whileLoop _ B PBody PExit =>
      fun X => X = B ∨ participationSet PBody X ∨ participationSet PExit X

notation "𝓛ₛ(" P ")" => participationSet P

section MembershipLemmas

variable {L C F Payload : Type}

@[simp] theorem participationSet_eps (X : L) :
    participationSet (Prog.eps : Prog L C F Payload) X = False := rfl

@[simp] theorem participationSet_msg (A B X : L) (xs ys : Payload) (h : A ≠ B) :
    participationSet (Prog.msg A xs B ys h : Prog L C F Payload) X ↔ X = A ∨ X = B := Iff.rfl

@[simp] theorem participationSet_act (A X : L) (ys : Payload) (f : F) (xs : Payload) :
    participationSet (Prog.act A ys f xs : Prog L C F Payload) X ↔ X = A := Iff.rfl

@[simp] theorem participationSet_seq (P1 P2 : Prog L C F Payload) (X : L) :
    participationSet (P1 ;; P2) X ↔ participationSet P1 X ∨ participationSet P2 X := Iff.rfl

@[simp] theorem participationSet_ite (c : C) (B X : L)
    (PTrue PFalse : Prog L C F Payload) :
    participationSet (Prog.ite c B PTrue PFalse) X ↔
      X = B ∨ participationSet PTrue X ∨ participationSet PFalse X := Iff.rfl

@[simp] theorem participationSet_while (c : C) (B X : L)
    (PBody PExit : Prog L C F Payload) :
    participationSet (Prog.whileLoop c B PBody PExit) X ↔
      X = B ∨ participationSet PBody X ∨ participationSet PExit X := Iff.rfl

end MembershipLemmas
