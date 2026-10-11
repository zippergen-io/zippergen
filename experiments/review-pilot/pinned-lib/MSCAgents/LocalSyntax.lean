/-
  MSCAgents/LocalSyntax.lean
  ==========================
  Formalization of:
    • Definition 9  — Local Syntax (sec:distributed)
    • Definition 10 — Distributed Programs (sec:distributed)
  from:
  "Provable Coordination for LLM Agents via Message Sequence Charts"
-/

import MSCAgents.Syntax

/-- Local programs owned by a fixed lifeline `A`. This is the local syntax
    from the paper's `LocProg` grammar, specialized to one owner lifeline. -/
inductive LocProg (L : Type) (C : Type) (F : Type) (Payload : Type) (A : L) : Type where
  | eps : LocProg L C F Payload A
  | send (xs : Payload) (target : L) : LocProg L C F Payload A
  | recv (ys : Payload) (source : L) : LocProg L C F Payload A
  | seq (S1 S2 : LocProg L C F Payload A) : LocProg L C F Payload A
  | act (ys : Payload) (f : F) (xs : Payload) : LocProg L C F Payload A
  | recvIf (ys : Payload) (source : L)
      (STrue SFalse : LocProg L C F Payload A) : LocProg L C F Payload A
  | recvWhile (ys : Payload) (source : L)
      (SBody SExit : LocProg L C F Payload A) : LocProg L C F Payload A
  | localIf (c : C)
      (STrue SFalse : LocProg L C F Payload A) : LocProg L C F Payload A
  | localWhile (c : C)
      (SBody SExit : LocProg L C F Payload A) : LocProg L C F Payload A

/-- Sequential composition notation for local programs. -/
infixl:50 " ;;ₗ " => LocProg.seq

/-- A distributed program is one local program per lifeline. -/
abbrev DistProg (L C F Payload : Type) : Type :=
  ∀ A : L, LocProg L C F Payload A

section Examples

open ExLifeline

abbrev ExLoc := LocProg ExLifeline Bool String ExPayload A

#check (LocProg.send (L := ExLifeline) (C := Bool) (F := String)
          (Payload := ExPayload) (A := A) [PayloadComp.val "hello"] B : ExLoc)

#check (LocProg.recvIf (L := ExLifeline) (C := Bool) (F := String)
          (Payload := ExPayload) (A := A) [PayloadComp.var] B
          LocProg.eps LocProg.eps : ExLoc)

#check ((fun
          | A => LocProg.eps
          | B => LocProg.eps
          | C => LocProg.eps) : DistProg ExLifeline Bool String ExPayload)

end Examples
