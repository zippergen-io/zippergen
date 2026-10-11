/-
  MSCAgents/WellTyped.lean
  =========================
  Formalization of Definition 5 (def:well-typed) from:
  "Provable Coordination for LLM Agents via Message Sequence Charts"

  Paper section: sec:semantics

  A global program is well typed if every message subterm
  `msg A(x⃗) → B(y⃗)` satisfies `match(x⃗, y⃗)`.
-/

import MSCAgents.Syntax
import MSCAgents.PayloadMatching

/-- Definition `def:well-typed`: every message node in the program tree uses
    compatible sender and receiver payloads. -/
def WellTypedProgram {L C F Payload : Type} [PayloadCompatiblePred Payload] :
    Prog L C F Payload → Prop
  | .eps => True
  | .msg _ xs _ ys _ => PayloadCompatible Payload xs ys
  | .act .. => True
  | .ite _ _ pTrue pFalse =>
      WellTypedProgram pTrue ∧ WellTypedProgram pFalse
  | .whileLoop _ _ pBody pExit =>
      WellTypedProgram pBody ∧ WellTypedProgram pExit
  | .seq p1 p2 =>
      WellTypedProgram p1 ∧ WellTypedProgram p2
