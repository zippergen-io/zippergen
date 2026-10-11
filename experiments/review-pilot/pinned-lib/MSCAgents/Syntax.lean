/-
  MSCAgents/Syntax.lean
  =====================
  Formalization of Definition 1 (def:phase-syntax) from:
  "Provable Coordination for LLM Agents via Message Sequence Charts"

  This file defines the abstract syntax of global agent programs (Prog).

  Paper section: sec:syntax (Global Agent Programs / Syntax)

  Grammar from the paper:
    P ::= ε
        | msg A(x⃗) → B(y⃗)
        | act A(y⃗) := f(x⃗)
        | if c@B then P_⊤ else P_⊥
        | while c@B do P_body exit P_exit
        | P₁ ; P₂

  Key constraints (from the paper):
    - 𝓛 is a finite set of lifelines (parameterized here as a type variable `L`)
    - In msg A(x⃗) → B(y⃗), we require A ≠ B (no self-channels)
    - Payloads are abstract tuples (parameterized over a type `Payload`)
    - The phase composition (;) is associative with neutral element ε
      (this identity holds in the semantics; see InductiveSemantics.lean)

  We parameterize over:
    L       : the type of lifelines
    C       : the type of conditions (guards for if/while)
    F       : the type of action function names
    Payload : the type of payload tuples (modeling x⃗, y⃗)
-/

import MSCAgents.Finiteness
import MSCAgents.PayloadMatching

/-- Abstract syntax of global programs (def:phase-syntax).

    `Prog L C F Payload` is the inductive type of global coordination programs
    over lifelines `L`, conditions `C`, action functions `F`, and payload
    tuples `Payload`.

    The paper's grammar is reproduced directly as constructors below.
-/
inductive Prog (L : Type) (C : Type) (F : Type) (Payload : Type) : Type where

  /-- ε — the empty (no-op) program. -/
  | eps : Prog L C F Payload

  /-- `msg sender xs receiver ys h`
      models `msg A(x⃗) → B(y⃗)`.
      Sender A sends payload xs; receiver B receives payload ys.
      The proof `h : sender ≠ receiver` enforces the paper's constraint:
      "in msg A(x⃗) → B(y⃗) we assume A ≠ B (no self-channels)". -/
  | msg (sender : L) (xs : Payload) (receiver : L) (ys : Payload)
      (h : sender ≠ receiver) : Prog L C F Payload

  /-- `act lifeline ys f xs`
      models `act A(y⃗) := f(x⃗)`.
      Lifeline A executes action f with input xs and output ys
      (e.g., an LLM call). -/
  | act (lifeline : L) (ys : Payload) (f : F) (xs : Payload)
      : Prog L C F Payload

  /-- `ite cond decider pTrue pFalse`
      models `if c@B then P_⊤ else P_⊥`.
      Conditional with condition c, decided at lifeline B. -/
  | ite (cond : C) (decider : L)
      (pTrue : Prog L C F Payload) (pFalse : Prog L C F Payload)
      : Prog L C F Payload

  /-- `whileLoop cond decider pBody pExit`
      models `while c@B do P_body exit P_exit`.
      Loop with condition c decided at lifeline B, loop body pBody,
      and explicit exit continuation pExit. -/
  | whileLoop (cond : C) (decider : L)
      (pBody : Prog L C F Payload) (pExit : Prog L C F Payload)
      : Prog L C F Payload

  /-- `seq p1 p2`
      models `P₁ ; P₂`.
      Phase (sequential) composition.
      The paper states that ; is associative with neutral element ε
      in the semantics (see InductiveSemantics.lean). -/
  | seq (p1 : Prog L C F Payload) (p2 : Prog L C F Payload)
      : Prog L C F Payload

------------------------------------------------------------------------
-- Notation: mirrors the paper's ; for sequential composition
------------------------------------------------------------------------

/-- Sequential composition, written `;;` to avoid conflict with Lean's `;`. -/
infixl:50 " ;; " => Prog.seq

------------------------------------------------------------------------
-- Well-formedness note
------------------------------------------------------------------------

section WellTypedness

variable {L C F Payload : Type}

/-- The `msg` constructor requires a proof `h : sender ≠ receiver`.
    This is the paper's constraint "no self-channels".
    Because `h` is bundled into the constructor, any well-typed `msg` term
    automatically satisfies the no-self-channel condition. -/
theorem Prog.msg_neq
    (sender receiver : L) (_xs _ys : Payload)
    (h : sender ≠ receiver) :
    sender ≠ receiver := h

end WellTypedness

------------------------------------------------------------------------
-- Example programs (smoke tests documenting the syntax)
------------------------------------------------------------------------

section Examples

/-- Example lifelines for documentation. -/
inductive ExLifeline : Type | A | B | C deriving DecidableEq, Repr

/-- Example payload type: lists of strings (modeling x⃗, y⃗). -/
abbrev ExPayload := PayloadTuple String

instance : PayloadCompatiblePred ExPayload where
  compat xs ys := PayloadMatches xs ys

instance : Fintype ExLifeline where
  elems := [ExLifeline.A, ExLifeline.B, ExLifeline.C]
  nodup := by simp
  complete
    | ExLifeline.A => by simp
    | ExLifeline.B => by simp
    | ExLifeline.C => by simp

open ExLifeline

-- msg A(("hello")) → B((x))
#check (Prog.msg A [PayloadComp.val "hello"] B [PayloadComp.var] (by decide)
          : Prog ExLifeline Bool String ExPayload)

-- act A(("y")) := "llm_call"(("x"))
#check (Prog.act A [PayloadComp.val "y"] "llm_call" [PayloadComp.val "x"] :
    Prog ExLifeline Bool String ExPayload)

-- if true@B then ε else ε
#check (Prog.ite true B Prog.eps Prog.eps :
    Prog ExLifeline Bool String ExPayload)

-- while true@B do ε exit ε
#check (Prog.whileLoop true B Prog.eps Prog.eps :
    Prog ExLifeline Bool String ExPayload)

-- (msg A → B) ;; ε
#check ((Prog.msg A [PayloadComp.val "hello"] B [PayloadComp.var] (by decide)) ;;
        Prog.eps :
        Prog ExLifeline Bool String ExPayload)

end Examples
