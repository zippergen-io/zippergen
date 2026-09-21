/-
  MSCAgents/Finiteness.lean
  =========================
  Minimal finiteness interface used for finite lifeline sets.
-/

/-- A project-local finite enumeration typeclass.

    This supplies the fixed finite lifeline set assumed by the paper without
    depending on external finite-type libraries. -/
class Fintype (α : Type) where
  elems : List α
  nodup : elems.Nodup
  complete : ∀ x : α, x ∈ elems

namespace Fintype

variable {α : Type}

/-- The chosen finite enumeration. -/
def enum (α : Type) [inst : Fintype α] : List α :=
  inst.elems

/-- Completeness of the chosen enumeration. -/
theorem complete_enum [inst : Fintype α] (x : α) : x ∈ enum α :=
  inst.complete x

/-- The chosen enumeration has no duplicates. -/
theorem nodup_enum [inst : Fintype α] : (enum α).Nodup :=
  inst.nodup

/-- Membership in the chosen enumeration. -/
theorem mem_enum [Fintype α] (x : α) : x ∈ enum α :=
  complete_enum x

/-- The cardinality of the chosen enumeration. -/
def card (α : Type) [Fintype α] : Nat :=
  (enum α).length

end Fintype
