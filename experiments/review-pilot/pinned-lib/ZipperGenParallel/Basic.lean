namespace ZipperGenParallel

universe u

/-- A local word over alphabet `α`. -/
abbrev Word (α : Type u) := List α

/-- A branch identifier for a parallel region. -/
abbrev BranchId := Nat

/-- A relation is acyclic if no vertex reaches itself by a nonempty path. -/
def Acyclic {α : Type u} (r : α → α → Prop) : Prop :=
  ∀ a, ¬ Relation.TransGen r a a

end ZipperGenParallel
