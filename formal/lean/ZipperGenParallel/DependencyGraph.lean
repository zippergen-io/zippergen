import ZipperGenParallel.MSC

namespace ZipperGenParallel

universe u v w x

/-- A lifeline is private if it is used by exactly one branch. -/
def PrivateLifeline {Branch Lifeline : Type u}
    (uses : Branch → Lifeline → Prop) (A : Lifeline) : Prop :=
  ∃ i, uses i A ∧ ∀ j, uses j A → j = i

/-- A lifeline is shared if it is used by two distinct branches. -/
def SharedLifeline {Branch Lifeline : Type u}
    (uses : Branch → Lifeline → Prop) (A : Lifeline) : Prop :=
  ∃ i j, i ≠ j ∧ uses i A ∧ uses j A

/--
Shared reachability graph.

This is the graph used by the paper's well-formedness condition: first take
nonempty reachability in the dependency graph, then restrict the endpoints
to shared lifelines.
-/
def SRG {Lifeline : Type u} (shared : Lifeline → Prop)
    (DG : Lifeline → Lifeline → Prop) : Lifeline → Lifeline → Prop :=
  fun A B => shared A ∧ shared B ∧ A ≠ B ∧ Relation.TransGen DG A B

/-- A parallel region is well formed when its shared reachability graph is acyclic. -/
def WellFormedParallel {Lifeline : Type u} (shared : Lifeline → Prop)
    (DG : Lifeline → Lifeline → Prop) : Prop :=
  Acyclic (SRG shared DG)

theorem srg_uses_reachability {Lifeline : Type u} {shared : Lifeline → Prop}
    {DG : Lifeline → Lifeline → Prop} {A B : Lifeline}
    (h : SRG shared DG A B) : Relation.TransGen DG A B :=
  h.2.2.2

theorem wellFormed_no_srg_cycle {Lifeline : Type u}
    {shared : Lifeline → Prop} {DG : Lifeline → Lifeline → Prop}
    (h : WellFormedParallel shared DG) (A : Lifeline) :
    ¬ Relation.TransGen (SRG shared DG) A A :=
  h A

/--
Dependency soundness interface.

This is the abstract form of the paper's dependency-soundness lemma.  A
syntactic dependency graph can be plugged in later.  Stage 4 can already use
this interface to turn a cross-lifeline causal path in a branch MSC into a path
in the dependency graph.
-/
structure DependencySoundness {Lifeline : Type u} {Channel : Type v}
    {Event : Type w}
    (M : MSC Lifeline Channel Event)
    (DG : Lifeline → Lifeline → Prop) : Prop where
  sound :
    ∀ {e f : Event},
      MSC.strictCausal M e f →
      M.lifeline e ≠ M.lifeline f →
      Relation.TransGen DG (M.lifeline e) (M.lifeline f)

theorem dependencySoundness_to_srg {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    {M : MSC Lifeline Channel Event}
    {DG : Lifeline → Lifeline → Prop}
    {shared : Lifeline → Prop} {e f : Event}
    (hsound : DependencySoundness M DG)
    (hpath : MSC.strictCausal M e f)
    (hshared_e : shared (M.lifeline e))
    (hshared_f : shared (M.lifeline f))
    (hne : M.lifeline e ≠ M.lifeline f) :
    SRG shared DG (M.lifeline e) (M.lifeline f) :=
  ⟨hshared_e, hshared_f, hne, hsound.sound hpath hne⟩

theorem transGen_mono {α : Type u} {r s : α → α → Prop}
    (hsub : ∀ {a b}, r a b → s a b) {a b : α}
    (h : Relation.TransGen r a b) : Relation.TransGen s a b := by
  induction h with
  | single hrel =>
      exact Relation.TransGen.single (hsub hrel)
  | tail hpath hrel ih =>
      exact Relation.TransGen.tail ih (hsub hrel)

/-- The dependency graph of a parallel region as the union of branch graphs. -/
def UnionDG {Branch : Type u} {Lifeline : Type v}
    (branchDG : Branch → Lifeline → Lifeline → Prop) :
    Lifeline → Lifeline → Prop :=
  fun A B => ∃ i, branchDG i A B

theorem transGen_to_unionDG {Branch : Type u} {Lifeline : Type v}
    {branchDG : Branch → Lifeline → Lifeline → Prop}
    (i : Branch) {A B : Lifeline}
    (h : Relation.TransGen (branchDG i) A B) :
    Relation.TransGen (UnionDG branchDG) A B :=
  transGen_mono (fun hrel => ⟨i, hrel⟩) h

/-- Branchwise dependency soundness for a family of branch MSCs. -/
def BranchDependencySoundness {Branch : Type u} {Lifeline : Type v}
    {Channel : Type w} {Event : Type x}
    (M : Branch → MSC Lifeline Channel Event)
    (branchDG : Branch → Lifeline → Lifeline → Prop) : Prop :=
  ∀ i, DependencySoundness (M i) (branchDG i)

theorem branchDependencySoundness_to_union {Branch : Type u}
    {Lifeline : Type v} {Channel : Type w} {Event : Type x}
    {M : Branch → MSC Lifeline Channel Event}
    {branchDG : Branch → Lifeline → Lifeline → Prop}
    (hsound : BranchDependencySoundness M branchDG)
    (i : Branch) {e f : Event}
    (hpath : MSC.strictCausal (M i) e f)
    (hne : (M i).lifeline e ≠ (M i).lifeline f) :
    Relation.TransGen (UnionDG branchDG)
      ((M i).lifeline e) ((M i).lifeline f) :=
  transGen_to_unionDG i ((hsound i).sound hpath hne)

theorem branchDependencySoundness_to_srg {Branch : Type u}
    {Lifeline : Type v} {Channel : Type w} {Event : Type x}
    {M : Branch → MSC Lifeline Channel Event}
    {branchDG : Branch → Lifeline → Lifeline → Prop}
    {shared : Lifeline → Prop}
    (hsound : BranchDependencySoundness M branchDG)
    (i : Branch) {e f : Event}
    (hpath : MSC.strictCausal (M i) e f)
    (hshared_e : shared ((M i).lifeline e))
    (hshared_f : shared ((M i).lifeline f))
    (hne : (M i).lifeline e ≠ (M i).lifeline f) :
    SRG shared (UnionDG branchDG)
      ((M i).lifeline e) ((M i).lifeline f) :=
  ⟨hshared_e, hshared_f, hne,
    branchDependencySoundness_to_union hsound i hpath hne⟩

end ZipperGenParallel
