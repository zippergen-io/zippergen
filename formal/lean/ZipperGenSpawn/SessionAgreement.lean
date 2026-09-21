import ZipperGenSpawn.Projection

namespace ZipperGenSpawn

universe u v

/-- Runtime identity of one dynamic visit to a spawn occurrence. -/
structure SpawnVisit where
  parent : SessionName
  site : Occurrence
  visit : Nat
deriving Repr, DecidableEq

namespace SpawnVisit

def child (v : SpawnVisit) : SessionName :=
  .child v.parent v.site v.visit

theorem child_injective : Function.Injective child := by
  intro left right h
  cases left with
  | mk leftParent leftSite leftVisit =>
      cases right with
      | mk rightParent rightSite rightVisit =>
          simp only [child] at h
          cases h
          rfl

end SpawnVisit

/-- The local event produced when participant `owner` executes its fork
    template for dynamic visit `v`. -/
structure ForkEvent (Lifeline : Type u) where
  owner : Lifeline
  visit : SpawnVisit
deriving Repr, DecidableEq

namespace ForkEvent

def parent (event : ForkEvent Lifeline) : SessionName := event.visit.parent
def site (event : ForkEvent Lifeline) : Occurrence := event.visit.site
def child (event : ForkEvent Lifeline) : SessionName := event.visit.child

/-- Endpoints that agree on the parent session, static occurrence, and ordinal
    visit necessarily allocate the same child name. This is the deterministic
    naming part of the projection argument; establishing ordinal-visit
    agreement is the separate control-correspondence obligation. -/
theorem child_agreement
    (A B : Lifeline) (v : SpawnVisit) :
    child ({ owner := A, visit := v } : ForkEvent Lifeline) =
      child ({ owner := B, visit := v } : ForkEvent Lifeline) := rfl

theorem different_visits_have_different_children
    {A B : Lifeline} {left right : SpawnVisit}
    (h : left ≠ right) :
    child ({ owner := A, visit := left } : ForkEvent Lifeline) ≠
      child ({ owner := B, visit := right } : ForkEvent Lifeline) := by
  intro hchild
  exact h (SpawnVisit.child_injective hchild)

end ForkEvent

/-- Every endpoint participating in the body receives the same static spawn
    site in its projected fork template. -/
theorem projected_fork_site_agreement
    {Lifeline : Type u} {Field : Type v} [DecidableEq Lifeline]
    (order : List Lifeline) (A B : Lifeline) (scope : ChannelScope)
    (site : Occurrence) (fresh : List (Lifeline × Field))
    (body : Prog Lifeline Field)
    (hA : A ∈ body.lifelines) (hB : B ∈ body.lifelines) :
    Prog.project order A scope (.spawn site fresh body) =
        .fork site (Prog.freshFor A fresh)
          (Prog.project order A .main body) /\
      Prog.project order B scope (.spawn site fresh body) =
        .fork site (Prog.freshFor B fresh)
          (Prog.project order B .main body) := by
  exact ⟨Prog.project_spawn_of_mem order A scope site fresh body hA,
    Prog.project_spawn_of_mem order B scope site fresh body hB⟩

end ZipperGenSpawn
