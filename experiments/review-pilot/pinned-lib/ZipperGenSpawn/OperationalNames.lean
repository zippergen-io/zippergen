import ZipperGenSpawn.SessionAgreement

namespace ZipperGenSpawn

universe u

/-- Each physical participant keeps a visit counter for a spawn site in a
    parent session. Counters are intentionally participant-local. -/
abbrev VisitCounters (Lifeline : Type u) :=
  Lifeline -> SessionName -> Occurrence -> Nat

def nextVisit (counters : VisitCounters Lifeline)
    (A : Lifeline) (parent : SessionName) (site : Occurrence) : Nat :=
  counters A parent site + 1

def nextChild (counters : VisitCounters Lifeline)
    (A : Lifeline) (parent : SessionName) (site : Occurrence) : SessionName :=
  .child parent site (nextVisit counters A parent site)

def incrementCounter [DecidableEq Lifeline]
    (counters : VisitCounters Lifeline)
    (A : Lifeline) (parent : SessionName) (site : Occurrence) :
    VisitCounters Lifeline :=
  fun B session occurrence =>
    if B = A ∧ session = parent ∧ occurrence = site then
      nextVisit counters A parent site
    else
      counters B session occurrence

@[simp] theorem incrementCounter_here [DecidableEq Lifeline]
    (counters : VisitCounters Lifeline)
    (A : Lifeline) (parent : SessionName) (site : Occurrence) :
    incrementCounter counters A parent site A parent site =
      counters A parent site + 1 := by
  simp [incrementCounter, nextVisit]

theorem nextChild_agrees_of_counter_agreement
    (counters : VisitCounters Lifeline)
    (A B : Lifeline) (parent : SessionName) (site : Occurrence)
    (hcount : counters A parent site = counters B parent site) :
    nextChild counters A parent site = nextChild counters B parent site := by
  simp [nextChild, nextVisit, hcount]

theorem nextChild_distinct_of_visit_distinct
    {parent : SessionName} {site : Occurrence} {k m : Nat}
    (hkm : k ≠ m) :
    SessionName.child parent site k ≠ SessionName.child parent site m := by
  intro h
  exact hkm ((SessionName.child_eq_iff.mp h).2.2)

/-- The runtime naming obligation isolated from scheduling: the `k`th local
    visit at every participating endpoint receives the same canonical name.
    The parallel/control correspondence theorem supplies equality of the
    ordinal `k`; this theorem performs the naming step. -/
theorem corresponding_forks_have_same_name
    (A B : Lifeline) (parent : SessionName)
    (site : Occurrence) (visit : Nat) :
    ForkEvent.child
        ({ owner := A,
           visit := { parent := parent, site := site, visit := visit } } :
          ForkEvent Lifeline) =
      ForkEvent.child
        ({ owner := B,
           visit := { parent := parent, site := site, visit := visit } } :
          ForkEvent Lifeline) := rfl

end ZipperGenSpawn
