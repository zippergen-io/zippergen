import ZipperGenParallel.Alphabet

namespace ZipperGenSpawn

universe u v

/-- Globally unique labels attached to parallel, decision, loop, and spawn
    occurrences in one source program. -/
abbrev Occurrence := Nat

/-- The root FIFO scope or one statically labelled branch of a parallel
    occurrence. Unlike the older parallel proof model, this is not a path. -/
inductive ChannelScope where
  | main
  | branch (site : Occurrence) (index : Nat)
deriving Repr, DecidableEq

/-- Canonical dynamic session names. `visit` is one-based and local to the
    pair `(parent, site)`. -/
inductive SessionName where
  | root
  | child (parent : SessionName) (site : Occurrence) (visit : Nat)
deriving Repr, DecidableEq

namespace SessionName

def parent : SessionName -> Option SessionName
  | root => none
  | child parent _ _ => some parent

@[simp] theorem parent_root : parent root = none := rfl

@[simp] theorem parent_child (s : SessionName) (o : Occurrence) (k : Nat) :
    parent (child s o k) = some s := rfl

theorem child_injective :
    Function.Injective
      (fun p : SessionName × Occurrence × Nat =>
        child p.1 p.2.1 p.2.2) := by
  intro left right h
  cases left with
  | mk leftParent leftRest =>
      cases leftRest with
      | mk leftSite leftVisit =>
          cases right with
          | mk rightParent rightRest =>
              cases rightRest with
              | mk rightSite rightVisit =>
                  cases h
                  rfl

theorem child_eq_iff
    {s t : SessionName} {o p : Occurrence} {k m : Nat} :
    child s o k = child t p m <-> s = t /\ o = p /\ k = m := by
  constructor
  · intro h
    cases h
    exact ⟨rfl, rfl, rfl⟩
  · rintro ⟨rfl, rfl, rfl⟩
    rfl

theorem child_ne_parent (parent : SessionName) (site : Occurrence)
    (visit : Nat) : child parent site visit ≠ parent := by
  induction parent generalizing site visit with
  | root =>
      intro h
      cases h
  | child ancestor ancestorSite ancestorVisit ih =>
      intro h
      have hparent := congrArg SessionName.parent h
      simp only [parent_child, Option.some.injEq] at hparent
      exact ih ancestorSite ancestorVisit hparent

end SessionName

/-- Symbolic choreography syntax used by the spawn proof. Payload values and
    action implementations are deliberately abstracted away, as in the
    checked parallel development. `Field` occurs only in spawn freshness
    declarations; it does not introduce shared memory. -/
inductive Prog (Lifeline : Type u) (Field : Type v) : Type (max u v) where
  | eps
  | msg (sender receiver : Lifeline) (h : sender ≠ receiver)
  | act (owner : Lifeline)
  | seq (first second : Prog Lifeline Field)
  | branch
      (site : Occurrence) (owner : Lifeline)
      (thenBranch elseBranch : Prog Lifeline Field)
  | loop
      (site : Occurrence) (owner : Lifeline)
      (body exit : Prog Lifeline Field)
  | par {n : Nat}
      (site : Occurrence) (branches : Fin n -> Prog Lifeline Field)
  | spawn
      (site : Occurrence) (fresh : List (Lifeline × Field))
      (body : Prog Lifeline Field)

namespace Prog

infixl:50 " ;;s " => Prog.seq

def concatLists {alpha : Type u} (lists : List (List alpha)) : List alpha :=
  lists.foldr (· ++ ·) []

def finConcatMap {n : Nat} {alpha : Type u}
    (f : Fin n -> List alpha) : List alpha :=
  concatLists (List.ofFn f)

/-- Lifelines that may act in a program. In particular, spawn has precisely
    the role set of its body. -/
def lifelines : Prog Lifeline Field -> List Lifeline
  | eps => []
  | msg sender receiver _ => [sender, receiver]
  | act owner => [owner]
  | seq first second => lifelines first ++ lifelines second
  | branch _ owner thenBranch elseBranch =>
      owner :: (lifelines thenBranch ++ lifelines elseBranch)
  | loop _ owner body exit =>
      owner :: (lifelines body ++ lifelines exit)
  | par _ branches => finConcatMap (fun i => lifelines (branches i))
  | spawn _ _ body => lifelines body

def mentions [DecidableEq Lifeline]
    (P : Prog Lifeline Field) (A : Lifeline) : Bool :=
  decide (A ∈ lifelines P)

/-- All labelled control occurrences, including occurrences nested inside a
    spawn body. -/
def occurrences : Prog Lifeline Field -> List Occurrence
  | eps | msg .. | act _ => []
  | seq first second => occurrences first ++ occurrences second
  | branch site _ thenBranch elseBranch =>
      site :: (occurrences thenBranch ++ occurrences elseBranch)
  | loop site _ body exit => site :: (occurrences body ++ occurrences exit)
  | par site branches => site :: finConcatMap (fun i => occurrences (branches i))
  | spawn site _ body => site :: occurrences body

/-- Static labels are globally unique in the complete source tree. Repeated
    dynamic visits reuse a label and are distinguished by their session and
    visit counter. -/
def WellLabelled (P : Prog Lifeline Field) : Prop :=
  P.occurrences.Nodup

/-- The fields owned by `A` that a local fork must copy. -/
def freshFor [DecidableEq Lifeline]
    (A : Lifeline) (fresh : List (Lifeline × Field)) : List Field :=
  fresh.filterMap fun entry =>
    if entry.1 = A then some entry.2 else none

end Prog

end ZipperGenSpawn
