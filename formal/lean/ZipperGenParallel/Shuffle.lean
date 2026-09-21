import ZipperGenParallel.MSC

namespace ZipperGenParallel

universe u v w z

/--
Binary list interleaving.

`Interleaving xs ys zs` says that `zs` preserves the internal order of `xs`
and `ys` and contains exactly their elements.  This is the basic local
interleaving relation used by `parloc`; later files can lift it to branch
tuples.
-/
inductive Interleaving {α : Type u} : List α → List α → List α → Prop where
  | nil : Interleaving [] [] []
  | left {x : α} {xs ys zs : List α} :
      Interleaving xs ys zs → Interleaving (x :: xs) ys (x :: zs)
  | right {y : α} {xs ys zs : List α} :
      Interleaving xs ys zs → Interleaving xs (y :: ys) (y :: zs)

/-- A branch-tagged event or local letter. -/
structure Tagged (Branch : Type u) (α : Type v) where
  branch : Branch
  val : α
deriving Repr

namespace Tagged

/-- Map the payload of a tagged value, preserving its branch tag. -/
def mapVal {Branch : Type u} {α : Type v} {β : Type w}
    (f : α → β) (x : Tagged Branch α) : Tagged Branch β :=
  { branch := x.branch, val := f x.val }

@[simp]
theorem mapVal_branch {Branch : Type u} {α : Type v} {β : Type w}
    (f : α → β) (x : Tagged Branch α) :
    (mapVal f x).branch = x.branch :=
  rfl

@[simp]
theorem mapVal_val {Branch : Type u} {α : Type v} {β : Type w}
    (f : α → β) (x : Tagged Branch α) :
    (mapVal f x).val = f x.val :=
  rfl

end Tagged

/-- Keep only the letters that came from branch `i`, erasing branch tags. -/
def eraseBranch {Branch : Type u} {α : Type v} [DecidableEq Branch] (i : Branch) :
    List (Tagged Branch α) → List α
  | [] => []
  | x :: xs =>
      if x.branch = i then
        x.val :: eraseBranch i xs
      else
        eraseBranch i xs

theorem eraseBranch_mapVal {Branch : Type u} {α : Type v} {β : Type w}
    [DecidableEq Branch] (i : Branch) (f : α → β)
    (xs : List (Tagged Branch α)) :
    eraseBranch i (xs.map (Tagged.mapVal f)) =
      (eraseBranch i xs).map f := by
  induction xs with
  | nil =>
      simp [eraseBranch]
  | cons x xs ih =>
      by_cases hx : x.branch = i
      · simp [eraseBranch, hx, ih]
      · simp [eraseBranch, hx, ih]

/--
Tuple-level branch shuffle.

For every lifeline `A` and branch `i`, erasing from the shuffled local
word of `A` all events not tagged by `i` yields the local word contributed
by branch `i`.
-/
def BranchShuffle {Branch : Type u} {Lifeline : Type v} {α : Type w}
    [DecidableEq Branch]
    (branchWord : Branch → Lifeline → Word α)
    (shuffled : Lifeline → Word (Tagged Branch α)) : Prop :=
  ∀ i A, eraseBranch i (shuffled A) = branchWord i A

/--
For an exhaustive two-branch index type, branch erasure induces the binary
`Interleaving` relation on the untagged shuffled word.
-/
theorem eraseBranch_two_interleaving {Branch : Type u} {α : Type v}
    [DecidableEq Branch] {left right : Branch}
    (hne : left ≠ right)
    (hexhaustive : ∀ k : Branch, k = left ∨ k = right)
    (xs : List (Tagged Branch α)) :
    Interleaving (eraseBranch left xs) (eraseBranch right xs)
      (xs.map Tagged.val) := by
  induction xs with
  | nil =>
      simp [eraseBranch]
      exact Interleaving.nil
  | cons x xs ih =>
      rcases hexhaustive x.branch with hleft | hright
      · have hxright : ¬ x.branch = right := by
          intro hx
          exact hne (hleft.symm.trans hx)
        simpa [eraseBranch, hleft, hxright, hne] using
          Interleaving.left ih
      · have hxleft : ¬ x.branch = left := by
          intro hx
          exact hne (hx.symm.trans hright)
        have hne_symm : ¬ right = left := by
          intro h
          exact hne h.symm
        simpa [eraseBranch, hright, hxleft, hne_symm] using
          Interleaving.right ih

/--
Binary branch shuffles are binary interleavings after forgetting branch tags,
provided the branch index type has exactly the two displayed branches.
-/
theorem branchShuffle_two_to_interleaving {Branch : Type u}
    {Lifeline : Type v} {α : Type w} [DecidableEq Branch]
    {branchWord : Branch → Lifeline → Word α}
    {shuffled : Lifeline → Word (Tagged Branch α)}
    {left right : Branch}
    (hne : left ≠ right)
    (hexhaustive : ∀ k : Branch, k = left ∨ k = right)
    (h : BranchShuffle branchWord shuffled)
    (A : Lifeline) :
    Interleaving (branchWord left A) (branchWord right A)
      ((shuffled A).map Tagged.val) := by
  simpa [h left A, h right A] using
    eraseBranch_two_interleaving (left := left) (right := right)
      hne hexhaustive (shuffled A)

/--
Conversely, a binary `Interleaving` has a branch-tagged witness whose erasures
recover the two input words.
-/
theorem interleaving_exists_two_branch_word {Branch : Type u} {α : Type v}
    [DecidableEq Branch] {left right : Branch}
    (hne : left ≠ right) {xs ys zs : List α}
    (h : Interleaving xs ys zs) :
    ∃ tagged : List (Tagged Branch α),
      tagged.map Tagged.val = zs ∧
      eraseBranch left tagged = xs ∧
      eraseBranch right tagged = ys := by
  induction h with
  | nil =>
      exact ⟨[], rfl, rfl, rfl⟩
  | left h ih =>
      rename_i x _ _ _
      rcases ih with ⟨tagged, hmap, hleft, hright⟩
      refine ⟨{ branch := left, val := x } :: tagged, ?_, ?_, ?_⟩
      · simp [hmap]
      · simp [eraseBranch, hleft]
      · simp [eraseBranch, hne, hright]
  | right h ih =>
      rename_i y _ _ _
      rcases ih with ⟨tagged, hmap, hleft, hright⟩
      have hne_symm : ¬ right = left := by
        intro hrl
        exact hne hrl.symm
      refine ⟨{ branch := right, val := y } :: tagged, ?_, ?_, ?_⟩
      · simp [hmap]
      · simp [eraseBranch, hne_symm, hleft]
      · simp [eraseBranch, hright]

theorem mem_eraseBranch_of_mem_branch {Branch : Type u} {α : Type v}
    [DecidableEq Branch] {i : Branch} {xs : List (Tagged Branch α)}
    {x : Tagged Branch α} (hx : x ∈ xs) (hbranch : x.branch = i) :
    x.val ∈ eraseBranch i xs := by
  induction xs with
  | nil =>
      cases hx
  | cons y ys ih =>
      simp [eraseBranch] at hx ⊢
      rcases hx with hxy | hxys
      · subst hxy
        simp [hbranch]
      · by_cases hy : y.branch = i
        · simp [hy, ih hxys]
        · simp [hy, ih hxys]

theorem exists_mem_branch_of_mem_eraseBranch {Branch : Type u} {α : Type v}
    [DecidableEq Branch] {i : Branch} {xs : List (Tagged Branch α)}
    {y : α} (hy : y ∈ eraseBranch i xs) :
    ∃ x : Tagged Branch α, x ∈ xs ∧ x.branch = i ∧ x.val = y := by
  induction xs with
  | nil =>
      simp [eraseBranch] at hy
  | cons x xs ih =>
      by_cases hxbranch : x.branch = i
      · simp [eraseBranch, hxbranch] at hy
        rcases hy with hxy | hyxs
        · exact ⟨x, by simp, hxbranch, hxy.symm⟩
        · rcases ih hyxs with ⟨z, hzmem, hzbranch, hzval⟩
          exact ⟨z, by simp [hzmem], hzbranch, hzval⟩
      · simp [eraseBranch, hxbranch] at hy
        rcases ih hy with ⟨z, hzmem, hzbranch, hzval⟩
        exact ⟨z, by simp [hzmem], hzbranch, hzval⟩

theorem branchShuffle_mem_branchWord_of_mem {Branch : Type u}
    {Lifeline : Type v} {α : Type w} [DecidableEq Branch]
    {branchWord : Branch → Lifeline → Word α}
    {shuffled : Lifeline → Word (Tagged Branch α)}
    (h : BranchShuffle branchWord shuffled)
    {i : Branch} {A : Lifeline} {x : Tagged Branch α}
    (hx : x ∈ shuffled A) (hbranch : x.branch = i) :
    x.val ∈ branchWord i A := by
  rw [← h i A]
  exact mem_eraseBranch_of_mem_branch hx hbranch

theorem branchShuffle_exists_mem_of_mem_branchWord {Branch : Type u}
    {Lifeline : Type v} {α : Type w} [DecidableEq Branch]
    {branchWord : Branch → Lifeline → Word α}
    {shuffled : Lifeline → Word (Tagged Branch α)}
    (h : BranchShuffle branchWord shuffled)
    {i : Branch} {A : Lifeline} {y : α}
    (hy : y ∈ branchWord i A) :
    ∃ x : Tagged Branch α, x ∈ shuffled A ∧ x.branch = i ∧ x.val = y := by
  rw [← h i A] at hy
  exact exists_mem_branch_of_mem_eraseBranch hy

theorem tagged_branch_unique {Branch : Type u} {α : Type v}
    {x : Tagged Branch α} {i j : Branch}
    (hi : x.branch = i) (hj : x.branch = j) : i = j := by
  rw [← hi, ← hj]

/--
Local decomposition of a branch shuffle as membership facts.

Every tagged event in a shuffled local word contributes to the erasure of
its tag, and every event of a branch word is represented by a tagged event in
the shuffled word.
-/
theorem local_shuffle_decompose {Branch : Type u} {Lifeline : Type v}
    {α : Type w} [DecidableEq Branch]
    {branchWord : Branch → Lifeline → Word α}
    {shuffled : Lifeline → Word (Tagged Branch α)}
    (h : BranchShuffle branchWord shuffled) :
    (∀ {i A} {x : Tagged Branch α},
        x ∈ shuffled A → x.branch = i → x.val ∈ branchWord i A) ∧
    (∀ {i A} {y : α},
        y ∈ branchWord i A →
          ∃ x : Tagged Branch α, x ∈ shuffled A ∧ x.branch = i ∧ x.val = y) :=
  ⟨fun hx hbranch => branchShuffle_mem_branchWord_of_mem h hx hbranch,
   fun hy => branchShuffle_exists_mem_of_mem_branchWord h hy⟩

/-- Branch shuffle specialized to a fixed finite branch index set. -/
abbrev FinBranchShuffle {n : Nat} {Lifeline : Type u} {α : Type v}
    (branchWord : Fin n → Lifeline → Word α)
    (shuffled : Lifeline → Word (Tagged (Fin n) α)) : Prop :=
  BranchShuffle branchWord shuffled

/--
Every event in a finite-indexed branch shuffle carries an in-range branch tag.
This is the coverage property needed before constructing shuffled MSCs.
-/
theorem finBranchShuffle_event_covered {n : Nat} {Lifeline : Type u}
    {α : Type v}
    {branchWord : Fin n → Lifeline → Word α}
    {shuffled : Lifeline → Word (Tagged (Fin n) α)}
    (_h : FinBranchShuffle branchWord shuffled)
    {A : Lifeline} {x : Tagged (Fin n) α}
    (_hx : x ∈ shuffled A) :
    ∃ i : Fin n, x.branch = i :=
  ⟨x.branch, rfl⟩

/-- MSCs whose events carry proof-level branch tags. -/
abbrev TaggedMSC (Branch : Type u) (Lifeline : Type v)
    (Channel : Type w) (Event : Type z) :=
  MSC Lifeline Channel (Tagged Branch Event)

/--
Every matched message in a tagged MSC stays within one proof-level branch.

In the paper, occurrence/path-based channel names imply this for parallel
branch subchannels.  The Lean development records the branch origin as a
proof-level event tag and uses this invariant as the corresponding restriction
principle.
-/
def TaggedMessagesSameBranch {Branch : Type u} {Lifeline : Type v}
    {Channel : Type w} {Event : Type z}
    (M : TaggedMSC Branch Lifeline Channel Event) : Prop :=
  ∀ {s r}, M.msg s r → s.branch = r.branch

theorem taggedMessages_recv_closed {Branch : Type u} {Lifeline : Type v}
    {Channel : Type w} {Event : Type z}
    (M : TaggedMSC Branch Lifeline Channel Event)
    (htag : TaggedMessagesSameBranch M) {i : Branch} {r : Tagged Branch Event}
    (hr : r.branch = i)
    (hrecv : ∃ A ch, M.kind r = EventKind.recv A ch) :
    ∃ s, s.branch = i ∧ M.msg s r := by
  rcases M.recv_matched hrecv with ⟨s, hmsg⟩
  refine ⟨s, ?_, hmsg⟩
  calc
    s.branch = r.branch := htag hmsg
    _ = i := hr

theorem taggedMessages_send_closed {Branch : Type u} {Lifeline : Type v}
    {Channel : Type w} {Event : Type z}
    (M : TaggedMSC Branch Lifeline Channel Event)
    (htag : TaggedMessagesSameBranch M) {i : Branch}
    {s r : Tagged Branch Event}
    (hs : s.branch = i) (hmsg : M.msg s r) :
    r.branch = i := by
  calc
    r.branch = s.branch := (htag hmsg).symm
    _ = i := hs

/--
Restrict a tagged MSC to the events of one proof-level branch.

The branch tags are proof witnesses for the channel/path branch origin.
Channels are left unchanged.
-/
def taggedBranchRestriction {Branch : Type u} {Lifeline : Type v}
    {Channel : Type w} {Event : Type z}
    (M : TaggedMSC Branch Lifeline Channel Event)
    (htag : TaggedMessagesSameBranch M) (i : Branch) :
    MSC Lifeline Channel {e : Tagged Branch Event // e.branch = i} :=
  M.restrict (fun e => e.branch = i) (taggedMessages_recv_closed M htag)

theorem taggedBranchRestriction_complete {Branch : Type u}
    {Lifeline : Type v} {Channel : Type w} {Event : Type z}
    (M : TaggedMSC Branch Lifeline Channel Event)
    (htag : TaggedMessagesSameBranch M) (i : Branch)
    (hcomplete : M.IsComplete) :
    (taggedBranchRestriction M htag i).IsComplete :=
  M.restrict_complete (fun e => e.branch = i)
    (taggedMessages_recv_closed M htag)
    (taggedMessages_send_closed M htag)
    hcomplete

end ZipperGenParallel
