import ZipperGenParallel.Shuffle

namespace ZipperGenParallel

universe u v w

/-- List-prefix relation, written explicitly to keep Stage 5 self-contained. -/
def ListPrefix {α : Type u} (xs ys : List α) : Prop :=
  ∃ zs, xs ++ zs = ys

namespace ListPrefix

theorem refl {α : Type u} (xs : List α) : ListPrefix xs xs := by
  exact ⟨[], by simp⟩

theorem trans {α : Type u} {xs ys zs : List α}
    (hxy : ListPrefix xs ys) (hyz : ListPrefix ys zs) :
    ListPrefix xs zs := by
  rcases hxy with ⟨xy, hxy⟩
  rcases hyz with ⟨yz, hyz⟩
  refine ⟨xy ++ yz, ?_⟩
  rw [← hyz, ← hxy, List.append_assoc]

theorem cons_cancel {α : Type u} {a : α} {xs ys : List α}
    (h : ListPrefix (a :: xs) (a :: ys)) :
    ListPrefix xs ys := by
  rcases h with ⟨zs, hzs⟩
  refine ⟨zs, ?_⟩
  simpa using congrArg List.tail hzs

theorem append_eq_left_prefix {α : Type u}
    {u1 u2 v1 v2 : List α}
    (hEq : u1 ++ u2 = v1 ++ v2)
    (hLen : u1.length ≤ v1.length) :
    ListPrefix u1 v1 := by
  refine ⟨v1.drop u1.length, ?_⟩
  have hTake := congrArg (fun xs => xs.take u1.length) hEq
  have hu1 : u1 = v1.take u1.length := by
    simpa [List.take_left, List.take_append_of_le_length hLen] using hTake
  rw [hu1, List.length_take_of_le hLen]
  exact List.take_append_drop u1.length v1

theorem append_eq_right_prefix {α : Type u}
    {u1 u2 v1 v2 : List α}
    (hEq : u1 ++ u2 = v1 ++ v2)
    (hLen : v1.length ≤ u1.length) :
    ListPrefix v1 u1 := by
  refine ⟨u1.drop v1.length, ?_⟩
  have hTake := congrArg (fun xs => xs.take v1.length) hEq
  have hv1 : v1 = u1.take v1.length := by
    simpa [List.take_left, List.take_append_of_le_length hLen] using hTake.symm
  rw [hv1, List.length_take_of_le hLen]
  exact List.take_append_drop v1.length u1

theorem append_split {α : Type u}
    {w u v : List α}
    (h : ListPrefix w (u ++ v)) :
    ∃ u' v',
      w = u' ++ v' ∧
      ListPrefix u' u ∧
      ListPrefix v' v ∧
      (v' ≠ [] → u' = u) := by
  rcases h with ⟨t, ht⟩
  by_cases hwu : w.length ≤ u.length
  · refine ⟨w, [], by simp, ?_, ?_, ?_⟩
    · have hTake := congrArg (fun xs => xs.take w.length) ht
      have hwEq : w = u.take w.length := by
        simpa [List.take_append_of_le_length hwu, List.take_left] using hTake
      refine ⟨u.drop w.length, ?_⟩
      rw [hwEq, List.length_take_of_le hwu]
      exact List.take_append_drop w.length u
    · exact ⟨v, by simp⟩
    · intro hne
      contradiction
  · have hwuLt : u.length < w.length := Nat.lt_of_not_ge hwu
    refine ⟨u, w.drop u.length, ?_, ⟨[], by simp⟩, ?_, ?_⟩
    · have hTake := congrArg (fun xs => xs.take u.length) ht
      have huEq : u = w.take u.length := by
        simpa [List.take_left,
          List.take_append_of_le_length (Nat.le_of_lt hwuLt)] using hTake.symm
      rw [huEq, List.length_take_of_le (Nat.le_of_lt hwuLt)]
      exact (List.take_append_drop u.length w).symm
    · refine ⟨t, ?_⟩
      have hwSplit : w = u ++ w.drop u.length := by
        have hTake := congrArg (fun xs => xs.take u.length) ht
        have huEq : u = w.take u.length := by
          simpa [List.take_left,
            List.take_append_of_le_length (Nat.le_of_lt hwuLt)] using hTake.symm
        rw [huEq, List.length_take_of_le (Nat.le_of_lt hwuLt)]
        exact (List.take_append_drop u.length w).symm
      rw [hwSplit] at ht
      have hcancel : w.drop u.length ++ t = v := by
        have hs : u ++ (w.drop u.length ++ t) = u ++ v := by
          simpa [List.append_assoc] using ht
        exact List.append_cancel_left hs
      exact hcancel
    · intro _
      rfl

end ListPrefix

/-- Prefix-freeness for a lifeline-indexed local complete-trace predicate. -/
def LocalPrefixFree {Lifeline : Type u} {α : Type v}
    (CompleteLocal : Lifeline → Word α → Prop) : Prop :=
  ∀ A {u v},
    CompleteLocal A u → CompleteLocal A v → ListPrefix u v → u = v

theorem eraseBranch_append {Branch : Type u} {α : Type v}
    [DecidableEq Branch] (i : Branch)
    (xs ys : List (Tagged Branch α)) :
    eraseBranch i (xs ++ ys) = eraseBranch i xs ++ eraseBranch i ys := by
  induction xs with
  | nil =>
      simp [eraseBranch]
  | cons x xs ih =>
      by_cases hx : x.branch = i
      · simp [eraseBranch, hx, ih]
      · simp [eraseBranch, hx, ih]

theorem eraseBranch_preserves_prefix {Branch : Type u} {α : Type v}
    [DecidableEq Branch] {i : Branch}
    {xs ys : List (Tagged Branch α)}
    (h : ListPrefix xs ys) :
    ListPrefix (eraseBranch i xs) (eraseBranch i ys) := by
  rcases h with ⟨zs, hzs⟩
  refine ⟨eraseBranch i zs, ?_⟩
  rw [← hzs, eraseBranch_append]

theorem list_map_preserves_prefix {α : Type u} {β : Type v}
    {xs ys : List α} (f : α → β)
    (h : ListPrefix xs ys) :
    ListPrefix (xs.map f) (ys.map f) := by
  rcases h with ⟨zs, hzs⟩
  refine ⟨zs.map f, ?_⟩
  rw [← hzs, List.map_append]

theorem listPrefix_lift_of_prefix_map {α : Type u} {β : Type v}
    {xs : List α} {ys : List β} (f : α → β)
    (h : ListPrefix ys (xs.map f)) :
    ∃ zs : List α, ListPrefix zs xs ∧ zs.map f = ys := by
  induction xs generalizing ys with
  | nil =>
      rcases h with ⟨suffix, hsuffix⟩
      cases ys with
      | nil =>
          exact ⟨[], ListPrefix.refl [], rfl⟩
      | cons y ys =>
          simp at hsuffix
  | cons x xs ih =>
      cases ys with
      | nil =>
          exact ⟨[], ⟨x :: xs, rfl⟩, rfl⟩
      | cons y ys =>
          rcases h with ⟨suffix, hsuffix⟩
          simp only [List.map_cons, List.cons_append] at hsuffix
          injection hsuffix with hhead htail
          rcases ih (ys := ys) ⟨suffix, htail⟩ with
            ⟨zs, hpzs, hmap⟩
          rcases hpzs with ⟨rest, hrest⟩
          refine ⟨x :: zs, ?_, ?_⟩
          · exact ⟨rest, by simp [hrest]⟩
          · simp [hhead.symm, hmap]

theorem taggedWord_lift_of_eraseBranch_map {Branch : Type u}
    {α : Type v} {β : Type w} [DecidableEq Branch]
    (f : α → β) (tagged : List (Tagged Branch β))
    (branches : Branch → List α)
    (herase : ∀ i, (branches i).map f = eraseBranch i tagged) :
    ∃ lifted : List (Tagged Branch α),
      lifted.map (Tagged.mapVal f) = tagged ∧
      ∀ i, eraseBranch i lifted = branches i := by
  induction tagged generalizing branches with
  | nil =>
      have hbranches : ∀ i, branches i = [] := by
        intro i
        cases h : branches i with
        | nil => rfl
        | cons a rest =>
            have hmap := herase i
            simp [eraseBranch, h] at hmap
      refine ⟨[], by simp, ?_⟩
      intro i
      simp [eraseBranch, hbranches i]
  | cons x xs ih =>
      let b : Branch := x.branch
      have hb : (branches b).map f = x.val :: eraseBranch b xs := by
        simpa [eraseBranch, b] using herase b
      cases hbranch : branches b with
      | nil =>
          simp [hbranch] at hb
      | cons a rest =>
          have hb' : f a :: rest.map f =
              x.val :: eraseBranch b xs := by
            simpa [hbranch] using hb
          injection hb' with hhead htail
          let tailBranches : Branch → List α :=
            fun i => if i = b then rest else branches i
          have htailErase :
              ∀ i, (tailBranches i).map f = eraseBranch i xs := by
            intro i
            by_cases hi : i = b
            · subst i
              simp [tailBranches, htail]
            · have hxi : x.branch ≠ i := by
                intro h
                exact hi (by simpa [b] using h.symm)
              have hiErase := herase i
              have hcons :
                  eraseBranch i (x :: xs) = eraseBranch i xs := by
                simp [eraseBranch, hxi]
              rw [hcons] at hiErase
              simpa [tailBranches, hi] using hiErase
          rcases ih tailBranches htailErase with
            ⟨liftedTail, hmapTail, heraseTail⟩
          let liftedHead : Tagged Branch α := { branch := b, val := a }
          refine ⟨liftedHead :: liftedTail, ?_, ?_⟩
          · simp [liftedHead, Tagged.mapVal, b, hhead, hmapTail]
          · intro i
            by_cases hi : i = b
            · subst i
              simp [liftedHead, eraseBranch, b, hbranch, heraseTail,
                tailBranches]
            · have hxi : b ≠ i := fun h => hi h.symm
              simp [liftedHead, eraseBranch, hxi, heraseTail, tailBranches,
                hi]

theorem list_append_eq_self_right_nil {α : Type u} {xs ys : List α}
    (h : xs = xs ++ ys) : ys = [] := by
  induction xs with
  | nil =>
      simpa using h.symm
  | cons _ xs ih =>
      simp at h
      exact h

theorem eraseBranch_all_empty_nil {n : Nat} {α : Type u}
    {xs : List (Tagged (Fin n) α)}
    (h : ∀ i, eraseBranch i xs = []) : xs = [] := by
  cases xs with
  | nil =>
      rfl
  | cons x xs =>
      have hx := h x.branch
      simp [eraseBranch] at hx

theorem listPrefix_eq_of_eraseBranch_eq {n : Nat} {α : Type u}
    {xs ys : List (Tagged (Fin n) α)}
    (hp : ListPrefix xs ys)
    (herase : ∀ i, eraseBranch i xs = eraseBranch i ys) :
    xs = ys := by
  rcases hp with ⟨zs, hzs⟩
  have hzs_empty : zs = [] := by
    apply eraseBranch_all_empty_nil
    intro i
    have hi := herase i
    rw [← hzs, eraseBranch_append] at hi
    exact list_append_eq_self_right_nil hi
  rw [← hzs, hzs_empty, List.append_nil]

/--
Generic branch-local trace tuple.

The specialized `ParLocTrace` below fixes `α` to `LocalLetter`; this generic
version is used by the recursive decorated local-trace proof.
-/
abbrev ParLocTraceOf (Lifeline : Type u) (α : Type v) :=
  Lifeline → Word α

/-- One generic branch-local trace assignment per parallel branch. -/
abbrev ParLocBranchesOf (n : Nat) (Lifeline : Type u) (α : Type v) :=
  Fin n → ParLocTraceOf Lifeline α

/-- Generic tagged local trace of a parallel region. -/
abbrev ParLocTaggedTraceOf (n : Nat) (Lifeline : Type u) (α : Type v) :=
  Lifeline → Word (Tagged (Fin n) α)

/-- Pointwise prefix relation on generic tagged local traces. -/
def TracePrefixOf {n : Nat} {Lifeline : Type u} {α : Type v}
    (pref full : ParLocTaggedTraceOf n Lifeline α) : Prop :=
  ∀ A, ListPrefix (pref A) (full A)

/-- Generic local parallel semantics, by branch erasure. -/
def ParLocOf {n : Nat} {Lifeline : Type u} {α : Type v}
    (branches : ParLocBranchesOf n Lifeline α)
    (trace : ParLocTaggedTraceOf n Lifeline α) : Prop :=
  FinBranchShuffle branches trace

namespace ParLocBranchesOf

def append {n : Nat} {Lifeline : Type u} {α : Type v}
    (left right : ParLocBranchesOf n Lifeline α) :
    ParLocBranchesOf n Lifeline α :=
  fun i A => left i A ++ right i A

end ParLocBranchesOf

namespace ParLocTaggedTraceOf

def append {n : Nat} {Lifeline : Type u} {α : Type v}
    (left right : ParLocTaggedTraceOf n Lifeline α) :
    ParLocTaggedTraceOf n Lifeline α :=
  fun A => left A ++ right A

end ParLocTaggedTraceOf

/-- Forget generic branch-origin tags. -/
def forgetBranchTagOf {n : Nat} {Lifeline : Type u} {α : Type v}
    (trace : ParLocTaggedTraceOf n Lifeline α) :
    ParLocTraceOf Lifeline α :=
  fun A => (trace A).map Tagged.val

/-- Untagged view of generic `ParLocOf`. -/
def UntaggedParLocOf {n : Nat} {Lifeline : Type u} {α : Type v}
    (branches : ParLocBranchesOf n Lifeline α)
    (trace : ParLocTraceOf Lifeline α) : Prop :=
  ∃ tagged : ParLocTaggedTraceOf n Lifeline α,
    ParLocOf branches tagged ∧ forgetBranchTagOf tagged = trace

theorem parlocOf_forgetBranch_eq {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {branches : ParLocBranchesOf n Lifeline α}
    {trace : ParLocTaggedTraceOf n Lifeline α}
    (hpar : ParLocOf branches trace) (i : Fin n) (A : Lifeline) :
    eraseBranch i (trace A) = branches i A :=
  hpar i A

theorem parlocOf_forgetBranchTag_witness {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {branches : ParLocBranchesOf n Lifeline α}
    {trace : ParLocTaggedTraceOf n Lifeline α}
    (hpar : ParLocOf branches trace) :
    UntaggedParLocOf branches (forgetBranchTagOf trace) := by
  exact ⟨trace, hpar, rfl⟩

theorem forgetBranchTagOf_preserves_prefix {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {pref full : ParLocTaggedTraceOf n Lifeline α}
    (hpref : TracePrefixOf pref full) :
    ∀ A, ListPrefix (forgetBranchTagOf pref A) (forgetBranchTagOf full A) := by
  intro A
  exact list_map_preserves_prefix Tagged.val (hpref A)

theorem tracePrefixOf_map {n : Nat}
    {Lifeline : Type u} {α : Type v} {β : Type w}
    (f : α → β)
    {pref full : ParLocTaggedTraceOf n Lifeline α}
    (hpref : TracePrefixOf pref full) :
    TracePrefixOf
      (fun A => (pref A).map (Tagged.mapVal f))
      (fun A => (full A).map (Tagged.mapVal f)) := by
  intro A
  exact list_map_preserves_prefix (Tagged.mapVal f) (hpref A)

theorem tracePrefixOf_append_left {n : Nat}
    {Lifeline : Type u} {α : Type v}
    (left right : ParLocTaggedTraceOf n Lifeline α) :
    TracePrefixOf left (ParLocTaggedTraceOf.append left right) := by
  intro A
  exact ⟨right A, rfl⟩

theorem parlocOf_append {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {leftBranches rightBranches : ParLocBranchesOf n Lifeline α}
    {leftTrace rightTrace : ParLocTaggedTraceOf n Lifeline α}
    (hleft : ParLocOf leftBranches leftTrace)
    (hright : ParLocOf rightBranches rightTrace) :
    ParLocOf (ParLocBranchesOf.append leftBranches rightBranches)
      (ParLocTaggedTraceOf.append leftTrace rightTrace) := by
  intro i A
  simp [ParLocBranchesOf.append, ParLocTaggedTraceOf.append,
    eraseBranch_append, hleft i A, hright i A]

theorem parlocOf_map {n : Nat}
    {Lifeline : Type u} {α : Type v} {β : Type w}
    (f : α → β)
    {branches : ParLocBranchesOf n Lifeline α}
    {trace : ParLocTaggedTraceOf n Lifeline α}
    (hpar : ParLocOf branches trace) :
    ParLocOf
      (fun i A => (branches i A).map f)
      (fun A => (trace A).map (Tagged.mapVal f)) := by
  intro i A
  calc
    eraseBranch i ((trace A).map (Tagged.mapVal f))
        = (eraseBranch i (trace A)).map f := eraseBranch_mapVal i f (trace A)
    _ = (branches i A).map f := by rw [hpar i A]

/-- Tag every letter of a word with the same branch. -/
def taggedWordOf {n : Nat} {α : Type u}
    (i : Fin n) (word : Word α) : Word (Tagged (Fin n) α) :=
  word.map (fun letter => ({ branch := i, val := letter } :
    Tagged (Fin n) α))

theorem eraseBranch_taggedWordOf_same {n : Nat} {α : Type u}
    (i : Fin n) (word : Word α) :
    eraseBranch i (taggedWordOf i word) = word := by
  induction word with
  | nil =>
      simp [taggedWordOf, eraseBranch]
  | cons x xs ih =>
      change
        eraseBranch i
          (({ branch := i, val := x } : Tagged (Fin n) α) ::
            taggedWordOf i xs) = x :: xs
      simp [eraseBranch, ih]

theorem eraseBranch_taggedWordOf_ne {n : Nat} {α : Type u}
    {i j : Fin n} (hij : i ≠ j) (word : Word α) :
    eraseBranch i (taggedWordOf j word) = [] := by
  induction word with
  | nil =>
      simp [taggedWordOf, eraseBranch]
  | cons x xs ih =>
      change
        eraseBranch i
          (({ branch := j, val := x } : Tagged (Fin n) α) ::
            taggedWordOf j xs) = []
      have hji : ¬ j = i := fun h => hij h.symm
      simp [eraseBranch, hji, ih]

theorem eraseBranch_zero_taggedWordOf_succ {n : Nat} {α : Type u}
    (i : Fin n) (word : Word α) :
    eraseBranch (0 : Fin (n + 1)) (taggedWordOf i.succ word) = [] := by
  exact eraseBranch_taggedWordOf_ne
    (fun h => Fin.succ_ne_zero i h.symm) word

theorem eraseBranch_succ_taggedWordOf_succ {n : Nat} {α : Type u}
    (i j : Fin n) (word : Word α) :
    eraseBranch j.succ (taggedWordOf i.succ word) =
      eraseBranch j (taggedWordOf i word) := by
  induction word with
  | nil =>
      simp [taggedWordOf, eraseBranch]
  | cons x xs ih =>
      change
        eraseBranch j.succ
          (({ branch := i.succ, val := x } : Tagged (Fin (n + 1)) α) ::
            taggedWordOf i.succ xs) =
        eraseBranch j
          (({ branch := i, val := x } : Tagged (Fin n) α) ::
            taggedWordOf i xs)
      by_cases hij : i = j
      · subst hij
        simp [eraseBranch, ih]
      · have hsij : ¬ i.succ = j.succ := by
          intro h
          exact hij (Fin.succ_inj.mp h)
        simp [eraseBranch, hij, hsij, ih]

/-- Concatenate a list of words. -/
def concatWordList {α : Type u} : List (Word α) → Word α
  | [] => []
  | word :: words => word ++ concatWordList words

theorem concatWordList_ofFn_nil {n : Nat} {α : Type u}
    (words : Fin n → Word α)
    (hwords : ∀ i, words i = []) :
    concatWordList (List.ofFn words) = [] := by
  induction n with
  | zero =>
      simp [concatWordList]
  | succ n ih =>
      simp [List.ofFn_succ, concatWordList, hwords 0,
        ih (fun i : Fin n => words i.succ) (fun i => hwords i.succ)]

theorem concatWordList_ofFn_congr {n : Nat} {α : Type u}
    {left right : Fin n → Word α}
    (hwords : ∀ i, left i = right i) :
    concatWordList (List.ofFn left) =
      concatWordList (List.ofFn right) := by
  induction n with
  | zero =>
      simp [concatWordList]
  | succ n ih =>
      simp [List.ofFn_succ, concatWordList, hwords 0,
        ih (fun i : Fin n => hwords i.succ)]

theorem eraseBranch_concatWordList {Branch : Type u} {α : Type v}
    [DecidableEq Branch] (i : Branch)
    (words : List (Word (Tagged Branch α))) :
    eraseBranch i (concatWordList words) =
      concatWordList (words.map (eraseBranch i)) := by
  induction words with
  | nil =>
      simp [concatWordList, eraseBranch]
  | cons _ _ ih =>
      simp [concatWordList, eraseBranch_append, ih]

/--
Serial tagged interleaving of finitely many branch words.

The order is branch-index order.  This is a canonical witness that any family
of branch-local words has at least one tagged `parloc` interleaving.
-/
def serialTaggedWordOf {n : Nat} {α : Type u}
    (branches : Fin n → Word α) : Word (Tagged (Fin n) α) :=
  concatWordList (List.ofFn (fun i : Fin n => taggedWordOf i (branches i)))

theorem eraseBranch_serialTaggedWordOf {n : Nat} {α : Type u}
    (branches : Fin n → Word α) (i : Fin n) :
    eraseBranch i (serialTaggedWordOf branches) = branches i := by
  classical
  unfold serialTaggedWordOf
  induction n with
  | zero =>
      exact Fin.elim0 i
  | succ n ih =>
      rcases Fin.eq_zero_or_eq_succ i with rfl | ⟨j, rfl⟩
      ·
        simp only [List.ofFn_succ, concatWordList]
        have htailEmpty :
            eraseBranch (0 : Fin (n + 1))
              (concatWordList
                (List.ofFn
                  (fun i : Fin n => taggedWordOf i.succ (branches i.succ)))) =
              [] := by
          rw [eraseBranch_concatWordList]
          rw [List.map_ofFn]
          exact concatWordList_ofFn_nil _ (fun i =>
            eraseBranch_zero_taggedWordOf_succ i (branches i.succ))
        rw [eraseBranch_append, eraseBranch_taggedWordOf_same,
          htailEmpty, List.append_nil]
      ·
        have htail :
            eraseBranch j
              (serialTaggedWordOf (fun k : Fin n => branches k.succ)) =
              branches j.succ :=
          ih (fun k : Fin n => branches k.succ) j
        simp only [List.ofFn_succ, concatWordList]
        have hheadEmpty :
            eraseBranch j.succ (taggedWordOf (0 : Fin (n + 1))
              (branches 0)) = [] := by
          exact eraseBranch_taggedWordOf_ne
            (fun h => Fin.succ_ne_zero j h) (branches 0)
        have htail' :
            eraseBranch j.succ
              (concatWordList
                (List.ofFn
                  (fun i : Fin n => taggedWordOf i.succ (branches i.succ)))) =
              branches j.succ := by
          rw [eraseBranch_concatWordList]
          rw [List.map_ofFn]
          unfold serialTaggedWordOf at htail
          rw [eraseBranch_concatWordList] at htail
          rw [List.map_ofFn] at htail
          have hcongr :
              concatWordList
                (List.ofFn
                  (eraseBranch j.succ ∘
                    fun i : Fin n => taggedWordOf i.succ (branches i.succ))) =
              concatWordList
                (List.ofFn
                  (eraseBranch j ∘
                    fun i : Fin n => taggedWordOf i (branches i.succ))) :=
            concatWordList_ofFn_congr
              (fun i =>
                eraseBranch_succ_taggedWordOf_succ i j (branches i.succ))
          rw [hcongr]
          exact htail
        rw [eraseBranch_append, hheadEmpty, htail']
        simp

def serialParLocTraceOf {n : Nat} {Lifeline : Type u} {α : Type v}
    (branches : ParLocBranchesOf n Lifeline α) :
    ParLocTaggedTraceOf n Lifeline α :=
  fun A => serialTaggedWordOf (fun i => branches i A)

theorem parlocOf_serial {n : Nat} {Lifeline : Type u} {α : Type v}
    (branches : ParLocBranchesOf n Lifeline α) :
    ParLocOf branches (serialParLocTraceOf branches) := by
  intro i A
  exact eraseBranch_serialTaggedWordOf (fun i => branches i A) i

theorem parlocOf_extend_from_residual {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {target prefBranches residual : ParLocBranchesOf n Lifeline α}
    {prefixTrace residualTrace : ParLocTaggedTraceOf n Lifeline α}
    (hprefix : ParLocOf prefBranches prefixTrace)
    (hresidual : ParLocOf residual residualTrace)
    (htarget : ∀ i A, target i A = prefBranches i A ++ residual i A) :
    ParLocOf target (ParLocTaggedTraceOf.append prefixTrace residualTrace) := by
  intro i A
  calc
    eraseBranch i ((ParLocTaggedTraceOf.append prefixTrace residualTrace) A)
        = prefBranches i A ++ residual i A := by
          simpa [ParLocTaggedTraceOf.append, hprefix i A, hresidual i A]
            using eraseBranch_append (i := i) (prefixTrace A) (residualTrace A)
    _ = target i A := (htarget i A).symm

/-- Generic componentwise prefix-freeness. -/
def ComponentPrefixFreeOf {n : Nat} {Lifeline : Type u} {α : Type v}
    (complete : Fin n → Lifeline → Word α → Prop) : Prop :=
  ∀ i A {xs ys},
    complete i A xs → complete i A ys → ListPrefix xs ys → xs = ys

/-- Generic complete tagged local trace for a parallel region. -/
def TaggedParLocCompleteOf {n : Nat} {Lifeline : Type u} {α : Type v}
    (complete : Fin n → Lifeline → Word α → Prop)
    (trace : ParLocTaggedTraceOf n Lifeline α) : Prop :=
  ∃ branches : ParLocBranchesOf n Lifeline α,
    ParLocOf branches trace ∧ ∀ i A, complete i A (branches i A)

/-- Per-lifeline complete generic tagged `parloc` word. -/
def TaggedParLocCompleteLocalOf {n : Nat} {Lifeline : Type u}
    {α : Type v}
    (complete : Fin n → Lifeline → Word α → Prop)
    (A : Lifeline) (word : Word (Tagged (Fin n) α)) :
    Prop :=
  ∃ branches : Fin n → Word α,
    (∀ i, eraseBranch i word = branches i) ∧
    ∀ i, complete i A (branches i)

theorem TaggedParLocCompleteOf.local {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    {trace : ParLocTaggedTraceOf n Lifeline α}
    (hcomplete : TaggedParLocCompleteOf complete trace) (A : Lifeline) :
    TaggedParLocCompleteLocalOf complete A (trace A) := by
  rcases hcomplete with ⟨branches, hpar, hbranchesComplete⟩
  exact ⟨fun i => branches i A, fun i => hpar i A,
    fun i => hbranchesComplete i A⟩

theorem TaggedParLocCompleteOf.of_components {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    {branches : ParLocBranchesOf n Lifeline α}
    {trace : ParLocTaggedTraceOf n Lifeline α}
    (hpar : ParLocOf branches trace)
    (hcomplete : ∀ i A, complete i A (branches i A)) :
    TaggedParLocCompleteOf complete trace :=
  ⟨branches, hpar, hcomplete⟩

theorem TaggedParLocCompleteOf.map {n : Nat}
    {Lifeline : Type u} {α : Type v} {β : Type w}
    {completeα : Fin n → Lifeline → Word α → Prop}
    {completeβ : Fin n → Lifeline → Word β → Prop}
    (f : α → β)
    (hcompleteMap :
      ∀ i A word, completeα i A word → completeβ i A (word.map f))
    {trace : ParLocTaggedTraceOf n Lifeline α}
    (hcomplete : TaggedParLocCompleteOf completeα trace) :
    TaggedParLocCompleteOf completeβ
      (fun A => (trace A).map (Tagged.mapVal f)) := by
  rcases hcomplete with ⟨branches, hpar, hbranchesComplete⟩
  refine ⟨fun i A => (branches i A).map f, ?_, ?_⟩
  · exact parlocOf_map f hpar
  · intro i A
    exact hcompleteMap i A (branches i A) (hbranchesComplete i A)

theorem TaggedParLocCompleteLocalOf.of_components {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    {branches : ParLocBranchesOf n Lifeline α}
    {trace : ParLocTaggedTraceOf n Lifeline α}
    (hpar : ParLocOf branches trace)
    {A : Lifeline}
    (hcomplete : ∀ i, complete i A (branches i A)) :
    TaggedParLocCompleteLocalOf complete A (trace A) := by
  exact ⟨fun i => branches i A, fun i => hpar i A, hcomplete⟩

theorem TaggedParLocCompleteLocalOf.map {n : Nat}
    {Lifeline : Type u} {α : Type v} {β : Type w}
    {completeα : Fin n → Lifeline → Word α → Prop}
    {completeβ : Fin n → Lifeline → Word β → Prop}
    (f : α → β)
    (hcompleteMap :
      ∀ i A word, completeα i A word → completeβ i A (word.map f))
    {A : Lifeline} {word : Word (Tagged (Fin n) α)}
    (hcomplete : TaggedParLocCompleteLocalOf completeα A word) :
    TaggedParLocCompleteLocalOf completeβ A
      (word.map (Tagged.mapVal f)) := by
  rcases hcomplete with ⟨branches, herase, hbranchesComplete⟩
  refine ⟨fun i => (branches i).map f, ?_, ?_⟩
  · intro i
    calc
      eraseBranch i (word.map (Tagged.mapVal f))
          = (eraseBranch i word).map f := eraseBranch_mapVal i f word
      _ = (branches i).map f := by rw [herase i]
  · intro i
    exact hcompleteMap i A (branches i) (hbranchesComplete i)

theorem TaggedParLocCompleteLocalOf.prefix_free {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    (hprefixFree : ComponentPrefixFreeOf complete) :
    LocalPrefixFree (TaggedParLocCompleteLocalOf complete) := by
  intro A u v hu hv hp
  rcases hu with ⟨branchesU, hparU, hcompleteU⟩
  rcases hv with ⟨branchesV, hparV, hcompleteV⟩
  apply listPrefix_eq_of_eraseBranch_eq hp
  intro i
  have hprefix_i : ListPrefix (branchesU i) (branchesV i) := by
    have hpErase := eraseBranch_preserves_prefix (i := i) hp
    simpa [hparU i, hparV i] using hpErase
  have hbranchEq : branchesU i = branchesV i :=
    hprefixFree i A (hcompleteU i) (hcompleteV i) hprefix_i
  calc
    eraseBranch i u = branchesU i := hparU i
    _ = branchesV i := hbranchEq
    _ = eraseBranch i v := (hparV i).symm

theorem parlocOf_residual_empty_of_components_empty {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {residual : ParLocBranchesOf n Lifeline α}
    {residualTrace : ParLocTaggedTraceOf n Lifeline α}
    (hresidual : ParLocOf residual residualTrace)
    {A : Lifeline}
    (hempty : ∀ i, residual i A = []) :
    residualTrace A = [] := by
  apply eraseBranch_all_empty_nil
  intro i
  rw [hresidual i A, hempty i]

theorem parlocOf_complete_prefix_free {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    (hprefixFree : ComponentPrefixFreeOf complete)
    {u v : ParLocTaggedTraceOf n Lifeline α}
    (hu : TaggedParLocCompleteOf complete u)
    (hv : TaggedParLocCompleteOf complete v)
    (hp : TracePrefixOf u v) :
    u = v := by
  rcases hu with ⟨branchesU, hparU, hcompleteU⟩
  rcases hv with ⟨branchesV, hparV, hcompleteV⟩
  funext A
  apply listPrefix_eq_of_eraseBranch_eq (hp A)
  intro i
  have hprefix_i : ListPrefix (branchesU i A) (branchesV i A) := by
    have hpErase := eraseBranch_preserves_prefix (i := i) (hp A)
    simpa [hparU i A, hparV i A] using hpErase
  have hbranchEq : branchesU i A = branchesV i A :=
    hprefixFree i A (hcompleteU i A) (hcompleteV i A) hprefix_i
  calc
    eraseBranch i (u A) = branchesU i A := hparU i A
    _ = branchesV i A := hbranchEq
    _ = eraseBranch i (v A) := (hparV i A).symm

theorem parlocOf_prefix_decompose {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {branches : ParLocBranchesOf n Lifeline α}
    {full pref : ParLocTaggedTraceOf n Lifeline α}
    (hpar : ParLocOf branches full)
    (hpref : TracePrefixOf pref full) :
    ∃ branchPrefix : ParLocBranchesOf n Lifeline α,
      ParLocOf branchPrefix pref ∧
      ∀ i A, ListPrefix (branchPrefix i A) (branches i A) := by
  refine ⟨fun i A => eraseBranch i (pref A), ?_, ?_⟩
  · intro i A
    rfl
  · intro i A
    have hp := eraseBranch_preserves_prefix (i := i) (hpref A)
    simpa [hpar i A] using hp

/-- A branch-local trace for each lifeline. -/
abbrev ParLocTrace (Lifeline : Type u) (Channel : Type v) :=
  Lifeline → Word (LocalLetter Lifeline Channel)

/-- One branch-local trace assignment per parallel branch. -/
abbrev ParLocBranches (n : Nat) (Lifeline : Type u) (Channel : Type v) :=
  Fin n → ParLocTrace Lifeline Channel

/--
Tagged local trace of a parallel region.

The tags are proof-level branch-origin information.  They make prefix
decomposition precise: erasing branch `i` from each lifeline recovers that
branch's local contribution.
-/
abbrev ParLocTaggedTrace (n : Nat) (Lifeline : Type u) (Channel : Type v) :=
  Lifeline → Word (Tagged (Fin n) (LocalLetter Lifeline Channel))

/-- Pointwise prefix relation on tagged local traces. -/
def TracePrefix {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (pref full : ParLocTaggedTrace n Lifeline Channel) : Prop :=
  ∀ A, ListPrefix (pref A) (full A)

/--
Local semantics of a parallel region, at the trace level.

For each lifeline and branch, erasing all events not tagged by that branch from
the shuffled local trace yields the branch-local trace on that lifeline.
-/
def ParLoc {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (branches : ParLocBranches n Lifeline Channel)
    (trace : ParLocTaggedTrace n Lifeline Channel) : Prop :=
  FinBranchShuffle branches trace

namespace ParLocBranches

def append {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (left right : ParLocBranches n Lifeline Channel) :
    ParLocBranches n Lifeline Channel :=
  fun i A => left i A ++ right i A

end ParLocBranches

namespace ParLocTaggedTrace

def append {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (left right : ParLocTaggedTrace n Lifeline Channel) :
    ParLocTaggedTrace n Lifeline Channel :=
  fun A => left A ++ right A

end ParLocTaggedTrace

/-- Forget branch-origin tags, recovering the observable untagged local trace. -/
def forgetBranchTag {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (trace : ParLocTaggedTrace n Lifeline Channel) :
    ParLocTrace Lifeline Channel :=
  fun A => (trace A).map Tagged.val

/--
Untagged view of `ParLoc`: an observable trace is a tag-forgetting projection
of some tagged trace satisfying the erasure characterization.
-/
def UntaggedParLoc {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (branches : ParLocBranches n Lifeline Channel)
    (trace : ParLocTrace Lifeline Channel) : Prop :=
  ∃ tagged : ParLocTaggedTrace n Lifeline Channel,
    ParLoc branches tagged ∧ forgetBranchTag tagged = trace

theorem parloc_trace_iff_finBranchShuffle {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : ParLocBranches n Lifeline Channel}
    {trace : ParLocTaggedTrace n Lifeline Channel} :
    ParLoc branches trace ↔ FinBranchShuffle branches trace :=
  Iff.rfl

/--
For a `ParLoc` trace, branch erasure recovers the untagged branch-local word.

The `eraseBranch` operation already drops the branch tag, so no extra
`Tagged.val` map is needed on the left-hand side.
-/
theorem parloc_forgetBranch_eq {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : ParLocBranches n Lifeline Channel}
    {trace : ParLocTaggedTrace n Lifeline Channel}
    (hpar : ParLoc branches trace) (i : Fin n) (A : Lifeline) :
    eraseBranch i (trace A) = branches i A :=
  hpar i A

theorem parloc_forgetBranchTag_witness {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : ParLocBranches n Lifeline Channel}
    {trace : ParLocTaggedTrace n Lifeline Channel}
    (hpar : ParLoc branches trace) :
    UntaggedParLoc branches (forgetBranchTag trace) := by
  exact ⟨trace, hpar, rfl⟩

theorem forgetBranchTag_preserves_prefix {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {pref full : ParLocTaggedTrace n Lifeline Channel}
    (hpref : TracePrefix pref full) :
    ∀ A, ListPrefix (forgetBranchTag pref A) (forgetBranchTag full A) := by
  intro A
  exact list_map_preserves_prefix Tagged.val (hpref A)

theorem tracePrefix_append_left {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    (left right : ParLocTaggedTrace n Lifeline Channel) :
    TracePrefix left (ParLocTaggedTrace.append left right) := by
  intro A
  exact ⟨right A, rfl⟩

theorem parloc_append {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {leftBranches rightBranches : ParLocBranches n Lifeline Channel}
    {leftTrace rightTrace : ParLocTaggedTrace n Lifeline Channel}
    (hleft : ParLoc leftBranches leftTrace)
    (hright : ParLoc rightBranches rightTrace) :
    ParLoc (ParLocBranches.append leftBranches rightBranches)
      (ParLocTaggedTrace.append leftTrace rightTrace) := by
  intro i A
  simp [ParLocBranches.append, ParLocTaggedTrace.append,
    eraseBranch_append, hleft i A, hright i A]

theorem parloc_extend_from_residual {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {target prefBranches residual : ParLocBranches n Lifeline Channel}
    {prefixTrace residualTrace : ParLocTaggedTrace n Lifeline Channel}
    (hprefix : ParLoc prefBranches prefixTrace)
    (hresidual : ParLoc residual residualTrace)
    (htarget : ∀ i A, target i A = prefBranches i A ++ residual i A) :
    ParLoc target (ParLocTaggedTrace.append prefixTrace residualTrace) := by
  intro i A
  calc
    eraseBranch i ((ParLocTaggedTrace.append prefixTrace residualTrace) A)
        = prefBranches i A ++ residual i A := by
          simpa [ParLocTaggedTrace.append, hprefix i A, hresidual i A]
            using eraseBranch_append (i := i) (prefixTrace A) (residualTrace A)
    _ = target i A := (htarget i A).symm

/--
Componentwise prefix-freeness for branch-local complete traces.

This abstracts the base Isola prefix-freeness theorem for the component
languages.  The `parloc` case below shows how this property lifts through
branch-origin tagged interleavings.
-/
def ComponentPrefixFree {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (complete : Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop) :
    Prop :=
  ∀ i A {xs ys},
    complete i A xs → complete i A ys → ListPrefix xs ys → xs = ys

/--
A complete tagged local trace for a parallel region.

The witness records the branch-local complete traces whose tagged interleaving
forms the observable parallel trace.
-/
def TaggedParLocComplete {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (complete : Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop)
    (trace : ParLocTaggedTrace n Lifeline Channel) : Prop :=
  ∃ branches : ParLocBranches n Lifeline Channel,
    ParLoc branches trace ∧ ∀ i A, complete i A (branches i A)

/--
Per-lifeline complete tagged `parloc` word.

This is the local-word version needed by `ZipPost`: a single tagged word is
complete at lifeline `A` when its branch erasures are complete component words
at `A`.
-/
def TaggedParLocCompleteLocal {n : Nat} {Lifeline : Type u}
    {Channel : Type v}
    (complete : Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop)
    (A : Lifeline) (word : Word (Tagged (Fin n) (LocalLetter Lifeline Channel))) :
    Prop :=
  ∃ branches : Fin n → Word (LocalLetter Lifeline Channel),
    (∀ i, eraseBranch i word = branches i) ∧
    ∀ i, complete i A (branches i)

theorem TaggedParLocComplete.local {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete : Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    {trace : ParLocTaggedTrace n Lifeline Channel}
    (hcomplete : TaggedParLocComplete complete trace) (A : Lifeline) :
    TaggedParLocCompleteLocal complete A (trace A) := by
  rcases hcomplete with ⟨branches, hpar, hbranchesComplete⟩
  exact ⟨fun i => branches i A, fun i => hpar i A,
    fun i => hbranchesComplete i A⟩

theorem TaggedParLocCompleteLocal.of_components {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete : Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    {branches : ParLocBranches n Lifeline Channel}
    {trace : ParLocTaggedTrace n Lifeline Channel}
    (hpar : ParLoc branches trace)
    {A : Lifeline}
    (hcomplete : ∀ i, complete i A (branches i A)) :
    TaggedParLocCompleteLocal complete A (trace A) := by
  exact ⟨fun i => branches i A, fun i => hpar i A, hcomplete⟩

theorem parloc_complete_prefix_decompose_local {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete : Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    {branches : ParLocBranches n Lifeline Channel}
    {full pref : ParLocTaggedTrace n Lifeline Channel}
    (hpar : ParLoc branches full)
    (hpref : TracePrefix pref full)
    {A : Lifeline}
    (hprefComplete : TaggedParLocCompleteLocal complete A (pref A)) :
    ∃ branchPrefix : Fin n → Word (LocalLetter Lifeline Channel),
      (∀ i, eraseBranch i (pref A) = branchPrefix i) ∧
      (∀ i, ListPrefix (branchPrefix i) (branches i A)) ∧
      ∀ i, complete i A (branchPrefix i) := by
  rcases hprefComplete with ⟨completeBranches, hcompletePar, hcomplete⟩
  refine ⟨fun i => eraseBranch i (pref A), ?_, ?_, ?_⟩
  · intro i
    rfl
  · intro i
    have hp := eraseBranch_preserves_prefix (i := i) (hpref A)
    simpa [hpar i A] using hp
  · intro i
    change complete i A (eraseBranch i (pref A))
    rw [hcompletePar i]
    exact hcomplete i

theorem parloc_residual_empty_of_components_empty {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {residual : ParLocBranches n Lifeline Channel}
    {residualTrace : ParLocTaggedTrace n Lifeline Channel}
    (hresidual : ParLoc residual residualTrace)
    {A : Lifeline}
    (hempty : ∀ i, residual i A = []) :
    residualTrace A = [] := by
  apply eraseBranch_all_empty_nil
  intro i
  rw [hresidual i A, hempty i]

theorem TaggedParLocCompleteLocal.prefix_free {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete : Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    (hprefixFree : ComponentPrefixFree complete) :
    LocalPrefixFree (TaggedParLocCompleteLocal complete) := by
  intro A u v hu hv hp
  rcases hu with ⟨branchesU, hparU, hcompleteU⟩
  rcases hv with ⟨branchesV, hparV, hcompleteV⟩
  apply listPrefix_eq_of_eraseBranch_eq hp
  intro i
  have hprefix_i : ListPrefix (branchesU i) (branchesV i) := by
    have hpErase := eraseBranch_preserves_prefix (i := i) hp
    simpa [hparU i, hparV i] using hpErase
  have hbranchEq : branchesU i = branchesV i :=
    hprefixFree i A (hcompleteU i) (hcompleteV i) hprefix_i
  calc
    eraseBranch i u = branchesU i := hparU i
    _ = branchesV i := hbranchEq
    _ = eraseBranch i v := (hparV i).symm

theorem parloc_complete_prefix_decompose {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete : Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    {branches : ParLocBranches n Lifeline Channel}
    {full pref : ParLocTaggedTrace n Lifeline Channel}
    (hpar : ParLoc branches full)
    (hpref : TracePrefix pref full)
    (hprefComplete : TaggedParLocComplete complete pref) :
    ∃ branchPrefix : ParLocBranches n Lifeline Channel,
      ParLoc branchPrefix pref ∧
      (∀ i A, ListPrefix (branchPrefix i A) (branches i A)) ∧
      ∀ i A, complete i A (branchPrefix i A) := by
  rcases hprefComplete with ⟨completeBranches, hcompletePar, hcomplete⟩
  refine ⟨fun i A => eraseBranch i (pref A), ?_, ?_, ?_⟩
  · intro i A
    rfl
  · intro i A
    have hp := eraseBranch_preserves_prefix (i := i) (hpref A)
    simpa [hpar i A]
      using hp
  · intro i A
    change complete i A (eraseBranch i (pref A))
    rw [hcompletePar i A]
    exact hcomplete i A

theorem parloc_complete_prefix_free {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete : Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    (hprefixFree : ComponentPrefixFree complete)
    {u v : ParLocTaggedTrace n Lifeline Channel}
    (hu : TaggedParLocComplete complete u)
    (hv : TaggedParLocComplete complete v)
    (hp : TracePrefix u v) :
    u = v := by
  rcases hu with ⟨branchesU, hparU, hcompleteU⟩
  rcases hv with ⟨branchesV, hparV, hcompleteV⟩
  funext A
  apply listPrefix_eq_of_eraseBranch_eq (hp A)
  intro i
  have hprefix_i : ListPrefix (branchesU i A) (branchesV i A) := by
    have hpErase := eraseBranch_preserves_prefix (i := i) (hp A)
    simpa [hparU i A, hparV i A]
      using hpErase
  have hbranchEq :
      branchesU i A = branchesV i A :=
    hprefixFree i A (hcompleteU i A) (hcompleteV i A) hprefix_i
  calc
    eraseBranch i (u A) = branchesU i A := hparU i A
    _ = branchesV i A := hbranchEq
    _ = eraseBranch i (v A) := (hparV i A).symm

/--
Prefix decomposition for local parallel traces.

If a tagged local trace is a pointwise prefix of a `ParLoc` trace, then each
branch contributes a pointwise prefix of its full branch-local trace.  No
completeness assumption is used.
-/
theorem parloc_prefix_decompose {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : ParLocBranches n Lifeline Channel}
    {full pref : ParLocTaggedTrace n Lifeline Channel}
    (hpar : ParLoc branches full)
    (hpref : TracePrefix pref full) :
    ∃ branchPrefix : ParLocBranches n Lifeline Channel,
      ParLoc branchPrefix pref ∧
      ∀ i A, ListPrefix (branchPrefix i A) (branches i A) := by
  refine ⟨fun i A => eraseBranch i (pref A), ?_, ?_⟩
  · intro i A
    rfl
  · intro i A
    have hp := eraseBranch_preserves_prefix (i := i) (hpref A)
    simpa [hpar i A]
      using hp

end ZipperGenParallel
