/-
  Causal-cut predicates for finite MSC prefixes.

  A safety claim about an unfinished execution should describe order at the
  cut, rather than merely assert that two events occur somewhere in a completed
  run.  `PrefixBefore p q w` says that every `q` event in a local word has a
  preceding `p` event in that same word.  It is consequently preserved when a
  componentwise prefix is observed.
-/

import ZipperGenSpawn.PrefixSafety

namespace ZipperGenSpawn

section CausalCut

variable {α : Type}

/-- Every letter selected by `q` has an earlier letter selected by `p`. -/
def PrefixBefore (p q : α → Bool) (word : List α) : Prop :=
  ∀ before x after,
    word = before ++ x :: after →
    q x = true →
    before.any p = true

/-- An empty word has no selected target event. -/
theorem PrefixBefore.nil (p q : α → Bool) : PrefixBefore p q [] := by
  intro before x after hsplit _
  simp at hsplit

/-- If a word has no selected target event, every predecessor requirement is
    vacuously true. -/
theorem PrefixBefore.of_no_target {p q : α → Bool} {word : List α}
    (h : ∀ x, x ∈ word → q x = false) : PrefixBefore p q word := by
  intro before x after hsplit hx
  have hxmem : x ∈ word := by
    rw [hsplit]
    simp
  have hfalse := h x hxmem
  simp [hfalse] at hx

/-- Local causal-cut order is stable under taking a word prefix. -/
theorem PrefixBefore.of_append {p q : α → Bool} {pre suffix : List α}
    (h : PrefixBefore p q (pre ++ suffix)) :
    PrefixBefore p q pre := by
  intro before x after hsplit hx
  have hfull : pre ++ suffix = before ++ x :: (after ++ suffix) := by
    rw [hsplit]
    simp [List.append_assoc]
  exact h before x (after ++ suffix) hfull hx

/-- A leading predecessor is enough to justify every later selected event. -/
theorem PrefixBefore.cons_of_pred {p q : α → Bool} {a : α} {tail : List α}
    (hp : p a = true) (hq : q a = false) :
    PrefixBefore p q (a :: tail) := by
  intro before x after hsplit hx
  cases before with
  | nil =>
      simp only [List.nil_append] at hsplit
      have hax : a = x := List.cons.inj hsplit |>.1
      subst x
      simp [hq] at hx
  | cons b rest =>
      simp only [List.cons_append] at hsplit
      have hab : a = b := List.cons.inj hsplit |>.1
      subst b
      simp [hp]

/-- A leading non-target event preserves an order fact for the tail. -/
theorem PrefixBefore.cons_of_non_target {p q : α → Bool} {a : α} {tail : List α}
    (hq : q a = false) (h : PrefixBefore p q tail) :
    PrefixBefore p q (a :: tail) := by
  intro before x after hsplit hx
  cases before with
  | nil =>
      simp only [List.nil_append] at hsplit
      have hax : a = x := List.cons.inj hsplit |>.1
      subst x
      simp [hq] at hx
  | cons b rest =>
      simp only [List.cons_append] at hsplit
      have hab : a = b := List.cons.inj hsplit |>.1
      subst b
      have htail : tail = rest ++ x :: after := List.cons.inj hsplit |>.2
      have hpref := h rest x after htail hx
      simp [hpref]

/-- Prepending a word with no selected target event preserves an order fact. -/
theorem PrefixBefore.append_left_of_no_target
    {p q : α → Bool} {pre tail : List α}
    (hpre : ∀ x, x ∈ pre → q x = false)
    (htail : PrefixBefore p q tail) :
    PrefixBefore p q (pre ++ tail) := by
  induction pre with
  | nil =>
      simpa using htail
  | cons a rest ih =>
      simp only [List.cons_append]
      apply PrefixBefore.cons_of_non_target (hpre a (by simp))
      apply ih
      · intro x hx
        exact hpre x (by simp [hx])

/-- Appending events that are not selected by `q` preserves an existing
    predecessor requirement. -/
theorem PrefixBefore.append_right_of_no_target
    {p q : α → Bool} {pre tail : List α}
    (hpre : PrefixBefore p q pre)
    (htail : ∀ x, x ∈ tail → q x = false) :
    PrefixBefore p q (pre ++ tail) := by
  intro before x after hsplit hx
  rcases (List.append_eq_append_iff.mp hsplit) with hleft | hright
  · rcases hleft with ⟨middle, hpreSplit, htailSplit⟩
    have hxmem : x ∈ tail := by
      rw [htailSplit]
      simp
    have hfalse := htail x hxmem
    simp [hfalse] at hx
  · rcases hright with ⟨middle, hpreSplit, htailSplit⟩
    cases middle with
    | nil =>
        simp only [List.nil_append] at htailSplit
        have hxmem : x ∈ tail := by
          rw [← htailSplit]
          simp
        have hfalse := htail x hxmem
        simp [hfalse] at hx
    | cons y middle =>
        simp only [List.cons_append] at htailSplit
        have hxy : x = y := List.cons.inj htailSplit |>.1
        subst y
        exact hpre before x middle hpreSplit hx

variable {L C F Payload : Type}

/-- Lift a local order fact to a tuple at one participant. -/
def AtParticipant (A : L) (p q : Letter L C F Payload → Bool)
    (M : WordTuple L C F Payload) : Prop :=
  PrefixBefore p q ((M A).map Subtype.val)

/-- A local causal fact depends only on the local word of its participant.
    This lets a global induction carry facts for later stages across a prefix
    that is empty at those participants. -/
theorem atParticipant_congr
    {A : L} {p q : Letter L C F Payload → Bool}
    {M N : WordTuple L C F Payload}
    (h : M A = N A) :
    AtParticipant A p q M ↔ AtParticipant A p q N := by
  unfold AtParticipant
  rw [h]

/-- A finite conjunction of local order facts is componentwise prefix-closed. -/
theorem atParticipant_prefix
    {A : L} {p q : Letter L C F Payload → Bool}
    {M M' : WordTuple L C F Payload}
    (hprefix : IsPrefixTuple M M')
    (h : AtParticipant A p q M') :
    AtParticipant A p q M := by
  rcases hprefix A with ⟨suffix, hsuffix⟩
  unfold AtParticipant at h ⊢
  rw [hsuffix, List.map_append] at h
  exact PrefixBefore.of_append h

end CausalCut

end ZipperGenSpawn
