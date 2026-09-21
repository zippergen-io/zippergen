import ZipperGenParallel.Parloc

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- Number of selected letters strictly before position `index`. -/
def selectedRank (pick : alpha -> Option beta) (word : List alpha)
    (index : Nat) : Nat :=
  ((word.take index).filterMap pick).length

@[simp] theorem selectedRank_nil (pick : alpha -> Option beta) (index : Nat) :
    selectedRank pick [] index = 0 := by
  simp [selectedRank]

@[simp] theorem selectedRank_zero (pick : alpha -> Option beta)
    (word : List alpha) :
    selectedRank pick word 0 = 0 := by
  simp [selectedRank]

@[simp] theorem selectedRank_cons_succ (pick : alpha -> Option beta)
    (head : alpha) (tail : List alpha) (index : Nat) :
    selectedRank pick (head :: tail) (index + 1) =
      (if (pick head).isSome then 1 else 0) +
        selectedRank pick tail index := by
  cases h : pick head <;> simp [selectedRank, h, Nat.add_comm]

/-- Selected positions are strictly ordered by their rank. -/
theorem selectedRank_strict (pick : alpha -> Option beta) :
    forall (word : List alpha) {left right : Nat},
      (hleft : left < word.length) -> right < word.length -> left < right ->
      (pick (word.get ⟨left, hleft⟩)).isSome = true ->
      selectedRank pick word left < selectedRank pick word right := by
  intro word
  induction word with
  | nil =>
      intro left right hleft
      simp at hleft
  | cons head tail ih =>
      intro left right hleft hright hlt hselected
      cases left with
      | zero =>
          cases right with
          | zero => omega
          | succ right =>
              simp only [selectedRank_zero, selectedRank_cons_succ]
              have hhead : (pick head).isSome = true := by
                simpa using hselected
              simp [hhead]
              omega
      | succ left =>
          cases right with
          | zero => omega
          | succ right =>
              simp only [selectedRank_cons_succ]
              have hleft' : left < tail.length := by simpa using hleft
              have hright' : right < tail.length := by simpa using hright
              have hlt' : left < right := by omega
              have hselected' :
                  (pick (tail.get ⟨left, hleft'⟩)).isSome = true := by
                simpa using hselected
              exact Nat.add_lt_add_left
                (ih hleft' hright' hlt' hselected') _

theorem selectedRank_injective_at_selected (pick : alpha -> Option beta)
    (word : List alpha) {left right : Nat}
    (hleft : left < word.length) (hright : right < word.length)
    (hleftSelected : (pick (word.get ⟨left, hleft⟩)).isSome = true)
    (hrightSelected : (pick (word.get ⟨right, hright⟩)).isSome = true)
    (hrank : selectedRank pick word left = selectedRank pick word right) :
    left = right := by
  rcases Nat.lt_trichotomy left right with hlt | heq | hgt
  · exact False.elim ((Nat.ne_of_lt
      (selectedRank_strict pick word hleft hright hlt hleftSelected)) hrank)
  · exact heq
  · exact False.elim ((Nat.ne_of_lt
      (selectedRank_strict pick word hright hleft hgt hrightSelected)) hrank.symm)

theorem selectedRank_lt_iff_at_selected (pick : alpha -> Option beta)
    (word : List alpha) {left right : Nat}
    (hleft : left < word.length) (hright : right < word.length)
    (hleftSelected : (pick (word.get ⟨left, hleft⟩)).isSome = true)
    (hrightSelected : (pick (word.get ⟨right, hright⟩)).isSome = true) :
    selectedRank pick word left < selectedRank pick word right <->
      left < right := by
  constructor
  · intro hrank
    rcases Nat.lt_trichotomy left right with hlt | heq | hgt
    · exact hlt
    · subst right
      exact False.elim (Nat.lt_irrefl _ hrank)
    · have hreverse := selectedRank_strict pick word hright hleft hgt
        hrightSelected
      exact False.elim (Nat.not_lt_of_ge (Nat.le_of_lt hreverse) hrank)
  · intro hlt
    exact selectedRank_strict pick word hleft hright hlt hleftSelected

/-- A selected source letter occurs in the filtered word exactly at its
    number of selected predecessors. -/
theorem filterMap_getElem?_selectedRank (pick : alpha -> Option beta) :
    forall (word : List alpha) (index : Nat) (item : beta),
      (hbound : index < word.length) ->
      pick (word.get ⟨index, hbound⟩) = some item ->
      (word.filterMap pick)[selectedRank pick word index]? = some item := by
  intro word
  induction word with
  | nil =>
      intro index item hbound
      simp at hbound
  | cons head tail ih =>
      intro index item hbound hpick
      cases index with
      | zero =>
          have hhead : pick head = some item := by simpa using hpick
          simp [hhead, selectedRank]
      | succ index =>
          have htail : index < tail.length := by simpa using hbound
          have hpickTail : pick (tail.get ⟨index, htail⟩) = some item := by
            simpa using hpick
          cases hhead : pick head with
          | none =>
              simpa [hhead, selectedRank] using ih index item htail hpickTail
          | some headItem =>
              simpa [hhead, selectedRank] using ih index item htail hpickTail

/-- Conversely, every item at a filtered index comes from a unique selected
    source position with that rank. -/
theorem exists_selected_of_filterMap_getElem? (pick : alpha -> Option beta) :
    forall (word : List alpha) (rank : Nat) (item : beta),
      (word.filterMap pick)[rank]? = some item ->
      exists index, exists hbound : index < word.length,
        pick (word.get ⟨index, hbound⟩) = some item /\
        selectedRank pick word index = rank := by
  intro word
  induction word with
  | nil =>
      intro rank item hget
      simp at hget
  | cons head tail ih =>
      intro rank item hget
      cases hhead : pick head with
      | none =>
          have htail : (tail.filterMap pick)[rank]? = some item := by
            simpa [hhead] using hget
          rcases ih rank item htail with ⟨index, hbound, hpick, hrank⟩
          refine ⟨index + 1, by simpa, ?_, ?_⟩
          · simpa using hpick
          · simpa [selectedRank, hhead] using hrank
      | some headItem =>
          cases rank with
          | zero =>
              have heq : headItem = item := by simpa [hhead] using hget
              subst headItem
              exact ⟨0, by simp, by simpa [hhead], by simp [selectedRank]⟩
          | succ rank =>
              have htail : (tail.filterMap pick)[rank]? = some item := by
                simpa [hhead] using hget
              rcases ih rank item htail with ⟨index, hbound, hpick, hrank⟩
              refine ⟨index + 1, by simpa, ?_, ?_⟩
              · simpa using hpick
              · simpa [selectedRank, hhead, hrank]

/-- List prefixes preserve every defined lookup. -/
theorem ListPrefix.getElem?_eq {left right : List alpha}
    (hprefix : ListPrefix left right) {index : Nat} {item : alpha}
    (hget : left[index]? = some item) :
    right[index]? = some item := by
  rcases hprefix with ⟨suffix, rfl⟩
  rw [List.getElem?_append]
  split
  · assumption
  · rename_i hnot
    have : index < left.length := List.getElem?_eq_some_iff.mp hget |>.1
    omega

theorem filterMap_take_selected_position (pick : alpha -> Option beta)
    (word : List alpha) (index : Nat) (hbound : index < word.length)
    (hselected : (pick (word.get ⟨index, hbound⟩)).isSome = true) :
    (word.take index).filterMap pick =
      (word.filterMap pick).take (selectedRank pick word index) := by
  have hprefix : ListPrefix ((word.take index).filterMap pick)
      (word.filterMap pick) := by
    refine ⟨(word.drop index).filterMap pick, ?_⟩
    rw [← List.filterMap_append, List.take_append_drop]
  rcases hprefix with ⟨suffix, hsuffix⟩
  rw [← hsuffix]
  simp [selectedRank]

theorem selectedRank_of_prefix (pick : alpha -> Option beta)
    {pre word : List alpha} (hprefix : ListPrefix pre word)
    {index : Nat} (hbound : index < pre.length) :
    selectedRank pick word index = selectedRank pick pre index := by
  rcases hprefix with ⟨suffix, rfl⟩
  simp [selectedRank, List.take_append_of_le_length (Nat.le_of_lt hbound)]


end ZipperGenSpawn
