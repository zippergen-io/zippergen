import ZipperGenParallel.LocalProgram

namespace ZipperGenParallel

universe u v w

/-- A tuple of local traces, indexed by lifeline. -/
abbrev TraceTuple (Lifeline : Type u) (α : Type v) :=
  Lifeline → Word α

/-- Pointwise concatenation of local trace tuples. -/
def TraceTuple.append {Lifeline : Type u} {α : Type v}
    (U V : TraceTuple Lifeline α) : TraceTuple Lifeline α :=
  fun A => U A ++ V A

def TraceTuple.empty (Lifeline : Type u) (α : Type v) :
    TraceTuple Lifeline α :=
  fun _ => []

theorem TraceTuple.append_empty {Lifeline : Type u} {α : Type v}
    (U : TraceTuple Lifeline α) :
    TraceTuple.append U (TraceTuple.empty Lifeline α) = U := by
  funext A
  simp [TraceTuple.append, TraceTuple.empty]

theorem TraceTuple.empty_append {Lifeline : Type u} {α : Type v}
    (U : TraceTuple Lifeline α) :
    TraceTuple.append (TraceTuple.empty Lifeline α) U = U := by
  funext A
  simp [TraceTuple.append, TraceTuple.empty]

namespace WordEvent

variable {Lifeline : Type u} {Channel : Type v}
variable {U V : TraceTuple Lifeline (LocalLetter Lifeline Channel)}

/-- Embed an event of the left component into a pointwise tuple append. -/
def appendLeft (e : WordEvent U) :
    WordEvent (TraceTuple.append U V) where
  owner := e.owner
  index := e.index
  inBounds := by
    have h := e.inBounds
    simp [TraceTuple.append, List.length_append]
    omega

/-- Embed an event of the right component into a pointwise tuple append. -/
def appendRight (e : WordEvent V) :
    WordEvent (TraceTuple.append U V) where
  owner := e.owner
  index := (U e.owner).length + e.index
  inBounds := by
    simpa [TraceTuple.append, List.length_append, Nat.add_assoc] using
      Nat.add_lt_add_left e.inBounds (List.length (U e.owner))

@[simp]
theorem appendLeft_owner (e : WordEvent U) :
    (appendLeft (U := U) (V := V) e).owner = e.owner :=
  rfl

@[simp]
theorem appendRight_owner (e : WordEvent V) :
    (appendRight (U := U) (V := V) e).owner = e.owner :=
  rfl

@[simp]
theorem appendLeft_index (e : WordEvent U) :
    (appendLeft (U := U) (V := V) e).index = e.index :=
  rfl

@[simp]
theorem appendRight_index (e : WordEvent V) :
    (appendRight (U := U) (V := V) e).index =
      (U e.owner).length + e.index :=
  rfl

@[simp]
theorem appendLeft_localIndex (e : WordEvent U) :
    (appendLeft (U := U) (V := V) e).localIndex = e.localIndex :=
  rfl

@[simp]
theorem appendRight_localIndex (e : WordEvent V) :
    (appendRight (U := U) (V := V) e).localIndex =
      (U e.owner).length + e.localIndex := by
  simp [WordEvent.localIndex, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

@[simp]
theorem appendLeft_letter (e : WordEvent U) :
    (appendLeft (U := U) (V := V) e).letter = e.letter := by
  cases e with
  | mk owner index inBounds =>
      simp only [appendLeft, WordEvent.letter, TraceTuple.append]
      exact List.getElem_append_left inBounds

@[simp]
theorem appendRight_letter (e : WordEvent V) :
    (appendRight (U := U) (V := V) e).letter = e.letter := by
  cases e with
  | mk owner index inBounds =>
      simp only [appendRight, WordEvent.letter, TraceTuple.append]
      have hbound :
          (U owner).length + index < (U owner ++ V owner).length := by
        simp [List.length_append]
        omega
      change
        (U owner ++ V owner)[(U owner).length + index]'hbound =
          (V owner)[index]'inBounds
      rw [List.getElem_append_right (h₁ := by omega)]
      simp

@[simp]
theorem appendLeft_kind (e : WordEvent U) :
    (appendLeft (U := U) (V := V) e).kind = e.kind := by
  simp [WordEvent.kind]

@[simp]
theorem appendRight_kind (e : WordEvent V) :
    (appendRight (U := U) (V := V) e).kind = e.kind := by
  simp [WordEvent.kind]

theorem appendLeft_injective :
    Function.Injective (appendLeft (U := U) (V := V)) := by
  intro e f h
  apply WordEvent.localIndex_inj
  · exact congrArg (fun x => x.owner) h
  · exact congrArg (fun x => x.localIndex) h

theorem appendRight_injective :
    Function.Injective (appendRight (U := U) (V := V)) := by
  intro e f h
  apply WordEvent.localIndex_inj
  · exact congrArg (fun x => x.owner) h
  · have howner : e.owner = f.owner :=
      congrArg (fun x => x.owner) h
    have hidx := congrArg (fun x => x.index) h
    change (U e.owner).length + e.index =
      (U f.owner).length + f.index at hidx
    rw [howner] at hidx
    simp at hidx
    exact congrArg Nat.succ hidx

theorem appendLeft_ne_appendRight
    (e : WordEvent U) (f : WordEvent V) :
    appendLeft (U := U) (V := V) e ≠ appendRight (U := U) (V := V) f := by
  intro h
  have howner : e.owner = f.owner := by
    exact congrArg (fun x => x.owner) h
  have hindex : e.index = (U f.owner).length + f.index := by
    exact congrArg (fun x => x.index) h
  have hleft : e.index < (U f.owner).length := by
    simpa [howner] using e.inBounds
  omega

/-- Every event of an appended tuple comes from the left or right component. -/
theorem append_cases
    (e : WordEvent (TraceTuple.append U V)) :
    (∃ eU : WordEvent U, e = appendLeft (U := U) (V := V) eU) ∨
      (∃ eV : WordEvent V, e = appendRight (U := U) (V := V) eV) := by
  cases e with
  | mk owner index inBounds =>
      by_cases hleft : index < (U owner).length
      · left
        refine ⟨⟨owner, index, hleft⟩, ?_⟩
        rfl
      · right
        have htotal : index < (U owner).length + (V owner).length := by
          simpa [TraceTuple.append, List.length_append] using inBounds
        have hright : index - (U owner).length < (V owner).length := by
          omega
        refine ⟨⟨owner, index - (U owner).length, hright⟩, ?_⟩
        apply WordEvent.localIndex_inj
        · rfl
        · simp [appendRight, WordEvent.localIndex]
          omega

end WordEvent

namespace WordMSCData

variable {Lifeline : Type u} {Channel : Type v}
variable {U V : TraceTuple Lifeline (LocalLetter Lifeline Channel)}

/-- Message matching for a pointwise tuple append: messages stay in one side. -/
def AppendMsg (DU : WordMSCData U) (DV : WordMSCData V) :
    WordEvent (TraceTuple.append U V) →
      WordEvent (TraceTuple.append U V) → Prop :=
  fun e f =>
    (∃ s r : WordEvent U,
      e = WordEvent.appendLeft (U := U) (V := V) s ∧
      f = WordEvent.appendLeft (U := U) (V := V) r ∧
      DU.msg s r) ∨
    (∃ s r : WordEvent V,
      e = WordEvent.appendRight (U := U) (V := V) s ∧
      f = WordEvent.appendRight (U := U) (V := V) r ∧
      DV.msg s r)

private theorem appendMsg_left_left_inv
    {DU : WordMSCData U} {DV : WordMSCData V}
    {s r : WordEvent U}
    (hmsg :
      AppendMsg DU DV
        (WordEvent.appendLeft (U := U) (V := V) s)
        (WordEvent.appendLeft (U := U) (V := V) r)) :
    DU.msg s r := by
  rcases hmsg with hleft | hright
  · rcases hleft with ⟨sU, rU, hs, hr, h⟩
    have hsEq : s = sU :=
      WordEvent.appendLeft_injective (U := U) (V := V) hs
    have hrEq : r = rU :=
      WordEvent.appendLeft_injective (U := U) (V := V) hr
    simpa [hsEq, hrEq] using h
  · rcases hright with ⟨sV, _rV, hs, _hr, _h⟩
    exact False.elim
      (WordEvent.appendLeft_ne_appendRight
        (U := U) (V := V) s sV hs)

private theorem appendMsg_right_right_inv
    {DU : WordMSCData U} {DV : WordMSCData V}
    {s r : WordEvent V}
    (hmsg :
      AppendMsg DU DV
        (WordEvent.appendRight (U := U) (V := V) s)
        (WordEvent.appendRight (U := U) (V := V) r)) :
    DV.msg s r := by
  rcases hmsg with hleft | hright
  · rcases hleft with ⟨sU, _rU, hs, _hr, _h⟩
    exact False.elim
      (WordEvent.appendLeft_ne_appendRight
        (U := U) (V := V) sU s hs.symm)
  · rcases hright with ⟨sV, rV, hs, hr, h⟩
    have hsEq : s = sV :=
      WordEvent.appendRight_injective (U := U) (V := V) hs
    have hrEq : r = rV :=
      WordEvent.appendRight_injective (U := U) (V := V) hr
    simpa [hsEq, hrEq] using h

private theorem appendMsg_right_left_false
    {DU : WordMSCData U} {DV : WordMSCData V}
    {s : WordEvent V} {r : WordEvent U}
    (hmsg :
      AppendMsg DU DV
        (WordEvent.appendRight (U := U) (V := V) s)
        (WordEvent.appendLeft (U := U) (V := V) r)) :
    False := by
  rcases hmsg with hleft | hright
  · rcases hleft with ⟨sU, _rU, hs, _hr, _h⟩
    exact
      WordEvent.appendLeft_ne_appendRight
        (U := U) (V := V) sU s hs.symm
  · rcases hright with ⟨_sV, rV, _hs, hr, _h⟩
    exact
      WordEvent.appendLeft_ne_appendRight
        (U := U) (V := V) r rV hr

private theorem appendMsg_left_right_false
    {DU : WordMSCData U} {DV : WordMSCData V}
    {s : WordEvent U} {r : WordEvent V}
    (hmsg :
      AppendMsg DU DV
        (WordEvent.appendLeft (U := U) (V := V) s)
        (WordEvent.appendRight (U := U) (V := V) r)) :
    False := by
  rcases hmsg with hleft | hright
  · rcases hleft with ⟨_sU, rU, _hs, hr, _h⟩
    exact
      WordEvent.appendLeft_ne_appendRight
        (U := U) (V := V) rU r hr.symm
  · rcases hright with ⟨sV, _rV, hs, _hr, _h⟩
    exact
      WordEvent.appendLeft_ne_appendRight
        (U := U) (V := V) s sV hs

private abbrev AppendBase (DU : WordMSCData U) (DV : WordMSCData V) :
    WordEvent (TraceTuple.append U V) →
      WordEvent (TraceTuple.append U V) → Prop :=
  BaseEdge
    (fun e : WordEvent (TraceTuple.append U V) => e.owner)
    (fun e : WordEvent (TraceTuple.append U V) => e.localIndex)
    (AppendMsg DU DV)

private abbrev ComponentBase
    (D : WordMSCData U) :
    WordEvent U → WordEvent U → Prop :=
  BaseEdge
    (fun e : WordEvent U => e.owner)
    (fun e : WordEvent U => e.localIndex)
    D.msg

private theorem append_base_left_left_inv
    {DU : WordMSCData U} {DV : WordMSCData V}
    {s r : WordEvent U}
    (hbase :
      AppendBase DU DV
        (WordEvent.appendLeft (U := U) (V := V) s)
        (WordEvent.appendLeft (U := U) (V := V) r)) :
    ComponentBase DU s r := by
  rcases hbase with hlocal | hmsg
  · exact Or.inl hlocal
  · exact Or.inr (appendMsg_left_left_inv hmsg)

private theorem append_base_right_right_inv
    {DU : WordMSCData U} {DV : WordMSCData V}
    {s r : WordEvent V}
    (hbase :
      AppendBase DU DV
        (WordEvent.appendRight (U := U) (V := V) s)
        (WordEvent.appendRight (U := U) (V := V) r)) :
    ComponentBase DV s r := by
  rcases hbase with hlocal | hmsg
  · rcases hlocal with ⟨howner, hidx⟩
    have howner' : s.owner = r.owner := by
      simpa using howner
    refine Or.inl ⟨howner, ?_⟩
    have hidx' : (U r.owner).length + r.localIndex =
        (U s.owner).length + s.localIndex + 1 := by
      simpa [WordEvent.appendRight_localIndex] using hidx
    rw [howner'] at hidx'
    have hidx'' : (U r.owner).length + r.localIndex =
        (U r.owner).length + (s.localIndex + 1) := by
      simpa [Nat.add_assoc] using hidx'
    exact Nat.add_left_cancel hidx''
  · exact Or.inr (appendMsg_right_right_inv hmsg)

private theorem append_base_right_left_false
    {DU : WordMSCData U} {DV : WordMSCData V}
    {s : WordEvent V} {r : WordEvent U}
    (hbase :
      AppendBase DU DV
        (WordEvent.appendRight (U := U) (V := V) s)
        (WordEvent.appendLeft (U := U) (V := V) r)) :
    False := by
  rcases hbase with hlocal | hmsg
  · rcases hlocal with ⟨howner, hidx⟩
    have howner' : s.owner = r.owner := by
      simpa using howner
    have hleft_le : r.localIndex ≤ (U r.owner).length := by
      have hb := r.inBounds
      dsimp [WordEvent.localIndex]
      omega
    have hright_gt :
        (U r.owner).length <
          (WordEvent.appendRight (U := U) (V := V) s).localIndex := by
      change (U r.owner).length < (U s.owner).length + s.localIndex
      rw [howner']
      simp [WordEvent.appendRight_localIndex]
      have hspos : 0 < s.localIndex := WordEvent.localIndex_pos s
      omega
    have hidx' : r.localIndex =
        (WordEvent.appendRight (U := U) (V := V) s).localIndex + 1 := by
      simpa using hidx
    omega
  · exact appendMsg_right_left_false hmsg

private theorem append_path_left
    {DU : WordMSCData U} {DV : WordMSCData V}
    {s : WordEvent U}
    {e : WordEvent (TraceTuple.append U V)}
    (hpath :
      Relation.TransGen (AppendBase DU DV)
        (WordEvent.appendLeft (U := U) (V := V) s) e) :
    (∃ r : WordEvent U,
      e = WordEvent.appendLeft (U := U) (V := V) r ∧
      Relation.TransGen (ComponentBase DU) s r) ∨
    (∃ r : WordEvent V,
      e = WordEvent.appendRight (U := U) (V := V) r) := by
  induction hpath with
  | single hbase =>
      rename_i b
      rcases WordEvent.append_cases (U := U) (V := V) b with
        ⟨r, he⟩ | ⟨r, he⟩
      · exact Or.inl
          ⟨r, he,
            Relation.TransGen.single
              (append_base_left_left_inv (by simpa [he] using hbase))⟩
      · exact Or.inr ⟨r, he⟩
  | tail hpath hbase ih =>
      rename_i mid last
      rcases ih with ⟨midU, hmid, hpathU⟩ | ⟨midV, hmid⟩
      · rw [hmid] at hbase
        rcases WordEvent.append_cases (U := U) (V := V) last with
          ⟨lastU, hlast⟩ | ⟨lastV, hlast⟩
        · exact Or.inl
            ⟨lastU, hlast,
              Relation.TransGen.tail hpathU
                (append_base_left_left_inv (by simpa [hlast] using hbase))⟩
        · exact Or.inr ⟨lastV, hlast⟩
      · rw [hmid] at hbase
        rcases WordEvent.append_cases (U := U) (V := V) last with
          ⟨lastU, hlast⟩ | ⟨lastV, hlast⟩
        · exact False.elim
            (append_base_right_left_false (by simpa [hlast] using hbase))
        · exact Or.inr ⟨lastV, hlast⟩

private theorem append_path_right
    {DU : WordMSCData U} {DV : WordMSCData V}
    {s : WordEvent V}
    {e : WordEvent (TraceTuple.append U V)}
    (hpath :
      Relation.TransGen (AppendBase DU DV)
        (WordEvent.appendRight (U := U) (V := V) s) e) :
    ∃ r : WordEvent V,
      e = WordEvent.appendRight (U := U) (V := V) r ∧
      Relation.TransGen (ComponentBase DV) s r := by
  induction hpath with
  | single hbase =>
      rename_i b
      rcases WordEvent.append_cases (U := U) (V := V) b with
        ⟨r, he⟩ | ⟨r, he⟩
      · exact False.elim
          (append_base_right_left_false (by simpa [he] using hbase))
      · exact
          ⟨r, he,
            Relation.TransGen.single
              (append_base_right_right_inv (by simpa [he] using hbase))⟩
  | tail hpath hbase ih =>
      rename_i mid last
      rcases ih with ⟨midV, hmid, hpathV⟩
      rw [hmid] at hbase
      rcases WordEvent.append_cases (U := U) (V := V) last with
        ⟨lastU, hlast⟩ | ⟨lastV, hlast⟩
      · exact False.elim
          (append_base_right_left_false (by simpa [hlast] using hbase))
      · exact
          ⟨lastV, hlast,
            Relation.TransGen.tail hpathV
              (append_base_right_right_inv (by simpa [hlast] using hbase))⟩

private theorem append_acyclic
    {DU : WordMSCData U} {DV : WordMSCData V} :
    Acyclic (AppendBase DU DV) := by
  intro e hcycle
  rcases WordEvent.append_cases (U := U) (V := V) e with
    ⟨eU, rfl⟩ | ⟨eV, rfl⟩
  · rcases append_path_left (DU := DU) (DV := DV) hcycle with
      ⟨r, hEq, hpath⟩ | ⟨r, hEq⟩
    · have hr : eU = r :=
        WordEvent.appendLeft_injective (U := U) (V := V) hEq
      subst hr
      exact DU.acyclic eU hpath
    · exact
        WordEvent.appendLeft_ne_appendRight
          (U := U) (V := V) eU r hEq
  · rcases append_path_right (DU := DU) (DV := DV) hcycle with
      ⟨r, hEq, hpath⟩
    have hr : eV = r :=
      WordEvent.appendRight_injective (U := U) (V := V) hEq
    subst hr
    exact DV.acyclic eV hpath

/-- Concatenate two complete word-MSC data structures pointwise. -/
def append (DU : WordMSCData U) (DV : WordMSCData V) :
    WordMSCData (TraceTuple.append U V) where
  msg := AppendMsg DU DV
  msg_send_recv := by
    intro s r hmsg
    rcases hmsg with hleft | hright
    · rcases hleft with ⟨sU, rU, hs, hr, h⟩
      simpa [hs, hr] using DU.msg_send_recv h
    · rcases hright with ⟨sV, rV, hs, hr, h⟩
      simpa [hs, hr] using DV.msg_send_recv h
  msg_letter_match := by
    intro s r hmsg
    rcases hmsg with hleft | hright
    · rcases hleft with ⟨sU, rU, hs, hr, h⟩
      simpa [hs, hr] using DU.msg_letter_match h
    · rcases hright with ⟨sV, rV, hs, hr, h⟩
      simpa [hs, hr] using DV.msg_letter_match h
  msg_lifeline_ne := by
    intro s r hmsg
    rcases hmsg with hleft | hright
    · rcases hleft with ⟨sU, rU, hs, hr, h⟩
      simpa [hs, hr] using DU.msg_lifeline_ne h
    · rcases hright with ⟨sV, rV, hs, hr, h⟩
      simpa [hs, hr] using DV.msg_lifeline_ne h
  msg_send_unique := by
    intro s r₁ r₂ h₁ h₂
    rcases h₁ with h₁L | h₁R
    · rcases h₁L with ⟨s₁, r₁U, hs₁, hr₁, hm₁⟩
      rcases h₂ with h₂L | h₂R
      · rcases h₂L with ⟨s₂, r₂U, hs₂, hr₂, hm₂⟩
        have hsEq : s₁ = s₂ :=
          WordEvent.appendLeft_injective (U := U) (V := V)
            (hs₁.symm.trans hs₂)
        have hrEq : r₁U = r₂U :=
          DU.msg_send_unique hm₁ (by simpa [hsEq] using hm₂)
        calc
          r₁ = WordEvent.appendLeft (U := U) (V := V) r₁U := hr₁
          _ = WordEvent.appendLeft (U := U) (V := V) r₂U := by
            simpa [hrEq]
          _ = r₂ := hr₂.symm
      · rcases h₂R with ⟨s₂, _r₂V, hs₂, _hr₂, _hm₂⟩
        exact False.elim
          (WordEvent.appendLeft_ne_appendRight
            (U := U) (V := V) s₁ s₂ (hs₁.symm.trans hs₂))
    · rcases h₁R with ⟨s₁, r₁V, hs₁, hr₁, hm₁⟩
      rcases h₂ with h₂L | h₂R
      · rcases h₂L with ⟨s₂, _r₂U, hs₂, _hr₂, _hm₂⟩
        exact False.elim
          (WordEvent.appendLeft_ne_appendRight
            (U := U) (V := V) s₂ s₁ (hs₂.symm.trans hs₁))
      · rcases h₂R with ⟨s₂, r₂V, hs₂, hr₂, hm₂⟩
        have hsEq : s₁ = s₂ :=
          WordEvent.appendRight_injective (U := U) (V := V)
            (hs₁.symm.trans hs₂)
        have hrEq : r₁V = r₂V :=
          DV.msg_send_unique hm₁ (by simpa [hsEq] using hm₂)
        calc
          r₁ = WordEvent.appendRight (U := U) (V := V) r₁V := hr₁
          _ = WordEvent.appendRight (U := U) (V := V) r₂V := by
            simpa [hrEq]
          _ = r₂ := hr₂.symm
  msg_recv_unique := by
    intro s₁ s₂ r h₁ h₂
    rcases h₁ with h₁L | h₁R
    · rcases h₁L with ⟨s₁U, r₁, hs₁, hr₁, hm₁⟩
      rcases h₂ with h₂L | h₂R
      · rcases h₂L with ⟨s₂U, r₂, hs₂, hr₂, hm₂⟩
        have hrEq : r₁ = r₂ :=
          WordEvent.appendLeft_injective (U := U) (V := V)
            (hr₁.symm.trans hr₂)
        have hsEq : s₁U = s₂U :=
          DU.msg_recv_unique hm₁ (by simpa [hrEq] using hm₂)
        calc
          s₁ = WordEvent.appendLeft (U := U) (V := V) s₁U := hs₁
          _ = WordEvent.appendLeft (U := U) (V := V) s₂U := by
            simpa [hsEq]
          _ = s₂ := hs₂.symm
      · rcases h₂R with ⟨_s₂V, r₂, _hs₂, hr₂, _hm₂⟩
        exact False.elim
          (WordEvent.appendLeft_ne_appendRight
            (U := U) (V := V) r₁ r₂ (hr₁.symm.trans hr₂))
    · rcases h₁R with ⟨s₁V, r₁, hs₁, hr₁, hm₁⟩
      rcases h₂ with h₂L | h₂R
      · rcases h₂L with ⟨_s₂U, r₂, _hs₂, hr₂, _hm₂⟩
        exact False.elim
          (WordEvent.appendLeft_ne_appendRight
            (U := U) (V := V) r₂ r₁ (hr₂.symm.trans hr₁))
      · rcases h₂R with ⟨s₂V, r₂, hs₂, hr₂, hm₂⟩
        have hrEq : r₁ = r₂ :=
          WordEvent.appendRight_injective (U := U) (V := V)
            (hr₁.symm.trans hr₂)
        have hsEq : s₁V = s₂V :=
          DV.msg_recv_unique hm₁ (by simpa [hrEq] using hm₂)
        calc
          s₁ = WordEvent.appendRight (U := U) (V := V) s₁V := hs₁
          _ = WordEvent.appendRight (U := U) (V := V) s₂V := by
            simpa [hsEq]
          _ = s₂ := hs₂.symm
  recv_matched := by
    intro r hrecv
    rcases WordEvent.append_cases (U := U) (V := V) r with
      ⟨rU, hr⟩ | ⟨rV, hr⟩
    · have hrecvU : ∃ A ch, rU.kind = EventKind.recv A ch := by
        simpa [hr] using hrecv
      rcases DU.recv_matched hrecvU with ⟨sU, hmsg⟩
      exact
        ⟨WordEvent.appendLeft (U := U) (V := V) sU,
          Or.inl ⟨sU, rU, rfl, hr, hmsg⟩⟩
    · have hrecvV : ∃ A ch, rV.kind = EventKind.recv A ch := by
        simpa [hr] using hrecv
      rcases DV.recv_matched hrecvV with ⟨sV, hmsg⟩
      exact
        ⟨WordEvent.appendRight (U := U) (V := V) sV,
          Or.inr ⟨sV, rV, rfl, hr, hmsg⟩⟩
  acyclic := append_acyclic
  channel_fifo := by
    intro s₁ s₂ r₁ r₂ A B ch hmsg₁ hmsg₂
      hs₁ hs₂ hr₁ hr₂ hks₁ hks₂ hkr₁ hkr₂ hlt
    rcases hmsg₁ with h₁L | h₁R
    · rcases h₁L with ⟨s₁U, r₁U, hs₁U, hr₁U, hm₁⟩
      rcases hmsg₂ with h₂L | h₂R
      · rcases h₂L with ⟨s₂U, r₂U, hs₂U, hr₂U, hm₂⟩
        have hltU : s₁U.localIndex < s₂U.localIndex := by
          simpa [hs₁U, hs₂U] using hlt
        simpa [hr₁U, hr₂U] using
          DU.channel_fifo hm₁ hm₂
            (by simpa [hs₁U] using hs₁)
            (by simpa [hs₂U] using hs₂)
            (by simpa [hr₁U] using hr₁)
            (by simpa [hr₂U] using hr₂)
            (by simpa [hs₁U] using hks₁)
            (by simpa [hs₂U] using hks₂)
            (by simpa [hr₁U] using hkr₁)
            (by simpa [hr₂U] using hkr₂)
            hltU
      · rcases h₂R with ⟨s₂V, r₂V, hs₂V, hr₂V, _hm₂⟩
        have hrecvOwner : r₁U.owner = r₂V.owner := by
          calc
            r₁U.owner = B := by simpa [hr₁U] using hr₁
            _ = r₂V.owner := by simpa [hr₂V] using hr₂.symm
        have hleft_le : r₁U.localIndex ≤ (U r₁U.owner).length := by
          have hb := r₁U.inBounds
          dsimp [WordEvent.localIndex]
          omega
        have hright_gt :
            (U r₁U.owner).length <
              (WordEvent.appendRight (U := U) (V := V) r₂V).localIndex := by
          change (U r₁U.owner).length <
            (U r₂V.owner).length + r₂V.localIndex
          rw [← hrecvOwner]
          have hpos : 0 < r₂V.localIndex := WordEvent.localIndex_pos r₂V
          omega
        calc
          r₁.localIndex =
              (WordEvent.appendLeft (U := U) (V := V) r₁U).localIndex := by
                rw [hr₁U]
          _ = r₁U.localIndex := rfl
          _ < (WordEvent.appendRight (U := U) (V := V) r₂V).localIndex :=
                Nat.lt_of_le_of_lt hleft_le hright_gt
          _ = r₂.localIndex := by rw [hr₂V]
    · rcases h₁R with ⟨s₁V, r₁V, hs₁V, hr₁V, hm₁⟩
      rcases hmsg₂ with h₂L | h₂R
      · rcases h₂L with ⟨s₂U, _r₂U, hs₂U, _hr₂U, _hm₂⟩
        have hsendOwner : s₁V.owner = s₂U.owner := by
          calc
            s₁V.owner = A := by simpa [hs₁V] using hs₁
            _ = s₂U.owner := by simpa [hs₂U] using hs₂.symm
        have hleft_le : s₂U.localIndex ≤ (U s₂U.owner).length := by
          have hb := s₂U.inBounds
          dsimp [WordEvent.localIndex]
          omega
        have hright_gt :
            (U s₂U.owner).length <
              (WordEvent.appendRight (U := U) (V := V) s₁V).localIndex := by
          change (U s₂U.owner).length <
            (U s₁V.owner).length + s₁V.localIndex
          rw [← hsendOwner]
          have hpos : 0 < s₁V.localIndex := WordEvent.localIndex_pos s₁V
          omega
        have hnot :
            ¬ s₁.localIndex < s₂.localIndex := by
          intro hbad
          rw [hs₁V, hs₂U] at hbad
          have hleAppend :
              (WordEvent.appendLeft (U := U) (V := V) s₂U).localIndex ≤
                (U s₂U.owner).length := by
            simpa using hleft_le
          omega
        exact False.elim (hnot hlt)
      · rcases h₂R with ⟨s₂V, r₂V, hs₂V, hr₂V, hm₂⟩
        have hsendOwner : s₁V.owner = s₂V.owner := by
          calc
            s₁V.owner = A := by simpa [hs₁V] using hs₁
            _ = s₂V.owner := by simpa [hs₂V] using hs₂.symm
        have hrecvOwner : r₁V.owner = r₂V.owner := by
          calc
            r₁V.owner = B := by simpa [hr₁V] using hr₁
            _ = r₂V.owner := by simpa [hr₂V] using hr₂.symm
        have hltV : s₁V.localIndex < s₂V.localIndex := by
          have hlt' := hlt
          rw [hs₁V, hs₂V] at hlt'
          simp [WordEvent.appendRight_localIndex, hsendOwner] at hlt'
          omega
        have hrecvLtV :
            r₁V.localIndex < r₂V.localIndex :=
          DV.channel_fifo hm₁ hm₂
            (by simpa [hs₁V] using hs₁)
            (by simpa [hs₂V] using hs₂)
            (by simpa [hr₁V] using hr₁)
            (by simpa [hr₂V] using hr₂)
            (by simpa [hs₁V] using hks₁)
            (by simpa [hs₂V] using hks₂)
            (by simpa [hr₁V] using hkr₁)
            (by simpa [hr₂V] using hkr₂)
            hltV
        rw [hr₁V, hr₂V]
        simp [WordEvent.appendRight_localIndex, hrecvOwner]
        omega

theorem append_complete (DU : WordMSCData U) (DV : WordMSCData V)
    (hU : DU.IsComplete) (hV : DV.IsComplete) :
    (append DU DV).IsComplete := by
  intro s hsend
  rcases WordEvent.append_cases (U := U) (V := V) s with
    ⟨sU, hs⟩ | ⟨sV, hs⟩
  · have hsendU : (DU.toMSC).IsSend sU := by
      rcases hsend with ⟨B, ch, hkind⟩
      exact ⟨B, ch, by simpa [WordMSCData.toMSC, hs] using hkind⟩
    rcases hU sU hsendU with ⟨rU, hmsg⟩
    exact
      ⟨WordEvent.appendLeft (U := U) (V := V) rU,
        Or.inl ⟨sU, rU, hs, rfl, hmsg⟩⟩
  · have hsendV : (DV.toMSC).IsSend sV := by
      rcases hsend with ⟨B, ch, hkind⟩
      exact ⟨B, ch, by simpa [WordMSCData.toMSC, hs] using hkind⟩
    rcases hV sV hsendV with ⟨rV, hmsg⟩
    exact
      ⟨WordEvent.appendRight (U := U) (V := V) rV,
        Or.inr ⟨sV, rV, hs, rfl, hmsg⟩⟩

theorem wordIsCompleteMSC_append
    (hU : WordIsCompleteMSC U) (hV : WordIsCompleteMSC V) :
    WordIsCompleteMSC (TraceTuple.append U V) := by
  rcases hU with ⟨DU, hDU⟩
  rcases hV with ⟨DV, hDV⟩
  exact ⟨append DU DV, append_complete DU DV hDU hDV⟩

end WordMSCData

def TraceTuple.map {Lifeline : Type u} {α : Type v} {β : Type w}
    (f : α → β) (M : TraceTuple Lifeline α) : TraceTuple Lifeline β :=
  fun A => (M A).map f

theorem TraceTuple.map_append {Lifeline : Type u} {α : Type v} {β : Type w}
    (f : α → β) (U V : TraceTuple Lifeline α) :
    TraceTuple.map f (TraceTuple.append U V) =
      TraceTuple.append (TraceTuple.map f U) (TraceTuple.map f V) := by
  funext A
  simp [TraceTuple.map, TraceTuple.append, List.map_append]

theorem TraceTuple.map_prefix_lift {Lifeline : Type u}
    {α : Type v} {β : Type w}
    (f : α → β) {U : TraceTuple Lifeline β}
    {M : TraceTuple Lifeline α}
    (hprefix : ∀ A, ListPrefix (U A) ((M A).map f)) :
    ∃ W : TraceTuple Lifeline α,
      (∀ A, ListPrefix (W A) (M A)) ∧ TraceTuple.map f W = U := by
  classical
  let W : TraceTuple Lifeline α :=
    fun A => Classical.choose
      (listPrefix_lift_of_prefix_map f (hprefix A))
  refine ⟨W, ?_, ?_⟩
  · intro A
    exact (Classical.choose_spec
      (listPrefix_lift_of_prefix_map f (hprefix A))).1
  · funext A
    exact (Classical.choose_spec
      (listPrefix_lift_of_prefix_map f (hprefix A))).2

theorem LocalLetter.eraseCtrlWord_preserves_prefix
    {Lifeline : Type u} {Channel : Type v}
    {left right : Word (LocalLetter Lifeline Channel)}
    (h : ListPrefix left right) :
    ListPrefix (LocalLetter.eraseCtrlWord left)
      (LocalLetter.eraseCtrlWord right) := by
  rcases h with ⟨suffix, hsuffix⟩
  refine ⟨LocalLetter.eraseCtrlWord suffix, ?_⟩
  rw [← hsuffix, LocalLetter.eraseCtrlWord_append]

/-- Erase compiler-inserted control letters pointwise from a trace tuple. -/
def TraceTuple.eraseCtrl {Lifeline : Type u} {Channel : Type v}
    (M : TraceTuple Lifeline (LocalLetter Lifeline Channel)) :
    TraceTuple Lifeline (LocalLetter Lifeline Channel) :=
  fun A => LocalLetter.eraseCtrlWord (M A)

theorem TraceTuple.eraseCtrl_append {Lifeline : Type u} {Channel : Type v}
    (U V : TraceTuple Lifeline (LocalLetter Lifeline Channel)) :
    TraceTuple.eraseCtrl (TraceTuple.append U V) =
      TraceTuple.append (TraceTuple.eraseCtrl U) (TraceTuple.eraseCtrl V) := by
  funext A
  exact LocalLetter.eraseCtrlWord_append (U A) (V A)

@[simp]
theorem TraceTuple.eraseCtrl_idem {Lifeline : Type u} {Channel : Type v}
    (M : TraceTuple Lifeline (LocalLetter Lifeline Channel)) :
    TraceTuple.eraseCtrl (TraceTuple.eraseCtrl M) = TraceTuple.eraseCtrl M := by
  funext A
  exact LocalLetter.eraseCtrlWord_idem (M A)

/--
The observable semantics obtained by erasing control letters from an internal
trace semantics.
-/
def ErasedSemantics {Lifeline : Type u} {Channel : Type v}
    (Internal :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop)
    (Observed :
      TraceTuple Lifeline (LocalLetter Lifeline Channel)) : Prop :=
  ∃ InternalTrace,
    Internal InternalTrace ∧ TraceTuple.eraseCtrl InternalTrace = Observed

theorem erasedSemantics_congr {Lifeline : Type u} {Channel : Type v}
    {Left Right :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (h : ∀ M, Left M ↔ Right M) :
    ∀ M, ErasedSemantics Left M ↔ ErasedSemantics Right M := by
  intro M
  constructor
  · intro hLeft
    rcases hLeft with ⟨internal, hinternal, herase⟩
    exact ⟨internal, (h internal).1 hinternal, herase⟩
  · intro hRight
    rcases hRight with ⟨internal, hinternal, herase⟩
    exact ⟨internal, (h internal).2 hinternal, herase⟩

/--
Control-erased correctness from equality of the internal semantics.

This is the abstract shape used by the paper's final correctness theorem:
first prove equality before erasing compiler controls, then transport that
equality through the observation map.
-/
theorem parallel_correctness_from_internal_equality
    {Lifeline : Type u} {Channel : Type v}
    {DistributedInternal EnrichedInternal :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (h :
      ∀ M, DistributedInternal M ↔ EnrichedInternal M) :
    ∀ M,
      ErasedSemantics DistributedInternal M ↔
        ErasedSemantics EnrichedInternal M :=
  erasedSemantics_congr h

/--
Abstract MSC-concatenation laws for trace tuples.

This is the self-contained interface used by Zipper-style arguments when the
concrete word-to-MSC construction is not part of the current file.
-/
structure TraceMSCConcat {Lifeline : Type u} {α : Type v}
    (CompleteMSC IsMSC : TraceTuple Lifeline α → Prop) : Prop where
  complete_implies_msc :
    ∀ M, CompleteMSC M → IsMSC M
  append_complete_left :
    ∀ U V, CompleteMSC U → IsMSC V → IsMSC (TraceTuple.append U V)
  append_complete_complete :
    ∀ U V, CompleteMSC U → CompleteMSC V →
      CompleteMSC (TraceTuple.append U V)
  suffix_of_complete_left :
    ∀ U V, CompleteMSC U → IsMSC (TraceTuple.append U V) → IsMSC V

/--
MSC obligations for the parallel Zipper assembly after mapping a proof-only
decorated alphabet back to the ordinary alphabet.

`completed` is the complete-MSC fact for the assembled prefix.  The
`extended_of_original` field is the Zipper-specific replacement lemma: if the
old prefix followed by the continuation is an MSC, then replacing that prefix
by the branchwise completion is also an MSC.
-/
structure MappedParallelZipMSCAssembly {n : Nat}
    {Lifeline : Type u} {α : Type v} {β : Type w}
    (f : α → β)
    (CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) β) → Prop)
    (prefixTrace residualTrace continuationTrace :
      TraceTuple Lifeline (Tagged (Fin n) α)) : Prop where
  completed :
    CompleteMSC
      (TraceTuple.map (Tagged.mapVal f)
        (ParLocTaggedTraceOf.append prefixTrace residualTrace))
  extended_of_original :
    IsMSC
      (TraceTuple.map (Tagged.mapVal f)
        (TraceTuple.append prefixTrace continuationTrace)) →
    IsMSC
      (TraceTuple.map (Tagged.mapVal f)
        (TraceTuple.append
          (ParLocTaggedTraceOf.append prefixTrace residualTrace)
          continuationTrace))

theorem MappedParallelZipMSCAssembly.of_completed_empty {n : Nat}
    {Lifeline : Type u} {α : Type v} {β : Type w}
    {f : α → β}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) β) → Prop}
    {prefixTrace residualTrace :
      TraceTuple Lifeline (Tagged (Fin n) α)}
    (hcomplete_implies_msc :
      ∀ M, CompleteMSC M → IsMSC M)
    (hcompleted :
      CompleteMSC
        (TraceTuple.map (Tagged.mapVal f)
          (ParLocTaggedTraceOf.append prefixTrace residualTrace))) :
    MappedParallelZipMSCAssembly f CompleteMSC IsMSC
      prefixTrace residualTrace
      (TraceTuple.empty Lifeline (Tagged (Fin n) α)) where
  completed := hcompleted
  extended_of_original := by
    intro _horiginal
    simpa [TraceTuple.append_empty] using
      hcomplete_implies_msc _ hcompleted

/--
The four-condition Zipper postcondition.

The predicates `CompleteLocal`, `CompleteMSC`, and `IsMSC` are parameters here:
this file records the postcondition shape and the generic consequence used by
the parallel case, while the full local and MSC semantics remain separate.
-/
structure ZipPost {Lifeline : Type u} {α : Type v}
    (CompleteLocal : Lifeline → Word α → Prop)
    (CompleteMSC IsMSC : TraceTuple Lifeline α → Prop)
    (U Ubar V : TraceTuple Lifeline α) : Prop where
  prefix_complete :
    ∀ A,
      ListPrefix (U A) (Ubar A) ∧
      CompleteLocal A (Ubar A) ∧
      (V A ≠ [] → Ubar A = U A)
  completed_msc : CompleteMSC Ubar
  original_msc : IsMSC (TraceTuple.append U V)
  extended_msc : IsMSC (TraceTuple.append Ubar V)

namespace ZipPost

theorem of_residuals {Lifeline : Type u} {α : Type v}
    {CompleteLocal : Lifeline → Word α → Prop}
    {CompleteMSC IsMSC : TraceTuple Lifeline α → Prop}
    {U Ubar V residual : TraceTuple Lifeline α}
    (hresidual : ∀ A, Ubar A = U A ++ residual A)
    (hcomplete : ∀ A, CompleteLocal A (Ubar A))
    (hstable : ∀ A, V A ≠ [] → residual A = [])
    (hcompleted : CompleteMSC Ubar)
    (horiginal : IsMSC (TraceTuple.append U V))
    (hextended : IsMSC (TraceTuple.append Ubar V)) :
    ZipPost CompleteLocal CompleteMSC IsMSC U Ubar V where
  prefix_complete := by
    intro A
    refine ⟨?_, hcomplete A, ?_⟩
    · exact ⟨residual A, (hresidual A).symm⟩
    · intro hcont
      rw [hresidual A, hstable A hcont, List.append_nil]
  completed_msc := hcompleted
  original_msc := horiginal
  extended_msc := hextended

theorem continuation_stable {Lifeline : Type u} {α : Type v}
    {CompleteLocal : Lifeline → Word α → Prop}
    {CompleteMSC IsMSC : TraceTuple Lifeline α → Prop}
    {U Ubar V : TraceTuple Lifeline α}
    (hzip : ZipPost CompleteLocal CompleteMSC IsMSC U Ubar V)
    {A : Lifeline} (hcontinuation : V A ≠ []) :
    Ubar A = U A := by
  exact (hzip.prefix_complete A).2.2 hcontinuation

theorem completed_prefix_msc {Lifeline : Type u} {α : Type v}
    {CompleteLocal : Lifeline → Word α → Prop}
    {CompleteMSC IsMSC : TraceTuple Lifeline α → Prop}
    {U Ubar V : TraceTuple Lifeline α}
    (hconcat : TraceMSCConcat CompleteMSC IsMSC)
    (hzip : ZipPost CompleteLocal CompleteMSC IsMSC U Ubar V) :
    IsMSC Ubar :=
  hconcat.complete_implies_msc Ubar hzip.completed_msc

theorem suffix_msc {Lifeline : Type u} {α : Type v}
    {CompleteLocal : Lifeline → Word α → Prop}
    {CompleteMSC IsMSC : TraceTuple Lifeline α → Prop}
    {U Ubar V : TraceTuple Lifeline α}
    (hconcat : TraceMSCConcat CompleteMSC IsMSC)
    (hzip : ZipPost CompleteLocal CompleteMSC IsMSC U Ubar V) :
    IsMSC V :=
  hconcat.suffix_of_complete_left Ubar V hzip.completed_msc
    hzip.extended_msc

theorem completion_preservation {Lifeline : Type u} {α : Type v}
    {CompleteLocal : Lifeline → Word α → Prop}
    {CompleteMSC IsMSC : TraceTuple Lifeline α → Prop}
    {U Ubar V : TraceTuple Lifeline α}
    (hprefixFree : LocalPrefixFree CompleteLocal)
    (hzip : ZipPost CompleteLocal CompleteMSC IsMSC U Ubar V)
    {A : Lifeline} (hcomplete : CompleteLocal A (U A)) :
    Ubar A = U A := by
  rcases hzip.prefix_complete A with ⟨hprefix, hbarComplete, _hcontinuation⟩
  exact (hprefixFree A hcomplete hbarComplete hprefix).symm

theorem residual_empty_of_complete_prefix {Lifeline : Type u} {α : Type v}
    {CompleteLocal : Lifeline → Word α → Prop}
    {CompleteMSC IsMSC : TraceTuple Lifeline α → Prop}
    {U Ubar V : TraceTuple Lifeline α}
    (hprefixFree : LocalPrefixFree CompleteLocal)
    (hzip : ZipPost CompleteLocal CompleteMSC IsMSC U Ubar V)
    {A : Lifeline} {residual : Word α}
    (hcomplete : CompleteLocal A (U A))
    (hresidual : Ubar A = U A ++ residual) :
    residual = [] := by
  have hstable := hzip.completion_preservation hprefixFree hcomplete
  exact list_append_eq_self_right_nil (hstable.symm.trans hresidual)

noncomputable def residual {Lifeline : Type u} {α : Type v}
    {CompleteLocal : Lifeline → Word α → Prop}
    {CompleteMSC IsMSC : TraceTuple Lifeline α → Prop}
    {U Ubar V : TraceTuple Lifeline α}
    (hzip : ZipPost CompleteLocal CompleteMSC IsMSC U Ubar V) :
    TraceTuple Lifeline α :=
  fun A => Classical.choose (hzip.prefix_complete A).1

theorem residual_eq {Lifeline : Type u} {α : Type v}
    {CompleteLocal : Lifeline → Word α → Prop}
    {CompleteMSC IsMSC : TraceTuple Lifeline α → Prop}
    {U Ubar V : TraceTuple Lifeline α}
    (hzip : ZipPost CompleteLocal CompleteMSC IsMSC U Ubar V)
    (A : Lifeline) :
    Ubar A = U A ++ hzip.residual A :=
  (Classical.choose_spec (hzip.prefix_complete A).1).symm

theorem map {Lifeline : Type u} {α : Type v} {β : Type w}
    {CompleteLocalα : Lifeline → Word α → Prop}
    {CompleteMSCα IsMSCα : TraceTuple Lifeline α → Prop}
    {CompleteLocalβ : Lifeline → Word β → Prop}
    {CompleteMSCβ IsMSCβ : TraceTuple Lifeline β → Prop}
    {U Ubar V : TraceTuple Lifeline α}
    (f : α → β)
    (hlocal :
      ∀ A word, CompleteLocalα A word →
        CompleteLocalβ A (word.map f))
    (hcompleted :
      CompleteMSCα Ubar → CompleteMSCβ (TraceTuple.map f Ubar))
    (horiginal :
      IsMSCα (TraceTuple.append U V) →
        IsMSCβ (TraceTuple.map f (TraceTuple.append U V)))
    (hextended :
      IsMSCα (TraceTuple.append Ubar V) →
        IsMSCβ (TraceTuple.map f (TraceTuple.append Ubar V)))
    (hzip : ZipPost CompleteLocalα CompleteMSCα IsMSCα U Ubar V) :
    ZipPost CompleteLocalβ CompleteMSCβ IsMSCβ
      (TraceTuple.map f U) (TraceTuple.map f Ubar) (TraceTuple.map f V) := by
  refine
    { prefix_complete := ?_
      completed_msc := hcompleted hzip.completed_msc
      original_msc := ?_
      extended_msc := ?_ }
  · intro A
    rcases hzip.prefix_complete A with
      ⟨hprefix, hbarComplete, hstable⟩
    refine ⟨list_map_preserves_prefix f hprefix,
      hlocal A (Ubar A) hbarComplete, ?_⟩
    intro hcont
    have hcontOriginal : V A ≠ [] := by
      intro hv
      apply hcont
      simp [TraceTuple.map, hv]
    exact congrArg (List.map f) (hstable hcontOriginal)
  · simpa [TraceTuple.map_append] using horiginal hzip.original_msc
  · simpa [TraceTuple.map_append] using hextended hzip.extended_msc

theorem parlocOf_residual_empty_of_complete_prefix {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) α) → Prop}
    {U Ubar V : TraceTuple Lifeline (Tagged (Fin n) α)}
    (hprefixFree : ComponentPrefixFreeOf complete)
    (hzip :
      ZipPost (TaggedParLocCompleteLocalOf complete) CompleteMSC IsMSC
        U Ubar V)
    {A : Lifeline}
    {residual : Word (Tagged (Fin n) α)}
    (hcomplete : TaggedParLocCompleteLocalOf complete A (U A))
    (hresidual : Ubar A = U A ++ residual) :
    residual = [] :=
  hzip.residual_empty_of_complete_prefix
    (TaggedParLocCompleteLocalOf.prefix_free hprefixFree)
    hcomplete hresidual

theorem parloc_residual_empty_of_complete_prefix {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete :
      Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {U Ubar V :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (hprefixFree : ComponentPrefixFree complete)
    (hzip :
      ZipPost (TaggedParLocCompleteLocal complete) CompleteMSC IsMSC U Ubar V)
    {A : Lifeline}
    {residual : Word (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (hcomplete : TaggedParLocCompleteLocal complete A (U A))
    (hresidual : Ubar A = U A ++ residual) :
    residual = [] :=
    hzip.residual_empty_of_complete_prefix
    (TaggedParLocCompleteLocal.prefix_free hprefixFree)
    hcomplete hresidual

theorem parloc_residual_empty_of_noPar_components {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {U Ubar V :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hnopar : ∀ i A, LocProg.NoPar (program i A))
    (hzip :
      ZipPost
        (TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word))
        CompleteMSC IsMSC U Ubar V)
    {A : Lifeline}
    {residual : Word (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (hcomplete :
      TaggedParLocCompleteLocal
        (fun i A word => LocProg.trace (program i A) word) A (U A))
    (hresidual : Ubar A = U A ++ residual) :
    residual = [] :=
  hzip.residual_empty_of_complete_prefix
    (LocProg.taggedParLocCompleteLocal_prefix_free_noPar hguard hnopar)
    hcomplete hresidual

theorem stable_trace_residual_of_branch_residuals_of {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {residual : ParLocBranchesOf n Lifeline α}
    {residualTrace : ParLocTaggedTraceOf n Lifeline α}
    {continuation : ParLocTaggedTraceOf n Lifeline α}
    (hresidualPar : ParLocOf residual residualTrace)
    (hbranchEmpty : ∀ A, continuation A ≠ [] → ∀ i, residual i A = []) :
    ∀ A, continuation A ≠ [] → residualTrace A = [] := by
  intro A hcont
  exact parlocOf_residual_empty_of_components_empty hresidualPar
    (hbranchEmpty A hcont)

theorem stable_trace_residual_of_branch_residuals {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {residual : ParLocBranches n Lifeline Channel}
    {residualTrace : ParLocTaggedTrace n Lifeline Channel}
    {continuation : ParLocTaggedTrace n Lifeline Channel}
    (hresidualPar : ParLoc residual residualTrace)
    (hbranchEmpty : ∀ A, continuation A ≠ [] → ∀ i, residual i A = []) :
    ∀ A, continuation A ≠ [] → residualTrace A = [] := by
  intro A hcont
  exact parloc_residual_empty_of_components_empty hresidualPar
    (hbranchEmpty A hcont)

theorem branch_residuals_empty_of_complete_prefixes_of {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    {CompleteMSC IsMSC :
      Fin n → TraceTuple Lifeline α → Prop}
    {branchPrefix completed residual continuation :
      ParLocBranchesOf n Lifeline α}
    {topContinuation : ParLocTaggedTraceOf n Lifeline α}
    (hprefixFree : ComponentPrefixFreeOf complete)
    (hzip :
      ∀ i, ZipPost (complete i) (CompleteMSC i) (IsMSC i)
        (branchPrefix i) (completed i) (continuation i))
    (hcompletePrefix :
      ∀ A, topContinuation A ≠ [] → ∀ i, complete i A (branchPrefix i A))
    (hresidual : ∀ i A, completed i A = branchPrefix i A ++ residual i A) :
    ∀ A, topContinuation A ≠ [] → ∀ i, residual i A = [] := by
  intro A hcont i
  exact (hzip i).residual_empty_of_complete_prefix
    (by
      intro A xs ys hxs hys hp
      exact hprefixFree i A hxs hys hp)
    (hcompletePrefix A hcont i)
    (hresidual i A)

theorem branch_residuals_empty_of_complete_prefixes {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete :
      Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    {CompleteMSC IsMSC :
      Fin n → TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {branchPrefix completed residual continuation :
      ParLocBranches n Lifeline Channel}
    {topContinuation : ParLocTaggedTrace n Lifeline Channel}
    (hprefixFree : ComponentPrefixFree complete)
    (hzip :
      ∀ i, ZipPost (complete i) (CompleteMSC i) (IsMSC i)
        (branchPrefix i) (completed i) (continuation i))
    (hcompletePrefix :
      ∀ A, topContinuation A ≠ [] → ∀ i, complete i A (branchPrefix i A))
    (hresidual : ∀ i A, completed i A = branchPrefix i A ++ residual i A) :
    ∀ A, topContinuation A ≠ [] → ∀ i, residual i A = [] := by
  intro A hcont i
  exact (hzip i).residual_empty_of_complete_prefix
    (by
      intro A xs ys hxs hys hp
      exact hprefixFree i A hxs hys hp)
    (hcompletePrefix A hcont i)
    (hresidual i A)

theorem stable_trace_residual_of_branch_zipPosts {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete :
      Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    {CompleteMSC IsMSC :
      Fin n → TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {branchPrefix completed residual continuation :
      ParLocBranches n Lifeline Channel}
    {residualTrace topContinuation : ParLocTaggedTrace n Lifeline Channel}
    (hresidualPar : ParLoc residual residualTrace)
    (hprefixFree : ComponentPrefixFree complete)
    (hzip :
      ∀ i, ZipPost (complete i) (CompleteMSC i) (IsMSC i)
        (branchPrefix i) (completed i) (continuation i))
    (hcompletePrefix :
      ∀ A, topContinuation A ≠ [] → ∀ i, complete i A (branchPrefix i A))
    (hresidual : ∀ i A, completed i A = branchPrefix i A ++ residual i A) :
    ∀ A, topContinuation A ≠ [] → residualTrace A = [] :=
  stable_trace_residual_of_branch_residuals
    (continuation := topContinuation) hresidualPar
    (branch_residuals_empty_of_complete_prefixes hprefixFree hzip
      hcompletePrefix hresidual)

theorem stable_trace_residual_of_branch_zipPosts_of {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    {CompleteMSC IsMSC :
      Fin n → TraceTuple Lifeline α → Prop}
    {branchPrefix completed residual continuation :
      ParLocBranchesOf n Lifeline α}
    {residualTrace topContinuation : ParLocTaggedTraceOf n Lifeline α}
    (hresidualPar : ParLocOf residual residualTrace)
    (hprefixFree : ComponentPrefixFreeOf complete)
    (hzip :
      ∀ i, ZipPost (complete i) (CompleteMSC i) (IsMSC i)
        (branchPrefix i) (completed i) (continuation i))
    (hcompletePrefix :
      ∀ A, topContinuation A ≠ [] → ∀ i, complete i A (branchPrefix i A))
    (hresidual : ∀ i A, completed i A = branchPrefix i A ++ residual i A) :
    ∀ A, topContinuation A ≠ [] → residualTrace A = [] :=
  stable_trace_residual_of_branch_residuals_of
    (continuation := topContinuation) hresidualPar
    (branch_residuals_empty_of_complete_prefixes_of hprefixFree hzip
      hcompletePrefix hresidual)

end ZipPost

/-- Prefix closure of decorated local traces for one local program. -/
def DTracePrefix {Lifeline : Type u} {Channel : Type v}
    (program : LocProg Lifeline Channel)
    (word : Word (LocProg.DecoratedLetter Lifeline Channel)) : Prop :=
  ∃ complete,
    LocProg.DTrace program complete ∧ ListPrefix word complete

/-- Pointwise decorated local-trace prefix property for a local program tuple. -/
def DTracePrefixTuple {Lifeline : Type u} {Channel : Type v}
    (program : Lifeline → LocProg Lifeline Channel)
    (U : TraceTuple Lifeline
      (LocProg.DecoratedLetter Lifeline Channel)) : Prop :=
  ∀ A, DTracePrefix (program A) (U A)

theorem DTracePrefix.of_prefixTrace {Lifeline : Type u} {Channel : Type v}
    {program : LocProg Lifeline Channel}
    {word : Word (LocalLetter Lifeline Channel)}
    (hprefix : LocProg.prefixTrace program word) :
    ∃ decorated : Word (LocProg.DecoratedLetter Lifeline Channel),
      DTracePrefix program decorated ∧
      decorated.map LocProg.DecoratedLetter.toLocal = word := by
  rcases hprefix with ⟨complete, hcomplete, hwordPrefix⟩
  rcases LocProg.trace_to_dtrace hcomplete with
    ⟨completeDecorated, hcompleteDecorated, hcompleteMap⟩
  have hwordPrefixDecorated :
      ListPrefix word
        (completeDecorated.map LocProg.DecoratedLetter.toLocal) := by
    simpa [hcompleteMap] using hwordPrefix
  rcases listPrefix_lift_of_prefix_map
      LocProg.DecoratedLetter.toLocal hwordPrefixDecorated with
    ⟨prefixDecorated, hprefixDecorated, hprefixMap⟩
  exact ⟨prefixDecorated,
    ⟨completeDecorated, hcompleteDecorated, hprefixDecorated⟩,
    hprefixMap⟩

theorem DTracePrefixTuple.of_distPrefixTrace
    {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hprefix : DistProg.prefixTrace program U) :
    ∃ Udecor : TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel),
      DTracePrefixTuple program Udecor ∧
      TraceTuple.map LocProg.DecoratedLetter.toLocal Udecor = U := by
  classical
  let Udecor : TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun A => Classical.choose
      (DTracePrefix.of_prefixTrace (hprefix A))
  refine ⟨Udecor, ?_, ?_⟩
  · intro A
    exact (Classical.choose_spec
      (DTracePrefix.of_prefixTrace (hprefix A))).1
  · funext A
    exact (Classical.choose_spec
      (DTracePrefix.of_prefixTrace (hprefix A))).2

theorem DTracePrefix.to_prefixTrace {Lifeline : Type u} {Channel : Type v}
    {program : LocProg Lifeline Channel}
    {word : Word (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : DTracePrefix program word) :
    LocProg.prefixTrace program
      (word.map LocProg.DecoratedLetter.toLocal) := by
  rcases hprefix with ⟨complete, hcomplete, hwordPrefix⟩
  refine ⟨complete.map LocProg.DecoratedLetter.toLocal, ?_, ?_⟩
  · exact LocProg.dtrace_to_trace hcomplete
  · exact list_map_preserves_prefix
      LocProg.DecoratedLetter.toLocal hwordPrefix

theorem DTracePrefixTuple.to_distPrefixTrace
    {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {U : TraceTuple Lifeline
      (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : DTracePrefixTuple program U) :
    DistProg.prefixTrace program
      (TraceTuple.map LocProg.DecoratedLetter.toLocal U) := by
  intro A
  exact DTracePrefix.to_prefixTrace (hprefix A)

theorem DTracePrefixTuple.to_distPrefixSemantics
    {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {IsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {U : TraceTuple Lifeline
      (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : DTracePrefixTuple program U)
    (hmsc :
      IsMSC (TraceTuple.map LocProg.DecoratedLetter.toLocal U)) :
    DistProg.prefixSemantics IsMSC program
      (TraceTuple.map LocProg.DecoratedLetter.toLocal U) :=
  ⟨DTracePrefixTuple.to_distPrefixTrace hprefix, hmsc⟩

theorem DTracePrefix.exists_iff_prefixTrace
    {Lifeline : Type u} {Channel : Type v}
    {program : LocProg Lifeline Channel}
    {word : Word (LocalLetter Lifeline Channel)} :
    (∃ decorated : Word (LocProg.DecoratedLetter Lifeline Channel),
      DTracePrefix program decorated ∧
      decorated.map LocProg.DecoratedLetter.toLocal = word) ↔
    LocProg.prefixTrace program word := by
  constructor
  · rintro ⟨decorated, hprefix, herase⟩
    rw [← herase]
    exact DTracePrefix.to_prefixTrace hprefix
  · exact DTracePrefix.of_prefixTrace

theorem DTracePrefixTuple.exists_iff_distPrefixTrace
    {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)} :
    (∃ Udecor : TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel),
      DTracePrefixTuple program Udecor ∧
      TraceTuple.map LocProg.DecoratedLetter.toLocal Udecor = U) ↔
    DistProg.prefixTrace program U := by
  constructor
  · rintro ⟨Udecor, hprefix, herase⟩
    rw [← herase]
    exact DTracePrefixTuple.to_distPrefixTrace hprefix
  · exact DTracePrefixTuple.of_distPrefixTrace

theorem DTraceTuple.to_distTrace
    {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {U : TraceTuple Lifeline
      (LocProg.DecoratedLetter Lifeline Channel)}
    (hcomplete :
      ∀ A, LocProg.DTrace (program A) (U A)) :
    DistProg.trace program
      (TraceTuple.map LocProg.DecoratedLetter.toLocal U) := by
  intro A
  exact LocProg.dtrace_to_trace (hcomplete A)

theorem DTraceTuple.to_distSemantics
    {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {CompleteMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {U : TraceTuple Lifeline
      (LocProg.DecoratedLetter Lifeline Channel)}
    (hcomplete :
      ∀ A, LocProg.DTrace (program A) (U A))
    (hmsc :
      CompleteMSC (TraceTuple.map LocProg.DecoratedLetter.toLocal U)) :
    DistProg.semantics CompleteMSC program
      (TraceTuple.map LocProg.DecoratedLetter.toLocal U) :=
  ⟨DTraceTuple.to_distTrace hcomplete, hmsc⟩

/--
A decorated branch prefix of a tagged `parloc` trace is an ordinary prefix
trace of the corresponding local `par` program after decorations and branch
tags are forgotten.
-/
theorem parallel_dtrace_prefix_to_distPrefixTrace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {prefixBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hbranchPrefix :
      ∀ i, DTracePrefixTuple (fun A => program i A) (prefixBranches i)) :
    DistProg.prefixTrace
      (fun A => LocProg.par (fun i => program i A))
      (TraceTuple.map Tagged.val
        (TraceTuple.map
          (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          prefixTrace)) := by
  classical
  intro A
  let completeBranches :
      Fin n → Word (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i => Classical.choose (hbranchPrefix i A)
  have hcomplete :
      ∀ i, LocProg.DTrace (program i A) (completeBranches i) := by
    intro i
    exact (Classical.choose_spec (hbranchPrefix i A)).1
  have hbranchListPrefix :
      ∀ i, ListPrefix (prefixBranches i A) (completeBranches i) := by
    intro i
    exact (Classical.choose_spec (hbranchPrefix i A)).2
  let residualBranches :
      Fin n → Word (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i => Classical.choose (hbranchListPrefix i)
  have hcompleteEq :
      ∀ i, prefixBranches i A ++ residualBranches i =
        completeBranches i := by
    intro i
    exact Classical.choose_spec (hbranchListPrefix i)
  let completeTagged :
      Word (Tagged (Fin n)
        (LocProg.DecoratedLetter Lifeline Channel)) :=
    prefixTrace A ++ serialTaggedWordOf residualBranches
  let completeTaggedLocal :
      Word (Tagged (Fin n) (LocalLetter Lifeline Channel)) :=
    completeTagged.map
      (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
  let completeLocal : Word (LocalLetter Lifeline Channel) :=
    completeTaggedLocal.map Tagged.val
  refine ⟨completeLocal, ?_, ?_⟩
  · have herase :
        ∀ i,
          eraseBranch i completeTagged = completeBranches i := by
      intro i
      calc
        eraseBranch i completeTagged
            = prefixBranches i A ++ residualBranches i := by
              simp [completeTagged, eraseBranch_append, hprefix i A,
                eraseBranch_serialTaggedWordOf]
        _ = completeBranches i := hcompleteEq i
    have heraseLocal :
        ∀ i,
          eraseBranch i completeTaggedLocal =
            (completeBranches i).map LocProg.DecoratedLetter.toLocal := by
      intro i
      calc
        eraseBranch i completeTaggedLocal
            = (eraseBranch i completeTagged).map
                LocProg.DecoratedLetter.toLocal := by
              exact eraseBranch_mapVal i
                LocProg.DecoratedLetter.toLocal completeTagged
        _ = (completeBranches i).map
                LocProg.DecoratedLetter.toLocal := by
              rw [herase i]
    exact LocProg.trace_par_intro
      (tagged := completeTaggedLocal)
      (branchWords :=
        fun i => (completeBranches i).map
          LocProg.DecoratedLetter.toLocal)
      rfl heraseLocal
      (fun i => LocProg.dtrace_to_trace (hcomplete i))
  · have hpTagged :
        ListPrefix (prefixTrace A) completeTagged :=
      ⟨serialTaggedWordOf residualBranches, rfl⟩
    have hpTaggedLocal :
        ListPrefix
          ((prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal))
          completeTaggedLocal :=
      list_map_preserves_prefix
        (Tagged.mapVal LocProg.DecoratedLetter.toLocal) hpTagged
    have hpLocal :
        ListPrefix
          (((prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)).map Tagged.val)
          completeLocal :=
      list_map_preserves_prefix Tagged.val hpTaggedLocal
    simpa [TraceTuple.map, completeLocal, completeTaggedLocal] using hpLocal

/--
Concrete-prefix decomposition interface for the parallel Zipper endgame.

It records that an ordinary local `par` prefix comes from a decorated tagged
`parloc` prefix, and that the branch restrictions satisfy the recursive
decorated prefix and MSC predicates.
-/
structure ParallelDTracePrefixPreimage {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    (program : Fin n → Lifeline → LocProg Lifeline Channel)
    (BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop)
    (LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop)
    (U : TraceTuple Lifeline (LocalLetter Lifeline Channel)) :
    Type (max u v) where
  prefixBranches :
    ParLocBranchesOf n Lifeline
      (LocProg.DecoratedLetter Lifeline Channel)
  prefixTrace :
    ParLocTaggedTraceOf n Lifeline
      (LocProg.DecoratedLetter Lifeline Channel)
  parloc : ParLocOf prefixBranches prefixTrace
  branch_prefix :
    ∀ i, DTracePrefixTuple (fun A => program i A)
      (prefixBranches i)
  branch_msc : ∀ i, BranchIsMSC i (prefixBranches i)
  local_msc :
    LocalIsMSC
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        prefixTrace)
  erases :
    U =
      TraceTuple.map Tagged.val
        (TraceTuple.map
          (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          prefixTrace)

theorem ParallelDTracePrefixPreimage.to_distPrefixTrace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hpre :
      ParallelDTracePrefixPreimage program BranchIsMSC LocalIsMSC U) :
    DistProg.prefixTrace
      (fun A => LocProg.par (fun i => program i A)) U := by
  rw [hpre.erases]
  exact parallel_dtrace_prefix_to_distPrefixTrace
    (program := program)
    hpre.parloc hpre.branch_prefix

theorem ParallelDTracePrefixPreimage.to_distPrefixSemantics {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hmscMap :
      ∀ M, LocalIsMSC M → OrdIsMSC (TraceTuple.map Tagged.val M))
    (hpre :
      ParallelDTracePrefixPreimage program BranchIsMSC LocalIsMSC U) :
    DistProg.prefixSemantics OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) U := by
  refine ⟨hpre.to_distPrefixTrace, ?_⟩
  rw [hpre.erases]
  exact hmscMap _ hpre.local_msc

noncomputable def ParallelDTracePrefixPreimage.of_complete_prefix {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {fullBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {fullTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hfullPar : ParLocOf fullBranches fullTrace)
    (hfullComplete :
      ∀ i A, LocProg.DTrace (program i A) (fullBranches i A))
    (hpref :
      ∀ A,
        ListPrefix (U A)
          (TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              fullTrace) A))
    (hprefixMSC :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) ∧
        ∀ i, BranchIsMSC i (prefixBranches i)) :
    ParallelDTracePrefixPreimage program BranchIsMSC LocalIsMSC U := by
  classical
  let eraseDecor :
      Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel) →
        LocalLetter Lifeline Channel :=
    fun x => LocProg.DecoratedLetter.toLocal x.val
  have hprefErase :
      ∀ A, ListPrefix (U A) ((fullTrace A).map eraseDecor) := by
    intro A
    simpa [TraceTuple.map, eraseDecor, List.map_map, Function.comp_def]
      using hpref A
  let prefixTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    Classical.choose (TraceTuple.map_prefix_lift eraseDecor hprefErase)
  have hprefixTrace :
      ∀ A, ListPrefix (prefixTrace A) (fullTrace A) :=
    (Classical.choose_spec
      (TraceTuple.map_prefix_lift eraseDecor hprefErase)).1
  have herasePrefix : TraceTuple.map eraseDecor prefixTrace = U :=
    (Classical.choose_spec
      (TraceTuple.map_prefix_lift eraseDecor hprefErase)).2
  let prefixBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    Classical.choose (parlocOf_prefix_decompose hfullPar hprefixTrace)
  have hprefixPar : ParLocOf prefixBranches prefixTrace :=
    (Classical.choose_spec
      (parlocOf_prefix_decompose hfullPar hprefixTrace)).1
  have hbranchPrefix :
      ∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A) :=
    (Classical.choose_spec
      (parlocOf_prefix_decompose hfullPar hprefixTrace)).2
  refine
    { prefixBranches := prefixBranches
      prefixTrace := prefixTrace
      parloc := hprefixPar
      branch_prefix := ?_
      branch_msc := (hprefixMSC hprefixPar hbranchPrefix).2
      local_msc := (hprefixMSC hprefixPar hbranchPrefix).1
      erases := ?_ }
  · intro i A
    exact ⟨fullBranches i A, hfullComplete i A, hbranchPrefix i A⟩
  · calc
      U = TraceTuple.map eraseDecor prefixTrace := herasePrefix.symm
      _ =
          TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              prefixTrace) := by
            funext A
            simp [TraceTuple.map, eraseDecor, List.map_map,
              Function.comp_def]

noncomputable def ParallelDTracePrefixPreimage.of_distPrefixTrace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hprefix :
      DistProg.prefixTrace
        (fun A => LocProg.par (fun i => program i A)) U)
    (hprefixMSC :
      ∀ {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) ∧
        ∀ i, BranchIsMSC i (prefixBranches i)) :
    ParallelDTracePrefixPreimage program BranchIsMSC LocalIsMSC U := by
  classical
  let fullWord : TraceTuple Lifeline (LocalLetter Lifeline Channel) :=
    fun A => Classical.choose (hprefix A)
  have hfullTraceOrd :
      ∀ A, LocProg.trace
        (LocProg.par (fun i => program i A)) (fullWord A) := by
    intro A
    exact (Classical.choose_spec (hprefix A)).1
  have hprefFull : ∀ A, ListPrefix (U A) (fullWord A) := by
    intro A
    exact (Classical.choose_spec (hprefix A)).2
  let fullDecorated :
      TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun A => Classical.choose (LocProg.trace_to_dtrace (hfullTraceOrd A))
  have hfullDecoratedTrace :
      ∀ A, LocProg.DTrace
        (LocProg.par (fun i => program i A)) (fullDecorated A) := by
    intro A
    exact (Classical.choose_spec
      (LocProg.trace_to_dtrace (hfullTraceOrd A))).1
  have hfullDecoratedMap :
      ∀ A, (fullDecorated A).map LocProg.DecoratedLetter.toLocal =
        fullWord A := by
    intro A
    exact (Classical.choose_spec
      (LocProg.trace_to_dtrace (hfullTraceOrd A))).2
  let fullTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun A => Classical.choose
      (LocProg.dtrace_par_elim (hfullDecoratedTrace A))
  have hfullDecoratedEq :
      ∀ A, fullDecorated A =
        (fullTrace A).map LocProg.wrapTaggedDecorated := by
    intro A
    exact (Classical.choose_spec
      (LocProg.dtrace_par_elim (hfullDecoratedTrace A))).1
  have hfullBranchTrace :
      ∀ i A, LocProg.DTrace (program i A) (eraseBranch i (fullTrace A)) := by
    intro i A
    exact (Classical.choose_spec
      (LocProg.dtrace_par_elim (hfullDecoratedTrace A))).2 i
  let fullBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i A => eraseBranch i (fullTrace A)
  have hfullPar : ParLocOf fullBranches fullTrace := by
    intro i A
    rfl
  have hfullComplete :
      ∀ i A, LocProg.DTrace (program i A) (fullBranches i A) := by
    intro i A
    exact hfullBranchTrace i A
  have hprefMapped :
      ∀ A,
        ListPrefix (U A)
          (TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              fullTrace) A) := by
    intro A
    have hfullErase :
        TraceTuple.map Tagged.val
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            fullTrace) A =
          fullWord A := by
      calc
        TraceTuple.map Tagged.val
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            fullTrace) A =
            (fullTrace A).map
              (fun x => LocProg.DecoratedLetter.toLocal x.val) := by
                simp [TraceTuple.map, Tagged.mapVal, List.map_map,
                  Function.comp_def]
        _ = ((fullTrace A).map LocProg.wrapTaggedDecorated).map
              LocProg.DecoratedLetter.toLocal := by
                simp [LocProg.wrapTaggedDecorated, List.map_map,
                  Function.comp_def]
        _ = (fullDecorated A).map LocProg.DecoratedLetter.toLocal := by
                rw [← hfullDecoratedEq A]
        _ = fullWord A := hfullDecoratedMap A
    rw [hfullErase]
    exact hprefFull A
  exact ParallelDTracePrefixPreimage.of_complete_prefix
    (program := program)
    (BranchIsMSC := BranchIsMSC)
    (LocalIsMSC := LocalIsMSC)
    (hfullPar := hfullPar)
    (hfullComplete := hfullComplete)
    (hpref := hprefMapped)
    (hprefixMSC := by
      intro prefixBranches prefixTrace hprefixPar hbranchPrefix
      exact hprefixMSC hfullPar hfullComplete hprefixPar hbranchPrefix)

/--
Branch-only concrete-prefix decomposition interface for the parallel Zipper
endgame.

This is the same decorated tagged prefix witness as
`ParallelDTracePrefixPreimage`, but it records only the branch MSC facts.  It
is useful for statements where the local tagged MSC predicate is an
intermediate proof device rather than a hypothesis of the final assembly.
-/
structure ParallelDTraceBranchPrefixPreimage {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    (program : Fin n → Lifeline → LocProg Lifeline Channel)
    (BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop)
    (U : TraceTuple Lifeline (LocalLetter Lifeline Channel)) :
    Type (max u v) where
  prefixBranches :
    ParLocBranchesOf n Lifeline
      (LocProg.DecoratedLetter Lifeline Channel)
  prefixTrace :
    ParLocTaggedTraceOf n Lifeline
      (LocProg.DecoratedLetter Lifeline Channel)
  parloc : ParLocOf prefixBranches prefixTrace
  branch_prefix :
    ∀ i, DTracePrefixTuple (fun A => program i A)
      (prefixBranches i)
  branch_msc : ∀ i, BranchIsMSC i (prefixBranches i)
  erases :
    U =
      TraceTuple.map Tagged.val
        (TraceTuple.map
          (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          prefixTrace)

theorem ParallelDTraceBranchPrefixPreimage.to_distPrefixTrace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hpre :
      ParallelDTraceBranchPrefixPreimage program BranchIsMSC U) :
    DistProg.prefixTrace
      (fun A => LocProg.par (fun i => program i A)) U := by
  rw [hpre.erases]
  exact parallel_dtrace_prefix_to_distPrefixTrace
    (program := program)
    hpre.parloc hpre.branch_prefix

noncomputable def ParallelDTraceBranchPrefixPreimage.of_complete_prefix
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {fullBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {fullTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hfullPar : ParLocOf fullBranches fullTrace)
    (hfullComplete :
      ∀ i A, LocProg.DTrace (program i A) (fullBranches i A))
    (hpref :
      ∀ A,
        ListPrefix (U A)
          (TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              fullTrace) A))
    (hprefixMSC :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i)) :
    ParallelDTraceBranchPrefixPreimage program BranchIsMSC U := by
  classical
  let hpre :
      ParallelDTracePrefixPreimage program BranchIsMSC
        (fun _ : TraceTuple Lifeline
          (Tagged (Fin n) (LocalLetter Lifeline Channel)) => True) U :=
    ParallelDTracePrefixPreimage.of_complete_prefix
      (program := program)
      (BranchIsMSC := BranchIsMSC)
      (LocalIsMSC :=
        fun _ : TraceTuple Lifeline
          (Tagged (Fin n) (LocalLetter Lifeline Channel)) => True)
      (hfullPar := hfullPar)
      (hfullComplete := hfullComplete)
      (hpref := hpref)
      (hprefixMSC := by
        intro prefixBranches prefixTrace hprefixPar hbranchPrefix
        exact ⟨trivial, hprefixMSC hprefixPar hbranchPrefix⟩)
  exact
    { prefixBranches := hpre.prefixBranches
      prefixTrace := hpre.prefixTrace
      parloc := hpre.parloc
      branch_prefix := hpre.branch_prefix
      branch_msc := hpre.branch_msc
      erases := hpre.erases }

noncomputable def ParallelDTraceBranchPrefixPreimage.of_distPrefixTrace
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hprefix :
      DistProg.prefixTrace
        (fun A => LocProg.par (fun i => program i A)) U)
    (hprefixMSC :
      ∀ {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i)) :
    ParallelDTraceBranchPrefixPreimage program BranchIsMSC U := by
  classical
  let hpre :
      ParallelDTracePrefixPreimage program BranchIsMSC
        (fun _ : TraceTuple Lifeline
          (Tagged (Fin n) (LocalLetter Lifeline Channel)) => True) U :=
    ParallelDTracePrefixPreimage.of_distPrefixTrace
      (program := program)
      (BranchIsMSC := BranchIsMSC)
      (LocalIsMSC :=
        fun _ : TraceTuple Lifeline
          (Tagged (Fin n) (LocalLetter Lifeline Channel)) => True)
      hprefix
      (by
        intro fullBranches fullTrace prefixBranches prefixTrace
          hfullPar hfullComplete hprefixPar hbranchPrefix
        exact
          ⟨trivial,
            hprefixMSC hfullPar hfullComplete hprefixPar hbranchPrefix⟩)
  exact
    { prefixBranches := hpre.prefixBranches
      prefixTrace := hpre.prefixTrace
      parloc := hpre.parloc
      branch_prefix := hpre.branch_prefix
      branch_msc := hpre.branch_msc
      erases := hpre.erases }

/--
Construct a branch-prefix preimage from a complete tagged `parloc` execution,
where branch MSC facts may use the fact that the chosen tagged prefix erases to
the original ordinary prefix.
-/
noncomputable def ParallelDTraceBranchPrefixPreimage.of_complete_prefix_with_erasure
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {fullBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {fullTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hfullPar : ParLocOf fullBranches fullTrace)
    (hfullComplete :
      ∀ i A, LocProg.DTrace (program i A) (fullBranches i A))
    (hpref :
      ∀ A,
        ListPrefix (U A)
          (TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              fullTrace) A))
    (hprefixMSC :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        U =
          TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i)) :
    ParallelDTraceBranchPrefixPreimage program BranchIsMSC U := by
  classical
  let hpre :
      ParallelDTracePrefixPreimage program
        (fun _ _ => True)
        (fun _ : TraceTuple Lifeline
          (Tagged (Fin n) (LocalLetter Lifeline Channel)) => True) U :=
    ParallelDTracePrefixPreimage.of_complete_prefix
      (program := program)
      (BranchIsMSC := fun _ _ => True)
      (LocalIsMSC :=
        fun _ : TraceTuple Lifeline
          (Tagged (Fin n) (LocalLetter Lifeline Channel)) => True)
      (hfullPar := hfullPar)
      (hfullComplete := hfullComplete)
      (hpref := hpref)
      (hprefixMSC := by
        intro prefixBranches prefixTrace hprefixPar hbranchPrefix
        exact ⟨trivial, fun _ => trivial⟩)
  exact
    { prefixBranches := hpre.prefixBranches
      prefixTrace := hpre.prefixTrace
      parloc := hpre.parloc
      branch_prefix := hpre.branch_prefix
      branch_msc := hprefixMSC hpre.parloc hpre.erases
      erases := hpre.erases }

/--
Construct a branch-prefix preimage directly from ordinary local `par` prefix
traces, exposing the erasure equality to the branch-MSC obligation.
-/
noncomputable def ParallelDTraceBranchPrefixPreimage.of_distPrefixTrace_with_erasure
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hprefix :
      DistProg.prefixTrace
        (fun A => LocProg.par (fun i => program i A)) U)
    (hprefixMSC :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        U =
          TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i)) :
    ParallelDTraceBranchPrefixPreimage program BranchIsMSC U := by
  classical
  let hpre :
      ParallelDTracePrefixPreimage program
        (fun _ _ => True)
        (fun _ : TraceTuple Lifeline
          (Tagged (Fin n) (LocalLetter Lifeline Channel)) => True) U :=
    ParallelDTracePrefixPreimage.of_distPrefixTrace
      (program := program)
      (BranchIsMSC := fun _ _ => True)
      (LocalIsMSC :=
        fun _ : TraceTuple Lifeline
          (Tagged (Fin n) (LocalLetter Lifeline Channel)) => True)
      hprefix
      (by
        intro fullBranches fullTrace prefixBranches prefixTrace
          hfullPar hfullComplete hprefixPar hbranchPrefix
        exact ⟨trivial, fun _ => trivial⟩)
  exact
    { prefixBranches := hpre.prefixBranches
      prefixTrace := hpre.prefixTrace
      parloc := hpre.parloc
      branch_prefix := hpre.branch_prefix
      branch_msc := hprefixMSC hpre.parloc hpre.erases
      erases := hpre.erases }

/--
Self-contained recursive Zipper property for branch programs.

The continuation is empty because this is the property needed when a branch
prefix of a parallel region is completed before reassembling the surrounding
parallel trace.
-/
def DTraceEmptyZipperProperty {Lifeline : Type u} {Channel : Type v}
    (program : Lifeline → LocProg Lifeline Channel)
    (CompleteMSC IsMSC :
      TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop) : Prop :=
  ∀ U,
    DTracePrefixTuple program U →
    IsMSC U →
    ∃ Ubar,
      ZipPost (fun A word => LocProg.DTrace (program A) word)
        CompleteMSC IsMSC U Ubar
        (TraceTuple.empty Lifeline
          (LocProg.DecoratedLetter Lifeline Channel))

/--
Build the empty-continuation decorated Zipper property from complete decorated
traces.

The theorem deliberately keeps the MSC predicates abstract.  It only needs a
way to recognize complete decorated trace tuples as complete MSCs, and the
usual implication from complete MSCs to prefix MSCs.
-/
theorem dtrace_empty_zipper_property_from_complete
    {Lifeline : Type u} {Channel : Type v}
    {program : Lifeline → LocProg Lifeline Channel}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    (hcompleteOfDTrace :
      ∀ U,
        (∀ A, LocProg.DTrace (program A) (U A)) →
        CompleteMSC U)
    (hcomplete_implies_msc :
      ∀ U, CompleteMSC U → IsMSC U) :
    DTraceEmptyZipperProperty program CompleteMSC IsMSC := by
  classical
  intro U hprefix hmsc
  let Ubar :
      TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun A => Classical.choose (hprefix A)
  have hcompleteLocal :
      ∀ A, LocProg.DTrace (program A) (Ubar A) := by
    intro A
    exact (Classical.choose_spec (hprefix A)).1
  have hprefixLocal :
      ∀ A, ListPrefix (U A) (Ubar A) := by
    intro A
    exact (Classical.choose_spec (hprefix A)).2
  have hcompleted : CompleteMSC Ubar :=
    hcompleteOfDTrace Ubar hcompleteLocal
  refine ⟨Ubar, ?_⟩
  refine
    { prefix_complete := ?_
      completed_msc := hcompleted
      original_msc := ?_
      extended_msc := ?_ }
  · intro A
    refine ⟨hprefixLocal A, hcompleteLocal A, ?_⟩
    intro hnonempty
    exact False.elim (hnonempty rfl)
  · simpa [TraceTuple.append_empty] using hmsc
  · simpa [TraceTuple.append_empty] using
      hcomplete_implies_msc Ubar hcompleted

/--
Direct prefix-completion property for decorated local traces.

This is weaker than the full empty-continuation `ZipPost`: it only asks for a
complete decorated extension of a valid decorated prefix.  The direct
deadlock-freedom endgame for `par` needs exactly this branch-level information.
-/
def DTracePrefixCompletionProperty {Lifeline : Type u} {Channel : Type v}
    (program : Lifeline → LocProg Lifeline Channel)
    (CompleteMSC IsMSC :
      TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop) : Prop :=
  ∀ U,
    DTracePrefixTuple program U →
    IsMSC U →
    ∃ Ubar,
      (∀ A, LocProg.DTrace (program A) (Ubar A)) ∧
      CompleteMSC Ubar ∧
      ∀ A, ListPrefix (U A) (Ubar A)

theorem dtrace_prefix_completion_from_empty_zipper_property
    {Lifeline : Type u} {Channel : Type v}
    {program : Lifeline → LocProg Lifeline Channel}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    (hzipper :
      DTraceEmptyZipperProperty program CompleteMSC IsMSC) :
    DTracePrefixCompletionProperty program CompleteMSC IsMSC := by
  intro U hprefix hmsc
  rcases hzipper U hprefix hmsc with ⟨Ubar, hzip⟩
  exact
    ⟨Ubar,
      fun A => (hzip.prefix_complete A).2.1,
      hzip.completed_msc,
      fun A => (hzip.prefix_complete A).1⟩

theorem dtrace_prefix_completion_property_from_complete
    {Lifeline : Type u} {Channel : Type v}
    {program : Lifeline → LocProg Lifeline Channel}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    (hcompleteOfDTrace :
      ∀ U,
        (∀ A, LocProg.DTrace (program A) (U A)) →
        CompleteMSC U) :
    DTracePrefixCompletionProperty program CompleteMSC IsMSC := by
  classical
  intro U hprefix _hmsc
  let Ubar :
      TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun A => Classical.choose (hprefix A)
  have hcompleteLocal :
      ∀ A, LocProg.DTrace (program A) (Ubar A) := by
    intro A
    exact (Classical.choose_spec (hprefix A)).1
  have hprefixLocal :
      ∀ A, ListPrefix (U A) (Ubar A) := by
    intro A
    exact (Classical.choose_spec (hprefix A)).2
  exact
    ⟨Ubar, hcompleteLocal,
      hcompleteOfDTrace Ubar hcompleteLocal,
      hprefixLocal⟩

/--
Every valid distributed prefix can be completed to a complete distributed
execution.

The MSC predicates are parameters because the concrete word-to-MSC
interpretation is intentionally outside this file.
-/
def PrefixCompletable {Lifeline : Type u} {Channel : Type v}
    (CompleteMSC IsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop)
    (program : DistProg Lifeline Channel) : Prop :=
  ∀ U,
    DistProg.prefixSemantics IsMSC program U →
    ∃ Ubar,
      DistProg.semantics CompleteMSC program Ubar ∧
      ∀ A, ListPrefix (U A) (Ubar A)

/--
Abstract deadlock-freedom consequence of the Zipper lemma.

If every valid prefix satisfies the empty-continuation Zipper postcondition,
then every such prefix has a complete distributed extension.
-/
theorem parallel_deadlock_free_from_zipper
    {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hzipper :
      ∀ U,
        DistProg.prefixSemantics IsMSC program U →
        ∃ Ubar,
          ZipPost (fun A word => LocProg.trace (program A) word)
            CompleteMSC IsMSC U Ubar
            (TraceTuple.empty Lifeline
              (LocalLetter Lifeline Channel))) :
    PrefixCompletable CompleteMSC IsMSC program := by
  intro U hprefix
  rcases hzipper U hprefix with ⟨Ubar, hpost⟩
  refine ⟨Ubar, ?_, ?_⟩
  · exact ⟨fun A => (hpost.prefix_complete A).2.1,
      hpost.completed_msc⟩
  · intro A
    exact (hpost.prefix_complete A).1

/--
Complete-local traces are complete MSCs whenever the projected program is
prefix-completable and its complete local language is prefix-free.

This is the abstract Isola-style endgame: complete the MSC prefix, then use
local prefix-freeness to show that the completion did not add any local
letters.
-/
theorem complete_of_prefixCompletable_and_local_prefix_free
    {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hcomplete :
      PrefixCompletable CompleteMSC IsMSC program)
    (hprefixFree :
      LocalPrefixFree (fun A word => LocProg.trace (program A) word))
    {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (htrace : DistProg.trace program U)
    (hmsc : IsMSC U) :
    CompleteMSC U := by
  rcases hcomplete U
      ⟨DistProg.trace_prefixTrace htrace, hmsc⟩ with
    ⟨Ubar, hsem, hpref⟩
  have heq : U = Ubar := by
    funext A
    exact hprefixFree A (htrace A) (hsem.1 A) (hpref A)
  simpa [heq] using hsem.2

/--
The completed prefix produced by the parallel Zipper postcondition is an
ordinary distributed trace of the local `par` program after branch tags are
forgotten.
-/
theorem parallel_zipPost_completed_distSemantics {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {TaggedCompleteMSC TaggedIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {CompleteMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {U Ubar V :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (hcompletedMap :
      ∀ M, TaggedCompleteMSC M →
        CompleteMSC (TraceTuple.map Tagged.val M))
    (hzip :
      ZipPost
        (TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word))
        TaggedCompleteMSC TaggedIsMSC U Ubar V) :
    DistProg.semantics CompleteMSC
      (fun A => LocProg.par (fun i => program i A))
      (TraceTuple.map Tagged.val Ubar) := by
  refine ⟨?_, hcompletedMap Ubar hzip.completed_msc⟩
  intro A
  exact LocProg.taggedParLocCompleteLocal_to_trace_par
    ((hzip.prefix_complete A).2.1)

theorem parallel_zipPost_prefix_extension {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {TaggedCompleteMSC TaggedIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {U Ubar V :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (hzip :
      ZipPost
        (TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word))
        TaggedCompleteMSC TaggedIsMSC U Ubar V) :
    ∀ A,
      ListPrefix
        (TraceTuple.map Tagged.val U A)
        (TraceTuple.map Tagged.val Ubar A) := by
  intro A
  exact list_map_preserves_prefix Tagged.val (hzip.prefix_complete A).1

theorem parallel_zipPost_to_prefixCompletable {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {TaggedCompleteMSC TaggedIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {CompleteMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {U Ubar V :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (hcompletedMap :
      ∀ M, TaggedCompleteMSC M →
        CompleteMSC (TraceTuple.map Tagged.val M))
    (hzip :
      ZipPost
        (TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word))
        TaggedCompleteMSC TaggedIsMSC U Ubar V) :
    ∃ UbarOrd,
      DistProg.semantics CompleteMSC
        (fun A => LocProg.par (fun i => program i A))
        UbarOrd ∧
      ∀ A,
        ListPrefix
          (TraceTuple.map Tagged.val U A)
          (UbarOrd A) := by
  refine ⟨TraceTuple.map Tagged.val Ubar, ?_, ?_⟩
  · exact parallel_zipPost_completed_distSemantics
      (program := program)
      (hcompletedMap := hcompletedMap)
      hzip
  · exact parallel_zipPost_prefix_extension
      (program := program)
      hzip

/--
Ordinary prefix completability from a decorated empty-continuation Zipper
property.

This is the erasure bridge for the recursive decorated local language: a
decorated valid prefix is completed by the decorated Zipper property, and the
completion is then mapped back to ordinary local traces.
-/
theorem prefixCompletable_from_dtrace_empty_zipper_preimage
    {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hzipper :
      DTraceEmptyZipperProperty program
        (fun M =>
          CompleteMSC
            (TraceTuple.map LocProg.DecoratedLetter.toLocal M))
        (fun M =>
          IsMSC
            (TraceTuple.map LocProg.DecoratedLetter.toLocal M))) :
    ∀ U,
      DTracePrefixTuple program U →
      IsMSC (TraceTuple.map LocProg.DecoratedLetter.toLocal U) →
      ∃ Ubar,
        DistProg.semantics CompleteMSC program Ubar ∧
        ∀ A,
          ListPrefix
            ((U A).map LocProg.DecoratedLetter.toLocal)
            (Ubar A) := by
  intro U hprefix hmsc
  rcases hzipper U hprefix hmsc with ⟨UbarDecor, hpost⟩
  refine
    ⟨TraceTuple.map LocProg.DecoratedLetter.toLocal UbarDecor,
      ?_, ?_⟩
  · exact DTraceTuple.to_distSemantics
      (program := program)
      (U := UbarDecor)
      (fun A => (hpost.prefix_complete A).2.1)
      hpost.completed_msc
  · intro A
    exact list_map_preserves_prefix
      LocProg.DecoratedLetter.toLocal
      (hpost.prefix_complete A).1

/--
Instantiate recursive branch Zipper properties for all branches of a parallel
region, producing the completed branch tuples used by the assembly lemmas.
-/
theorem branch_zipPosts_from_dtrace_empty_properties {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n →
        TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
          Prop}
    {prefixBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchPrefix :
      ∀ i, DTracePrefixTuple (fun A => program i A) (prefixBranches i))
    (hbranchMSC :
      ∀ i, BranchIsMSC i (prefixBranches i)) :
    ∃ completedBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel),
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (BranchCompleteMSC i) (BranchIsMSC i)
          (prefixBranches i) (completedBranches i)
          (TraceTuple.empty Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)) := by
  classical
  have h :
      ∀ i,
        ∃ completed :
          TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel),
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) completed
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel)) := by
    intro i
    exact hbranchZipper i (prefixBranches i)
      (hbranchPrefix i) (hbranchMSC i)
  let completed :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i => Classical.choose (h i)
  exact ⟨completed, fun i => Classical.choose_spec (h i)⟩

/--
Generic residual-form assembly of the parallel Zipper case.

This is the same abstract assembly as `zipper_parallel_zipPost_from_residuals`,
but parameterized by the local alphabet.  It is used to instantiate the proof
with recursively decorated local traces for nested `par`.
-/
theorem zipper_parallel_zipPost_from_residuals_of {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) α) → Prop}
    {prefixBranches residualBranches targetBranches :
      ParLocBranchesOf n Lifeline α}
    {prefixTrace residualTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline α}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hresidual : ParLocOf residualBranches residualTrace)
    (htarget :
      ∀ i A, targetBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hcomplete : ∀ i A, complete i A (targetBranches i A))
    (hbranchEmpty :
      ∀ A, continuationTrace A ≠ [] → ∀ i, residualBranches i A = [])
    (hcompleted :
      CompleteMSC (ParLocTaggedTraceOf.append prefixTrace residualTrace))
    (horiginal : IsMSC (TraceTuple.append prefixTrace continuationTrace))
    (hextended :
      IsMSC (TraceTuple.append
        (ParLocTaggedTraceOf.append prefixTrace residualTrace)
        continuationTrace)) :
    ZipPost (TaggedParLocCompleteLocalOf complete) CompleteMSC IsMSC
      prefixTrace (ParLocTaggedTraceOf.append prefixTrace residualTrace)
      continuationTrace := by
  refine ZipPost.of_residuals
    (U := prefixTrace)
    (Ubar := ParLocTaggedTraceOf.append prefixTrace residualTrace)
    (V := continuationTrace)
    (residual := residualTrace)
    ?_ ?_ ?_ hcompleted horiginal hextended
  · intro A
    rfl
  · intro A
    exact TaggedParLocCompleteLocalOf.of_components
      (parlocOf_extend_from_residual hprefix hresidual htarget)
      (fun i => hcomplete i A)
  · exact ZipPost.stable_trace_residual_of_branch_residuals_of hresidual
      hbranchEmpty

/-- Generic parallel `ZipPost` assembly from branch-level `ZipPost`s. -/
theorem zipper_parallel_zipPost_from_branch_zipPosts_of {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {complete : Fin n → Lifeline → Word α → Prop}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline α → Prop}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) α) → Prop}
    {prefixBranches completedBranches residualBranches continuationBranches :
      ParLocBranchesOf n Lifeline α}
    {prefixTrace residualTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline α}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hresidual : ParLocOf residualBranches residualTrace)
    (hresidualEq :
      ∀ i A, completedBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hprefixFree : ComponentPrefixFreeOf complete)
    (hbranchZip :
      ∀ i, ZipPost (complete i) (BranchCompleteMSC i) (BranchIsMSC i)
        (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, complete i A (prefixBranches i A))
    (hcompleted :
      CompleteMSC (ParLocTaggedTraceOf.append prefixTrace residualTrace))
    (horiginal : IsMSC (TraceTuple.append prefixTrace continuationTrace))
    (hextended :
      IsMSC (TraceTuple.append
        (ParLocTaggedTraceOf.append prefixTrace residualTrace)
        continuationTrace)) :
    ZipPost (TaggedParLocCompleteLocalOf complete) CompleteMSC IsMSC
      prefixTrace (ParLocTaggedTraceOf.append prefixTrace residualTrace)
      continuationTrace := by
  refine zipper_parallel_zipPost_from_residuals_of
    (hprefix := hprefix)
    (hresidual := hresidual)
    (htarget := hresidualEq)
    (hcomplete := ?_)
    (hbranchEmpty := ?_)
    hcompleted horiginal hextended
  · intro i A
    exact ((hbranchZip i).prefix_complete A).2.1
  · exact ZipPost.branch_residuals_empty_of_complete_prefixes_of
      hprefixFree hbranchZip hcompletePrefix hresidualEq

/--
Abstract residual-form assembly of the parallel Zipper case.

The theorem assumes the semantic ingredients that still belong to the full
local and MSC development: completed branch words are complete, residuals
vanish on lifelines with nonempty continuations, and the three MSC conditions
hold.  From these hypotheses it constructs the four-condition `ZipPost` for the
tagged `parloc` trace.
-/
theorem zipper_parallel_zipPost_from_residuals {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete :
      Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {prefixBranches residualBranches targetBranches :
      ParLocBranches n Lifeline Channel}
    {prefixTrace residualTrace continuationTrace :
      ParLocTaggedTrace n Lifeline Channel}
    (hprefix : ParLoc prefixBranches prefixTrace)
    (hresidual : ParLoc residualBranches residualTrace)
    (htarget :
      ∀ i A, targetBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hcomplete : ∀ i A, complete i A (targetBranches i A))
    (hbranchEmpty :
      ∀ A, continuationTrace A ≠ [] → ∀ i, residualBranches i A = [])
    (hcompleted :
      CompleteMSC (ParLocTaggedTrace.append prefixTrace residualTrace))
    (horiginal : IsMSC (TraceTuple.append prefixTrace continuationTrace))
    (hextended :
      IsMSC (TraceTuple.append
        (ParLocTaggedTrace.append prefixTrace residualTrace)
        continuationTrace)) :
    ZipPost (TaggedParLocCompleteLocal complete) CompleteMSC IsMSC
      prefixTrace (ParLocTaggedTrace.append prefixTrace residualTrace)
      continuationTrace := by
  refine ZipPost.of_residuals
    (U := prefixTrace)
    (Ubar := ParLocTaggedTrace.append prefixTrace residualTrace)
    (V := continuationTrace)
    (residual := residualTrace)
    ?_ ?_ ?_ hcompleted horiginal hextended
  · intro A
    rfl
  · intro A
    exact TaggedParLocCompleteLocal.of_components
      (parloc_extend_from_residual hprefix hresidual htarget)
      (fun i => hcomplete i A)
  · exact ZipPost.stable_trace_residual_of_branch_residuals hresidual
      hbranchEmpty

/--
Parallel `ZipPost` assembly from branch-level `ZipPost`s.

This is the abstract shape of the paper's parallel Zipper case.  The branch
postconditions provide complete branch words and residual equations; component
prefix-freeness plus the explicit `hcompletePrefix` hypothesis makes branch
residuals vanish when the top-level continuation is nonempty.  The remaining
MSC hypotheses are kept explicit.
-/
theorem zipper_parallel_zipPost_from_branch_zipPosts {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {complete :
      Fin n → Lifeline → Word (LocalLetter Lifeline Channel) → Prop}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {prefixBranches completedBranches residualBranches continuationBranches :
      ParLocBranches n Lifeline Channel}
    {prefixTrace residualTrace continuationTrace :
      ParLocTaggedTrace n Lifeline Channel}
    (hprefix : ParLoc prefixBranches prefixTrace)
    (hresidual : ParLoc residualBranches residualTrace)
    (hresidualEq :
      ∀ i A, completedBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hprefixFree : ComponentPrefixFree complete)
    (hbranchZip :
      ∀ i, ZipPost (complete i) (BranchCompleteMSC i) (BranchIsMSC i)
        (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, complete i A (prefixBranches i A))
    (hcompleted :
      CompleteMSC (ParLocTaggedTrace.append prefixTrace residualTrace))
    (horiginal : IsMSC (TraceTuple.append prefixTrace continuationTrace))
    (hextended :
      IsMSC (TraceTuple.append
        (ParLocTaggedTrace.append prefixTrace residualTrace)
        continuationTrace)) :
    ZipPost (TaggedParLocCompleteLocal complete) CompleteMSC IsMSC
      prefixTrace (ParLocTaggedTrace.append prefixTrace residualTrace)
      continuationTrace := by
  refine zipper_parallel_zipPost_from_residuals
    (hprefix := hprefix)
    (hresidual := hresidual)
    (htarget := hresidualEq)
    (hcomplete := ?_)
    (hbranchEmpty := ?_)
    hcompleted horiginal hextended
  · intro i A
    exact ((hbranchZip i).prefix_complete A).2.1
  · exact ZipPost.branch_residuals_empty_of_complete_prefixes
      hprefixFree hbranchZip hcompletePrefix hresidualEq

theorem zipper_parallel_zipPost_from_noPar_branch_zipPosts {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {prefixBranches completedBranches residualBranches continuationBranches :
      ParLocBranches n Lifeline Channel}
    {prefixTrace residualTrace continuationTrace :
      ParLocTaggedTrace n Lifeline Channel}
    (hprefix : ParLoc prefixBranches prefixTrace)
    (hresidual : ParLoc residualBranches residualTrace)
    (hresidualEq :
      ∀ i A, completedBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hnopar : ∀ i A, LocProg.NoPar (program i A))
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.trace (program i A) word)
          (BranchCompleteMSC i) (BranchIsMSC i)
          (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.trace (program i A) (prefixBranches i A))
    (hcompleted :
      CompleteMSC (ParLocTaggedTrace.append prefixTrace residualTrace))
    (horiginal : IsMSC (TraceTuple.append prefixTrace continuationTrace))
    (hextended :
      IsMSC (TraceTuple.append
        (ParLocTaggedTrace.append prefixTrace residualTrace)
        continuationTrace)) :
    ZipPost
      (TaggedParLocCompleteLocal
        (fun i A word => LocProg.trace (program i A) word))
      CompleteMSC IsMSC
      prefixTrace (ParLocTaggedTrace.append prefixTrace residualTrace)
      continuationTrace :=
  zipper_parallel_zipPost_from_branch_zipPosts
    (hprefix := hprefix)
    (hresidual := hresidual)
    (hresidualEq := hresidualEq)
    (hprefixFree := LocProg.componentPrefixFree_noPar hguard hnopar)
    (hbranchZip := hbranchZip)
    (hcompletePrefix := hcompletePrefix)
    hcompleted horiginal hextended

theorem zipper_zipPost_dtrace_to_trace
    {Lifeline : Type u} {Channel : Type v}
    {program : Lifeline → LocProg Lifeline Channel}
    {DecorCompleteMSC DecorIsMSC :
      TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {U Ubar V :
      TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel)}
    (hcompleted :
      DecorCompleteMSC Ubar →
        LocalCompleteMSC
          (TraceTuple.map LocProg.DecoratedLetter.toLocal Ubar))
    (horiginal :
      DecorIsMSC (TraceTuple.append U V) →
        LocalIsMSC
          (TraceTuple.map LocProg.DecoratedLetter.toLocal
            (TraceTuple.append U V)))
    (hextended :
      DecorIsMSC (TraceTuple.append Ubar V) →
        LocalIsMSC
          (TraceTuple.map LocProg.DecoratedLetter.toLocal
            (TraceTuple.append Ubar V)))
    (hzip :
      ZipPost (fun A word => LocProg.DTrace (program A) word)
        DecorCompleteMSC DecorIsMSC U Ubar V) :
    ZipPost (fun A word => LocProg.trace (program A) word)
      LocalCompleteMSC LocalIsMSC
      (TraceTuple.map LocProg.DecoratedLetter.toLocal U)
      (TraceTuple.map LocProg.DecoratedLetter.toLocal Ubar)
      (TraceTuple.map LocProg.DecoratedLetter.toLocal V) :=
  hzip.map
    LocProg.DecoratedLetter.toLocal
    (fun _ _ hcomplete => LocProg.dtrace_to_trace hcomplete)
    hcompleted horiginal hextended

theorem zipper_zipPost_dtrace_to_trace_preimage
    {Lifeline : Type u} {Channel : Type v}
    {program : Lifeline → LocProg Lifeline Channel}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {U Ubar V :
      TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel)}
    (hzip :
      ZipPost (fun A word => LocProg.DTrace (program A) word)
        (fun M => LocalCompleteMSC
          (TraceTuple.map LocProg.DecoratedLetter.toLocal M))
        (fun M => LocalIsMSC
          (TraceTuple.map LocProg.DecoratedLetter.toLocal M))
        U Ubar V) :
    ZipPost (fun A word => LocProg.trace (program A) word)
      LocalCompleteMSC LocalIsMSC
      (TraceTuple.map LocProg.DecoratedLetter.toLocal U)
      (TraceTuple.map LocProg.DecoratedLetter.toLocal Ubar)
      (TraceTuple.map LocProg.DecoratedLetter.toLocal V) :=
  zipper_zipPost_dtrace_to_trace
    (hcompleted := fun h => h)
    (horiginal := fun h => h)
    (hextended := fun h => h)
    hzip

theorem zipper_branch_zipPosts_dtrace_to_trace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {DecorCompleteMSC DecorIsMSC :
      Fin n →
        TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      Fin n → TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {U Ubar V :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hcompleted :
      ∀ i M, DecorCompleteMSC i M →
        LocalCompleteMSC i
          (TraceTuple.map LocProg.DecoratedLetter.toLocal M))
    (horiginal :
      ∀ i M, DecorIsMSC i M →
        LocalIsMSC i
          (TraceTuple.map LocProg.DecoratedLetter.toLocal M))
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (DecorCompleteMSC i) (DecorIsMSC i)
          (U i) (Ubar i) (V i)) :
    ∀ i,
      ZipPost (fun A word => LocProg.trace (program i A) word)
        (LocalCompleteMSC i) (LocalIsMSC i)
        (TraceTuple.map LocProg.DecoratedLetter.toLocal (U i))
        (TraceTuple.map LocProg.DecoratedLetter.toLocal (Ubar i))
        (TraceTuple.map LocProg.DecoratedLetter.toLocal (V i)) := by
  intro i
  exact zipper_zipPost_dtrace_to_trace
    (program := fun A => program i A)
    (hcompleted := fun h => hcompleted i _ h)
    (horiginal := fun h => horiginal i _ h)
    (hextended := fun h => horiginal i _ h)
    (hbranchZip i)

theorem zipper_branch_zipPosts_dtrace_to_trace_preimage {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {LocalCompleteMSC LocalIsMSC :
      Fin n → TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {U Ubar V :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (fun M => LocalCompleteMSC i
            (TraceTuple.map LocProg.DecoratedLetter.toLocal M))
          (fun M => LocalIsMSC i
            (TraceTuple.map LocProg.DecoratedLetter.toLocal M))
          (U i) (Ubar i) (V i)) :
    ∀ i,
      ZipPost (fun A word => LocProg.trace (program i A) word)
        (LocalCompleteMSC i) (LocalIsMSC i)
        (TraceTuple.map LocProg.DecoratedLetter.toLocal (U i))
        (TraceTuple.map LocProg.DecoratedLetter.toLocal (Ubar i))
        (TraceTuple.map LocProg.DecoratedLetter.toLocal (V i)) := by
  intro i
  exact zipper_zipPost_dtrace_to_trace_preimage
    (program := fun A => program i A)
    (hbranchZip i)

theorem zipper_parallel_zipPost_from_dtrace_branch_zipPosts {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel)) → Prop}
    {prefixBranches completedBranches residualBranches continuationBranches :
      ParLocBranchesOf n Lifeline (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace residualTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hresidual : ParLocOf residualBranches residualTrace)
    (hresidualEq :
      ∀ i A, completedBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (BranchCompleteMSC i) (BranchIsMSC i)
          (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.DTrace (program i A) (prefixBranches i A))
    (hcompleted :
      CompleteMSC (ParLocTaggedTraceOf.append prefixTrace residualTrace))
    (horiginal : IsMSC (TraceTuple.append prefixTrace continuationTrace))
    (hextended :
      IsMSC (TraceTuple.append
        (ParLocTaggedTraceOf.append prefixTrace residualTrace)
        continuationTrace)) :
    ZipPost
      (TaggedParLocCompleteLocalOf
        (fun i A word => LocProg.DTrace (program i A) word))
      CompleteMSC IsMSC
      prefixTrace (ParLocTaggedTraceOf.append prefixTrace residualTrace)
      continuationTrace :=
  zipper_parallel_zipPost_from_branch_zipPosts_of
    (hprefix := hprefix)
    (hresidual := hresidual)
    (hresidualEq := hresidualEq)
    (hprefixFree := LocProg.componentPrefixFree_dtrace hguard)
    (hbranchZip := hbranchZip)
    (hcompletePrefix := hcompletePrefix)
    hcompleted horiginal hextended

theorem zipper_parallel_zipPost_dtrace_to_trace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {DecorCompleteMSC DecorIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel)) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {U Ubar V :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel))}
    (hcompleted :
      DecorCompleteMSC Ubar →
        LocalCompleteMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            Ubar))
    (horiginal :
      DecorIsMSC (TraceTuple.append U V) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (TraceTuple.append U V)))
    (hextended :
      DecorIsMSC (TraceTuple.append Ubar V) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (TraceTuple.append Ubar V)))
    (hzip :
      ZipPost
        (TaggedParLocCompleteLocalOf
          (fun i A word => LocProg.DTrace (program i A) word))
        DecorCompleteMSC DecorIsMSC U Ubar V) :
    ZipPost
      (TaggedParLocCompleteLocal
        (fun i A word => LocProg.trace (program i A) word))
      LocalCompleteMSC LocalIsMSC
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) U)
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) Ubar)
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) V) :=
  hzip.map
    (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
    (fun _ _ hcomplete =>
      LocProg.dtrace_taggedParLocCompleteLocal_to_trace hcomplete)
    hcompleted horiginal hextended

theorem zipper_parallel_zipPost_from_dtrace_branch_zipPosts_to_trace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    {DecorCompleteMSC DecorIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel)) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {prefixBranches completedBranches residualBranches continuationBranches :
      ParLocBranchesOf n Lifeline (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace residualTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hresidual : ParLocOf residualBranches residualTrace)
    (hresidualEq :
      ∀ i A, completedBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (BranchCompleteMSC i) (BranchIsMSC i)
          (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.DTrace (program i A) (prefixBranches i A))
    (hcompleted :
      DecorCompleteMSC (ParLocTaggedTraceOf.append prefixTrace residualTrace))
    (horiginal : DecorIsMSC (TraceTuple.append prefixTrace continuationTrace))
    (hextended :
      DecorIsMSC (TraceTuple.append
        (ParLocTaggedTraceOf.append prefixTrace residualTrace)
        continuationTrace))
    (hcompletedMap :
      ∀ M, DecorCompleteMSC M →
        LocalCompleteMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) M))
    (hisMSCMap :
      ∀ M, DecorIsMSC M →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) M)) :
    ZipPost
      (TaggedParLocCompleteLocal
        (fun i A word => LocProg.trace (program i A) word))
      LocalCompleteMSC LocalIsMSC
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        prefixTrace)
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        (ParLocTaggedTraceOf.append prefixTrace residualTrace))
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        continuationTrace) := by
  have hdecor :
      ZipPost
        (TaggedParLocCompleteLocalOf
          (fun i A word => LocProg.DTrace (program i A) word))
        DecorCompleteMSC DecorIsMSC
        prefixTrace (ParLocTaggedTraceOf.append prefixTrace residualTrace)
        continuationTrace :=
    zipper_parallel_zipPost_from_dtrace_branch_zipPosts
      (hprefix := hprefix)
      (hresidual := hresidual)
      (hresidualEq := hresidualEq)
      (hguard := hguard)
      (hbranchZip := hbranchZip)
      (hcompletePrefix := hcompletePrefix)
      hcompleted horiginal hextended
  exact zipper_parallel_zipPost_dtrace_to_trace
    (program := program)
    (hcompleted := fun h => hcompletedMap _ h)
    (horiginal := fun h => hisMSCMap _ h)
    (hextended := fun h => hisMSCMap _ h)
    hdecor

/--
Paper-facing parallel case of the Zipper lemma.

This theorem is still local-program-parametric: it does not depend on the
source syntax used to produce the branch local programs.  Accordingly, it
exposes the semantic ingredients as hypotheses: branch Zipper postconditions,
prefix/residual decomposition, the three MSC facts for the assembled tuple,
and the maps that forget recursive proof decorations back to ordinary local
letters.  Under those hypotheses it proves the full four-condition `ZipPost`
for the projected local parallel program.
-/
theorem zipper_lemma_parallel {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    {DecorCompleteMSC DecorIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel)) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {prefixBranches completedBranches residualBranches continuationBranches :
      ParLocBranchesOf n Lifeline (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace residualTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hresidual : ParLocOf residualBranches residualTrace)
    (hresidualEq :
      ∀ i A, completedBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (BranchCompleteMSC i) (BranchIsMSC i)
          (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.DTrace (program i A) (prefixBranches i A))
    (hcompleted :
      DecorCompleteMSC (ParLocTaggedTraceOf.append prefixTrace residualTrace))
    (horiginal : DecorIsMSC (TraceTuple.append prefixTrace continuationTrace))
    (hextended :
      DecorIsMSC (TraceTuple.append
        (ParLocTaggedTraceOf.append prefixTrace residualTrace)
        continuationTrace))
    (hcompletedMap :
      ∀ M, DecorCompleteMSC M →
        LocalCompleteMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) M))
    (hisMSCMap :
      ∀ M, DecorIsMSC M →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) M)) :
    ZipPost
      (TaggedParLocCompleteLocal
        (fun i A word => LocProg.trace (program i A) word))
      LocalCompleteMSC LocalIsMSC
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        prefixTrace)
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        (ParLocTaggedTraceOf.append prefixTrace residualTrace))
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        continuationTrace) :=
  zipper_parallel_zipPost_from_dtrace_branch_zipPosts_to_trace
    (hprefix := hprefix)
    (hresidual := hresidual)
    (hresidualEq := hresidualEq)
    (hguard := hguard)
    (hbranchZip := hbranchZip)
    (hcompletePrefix := hcompletePrefix)
    hcompleted horiginal hextended hcompletedMap hisMSCMap

/--
Parallel Zipper lemma with decorated MSC predicates chosen as preimages.

This removes the two explicit map assumptions from `zipper_lemma_parallel`.
The completed/original/extended MSC hypotheses are stated directly after
forgetting recursive proof decorations back to ordinary local letters.
-/
theorem zipper_lemma_parallel_preimage {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {prefixBranches completedBranches residualBranches continuationBranches :
      ParLocBranchesOf n Lifeline (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace residualTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hresidual : ParLocOf residualBranches residualTrace)
    (hresidualEq :
      ∀ i A, completedBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (BranchCompleteMSC i) (BranchIsMSC i)
          (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.DTrace (program i A) (prefixBranches i A))
    (hcompleted :
      LocalCompleteMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (ParLocTaggedTraceOf.append prefixTrace residualTrace)))
    (horiginal :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append prefixTrace continuationTrace)))
    (hextended :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append
            (ParLocTaggedTraceOf.append prefixTrace residualTrace)
            continuationTrace))) :
    ZipPost
      (TaggedParLocCompleteLocal
        (fun i A word => LocProg.trace (program i A) word))
      LocalCompleteMSC LocalIsMSC
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        prefixTrace)
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        (ParLocTaggedTraceOf.append prefixTrace residualTrace))
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        continuationTrace) :=
  zipper_lemma_parallel
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (DecorCompleteMSC := fun M =>
      LocalCompleteMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) M))
    (DecorIsMSC := fun M =>
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) M))
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (hprefix := hprefix)
    (hresidual := hresidual)
    (hresidualEq := hresidualEq)
    (hguard := hguard)
    (hbranchZip := hbranchZip)
    (hcompletePrefix := hcompletePrefix)
    hcompleted horiginal hextended
    (fun _ h => h)
    (fun _ h => h)

/--
Parallel Zipper lemma using the canonical serial residual interleaving.

This removes the explicit residual `ParLocOf` witness: residual branch words are
serialized branch by branch, which is always a valid tagged `parloc`
interleaving by `parlocOf_serial`.
-/
theorem zipper_lemma_parallel_preimage_serial {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {prefixBranches completedBranches residualBranches continuationBranches :
      ParLocBranchesOf n Lifeline (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hresidualEq :
      ∀ i A, completedBranches i A =
        prefixBranches i A ++ residualBranches i A)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (BranchCompleteMSC i) (BranchIsMSC i)
          (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.DTrace (program i A) (prefixBranches i A))
    (hcompleted :
      LocalCompleteMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (ParLocTaggedTraceOf.append prefixTrace
            (serialParLocTraceOf residualBranches))))
    (horiginal :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append prefixTrace continuationTrace)))
    (hextended :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf residualBranches))
            continuationTrace))) :
    ZipPost
      (TaggedParLocCompleteLocal
        (fun i A word => LocProg.trace (program i A) word))
      LocalCompleteMSC LocalIsMSC
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        prefixTrace)
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        (ParLocTaggedTraceOf.append prefixTrace
          (serialParLocTraceOf residualBranches)))
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        continuationTrace) :=
  zipper_lemma_parallel_preimage
    (hprefix := hprefix)
    (hresidual := parlocOf_serial residualBranches)
    (hresidualEq := hresidualEq)
    (hguard := hguard)
    (hbranchZip := hbranchZip)
    (hcompletePrefix := hcompletePrefix)
    hcompleted horiginal hextended

/--
Parallel Zipper lemma deriving residual branch words from branch `ZipPost`s.

The branch completions already contain prefix witnesses.  This theorem chooses
their suffixes as the residual branch words and serializes them, so callers do
not need to provide residual words or residual equations separately.
-/
theorem zipper_lemma_parallel_preimage_from_branch_zipPosts {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {prefixBranches completedBranches continuationBranches :
      ParLocBranchesOf n Lifeline (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (BranchCompleteMSC i) (BranchIsMSC i)
          (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.DTrace (program i A) (prefixBranches i A))
    (hcompleted :
      LocalCompleteMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (ParLocTaggedTraceOf.append prefixTrace
            (serialParLocTraceOf
              (fun i A => (hbranchZip i).residual A)))))
    (horiginal :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append prefixTrace continuationTrace)))
    (hextended :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A)))
            continuationTrace))) :
    ZipPost
      (TaggedParLocCompleteLocal
        (fun i A word => LocProg.trace (program i A) word))
      LocalCompleteMSC LocalIsMSC
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        prefixTrace)
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        (ParLocTaggedTraceOf.append prefixTrace
          (serialParLocTraceOf
            (fun i A => (hbranchZip i).residual A))))
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        continuationTrace) :=
  zipper_lemma_parallel_preimage_serial
    (hprefix := hprefix)
    (hresidualEq := fun i A => ZipPost.residual_eq (hbranchZip i) A)
    (hguard := hguard)
    (hbranchZip := hbranchZip)
    (hcompletePrefix := hcompletePrefix)
    hcompleted horiginal hextended

/--
Parallel Zipper lemma from branch `ZipPost`s and a single MSC assembly
interface.

The only MSC hypothesis exposed directly is the original one, matching the Zipper
lemma hypothesis.  The completed-prefix and extended-prefix MSC facts are
provided by `MappedParallelZipMSCAssembly`.
-/
theorem zipper_lemma_parallel_preimage_from_branch_zipPosts_assembly {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline (LocProg.DecoratedLetter Lifeline Channel) →
        Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline (Tagged (Fin n) (LocalLetter Lifeline Channel)) →
        Prop}
    {prefixBranches completedBranches continuationBranches :
      ParLocBranchesOf n Lifeline (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZip :
      ∀ i,
        ZipPost (fun A word => LocProg.DTrace (program i A) word)
          (BranchCompleteMSC i) (BranchIsMSC i)
          (prefixBranches i) (completedBranches i) (continuationBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.DTrace (program i A) (prefixBranches i A))
    (hassembly :
      MappedParallelZipMSCAssembly
        LocProg.DecoratedLetter.toLocal
        LocalCompleteMSC LocalIsMSC
        prefixTrace
        (serialParLocTraceOf
          (fun i A => (hbranchZip i).residual A))
        continuationTrace)
    (horiginal :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append prefixTrace continuationTrace))) :
    ZipPost
      (TaggedParLocCompleteLocal
        (fun i A word => LocProg.trace (program i A) word))
      LocalCompleteMSC LocalIsMSC
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        prefixTrace)
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        (ParLocTaggedTraceOf.append prefixTrace
          (serialParLocTraceOf
            (fun i A => (hbranchZip i).residual A))))
      (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        continuationTrace) :=
  zipper_lemma_parallel_preimage_from_branch_zipPosts
    (hprefix := hprefix)
    (hguard := hguard)
    (hbranchZip := hbranchZip)
    (hcompletePrefix := hcompletePrefix)
    hassembly.completed
    horiginal
    (hassembly.extended_of_original horiginal)

/--
Parallel Zipper lemma from recursive branch Zipper properties.

This closes the branch-completion gap left by
`zipper_lemma_parallel_preimage_from_branch_zipPosts_assembly`: branch
`ZipPost`s are obtained from the recursive empty-continuation Zipper property
for each component.  The remaining assembly hypothesis is the concrete
word-to-MSC obligation for the surrounding parallel node.
-/
theorem zipper_lemma_parallel_exists_from_branch_properties_assembly
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {prefixBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchPrefix :
      ∀ i, DTracePrefixTuple (fun A => program i A) (prefixBranches i))
    (hbranchMSC :
      ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.DTrace (program i A) (prefixBranches i A))
    (hassembly :
      ∀ (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        MappedParallelZipMSCAssembly
          LocProg.DecoratedLetter.toLocal
          LocalCompleteMSC LocalIsMSC
          prefixTrace
          (serialParLocTraceOf
            (fun i A => (hbranchZip i).residual A))
          continuationTrace)
    (horiginal :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append prefixTrace continuationTrace))) :
    ∃ Ubar,
      ZipPost
        (TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word))
        LocalCompleteMSC LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          prefixTrace)
        Ubar
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          continuationTrace) := by
  classical
  obtain ⟨completedBranches, hbranchZip⟩ :=
    branch_zipPosts_from_dtrace_empty_properties
      (program := program)
      (BranchCompleteMSC := BranchCompleteMSC)
      (BranchIsMSC := BranchIsMSC)
      (prefixBranches := prefixBranches)
      hbranchZipper hbranchPrefix hbranchMSC
  let hzip :=
    zipper_lemma_parallel_preimage_from_branch_zipPosts_assembly
      (program := program)
      (BranchCompleteMSC := BranchCompleteMSC)
      (BranchIsMSC := BranchIsMSC)
      (LocalCompleteMSC := LocalCompleteMSC)
      (LocalIsMSC := LocalIsMSC)
      (hprefix := hprefix)
      (hguard := hguard)
      (hbranchZip := hbranchZip)
      (hcompletePrefix := hcompletePrefix)
      (hassembly := hassembly completedBranches hbranchZip)
      (horiginal := horiginal)
  exact ⟨_, hzip⟩

/--
Parallel branch-property Zipper theorem, projected to ordinary local `par`
executions.

This composes the decorated/tagged parallel Zipper theorem with the
tag-forgetting bridge to `LocProg.par`.  The only extra assumption is the map
from complete MSCs over tagged local words to complete MSCs over the ordinary
local words obtained by forgetting branch tags.
-/
theorem zipper_lemma_parallel_exists_dist_completion_from_branch_properties_assembly
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {prefixBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace continuationTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchPrefix :
      ∀ i, DTracePrefixTuple (fun A => program i A) (prefixBranches i))
    (hbranchMSC :
      ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletePrefix :
      ∀ A, continuationTrace A ≠ [] →
        ∀ i, LocProg.DTrace (program i A) (prefixBranches i A))
    (hassembly :
      ∀ (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        MappedParallelZipMSCAssembly
          LocProg.DecoratedLetter.toLocal
          LocalCompleteMSC LocalIsMSC
          prefixTrace
          (serialParLocTraceOf
            (fun i A => (hbranchZip i).residual A))
          continuationTrace)
    (horiginal :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append prefixTrace continuationTrace)))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    ∃ UbarOrd,
      DistProg.semantics OrdCompleteMSC
        (fun A => LocProg.par (fun i => program i A))
        UbarOrd ∧
      ∀ A,
        ListPrefix
          (TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              prefixTrace) A)
          (UbarOrd A) := by
  obtain ⟨UbarTagged, hzip⟩ :=
    zipper_lemma_parallel_exists_from_branch_properties_assembly
      (program := program)
      (BranchCompleteMSC := BranchCompleteMSC)
      (BranchIsMSC := BranchIsMSC)
      (LocalCompleteMSC := LocalCompleteMSC)
      (LocalIsMSC := LocalIsMSC)
      (hprefix := hprefix)
      (hguard := hguard)
      (hbranchZipper := hbranchZipper)
      (hbranchPrefix := hbranchPrefix)
      (hbranchMSC := hbranchMSC)
      (hcompletePrefix := hcompletePrefix)
      (hassembly := hassembly)
      (horiginal := horiginal)
  exact parallel_zipPost_to_prefixCompletable
    (program := program)
    (CompleteMSC := OrdCompleteMSC)
    (hcompletedMap := hcompletedMap)
    hzip

/--
Empty-continuation specialization of the previous theorem.

This is the form used for prefix-completability: the old prefix itself is the
MSC prefix being completed, so the Zipper continuation is empty.
-/
theorem zipper_lemma_parallel_prefix_completion_from_branch_properties_assembly
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    {prefixBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {prefixTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    (hprefix : ParLocOf prefixBranches prefixTrace)
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchPrefix :
      ∀ i, DTracePrefixTuple (fun A => program i A) (prefixBranches i))
    (hbranchMSC :
      ∀ i, BranchIsMSC i (prefixBranches i))
    (hassembly :
      ∀ (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        MappedParallelZipMSCAssembly
          LocProg.DecoratedLetter.toLocal
          LocalCompleteMSC LocalIsMSC
          prefixTrace
          (serialParLocTraceOf
            (fun i A => (hbranchZip i).residual A))
          (TraceTuple.empty Lifeline
            (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel))))
    (horiginal :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          prefixTrace))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    ∃ UbarOrd,
      DistProg.semantics OrdCompleteMSC
        (fun A => LocProg.par (fun i => program i A))
        UbarOrd ∧
      ∀ A,
        ListPrefix
          (TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              prefixTrace) A)
          (UbarOrd A) := by
  refine
    zipper_lemma_parallel_exists_dist_completion_from_branch_properties_assembly
      (program := program)
      (BranchCompleteMSC := BranchCompleteMSC)
      (BranchIsMSC := BranchIsMSC)
      (LocalCompleteMSC := LocalCompleteMSC)
      (LocalIsMSC := LocalIsMSC)
      (OrdCompleteMSC := OrdCompleteMSC)
      (hprefix := hprefix)
      (hguard := hguard)
      (hbranchZipper := hbranchZipper)
      (hbranchPrefix := hbranchPrefix)
      (hbranchMSC := hbranchMSC)
      (hcompletePrefix := ?_)
      (hassembly := hassembly)
      (horiginal := ?_)
      (hcompletedMap := hcompletedMap)
  · intro A hcontinuation _i
    simp [TraceTuple.empty] at hcontinuation
  · simpa [TraceTuple.append_empty] using horiginal

/--
Prefix-completability for a local `par` program from the decorated preimage
interface.

The theorem isolates the remaining concrete semantic ingredient: every
ordinary valid prefix must decompose into a decorated tagged `parloc` prefix
with branch MSC facts.  Once that is supplied, the recursive branch Zipper
properties and the parallel MSC assembly theorem yield an ordinary complete
distributed execution extending the prefix.
-/
theorem parallel_prefixCompletable_from_dtrace_preimage_branch_properties_assembly
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hpreimage :
      ∀ U,
        DistProg.prefixSemantics OrdIsMSC
          (fun A => LocProg.par (fun i => program i A)) U →
        ParallelDTracePrefixPreimage program BranchIsMSC LocalIsMSC U)
    (hassembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        MappedParallelZipMSCAssembly
          LocProg.DecoratedLetter.toLocal
          LocalCompleteMSC LocalIsMSC
          prefixTrace
          (serialParLocTraceOf
            (fun i A => (hbranchZip i).residual A))
          (TraceTuple.empty Lifeline
            (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) := by
  intro U hsem
  let hpre := hpreimage U hsem
  rcases
    zipper_lemma_parallel_prefix_completion_from_branch_properties_assembly
      (program := program)
      (BranchCompleteMSC := BranchCompleteMSC)
      (BranchIsMSC := BranchIsMSC)
      (LocalCompleteMSC := LocalCompleteMSC)
      (LocalIsMSC := LocalIsMSC)
      (OrdCompleteMSC := OrdCompleteMSC)
      (prefixBranches := hpre.prefixBranches)
      (prefixTrace := hpre.prefixTrace)
      (hprefix := hpre.parloc)
      (hguard := hguard)
      (hbranchZipper := hbranchZipper)
      (hbranchPrefix := hpre.branch_prefix)
      (hbranchMSC := hpre.branch_msc)
      (hassembly := by
        intro completedBranches hbranchZip
        exact hassembly hpre.parloc hpre.branch_prefix hpre.branch_msc
          completedBranches hbranchZip)
      (horiginal := hpre.local_msc)
      (hcompletedMap := hcompletedMap) with
    ⟨UbarOrd, hcomplete, hprefixOrd⟩
  refine ⟨UbarOrd, hcomplete, ?_⟩
  intro A
  simpa [hpre.erases] using hprefixOrd A

/--
Prefix-completability with the decorated preimage derived from ordinary prefix
traces.

Compared with
`parallel_prefixCompletable_from_dtrace_preimage_branch_properties_assembly`,
this theorem no longer assumes a preimage function.  It constructs the
decorated tagged prefix witness from the ordinary local `par` prefix trace and
keeps only the semantic MSC facts for the extracted branch/local prefixes as
an explicit hypothesis.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_properties_assembly
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hprefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) ∧
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hassembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        MappedParallelZipMSCAssembly
          LocProg.DecoratedLetter.toLocal
          LocalCompleteMSC LocalIsMSC
          prefixTrace
          (serialParLocTraceOf
            (fun i A => (hbranchZip i).residual A))
          (TraceTuple.empty Lifeline
            (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_dtrace_preimage_branch_properties_assembly
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (OrdCompleteMSC := OrdCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hguard := hguard)
    (hbranchZipper := hbranchZipper)
    (hpreimage := by
      intro U hsem
      exact ParallelDTracePrefixPreimage.of_distPrefixTrace
        (program := program)
        (BranchIsMSC := BranchIsMSC)
        (LocalIsMSC := LocalIsMSC)
        hsem.1
        (by
          intro fullBranches fullTrace prefixBranches prefixTrace
            hfullPar hfullComplete hprefixPar hbranchPrefix
          exact hprefixMSC hsem hfullPar hfullComplete hprefixPar
            hbranchPrefix))
    (hassembly := hassembly)
    (hcompletedMap := hcompletedMap)

/--
Deadlock-freedom endgame with only completed-MSC assembly.

For the empty-continuation prefix-completability theorem, the full
replacement-style `MappedParallelZipMSCAssembly` can be derived from the
completed-MSC fact and the standard implication from complete MSCs to MSCs.
-/
theorem parallel_prefixCompletable_from_distPrefix_completed_assembly
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalCompleteImpliesMSC :
      ∀ M, LocalCompleteMSC M → LocalIsMSC M)
    (hprefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) ∧
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A)))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_properties_assembly
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (OrdCompleteMSC := OrdCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hguard := hguard)
    (hbranchZipper := hbranchZipper)
    (hprefixMSC := hprefixMSC)
    (hassembly := by
      intro prefixBranches prefixTrace hprefix hbranchPrefix hbranchMSC
        completedBranches hbranchZip
      exact MappedParallelZipMSCAssembly.of_completed_empty
        (f := LocProg.DecoratedLetter.toLocal)
        (CompleteMSC := LocalCompleteMSC)
        (IsMSC := LocalIsMSC)
        hlocalCompleteImpliesMSC
        (hcompletedAssembly hprefix hbranchPrefix hbranchMSC
          completedBranches hbranchZip))
    (hcompletedMap := hcompletedMap)

/--
Deadlock-freedom endgame with prefix-MSC extraction split into two obligations.

The first obligation proves the extracted tagged local prefix is an MSC.  The
second is the branch-restriction principle for such prefixes.  This mirrors
the paper proof more closely than the bundled `hprefixMSC` hypothesis.
-/
theorem parallel_prefixCompletable_from_distPrefix_local_branch_assembly
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalCompleteImpliesMSC :
      ∀ M, LocalCompleteMSC M → LocalIsMSC M)
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A)))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_completed_assembly
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (OrdCompleteMSC := OrdCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hguard := hguard)
    (hbranchZipper := hbranchZipper)
    (hlocalCompleteImpliesMSC := hlocalCompleteImpliesMSC)
    (hprefixMSC := by
      intro U hsem fullBranches fullTrace prefixBranches prefixTrace
        hfullPar hfullComplete hprefixPar hbranchPrefix
      have hlocal :=
        hlocalPrefixMSC hsem hfullPar hfullComplete hprefixPar
          hbranchPrefix
      exact ⟨hlocal, hbranchMSCOfPrefix hprefixPar hlocal⟩)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := hcompletedMap)

/--
Same endgame as
`parallel_prefixCompletable_from_distPrefix_local_branch_assembly`, using the
standard MSC-concatenation interface to obtain `complete implies MSC`.
-/
theorem parallel_prefixCompletable_from_distPrefix_local_branch_concat_assembly
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalConcat : TraceMSCConcat LocalCompleteMSC LocalIsMSC)
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A)))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_local_branch_assembly
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (OrdCompleteMSC := OrdCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hguard := hguard)
    (hbranchZipper := hbranchZipper)
    (hlocalCompleteImpliesMSC := hlocalConcat.complete_implies_msc)
    (hlocalPrefixMSC := hlocalPrefixMSC)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := hcompletedMap)

/--
Image-semantics version of
`parallel_prefixCompletable_from_distPrefix_local_branch_concat_assembly`.

Here the complete ordinary MSC predicate is exactly the image of the tagged
complete MSC predicate under branch-tag erasure, so no separate completed-map
hypothesis is needed.
-/
theorem parallel_prefixCompletable_from_distPrefix_local_branch_concat_assembly_image
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hguard : ∀ i A, LocProg.GuardDistinct (program i A))
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalConcat : TraceMSCConcat LocalCompleteMSC LocalIsMSC)
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A))))) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_local_branch_concat_assembly
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (OrdCompleteMSC :=
      fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
    (OrdIsMSC := OrdIsMSC)
    (hguard := hguard)
    (hbranchZipper := hbranchZipper)
    (hlocalConcat := hlocalConcat)
    (hlocalPrefixMSC := hlocalPrefixMSC)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := by
      intro M hcomplete
      exact ⟨M, hcomplete, rfl⟩)

/--
Empty-continuation parallel Zipper lemma without a separate `GuardDistinct`
hypothesis.

The guard/prefix-freeness assumption is needed in the general Zipper parallel
case to prove that residuals vanish on lifelines with nonempty continuations.
Here the continuation is empty, so that stability hypothesis is immediate.  The
result is the full `ZipPost` for the branch-tagged local trace produced by the
parallel prefix decomposition; ordinary deadlock-freedom is only a corollary.
-/
theorem parallel_zipper_from_distPrefix_local_branch_completed_assembly_no_guard
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalCompleteImpliesMSC :
      ∀ M, LocalCompleteMSC M → LocalIsMSC M)
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A))))) :
    ∀ U,
      DistProg.prefixSemantics OrdIsMSC
        (fun A => LocProg.par (fun i => program i A)) U →
      ∃ Utag Ubar,
        U = TraceTuple.map Tagged.val Utag ∧
        ZipPost
          (TaggedParLocCompleteLocal
            (fun i A word => LocProg.trace (program i A) word))
          LocalCompleteMSC LocalIsMSC
          Utag Ubar
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (TraceTuple.empty Lifeline
              (Tagged (Fin n)
                (LocProg.DecoratedLetter Lifeline Channel)))) := by
  classical
  intro U hsem
  let hpre : ParallelDTracePrefixPreimage program BranchIsMSC LocalIsMSC U :=
    ParallelDTracePrefixPreimage.of_distPrefixTrace
      (program := program)
      (BranchIsMSC := BranchIsMSC)
      (LocalIsMSC := LocalIsMSC)
      hsem.1
      (by
        intro fullBranches fullTrace prefixBranches prefixTrace
          hfullPar hfullComplete hprefixPar hbranchPrefix
        have hlocal :=
          hlocalPrefixMSC hsem hfullPar hfullComplete hprefixPar
            hbranchPrefix
        exact ⟨hlocal, hbranchMSCOfPrefix hprefixPar hlocal⟩)
  obtain ⟨completedBranches, hbranchZip⟩ :=
    branch_zipPosts_from_dtrace_empty_properties
      (program := program)
      (BranchCompleteMSC := BranchCompleteMSC)
      (BranchIsMSC := BranchIsMSC)
      (prefixBranches := hpre.prefixBranches)
      hbranchZipper hpre.branch_prefix hpre.branch_msc
  let residualTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    serialParLocTraceOf (fun i A => (hbranchZip i).residual A)
  have hassembly :
      MappedParallelZipMSCAssembly
        LocProg.DecoratedLetter.toLocal
        LocalCompleteMSC LocalIsMSC
        hpre.prefixTrace residualTrace
        (TraceTuple.empty Lifeline
          (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel))) :=
    MappedParallelZipMSCAssembly.of_completed_empty
      (f := LocProg.DecoratedLetter.toLocal)
      (CompleteMSC := LocalCompleteMSC)
      (IsMSC := LocalIsMSC)
      hlocalCompleteImpliesMSC
      (hcompletedAssembly hpre.parloc hpre.branch_prefix hpre.branch_msc
        completedBranches hbranchZip)
  have horiginalDecor :
      LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.append hpre.prefixTrace
            (TraceTuple.empty Lifeline
              (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel))))) := by
    simpa [TraceTuple.append_empty] using hpre.local_msc
  have hdecorZip :
      ZipPost
        (TaggedParLocCompleteLocalOf
          (fun i A word => LocProg.DTrace (program i A) word))
        (fun M =>
          LocalCompleteMSC
            (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) M))
        (fun M =>
          LocalIsMSC
            (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal) M))
        hpre.prefixTrace
        (ParLocTaggedTraceOf.append hpre.prefixTrace residualTrace)
        (TraceTuple.empty Lifeline
          (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel))) := by
    refine zipper_parallel_zipPost_from_residuals_of
      (complete := fun i A word => LocProg.DTrace (program i A) word)
      (targetBranches := completedBranches)
      (hprefix := hpre.parloc)
      (hresidual := parlocOf_serial
        (fun i A => (hbranchZip i).residual A))
      (htarget := ?_)
      (hcomplete := ?_)
      (hbranchEmpty := ?_)
      hassembly.completed
      horiginalDecor
      ?_
    · intro i A
      exact ZipPost.residual_eq (hbranchZip i) A
    · intro i A
      exact ((hbranchZip i).prefix_complete A).2.1
    · intro A hcontinuation _i
      simp [TraceTuple.empty] at hcontinuation
    · exact hassembly.extended_of_original horiginalDecor
  have hlocalZip :
      ZipPost
        (TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word))
        LocalCompleteMSC LocalIsMSC
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          hpre.prefixTrace)
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (ParLocTaggedTraceOf.append hpre.prefixTrace residualTrace))
        (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          (TraceTuple.empty Lifeline
            (Tagged (Fin n) (LocProg.DecoratedLetter Lifeline Channel)))) :=
    zipper_parallel_zipPost_dtrace_to_trace
      (program := program)
      (hcompleted := fun h => h)
      (horiginal := fun h => h)
      (hextended := fun h => h)
      hdecorZip
  refine ⟨
    TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
      hpre.prefixTrace,
    TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
      (ParLocTaggedTraceOf.append hpre.prefixTrace residualTrace),
    hpre.erases,
    hlocalZip⟩

/--
Empty-continuation deadlock endgame without a separate `GuardDistinct`
hypothesis.

This is the ordinary prefix-completability corollary of
`parallel_zipper_from_distPrefix_local_branch_completed_assembly_no_guard`.
-/
theorem parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_no_guard
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalCompleteImpliesMSC :
      ∀ M, LocalCompleteMSC M → LocalIsMSC M)
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A)))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) := by
  classical
  intro U hsem
  rcases
      parallel_zipper_from_distPrefix_local_branch_completed_assembly_no_guard
        (program := program)
        (BranchCompleteMSC := BranchCompleteMSC)
        (BranchIsMSC := BranchIsMSC)
        (LocalCompleteMSC := LocalCompleteMSC)
        (LocalIsMSC := LocalIsMSC)
        (OrdIsMSC := OrdIsMSC)
        (hbranchZipper := hbranchZipper)
        (hlocalCompleteImpliesMSC := hlocalCompleteImpliesMSC)
        (hlocalPrefixMSC := hlocalPrefixMSC)
        (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
        (hcompletedAssembly := hcompletedAssembly)
        U hsem with
    ⟨Utag, _UbarTag, hU, hlocalZip⟩
  rcases parallel_zipPost_to_prefixCompletable
      (program := program)
      (CompleteMSC := OrdCompleteMSC)
      (hcompletedMap := hcompletedMap)
      hlocalZip with
    ⟨UbarOrd, hcomplete, hprefixOrd⟩
  refine ⟨UbarOrd, hcomplete, ?_⟩
  intro A
  simpa [hU] using hprefixOrd A

/--
`TraceMSCConcat` wrapper around
`parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_no_guard`.
-/
theorem parallel_prefixCompletable_from_distPrefix_local_branch_concat_assembly_no_guard
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalConcat : TraceMSCConcat LocalCompleteMSC LocalIsMSC)
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A)))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_no_guard
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (OrdCompleteMSC := OrdCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hbranchZipper := hbranchZipper)
    (hlocalCompleteImpliesMSC := hlocalConcat.complete_implies_msc)
    (hlocalPrefixMSC := hlocalPrefixMSC)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := hcompletedMap)

/--
Image-semantics specialization of
`parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_no_guard`.
-/
theorem parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_image_no_guard
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalCompleteImpliesMSC :
      ∀ M, LocalCompleteMSC M → LocalIsMSC M)
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A))))) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_no_guard
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (OrdCompleteMSC :=
      fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
    (OrdIsMSC := OrdIsMSC)
    (hbranchZipper := hbranchZipper)
    (hlocalCompleteImpliesMSC := hlocalCompleteImpliesMSC)
    (hlocalPrefixMSC := hlocalPrefixMSC)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := by
      intro M hcomplete
      exact ⟨M, hcomplete, rfl⟩)

/--
`TraceMSCConcat` wrapper around
`parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_image_no_guard`.
-/
theorem parallel_prefixCompletable_from_distPrefix_local_branch_concat_assembly_image_no_guard
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalConcat : TraceMSCConcat LocalCompleteMSC LocalIsMSC)
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A))))) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_image_no_guard
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (OrdIsMSC := OrdIsMSC)
    (hbranchZipper := hbranchZipper)
    (hlocalCompleteImpliesMSC := hlocalConcat.complete_implies_msc)
    (hlocalPrefixMSC := hlocalPrefixMSC)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)

/--
Direct empty-continuation prefix-completability endgame.

Unlike the ZipPost-based no-guard wrappers above, this theorem constructs the
completed distributed trace directly from the branch completions.  Therefore it
does not need `LocalCompleteMSC → LocalIsMSC`: the completed MSC fact is used
only to inhabit the complete distributed semantics after branch-tag erasure.
-/
theorem parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A)))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) := by
  classical
  intro U hsem
  let hpre : ParallelDTracePrefixPreimage program BranchIsMSC LocalIsMSC U :=
    ParallelDTracePrefixPreimage.of_distPrefixTrace
      (program := program)
      (BranchIsMSC := BranchIsMSC)
      (LocalIsMSC := LocalIsMSC)
      hsem.1
      (by
        intro fullBranches fullTrace prefixBranches prefixTrace
          hfullPar hfullComplete hprefixPar hbranchPrefix
        have hlocal :=
          hlocalPrefixMSC hsem hfullPar hfullComplete hprefixPar
            hbranchPrefix
        exact ⟨hlocal, hbranchMSCOfPrefix hprefixPar hlocal⟩)
  obtain ⟨completedBranches, hbranchZip⟩ :=
    branch_zipPosts_from_dtrace_empty_properties
      (program := program)
      (BranchCompleteMSC := BranchCompleteMSC)
      (BranchIsMSC := BranchIsMSC)
      (prefixBranches := hpre.prefixBranches)
      hbranchZipper hpre.branch_prefix hpre.branch_msc
  let residualBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i A => (hbranchZip i).residual A
  let residualTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    serialParLocTraceOf residualBranches
  let completedTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    ParLocTaggedTraceOf.append hpre.prefixTrace residualTrace
  let completedLocal :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) :=
    TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
      completedTrace
  let completedOrd :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) :=
    TraceTuple.map Tagged.val completedLocal
  have hcompletedLocal : LocalCompleteMSC completedLocal := by
    exact hcompletedAssembly hpre.parloc hpre.branch_prefix hpre.branch_msc
      completedBranches hbranchZip
  refine ⟨completedOrd, ?_, ?_⟩
  · refine ⟨?_, hcompletedMap completedLocal hcompletedLocal⟩
    intro A
    have hparCompleted :
        ParLocOf completedBranches completedTrace :=
      parlocOf_extend_from_residual
        hpre.parloc
        (parlocOf_serial residualBranches)
        (fun i A => ZipPost.residual_eq (hbranchZip i) A)
    have hcompleteDecor :
        TaggedParLocCompleteLocalOf
          (fun i A word => LocProg.DTrace (program i A) word)
          A (completedTrace A) :=
      TaggedParLocCompleteLocalOf.of_components
        hparCompleted
        (fun i => ((hbranchZip i).prefix_complete A).2.1)
    have hcompleteLocal :
        TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word)
          A (completedLocal A) := by
      simpa [completedLocal, TraceTuple.map]
        using
          (LocProg.dtrace_taggedParLocCompleteLocal_to_trace
            (program := program)
            hcompleteDecor)
    simpa [completedOrd, TraceTuple.map] using
      LocProg.taggedParLocCompleteLocal_to_trace_par hcompleteLocal
  · intro A
    have hpTagged :
        ListPrefix (hpre.prefixTrace A) (completedTrace A) :=
      ⟨residualTrace A, rfl⟩
    have hpLocal :
        ListPrefix
          ((hpre.prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal))
          (completedLocal A) := by
      simpa [completedLocal, completedTrace, TraceTuple.map] using
        list_map_preserves_prefix
          (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          hpTagged
    have hpOrd :
        ListPrefix
          (((hpre.prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)).map
              Tagged.val)
          (completedOrd A) := by
      simpa [completedOrd, TraceTuple.map] using
        list_map_preserves_prefix Tagged.val hpLocal
    simpa [hpre.erases, TraceTuple.map] using hpOrd

/--
Image-semantics specialization of
`parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_direct`.

This is the smallest current abstract deadlock-freedom endgame: after the
semantic prefix/branch/assembly obligations are supplied, no
complete-implies-MSC, concat, top-level guard, or completed-map hypothesis remains.
-/
theorem parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_image_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC LocalIsMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hlocalPrefixMSC :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace))
    (hbranchMSCOfPrefix :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        LocalIsMSC
          (TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A))))) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_direct
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (LocalIsMSC := LocalIsMSC)
    (OrdCompleteMSC :=
      fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
    (OrdIsMSC := OrdIsMSC)
    (hbranchZipper := hbranchZipper)
    (hlocalPrefixMSC := hlocalPrefixMSC)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := by
      intro M hcomplete
      exact ⟨M, hcomplete, rfl⟩)

/--
Branch-prefix direct endgame for empty-continuation parallel completion.

Compared with
`parallel_prefixCompletable_from_distPrefix_local_branch_completed_assembly_direct`,
the local tagged prefix-MSC predicate is no longer a hypothesis.  The semantic
boundary is a single branch-prefix extraction obligation: every ordinary valid
`par` prefix has a decorated tagged `parloc` preimage whose branch
restrictions satisfy the recursive branch MSC predicates.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completed_assembly_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A)))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) := by
  classical
  intro U hsem
  let hpre :
      ParallelDTraceBranchPrefixPreimage program BranchIsMSC U :=
    ParallelDTraceBranchPrefixPreimage.of_distPrefixTrace
      (program := program)
      (BranchIsMSC := BranchIsMSC)
      hsem.1
      (by
        intro fullBranches fullTrace prefixBranches prefixTrace
          hfullPar hfullComplete hprefixPar hbranchPrefix
        exact
          hbranchMSCOfPrefix (U := U) hsem hfullPar hfullComplete
            hprefixPar hbranchPrefix)
  obtain ⟨completedBranches, hbranchZip⟩ :=
    branch_zipPosts_from_dtrace_empty_properties
      (program := program)
      (BranchCompleteMSC := BranchCompleteMSC)
      (BranchIsMSC := BranchIsMSC)
      (prefixBranches := hpre.prefixBranches)
      hbranchZipper hpre.branch_prefix hpre.branch_msc
  let residualBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i A => (hbranchZip i).residual A
  let residualTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    serialParLocTraceOf residualBranches
  let completedTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    ParLocTaggedTraceOf.append hpre.prefixTrace residualTrace
  let completedLocal :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) :=
    TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
      completedTrace
  let completedOrd :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) :=
    TraceTuple.map Tagged.val completedLocal
  have hcompletedLocal : LocalCompleteMSC completedLocal := by
    exact hcompletedAssembly hpre.parloc hpre.branch_prefix hpre.branch_msc
      completedBranches hbranchZip
  refine ⟨completedOrd, ?_, ?_⟩
  · refine ⟨?_, hcompletedMap completedLocal hcompletedLocal⟩
    intro A
    have hparCompleted :
        ParLocOf completedBranches completedTrace :=
      parlocOf_extend_from_residual
        hpre.parloc
        (parlocOf_serial residualBranches)
        (fun i A => ZipPost.residual_eq (hbranchZip i) A)
    have hcompleteDecor :
        TaggedParLocCompleteLocalOf
          (fun i A word => LocProg.DTrace (program i A) word)
          A (completedTrace A) :=
      TaggedParLocCompleteLocalOf.of_components
        hparCompleted
        (fun i => ((hbranchZip i).prefix_complete A).2.1)
    have hcompleteLocal :
        TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word)
          A (completedLocal A) := by
      simpa [completedLocal, TraceTuple.map]
        using
          (LocProg.dtrace_taggedParLocCompleteLocal_to_trace
            (program := program)
            hcompleteDecor)
    simpa [completedOrd, TraceTuple.map] using
      LocProg.taggedParLocCompleteLocal_to_trace_par hcompleteLocal
  · intro A
    have hpTagged :
        ListPrefix (hpre.prefixTrace A) (completedTrace A) :=
      ⟨residualTrace A, rfl⟩
    have hpLocal :
        ListPrefix
          ((hpre.prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal))
          (completedLocal A) := by
      simpa [completedLocal, completedTrace, TraceTuple.map] using
        list_map_preserves_prefix
          (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          hpTagged
    have hpOrd :
        ListPrefix
          (((hpre.prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)).map
              Tagged.val)
          (completedOrd A) := by
      simpa [completedOrd, TraceTuple.map] using
        list_map_preserves_prefix Tagged.val hpLocal
    simpa [hpre.erases, TraceTuple.map] using hpOrd

/--
Image-semantics specialization of the branch-prefix direct endgame.

This is the current smallest abstract deadlock-freedom wrapper: the remaining
semantic ingredients are recursive branch completion, branch-prefix MSC
extraction, and completed tagged-assembly.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completed_assembly_image_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchZipper :
      ∀ i, DTraceEmptyZipperProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A))))) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completed_assembly_direct
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC :=
      fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
    (OrdIsMSC := OrdIsMSC)
    (hbranchZipper := hbranchZipper)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := by
      intro M hcomplete
      exact ⟨M, hcomplete, rfl⟩)

/--
Branch-prefix direct endgame with recursive branch Zipper properties generated
from complete decorated traces.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completed_assembly_direct_from_complete
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompleteOfDTrace :
      ∀ i U,
        (∀ A, LocProg.DTrace (program i A) (U A)) →
        BranchCompleteMSC i U)
    (hbranchComplete_implies_msc :
      ∀ i U, BranchCompleteMSC i U → BranchIsMSC i U)
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A)))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completed_assembly_direct
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC := OrdCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hbranchZipper := by
      intro i
      exact dtrace_empty_zipper_property_from_complete
        (program := fun A => program i A)
        (CompleteMSC := BranchCompleteMSC i)
        (IsMSC := BranchIsMSC i)
        (hbranchCompleteOfDTrace i)
        (hbranchComplete_implies_msc i))
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := hcompletedMap)

/--
Image-semantics specialization with branch Zipper properties generated from
complete decorated traces.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completed_assembly_image_direct_from_complete
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompleteOfDTrace :
      ∀ i U,
        (∀ A, LocProg.DTrace (program i A) (U A)) →
        BranchCompleteMSC i U)
    (hbranchComplete_implies_msc :
      ∀ i U, BranchCompleteMSC i U → BranchIsMSC i U)
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (hbranchZip :
          ∀ i,
          ZipPost (fun A word => LocProg.DTrace (program i A) word)
            (BranchCompleteMSC i) (BranchIsMSC i)
            (prefixBranches i) (completedBranches i)
            (TraceTuple.empty Lifeline
              (LocProg.DecoratedLetter Lifeline Channel))),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => (hbranchZip i).residual A))))) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completed_assembly_direct_from_complete
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC :=
      fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompleteOfDTrace := hbranchCompleteOfDTrace)
    (hbranchComplete_implies_msc := hbranchComplete_implies_msc)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := by
      intro M hcomplete
      exact ⟨M, hcomplete, rfl⟩)

/--
Direct branch-completion endgame.

This variant avoids branch `ZipPost`s entirely.  Each branch only has to supply
a complete decorated trace extending the branch prefix; the residuals used in
the assembled `parloc` trace are chosen from those prefix witnesses.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_assembly_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompletion :
      ∀ i, DTracePrefixCompletionProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (_hbranchTrace :
          ∀ i A, LocProg.DTrace (program i A)
            (completedBranches i A))
        (_hbranchCompleteMSC :
          ∀ i, BranchCompleteMSC i (completedBranches i))
        (hbranchExt :
          ∀ i A, ListPrefix (prefixBranches i A)
            (completedBranches i A)),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => Classical.choose (hbranchExt i A))))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) := by
  classical
  intro U hsem
  let hpre :
      ParallelDTraceBranchPrefixPreimage program BranchIsMSC U :=
    ParallelDTraceBranchPrefixPreimage.of_distPrefixTrace
      (program := program)
      (BranchIsMSC := BranchIsMSC)
      hsem.1
      (by
        intro fullBranches fullTrace prefixBranches prefixTrace
          hfullPar hfullComplete hprefixPar hbranchPrefix
        exact
          hbranchMSCOfPrefix (U := U) hsem hfullPar hfullComplete
            hprefixPar hbranchPrefix)
  have hcompletion :
      ∀ i,
        ∃ completed :
          TraceTuple Lifeline
            (LocProg.DecoratedLetter Lifeline Channel),
          (∀ A, LocProg.DTrace (program i A) (completed A)) ∧
          BranchCompleteMSC i completed ∧
          ∀ A, ListPrefix (hpre.prefixBranches i A) (completed A) := by
    intro i
    exact hbranchCompletion i (hpre.prefixBranches i)
      (hpre.branch_prefix i) (hpre.branch_msc i)
  let completedBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i => Classical.choose (hcompletion i)
  have hbranchTrace :
      ∀ i A, LocProg.DTrace (program i A)
        (completedBranches i A) := by
    intro i A
    exact (Classical.choose_spec (hcompletion i)).1 A
  have hbranchCompleteMSC :
      ∀ i, BranchCompleteMSC i (completedBranches i) := by
    intro i
    exact (Classical.choose_spec (hcompletion i)).2.1
  have hbranchExt :
      ∀ i A, ListPrefix (hpre.prefixBranches i A)
        (completedBranches i A) := by
    intro i A
    exact (Classical.choose_spec (hcompletion i)).2.2 A
  let residualBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i A => Classical.choose (hbranchExt i A)
  have hresidualEq :
      ∀ i A, completedBranches i A =
        hpre.prefixBranches i A ++ residualBranches i A := by
    intro i A
    exact (Classical.choose_spec (hbranchExt i A)).symm
  let residualTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    serialParLocTraceOf residualBranches
  let completedTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    ParLocTaggedTraceOf.append hpre.prefixTrace residualTrace
  let completedLocal :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) :=
    TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
      completedTrace
  let completedOrd :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) :=
    TraceTuple.map Tagged.val completedLocal
  have hcompletedLocal : LocalCompleteMSC completedLocal := by
    exact hcompletedAssembly hpre.parloc hpre.branch_prefix hpre.branch_msc
      completedBranches hbranchTrace hbranchCompleteMSC hbranchExt
  refine ⟨completedOrd, ?_, ?_⟩
  · refine ⟨?_, hcompletedMap completedLocal hcompletedLocal⟩
    intro A
    have hparCompleted :
        ParLocOf completedBranches completedTrace :=
      parlocOf_extend_from_residual
        hpre.parloc
        (parlocOf_serial residualBranches)
        hresidualEq
    have hcompleteDecor :
        TaggedParLocCompleteLocalOf
          (fun i A word => LocProg.DTrace (program i A) word)
          A (completedTrace A) :=
      TaggedParLocCompleteLocalOf.of_components
        hparCompleted
        (fun i => hbranchTrace i A)
    have hcompleteLocal :
        TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word)
          A (completedLocal A) := by
      simpa [completedLocal, TraceTuple.map]
        using
          (LocProg.dtrace_taggedParLocCompleteLocal_to_trace
            (program := program)
            hcompleteDecor)
    simpa [completedOrd, TraceTuple.map] using
      LocProg.taggedParLocCompleteLocal_to_trace_par hcompleteLocal
  · intro A
    have hpTagged :
        ListPrefix (hpre.prefixTrace A) (completedTrace A) :=
      ⟨residualTrace A, rfl⟩
    have hpLocal :
        ListPrefix
          ((hpre.prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal))
          (completedLocal A) := by
      simpa [completedLocal, completedTrace, TraceTuple.map] using
        list_map_preserves_prefix
          (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          hpTagged
    have hpOrd :
        ListPrefix
          (((hpre.prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)).map
              Tagged.val)
          (completedOrd A) := by
      simpa [completedOrd, TraceTuple.map] using
        list_map_preserves_prefix Tagged.val hpLocal
    simpa [hpre.erases, TraceTuple.map] using hpOrd

/--
Erasure-aware direct branch-completion endgame.

This variant exposes the equality between the ordinary prefix `U` and the
chosen decorated tagged prefix.  The equality is needed by concrete
word-level branch-restriction proofs: without it, the enclosing `WordIsMSC U`
witness cannot be related to the branch-prefix events extracted from the
tagged `parloc` witness.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_assembly_erasure_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompletion :
      ∀ i, DTracePrefixCompletionProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf prefixBranches prefixTrace →
        (∀ i, DTracePrefixTuple (fun A => program i A)
          (prefixBranches i)) →
        U =
          TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
              prefixTrace) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (_hbranchTrace :
          ∀ i A, LocProg.DTrace (program i A)
            (completedBranches i A))
        (_hbranchCompleteMSC :
          ∀ i, BranchCompleteMSC i (completedBranches i))
        (hbranchExt :
          ∀ i A, ListPrefix (prefixBranches i A)
            (completedBranches i A)),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => Classical.choose (hbranchExt i A))))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) := by
  classical
  intro U hsem
  let hpreTrue :
      ParallelDTraceBranchPrefixPreimage
        (program := program) (fun _ _ => True) U :=
    ParallelDTraceBranchPrefixPreimage.of_distPrefixTrace
      (program := program)
      (BranchIsMSC := fun _ _ => True)
      hsem.1
      (by
        intro fullBranches fullTrace prefixBranches prefixTrace
          hfullPar hfullComplete hprefixPar hbranchPrefix i
        trivial)
  let hpre :
      ParallelDTraceBranchPrefixPreimage program BranchIsMSC U :=
    { prefixBranches := hpreTrue.prefixBranches
      prefixTrace := hpreTrue.prefixTrace
      parloc := hpreTrue.parloc
      branch_prefix := hpreTrue.branch_prefix
      branch_msc :=
        hbranchMSCOfPrefix hsem hpreTrue.parloc hpreTrue.branch_prefix
          hpreTrue.erases
      erases := hpreTrue.erases }
  have hcompletion :
      ∀ i,
        ∃ completed :
          TraceTuple Lifeline
            (LocProg.DecoratedLetter Lifeline Channel),
          (∀ A, LocProg.DTrace (program i A) (completed A)) ∧
          BranchCompleteMSC i completed ∧
          ∀ A, ListPrefix (hpre.prefixBranches i A) (completed A) := by
    intro i
    exact hbranchCompletion i (hpre.prefixBranches i)
      (hpre.branch_prefix i) (hpre.branch_msc i)
  let completedBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i => Classical.choose (hcompletion i)
  have hbranchTrace :
      ∀ i A, LocProg.DTrace (program i A)
        (completedBranches i A) := by
    intro i A
    exact (Classical.choose_spec (hcompletion i)).1 A
  have hbranchCompleteMSC :
      ∀ i, BranchCompleteMSC i (completedBranches i) := by
    intro i
    exact (Classical.choose_spec (hcompletion i)).2.1
  have hbranchExt :
      ∀ i A, ListPrefix (hpre.prefixBranches i A)
        (completedBranches i A) := by
    intro i A
    exact (Classical.choose_spec (hcompletion i)).2.2 A
  let residualBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    fun i A => Classical.choose (hbranchExt i A)
  have hresidualEq :
      ∀ i A, completedBranches i A =
        hpre.prefixBranches i A ++ residualBranches i A := by
    intro i A
    exact (Classical.choose_spec (hbranchExt i A)).symm
  let residualTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    serialParLocTraceOf residualBranches
  let completedTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) :=
    ParLocTaggedTraceOf.append hpre.prefixTrace residualTrace
  let completedLocal :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) :=
    TraceTuple.map (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
      completedTrace
  let completedOrd :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) :=
    TraceTuple.map Tagged.val completedLocal
  have hcompletedLocal : LocalCompleteMSC completedLocal := by
    exact hcompletedAssembly hpre.parloc hpre.branch_prefix hpre.branch_msc
      completedBranches hbranchTrace hbranchCompleteMSC hbranchExt
  refine ⟨completedOrd, ?_, ?_⟩
  · refine ⟨?_, hcompletedMap completedLocal hcompletedLocal⟩
    intro A
    have hparCompleted :
        ParLocOf completedBranches completedTrace :=
      parlocOf_extend_from_residual
        hpre.parloc
        (parlocOf_serial residualBranches)
        hresidualEq
    have hcompleteDecor :
        TaggedParLocCompleteLocalOf
          (fun i A word => LocProg.DTrace (program i A) word)
          A (completedTrace A) :=
      TaggedParLocCompleteLocalOf.of_components
        hparCompleted
        (fun i => hbranchTrace i A)
    have hcompleteLocal :
        TaggedParLocCompleteLocal
          (fun i A word => LocProg.trace (program i A) word)
          A (completedLocal A) := by
      simpa [completedLocal, TraceTuple.map]
        using
          (LocProg.dtrace_taggedParLocCompleteLocal_to_trace
            (program := program)
            hcompleteDecor)
    simpa [completedOrd, TraceTuple.map] using
      LocProg.taggedParLocCompleteLocal_to_trace_par hcompleteLocal
  · intro A
    have hpTagged :
        ListPrefix (hpre.prefixTrace A) (completedTrace A) :=
      ⟨residualTrace A, rfl⟩
    have hpLocal :
        ListPrefix
          ((hpre.prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal))
          (completedLocal A) := by
      simpa [completedLocal, completedTrace, TraceTuple.map] using
        list_map_preserves_prefix
          (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
          hpTagged
    have hpOrd :
        ListPrefix
          (((hpre.prefixTrace A).map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)).map
              Tagged.val)
          (completedOrd A) := by
      simpa [completedOrd, TraceTuple.map] using
        list_map_preserves_prefix Tagged.val hpLocal
    simpa [hpre.erases, TraceTuple.map] using hpOrd

/-- Image-semantics specialization of the direct branch-completion endgame. -/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_assembly_image_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompletion :
      ∀ i, DTracePrefixCompletionProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (_hbranchTrace :
          ∀ i A, LocProg.DTrace (program i A)
            (completedBranches i A))
        (_hbranchCompleteMSC :
          ∀ i, BranchCompleteMSC i (completedBranches i))
        (hbranchExt :
          ∀ i A, ListPrefix (prefixBranches i A)
            (completedBranches i A)),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => Classical.choose (hbranchExt i A)))))) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_assembly_direct
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC :=
      fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompletion := hbranchCompletion)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := by
      intro M hcomplete
      exact ⟨M, hcomplete, rfl⟩)

/--
Branch-completion endgame where branch completion is generated from complete
decorated traces.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_assembly_direct_from_complete
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompleteOfDTrace :
      ∀ i U,
        (∀ A, LocProg.DTrace (program i A) (U A)) →
        BranchCompleteMSC i U)
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (_hbranchTrace :
          ∀ i A, LocProg.DTrace (program i A)
            (completedBranches i A))
        (_hbranchCompleteMSC :
          ∀ i, BranchCompleteMSC i (completedBranches i))
        (hbranchExt :
          ∀ i A, ListPrefix (prefixBranches i A)
            (completedBranches i A)),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => Classical.choose (hbranchExt i A))))))
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_assembly_direct
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC := OrdCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompletion := by
      intro i
      exact dtrace_prefix_completion_property_from_complete
        (program := fun A => program i A)
        (CompleteMSC := BranchCompleteMSC i)
        (IsMSC := BranchIsMSC i)
        (hbranchCompleteOfDTrace i))
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := hcompletedMap)

/--
Image-semantics branch-completion endgame where branch completion is generated
from complete decorated traces.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_assembly_image_direct_from_complete
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompleteOfDTrace :
      ∀ i U,
        (∀ A, LocProg.DTrace (program i A) (U A)) →
        BranchCompleteMSC i U)
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      ∀ {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        (_hprefix : ParLocOf prefixBranches prefixTrace)
        (_hbranchPrefix :
          ∀ i, DTracePrefixTuple (fun A => program i A)
            (prefixBranches i))
        (_hbranchMSC : ∀ i, BranchIsMSC i (prefixBranches i))
        (completedBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel))
        (_hbranchTrace :
          ∀ i A, LocProg.DTrace (program i A)
            (completedBranches i A))
        (_hbranchCompleteMSC :
          ∀ i, BranchCompleteMSC i (completedBranches i))
        (hbranchExt :
          ∀ i A, ListPrefix (prefixBranches i A)
            (completedBranches i A)),
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf
                (fun i A => Classical.choose (hbranchExt i A)))))) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_assembly_direct_from_complete
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC :=
      fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompleteOfDTrace := hbranchCompleteOfDTrace)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := by
      intro M hcomplete
      exact ⟨M, hcomplete, rfl⟩)

/--
Completed local assembly for a tagged `parloc` trace.

This is the semantic assembly obligation needed by the direct
branch-completion endgame: once the branches are complete decorated traces and
their branch MSC predicates hold, any tagged `parloc` assembly of them is a
complete local MSC after forgetting the recursive decoration wrapper.
-/
def CompletedParLocAssembly {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    (program : Fin n → Lifeline → LocProg Lifeline Channel)
    (BranchCompleteMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop)
    (LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop) :
    Prop :=
  ∀ {completedBranches :
      ParLocBranchesOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)}
    {completedTrace :
      ParLocTaggedTraceOf n Lifeline
        (LocProg.DecoratedLetter Lifeline Channel)},
    ParLocOf completedBranches completedTrace →
    (∀ i A, LocProg.DTrace (program i A) (completedBranches i A)) →
    (∀ i, BranchCompleteMSC i (completedBranches i)) →
    LocalCompleteMSC
      (TraceTuple.map
        (Tagged.mapVal LocProg.DecoratedLetter.toLocal)
        completedTrace)

theorem CompletedParLocAssembly.of_dtrace_complete {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    (hlocalComplete :
      ∀ trace,
        TaggedParLocCompleteOf
          (fun i A word => LocProg.DTrace (program i A) word) trace →
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal) trace)) :
    CompletedParLocAssembly program BranchCompleteMSC LocalCompleteMSC := by
  intro completedBranches completedTrace hpar hbranchTrace _hbranchCompleteMSC
  exact hlocalComplete completedTrace
    (TaggedParLocCompleteOf.of_components hpar hbranchTrace)

theorem CompletedParLocAssembly.of_trace_complete {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    (hlocalComplete :
      ∀ trace,
        TaggedParLocComplete
          (fun i A word => LocProg.trace (program i A) word) trace →
        LocalCompleteMSC trace) :
    CompletedParLocAssembly program BranchCompleteMSC LocalCompleteMSC := by
  intro completedBranches completedTrace hpar hbranchTrace _hbranchCompleteMSC
  exact hlocalComplete
    (TraceTuple.map
      (Tagged.mapVal LocProg.DecoratedLetter.toLocal) completedTrace)
    (LocProg.dtrace_taggedParLocComplete_to_trace
      (program := program)
      (TaggedParLocCompleteOf.of_components hpar hbranchTrace))

/--
Direct branch-completion endgame with prefix-independent `parloc` assembly.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_parloc_assembly_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompletion :
      ∀ i, DTracePrefixCompletionProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      CompletedParLocAssembly program BranchCompleteMSC LocalCompleteMSC)
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_assembly_direct
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC := OrdCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompletion := hbranchCompletion)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := by
      intro prefixBranches prefixTrace hprefix _hbranchPrefix _hbranchMSC
        completedBranches hbranchTrace hbranchCompleteMSC hbranchExt
      let residualBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel) :=
        fun i A => Classical.choose (hbranchExt i A)
      have hresidualEq :
          ∀ i A, completedBranches i A =
            prefixBranches i A ++ residualBranches i A := by
        intro i A
        exact (Classical.choose_spec (hbranchExt i A)).symm
      have hparCompleted :
          ParLocOf completedBranches
            (ParLocTaggedTraceOf.append prefixTrace
              (serialParLocTraceOf residualBranches)) :=
        parlocOf_extend_from_residual
          hprefix
          (parlocOf_serial residualBranches)
          hresidualEq
      exact hcompletedAssembly hparCompleted hbranchTrace
        hbranchCompleteMSC)
    (hcompletedMap := hcompletedMap)

/--
Image-semantics specialization with prefix-independent `parloc` assembly.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_parloc_assembly_image_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompletion :
      ∀ i, DTracePrefixCompletionProperty
        (fun A => program i A)
        (BranchCompleteMSC i) (BranchIsMSC i))
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      CompletedParLocAssembly program BranchCompleteMSC LocalCompleteMSC) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_parloc_assembly_direct
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC :=
      fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompletion := hbranchCompletion)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := by
      intro M hcomplete
      exact ⟨M, hcomplete, rfl⟩)

/--
Prefix-independent `parloc` assembly, with branch completion generated from
complete decorated traces.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_parloc_assembly_direct_from_complete
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdCompleteMSC OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompleteOfDTrace :
      ∀ i U,
        (∀ A, LocProg.DTrace (program i A) (U A)) →
        BranchCompleteMSC i U)
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      CompletedParLocAssembly program BranchCompleteMSC LocalCompleteMSC)
    (hcompletedMap :
      ∀ M, LocalCompleteMSC M →
        OrdCompleteMSC (TraceTuple.map Tagged.val M)) :
    PrefixCompletable OrdCompleteMSC OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_parloc_assembly_direct
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC := OrdCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompletion := by
      intro i
      exact dtrace_prefix_completion_property_from_complete
        (program := fun A => program i A)
        (CompleteMSC := BranchCompleteMSC i)
        (IsMSC := BranchIsMSC i)
        (hbranchCompleteOfDTrace i))
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := hcompletedMap)

/--
Image-semantics specialization of the prefix-independent `parloc` assembly
wrapper, with branch completion generated from complete decorated traces.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_parloc_assembly_image_direct_from_complete
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompleteOfDTrace :
      ∀ i U,
        (∀ A, LocProg.DTrace (program i A) (U A)) →
        BranchCompleteMSC i U)
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hcompletedAssembly :
      CompletedParLocAssembly program BranchCompleteMSC LocalCompleteMSC) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_parloc_assembly_direct_from_complete
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdCompleteMSC :=
      fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompleteOfDTrace := hbranchCompleteOfDTrace)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly := hcompletedAssembly)
    (hcompletedMap := by
      intro M hcomplete
      exact ⟨M, hcomplete, rfl⟩)

/--
Strongest current image endgame using decorated tagged `parloc` completeness
as the local completed-assembly hypothesis.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_dtrace_complete_image_direct_from_complete
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompleteOfDTrace :
      ∀ i U,
        (∀ A, LocProg.DTrace (program i A) (U A)) →
        BranchCompleteMSC i U)
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hlocalComplete :
      ∀ trace,
        TaggedParLocCompleteOf
          (fun i A word => LocProg.DTrace (program i A) word) trace →
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal) trace)) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_parloc_assembly_image_direct_from_complete
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompleteOfDTrace := hbranchCompleteOfDTrace)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly :=
      CompletedParLocAssembly.of_dtrace_complete
        (program := program)
        (BranchCompleteMSC := BranchCompleteMSC)
        (LocalCompleteMSC := LocalCompleteMSC)
        hlocalComplete)

/--
Image endgame using ordinary tagged `parloc` trace completeness as the local
completed-assembly hypothesis.
-/
theorem parallel_prefixCompletable_from_distPrefix_branch_completion_trace_complete_image_direct_from_complete
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {BranchCompleteMSC BranchIsMSC :
      Fin n → TraceTuple Lifeline
        (LocProg.DecoratedLetter Lifeline Channel) → Prop}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hbranchCompleteOfDTrace :
      ∀ i U,
        (∀ A, LocProg.DTrace (program i A) (U A)) →
        BranchCompleteMSC i U)
    (hbranchMSCOfPrefix :
      ∀ {U : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
        (_hsem :
          DistProg.prefixSemantics OrdIsMSC
            (fun A => LocProg.par (fun i => program i A)) U)
        {fullBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {fullTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixBranches :
          ParLocBranchesOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)}
        {prefixTrace :
          ParLocTaggedTraceOf n Lifeline
            (LocProg.DecoratedLetter Lifeline Channel)},
        ParLocOf fullBranches fullTrace →
        (∀ i A, LocProg.DTrace (program i A) (fullBranches i A)) →
        ParLocOf prefixBranches prefixTrace →
        (∀ i A, ListPrefix (prefixBranches i A) (fullBranches i A)) →
        ∀ i, BranchIsMSC i (prefixBranches i))
    (hlocalComplete :
      ∀ trace,
        TaggedParLocComplete
          (fun i A word => LocProg.trace (program i A) word) trace →
        LocalCompleteMSC trace) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_parloc_assembly_image_direct_from_complete
    (program := program)
    (BranchCompleteMSC := BranchCompleteMSC)
    (BranchIsMSC := BranchIsMSC)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompleteOfDTrace := hbranchCompleteOfDTrace)
    (hbranchMSCOfPrefix := hbranchMSCOfPrefix)
    (hcompletedAssembly :=
      CompletedParLocAssembly.of_trace_complete
        (program := program)
        (BranchCompleteMSC := BranchCompleteMSC)
        (LocalCompleteMSC := LocalCompleteMSC)
        hlocalComplete)

/--
Closed trace-level direct image endgame.

This corollary instantiates the branch MSC predicates by `True`.  Hence no
branch-prefix MSC extraction is needed: any ordinary syntactic prefix of a
local `par` program is completed by completing each decorated branch prefix and
assembling the resulting tagged `parloc` trace.  The remaining hypothesis says
that such completed decorated `parloc` traces are accepted by the chosen local
complete predicate.
-/
theorem parallel_prefixCompletable_from_distPrefix_dtrace_complete_image_direct
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop}
    (hlocalComplete :
      ∀ trace,
        TaggedParLocCompleteOf
          (fun i A word => LocProg.DTrace (program i A) word) trace →
        LocalCompleteMSC
          (TraceTuple.map
            (Tagged.mapVal LocProg.DecoratedLetter.toLocal) trace)) :
    PrefixCompletable
      (fun M =>
        ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  parallel_prefixCompletable_from_distPrefix_branch_completion_dtrace_complete_image_direct_from_complete
    (program := program)
    (BranchCompleteMSC :=
      fun _ _ => True)
    (BranchIsMSC :=
      fun _ _ => True)
    (LocalCompleteMSC := LocalCompleteMSC)
    (OrdIsMSC := OrdIsMSC)
    (hbranchCompleteOfDTrace := by
      intro _ _ _htrace
      trivial)
    (hbranchMSCOfPrefix := by
      intro U hsem fullBranches fullTrace prefixBranches prefixTrace
        hfullPar hfullComplete hprefixPar hbranchPrefix i
      trivial)
    (hlocalComplete := hlocalComplete)

/--
Fully closed syntactic image version of the direct endgame.

The complete predicate is exactly the image, after decoration and branch-tag
erasure, of completed decorated tagged `parloc` traces.
-/
theorem parallel_prefixCompletable_from_distPrefix_dtrace_complete_image
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop} :
    PrefixCompletable
      (fun M =>
        ∃ T,
          TaggedParLocCompleteOf
            (fun i A word => LocProg.DTrace (program i A) word) T ∧
          TraceTuple.map Tagged.val
            (TraceTuple.map
              (Tagged.mapVal LocProg.DecoratedLetter.toLocal) T) = M)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) := by
  classical
  let LocalCompleteMSC :
      TraceTuple Lifeline
        (Tagged (Fin n) (LocalLetter Lifeline Channel)) → Prop :=
    fun T =>
      ∃ D,
        TaggedParLocCompleteOf
          (fun i A word => LocProg.DTrace (program i A) word) D ∧
        TraceTuple.map
          (Tagged.mapVal LocProg.DecoratedLetter.toLocal) D = T
  have hprefix :
      PrefixCompletable
        (fun M =>
          ∃ T, LocalCompleteMSC T ∧ TraceTuple.map Tagged.val T = M)
        OrdIsMSC
        (fun A => LocProg.par (fun i => program i A)) :=
    parallel_prefixCompletable_from_distPrefix_dtrace_complete_image_direct
      (program := program)
      (LocalCompleteMSC := LocalCompleteMSC)
      (OrdIsMSC := OrdIsMSC)
      (by
        intro trace hcomplete
        exact ⟨trace, hcomplete, rfl⟩)
  intro U hsem
  rcases hprefix U hsem with ⟨Ubar, hsemComplete, hpref⟩
  rcases hsemComplete with ⟨htrace, hcomplete⟩
  rcases hcomplete with ⟨T, hlocal, hT⟩
  rcases hlocal with ⟨D, hDcomplete, hDmap⟩
  refine ⟨Ubar, ?_, hpref⟩
  refine ⟨htrace, ?_⟩
  refine ⟨D, hDcomplete, ?_⟩
  calc
    TraceTuple.map Tagged.val
        (TraceTuple.map
          (Tagged.mapVal LocProg.DecoratedLetter.toLocal) D)
        = TraceTuple.map Tagged.val T := by
          rw [hDmap]
    _ = Ubar := hT

/--
Complete local-parallel traces after decoration and branch-tag erasure.

This is the trace-level complete predicate closed by the current parallel
Zipper development.  A later concrete word-to-MSC layer can map this predicate
to the paper's complete-MSC predicate.
-/
def DTraceParCompleteImage {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    (program : Fin n → Lifeline → LocProg Lifeline Channel)
    (M : TraceTuple Lifeline (LocalLetter Lifeline Channel)) : Prop :=
  ∃ T,
    TaggedParLocCompleteOf
      (fun i A word => LocProg.DTrace (program i A) word) T ∧
    TraceTuple.map Tagged.val
      (TraceTuple.map
        (Tagged.mapVal LocProg.DecoratedLetter.toLocal) T) = M

/--
Closed trace-level deadlock-freedom theorem for a local parallel region.

This is just the preceding prefix-completability theorem under the name used by
the paper-level consequence: the complete executions are exactly the image,
after decoration and branch-tag erasure, of completed decorated tagged `parloc`
traces.
-/
theorem parallel_deadlock_free_dtrace_complete_image
    {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {OrdIsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) → Prop} :
    PrefixCompletable
      (DTraceParCompleteImage program)
      OrdIsMSC
      (fun A => LocProg.par (fun i => program i A)) :=
  by
    unfold DTraceParCompleteImage
    exact
      parallel_prefixCompletable_from_distPrefix_dtrace_complete_image
        (program := program)
        (OrdIsMSC := OrdIsMSC)

/--
The word-MSC bridge is not derivable from local trace completeness alone.

A single completed local send, with no matching receive anywhere, is a
completed decorated `parloc` image but not a complete word-level MSC.
-/
theorem dtraceParCompleteImage_not_word_msc_closed :
    ∃ (program : Fin 1 → Unit → LocProg Unit Unit)
      (M : TraceTuple Unit (LocalLetter Unit Unit)),
      DTraceParCompleteImage program M ∧ ¬ WordIsCompleteMSC M := by
  let sendLetter : LocalLetter Unit Unit := LocalLetter.send () ()
  let program : Fin 1 → Unit → LocProg Unit Unit :=
    fun _ _ => LocProg.letter sendLetter
  let M : TraceTuple Unit (LocalLetter Unit Unit) :=
    fun _ => [sendLetter]
  refine ⟨program, M, ?_, ?_⟩
  · let decorated : LocProg.DecoratedLetter Unit Unit :=
      LocProg.DecoratedLetter.atom sendLetter
    let taggedTrace :
        ParLocTaggedTraceOf 1 Unit (LocProg.DecoratedLetter Unit Unit) :=
      fun _ => [{ branch := (0 : Fin 1), val := decorated }]
    refine ⟨taggedTrace, ?_, ?_⟩
    · refine TaggedParLocCompleteOf.of_components
        (branches := fun _ _ => [decorated])
        (trace := taggedTrace)
        ?_ ?_
      · intro i A
        cases A
        have hi : i = (0 : Fin 1) := by
          apply Fin.ext
          exact Nat.lt_one_iff.mp i.isLt
        rw [hi]
        rfl
      · intro i A
        cases A
        have hi : i = (0 : Fin 1) := by
          apply Fin.ext
          exact Nat.lt_one_iff.mp i.isLt
        rw [hi]
        exact LocProg.DTrace.letter sendLetter
    · funext A
      cases A
      rfl
  · intro hcomplete
    rcases hcomplete with ⟨D, hDcomplete⟩
    let sendEvent : WordEvent M :=
      { owner := ()
        index := 0
        inBounds := by simp [M] }
    have hsend : MSC.IsSend D.toMSC sendEvent := by
      refine ⟨(), (), ?_⟩
      rfl
    rcases hDcomplete sendEvent hsend with ⟨recvEvent, hmsg⟩
    rcases D.msg_send_recv hmsg with
      ⟨A, B, ch, _hsOwner, _hrOwner, _hsKind, hrKind⟩
    cases A
    cases B
    cases ch
    cases recvEvent with
    | mk owner index inBounds =>
        cases owner
        have hindex : index = 0 := by
          exact Nat.lt_one_iff.mp (by simpa [M] using inBounds)
        subst index
        simp [WordEvent.kind, WordEvent.letter, M, sendLetter] at hrKind

/-- Generic tagged prefix theorem for local parallel traces. -/
theorem zipper_parallel_case_tagged_of {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {branches : ParLocBranchesOf n Lifeline α}
    {full pref : ParLocTaggedTraceOf n Lifeline α}
    (hfull : ParLocOf branches full)
    (hpref : TracePrefixOf pref full) :
    ∃ branchPrefix : ParLocBranchesOf n Lifeline α,
      ParLocOf branchPrefix pref ∧
      ∀ i A, ListPrefix (branchPrefix i A) (branches i A) :=
  parlocOf_prefix_decompose hfull hpref

/-- Generic observable prefix theorem for local parallel traces. -/
theorem zipper_parallel_case_of {n : Nat}
    {Lifeline : Type u} {α : Type v}
    {branches : ParLocBranchesOf n Lifeline α}
    {tagged_full tagged_pref : ParLocTaggedTraceOf n Lifeline α}
    (hfull : ParLocOf branches tagged_full)
    (hpref : TracePrefixOf tagged_pref tagged_full) :
    ∃ branchPrefix : ParLocBranchesOf n Lifeline α,
      UntaggedParLocOf branchPrefix (forgetBranchTagOf tagged_pref) ∧
      ∀ i A, ListPrefix (branchPrefix i A) (branches i A) := by
  obtain ⟨branchPrefix, hprefixParLoc, hbranchPrefix⟩ :=
    parlocOf_prefix_decompose hfull hpref
  exact ⟨branchPrefix, parlocOf_forgetBranchTag_witness hprefixParLoc,
    hbranchPrefix⟩

/--
Parallel Zipper prefix theorem, tagged version.

If `full` is a tagged `ParLoc` execution and `pref` is a pointwise prefix of
`full`, then each branch has a valid prefix contribution, and those
contributions are pointwise prefixes of the full branch traces.

This is the trace-prefix property used by the parallel case of the Zipper
lemma.  The full four-condition postcondition assembly is formalized above as
`zipper_lemma_parallel`; this theorem is only its branch-prefix decomposition
ingredient.
-/
theorem zipper_parallel_case_tagged {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : ParLocBranches n Lifeline Channel}
    {full pref : ParLocTaggedTrace n Lifeline Channel}
    (hfull : ParLoc branches full)
    (hpref : TracePrefix pref full) :
    ∃ branchPrefix : ParLocBranches n Lifeline Channel,
      ParLoc branchPrefix pref ∧
      ∀ i A, ListPrefix (branchPrefix i A) (branches i A) :=
  parloc_prefix_decompose hfull hpref

/--
Parallel Zipper prefix theorem, observable version.

The observable trace is obtained by forgetting branch-origin tags from the
tagged prefix.  The tagged witnesses are explicit: an arbitrary untagged prefix
of an untagged trace cannot be lifted to a compatible tagged prefix without
extra information.
-/
theorem zipper_parallel_case {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : ParLocBranches n Lifeline Channel}
    {tagged_full tagged_pref : ParLocTaggedTrace n Lifeline Channel}
    (hfull : ParLoc branches tagged_full)
    (hpref : TracePrefix tagged_pref tagged_full) :
    ∃ branchPrefix : ParLocBranches n Lifeline Channel,
      UntaggedParLoc branchPrefix (forgetBranchTag tagged_pref) ∧
      ∀ i A, ListPrefix (branchPrefix i A) (branches i A) := by
  obtain ⟨branchPrefix, hprefixParLoc, hbranchPrefix⟩ :=
    parloc_prefix_decompose hfull hpref
  exact ⟨branchPrefix, parloc_forgetBranchTag_witness hprefixParLoc, hbranchPrefix⟩

end ZipperGenParallel
