/-
  MSCAgents/Projection.lean
  =========================
  Formalization of the projection definitions from sec:projection
  ("From Global to Distributed Programs") of:
  "Provable Coordination for LLM Agents via Message Sequence Charts"
-/

import MSCAgents.ControlPayload
import MSCAgents.LocalSyntax
import MSCAgents.ParticipationSets
import MSCAgents.WellTyped

attribute [local instance] Classical.propDecidable

/-- A fixed enumeration of the lifeline set, used as the projection order for
    decision broadcasts. This plays the role of the paper's fixed total order. -/
noncomputable def lifelineOrder {L : Type} [Fintype L] : List L :=
  Fintype.enum L

/-- Sequence a list of local programs left-to-right. -/
def seqLocList {L C F Payload : Type} {A : L}
    (Ss : List (LocProg L C F Payload A)) : LocProg L C F Payload A :=
  Ss.foldr LocProg.seq LocProg.eps

@[simp]
theorem seqLocList_nil {L C F Payload : Type} {A : L} :
    seqLocList (L := L) (C := C) (F := F) (Payload := Payload)
      (A := A) [] = LocProg.eps := rfl

@[simp]
theorem seqLocList_cons {L C F Payload : Type} {A : L}
    (S : LocProg L C F Payload A) (Ss : List (LocProg L C F Payload A)) :
    seqLocList (L := L) (C := C) (F := F) (Payload := Payload)
      (A := A) (S :: Ss) = (S ;;ₗ seqLocList Ss) := rfl

/-- Structural program size corresponding to the paper's `|P|`. -/
def progSize {L C F Payload : Type} : Prog L C F Payload → Nat
  | .eps => 0
  | .msg .. => 1
  | .act .. => 1
  | .seq P1 P2 => progSize P1 + progSize P2
  | .ite _ _ PTrue PFalse => 1 + progSize PTrue + progSize PFalse
  | .whileLoop _ _ PBody PExit => 1 + progSize PBody + progSize PExit

/-- Structural local-program size. Sequencing is additive so a control
    broadcast has size equal to the number of generated sends. -/
def locProgSize {L C F Payload : Type} {A : L} :
    LocProg L C F Payload A → Nat
  | .eps => 0
  | .send .. => 1
  | .recv .. => 1
  | .seq S1 S2 => locProgSize S1 + locProgSize S2
  | .act .. => 1
  | .recvIf _ _ STrue SFalse => 1 + locProgSize STrue + locProgSize SFalse
  | .recvWhile _ _ SBody SExit => 1 + locProgSize SBody + locProgSize SExit
  | .localIf _ STrue SFalse => 1 + locProgSize STrue + locProgSize SFalse
  | .localWhile _ SBody SExit => 1 + locProgSize SBody + locProgSize SExit

section SizeLemmas

variable {α : Type}

@[simp]
theorem sum_map_const_one_eq_length (xs : List α) :
    (xs.map (fun _ => (1 : Nat))).sum = xs.length := by
  induction xs with
  | nil => simp
  | cons _ xs ih =>
      simp [ih]
      omega

@[simp]
theorem scaledEqIndicatorSum_eq_mul_count [DecidableEq α] (a : α) (k : Nat) (xs : List α) :
    (xs.map (fun x => if x = a then k else 0)).sum = k * xs.count a := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      by_cases h : x = a
      · simpa [h, ih, Nat.mul_add, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
          using rfl
      · simp [h, ih]

theorem sum_map_le (xs : List α) {f g : α → Nat} (h : ∀ x, f x ≤ g x) :
    (xs.map f).sum ≤ (xs.map g).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp
      exact Nat.add_le_add (h x) ih

@[simp]
theorem sum_map_add (xs : List α) (f g : α → Nat) :
    (xs.map (fun x => f x + g x)).sum = (xs.map f).sum + (xs.map g).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp [ih]
      omega

@[simp]
theorem sum_map_add3 (xs : List α) (f g h : α → Nat) :
    (xs.map (fun x => f x + g x + h x)).sum =
      (xs.map f).sum + (xs.map g).sum + (xs.map h).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp [ih]
      omega

end SizeLemmas

@[simp]
theorem locProgSize_seqLocList {L C F Payload : Type} {A : L}
    (Ss : List (LocProg L C F Payload A)) :
    locProgSize (seqLocList (L := L) (C := C) (F := F) (Payload := Payload)
      (A := A) Ss) = (Ss.map locProgSize).sum := by
  induction Ss with
  | nil => simp [seqLocList, locProgSize]
  | cons S Ss ih =>
      simp [seqLocList_cons, locProgSize, ih]

/-- Projection-recipient set for `if`. -/
def ifRecipients {L C F Payload : Type} [DecidableEq L]
    (B : L) (PTrue PFalse : Prog L C F Payload) : L → Prop :=
  fun X => X ≠ B ∧ (participationSet PTrue X ∨ participationSet PFalse X)

/-- Projection-recipient set for `while`. -/
def whileRecipients {L C F Payload : Type} [DecidableEq L]
    (B : L) (PBody PExit : Prog L C F Payload) : L → Prop :=
  fun X => X ≠ B ∧ (participationSet PBody X ∨ participationSet PExit X)

/-- Ordered list of control-broadcast recipients in the fixed lifeline order. -/
noncomputable def controlRecipients
    {L C F Payload : Type} [DecidableEq L] [Fintype L]
    (recips : L → Prop) : List L := by
  classical
  exact (lifelineOrder).filterMap fun X => if h : recips X then some X else none

/-- Control-broadcast send targets, with any self-target defensively removed. -/
noncomputable def controlSendTargets
    {L C F Payload : Type} [DecidableEq L] [Fintype L]
    (A : L) (recips : L → Prop) : List L := by
  classical
  exact (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).filter
    (fun X => decide (A ≠ X))

section ControlRecipientsLemmas

variable {L C F Payload : Type} [DecidableEq L] [Fintype L]

@[simp]
theorem mem_controlRecipients {recips : L → Prop} {X : L} :
    X ∈ controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips ↔ recips X := by
  classical
  unfold controlRecipients lifelineOrder
  constructor
  · intro hX
    rcases List.mem_filterMap.mp hX with ⟨Y, hY, hSome⟩
    by_cases hRecip : recips Y
    · simp [hRecip] at hSome
      subst hSome
      exact hRecip
    · simp [hRecip] at hSome
  · intro hRecip
    apply List.mem_filterMap.mpr
    refine ⟨X, Fintype.mem_enum X, ?_⟩
    simp [hRecip]

theorem controlRecipients_no_self {recips : L → Prop} {B : L}
    (hRecips : ∀ X, recips X → X ≠ B) :
    ∀ X ∈ controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips, X ≠ B := by
  intro X hX
  exact hRecips X ((mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp hX)

theorem controlSendTargets_no_self {recips : L → Prop} {A X : L}
    (hX : X ∈ controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips) :
    A ≠ X := by
  classical
  unfold controlSendTargets at hX
  exact of_decide_eq_true ((List.mem_filter.mp hX).2)

theorem controlSendTargets_length_le_controlRecipients (A : L) (recips : L → Prop) :
    (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips).length ≤
      (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length := by
  classical
  unfold controlSendTargets
  exact List.length_filter_le _ _

@[simp]
theorem indicatorSum_eq_controlRecipients_length {recips : L → Prop} :
    ((lifelineOrder (L := L)).map fun X => if recips X then (1 : Nat) else 0).sum =
      (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length := by
  classical
  unfold controlRecipients lifelineOrder
  induction Fintype.enum L with
  | nil => simp
  | cons X Xs ih =>
      by_cases h : recips X
      · simp [h, ih]
        omega
      · simp [h, ih]

theorem controlRecipients_length_le_card {recips : L → Prop} :
    (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length ≤
      Fintype.card L := by
  classical
  unfold controlRecipients lifelineOrder Fintype.card
  simpa using
    List.length_filterMap_le
      (fun X => if h : recips X then some X else none) (Fintype.enum L)

theorem scaledSingletonIndicatorSum_le (B : L) (k : Nat) :
    ((lifelineOrder (L := L)).map fun X => if X = B then k else 0).sum ≤ k := by
  have hcount :
      (lifelineOrder (L := L)).count B = 1 := by
    simpa [lifelineOrder, Fintype.mem_enum B] using
      (List.Nodup.count (a := B) (l := Fintype.enum L) Fintype.nodup_enum)
  rw [scaledEqIndicatorSum_eq_mul_count]
  simp [hcount]

end ControlRecipientsLemmas

/-- Control-send sequence sent by the decider in the fixed lifeline order. -/
noncomputable def controlBroadcast
    {L C F Payload : Type} [DecidableEq L] [Fintype L] [ControlPayload Payload]
    (A : L) (recips : L → Prop) (decision : Bool) :
    LocProg L C F Payload A := by
  classical
  exact seqLocList <|
    (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips).map
      (fun X => LocProg.send (A := A)
        (ControlPayload.setDecision decision ControlPayload.ctrlPattern) X)

@[simp]
theorem locProgSize_controlBroadcast
    {L C F Payload : Type} [DecidableEq L] [Fintype L] [ControlPayload Payload]
    (A : L) (recips : L → Prop) (decision : Bool) :
    locProgSize (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
      A recips decision) =
      (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips).length := by
  classical
  unfold controlBroadcast
  rw [locProgSize_seqLocList]
  induction controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips with
  | nil => simp
  | cons X Xs ih =>
      simpa [locProgSize, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
        congrArg Nat.succ ih

/-- Structural projection `π_A(P)` from global programs to local programs. -/
noncomputable def project
    {L C F Payload : Type} [DecidableEq L] [Fintype L] [ControlPayload Payload]
    (A : L) : Prog L C F Payload → LocProg L C F Payload A := by
  classical
  intro P
  exact match P with
  | .eps => .eps
  | .msg X xs Y ys _ =>
      if hAX : A = X then
        hAX ▸ LocProg.send xs Y
      else if hAY : A = Y then
        hAY ▸ LocProg.recv ys X
      else
        .eps
  | .act X ys f xs =>
      if hAX : A = X then
        hAX ▸ LocProg.act ys f xs
      else
        .eps
  | .seq P1 P2 =>
      project A P1 ;;ₗ project A P2
  | .ite c B PTrue PFalse =>
      if hAB : A = B then
        hAB ▸ LocProg.localIf c
          (controlBroadcast B (ifRecipients B PTrue PFalse) true ;;ₗ project B PTrue)
          (controlBroadcast B (ifRecipients B PTrue PFalse) false ;;ₗ project B PFalse)
      else if hRecip : ifRecipients B PTrue PFalse A then
        LocProg.recvIf ControlPayload.ctrlPattern B (project A PTrue) (project A PFalse)
      else
        .eps
  | .whileLoop c B PBody PExit =>
      if hAB : A = B then
        hAB ▸ LocProg.localWhile c
          (controlBroadcast B (whileRecipients B PBody PExit) true ;;ₗ project B PBody)
          (controlBroadcast B (whileRecipients B PBody PExit) false ;;ₗ project B PExit)
      else if hRecip : whileRecipients B PBody PExit A then
        LocProg.recvWhile ControlPayload.ctrlPattern B (project A PBody) (project A PExit)
      else
        .eps

/-- Distributed projection `𝒟_P = (π_A(P))_A`. -/
noncomputable def projectDist
    {L C F Payload : Type} [DecidableEq L] [Fintype L] [ControlPayload Payload]
    (P : Prog L C F Payload) : DistProg L C F Payload :=
  fun A => project A P

/-- Sum of projected local-program sizes over the fixed lifeline order. -/
noncomputable def totalProjectionSize
    {L C F Payload : Type} [DecidableEq L] [Fintype L] [ControlPayload Payload]
    (P : Prog L C F Payload) : Nat :=
  ((lifelineOrder (L := L)).map fun A =>
    locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A P)).sum

section ProjectionLemmas

variable {L C F Payload : Type} [DecidableEq L] [Fintype L] [ControlPayload Payload]

@[simp]
theorem project_eps (A : L) :
    project (L := L) (C := C) (F := F) (Payload := Payload) A Prog.eps = LocProg.eps := by
  simp [project]

theorem project_if_size_decompose (A B : L) (c : C)
    (PTrue PFalse : Prog L C F Payload) :
    let recips := ifRecipients B PTrue PFalse
    let nrecips := (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length
    locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
      (Prog.ite c B PTrue PFalse)) ≤
      (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
        locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue) +
        locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse) := by
  classical
  let recips := ifRecipients B PTrue PFalse
  let nrecips :=
    (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length
  by_cases hAB : A = B
  · subst hAB
    have hTargetsLe :
        (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)).length ≤
          (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload)
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)).length := by
      exact controlSendTargets_length_le_controlRecipients (L := L) (C := C) (F := F)
        (Payload := Payload) A
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
    simp [project, recips, nrecips, locProgSize, locProgSize_controlBroadcast]
    omega
  · by_cases hRecip : recips A
    · simp [project, hAB, hRecip, recips, nrecips, locProgSize]
    · simp [project, hAB, hRecip, recips, nrecips, locProgSize]

theorem project_while_size_decompose (A B : L) (c : C)
    (PBody PExit : Prog L C F Payload) :
    let recips := whileRecipients B PBody PExit
    let nrecips := (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length
    locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
      (Prog.whileLoop c B PBody PExit)) ≤
      (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
        locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PBody) +
        locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PExit) := by
  classical
  let recips := whileRecipients B PBody PExit
  let nrecips :=
    (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length
  by_cases hAB : A = B
  · subst hAB
    have hTargetsLe :
        (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)).length ≤
          (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload)
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)).length := by
      exact controlSendTargets_length_le_controlRecipients (L := L) (C := C) (F := F)
        (Payload := Payload) A
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)
    simp [project, recips, nrecips, locProgSize, locProgSize_controlBroadcast]
    omega
  · by_cases hRecip : recips A
    · simp [project, hAB, hRecip, recips, nrecips, locProgSize]
    · simp [project, hAB, hRecip, recips, nrecips, locProgSize]

theorem project_size_linear (A : L) :
    ∀ P : Prog L C F Payload,
      locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A P) ≤
        (2 * Fintype.card L + 1) * progSize P
  | .eps => by simp [project, progSize, locProgSize]
  | .msg X xs Y ys h => by
      by_cases hAX : A = X
      · subst hAX
        simp [project, progSize, locProgSize]
      · by_cases hAY : A = Y
        · subst hAY
          simp [project, hAX, progSize, locProgSize]
        · simp [project, hAX, hAY, progSize, locProgSize]
  | .act X ys f xs => by
      by_cases hAX : A = X
      · subst hAX
        simp [project, progSize, locProgSize]
      · simp [project, hAX, progSize, locProgSize]
  | .seq P1 P2 => by
      have h1 := project_size_linear A P1
      have h2 := project_size_linear A P2
      calc
        locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A (P1 ;; P2))
            = locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A P1) +
              locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A P2) := by
                simp [project, locProgSize]
        _ ≤ (2 * Fintype.card L + 1) * progSize P1 +
              (2 * Fintype.card L + 1) * progSize P2 := by
                exact Nat.add_le_add h1 h2
        _ = (2 * Fintype.card L + 1) * progSize (P1 ;; P2) := by
              simp [progSize, Nat.left_distrib]
  | .ite c B PTrue PFalse => by
      classical
      let recips := ifRecipients B PTrue PFalse
      let nrecips :=
        (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length
      let k := 2 * Fintype.card L + 1
      have htrue := project_size_linear A PTrue
      have hfalse := project_size_linear A PFalse
      have hsub :
          locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue) +
            locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse) ≤
              k * progSize PTrue + k * progSize PFalse := by
        exact Nat.add_le_add htrue hfalse
      have hn : nrecips ≤ Fintype.card L := by
        exact controlRecipients_length_le_card (recips := recips)
      have hextra :
          (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) ≤ k := by
        by_cases hAB : A = B
        · simp [hAB, k]
          omega
        · by_cases hRecip : recips A
          · simp [hAB, hRecip, k]
          · simp [hAB, hRecip, k]
      have hdecomp := project_if_size_decompose (L := L) (C := C) (F := F)
        (Payload := Payload) A B c PTrue PFalse
      calc
        locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
            (Prog.ite c B PTrue PFalse))
            ≤ (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
              locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue) +
              locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse) := by
                simpa [recips, nrecips] using hdecomp
        _ ≤ (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
              (k * progSize PTrue + k * progSize PFalse) := by
                simpa [Nat.add_assoc] using
                  Nat.add_le_add_left hsub
                    (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)
        _ ≤ k + (k * progSize PTrue + k * progSize PFalse) := by
              exact Nat.add_le_add_right hextra _
        _ = (2 * Fintype.card L + 1) * progSize (Prog.ite c B PTrue PFalse) := by
              simp [progSize, k, Nat.left_distrib, Nat.right_distrib]
              omega
  | .whileLoop c B PBody PExit => by
      classical
      let recips := whileRecipients B PBody PExit
      let nrecips :=
        (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length
      let k := 2 * Fintype.card L + 1
      have hbody := project_size_linear A PBody
      have hexit := project_size_linear A PExit
      have hsub :
          locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PBody) +
            locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PExit) ≤
              k * progSize PBody + k * progSize PExit := by
        exact Nat.add_le_add hbody hexit
      have hn : nrecips ≤ Fintype.card L := by
        exact controlRecipients_length_le_card (recips := recips)
      have hextra :
          (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) ≤ k := by
        by_cases hAB : A = B
        · simp [hAB, k]
          omega
        · by_cases hRecip : recips A
          · simp [hAB, hRecip, k]
          · simp [hAB, hRecip, k]
      have hdecomp := project_while_size_decompose (L := L) (C := C) (F := F)
        (Payload := Payload) A B c PBody PExit
      calc
        locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
            (Prog.whileLoop c B PBody PExit))
            ≤ (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
              locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PBody) +
              locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PExit) := by
                simpa [recips, nrecips] using hdecomp
        _ ≤ (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
              (k * progSize PBody + k * progSize PExit) := by
                simpa [Nat.add_assoc] using
                  Nat.add_le_add_left hsub
                    (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)
        _ ≤ k + (k * progSize PBody + k * progSize PExit) := by
              exact Nat.add_le_add_right hextra _
        _ = (2 * Fintype.card L + 1) * progSize (Prog.whileLoop c B PBody PExit) := by
              simp [progSize, k, Nat.left_distrib, Nat.right_distrib]
              omega

theorem totalProjectionSize_linear :
    ∀ P : Prog L C F Payload,
      totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) P ≤
        (3 * Fintype.card L + 2) * progSize P
  | .eps => by
      unfold totalProjectionSize
      unfold lifelineOrder
      have hzero : (List.map (fun A : L => (0 : Nat)) (Fintype.enum L)).sum = 0 := by
        induction Fintype.enum L with
        | nil => simp
        | cons X Xs ih => simp [ih]
      simpa [progSize, project, locProgSize] using hzero
  | .msg X xs Y ys h => by
      have hpoint :
          ∀ A,
            locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
              (Prog.msg X xs Y ys h)) ≤
              if A = X then 1 else if A = Y then 1 else 0 := by
        intro A
        by_cases hAX : A = X
        · subst hAX
          simp [project, locProgSize]
        · by_cases hAY : A = Y
          · subst hAY
            simp [project, hAX, locProgSize]
          · simp [project, hAX, hAY, locProgSize]
      have hsplit :
          ((lifelineOrder (L := L)).map fun A => if A = X then 1 else if A = Y then 1 else 0).sum ≤
            ((lifelineOrder (L := L)).map
              (fun A => (if A = X then 1 else 0) + (if A = Y then 1 else 0))).sum := by
        exact sum_map_le _ (fun A => by
          by_cases hAX : A = X <;> by_cases hAY : A = Y <;> simp [hAX, hAY])
      calc
        totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload)
            (Prog.msg X xs Y ys h)
            = ((lifelineOrder (L := L)).map
                (fun A =>
                  locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
                    (Prog.msg X xs Y ys h)))).sum := rfl
        _ ≤ ((lifelineOrder (L := L)).map fun A => if A = X then 1 else if A = Y then 1 else 0).sum := by
              exact sum_map_le _ hpoint
        _ ≤ ((lifelineOrder (L := L)).map
              (fun A => (if A = X then 1 else 0) + (if A = Y then 1 else 0))).sum := hsplit
        _ = ((lifelineOrder (L := L)).map fun A => if A = X then 1 else 0).sum +
              ((lifelineOrder (L := L)).map fun A => if A = Y then 1 else 0).sum := by
                simp [sum_map_add]
        _ ≤ 1 + 1 := by
              exact Nat.add_le_add (scaledSingletonIndicatorSum_le (B := X) 1)
                (scaledSingletonIndicatorSum_le (B := Y) 1)
        _ ≤ (3 * Fintype.card L + 2) * progSize (Prog.msg X xs Y ys h) := by
              simp [progSize]
  | .act X ys f xs => by
      have hpoint :
          ∀ A,
            locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
              (Prog.act X ys f xs)) ≤
              if A = X then 1 else 0 := by
        intro A
        by_cases hAX : A = X
        · subst hAX
          simp [project, locProgSize]
        · simp [project, hAX, locProgSize]
      calc
        totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload)
            (Prog.act X ys f xs)
            = ((lifelineOrder (L := L)).map
                (fun A =>
                  locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
                    (Prog.act X ys f xs)))).sum := rfl
        _ ≤ ((lifelineOrder (L := L)).map fun A => if A = X then 1 else 0).sum := by
              exact sum_map_le _ hpoint
        _ ≤ 1 := scaledSingletonIndicatorSum_le (B := X) 1
        _ ≤ (3 * Fintype.card L + 2) * progSize (Prog.act X ys f xs) := by
              simp [progSize]
  | .seq P1 P2 => by
      have h1 := totalProjectionSize_linear P1
      have h2 := totalProjectionSize_linear P2
      calc
        totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) (P1 ;; P2)
            = totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) P1 +
              totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) P2 := by
                simp [totalProjectionSize, project, locProgSize, sum_map_add]
        _ ≤ (3 * Fintype.card L + 2) * progSize P1 +
              (3 * Fintype.card L + 2) * progSize P2 := by
                exact Nat.add_le_add h1 h2
        _ = (3 * Fintype.card L + 2) * progSize (P1 ;; P2) := by
              simp [progSize, Nat.left_distrib]
  | .ite c B PTrue PFalse => by
      classical
      let recips := ifRecipients B PTrue PFalse
      let nrecips :=
        (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length
      let k := 3 * Fintype.card L + 2
      have htrue := totalProjectionSize_linear PTrue
      have hfalse := totalProjectionSize_linear PFalse
      have hsub :
          totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PTrue +
            totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PFalse ≤
              k * progSize PTrue + k * progSize PFalse := by
        exact Nat.add_le_add htrue hfalse
      have hn : nrecips ≤ Fintype.card L := by
        exact controlRecipients_length_le_card (recips := recips)
      have hpoint :
          ∀ A,
            locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
              (Prog.ite c B PTrue PFalse)) ≤
              (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
                locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue) +
                locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse) := by
        intro A
        simpa [recips, nrecips] using
          project_if_size_decompose (L := L) (C := C) (F := F) (Payload := Payload)
            A B c PTrue PFalse
      have hsplit :
          ((lifelineOrder (L := L)).map
            (fun A => if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)).sum ≤
              ((lifelineOrder (L := L)).map
                (fun A => (if A = B then 1 + 2 * nrecips else 0) + (if recips A then 1 else 0))).sum := by
        exact sum_map_le _ (fun A => by
          by_cases hAB : A = B <;> by_cases hRecip : recips A <;> simp [hAB, hRecip])
      have hOverhead :
          ((lifelineOrder (L := L)).map
            (fun A => if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)).sum ≤
              1 + 3 * Fintype.card L := by
        calc
          ((lifelineOrder (L := L)).map
            (fun A => if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)).sum
              ≤ ((lifelineOrder (L := L)).map
                  (fun A => (if A = B then 1 + 2 * nrecips else 0) + (if recips A then 1 else 0))).sum := hsplit
          _ = ((lifelineOrder (L := L)).map fun A => if A = B then 1 + 2 * nrecips else 0).sum +
                ((lifelineOrder (L := L)).map fun A => if recips A then 1 else 0).sum := by
                  simp [sum_map_add]
          _ ≤ (1 + 2 * nrecips) + nrecips := by
                exact Nat.add_le_add
                  (scaledSingletonIndicatorSum_le (B := B) (1 + 2 * nrecips))
                  (by
                    simpa [nrecips] using
                      (Nat.le_of_eq (indicatorSum_eq_controlRecipients_length (C := C) (F := F)
                        (Payload := Payload) (recips := recips))))
          _ ≤ 1 + 3 * Fintype.card L := by
                omega
      have hextra : 1 + 3 * Fintype.card L ≤ k := by
        simp [k]
        omega
      calc
        totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload)
            (Prog.ite c B PTrue PFalse)
            = ((lifelineOrder (L := L)).map
                (fun A =>
                  locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
                    (Prog.ite c B PTrue PFalse)))).sum := rfl
        _ ≤ ((lifelineOrder (L := L)).map
              (fun A =>
                (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
                locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue) +
                locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse))).sum := by
                  exact sum_map_le _ hpoint
        _ = ((lifelineOrder (L := L)).map
              (fun A => if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)).sum +
              totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PTrue +
              totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PFalse := by
                simp [totalProjectionSize, sum_map_add3]
        _ ≤ (1 + 3 * Fintype.card L) +
              totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PTrue +
              totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PFalse := by
                simpa [Nat.add_assoc] using
                  Nat.add_le_add_right hOverhead
                    (totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PTrue +
                      totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PFalse)
        _ ≤ k + (k * progSize PTrue + k * progSize PFalse) := by
              have hk' : (1 + 3 * Fintype.card L) +
                  (totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PTrue +
                    totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PFalse) ≤
                    k + (k * progSize PTrue + k * progSize PFalse) := by
                exact Nat.add_le_add hextra hsub
              simpa [Nat.add_assoc] using hk'
        _ = (3 * Fintype.card L + 2) * progSize (Prog.ite c B PTrue PFalse) := by
              simp [progSize, k, Nat.left_distrib, Nat.right_distrib]
              omega
  | .whileLoop c B PBody PExit => by
      classical
      let recips := whileRecipients B PBody PExit
      let nrecips :=
        (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).length
      let k := 3 * Fintype.card L + 2
      have hbody := totalProjectionSize_linear PBody
      have hexit := totalProjectionSize_linear PExit
      have hsub :
          totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PBody +
            totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PExit ≤
              k * progSize PBody + k * progSize PExit := by
        exact Nat.add_le_add hbody hexit
      have hn : nrecips ≤ Fintype.card L := by
        exact controlRecipients_length_le_card (recips := recips)
      have hpoint :
          ∀ A,
            locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
              (Prog.whileLoop c B PBody PExit)) ≤
              (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
                locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PBody) +
                locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PExit) := by
        intro A
        simpa [recips, nrecips] using
          project_while_size_decompose (L := L) (C := C) (F := F) (Payload := Payload)
            A B c PBody PExit
      have hsplit :
          ((lifelineOrder (L := L)).map
            (fun A => if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)).sum ≤
              ((lifelineOrder (L := L)).map
                (fun A => (if A = B then 1 + 2 * nrecips else 0) + (if recips A then 1 else 0))).sum := by
        exact sum_map_le _ (fun A => by
          by_cases hAB : A = B <;> by_cases hRecip : recips A <;> simp [hAB, hRecip])
      have hOverhead :
          ((lifelineOrder (L := L)).map
            (fun A => if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)).sum ≤
              1 + 3 * Fintype.card L := by
        calc
          ((lifelineOrder (L := L)).map
            (fun A => if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)).sum
              ≤ ((lifelineOrder (L := L)).map
                  (fun A => (if A = B then 1 + 2 * nrecips else 0) + (if recips A then 1 else 0))).sum := hsplit
          _ = ((lifelineOrder (L := L)).map fun A => if A = B then 1 + 2 * nrecips else 0).sum +
                ((lifelineOrder (L := L)).map fun A => if recips A then 1 else 0).sum := by
                  simp [sum_map_add]
          _ ≤ (1 + 2 * nrecips) + nrecips := by
                exact Nat.add_le_add
                  (scaledSingletonIndicatorSum_le (B := B) (1 + 2 * nrecips))
                  (by
                    simpa [nrecips] using
                      (Nat.le_of_eq (indicatorSum_eq_controlRecipients_length (C := C) (F := F)
                        (Payload := Payload) (recips := recips))))
          _ ≤ 1 + 3 * Fintype.card L := by
                omega
      have hextra : 1 + 3 * Fintype.card L ≤ k := by
        simp [k]
        omega
      calc
        totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload)
            (Prog.whileLoop c B PBody PExit)
            = ((lifelineOrder (L := L)).map
                (fun A =>
                  locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A
                    (Prog.whileLoop c B PBody PExit)))).sum := rfl
        _ ≤ ((lifelineOrder (L := L)).map
              (fun A =>
                (if A = B then 1 + 2 * nrecips else if recips A then 1 else 0) +
                locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PBody) +
                locProgSize (project (L := L) (C := C) (F := F) (Payload := Payload) A PExit))).sum := by
                  exact sum_map_le _ hpoint
        _ = ((lifelineOrder (L := L)).map
              (fun A => if A = B then 1 + 2 * nrecips else if recips A then 1 else 0)).sum +
              totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PBody +
              totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PExit := by
                simp [totalProjectionSize, sum_map_add3]
        _ ≤ (1 + 3 * Fintype.card L) +
              totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PBody +
              totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PExit := by
                simpa [Nat.add_assoc] using
                  Nat.add_le_add_right hOverhead
                    (totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PBody +
                      totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PExit)
        _ ≤ k + (k * progSize PBody + k * progSize PExit) := by
              have hk' : (1 + 3 * Fintype.card L) +
                  (totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PBody +
                    totalProjectionSize (L := L) (C := C) (F := F) (Payload := Payload) PExit) ≤
                    k + (k * progSize PBody + k * progSize PExit) := by
                exact Nat.add_le_add hextra hsub
              simpa [Nat.add_assoc] using hk'
        _ = (3 * Fintype.card L + 2) * progSize (Prog.whileLoop c B PBody PExit) := by
              simp [progSize, k, Nat.left_distrib, Nat.right_distrib]
              omega

end ProjectionLemmas
