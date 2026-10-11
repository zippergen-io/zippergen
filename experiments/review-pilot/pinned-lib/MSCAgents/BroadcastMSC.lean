/-
  MSCAgents/BroadcastMSC.lean
  ===========================
  Helper MSCs for the projection-generated control broadcasts from sec:projection.
-/

import MSCAgents.Projection
import MSCAgents.Erasure

section BroadcastMSC

variable {L C F Payload : Type} [DecidableEq L] [Fintype L]
variable [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]

/-- The payload used by a control broadcast with decision `ν`. -/
def controlDecisionPayload (decision : Bool) : Payload :=
  ControlPayload.setDecision decision ControlPayload.ctrlPattern

/-- The list of canonical control-message MSCs used in a broadcast. -/
noncomputable def controlBroadcastSteps
    (B : L) (recips : L → Prop) (decision : Bool)
    (hRecips : ∀ X, recips X → X ≠ B) :
    List (WordTuple L C F Payload) := by
  classical
  exact (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).attach.map
    (fun (X : { x //
        x ∈ controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips }) =>
      let hX : recips X.1 :=
        (mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp X.2
      mscMsg (C := C) (F := F)
        B (controlDecisionPayload (Payload := Payload) decision)
        X.1 (controlDecisionPayload (Payload := Payload) decision)
        (hRecips X.1 hX).symm)

/-- The MSC implementing one whole control broadcast. -/
noncomputable def controlBroadcastMSC
    (B : L) (recips : L → Prop) (decision : Bool)
    (hRecips : ∀ X, recips X → X ≠ B) :
    WordTuple L C F Payload :=
  WordTuple.concatList (controlBroadcastSteps (C := C) (F := F) (Payload := Payload)
    B recips decision hRecips)

private theorem controlDecisionPayload_compat (decision : Bool) :
    PayloadCompatible Payload
      (controlDecisionPayload (Payload := Payload) decision)
      (controlDecisionPayload (Payload := Payload) decision) := by
  by_cases h : decision = true
  · subst h
    simpa [controlDecisionPayload, ctrlTruePayload] using
      (ControlPayloadSpec.compat_ctrl_true (Payload := Payload))
  · have h' : decision = false := by cases decision <;> simp at h ⊢
    subst h'
    simpa [controlDecisionPayload, ctrlFalsePayload] using
      (ControlPayloadSpec.compat_ctrl_false (Payload := Payload))

private theorem controlDecisionPayload_isControl (decision : Bool) :
    isControlPayload (controlDecisionPayload (Payload := Payload) decision) = true := by
  by_cases h : decision = true
  · subst h
    simpa [controlDecisionPayload, ctrlTruePayload] using
      (ControlPayloadSpec.isControl_true (Payload := Payload))
  · have h' : decision = false := by
      cases decision <;> simp at h ⊢
    subst h'
    simpa [controlDecisionPayload, ctrlFalsePayload] using
      (ControlPayloadSpec.isControl_false (Payload := Payload))

theorem controlRecipients_nodup (recips : L → Prop) :
    (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips).Nodup := by
  classical
  unfold controlRecipients lifelineOrder
  rw [List.nodup_iff_pairwise_ne]
  exact List.Pairwise.filterMap
    (fun X => if h : recips X then some X else none)
    (fun a a' hne b hb b' hb' => by
      rcases (by simpa using hb) with ⟨_, rfl⟩
      rcases (by simpa using hb') with ⟨_, rfl⟩
      exact hne)
    (List.nodup_iff_pairwise_ne.mp (Fintype.nodup_enum (α := L)))

private theorem controlSendTargets_eq_controlRecipients
    (B : L) (recips : L → Prop)
    (hRecips : ∀ X, recips X → X ≠ B) :
    controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) B recips =
      controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips := by
  classical
  unfold controlSendTargets
  apply List.filter_eq_self.2
  intro X hX
  have hXB : X ≠ B :=
    hRecips X ((mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp hX)
  simp [hXB.symm]

private theorem controlBroadcast_component
    (ls : List L) (B X : L) (decision : Bool)
    (hNoSelf : ∀ Y ∈ ls, Y ≠ B) (hXB : X ≠ B) (hNodup : ls.Nodup) :
    (WordTuple.concatList
      (ls.attach.map fun Y =>
        mscMsg (C := C) (F := F)
          B (controlDecisionPayload (Payload := Payload) decision)
          Y.1 (controlDecisionPayload (Payload := Payload) decision)
          ((hNoSelf Y.1 Y.2).symm))) X =
      if X ∈ ls then
        [AlphabetOf.mkRecv (C := C) (F := F) X
          (controlDecisionPayload (Payload := Payload) decision) B]
      else [] := by
  induction ls generalizing X with
  | nil =>
      simp [WordTuple.concatList, WordTuple.empty]
  | cons Y Ys ih =>
      have hYB : Y ≠ B := hNoSelf Y (by simp)
      have hNoSelfYs : ∀ Z ∈ Ys, Z ≠ B := by
        intro Z hZ
        exact hNoSelf Z (List.mem_cons_of_mem _ hZ)
      have hNodupYs : Ys.Nodup := hNodup.of_cons
      by_cases hXY : X = Y
      · subst hXY
        have hNotMem : X ∉ Ys := (List.nodup_cons.mp hNodup).1
        have hTail := ih X hNoSelfYs hXB hNodupYs
        simp [hNotMem] at hTail
        simpa [List.attach_cons, WordTuple.concatList_cons, WordTuple.concat,
          mscMsg_receiver, hXB] using hTail
      · have hTail := ih X hNoSelfYs hXB hNodupYs
        simpa [List.attach_cons, WordTuple.concatList_cons, WordTuple.concat,
          mscMsg_other, hXB, hXY] using hTail

/-- Every control broadcast MSC is complete. -/
theorem controlBroadcastMSC_isCompleteMSC
    (B : L) (recips : L → Prop) (decision : Bool)
    (hRecips : ∀ X, recips X → X ≠ B) :
    IsCompleteMSC (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
      B recips decision hRecips) := by
  classical
  unfold controlBroadcastMSC controlBroadcastSteps
  apply concatList_complete
  intro M hM
  simp only [List.mem_map] at hM
  rcases hM with ⟨X, hX, rfl⟩
  have hRecipX : recips X :=
    (mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp X.2
  exact mscMsg_isCompleteMSC
    (C := C) (F := F)
    B (controlDecisionPayload (Payload := Payload) decision)
    X.1 (controlDecisionPayload (Payload := Payload) decision)
    ((hRecips X.1 hRecipX).symm)
    (controlDecisionPayload_compat (Payload := Payload) decision)

/-- The concrete decision-prefix tuple for a projected if-construct: the
    decider's choice letter followed by the generated control broadcast. -/
noncomputable def ifDecisionPrefixMSC
    (c : C) (B : L) (PTrue PFalse : Prog L C F Payload)
    (decision : Bool) : WordTuple L C F Payload :=
  match decision with
  | true =>
      mscIfTrue (C := C) (F := F) (Payload := Payload) c B ∘ₘ
        controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
          B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          true (by intro X hX; exact hX.1)
  | false =>
      mscIfFalse (C := C) (F := F) (Payload := Payload) c B ∘ₘ
        controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
          B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          false (by intro X hX; exact hX.1)

/-- The concrete decision-prefix tuple for a projected while-construct: the
    decider's choice letter followed by the generated control broadcast. -/
noncomputable def whileDecisionPrefixMSC
    (c : C) (B : L) (PBody PExit : Prog L C F Payload)
    (decision : Bool) : WordTuple L C F Payload :=
  match decision with
  | true =>
      mscWhileTrue (C := C) (F := F) (Payload := Payload) c B ∘ₘ
        controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
          B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          true (by intro X hX; exact hX.1)
  | false =>
      mscWhileFalse (C := C) (F := F) (Payload := Payload) c B ∘ₘ
        controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
          B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          false (by intro X hX; exact hX.1)

/-- `D` is exactly the decision-prefix tuple generated by projection for the
    if/while construct `P`, decider `B`, and decision value `ν`. -/
inductive IsDecisionPrefix :
    Prog L C F Payload → L → Bool → WordTuple L C F Payload → Prop where
  | if_true (c : C) (B : L) (PTrue PFalse : Prog L C F Payload) :
      IsDecisionPrefix (.ite c B PTrue PFalse) B true
        (ifDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
          c B PTrue PFalse true)
  | if_false (c : C) (B : L) (PTrue PFalse : Prog L C F Payload) :
      IsDecisionPrefix (.ite c B PTrue PFalse) B false
        (ifDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
          c B PTrue PFalse false)
  | while_true (c : C) (B : L) (PBody PExit : Prog L C F Payload) :
      IsDecisionPrefix (.whileLoop c B PBody PExit) B true
        (whileDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
          c B PBody PExit true)
  | while_false (c : C) (B : L) (PBody PExit : Prog L C F Payload) :
      IsDecisionPrefix (.whileLoop c B PBody PExit) B false
        (whileDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
          c B PBody PExit false)

theorem ifDecisionPrefix_isDecisionPrefix
    (c : C) (B : L) (PTrue PFalse : Prog L C F Payload)
    (decision : Bool) :
    IsDecisionPrefix (L := L) (C := C) (F := F) (Payload := Payload)
      (.ite c B PTrue PFalse) B decision
      (ifDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
        c B PTrue PFalse decision) := by
  cases decision
  · exact IsDecisionPrefix.if_false c B PTrue PFalse
  · exact IsDecisionPrefix.if_true c B PTrue PFalse

theorem whileDecisionPrefix_isDecisionPrefix
    (c : C) (B : L) (PBody PExit : Prog L C F Payload)
    (decision : Bool) :
    IsDecisionPrefix (L := L) (C := C) (F := F) (Payload := Payload)
      (.whileLoop c B PBody PExit) B decision
      (whileDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
        c B PBody PExit decision) := by
  cases decision
  · exact IsDecisionPrefix.while_false c B PBody PExit
  · exact IsDecisionPrefix.while_true c B PBody PExit

/-- The concrete decision-prefix data generated from a projected if/while
    construct satisfies `IsDecisionPrefix`.  The prefix tuples below include
    the decider choice letter and the `controlBroadcastMSC` messages. -/
theorem controlBroadcastMSC_isDecisionPrefix
    {P : Prog L C F Payload} {B : L} {decision : Bool}
    {D : WordTuple L C F Payload}
    (hConcrete :
      (∃ (c : C) (PTrue PFalse : Prog L C F Payload),
        P = .ite c B PTrue PFalse ∧
        D = ifDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
          c B PTrue PFalse decision) ∨
      (∃ (c : C) (PBody PExit : Prog L C F Payload),
        P = .whileLoop c B PBody PExit ∧
        D = whileDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
          c B PBody PExit decision)) :
    IsDecisionPrefix (L := L) (C := C) (F := F) (Payload := Payload)
      P B decision D := by
  rcases hConcrete with hIf | hWhile
  · rcases hIf with ⟨c, PTrue, PFalse, rfl, rfl⟩
    exact ifDecisionPrefix_isDecisionPrefix
      (L := L) (C := C) (F := F) (Payload := Payload) c B PTrue PFalse decision
  · rcases hWhile with ⟨c, PBody, PExit, rfl, rfl⟩
    exact whileDecisionPrefix_isDecisionPrefix
      (L := L) (C := C) (F := F) (Payload := Payload) c B PBody PExit decision

theorem ifDecisionPrefix_isCompleteMSC
    (c : C) (B : L) (PTrue PFalse : Prog L C F Payload)
    (decision : Bool) :
    IsCompleteMSC (ifDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
      c B PTrue PFalse decision) := by
  cases decision
  · simp [ifDecisionPrefixMSC]
    exact concat_complete_complete
      (C := C) (F := F) (Payload := Payload)
      (mscIfFalse (C := C) (F := F) (Payload := Payload) c B)
      (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
        B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        false (by intro X hX; exact hX.1))
      (mscIfFalse_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
      (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
        B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        false (by intro X hX; exact hX.1))
  · simp [ifDecisionPrefixMSC]
    exact concat_complete_complete
      (C := C) (F := F) (Payload := Payload)
      (mscIfTrue (C := C) (F := F) (Payload := Payload) c B)
      (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
        B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        true (by intro X hX; exact hX.1))
      (mscIfTrue_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
      (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
        B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        true (by intro X hX; exact hX.1))

theorem whileDecisionPrefix_isCompleteMSC
    (c : C) (B : L) (PBody PExit : Prog L C F Payload)
    (decision : Bool) :
    IsCompleteMSC (whileDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
      c B PBody PExit decision) := by
  cases decision
  · simp [whileDecisionPrefixMSC]
    exact concat_complete_complete
      (C := C) (F := F) (Payload := Payload)
      (mscWhileFalse (C := C) (F := F) (Payload := Payload) c B)
      (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
        B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        false (by intro X hX; exact hX.1))
      (mscWhileFalse_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
      (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
        B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        false (by intro X hX; exact hX.1))
  · simp [whileDecisionPrefixMSC]
    exact concat_complete_complete
      (C := C) (F := F) (Payload := Payload)
      (mscWhileTrue (C := C) (F := F) (Payload := Payload) c B)
      (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
        B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        true (by intro X hX; exact hX.1))
      (mscWhileTrue_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
      (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
        B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        true (by intro X hX; exact hX.1))

theorem isDecisionPrefix_isCompleteMSC
    {P : Prog L C F Payload} {B : L} {decision : Bool}
    {D : WordTuple L C F Payload}
    (hD : IsDecisionPrefix (L := L) (C := C) (F := F) (Payload := Payload)
      P B decision D) :
    IsCompleteMSC D := by
  cases hD with
  | if_true c B PTrue PFalse =>
      exact ifDecisionPrefix_isCompleteMSC
        (L := L) (C := C) (F := F) (Payload := Payload) c B PTrue PFalse true
  | if_false c B PTrue PFalse =>
      exact ifDecisionPrefix_isCompleteMSC
        (L := L) (C := C) (F := F) (Payload := Payload) c B PTrue PFalse false
  | while_true c B PBody PExit =>
      exact whileDecisionPrefix_isCompleteMSC
        (L := L) (C := C) (F := F) (Payload := Payload) c B PBody PExit true
  | while_false c B PBody PExit =>
      exact whileDecisionPrefix_isCompleteMSC
        (L := L) (C := C) (F := F) (Payload := Payload) c B PBody PExit false

/-- **Lemma `lem:strip-decision-prefix`** for the concrete decision-prefix
    tuples generated by projection. -/
theorem strip_decision_prefix
    {P : Prog L C F Payload} {B : L} {decision : Bool}
    {D : WordTuple L C F Payload}
    (hConcrete :
      (∃ (c : C) (PTrue PFalse : Prog L C F Payload),
        P = .ite c B PTrue PFalse ∧
        D = ifDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
          c B PTrue PFalse decision) ∨
      (∃ (c : C) (PBody PExit : Prog L C F Payload),
        P = .whileLoop c B PBody PExit ∧
        D = whileDecisionPrefixMSC (C := C) (F := F) (Payload := Payload)
          c B PBody PExit decision))
    (M : WordTuple L C F Payload)
    (hM : IsMSC M)
    (hForm : ∀ X,
      IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload)
        (M X) (D X) ∨
      ∃ s, M X = D X ++ s) :
    IsMSC (stripDecisionPrefix (L := L) (C := C) (F := F) (Payload := Payload)
      D M) := by
  have hD :
      IsDecisionPrefix (L := L) (C := C) (F := F) (Payload := Payload)
        P B decision D :=
    controlBroadcastMSC_isDecisionPrefix
      (L := L) (C := C) (F := F) (Payload := Payload) hConcrete
  exact strip_complete_prefix_like_msc
    (L := L) (C := C) (F := F) (Payload := Payload)
    D (isDecisionPrefix_isCompleteMSC
      (L := L) (C := C) (F := F) (Payload := Payload) hD)
    M hM hForm

/-- The decider's local word in the broadcast MSC is exactly the projected
    control-send sequence. -/
theorem controlBroadcastMSC_decider
    (B : L) (recips : L → Prop) (decision : Bool)
    (hRecips : ∀ X, recips X → X ≠ B) :
    (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
      B recips decision hRecips) B =
      controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
        B recips decision := by
  classical
  unfold controlBroadcastMSC controlBroadcastSteps controlBroadcastWord
  have hAux :
      WordTuple.concatList
        (List.map
          (fun (X : { x //
              x ∈ controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips }) =>
            let hX : recips X.1 :=
              (mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp X.2
            mscMsg (C := C) (F := F)
              B (controlDecisionPayload (Payload := Payload) decision)
              X.1 (controlDecisionPayload (Payload := Payload) decision)
              ((hRecips X.1 hX).symm))
          (List.attach
            (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips))) B =
      (List.attach
        (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips)).map
          (fun X => AlphabetOf.mkSend (C := C) (F := F) B
            (controlDecisionPayload (Payload := Payload) decision) X.1
            ((hRecips X.1
              ((mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp X.2)).symm)) := by
    induction (List.attach
        (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips)) with
    | nil =>
        simp [WordTuple.concatList, WordTuple.empty]
    | cons X Xs ih =>
        have hRecipX : recips X.1 :=
          (mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp X.2
        simpa [List.map_cons, WordTuple.concatList_cons, WordTuple.concat, mscMsg_sender] using ih
  have hNoSelf :
      ∀ X,
        X ∈ controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips →
          B ≠ X := by
    intro X hX
    exact (hRecips X
      ((mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp hX)).symm
  have hTargets :
      sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
        (A := B)
        (ControlPayload.setDecision decision ControlPayload.ctrlPattern)
        (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) B recips)
        (by
          intro X hX
          exact controlSendTargets_no_self
            (L := L) (C := C) (F := F) (Payload := Payload) hX) =
      sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
        (A := B)
        (ControlPayload.setDecision decision ControlPayload.ctrlPattern)
        (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips)
        hNoSelf := by
    unfold controlSendTargets
    simpa using
      (sendWordForTargets_filter_eq_self
        (L := L) (C := C) (F := F) (Payload := Payload) (A := B)
        (ControlPayload.setDecision decision ControlPayload.ctrlPattern)
        (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips)
        hNoSelf)
  rw [hTargets]
  rw [sendWordForTargets_eq_attach_map]
  simpa [controlDecisionPayload] using hAux

/-- A broadcast recipient sees exactly one control receive from the decider. -/
theorem controlBroadcastMSC_recipient
    (B X : L) (recips : L → Prop) (decision : Bool)
    (hRecips : ∀ Y, recips Y → Y ≠ B)
    (hX : recips X) :
    (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
      B recips decision hRecips) X =
      [AlphabetOf.mkRecv (C := C) (F := F) X
        (controlDecisionPayload (Payload := Payload) decision) B] := by
  classical
  unfold controlBroadcastMSC controlBroadcastSteps
  have hXB : X ≠ B := hRecips X hX
  simpa [(mem_controlRecipients (C := C) (F := F) (Payload := Payload)).2 hX, hXB] using
    controlBroadcast_component
      (C := C) (F := F) (Payload := Payload)
      (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips)
      B X decision
      (controlRecipients_no_self (C := C) (F := F) (Payload := Payload) hRecips)
      hXB
      (controlRecipients_nodup (C := C) (F := F) (Payload := Payload) recips)

/-- Lifelines outside the recipient set see no local event from the control
    broadcast. -/
theorem controlBroadcastMSC_nonRecipient
    (B X : L) (recips : L → Prop) (decision : Bool)
    (hRecips : ∀ Y, recips Y → Y ≠ B)
    (hXB : X ≠ B) (hX : ¬ recips X) :
    (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
      B recips decision hRecips) X = [] := by
  classical
  unfold controlBroadcastMSC controlBroadcastSteps
  have hNotMem : X ∉ controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips := by
    intro hMem
    exact hX ((mem_controlRecipients (C := C) (F := F) (Payload := Payload)).1 hMem)
  simpa [hNotMem, hXB] using
    controlBroadcast_component
      (C := C) (F := F) (Payload := Payload)
      (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips)
      B X decision
      (controlRecipients_no_self (C := C) (F := F) (Payload := Payload) hRecips)
      hXB
      (controlRecipients_nodup (C := C) (F := F) (Payload := Payload) recips)

/-- Erasing a control broadcast yields the empty MSC. -/
theorem erase_controlBroadcastMSC
    (B : L) (recips : L → Prop) (decision : Bool)
    (hRecips : ∀ X, recips X → X ≠ B) :
    erase (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
      B recips decision hRecips) = mscEmpty := by
  classical
  unfold controlBroadcastMSC controlBroadcastSteps
  rw [eraseTuple_concatList]
  induction (List.attach
      (controlRecipients (L := L) (C := C) (F := F) (Payload := Payload) recips)) with
  | nil =>
      simp [WordTuple.concatList, mscEmpty]
  | cons X Xs ih =>
      have hRecipX : recips X.1 :=
        (mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp X.2
      have hHead :
          erase
              (mscMsg (C := C) (F := F)
                B (controlDecisionPayload (Payload := Payload) decision) X.1
                (controlDecisionPayload (Payload := Payload) decision)
                ((hRecips X.1 hRecipX).symm)) = mscEmpty := by
        apply eraseTuple_mscMsg_control
        · exact controlDecisionPayload_isControl (Payload := Payload) decision
        · exact controlDecisionPayload_isControl (Payload := Payload) decision
      simp only [List.map_cons, WordTuple.concatList_cons, hHead, ih, mscEmpty,
        WordTuple.concat_eps_left]

end BroadcastMSC
