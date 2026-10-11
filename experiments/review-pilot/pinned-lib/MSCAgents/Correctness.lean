/-
  MSCAgents/Correctness.lean
  ==========================
  Formalization of Theorem (thm:correctness) from sec:projection:
    • Item 1: realization_complete (L1927) — every M ∈ ⟦P⟧ has a distributed realization
    • Item 2: realization_sound    (L2386) — every distributed execution erases to ⟦P⟧
-/

import MSCAgents.BroadcastMSC
import MSCAgents.ZipperLemma

section Correctness

variable {L C F Payload : Type} [DecidableEq L] [Fintype L]
variable [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]

-- `distSemantics_if_decompose` and `distSemantics_while_decompose` are proved
-- below, after the local helper layer.

private theorem localTrace_implies_localPrefix {A : L}
    (S : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (hTrace :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) S w) :
    localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) S w := by
  exact ⟨w, hTrace, ⟨[], by simp [IsPrefixWord]⟩⟩

private theorem localTraceSemantics_send_eq
    {A : L} (xs : Payload) (B : L) (hAB : A ≠ B)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.send xs B) w) :
    w = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] := by
  rcases h with ⟨hAB', hw⟩
  calc
    w = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB'] := hw
    _ = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] := by
      simp [AlphabetOf.mkSend]

private theorem ifHead_true_ne_false {A : L} {c : C} :
    AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c ≠
      AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c := by
  intro h
  have hVal :
      Letter.ifTrueLetter c A = Letter.ifFalseLetter c A :=
    congrArg Subtype.val h
  cases hVal

private theorem ifHead_false_ne_true {A : L} {c : C} :
    AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c ≠
      AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c := by
  intro h
  have hVal :
      Letter.ifFalseLetter c A = Letter.ifTrueLetter c A :=
    congrArg Subtype.val h
  cases hVal

private theorem tupleComplete_suffix
    (M1 M2 : WordTuple L C F Payload)
    (h1 : tupleComplete M1)
    (h : tupleComplete (M1 ∘ₘ M2)) :
    tupleComplete M2 := by
  intro A B
  have h1AB := h1 A B
  have hAB := h A B
  simp [channelComplete, sndCount, rcvCount, WordTuple.concat,
    countSends_append, countRecvs_append] at h1AB hAB ⊢
  omega

private theorem complete_suffix_of_complete_prefix
    (M1 M2 : WordTuple L C F Payload)
    (h1 : IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) M1)
    (h : IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) (M1 ∘ₘ M2)) :
    IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) M2 := by
  have hMSC :
      IsMSC (L := L) (C := C) (F := F) (Payload := Payload) (M1 ∘ₘ M2) :=
    isCompleteMSC_implies_isMSC (L := L) (C := C) (F := F) (Payload := Payload)
      (M1 ∘ₘ M2) h
  have hSuffixMSC :
      IsMSC (L := L) (C := C) (F := F) (Payload := Payload) M2 :=
    suffix_msc_of_complete_prefix (L := L) (C := C) (F := F) (Payload := Payload)
      M1 M2 h1 hMSC
  exact
    { complete := tupleComplete_suffix (L := L) (C := C) (F := F) (Payload := Payload)
        M1 M2 h1.complete h.complete
      labelCompat := hSuffixMSC.labelCompat
      acyclic := hSuffixMSC.acyclic }

private theorem localTraceSemantics_sendList_eq {A : L}
    (payload : Payload) :
    ∀ (targets : List L) (hTargets : ∀ X, X ∈ targets → A ≠ X) (w),
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload) (A := A)
          (seqLocList (targets.map (fun X => LocProg.send (A := A) payload X)))
          w →
        w = sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
          (A := A) payload targets hTargets
  | [], _hTargets, w, h => by
      simpa [seqLocList, localTraceSemantics] using h
  | X :: Xs, hTargets, w, h => by
      simp [seqLocList] at h
      rcases h with ⟨u1, u2, hu1, hu2, rfl⟩
      have hu1' :=
        localTraceSemantics_send_eq
          (L := L) (C := C) (F := F) (Payload := Payload)
          (A := A) payload X (hTargets X (by simp)) u1 hu1
      have hTail : ∀ Y, Y ∈ Xs → A ≠ Y := by
        intro Y hY
        exact hTargets Y (by simp [hY])
      have hu2' :=
        localTraceSemantics_sendList_eq (A := A) (payload := payload) Xs hTail u2 hu2
      rw [hu1', hu2']
      simp [sendWordForTargets]

private theorem controlBroadcast_trace_eq
    (A : L) (recips : L → Prop) (decision : Bool)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload) (A := A)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) A recips decision)
        w) :
    w =
      controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload) A recips decision := by
  classical
  have h' :=
    localTraceSemantics_sendList_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := A)
      (ControlPayload.setDecision decision ControlPayload.ctrlPattern)
      (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips)
      (by
        intro X hX
        exact controlSendTargets_no_self (L := L) (C := C) (F := F) (Payload := Payload) hX)
      w
      (by simpa [controlBroadcast] using h)
  simpa [controlBroadcastWord] using h'

private theorem sendPayloads_sendWordForTargets_eq {A : L}
    (payload : Payload) :
    ∀ targets : List L,
      ∀ hTargets : ∀ Y, Y ∈ targets → A ≠ Y,
      targets.Nodup →
      ∀ X : L,
        sendPayloads (C := C) (F := F) (Payload := Payload) X
          (sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
            (A := A) payload targets hTargets) =
        if X ∈ targets then [payload] else []
  | [], _hTargets, _hNodup, X => by
      simp [sendWordForTargets, sendPayloads]
  | Y :: Ys, hTargets, hNodup, X => by
      have hNodupYs : Ys.Nodup := (List.nodup_cons.mp hNodup).2
      have hTailTargets : ∀ Z, Z ∈ Ys → A ≠ Z := by
        intro Z hZ
        exact hTargets Z (by simp [hZ])
      have ih :=
        sendPayloads_sendWordForTargets_eq
          (A := A) (payload := payload) (targets := Ys) hTailTargets hNodupYs X
      by_cases hXY : X = Y
      · subst hXY
        have hNotMem : X ∉ Ys := (List.nodup_cons.mp hNodup).1
        have hTailEq :
            sendPayloads (C := C) (F := F) (Payload := Payload) X
              (sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
                (A := A) payload Ys hTailTargets) = [] := by
          simpa [hNotMem] using ih
        simpa [sendWordForTargets, sendPayloads, AlphabetOf.mkSend, hTailEq]
      · have hYX : Y ≠ X := by
          intro h
          exact hXY h.symm
        have hTailEq :
            sendPayloads (C := C) (F := F) (Payload := Payload) X
              (sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
                (A := A) payload Ys hTailTargets) =
              if X ∈ Ys then [payload] else [] := ih
        calc
          sendPayloads (C := C) (F := F) (Payload := Payload) X
              (sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
                (A := A) payload (Y :: Ys) hTargets)
              =
              sendPayloads (C := C) (F := F) (Payload := Payload) X
                (sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
                  (A := A) payload Ys hTailTargets) := by
                  simp [sendWordForTargets, sendPayloads, AlphabetOf.mkSend, hYX]
          _ = if X ∈ Ys then [payload] else [] := hTailEq
          _ = if X = Y ∨ X ∈ Ys then [payload] else [] := by
                simp [List.mem_cons, hXY]
          _ = if X ∈ Y :: Ys then [payload] else [] := by
                simp [List.mem_cons]

private theorem sendPayloads_controlBroadcastWord_recipient
    (A X : L) (recips : L → Prop) (decision : Bool)
    (hRecips : ∀ Y, recips Y → Y ≠ A)
    (hX : recips X) :
    sendPayloads (C := C) (F := F) (Payload := Payload) X
      (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload) A recips decision) =
      [ControlPayload.setDecision decision ControlPayload.ctrlPattern] := by
  classical
  unfold controlBroadcastWord
  have hTargets :
      ∀ Y,
        Y ∈ controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips →
          A ≠ Y := by
    intro Y hY
    exact controlSendTargets_no_self (L := L) (C := C) (F := F) (Payload := Payload) hY
  have hXTarget :
      X ∈ controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips := by
    unfold controlSendTargets
    exact List.mem_filter.mpr
      ⟨(mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mpr hX,
        decide_eq_true ((hRecips X hX).symm)⟩
  rw [sendPayloads_sendWordForTargets_eq
    (L := L) (C := C) (F := F) (Payload := Payload)
    (A := A)
    (ControlPayload.setDecision decision ControlPayload.ctrlPattern)
    (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips)
    hTargets
    (List.Nodup.sublist List.filter_sublist
      (controlRecipients_nodup (L := L) (C := C) (F := F) (Payload := Payload) recips))
    X]
  simp [hXTarget]

private theorem if_recipient_recvPayloads_head
    (B X : L) (decision : Bool)
    (M : WordTuple L C F Payload)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hX :
      M X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision decision ControlPayload.ctrlPattern) B :: uX) :
    ∃ rs,
      recvPayloads (C := C) (F := F) (Payload := Payload) B (M X) =
        ControlPayload.setDecision decision ControlPayload.ctrlPattern :: rs := by
  refine ⟨recvPayloads (C := C) (F := F) (Payload := Payload) B uX, ?_⟩
  rw [hX]
  simp [recvPayloads, AlphabetOf.mkRecv, Letter.isRecvFrom]

private theorem if_true_recipient_false_impossible
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (Mhat : WordTuple L C F Payload)
    (uB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (hMB :
      Mhat B =
        AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c ::
          (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true ++ uB))
    (hMX :
      Mhat X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: uX)
    (hMSC : IsMSC (L := L) (C := C) (F := F) (Payload := Payload) Mhat) :
    False := by
  have hs :
      ∃ ss,
        sendPayloads (C := C) (F := F) (Payload := Payload) X (Mhat B) =
          ControlPayload.setDecision true ControlPayload.ctrlPattern :: ss := by
    refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) X uB, ?_⟩
    rw [hMB]
    simp [sendPayloads, AlphabetOf.mkIfTrue, Letter.isSendTo,
      sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        true (by intro Y hY; exact hY.1) hRecip]
  have hr :=
    if_recipient_recvPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X false Mhat uX hMX
  have hEq :=
    firstControlDecisions_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      Mhat hMSC B X true false hs hr
  cases hEq

private theorem if_false_recipient_true_impossible
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (Mhat : WordTuple L C F Payload)
    (uB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (hMB :
      Mhat B =
        AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c ::
          (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false ++ uB))
    (hMX :
      Mhat X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: uX)
    (hMSC : IsMSC (L := L) (C := C) (F := F) (Payload := Payload) Mhat) :
    False := by
  have hs :
      ∃ ss,
        sendPayloads (C := C) (F := F) (Payload := Payload) X (Mhat B) =
          ControlPayload.setDecision false ControlPayload.ctrlPattern :: ss := by
    refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) X uB, ?_⟩
    rw [hMB]
    simp [sendPayloads, AlphabetOf.mkIfFalse, Letter.isSendTo,
      sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        false (by intro Y hY; exact hY.1) hRecip]
  have hr :=
    if_recipient_recvPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X true Mhat uX hMX
  have hEq :=
    firstControlDecisions_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      Mhat hMSC B X false true hs hr
  cases hEq

private theorem while_true_recipient_false_impossible
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (Mhat : WordTuple L C F Payload)
    (uB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (hMB :
      Mhat B =
        AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
          (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            true ++ uB))
    (hMX :
      Mhat X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: uX)
    (hMSC : IsMSC (L := L) (C := C) (F := F) (Payload := Payload) Mhat) :
    False := by
  have hs :
      ∃ ss,
        sendPayloads (C := C) (F := F) (Payload := Payload) X (Mhat B) =
          ControlPayload.setDecision true ControlPayload.ctrlPattern :: ss := by
    refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) X uB, ?_⟩
    rw [hMB]
    simp [sendPayloads, AlphabetOf.mkWhileTrue, Letter.isSendTo,
      sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        true (by intro Y hY; exact hY.1) hRecip]
  have hr :=
    if_recipient_recvPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X false Mhat uX hMX
  have hEq :=
    firstControlDecisions_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      Mhat hMSC B X true false hs hr
  cases hEq

private theorem while_false_recipient_true_impossible
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (Mhat : WordTuple L C F Payload)
    (uB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (hMB :
      Mhat B =
        AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
          (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            false ++ uB))
    (hMX :
      Mhat X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: uX)
    (hMSC : IsMSC (L := L) (C := C) (F := F) (Payload := Payload) Mhat) :
    False := by
  have hs :
      ∃ ss,
        sendPayloads (C := C) (F := F) (Payload := Payload) X (Mhat B) =
          ControlPayload.setDecision false ControlPayload.ctrlPattern :: ss := by
    refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) X uB, ?_⟩
    rw [hMB]
    simp [sendPayloads, AlphabetOf.mkWhileFalse, Letter.isSendTo,
      sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        false (by intro Y hY; exact hY.1) hRecip]
  have hr :=
    if_recipient_recvPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X true Mhat uX hMX
  have hEq :=
    firstControlDecisions_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      Mhat hMSC B X false true hs hr
  cases hEq

theorem distSemantics_if_decompose
    (c : C) (B : L) (PTrue PFalse : Prog L C F Payload)
    (Mhat : WordTuple L C F Payload)
    (hDist :
      distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
          (.ite c B PTrue PFalse))
        Mhat) :
    ((∃ MTrue,
      distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PTrue) MTrue ∧
      Mhat =
        mscIfTrue (C := C) (F := F) (Payload := Payload) c B
          ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                true
                (by
                  intro X hX
                  exact hX.1)
          ∘ₘ MTrue) ∧
      ¬ (∃ MFalse,
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PFalse) MFalse ∧
        Mhat =
          mscIfFalse (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false
                  (by
                    intro X hX
                    exact hX.1)
            ∘ₘ MFalse)) ∨
    ((∃ MFalse,
      distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PFalse) MFalse ∧
      Mhat =
        mscIfFalse (C := C) (F := F) (Payload := Payload) c B
          ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                false
                (by
                  intro X hX
                  exact hX.1)
          ∘ₘ MFalse) ∧
      ¬ (∃ MTrue,
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PTrue) MTrue ∧
        Mhat =
          mscIfTrue (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  true
                  (by
                    intro X hX
                    exact hX.1)
            ∘ₘ MTrue)) := by
  classical
  rcases hDist with ⟨hTrace, hComplete⟩
  let recips : L → Prop :=
    ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse
  have hMSC :
      IsMSC (L := L) (C := C) (F := F) (Payload := Payload) Mhat :=
    isCompleteMSC_implies_isMSC (L := L) (C := C) (F := F) (Payload := Payload) Mhat hComplete
  have hBProj :
      projectDist (L := L) (C := C) (F := F) (Payload := Payload)
        (.ite c B PTrue PFalse) B =
        LocProg.localIf c
          ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips true)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PTrue)
          ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips false)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PFalse) := by
    simp [projectDist, project, recips]
  have hBTrace :
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (LocProg.localIf c
          ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips true)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PTrue)
          ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips false)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PFalse))
        (Mhat B) := by
    rw [← hBProj]
    exact hTrace B
  simp [localTraceSemantics] at hBTrace
  rcases hBTrace with ⟨uB, huB, hMB⟩ | ⟨uB, huB, hMB⟩
  · rcases huB with ⟨sB, hsB, rB, hrB, huB⟩
    have hSWord :
        sB =
          controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
            B recips true :=
      controlBroadcast_trace_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B recips true sB hsB
    have hMB' :
        Mhat B =
          AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c ::
            (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              B recips true ++ rB) := by
      simpa [huB, hSWord, List.cons_append, List.append_assoc] using hMB
    have hRecipTrue :
        ∀ X, X ≠ B → recips X →
          ∃ uX,
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue) uX ∧
            Mhat X =
              AlphabetOf.mkRecv (C := C) (F := F) X
                (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: uX := by
      intro X hXB hRecip
      have hXProj :
          projectDist (L := L) (C := C) (F := F) (Payload := Payload)
            (.ite c B PTrue PFalse) X =
            LocProg.recvIf ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse) := by
        simp [projectDist, project, recips, hXB, hRecip]
      have hXTrace :
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (LocProg.recvIf ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse))
            (Mhat X) := by
        rw [← hXProj]
        exact hTrace X
      simp [localTraceSemantics] at hXTrace
      rcases hXTrace with hXTrue | hXFalse
      · exact hXTrue
      · rcases hXFalse with ⟨uX, huX, hXFalse⟩
        exact False.elim <|
          if_true_recipient_false_impossible
            (L := L) (C := C) (F := F) (Payload := Payload)
            c B X PTrue PFalse Mhat rB uX hRecip hMB' hXFalse hMSC
    let MTrue : WordTuple L C F Payload := fun X =>
      if hXB : X = B then
        by
          subst hXB
          exact rB
      else if hRecip : recips X then
        Classical.choose (hRecipTrue X hXB hRecip)
      else
        []
    have hTraceTrue :
        ∀ X,
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PTrue X)
            (MTrue X) := by
      intro X
      by_cases hXB : X = B
      · subst X
        dsimp [MTrue]
        simp [projectDist, hrB]
      · by_cases hRecip : recips X
        · simpa [MTrue, hXB, hRecip] using
            (Classical.choose_spec (hRecipTrue X hXB hRecip)).1
        · have hNotPartTrue : ¬ participationSet PTrue X := by
            intro hPart
            exact hRecip ⟨hXB, Or.inl hPart⟩
          simpa [MTrue, hXB, hRecip, projectDist] using
            (project_empty_trace_of_not_participating
              (L := L) (C := C) (F := F) (Payload := Payload)
              X PTrue hNotPartTrue)
    have hEqTrue :
        Mhat =
          mscIfTrue (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips true (by intro X hX; exact hX.1)
            ∘ₘ MTrue := by
      funext X
      by_cases hXB : X = B
      · subst X
        have hMTrueB : MTrue B = rB := by
          simp [MTrue]
        simpa [WordTuple.concat, mscIfTrue, choiceIfTrue, controlBroadcastMSC_decider,
          List.append_assoc, hMTrueB] using hMB'
      · by_cases hRecip : recips X
        · have hXEq := (Classical.choose_spec (hRecipTrue X hXB hRecip)).2
          have hMTrueX : MTrue X = Classical.choose (hRecipTrue X hXB hRecip) := by
            simp [MTrue, hXB, hRecip]
          have hChoice :
              mscIfTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceIfTrue (C := C) (F := F) (Payload := Payload) c B) hXB
          have hCtrl :
              (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips true (by intro Y hY; exact hY.1)) X =
                [AlphabetOf.mkRecv (C := C) (F := F) X
                  (ControlPayload.setDecision true ControlPayload.ctrlPattern) B] := by
            simpa [controlDecisionPayload] using
              (controlBroadcastMSC_recipient
                (C := C) (F := F) (Payload := Payload)
                B X recips true (by intro Y hY; exact hY.1) hRecip)
          simpa [WordTuple.concat, hChoice, hCtrl, hMTrueX] using hXEq
        · have hChoice :
              mscIfTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceIfTrue (C := C) (F := F) (Payload := Payload) c B) hXB
          have hCtrl :
              (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips true (by intro Y hY; exact hY.1)) X = [] :=
            controlBroadcastMSC_nonRecipient
              (C := C) (F := F) (Payload := Payload)
              B X recips true (by intro Y hY; exact hY.1) hXB hRecip
          have hXProj :
              projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                (.ite c B PTrue PFalse) X = LocProg.eps := by
            simp [projectDist, project, recips, hXB, hRecip]
          have hMXNil : Mhat X = [] := by
            have hTraceX := hTrace X
            rw [hXProj] at hTraceX
            simpa [localTraceSemantics] using hTraceX
          have hMTrueX : MTrue X = [] := by
            simp [MTrue, hXB, hRecip]
          simpa [WordTuple.concat, hChoice, hCtrl, hMTrueX] using hMXNil
    have hPrefixComplete :
        IsCompleteMSC
          (L := L) (C := C) (F := F) (Payload := Payload)
          (mscIfTrue (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips true (by intro X hX; exact hX.1)) := by
      exact concat_complete_complete _ _
        (mscIfTrue_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
        (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
          B recips true (by intro X hX; exact hX.1))
    have hAll :
        IsCompleteMSC
          (L := L) (C := C) (F := F) (Payload := Payload)
          (mscIfTrue (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips true (by intro X hX; exact hX.1)
            ∘ₘ MTrue) := by
      simpa [hEqTrue] using hComplete
    have hCompleteTrue :
        IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) MTrue :=
      complete_suffix_of_complete_prefix
        (L := L) (C := C) (F := F) (Payload := Payload)
        (mscIfTrue (C := C) (F := F) (Payload := Payload) c B
          ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips true (by intro X hX; exact hX.1))
        MTrue hPrefixComplete hAll
    have hNoFalse :
        ¬ (∃ MFalse,
          distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PFalse) MFalse ∧
          Mhat =
            mscIfFalse (C := C) (F := F) (Payload := Payload) c B
              ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B recips false (by intro X hX; exact hX.1)
              ∘ₘ MFalse) := by
      rintro ⟨MFalse, _hDistFalse, hEqFalse'⟩
      have hBFalse :
          Mhat B =
            AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c ::
              (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B recips false ++ MFalse B) := by
        have hB := congrFun hEqFalse' B
        simpa [WordTuple.concat, mscIfFalse, choiceIfFalse, controlBroadcastMSC_decider,
          List.append_assoc] using hB
      have hHead :
          AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c =
            AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :=
        (List.cons.inj (hMB'.symm.trans hBFalse)).1
      exact ifHead_true_ne_false (C := C) (F := F) (Payload := Payload) hHead
    exact Or.inl
      ⟨⟨MTrue, ⟨hTraceTrue, hCompleteTrue⟩, by simpa [recips] using hEqTrue⟩,
        by simpa [recips] using hNoFalse⟩
  · rcases huB with ⟨sB, hsB, rB, hrB, huB⟩
    have hSWord :
        sB =
          controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
            B recips false :=
      controlBroadcast_trace_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B recips false sB hsB
    have hMB' :
        Mhat B =
          AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c ::
            (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              B recips false ++ rB) := by
      simpa [huB, hSWord, List.cons_append, List.append_assoc] using hMB
    have hRecipFalse :
        ∀ X, X ≠ B → recips X →
          ∃ uX,
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse) uX ∧
            Mhat X =
              AlphabetOf.mkRecv (C := C) (F := F) X
                (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: uX := by
      intro X hXB hRecip
      have hXProj :
          projectDist (L := L) (C := C) (F := F) (Payload := Payload)
            (.ite c B PTrue PFalse) X =
            LocProg.recvIf ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse) := by
        simp [projectDist, project, recips, hXB, hRecip]
      have hXTrace :
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (LocProg.recvIf ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse))
            (Mhat X) := by
        rw [← hXProj]
        exact hTrace X
      simp [localTraceSemantics] at hXTrace
      rcases hXTrace with hXTrue | hXFalse
      · rcases hXTrue with ⟨uX, huX, hXTrue⟩
        exact False.elim <|
          if_false_recipient_true_impossible
            (L := L) (C := C) (F := F) (Payload := Payload)
            c B X PTrue PFalse Mhat rB uX hRecip hMB' hXTrue hMSC
      · exact hXFalse
    let MFalse : WordTuple L C F Payload := fun X =>
      if hXB : X = B then
        by
          subst hXB
          exact rB
      else if hRecip : recips X then
        Classical.choose (hRecipFalse X hXB hRecip)
      else
        []
    have hTraceFalse :
        ∀ X,
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PFalse X)
            (MFalse X) := by
      intro X
      by_cases hXB : X = B
      · subst X
        dsimp [MFalse]
        simp [projectDist, hrB]
      · by_cases hRecip : recips X
        · simpa [MFalse, hXB, hRecip] using
            (Classical.choose_spec (hRecipFalse X hXB hRecip)).1
        · have hNotPartFalse : ¬ participationSet PFalse X := by
            intro hPart
            exact hRecip ⟨hXB, Or.inr hPart⟩
          simpa [MFalse, hXB, hRecip, projectDist] using
            (project_empty_trace_of_not_participating
              (L := L) (C := C) (F := F) (Payload := Payload)
              X PFalse hNotPartFalse)
    have hEqFalse :
        Mhat =
          mscIfFalse (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips false (by intro X hX; exact hX.1)
            ∘ₘ MFalse := by
      funext X
      by_cases hXB : X = B
      · subst X
        have hMFalseB : MFalse B = rB := by
          simp [MFalse]
        simpa [WordTuple.concat, mscIfFalse, choiceIfFalse, controlBroadcastMSC_decider,
          List.append_assoc, hMFalseB] using hMB'
      · by_cases hRecip : recips X
        · have hXEq := (Classical.choose_spec (hRecipFalse X hXB hRecip)).2
          have hMFalseX : MFalse X = Classical.choose (hRecipFalse X hXB hRecip) := by
            simp [MFalse, hXB, hRecip]
          have hChoice :
              mscIfFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceIfFalse (C := C) (F := F) (Payload := Payload) c B) hXB
          have hCtrl :
              (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips false (by intro Y hY; exact hY.1)) X =
                [AlphabetOf.mkRecv (C := C) (F := F) X
                  (ControlPayload.setDecision false ControlPayload.ctrlPattern) B] := by
            simpa [controlDecisionPayload] using
              (controlBroadcastMSC_recipient
                (C := C) (F := F) (Payload := Payload)
                B X recips false (by intro Y hY; exact hY.1) hRecip)
          simpa [WordTuple.concat, hChoice, hCtrl, hMFalseX] using hXEq
        · have hChoice :
              mscIfFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceIfFalse (C := C) (F := F) (Payload := Payload) c B) hXB
          have hCtrl :
              (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips false (by intro Y hY; exact hY.1)) X = [] :=
            controlBroadcastMSC_nonRecipient
              (C := C) (F := F) (Payload := Payload)
              B X recips false (by intro Y hY; exact hY.1) hXB hRecip
          have hXProj :
              projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                (.ite c B PTrue PFalse) X = LocProg.eps := by
            simp [projectDist, project, recips, hXB, hRecip]
          have hMXNil : Mhat X = [] := by
            have hTraceX := hTrace X
            rw [hXProj] at hTraceX
            simpa [localTraceSemantics] using hTraceX
          have hMFalseX : MFalse X = [] := by
            simp [MFalse, hXB, hRecip]
          simpa [WordTuple.concat, hChoice, hCtrl, hMFalseX] using hMXNil
    have hPrefixComplete :
        IsCompleteMSC
          (L := L) (C := C) (F := F) (Payload := Payload)
          (mscIfFalse (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips false (by intro X hX; exact hX.1)) := by
      exact concat_complete_complete _ _
        (mscIfFalse_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
        (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
          B recips false (by intro X hX; exact hX.1))
    have hAll :
        IsCompleteMSC
          (L := L) (C := C) (F := F) (Payload := Payload)
          (mscIfFalse (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips false (by intro X hX; exact hX.1)
            ∘ₘ MFalse) := by
      simpa [hEqFalse] using hComplete
    have hCompleteFalse :
        IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) MFalse :=
      complete_suffix_of_complete_prefix
        (L := L) (C := C) (F := F) (Payload := Payload)
        (mscIfFalse (C := C) (F := F) (Payload := Payload) c B
          ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips false (by intro X hX; exact hX.1))
        MFalse hPrefixComplete hAll
    have hNoTrue :
        ¬ (∃ MTrue,
          distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PTrue) MTrue ∧
          Mhat =
            mscIfTrue (C := C) (F := F) (Payload := Payload) c B
              ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B recips true (by intro X hX; exact hX.1)
              ∘ₘ MTrue) := by
      rintro ⟨MTrue, _hDistTrue, hEqTrue'⟩
      have hBTrue :
          Mhat B =
            AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c ::
              (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B recips true ++ MTrue B) := by
        have hB := congrFun hEqTrue' B
        simpa [WordTuple.concat, mscIfTrue, choiceIfTrue, controlBroadcastMSC_decider,
          List.append_assoc] using hB
      have hHead :
          AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c =
            AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :=
        (List.cons.inj (hMB'.symm.trans hBTrue)).1
      exact ifHead_false_ne_true (C := C) (F := F) (Payload := Payload) hHead
    exact Or.inr
      ⟨⟨MFalse, ⟨hTraceFalse, hCompleteFalse⟩, by simpa [recips] using hEqFalse⟩,
        by simpa [recips] using hNoTrue⟩

private theorem concatLocalWords_foldl
    {A : L}
    (acc : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (ws : List (LocalWord (C := C) (F := F) (Payload := Payload) A)) :
    ws.foldl (· ++ ·) acc =
      acc ++ ws.foldl (· ++ ·) [] := by
  induction ws generalizing acc with
  | nil =>
      simp
  | cons w ws ih =>
      simp only [List.foldl_cons]
      rw [ih (acc ++ w)]
      simpa [List.append_assoc] using (ih w).symm

@[simp]
private theorem concatLocalWords_cons {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (ws : List (LocalWord (C := C) (F := F) (Payload := Payload) A)) :
    concatLocalWords (C := C) (F := F) (Payload := Payload) (w :: ws) =
      w ++ concatLocalWords (C := C) (F := F) (Payload := Payload) ws := by
  simpa [concatLocalWords] using
    (concatLocalWords_foldl (C := C) (F := F) (Payload := Payload) w ws)

@[simp]
private theorem concatLocalWords_append {A : L}
    (ws1 ws2 : List (LocalWord (C := C) (F := F) (Payload := Payload) A)) :
    concatLocalWords (C := C) (F := F) (Payload := Payload) (ws1 ++ ws2) =
      concatLocalWords (C := C) (F := F) (Payload := Payload) ws1 ++
      concatLocalWords (C := C) (F := F) (Payload := Payload) ws2 := by
  induction ws1 with
  | nil =>
      simp [concatLocalWords]
  | cons w ws ih =>
      simp [List.cons_append, concatLocalWords_cons, ih, List.append_assoc]

@[simp]
private theorem concatLocalWords_replicate_nil {A : L} (n : Nat) :
    concatLocalWords (C := C) (F := F) (Payload := Payload)
      (List.replicate n ([] : LocalWord (C := C) (F := F) (Payload := Payload) A)) = [] := by
  induction n with
  | zero =>
      simp [concatLocalWords]
  | succ n ih =>
      simp [List.replicate_succ, concatLocalWords_cons, ih]

@[simp]
private theorem concatList_apply
    (Ms : List (WordTuple L C F Payload)) (A : L) :
    WordTuple.concatList Ms A =
      concatLocalWords (C := C) (F := F) (Payload := Payload)
        (Ms.map (fun M => M A)) := by
  induction Ms with
  | nil =>
      simp [WordTuple.concatList, concatLocalWords, WordTuple.empty]
  | cons M Ms ih =>
      simp [WordTuple.concatList_cons, WordTuple.concat, ih]

@[simp]
private theorem concatList_append_singleton
    (Ms : List (WordTuple L C F Payload)) (M : WordTuple L C F Payload) :
    WordTuple.concatList (Ms ++ [M]) = WordTuple.concatList Ms ∘ₘ M := by
  induction Ms with
  | nil =>
      simp [WordTuple.concatList, WordTuple.concat_eps_left]
  | cons hd tl ih =>
      simp [WordTuple.concatList_cons, ih, WordTuple.concat_assoc]

private theorem ifRecipients_no_self
    (B : L) (PTrue PFalse : Prog L C F Payload) :
    ∀ X, ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X → X ≠ B := by
  intro X hX
  exact hX.1

private theorem whileRecipients_no_self
    (B : L) (PBody PExit : Prog L C F Payload) :
    ∀ X, whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X → X ≠ B := by
  intro X hX
  exact hX.1

/-- Auxiliary invariant used in the constructive correctness proof: lifelines
    outside the structural participation set carry the empty local word. -/
def EmptyOutsideParticipants
    (P : Prog L C F Payload) (M : WordTuple L C F Payload) : Prop :=
  ∀ A, ¬ participationSet P A → M A = []

/-- Distributed semantics is closed under pointwise sequential composition. -/
theorem distSemantics_seq
    (D1 D2 : DistProg L C F Payload)
    (M1 M2 : WordTuple L C F Payload)
    (h1 : distSemantics (L := L) (C := C) (F := F) (Payload := Payload) D1 M1)
    (h2 : distSemantics (L := L) (C := C) (F := F) (Payload := Payload) D2 M2) :
    distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (fun A => D1 A ;;ₗ D2 A) (M1 ∘ₘ M2) := by
  rcases h1 with ⟨h1Trace, h1Complete⟩
  rcases h2 with ⟨h2Trace, h2Complete⟩
  refine ⟨?_, concat_complete_complete _ _ h1Complete h2Complete⟩
  intro A
  refine ⟨M1 A, M2 A, h1Trace A, h2Trace A, rfl⟩

/-- The projected empty program has the empty MSC as a distributed trace. -/
theorem distSemantics_project_eps :
    distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload) Prog.eps)
      (mscEmpty (L := L) (C := C) (F := F) (Payload := Payload)) := by
  refine ⟨?_, mscEmpty_isCompleteMSC⟩
  intro A
  simp [projectDist, project, localTraceSemantics, mscEmpty, WordTuple.empty]

/-- Non-participating lifelines are empty in the empty MSC. -/
theorem emptyOutsideParticipants_eps :
    EmptyOutsideParticipants
      (L := L) (C := C) (F := F) (Payload := Payload)
      Prog.eps (mscEmpty (L := L) (C := C) (F := F) (Payload := Payload)) := by
  intro A hA
  simp [EmptyOutsideParticipants, mscEmpty, WordTuple.empty]

/-- The canonical message MSC realizes the distributed projection of a message
    statement, assuming user payloads are disjoint from the reserved control
    payload space. -/
theorem distSemantics_project_msg
    (A : L) (xs : Payload) (B : L) (ys : Payload) (h : A ≠ B)
    (hCompat : PayloadCompatible Payload xs ys) :
    distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
        (.msg A xs B ys h))
      (mscMsg (C := C) (F := F) A xs B ys h) := by
  refine ⟨?_, mscMsg_isCompleteMSC (C := C) (F := F) A xs B ys h hCompat⟩
  · intro X
    by_cases hXA : X = A
    · subst hXA
      simpa [projectDist, project, localTraceSemantics, mscMsg_sender] using
        (Exists.intro h rfl)
    · by_cases hXB : X = B
      · subst hXB
        simp [projectDist, project, h, hXA, localTraceSemantics, mscMsg_receiver]
      · simp [projectDist, project, hXA, hXB, localTraceSemantics, mscMsg_other]

/-- The canonical action MSC realizes the distributed projection of a local
    action. -/
theorem distSemantics_project_act
    (A : L) (ys : Payload) (f : F) (xs : Payload) :
    distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
        (.act A ys f xs))
      (mscAct (C := C) (F := F) A ys f xs) := by
  refine ⟨?_, mscAct_isCompleteMSC (C := C) (F := F) (Payload := Payload) A ys f xs⟩
  intro X
  by_cases hXA : X = A
  · subst hXA
    simp [projectDist, project, localTraceSemantics, mscAct_owner]
  · simp [projectDist, project, hXA, localTraceSemantics, mscAct_other]

/-- The projected `if`-program realizes the true branch by prepending the
    canonical choice MSC and the projection-generated control broadcast. -/
theorem distSemantics_project_if_true
    (c : C) (B : L) (PTrue PFalse : Prog L C F Payload)
    (MTrue : WordTuple L C F Payload)
    (hTrue : distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PTrue) MTrue) :
    distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
        (.ite c B PTrue PFalse))
      (mscIfTrue (C := C) (F := F) (Payload := Payload) c B
        ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              true
              (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                B PTrue PFalse)
        ∘ₘ MTrue) := by
  rcases hTrue with ⟨hTraceTrue, hCompleteTrue⟩
  refine ⟨?_, ?_⟩
  · intro X
    by_cases hXB : X = B
    · subst hXB
      simpa [projectDist, project, localTraceSemantics, WordTuple.concat_assoc,
        controlBroadcastMSC_decider] using
        (Or.inl (show
          ∃ u,
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  X
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
                  true)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue) u ∧
            ((mscIfTrue (C := C) (F := F) (Payload := Payload) c X
                ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    X
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
                    true
                    (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                      X PTrue PFalse)
                ∘ₘ MTrue) X) =
              AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) X c :: u from
          ⟨(controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              X
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
              true)
            ++ MTrue X,
            localTraceSemantics_seq_intro
              (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                X
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
                true)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
              _ _
              (controlBroadcast_trace (L := L) (C := C) (F := F) (Payload := Payload)
                X
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
                true)
              (hTraceTrue X),
            by
              simp [WordTuple.concat, mscIfTrue, choiceIfTrue,
                controlBroadcastMSC_decider, List.append_assoc]⟩))
    · by_cases hRecip :
        ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X
      · simpa [projectDist, project, hXB, hRecip, localTraceSemantics,
          WordTuple.concat_assoc] using
          (Or.inl (show
            ∃ u,
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue) u ∧
              ((mscIfTrue (C := C) (F := F) (Payload := Payload) c B
                  ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                        B
                        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                        true
                        (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                          B PTrue PFalse)
                  ∘ₘ MTrue) X) =
                AlphabetOf.mkRecv (C := C) (F := F) X
                  (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: u from
            ⟨MTrue X, hTraceTrue X, by
              have hChoice :
                  mscIfTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
                mscChoice_other (C := C) (F := F) (Payload := Payload)
                  B X (choiceIfTrue (C := C) (F := F) (Payload := Payload) c B) hXB
              have hCtrl :
                  (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                    true
                    (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                      B PTrue PFalse)) X =
                  [AlphabetOf.mkRecv (C := C) (F := F) X
                    (ControlPayload.setDecision true ControlPayload.ctrlPattern) B] :=
                controlBroadcastMSC_recipient (C := C) (F := F) (Payload := Payload)
                  B X
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  true
                  (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                    B PTrue PFalse)
                  hRecip
              simp [WordTuple.concat, hChoice, hCtrl]⟩))
      · have hNoPartTrue : ¬ participationSet PTrue X := by
          intro hPart
          exact hRecip ⟨hXB, Or.inl hPart⟩
        have hMTrueNil : MTrue X = [] :=
          project_trace_nil_of_not_participating
            (L := L) (C := C) (F := F) (Payload := Payload) X PTrue hNoPartTrue
            (MTrue X) (hTraceTrue X)
        have hXB' : X ≠ B := hXB
        have hBroadcastNil :
            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              true
              (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                B PTrue PFalse)) X = [] :=
          controlBroadcastMSC_nonRecipient (C := C) (F := F) (Payload := Payload)
            B X
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true
            (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
              B PTrue PFalse)
            hXB' hRecip
        have hWord :
            (mscIfTrue (C := C) (F := F) (Payload := Payload) c B
              ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                    true
                    (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                      B PTrue PFalse)
              ∘ₘ MTrue) X = [] := by
          have hChoice :
              mscIfTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceIfTrue (C := C) (F := F) (Payload := Payload) c B) hXB'
          simp [WordTuple.concat, hChoice, hBroadcastNil, hMTrueNil]
        simpa [projectDist, project, hXB, hRecip, localTraceSemantics, hWord] using
          (localTraceSemantics_eps_nil
            (L := L) (C := C) (F := F) (Payload := Payload) (A := X))
  · apply concat_complete_complete
    · exact concat_complete_complete _ _
        (mscIfTrue_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
        (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          true
          (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
            B PTrue PFalse))
    · exact hCompleteTrue

/-- The projected `if`-program realizes the false branch symmetrically. -/
theorem distSemantics_project_if_false
    (c : C) (B : L) (PTrue PFalse : Prog L C F Payload)
    (MFalse : WordTuple L C F Payload)
    (hFalse : distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PFalse) MFalse) :
    distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
        (.ite c B PTrue PFalse))
      (mscIfFalse (C := C) (F := F) (Payload := Payload) c B
        ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              false
              (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                B PTrue PFalse)
        ∘ₘ MFalse) := by
  rcases hFalse with ⟨hTraceFalse, hCompleteFalse⟩
  refine ⟨?_, ?_⟩
  · intro X
    by_cases hXB : X = B
    · subst hXB
      simpa [projectDist, project, localTraceSemantics, WordTuple.concat_assoc,
        controlBroadcastMSC_decider] using
        (Or.inr (show
          ∃ u,
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  X
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
                  false)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse) u ∧
            ((mscIfFalse (C := C) (F := F) (Payload := Payload) c X
                ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    X
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
                    false
                    (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                      X PTrue PFalse)
                ∘ₘ MFalse) X) =
              AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) X c :: u from
          ⟨(controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              X
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
              false)
            ++ MFalse X,
            localTraceSemantics_seq_intro
              (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                X
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
                false)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
              _ _
              (controlBroadcast_trace (L := L) (C := C) (F := F) (Payload := Payload)
                X
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PTrue PFalse)
                false)
              (hTraceFalse X),
            by
              simp [WordTuple.concat, mscIfFalse, choiceIfFalse,
                controlBroadcastMSC_decider, List.append_assoc]⟩))
    · by_cases hRecip :
        ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X
      · simpa [projectDist, project, hXB, hRecip, localTraceSemantics,
          WordTuple.concat_assoc] using
          (Or.inr (show
            ∃ u,
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse) u ∧
              ((mscIfFalse (C := C) (F := F) (Payload := Payload) c B
                  ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                        B
                        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                        false
                        (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                          B PTrue PFalse)
                  ∘ₘ MFalse) X) =
                AlphabetOf.mkRecv (C := C) (F := F) X
                  (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: u from
            ⟨MFalse X, hTraceFalse X, by
              have hChoice :
                  mscIfFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
                mscChoice_other (C := C) (F := F) (Payload := Payload)
                  B X (choiceIfFalse (C := C) (F := F) (Payload := Payload) c B) hXB
              have hCtrl :
                  (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                    false
                    (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                      B PTrue PFalse)) X =
                  [AlphabetOf.mkRecv (C := C) (F := F) X
                    (ControlPayload.setDecision false ControlPayload.ctrlPattern) B] :=
                controlBroadcastMSC_recipient (C := C) (F := F) (Payload := Payload)
                  B X
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false
                  (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                    B PTrue PFalse)
                  hRecip
              simp [WordTuple.concat, hChoice, hCtrl]⟩))
      · have hNoPartFalse : ¬ participationSet PFalse X := by
          intro hPart
          exact hRecip ⟨hXB, Or.inr hPart⟩
        have hMFalseNil : MFalse X = [] :=
          project_trace_nil_of_not_participating
            (L := L) (C := C) (F := F) (Payload := Payload) X PFalse hNoPartFalse
            (MFalse X) (hTraceFalse X)
        have hXB' : X ≠ B := hXB
        have hBroadcastNil :
            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              false
              (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                B PTrue PFalse)) X = [] :=
          controlBroadcastMSC_nonRecipient (C := C) (F := F) (Payload := Payload)
            B X
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false
            (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
              B PTrue PFalse)
            hXB' hRecip
        have hWord :
            (mscIfFalse (C := C) (F := F) (Payload := Payload) c B
              ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                    false
                    (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                      B PTrue PFalse)
              ∘ₘ MFalse) X = [] := by
          have hChoice :
              mscIfFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceIfFalse (C := C) (F := F) (Payload := Payload) c B) hXB'
          simp [WordTuple.concat, hChoice, hBroadcastNil, hMFalseNil]
        simpa [projectDist, project, hXB, hRecip, localTraceSemantics, hWord] using
          (localTraceSemantics_eps_nil
            (L := L) (C := C) (F := F) (Payload := Payload) (A := X))
  · apply concat_complete_complete
    · exact concat_complete_complete _ _
        (mscIfFalse_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
        (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          false
          (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
            B PTrue PFalse))
    · exact hCompleteFalse

/-- Complete distributed traces of a projected `while` decompose into a finite
    sequence of complete body realizations followed by a complete exit
    realization, with each component preceded by the corresponding projected
    control decision. -/
theorem distSemantics_while_decompose
    (c : C) (B : L) (PBody PExit : Prog L C F Payload)
    (hWFBody : WellTypedProgram PBody)
    (Mhat : WordTuple L C F Payload)
    (hDist :
      distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
          (.whileLoop c B PBody PExit))
        Mhat) :
    ∃ k : Nat,
      ∃ bodiesHat : Fin k → WordTuple L C F Payload,
      (∀ i,
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PBody)
          (bodiesHat i)) ∧
      ∃ exitHat,
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PExit)
          exitHat ∧
        Mhat =
          WordTuple.concatList
            (List.ofFn (fun i =>
              mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
                ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      true
                      (whileRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                ∘ₘ bodiesHat i) ++
             [mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
                ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      false
                      (whileRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                ∘ₘ exitHat]) := by
  classical
  let decompResult : WordTuple L C F Payload → Prop := fun Mhat =>
    ∃ k : Nat,
      ∃ bodiesHat : Fin k → WordTuple L C F Payload,
      (∀ i,
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PBody)
          (bodiesHat i)) ∧
      ∃ exitHat,
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PExit)
          exitHat ∧
        Mhat =
          WordTuple.concatList
            (List.ofFn (fun i =>
              mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
                ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      true
                      (whileRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                ∘ₘ bodiesHat i) ++
             [mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
                ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      false
                      (whileRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                ∘ₘ exitHat])
  change decompResult Mhat
  let motive : Nat → Prop := fun n =>
    ∀ Mhat : WordTuple L C F Payload,
      (Mhat B).length = n →
      distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
          (.whileLoop c B PBody PExit))
        Mhat →
      decompResult Mhat
  refine (Nat.strongRecOn (Mhat B).length (motive := motive) ?_) Mhat rfl hDist
  intro n ih Mhat hn hDist
  rcases hDist with ⟨hTrace, hComplete⟩
  let recips : L → Prop :=
    whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit
  have hMSC :
      IsMSC (L := L) (C := C) (F := F) (Payload := Payload) Mhat :=
    isCompleteMSC_implies_isMSC (L := L) (C := C) (F := F) (Payload := Payload)
      Mhat hComplete
  have hBProj :
      projectDist (L := L) (C := C) (F := F) (Payload := Payload)
        (.whileLoop c B PBody PExit) B =
        LocProg.localWhile c
          ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips true)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
          ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips false)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit) := by
    simp [projectDist, project, recips]
  have hBTrace :
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (LocProg.localWhile c
          ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips true)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
          ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips false)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit))
        (Mhat B) := by
    rw [← hBProj]
    exact hTrace B
  rcases (localTraceSemantics_localWhile_unfold
      (L := L) (C := C) (F := F) (Payload := Payload)
      c
      ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips true)
        ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
      ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) B recips false)
        ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit)
      (Mhat B)).mp hBTrace with
    ⟨exitWordB, hExitSeqB, hMB⟩ | ⟨bodyWordB, restB, hBodySeqB, hRestB, hMB⟩
  · rcases hExitSeqB with ⟨sB, rB, hsB, hrB, hExitEqB⟩
    have hSWord :
        sB =
          controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
            B recips false :=
      controlBroadcast_trace_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B recips false sB hsB
    have hMB' :
        Mhat B =
          AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
            (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              B recips false ++ rB) := by
      simpa [hExitEqB, hSWord, List.cons_append, List.append_assoc] using hMB
    have hRecipExit :
        ∀ X, X ≠ B → recips X →
          ∃ uX,
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit) uX ∧
            Mhat X =
              AlphabetOf.mkRecv (C := C) (F := F) X
                (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: uX := by
      intro X hXB hRecip
      have hXProj :
          projectDist (L := L) (C := C) (F := F) (Payload := Payload)
            (.whileLoop c B PBody PExit) X =
            LocProg.recvWhile ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit) := by
        simp [projectDist, project, recips, hXB, hRecip]
      have hXTrace :
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (LocProg.recvWhile ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit))
            (Mhat X) := by
        rw [← hXProj]
        exact hTrace X
      rcases (localTraceSemantics_recvWhile_unfold
          (L := L) (C := C) (F := F) (Payload := Payload)
          ControlPayload.ctrlPattern B
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
          (Mhat X)).mp hXTrace with
        ⟨exitWordX, hExitX, hMX⟩ | ⟨bodyWordX, restX, _hBodyX, _hRestX, hMX⟩
      · exact ⟨exitWordX, hExitX, hMX⟩
      · exact False.elim <|
          while_false_recipient_true_impossible
            (L := L) (C := C) (F := F) (Payload := Payload)
            c B X PBody PExit Mhat rB (bodyWordX ++ restX)
            hRecip hMB' (by simpa [List.cons_append] using hMX) hMSC
    let exitHat : WordTuple L C F Payload := fun X =>
      if hXB : X = B then
        by
          subst hXB
          exact rB
      else if hRecip : recips X then
        Classical.choose (hRecipExit X hXB hRecip)
      else
        []
    have hTraceExit :
        ∀ X,
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PExit X)
            (exitHat X) := by
      intro X
      by_cases hXB : X = B
      · subst X
        dsimp [exitHat]
        simp [projectDist, hrB]
      · by_cases hRecip : recips X
        · simpa [exitHat, hXB, hRecip] using
            (Classical.choose_spec (hRecipExit X hXB hRecip)).1
        · have hNotPartExit : ¬ participationSet PExit X := by
            intro hPart
            exact hRecip ⟨hXB, Or.inr hPart⟩
          simpa [exitHat, hXB, hRecip, projectDist] using
            (project_empty_trace_of_not_participating
              (L := L) (C := C) (F := F) (Payload := Payload)
              X PExit hNotPartExit)
    have hEqExit :
        Mhat =
          mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips false (whileRecipients_no_self
                    (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            ∘ₘ exitHat := by
      funext X
      by_cases hXB : X = B
      · subst X
        have hExitHatB : exitHat B = rB := by
          simp [exitHat]
        simpa [WordTuple.concat, mscWhileFalse, choiceWhileFalse,
          controlBroadcastMSC_decider, hExitHatB, List.append_assoc] using hMB'
      · by_cases hRecip : recips X
        · have hXEq := (Classical.choose_spec (hRecipExit X hXB hRecip)).2
          have hExitHatX :
              exitHat X = Classical.choose (hRecipExit X hXB hRecip) := by
            simp [exitHat, hXB, hRecip]
          have hChoice :
              mscWhileFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceWhileFalse (C := C) (F := F) (Payload := Payload) c B) hXB
          have hCtrl :
              (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips false (whileRecipients_no_self
                  (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)) X =
                [AlphabetOf.mkRecv (C := C) (F := F) X
                  (ControlPayload.setDecision false ControlPayload.ctrlPattern) B] := by
            simpa [controlDecisionPayload] using
              (controlBroadcastMSC_recipient
                (C := C) (F := F) (Payload := Payload)
                B X recips false (whileRecipients_no_self
                  (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                hRecip)
          simpa [WordTuple.concat, hChoice, hCtrl, hExitHatX] using hXEq
        · have hChoice :
              mscWhileFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceWhileFalse (C := C) (F := F) (Payload := Payload) c B) hXB
          have hCtrl :
              (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips false (whileRecipients_no_self
                  (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)) X = [] :=
            controlBroadcastMSC_nonRecipient
              (C := C) (F := F) (Payload := Payload)
              B X recips false (whileRecipients_no_self
                (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              hXB hRecip
          have hXProj :
              projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                (.whileLoop c B PBody PExit) X = LocProg.eps := by
            simp [projectDist, project, recips, hXB, hRecip]
          have hMXNil : Mhat X = [] := by
            have hTraceX := hTrace X
            rw [hXProj] at hTraceX
            simpa [localTraceSemantics] using hTraceX
          have hExitHatX : exitHat X = [] := by
            simp [exitHat, hXB, hRecip]
          simpa [WordTuple.concat, hChoice, hCtrl, hExitHatX] using hMXNil
    have hPrefixComplete :
        IsCompleteMSC
          (L := L) (C := C) (F := F) (Payload := Payload)
          (mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips false (whileRecipients_no_self
                    (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)) := by
      exact concat_complete_complete _ _
        (mscWhileFalse_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
        (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
          B recips false (whileRecipients_no_self
            (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit))
    have hAll :
        IsCompleteMSC
          (L := L) (C := C) (F := F) (Payload := Payload)
          (mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips false (whileRecipients_no_self
                    (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            ∘ₘ exitHat) := by
      simpa [hEqExit] using hComplete
    have hCompleteExit :
        IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) exitHat :=
      complete_suffix_of_complete_prefix
        (L := L) (C := C) (F := F) (Payload := Payload)
        (mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
          ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips false (whileRecipients_no_self
                  (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit))
        exitHat hPrefixComplete hAll
    refine ⟨0, Fin.elim0, ?_, exitHat, ⟨hTraceExit, hCompleteExit⟩, ?_⟩
    · intro i
      exact Fin.elim0 i
    · simpa [WordTuple.concatList, WordTuple.concat_eps_left] using hEqExit
  · rcases hBodySeqB with ⟨sB, rB, hsB, hrB, hBodyEqB⟩
    have hSWord :
        sB =
          controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
            B recips true :=
      controlBroadcast_trace_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B recips true sB hsB
    have hMB' :
        Mhat B =
          AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
            (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              B recips true ++ rB ++ restB) := by
      simpa [hBodyEqB, hSWord, List.cons_append, List.append_assoc] using hMB
    have hRecipBodyRest :
        ∀ X, X ≠ B → recips X →
          ∃ bodyX,
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody) bodyX ∧
          ∃ restX,
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                (.whileLoop c B PBody PExit) X) restX ∧
            Mhat X =
              AlphabetOf.mkRecv (C := C) (F := F) X
                (ControlPayload.setDecision true ControlPayload.ctrlPattern) B ::
                bodyX ++ restX := by
      intro X hXB hRecip
      have hXProj :
          projectDist (L := L) (C := C) (F := F) (Payload := Payload)
            (.whileLoop c B PBody PExit) X =
            LocProg.recvWhile ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit) := by
        simp [projectDist, project, recips, hXB, hRecip]
      have hXTrace :
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (LocProg.recvWhile ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit))
            (Mhat X) := by
        rw [← hXProj]
        exact hTrace X
      rcases (localTraceSemantics_recvWhile_unfold
          (L := L) (C := C) (F := F) (Payload := Payload)
          ControlPayload.ctrlPattern B
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
          (Mhat X)).mp hXTrace with
        ⟨exitWordX, _hExitX, hMX⟩ | ⟨bodyWordX, restX, hBodyX, hRestX, hMX⟩
      · exact False.elim <|
          while_true_recipient_false_impossible
            (L := L) (C := C) (F := F) (Payload := Payload)
            c B X PBody PExit Mhat (rB ++ restB) exitWordX
            hRecip (by simpa [List.append_assoc] using hMB') hMX hMSC
      · refine ⟨bodyWordX, hBodyX, restX, ?_, ?_⟩
        · simpa [hXProj] using hRestX
        · simpa [List.cons_append] using hMX
    let MBody : WordTuple L C F Payload := fun X =>
      if hXB : X = B then
        by
          subst hXB
          exact rB
      else if hRecip : recips X then
        Classical.choose (hRecipBodyRest X hXB hRecip)
      else
        []
    let MRest : WordTuple L C F Payload := fun X =>
      if hXB : X = B then
        by
          subst hXB
          exact restB
      else if hRecip : recips X then
        Classical.choose ((Classical.choose_spec (hRecipBodyRest X hXB hRecip)).2)
      else
        []
    have hTraceBody :
        ∀ X,
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PBody X)
            (MBody X) := by
      intro X
      by_cases hXB : X = B
      · subst X
        dsimp [MBody]
        simp [projectDist, hrB]
      · by_cases hRecip : recips X
        · simpa [MBody, hXB, hRecip] using
            (Classical.choose_spec (hRecipBodyRest X hXB hRecip)).1
        · have hNotPartBody : ¬ participationSet PBody X := by
            intro hPart
            exact hRecip ⟨hXB, Or.inl hPart⟩
          simpa [MBody, hXB, hRecip, projectDist] using
            (project_empty_trace_of_not_participating
              (L := L) (C := C) (F := F) (Payload := Payload)
              X PBody hNotPartBody)
    have hTraceRest :
        ∀ X,
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
              (.whileLoop c B PBody PExit) X)
            (MRest X) := by
      intro X
      by_cases hXB : X = B
      · subst X
        have hMRestB' : MRest B = restB := by
          simp [MRest]
        rw [hMRestB']
        rw [hBProj]
        exact hRestB
      · by_cases hRecip : recips X
        · simpa [MRest, hXB, hRecip] using
            (Classical.choose_spec
              ((Classical.choose_spec (hRecipBodyRest X hXB hRecip)).2)).1
        · have hProj :
              projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                (.whileLoop c B PBody PExit) X = LocProg.eps := by
            simp [projectDist, project, recips, hXB, hRecip]
          simpa [MRest, hXB, hRecip, hProj, localTraceSemantics]
    have hEqStep :
        Mhat =
          mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                  B recips true (whileRecipients_no_self
                    (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            ∘ₘ MBody ∘ₘ MRest := by
      funext X
      by_cases hXB : X = B
      · subst X
        have hMBodyB : MBody B = rB := by
          simp [MBody]
        have hMRestB : MRest B = restB := by
          simp [MRest]
        simpa [WordTuple.concat, mscWhileTrue, choiceWhileTrue,
          controlBroadcastMSC_decider, hMBodyB, hMRestB, List.append_assoc] using hMB'
      · by_cases hRecip : recips X
        · have hSpec := hRecipBodyRest X hXB hRecip
          have hBodyX : MBody X = Classical.choose hSpec := by
            simp [MBody, hXB, hRecip]
          have hRestX :
              MRest X = Classical.choose ((Classical.choose_spec hSpec).2) := by
            simp [MRest, hXB, hRecip]
          have hXEq :=
            (Classical.choose_spec ((Classical.choose_spec hSpec).2)).2
          have hChoice :
              mscWhileTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceWhileTrue (C := C) (F := F) (Payload := Payload) c B) hXB
          have hCtrl :
              (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips true (whileRecipients_no_self
                  (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)) X =
                [AlphabetOf.mkRecv (C := C) (F := F) X
                  (ControlPayload.setDecision true ControlPayload.ctrlPattern) B] := by
            simpa [controlDecisionPayload] using
              (controlBroadcastMSC_recipient
                (C := C) (F := F) (Payload := Payload)
                B X recips true (whileRecipients_no_self
                  (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                hRecip)
          simpa [WordTuple.concat, hChoice, hCtrl, hBodyX, hRestX,
            List.append_assoc, List.cons_append] using hXEq
        · have hChoice :
              mscWhileTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceWhileTrue (C := C) (F := F) (Payload := Payload) c B) hXB
          have hCtrl :
              (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B recips true (whileRecipients_no_self
                  (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)) X = [] :=
            controlBroadcastMSC_nonRecipient
              (C := C) (F := F) (Payload := Payload)
              B X recips true (whileRecipients_no_self
                (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              hXB hRecip
          have hXProj :
              projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                (.whileLoop c B PBody PExit) X = LocProg.eps := by
            simp [projectDist, project, recips, hXB, hRecip]
          have hMXNil : Mhat X = [] := by
            have hTraceX := hTrace X
            rw [hXProj] at hTraceX
            simpa [localTraceSemantics] using hTraceX
          have hMBodyX : MBody X = [] := by
            simp [MBody, hXB, hRecip]
          have hMRestX : MRest X = [] := by
            simp [MRest, hXB, hRecip]
          simpa [WordTuple.concat, hChoice, hCtrl, hMBodyX, hMRestX] using hMXNil
    let stepPrefix : WordTuple L C F Payload :=
      mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
        ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B recips true (whileRecipients_no_self
                (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
    have hStepPrefixComplete :
        IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) stepPrefix := by
      exact concat_complete_complete _ _
        (mscWhileTrue_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
        (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
          B recips true (whileRecipients_no_self
            (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit))
    have hMSCBodyRest :
        IsMSC (L := L) (C := C) (F := F) (Payload := Payload) (MBody ∘ₘ MRest) :=
      suffix_msc_of_complete_prefix
        (L := L) (C := C) (F := F) (Payload := Payload)
        stepPrefix (MBody ∘ₘ MRest) hStepPrefixComplete
        (by
          simpa [stepPrefix, WordTuple.concat_assoc, hEqStep] using hMSC)
    have hZip :=
      uniformZipper (L := L) (C := C) (F := F) (Payload := Payload) PBody
        hWFBody MBody MRest
        (fun X =>
          localTrace_implies_localPrefix
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
            (MBody X) (by simpa [projectDist] using hTraceBody X))
        (fun X _hX => by simpa [projectDist] using hTraceBody X)
        hMSCBodyRest
    rcases hZip with ⟨_MBodyBar, hZipPost⟩
    have hCompleteBody :
        IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) MBody :=
      zipPost_left_isCompleteMSC
        (L := L) (C := C) (F := F) (Payload := Payload)
        hZipPost
        (fun X => by simpa [projectDist] using hTraceBody X)
    have hPrefixBodyComplete :
        IsCompleteMSC
          (L := L) (C := C) (F := F) (Payload := Payload)
          (stepPrefix ∘ₘ MBody) :=
      concat_complete_complete _ _ hStepPrefixComplete hCompleteBody
    have hAllBodyRest :
        IsCompleteMSC
          (L := L) (C := C) (F := F) (Payload := Payload)
          ((stepPrefix ∘ₘ MBody) ∘ₘ MRest) := by
      simpa [stepPrefix, WordTuple.concat_assoc, hEqStep] using hComplete
    have hCompleteRest :
        IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) MRest :=
      complete_suffix_of_complete_prefix
        (L := L) (C := C) (F := F) (Payload := Payload)
        (stepPrefix ∘ₘ MBody) MRest hPrefixBodyComplete hAllBodyRest
    have hBodyDist :
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PBody) MBody :=
      ⟨hTraceBody, hCompleteBody⟩
    have hRestDist :
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
            (.whileLoop c B PBody PExit)) MRest :=
      ⟨hTraceRest, hCompleteRest⟩
    have hMRestB : MRest B = restB := by
      simp [MRest]
    have hRestLt : (MRest B).length < n := by
      rw [hMRestB]
      have hLenDrop : restB.length < (Mhat B).length := by
        rw [hMB']
        simp [List.length_append]
        omega
      omega
    rcases ih (MRest B).length hRestLt MRest rfl hRestDist with
      ⟨kRest, bodiesRest, hBodiesRest, exitHat, hExitHat, hRestEq⟩
    let bodiesHat : Fin (kRest + 1) → WordTuple L C F Payload :=
      Fin.cases MBody bodiesRest
    refine ⟨kRest + 1, bodiesHat, ?_, exitHat, hExitHat, ?_⟩
    · intro i
      subst bodiesHat
      exact Fin.cases hBodyDist hBodiesRest i
    · calc
        Mhat =
            (mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
              ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B recips true (whileRecipients_no_self
                      (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              ∘ₘ MBody) ∘ₘ MRest := by
                simpa [WordTuple.concat_assoc] using hEqStep
        _ =
            WordTuple.concatList
              (List.ofFn (fun i =>
                mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
                  ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                        B
                        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                        true
                        (whileRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                  ∘ₘ bodiesHat i) ++
               [mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
                  ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                        B
                        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                        false
                        (whileRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                  ∘ₘ exitHat]) := by
                rw [hRestEq]
                simp [bodiesHat, recips, List.ofFn_succ, WordTuple.concatList_cons,
                  WordTuple.concat_assoc]

/-- The constructive direction of Theorem `thm:correctness`:
    every global MSC has an enriched distributed realization whose erasure is
    the original MSC. -/
theorem realization_complete
    (prog : Prog L C F Payload)
    (hWellFormed : WellTypedProgram prog)
    (hCtrl : ControlDistinguishableProgram (L := L) (C := C) (F := F) (Payload := Payload) prog) :
    ∀ M, ⟦prog⟧ M →
      ∃ Mhat,
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) prog) Mhat ∧
        eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) Mhat = M := by
  induction prog with
  | eps =>
      intro M hM
      simp [mscSemantics] at hM
      subst hM
      refine ⟨mscEmpty, distSemantics_project_eps (L := L) (C := C) (F := F) (Payload := Payload), ?_⟩
      simpa using (eraseTuple_mscEmpty (L := L) (C := C) (F := F) (Payload := Payload))
  | msg A xs B ys h =>
      intro M hM
      simp [mscSemantics] at hM
      subst hM
      rcases hCtrl with ⟨hxs, hys⟩
      refine ⟨mscMsg (C := C) (F := F) A xs B ys h,
        distSemantics_project_msg (L := L) (C := C) (F := F) (Payload := Payload)
          A xs B ys h (by simpa [WellTypedProgram] using hWellFormed), ?_⟩
      simpa [eraseTuple_mscMsg] using
        (eraseTuple_mscMsg (L := L) (C := C) (F := F) (Payload := Payload) A xs B ys h hxs hys)
  | act A ys f xs =>
      intro M hM
      simp [mscSemantics] at hM
      subst hM
      refine ⟨mscAct (C := C) (F := F) A ys f xs,
        distSemantics_project_act (L := L) (C := C) (F := F) (Payload := Payload) A ys f xs, ?_⟩
      simpa using
        (eraseTuple_mscAct (L := L) (C := C) (F := F) (Payload := Payload) A ys f xs)
  | seq P1 P2 ih1 ih2 =>
      intro M hM
      rcases hWellFormed with ⟨hWF1, hWF2⟩
      rcases hCtrl with ⟨hCtrl1, hCtrl2⟩
      simp [mscSemantics] at hM
      rcases hM with ⟨M1, hM1, M2, hM2, rfl⟩
      rcases ih1 hWF1 hCtrl1 M1 hM1 with ⟨Mhat1, hDist1, hErase1⟩
      rcases ih2 hWF2 hCtrl2 M2 hM2 with ⟨Mhat2, hDist2, hErase2⟩
      refine ⟨Mhat1 ∘ₘ Mhat2,
        distSemantics_seq (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) P1)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) P2)
          Mhat1 Mhat2 hDist1 hDist2, ?_⟩
      simp [hErase1, hErase2]
  | ite c B PTrue PFalse ihTrue ihFalse =>
      intro M hM
      rcases hWellFormed with ⟨hWFTrue, hWFFalse⟩
      rcases hCtrl with ⟨hCtrlTrue, hCtrlFalse⟩
      simp [mscSemantics] at hM
      rcases hM with ⟨MTrue, hMTrue, rfl⟩ | ⟨MFalse, hMFalse, rfl⟩
      · rcases ihTrue hWFTrue hCtrlTrue MTrue hMTrue with ⟨MhatTrue, hDistTrue, hEraseTrue⟩
        refine ⟨mscIfTrue (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                true
                (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                  B PTrue PFalse)
            ∘ₘ MhatTrue,
          distSemantics_project_if_true (L := L) (C := C) (F := F) (Payload := Payload)
            c B PTrue PFalse MhatTrue hDistTrue, ?_⟩
        rw [eraseTuple_concat, eraseTuple_concat,
          eraseTuple_mscIfTrue, erase_controlBroadcastMSC, hEraseTrue,
          WordTuple.concat_assoc]
        funext X
        simp [WordTuple.concat, mscEmpty, WordTuple.empty]
      · rcases ihFalse hWFFalse hCtrlFalse MFalse hMFalse with ⟨MhatFalse, hDistFalse, hEraseFalse⟩
        refine ⟨mscIfFalse (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                false
                (ifRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                  B PTrue PFalse)
            ∘ₘ MhatFalse,
          distSemantics_project_if_false (L := L) (C := C) (F := F) (Payload := Payload)
            c B PTrue PFalse MhatFalse hDistFalse, ?_⟩
        rw [eraseTuple_concat, eraseTuple_concat,
          eraseTuple_mscIfFalse, erase_controlBroadcastMSC, hEraseFalse,
          WordTuple.concat_assoc]
        funext X
        simp [WordTuple.concat, mscEmpty, WordTuple.empty]
  | whileLoop c B PBody PExit ihBody ihExit =>
      classical
      intro M hM
      rcases hWellFormed with ⟨hWFBody, hWFExit⟩
      rcases hCtrl with ⟨hCtrlBody, hCtrlExit⟩
      simp [mscSemantics] at hM
      rcases hM with ⟨k, bodies, hBodies, exitMSC, hExitMSC, rfl⟩
      let bodyRecips :=
        whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit
      have hBodyRecips : ∀ X, bodyRecips X → X ≠ B :=
        whileRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit
      let bodyHats : Fin k → WordTuple L C F Payload :=
        fun i => Classical.choose (ihBody hWFBody hCtrlBody (bodies i) (hBodies i))
      have hBodyDist : ∀ i : Fin k,
          distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PBody)
            (bodyHats i) := by
        intro i
        exact (Classical.choose_spec (ihBody hWFBody hCtrlBody (bodies i) (hBodies i))).1
      have hBodyErase : ∀ i : Fin k,
          eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) (bodyHats i) = bodies i := by
        intro i
        exact (Classical.choose_spec (ihBody hWFBody hCtrlBody (bodies i) (hBodies i))).2
      rcases ihExit hWFExit hCtrlExit exitMSC hExitMSC with ⟨exitHat, hExitDist, hExitErase⟩
      let bodyStep : Fin k → WordTuple L C F Payload :=
        fun i =>
          mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
            ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload) B bodyRecips true hBodyRecips
            ∘ₘ bodyHats i
      let exitStep : WordTuple L C F Payload :=
        mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
          ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload) B bodyRecips false hBodyRecips
          ∘ₘ exitHat
      refine ⟨WordTuple.concatList (List.ofFn bodyStep ++ [exitStep]), ?_, ?_⟩
      · refine ⟨?_, ?_⟩
        · intro X
          by_cases hXB : X = B
          · subst hXB
            simpa [projectDist, project, localTraceSemantics, bodyStep, exitStep,
              concatList_apply, WordTuple.concat_assoc, controlBroadcastMSC_decider,
              List.map_append] using
              (show
                ∃ k' : Nat,
                ∃ bodies' : Fin k' →
                  LocalWord (C := C) (F := F) (Payload := Payload) X,
                (∀ i, localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      X bodyRecips true)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                  (bodies' i)) ∧
                ∃ exitWord,
                  localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                        X bodyRecips false)
                      ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
                    exitWord ∧
                  WordTuple.concatList (List.ofFn bodyStep ++ [exitStep]) X =
                    concatLocalWords
                      (C := C) (F := F) (Payload := Payload)
                      (List.ofFn (fun i =>
                        AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) X c
                          :: bodies' i))
                    ++
                    (AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) X c
                      :: exitWord) from
                ⟨k,
                  (fun i =>
                    (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                      X bodyRecips true) ++ bodyHats i X),
                  (by
                    intro i
                    exact localTraceSemantics_seq_intro
                      (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                        X bodyRecips true)
                      (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                      _ _
                      (controlBroadcast_trace (L := L) (C := C) (F := F) (Payload := Payload)
                        X bodyRecips true)
                      ((hBodyDist i).1 X)),
                  (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                      X bodyRecips false) ++ exitHat X,
                  localTraceSemantics_seq_intro
                    (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      X bodyRecips false)
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
                    _ _
                    (controlBroadcast_trace (L := L) (C := C) (F := F) (Payload := Payload)
                      X bodyRecips false)
                    (hExitDist.1 X),
                  by
                    have hBodyMap :
                        List.map (fun M => M X) (List.ofFn bodyStep) =
                          List.ofFn (fun i =>
                            AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) X c
                              :: (controlBroadcastWord (L := L) (C := C) (F := F)
                                    (Payload := Payload) X bodyRecips true ++ bodyHats i X)) := by
                      ext i
                      simp [bodyStep, WordTuple.concat, mscWhileTrue, choiceWhileTrue,
                        controlBroadcastMSC_decider, List.append_assoc]
                    have hExitWord :
                        exitStep X =
                          AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) X c
                            :: (controlBroadcastWord (L := L) (C := C) (F := F)
                                  (Payload := Payload) X bodyRecips false ++ exitHat X) := by
                      simp [exitStep, WordTuple.concat, mscWhileFalse, choiceWhileFalse,
                        controlBroadcastMSC_decider, List.append_assoc]
                    rw [concatList_apply, List.map_append, hBodyMap]
                    simp [hExitWord, concatLocalWords_append, concatLocalWords_cons,
                      concatLocalWords]⟩)
          · by_cases hRecip : bodyRecips X
            · have hRecip' :
                  whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit X := hRecip
              have hTraceRecv :
                  ∃ k' : Nat,
                  ∃ bodies' : Fin k' →
                    LocalWord (C := C) (F := F) (Payload := Payload) X,
                  (∀ i, localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                    (bodies' i)) ∧
                  ∃ exitWord,
                    localTraceSemantics
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
                      exitWord ∧
                    WordTuple.concatList (List.ofFn bodyStep ++ [exitStep]) X =
                      concatLocalWords
                        (C := C) (F := F) (Payload := Payload)
                        (List.ofFn (fun i =>
                          AlphabetOf.mkRecv (C := C) (F := F) X
                            (ControlPayload.setDecision true ControlPayload.ctrlPattern) B
                            :: bodies' i))
                      ++
                      (AlphabetOf.mkRecv (C := C) (F := F) X
                        (ControlPayload.setDecision false ControlPayload.ctrlPattern) B
                        :: exitWord) := by
                  exact ⟨k, (fun i => bodyHats i X), (by intro i; exact (hBodyDist i).1 X),
                    exitHat X, (hExitDist.1 X),
                    by
                      have hBodyMap :
                          List.map (fun M => M X) (List.ofFn bodyStep) =
                            List.ofFn (fun i =>
                              AlphabetOf.mkRecv (C := C) (F := F) X
                                (ControlPayload.setDecision true ControlPayload.ctrlPattern) B
                                  :: bodyHats i X) := by
                        ext i
                        have hChoice :
                            mscWhileTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
                          mscChoice_other (C := C) (F := F) (Payload := Payload)
                            B X (choiceWhileTrue (C := C) (F := F) (Payload := Payload) c B) hXB
                        have hCtrl :
                            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                              B bodyRecips true hBodyRecips) X =
                            [AlphabetOf.mkRecv (C := C) (F := F) X
                              (ControlPayload.setDecision true ControlPayload.ctrlPattern) B] :=
                          controlBroadcastMSC_recipient (C := C) (F := F) (Payload := Payload)
                            B X bodyRecips true hBodyRecips hRecip
                        simp [bodyStep, WordTuple.concat, hChoice, hCtrl]
                      have hExitWord :
                          exitStep X =
                            AlphabetOf.mkRecv (C := C) (F := F) X
                              (ControlPayload.setDecision false ControlPayload.ctrlPattern) B
                                :: exitHat X := by
                        have hChoice :
                            mscWhileFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
                          mscChoice_other (C := C) (F := F) (Payload := Payload)
                            B X (choiceWhileFalse (C := C) (F := F) (Payload := Payload) c B) hXB
                        have hCtrl :
                            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                              B bodyRecips false hBodyRecips) X =
                            [AlphabetOf.mkRecv (C := C) (F := F) X
                              (ControlPayload.setDecision false ControlPayload.ctrlPattern) B] :=
                          controlBroadcastMSC_recipient (C := C) (F := F) (Payload := Payload)
                            B X bodyRecips false hBodyRecips hRecip
                        simp [exitStep, WordTuple.concat, hChoice, hCtrl]
                      rw [concatList_apply, List.map_append, hBodyMap]
                      simp [hExitWord, concatLocalWords_append, concatLocalWords_cons,
                        concatLocalWords]⟩
              have hProj :
                  projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                    (.whileLoop c B PBody PExit) X =
                  LocProg.recvWhile ControlPayload.ctrlPattern B
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit) := by
                simp [projectDist, project, hXB, hRecip']
              rw [hProj]
              simpa [localTraceSemantics] using hTraceRecv
            · have hNoPartBody : ¬ participationSet PBody X := by
                intro hPart
                exact hRecip ⟨hXB, Or.inl hPart⟩
              have hNoPartExit : ¬ participationSet PExit X := by
                intro hPart
                exact hRecip ⟨hXB, Or.inr hPart⟩
              have hBodyNil : ∀ i : Fin k, bodyHats i X = [] := by
                intro i
                exact project_trace_nil_of_not_participating
                  (L := L) (C := C) (F := F) (Payload := Payload) X PBody hNoPartBody
                  (bodyHats i X) ((hBodyDist i).1 X)
              have hExitNil : exitHat X = [] :=
                project_trace_nil_of_not_participating
                  (L := L) (C := C) (F := F) (Payload := Payload) X PExit hNoPartExit
                  (exitHat X) (hExitDist.1 X)
              have hXB' : X ≠ B := hXB
              have hBodyStepNil : ∀ i : Fin k, bodyStep i X = [] := by
                intro i
                have hChoice :
                    mscWhileTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
                  mscChoice_other (C := C) (F := F) (Payload := Payload)
                    B X (choiceWhileTrue (C := C) (F := F) (Payload := Payload) c B) hXB'
                have hCtrl :
                    (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                      B bodyRecips true hBodyRecips) X = [] :=
                  controlBroadcastMSC_nonRecipient (C := C) (F := F) (Payload := Payload)
                    B X bodyRecips true hBodyRecips hXB' hRecip
                simp [bodyStep, WordTuple.concat, hChoice, hCtrl, hBodyNil i]
              have hExitStepNil : exitStep X = [] := by
                have hChoice :
                    mscWhileFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
                  mscChoice_other (C := C) (F := F) (Payload := Payload)
                    B X (choiceWhileFalse (C := C) (F := F) (Payload := Payload) c B) hXB'
                have hCtrl :
                    (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                      B bodyRecips false hBodyRecips) X = [] :=
                  controlBroadcastMSC_nonRecipient (C := C) (F := F) (Payload := Payload)
                    B X bodyRecips false hBodyRecips hXB' hRecip
                simp [exitStep, WordTuple.concat, hChoice, hCtrl, hExitNil]
              have hBodyWordMap :
                  List.map (fun M => M X) (List.ofFn bodyStep) =
                    List.replicate k
                      ([] : LocalWord (C := C) (F := F) (Payload := Payload) X) := by
                apply List.ext_getElem?
                intro i
                rw [List.getElem?_map, List.getElem?_ofFn, List.getElem?_replicate]
                by_cases hi : i < k
                · simp [hi, hBodyStepNil ⟨i, hi⟩]
                · simp [hi]
              have hBodyWordNil :
                  concatLocalWords (C := C) (F := F) (Payload := Payload)
                    (List.map (fun M => M X) (List.ofFn bodyStep)) = [] := by
                rw [hBodyWordMap]
                simpa using
                  (concatLocalWords_replicate_nil (C := C) (F := F) (Payload := Payload)
                    (A := X) k)
              have hWord :
                  WordTuple.concatList (List.ofFn bodyStep ++ [exitStep]) X = [] := by
                rw [concatList_apply, List.map_append, concatLocalWords_append]
                rw [hBodyWordNil]
                simp [concatLocalWords, hExitStepNil]
              have hProj :
                  project (L := L) (C := C) (F := F) (Payload := Payload) X
                    (.whileLoop c B PBody PExit) = LocProg.eps := by
                have hRecip' :
                    ¬ whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit X := hRecip
                unfold project
                simp [hXB, hRecip']
              have hProjDist :
                  projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                    (.whileLoop c B PBody PExit) X = LocProg.eps := by
                simp [projectDist, hProj]
              rw [hProjDist, hWord]
              simpa using
                localTraceSemantics_eps_nil
                (L := L) (C := C) (F := F) (Payload := Payload) (A := X)
        · have hBodyStepComplete : ∀ i : Fin k, IsCompleteMSC (bodyStep i) := by
            intro i
            exact concat_complete_complete _ _
              (concat_complete_complete _ _
                (mscWhileTrue_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
                (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
                  B bodyRecips true hBodyRecips))
              ((hBodyDist i).2)
          have hExitStepComplete : IsCompleteMSC exitStep := by
            exact concat_complete_complete _ _
              (concat_complete_complete _ _
                (mscWhileFalse_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
                (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
                  B bodyRecips false hBodyRecips))
              (hExitDist.2)
          exact concatList_complete (L := L) (C := C) (F := F) (Payload := Payload)
            (List.ofFn bodyStep ++ [exitStep]) (by
              intro M hM
              rw [List.mem_append, List.mem_ofFn] at hM
              rcases hM with hM | hM
              · rcases hM with ⟨i, rfl⟩
                exact hBodyStepComplete i
              · simp at hM
                rcases hM with rfl
                exact hExitStepComplete)
      · rw [eraseTuple_concatList, List.map_append]
        simp only [List.map]
        rw [concatList_append_singleton, eraseTuple_concat, eraseTuple_concat,
          eraseTuple_mscWhileFalse, erase_controlBroadcastMSC, hExitErase]
        have hBodyMap :
            List.map (eraseTuple (L := L) (C := C) (F := F) (Payload := Payload))
              (List.ofFn bodyStep) =
            List.ofFn (fun i => mscWhileTrue (C := C) (F := F) (Payload := Payload) c B ∘ₘ bodies i) := by
          apply List.ext_getElem <;> simp
          intro i hi1 hi2
          simp [bodyStep, eraseTuple_concat, eraseTuple_mscWhileTrue,
            erase_controlBroadcastMSC, hBodyErase, WordTuple.concat_assoc]
          ext X
          simp [WordTuple.concat, mscEmpty, WordTuple.empty]
        rw [hBodyMap]
        ext X
        simp [WordTuple.concat, WordTuple.concat_assoc, mscEmpty, WordTuple.empty]

/-- Existence of a complete distributed realization for every well-typed,
    control-distinguishable global program. This combines semantic nonemptiness
    with the constructive realization theorem and is intended as a reusable
    helper for the zipper/control proofs. -/
theorem exists_distSemantics
    (prog : Prog L C F Payload)
    (hWellFormed : WellTypedProgram prog)
    (hCtrl : ControlDistinguishableProgram (L := L) (C := C) (F := F) (Payload := Payload) prog) :
    ∃ Mhat,
      distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (projectDist (L := L) (C := C) (F := F) (Payload := Payload) prog) Mhat := by
  rcases mscSemantics_nonempty (L := L) (C := C) (F := F) (Payload := Payload) prog hWellFormed with
    ⟨M, hM⟩
  rcases realization_complete (L := L) (C := C) (F := F) (Payload := Payload)
      prog hWellFormed hCtrl M hM with ⟨Mhat, hDist, _hErase⟩
  exact ⟨Mhat, hDist⟩

/-- **lem:complete-locals-complete-msc**: If every lifeline's local word satisfies
    the projection semantics of a well-typed program,
    AND the resulting word tuple is already an MSC (has no unmatched receives,
    compatible matched payloads, and acyclic causality), then it is a *complete* MSC
    (sends = receives on every channel).

    The `IsMSC` hypothesis is essential: it prevents branch mismatch in `ite`/`while`
    via `matchedLabelsCompatible`, and supplies `labelCompat`/`acyclic` directly.
    The remaining content is proving `tupleComplete` (sndCount = rcvCount), whose
    missing direction (sndCount ≤ rcvCount) follows from the program structure. -/
theorem complete_locals_isCompleteMSC
    (prog : Prog L C F Payload)
    (hWellFormed : WellTypedProgram prog)
    (w : (A : L) → LocalWord (C := C) (F := F) (Payload := Payload) A)
    (hTrace : ∀ A, localTraceSemantics
      (project (L := L) (C := C) (F := F) (Payload := Payload) A prog) (w A))
    (hMSC : IsMSC (fun A => w A)) :
    IsCompleteMSC (fun A => w A) := by
  rcases uniformZipper (L := L) (C := C) (F := F) (Payload := Payload) prog
      hWellFormed (fun A => w A) (WordTuple.empty (L := L) (C := C) (F := F) (Payload := Payload))
      (fun A => localTrace_implies_localPrefix
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) A prog)
        (w A) (hTrace A))
      (by
        intro A hA
        simpa [WordTuple.empty] using hA)
      (by simpa using hMSC) with ⟨wbar, hZip⟩
  exact zipPost_left_isCompleteMSC
    (L := L) (C := C) (F := F) (Payload := Payload)
    hZip hTrace

/-- **Soundness of realization**: every distributed execution of `projectDist prog`
    erases to an MSC in the global semantics ⟦prog⟧.

    This is the converse of `realization_complete`. Together they show
    `⟦prog⟧ M ↔ ∃ Mhat, distSemantics (projectDist prog) Mhat ∧ eraseTuple Mhat = M`.

    Proof sketch for compound cases:
    - seq: decompose each `Mhat A = w1_A ++ w2_A` using the seq local semantics,
      define Mhat1/Mhat2; the completeness of Mhat together with channel-completeness
      of P1 forces IsCompleteMSC Mhat1 (and Mhat2); apply IH.
    - ite: the control letter at B determines the branch (true/false); extract the
      sub-execution and apply the IH for that branch.
    - while: the control letters at B determine the number of iterations; extract
      body executions and exit execution; apply IHs. -/
theorem realization_sound
    (prog : Prog L C F Payload)
    (hWellFormed : WellTypedProgram prog)
    (hCtrl : ControlDistinguishableProgram (L := L) (C := C) (F := F) (Payload := Payload) prog) :
    ∀ Mhat, distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload) prog) Mhat →
      ⟦prog⟧ (eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) Mhat) := by
  induction prog with
  | eps =>
      intro Mhat ⟨hTrace, _⟩
      simp only [mscSemantics]
      funext A
      have hA := hTrace A
      simp [projectDist, project, localTraceSemantics] at hA
      simp [eraseTuple, eraseWord, mscEmpty, WordTuple.empty, hA]
  | msg A xs B ys h =>
      intro Mhat ⟨hTrace, _⟩
      rcases hCtrl with ⟨hxs, hys⟩
      simp only [mscSemantics]
      -- local trace semantics forces Mhat = mscMsg A xs B ys h
      suffices heq : Mhat = mscMsg (C := C) (F := F) A xs B ys h by
        rw [heq]; exact eraseTuple_mscMsg A xs B ys h hxs hys
      funext X
      have hX := hTrace X
      by_cases hXA : X = A
      · subst X
        have hX' :=
          localTraceSemantics_send_eq
            (L := L) (C := C) (F := F) (Payload := Payload)
            (A := A) xs B h (Mhat A)
            (by simpa [projectDist, project] using hX)
        simpa [mscMsg_sender] using hX'
      · by_cases hXB : X = B
        · subst X
          simp [projectDist, project, hXA, localTraceSemantics] at hX
          simp [hX]
        · simp [projectDist, project, hXA, hXB, localTraceSemantics] at hX
          simp [hXA, hXB, hX]
  | act A ys f xs =>
      intro Mhat ⟨hTrace, _⟩
      simp only [mscSemantics]
      suffices heq : Mhat = mscAct (C := C) (F := F) A ys f xs by
        rw [heq]; exact eraseTuple_mscAct A ys f xs
      funext X
      have hX := hTrace X
      by_cases hXA : X = A
      · subst hXA
        simp [projectDist, project, localTraceSemantics] at hX
        simp [hX]
      · simp [projectDist, project, hXA, localTraceSemantics] at hX
        simp [hXA, hX]
  | seq P1 P2 ih1 ih2 =>
      intro Mhat ⟨hTrace, hComplete⟩
      rcases hWellFormed with ⟨hWF1, hWF2⟩
      rcases hCtrl with ⟨hCtrl1, hCtrl2⟩
      simp only [mscSemantics]
      -- Each local trace decomposes: Mhat A = u1 A ++ u2 A
      have hDecomp : ∀ A, ∃ u1 u2,
          localTraceSemantics (project A P1) u1 ∧
          localTraceSemantics (project A P2) u2 ∧
          Mhat A = u1 ++ u2 := fun A => by
        have hA := hTrace A
        simp only [projectDist, project, localTraceSemantics] at hA
        obtain ⟨u1, u2, hu1, hu2, hMhat⟩ := hA
        exact ⟨u1, u2, hu1, hu2, hMhat⟩
      -- Choose decompositions pointwise
      let Mhat1 : (A : L) → LocalWord (C := C) (F := F) (Payload := Payload) A :=
        fun A => (hDecomp A).choose
      let Mhat2 : (A : L) → LocalWord (C := C) (F := F) (Payload := Payload) A :=
        fun A => (hDecomp A).choose_spec.choose
      have hw1 : ∀ A, localTraceSemantics (project A P1) (Mhat1 A) :=
        fun A => (hDecomp A).choose_spec.choose_spec.1
      have hw2 : ∀ A, localTraceSemantics (project A P2) (Mhat2 A) :=
        fun A => (hDecomp A).choose_spec.choose_spec.2.1
      have hw : ∀ A, Mhat A = Mhat1 A ++ Mhat2 A :=
        fun A => (hDecomp A).choose_spec.choose_spec.2.2
      have hMSC : IsMSC Mhat :=
        isCompleteMSC_implies_isMSC
          (L := L) (C := C) (F := F) (Payload := Payload) Mhat hComplete
      have heqMhat : Mhat = (fun A => Mhat1 A) ∘ₘ (fun A => Mhat2 A) := by
        funext A
        simp [WordTuple.concat, hw A]
      have hZip1 :=
        uniformZipper (L := L) (C := C) (F := F) (Payload := Payload) P1
          hWF1
          (fun A => Mhat1 A)
          (fun A => Mhat2 A)
          (fun A => localTrace_implies_localPrefix
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) A P1)
            (Mhat1 A) (hw1 A))
          (by
            intro A _hA
            exact hw1 A)
          (by simpa [heqMhat] using hMSC)
      rcases hZip1 with ⟨Mhat1bar, hZip1⟩
      have h1Complete : IsCompleteMSC (fun A => Mhat1 A) :=
        zipPost_left_isCompleteMSC
          (L := L) (C := C) (F := F) (Payload := Payload)
          hZip1 hw1
      have hMSC2 : IsMSC (fun A => Mhat2 A) :=
        suffix_msc_of_complete_prefix
          (L := L) (C := C) (F := F) (Payload := Payload)
          (fun A => Mhat1 A) (fun A => Mhat2 A) h1Complete
          (by simpa [heqMhat] using hMSC)
      have h2Complete : IsCompleteMSC (fun A => Mhat2 A) :=
        complete_locals_isCompleteMSC P2 hWF2 (fun A => Mhat2 A) hw2 hMSC2
      -- Apply induction hypotheses
      have hSem1 : ⟦P1⟧ (eraseTuple (fun A => Mhat1 A)) :=
        ih1 hWF1 hCtrl1 (fun A => Mhat1 A)
          ⟨fun A => by simpa [projectDist] using hw1 A, h1Complete⟩
      have hSem2 : ⟦P2⟧ (eraseTuple (fun A => Mhat2 A)) :=
        ih2 hWF2 hCtrl2 (fun A => Mhat2 A)
          ⟨fun A => by simpa [projectDist] using hw2 A, h2Complete⟩
      rw [heqMhat, eraseTuple_concat]
      exact ⟨eraseTuple (fun A => Mhat1 A), hSem1,
             eraseTuple (fun A => Mhat2 A), hSem2, rfl⟩
  | ite c B PTrue PFalse ihTrue ihFalse =>
      intro Mhat hDist
      rcases hWellFormed with ⟨hWFTrue, hWFFalse⟩
      rcases hCtrl with ⟨hCtrlTrue, hCtrlFalse⟩
      simp only [mscSemantics]
      rcases distSemantics_if_decompose
          (L := L) (C := C) (F := F) (Payload := Payload)
          c B PTrue PFalse Mhat hDist with
        ⟨⟨MTrue, hDistTrue, rfl⟩, _hNoFalse⟩ |
        ⟨⟨MFalse, hDistFalse, rfl⟩, _hNoTrue⟩
      · have hSemTrue : ⟦PTrue⟧
            (eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) MTrue) :=
          ihTrue hWFTrue hCtrlTrue MTrue hDistTrue
        refine Or.inl ?_
        refine ⟨eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) MTrue,
          hSemTrue, ?_⟩
        rw [eraseTuple_concat, eraseTuple_concat,
          eraseTuple_mscIfTrue, erase_controlBroadcastMSC]
        ext X
        simp [WordTuple.concat, mscEmpty, WordTuple.empty]
      · have hSemFalse : ⟦PFalse⟧
            (eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) MFalse) :=
          ihFalse hWFFalse hCtrlFalse MFalse hDistFalse
        refine Or.inr ?_
        refine ⟨eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) MFalse,
          hSemFalse, ?_⟩
        rw [eraseTuple_concat, eraseTuple_concat,
          eraseTuple_mscIfFalse, erase_controlBroadcastMSC]
        ext X
        simp [WordTuple.concat, mscEmpty, WordTuple.empty]
  | whileLoop c B PBody PExit ihBody ihExit =>
      intro Mhat hDist
      rcases hWellFormed with ⟨hWFBody, hWFExit⟩
      rcases hCtrl with ⟨hCtrlBody, hCtrlExit⟩
      simp only [mscSemantics]
      rcases distSemantics_while_decompose
          (L := L) (C := C) (F := F) (Payload := Payload)
          c B PBody PExit hWFBody Mhat hDist with
        ⟨k, bodiesHat, hBodiesHat, exitHat, hExitHat, rfl⟩
      have hBodies :
          ∀ i : Fin k,
            ⟦PBody⟧ (eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) (bodiesHat i)) := by
        intro i
        exact ihBody hWFBody hCtrlBody (bodiesHat i) (hBodiesHat i)
      have hExit :
          ⟦PExit⟧ (eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) exitHat) :=
        ihExit hWFExit hCtrlExit exitHat hExitHat
      refine ⟨k, (fun i =>
        eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) (bodiesHat i)),
        hBodies,
        eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) exitHat,
        hExit, ?_⟩
      rw [eraseTuple_concatList, List.map_append]
      simp only [List.map]
      rw [concatList_append_singleton, eraseTuple_concat, eraseTuple_concat,
        eraseTuple_mscWhileFalse, erase_controlBroadcastMSC]
      have hBodyMap :
          List.map
              (eraseTuple (L := L) (C := C) (F := F) (Payload := Payload))
              (List.ofFn (fun i =>
                mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
                  ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                        B
                        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                        true
                        (whileRecipients_no_self (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                  ∘ₘ bodiesHat i)) =
            List.ofFn (fun i =>
              mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
                ∘ₘ eraseTuple (L := L) (C := C) (F := F) (Payload := Payload) (bodiesHat i)) := by
        apply List.ext_getElem <;> simp
        intro i hi1 hi2
        simp [eraseTuple_concat, eraseTuple_mscWhileTrue,
          erase_controlBroadcastMSC, WordTuple.concat_assoc]
        ext X
        simp [WordTuple.concat, mscEmpty, WordTuple.empty]
      rw [hBodyMap]
      ext X
      simp [WordTuple.concat, WordTuple.concat_assoc, mscEmpty, WordTuple.empty]

end Correctness
