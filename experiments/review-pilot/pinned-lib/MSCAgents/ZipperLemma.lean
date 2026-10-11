/-
  MSCAgents/ZipperLemma.lean
  ==========================
  Structural proof infrastructure for the Uniform Zipper lemma from sec:projection.
-/

import MSCAgents.ZipperPost
import MSCAgents.BroadcastMSC

section ZipperLemma

variable {L C F Payload : Type} [DecidableEq L] [Fintype L]
variable [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]

private theorem mkSend_proof_irrel
    (A : L) (xs : Payload) (B : L) (h₁ h₂ : A ≠ B) :
    AlphabetOf.mkSend (C := C) (F := F) A xs B h₁ =
      AlphabetOf.mkSend (C := C) (F := F) A xs B h₂ := by
  simp [AlphabetOf.mkSend]

private theorem drop_concat_prefix_eq
    (U V : WordTuple L C F Payload) (k : L → Nat)
    (hk : ∀ X, k X ≤ (U X).length) :
    (fun X => ((U ∘ₘ V) X).drop (k X)) =
      (fun X => (U X).drop (k X)) ∘ₘ V := by
  funext X
  simp [WordTuple.concat]
  exact List.drop_append_of_le_length (hk X)

private theorem localPrefixSemantics_send_cases {A : L}
    (xs : Payload) (B : L) (hAB : A ≠ B)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.send xs B) w) :
    w = [] ∨ w = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] := by
  rcases h with ⟨w', hw', hpref⟩
  rcases hw' with ⟨hAB', hw'⟩
  subst hw'
  rcases hpref with ⟨t, ht⟩
  cases w with
  | nil =>
      left
      rfl
  | cons hd tl =>
      cases tl with
      | nil =>
          cases t with
          | nil =>
              right
              calc
                [hd] = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB'] := by
                  simpa using ht.symm
                _ = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] := by
                  simp [mkSend_proof_irrel (C := C) (F := F) A xs B hAB' hAB]
          | cons th tt =>
              simp at ht
      | cons hd' tl' =>
          simp at ht

private theorem localPrefixSemantics_recv_cases {A : L}
    (ys : Payload) (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.recv ys B) w) :
    w = [] ∨ w = [AlphabetOf.mkRecv (C := C) (F := F) A ys B] := by
  rcases h with ⟨w', hw', hpref⟩
  simp [localTraceSemantics] at hw'
  subst hw'
  rcases hpref with ⟨t, ht⟩
  cases w with
  | nil =>
      left
      rfl
  | cons hd tl =>
      cases tl with
      | nil =>
          cases t with
          | nil =>
              right
              simpa using ht.symm
          | cons th tt =>
              simp at ht
      | cons hd' tl' =>
          simp at ht

private theorem localPrefixSemantics_act_cases {A : L}
    (ys : Payload) (f : F) (xs : Payload)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.act ys f xs) w) :
    w = [] ∨ w = [AlphabetOf.mkAct (C := C) A ys f xs] := by
  rcases h with ⟨w', hw', hpref⟩
  simp [localTraceSemantics] at hw'
  subst hw'
  rcases hpref with ⟨t, ht⟩
  cases w with
  | nil =>
      left
      rfl
  | cons hd tl =>
      cases tl with
      | nil =>
          cases t with
          | nil =>
              right
              simpa using ht.symm
          | cons th tt =>
              simp at ht
      | cons hd' tl' =>
          simp at ht

private theorem localPrefixSemantics_eps_eq_nil {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.eps) w) :
    w = [] := by
  rcases h with ⟨w', hw', hpref⟩
  simp [localTraceSemantics] at hw'
  subst hw'
  rcases hpref with ⟨t, ht⟩
  cases w with
  | nil => rfl
  | cons hd tl =>
      simp at ht

private def msgSendPrefix
    (A : L) (xs : Payload) (B : L) (hAB : A ≠ B) :
    WordTuple L C F Payload :=
  fun X =>
    if hX : X = A then
      hX ▸ [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB]
    else
      []

@[simp]
private theorem msgSendPrefix_owner
    (A : L) (xs : Payload) (B : L) (hAB : A ≠ B) :
    msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB A =
      [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] := by
  simp [msgSendPrefix]

@[simp]
private theorem msgSendPrefix_other
    (A X : L) (xs : Payload) (B : L) (hAB : A ≠ B)
    (hXA : X ≠ A) :
    msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB X = [] := by
  simp [msgSendPrefix, hXA]

private theorem sndCount_msgSendPrefix
    (A : L) (xs : Payload) (B : L) (hAB : A ≠ B) (S T : L) :
    sndCount (msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB) S T =
      if S = A ∧ T = B then 1 else 0 := by
  by_cases hSA : S = A
  · subst S
    by_cases hTB : T = B
    · subst T
      simp [sndCount, msgSendPrefix_owner, countSends, AlphabetOf.mkSend, Letter.isSendTo]
    · have hBT : B ≠ T := by simpa [eq_comm] using hTB
      simp [sndCount, msgSendPrefix_owner, countSends, AlphabetOf.mkSend, Letter.isSendTo, hTB, hBT]
  · simp [sndCount, msgSendPrefix_other, hSA, countSends]

private theorem rcvCount_msgSendPrefix
    (A : L) (xs : Payload) (B : L) (hAB : A ≠ B) (S T : L) :
    rcvCount (msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB) S T = 0 := by
  by_cases hTA : T = A
  · subst T
    simp [rcvCount, msgSendPrefix_owner, countRecvs, AlphabetOf.mkSend, Letter.isRecvFrom]
  · simp [rcvCount, msgSendPrefix_other, hTA, countRecvs]

private theorem sendPayloads_msgSendPrefix
    (A : L) (xs : Payload) (B : L) (hAB : A ≠ B) (S T : L) :
    sendPayloads (C := C) (F := F) (Payload := Payload) T
      (msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB S) =
      if S = A ∧ T = B then [xs] else [] := by
  by_cases hSA : S = A
  · subst S
    by_cases hTB : T = B
    · subst T
      simp [msgSendPrefix_owner, sendPayloads, AlphabetOf.mkSend]
    · have hBT : B ≠ T := by simpa [eq_comm] using hTB
      simp [msgSendPrefix_owner, sendPayloads, AlphabetOf.mkSend, hTB, hBT]
  · simp [msgSendPrefix_other, hSA, sendPayloads]

private theorem recvPayloads_msgSendPrefix
    (A : L) (xs : Payload) (B : L) (hAB : A ≠ B) (S T : L) :
    recvPayloads (C := C) (F := F) (Payload := Payload) S
      (msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB T) = [] := by
  by_cases hTA : T = A
  · subst T
    simp [msgSendPrefix_owner, recvPayloads, AlphabetOf.mkSend]
  · simp [msgSendPrefix_other, hTA, recvPayloads]

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
      simp [mkSend_proof_irrel (C := C) (F := F) A xs B hAB' hAB]

private theorem localTraceSemantics_recv_eq
    {A : L} (ys : Payload) (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.recv ys B) w) :
    w = [AlphabetOf.mkRecv (C := C) (F := F) A ys B] := by
  simpa [localTraceSemantics] using h

private theorem exists_project_trace
    (A : L) :
    ∀ P : Prog L C F Payload,
      ∃ w,
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) A P) w
  | .eps =>
      ⟨[], by simp [project, localTraceSemantics]⟩
  | .msg X xs Y ys hXY => by
      by_cases hAX : A = X
      · subst hAX
        refine ⟨[AlphabetOf.mkSend (C := C) (F := F) A xs Y hXY], ?_⟩
        simpa [project, localTraceSemantics] using (Exists.intro hXY rfl)
      · by_cases hAY : A = Y
        · subst hAY
          refine ⟨[AlphabetOf.mkRecv (C := C) (F := F) A ys X], ?_⟩
          simp [project, hAX, localTraceSemantics]
        · exact ⟨[], by simp [project, hAX, hAY, localTraceSemantics]⟩
  | .act X ys f xs => by
      by_cases hAX : A = X
      · subst hAX
        refine ⟨[AlphabetOf.mkAct (C := C) A ys f xs], ?_⟩
        simp [project, localTraceSemantics]
      · exact ⟨[], by simp [project, hAX, localTraceSemantics]⟩
  | .seq P1 P2 => by
      let ⟨w1, hw1⟩ := exists_project_trace A P1
      let ⟨w2, hw2⟩ := exists_project_trace A P2
      exact ⟨w1 ++ w2,
        localTraceSemantics_seq_intro
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) A P1)
          (project (L := L) (C := C) (F := F) (Payload := Payload) A P2)
          w1 w2 hw1 hw2⟩
  | .ite c B PTrue PFalse => by
      by_cases hAB : A = B
      · subst hAB
        let ⟨wFalse, hwFalse⟩ := exists_project_trace A PFalse
        refine ⟨AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c ::
            controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              A
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
              false ++ wFalse, ?_⟩
        have hSeq :
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  A
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                  false)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse)
              (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  A
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                  false ++ wFalse) := by
          exact ⟨controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              A
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
              false,
            wFalse,
            controlBroadcast_trace
              (L := L) (C := C) (F := F) (Payload := Payload)
              A
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
              false,
            hwFalse,
            rfl⟩
        have hGoal :
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (LocProg.localIf c
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    A
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                    true)
                  ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue)
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    A
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                    false)
                  ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse))
              (AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c ::
                controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  A
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                  false ++ wFalse) := by
          change
            (∃ u,
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    A
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                    true)
                  ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue) u ∧
              AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c ::
                  controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                    A
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                    false ++ wFalse =
                AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c :: u) ∨
            (∃ u,
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    A
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                    false)
                  ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse) u ∧
              AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c ::
                  controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                    A
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                    false ++ wFalse =
                AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c :: u)
          exact Or.inr ⟨
            controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              A
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
              false ++ wFalse,
            hSeq,
            rfl⟩
        simpa [project] using hGoal
      · by_cases hRecip :
          ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse A
        · let ⟨wFalse, hwFalse⟩ := exists_project_trace A PFalse
          refine ⟨AlphabetOf.mkRecv (C := C) (F := F) A
              (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: wFalse, ?_⟩
          have hGoal :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (LocProg.recvIf ControlPayload.ctrlPattern B
                  (project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse))
                (AlphabetOf.mkRecv (C := C) (F := F) A
                  (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: wFalse) := by
            change
              (∃ u,
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue) u ∧
                AlphabetOf.mkRecv (C := C) (F := F) A
                    (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: wFalse =
                  AlphabetOf.mkRecv (C := C) (F := F) A
                    (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: u) ∨
              (∃ u,
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse) u ∧
                AlphabetOf.mkRecv (C := C) (F := F) A
                    (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: wFalse =
                  AlphabetOf.mkRecv (C := C) (F := F) A
                    (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: u)
            exact Or.inr ⟨wFalse, hwFalse, rfl⟩
          simpa [project, hAB, hRecip] using hGoal
        · exact ⟨[], by simp [project, hAB, hRecip, localTraceSemantics]⟩
  | .whileLoop c B PBody PExit => by
      by_cases hAB : A = B
      · subst hAB
        let ⟨wExit, hwExit⟩ := exists_project_trace A PExit
        refine ⟨AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c ::
            controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
              A
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)
              false ++ wExit, ?_⟩
        simpa [project] using
          ((localTraceSemantics_localWhile_unfold
            (L := L) (C := C) (F := F) (Payload := Payload)
            (A := A)
            c
            ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                A
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)
                true)
              ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PBody)
            ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                A
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)
                false)
              ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PExit)
            _).2 <|
            Or.inl ⟨controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                A
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)
                false ++ wExit,
              ⟨controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  A
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)
                  false,
                wExit,
                controlBroadcast_trace
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  A
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)
                  false,
                hwExit,
                rfl⟩,
              rfl⟩)
      · by_cases hRecip :
          whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit A
        · let ⟨wExit, hwExit⟩ := exists_project_trace A PExit
          refine ⟨AlphabetOf.mkRecv (C := C) (F := F) A
              (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: wExit, ?_⟩
          simpa [project, hAB, hRecip] using
            ((localTraceSemantics_recvWhile_unfold
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := A)
              ControlPayload.ctrlPattern
              B
              (project (L := L) (C := C) (F := F) (Payload := Payload) A PBody)
              (project (L := L) (C := C) (F := F) (Payload := Payload) A PExit)
              _).2 <|
              Or.inl ⟨wExit, hwExit, rfl⟩)
        · exact ⟨[], by simp [project, hAB, hRecip, localTraceSemantics]⟩

private theorem localPrefixSemantics_project_nil
    (A : L) (P : Prog L C F Payload) :
    localPrefixSemantics
      (L := L) (C := C) (F := F) (Payload := Payload)
      (project (L := L) (C := C) (F := F) (Payload := Payload) A P) [] := by
  rcases exists_project_trace (L := L) (C := C) (F := F) (Payload := Payload) A P with ⟨w, hw⟩
  exact ⟨w, hw, ⟨w, by simp [IsPrefixWord]⟩⟩

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

private def IsPrefixList {α : Type} (xs ys : List α) : Prop :=
  ∃ t, ys = xs ++ t

private theorem sendPayloads_prefix {A : L} (B : L)
    {u u' : LocalWord (C := C) (F := F) (Payload := Payload) A}
    (h : IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) u u') :
    IsPrefixList
      (sendPayloads (C := C) (F := F) (Payload := Payload) B u)
      (sendPayloads (C := C) (F := F) (Payload := Payload) B u') := by
  rcases h with ⟨t, rfl⟩
  refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) B t, ?_⟩
  simp [IsPrefixList]

private theorem prefix_of_singleton_nonempty {α : Type} {xs : List α} {a : α}
    (hPref : IsPrefixList xs [a])
    (hNe : xs ≠ []) :
    xs = [a] := by
  rcases hPref with ⟨t, ht⟩
  cases xs with
  | nil =>
      contradiction
  | cons x xs =>
      cases xs with
      | nil =>
          cases t with
          | nil =>
              simp at ht
              simp [ht]
          | cons y ys =>
              simp at ht
      | cons y ys =>
          cases t <;> simp at ht

private theorem prefix_of_nil_eq_nil {α : Type} {xs : List α}
    (hPref : IsPrefixList xs []) :
    xs = [] := by
  rcases hPref with ⟨t, ht⟩
  cases xs with
  | nil =>
      rfl
  | cons x xs =>
      simp at ht

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
        localTraceSemantics_sendList_eq payload Xs hTail u2 hu2
      rw [hu1', hu2']
      simp [sendWordForTargets]

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
    (A := A)
    (payload := ControlPayload.setDecision decision ControlPayload.ctrlPattern)
    (targets := controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips)
    hTargets
    (List.Nodup.sublist List.filter_sublist
      (controlRecipients_nodup (C := C) (F := F) (Payload := Payload) recips))
    X]
  simp [hXTarget]

private theorem sendPayloads_controlBroadcastWord_nonRecipient
    (A X : L) (recips : L → Prop) (decision : Bool)
    (hX : ¬ recips X) :
    sendPayloads (C := C) (F := F) (Payload := Payload) X
      (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload) A recips decision) =
      [] := by
  classical
  unfold controlBroadcastWord
  have hTargets :
      ∀ Y,
        Y ∈ controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips →
          A ≠ Y := by
    intro Y hY
    exact controlSendTargets_no_self (L := L) (C := C) (F := F) (Payload := Payload) hY
  have hXTarget :
      X ∉ controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips := by
    intro hMem
    exact hX ((mem_controlRecipients (C := C) (F := F) (Payload := Payload)).mp
      (List.mem_filter.mp hMem).1)
  rw [sendPayloads_sendWordForTargets_eq
    (A := A)
    (payload := ControlPayload.setDecision decision ControlPayload.ctrlPattern)
    (targets := controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips)
    hTargets
    (List.Nodup.sublist List.filter_sublist
      (controlRecipients_nodup (C := C) (F := F) (Payload := Payload) recips))
    X]
  simp [hXTarget]

private theorem localTraceSemantics_controlBroadcast_eq
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
      (A := A)
      (ControlPayload.setDecision decision ControlPayload.ctrlPattern)
      (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips)
      (by
        intro X hX
        exact controlSendTargets_no_self (L := L) (C := C) (F := F) (Payload := Payload) hX)
      w
      (by simpa [controlBroadcast] using h)
  simpa [controlBroadcastWord] using h'

private theorem localPrefixSemantics_controlBroadcast_prefix
    (A : L) (recips : L → Prop) (decision : Bool)
    (u : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload) (A := A)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) A recips decision)
        u) :
    IsPrefixWord
      (L := L) (C := C) (F := F) (Payload := Payload)
      u
      (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload) A recips decision) := by
  rcases h with ⟨w, hw, hpref⟩
  rw [localTraceSemantics_controlBroadcast_eq
      (L := L) (C := C) (F := F) (Payload := Payload) A recips decision w hw] at hpref
  exact hpref

private theorem controlBroadcast_prefix_send_count_zero_nonRecipient
    (A X : L) (recips : L → Prop) (decision : Bool)
    (sA : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) A recips decision)
        sA)
    (hX : ¬ recips X) :
    countSends X sA = 0 := by
  have hSendPref :
      IsPrefixWord
        (L := L) (C := C) (F := F) (Payload := Payload)
        sA
        (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload) A recips decision) :=
    localPrefixSemantics_controlBroadcast_prefix
      (L := L) (C := C) (F := F) (Payload := Payload) A recips decision sA hSPref
  have hPayloadPref :
      IsPrefixList
        (sendPayloads (C := C) (F := F) (Payload := Payload) X sA)
        [] := by
    simpa [sendPayloads_controlBroadcastWord_nonRecipient
      (L := L) (C := C) (F := F) (Payload := Payload) A X recips decision hX] using
      (sendPayloads_prefix
        (L := L) (C := C) (F := F) (Payload := Payload) X hSendPref)
  have hPayloadNil :
      sendPayloads (C := C) (F := F) (Payload := Payload) X sA = [] :=
    prefix_of_nil_eq_nil hPayloadPref
  rw [← sendPayloads_length (C := C) (F := F) (Payload := Payload) X sA]
  simp [hPayloadNil]

private theorem controlBroadcast_prefix_send_count_le_one_recipient
    (A X : L) (recips : L → Prop) (decision : Bool)
    (sA : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) A recips decision)
        sA)
    (hRecips : ∀ Y, recips Y → Y ≠ A)
    (hX : recips X) :
    countSends X sA ≤ 1 := by
  have hSendPref :
      IsPrefixWord
        (L := L) (C := C) (F := F) (Payload := Payload)
        sA
        (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload) A recips decision) :=
    localPrefixSemantics_controlBroadcast_prefix
      (L := L) (C := C) (F := F) (Payload := Payload) A recips decision sA hSPref
  have hPayloadPref :
      IsPrefixList
        (sendPayloads (C := C) (F := F) (Payload := Payload) X sA)
        [ControlPayload.setDecision decision ControlPayload.ctrlPattern] := by
    simpa [sendPayloads_controlBroadcastWord_recipient
      (L := L) (C := C) (F := F) (Payload := Payload) A X recips decision hRecips hX] using
      (sendPayloads_prefix
        (L := L) (C := C) (F := F) (Payload := Payload) X hSendPref)
  have hLen :
      (sendPayloads (C := C) (F := F) (Payload := Payload) X sA).length ≤ 1 := by
    rcases hPayloadPref with ⟨t, ht⟩
    cases hs : sendPayloads (C := C) (F := F) (Payload := Payload) X sA with
    | nil =>
        simp [hs]
    | cons hd tl =>
        cases tl with
        | nil =>
            simp [hs]
        | cons hd' tl' =>
            simp [hs] at ht
  rw [← sendPayloads_length (C := C) (F := F) (Payload := Payload) X sA]
  exact hLen

/-- Any prefix of a controlBroadcast word has zero receives from any lifeline,
    because all letters in controlBroadcastWord are sends (mkSend). -/
private theorem controlBroadcast_prefix_recv_count_zero
    (A X : L) (recips : L → Prop) (decision : Bool)
    (sA : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) A recips decision)
        sA) :
    countRecvs X sA = 0 := by
  obtain ⟨t, ht⟩ := localPrefixSemantics_controlBroadcast_prefix
    (L := L) (C := C) (F := F) (Payload := Payload) A recips decision sA hSPref
  have hFull : countRecvs X
      (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
        A recips decision) = 0 := by
    classical
    unfold controlBroadcastWord
    rw [sendWordForTargets_eq_attach_map]
    induction (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips).attach with
    | nil => simp [countRecvs]
    | cons R rs ih =>
        simpa [countRecvs, AlphabetOf.mkSend, Letter.isRecvFrom] using ih
  rw [ht, countRecvs_append] at hFull
  omega

private theorem if_recipient_nil_tail_nil
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (hSide : ∀ Y, V Y ≠ [] →
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) Y (.ite c B PTrue PFalse))
        (U Y))
    (hXNil : U X = []) :
    V X = [] := by
  by_cases hVX : V X = []
  · exact hVX
  · have hPartX : participationSet (.ite c B PTrue PFalse) X := by
      simp [participationSet_ite, hXB, hRecip.2]
    have hXTrace := hSide X hVX
    have hXNonempty :=
      project_trace_ne_nil_of_if_participating
        (L := L) (C := C) (F := F) (Payload := Payload)
        X c B PTrue PFalse (U X) hPartX hXTrace
    rw [hXNil] at hXNonempty
    simp at hXNonempty

private theorem if_true_decider_sendPayloads_head
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (hUB :
      U B =
        AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          true)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true)
          sB)
    (hXVisible :
      ∃ p uX,
        U X =
          AlphabetOf.mkRecv (C := C) (F := F) X p B :: uX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    ∃ ss,
      sendPayloads (C := C) (F := F) (Payload := Payload) X ((U ∘ₘ V) B) =
        ControlPayload.setDecision true ControlPayload.ctrlPattern :: ss := by
  have hSendPref :
      IsPrefixWord
        (L := L) (C := C) (F := F) (Payload := Payload)
        sB
        (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          true) :=
    localPrefixSemantics_controlBroadcast_prefix
      (L := L) (C := C) (F := F) (Payload := Payload)
      B
      (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
      true
      sB
      hSPref
  by_cases hCont : rB ++ V B = []
  · have hVB : V B = [] := (List.eq_nil_of_append_eq_nil hCont).2
    have hrB : rB = [] := (List.eq_nil_of_append_eq_nil hCont).1
    rcases hXVisible with ⟨p, uX, hXVisible⟩
    have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
      rw [rcvCount, WordTuple.concat, hXVisible]
      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
    have hSendPos : 1 ≤ sndCount (U ∘ₘ V) B X := by
      have hNoUnmatched := hMSC.noUnmatchedRecv B X
      omega
    have hSendPos' : 1 ≤ countSends X sB := by
      simpa [sndCount, WordTuple.concat, hUB, hVB, hrB, countSends,
        AlphabetOf.mkIfTrue, Letter.isSendTo] using hSendPos
    have hNonempty : sendPayloads (C := C) (F := F) (Payload := Payload) X sB ≠ [] := by
      intro hNil
      have hLen :
          (sendPayloads (C := C) (F := F) (Payload := Payload) X sB).length = 0 := by
        simp [hNil]
      rw [sendPayloads_length] at hLen
      omega
    have hPrefPayloads :
        IsPrefixList
          (sendPayloads (C := C) (F := F) (Payload := Payload) X sB)
          [ControlPayload.setDecision true ControlPayload.ctrlPattern] := by
      simpa [sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        true (by intro Y hY; exact hY.1) hRecip] using
          (sendPayloads_prefix
            (L := L) (C := C) (F := F) (Payload := Payload) X hSendPref)
    have hPayloads :
        sendPayloads (C := C) (F := F) (Payload := Payload) X sB =
          [ControlPayload.setDecision true ControlPayload.ctrlPattern] :=
      prefix_of_singleton_nonempty hPrefPayloads hNonempty
    refine ⟨[], ?_⟩
    simp [WordTuple.concat, hUB, hPayloads, hrB, hVB, sendPayloads,
      AlphabetOf.mkIfTrue]
  · have hSWord :
      sB =
        controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          true := by
      exact localTraceSemantics_controlBroadcast_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        true
        sB
        (hSFull hCont)
    refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) X rB ++
      sendPayloads (C := C) (F := F) (Payload := Payload) X (V B), ?_⟩
    simp [WordTuple.concat, hUB, hSWord, sendPayloads_append,
      sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        true (by intro Y hY; exact hY.1) hRecip,
      List.append_assoc, sendPayloads, AlphabetOf.mkIfTrue]

private theorem if_true_prefix_send_count_pos
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hUB :
      U B =
        AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          true)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true)
          sB)
    (hXVisible :
      U X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: uX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    1 ≤ countSends X sB := by
  by_cases hCont : rB ++ V B = []
  · have hVB : V B = [] := (List.eq_nil_of_append_eq_nil hCont).2
    have hrB : rB = [] := (List.eq_nil_of_append_eq_nil hCont).1
    have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
      rw [rcvCount, WordTuple.concat, hXVisible]
      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
    have hSendPos : 1 ≤ sndCount (U ∘ₘ V) B X := by
      have hNoUnmatched := hMSC.noUnmatchedRecv B X
      omega
    simpa [sndCount, WordTuple.concat, hUB, hVB, hrB, countSends,
      AlphabetOf.mkIfTrue, Letter.isSendTo] using hSendPos
  · have hSWord :
      sB =
        controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          true := by
      exact localTraceSemantics_controlBroadcast_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        true
        sB
        (hSFull hCont)
    rw [hSWord]
    rw [← sendPayloads_length]
    simpa [sendPayloads_controlBroadcastWord_recipient
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X
      (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
      true (by intro Y hY; exact hY.1) hRecip]

private theorem if_false_decider_sendPayloads_head
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (hUB :
      U B =
        AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          false)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false)
          sB)
    (hXVisible :
      ∃ p uX,
        U X =
          AlphabetOf.mkRecv (C := C) (F := F) X p B :: uX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    ∃ ss,
      sendPayloads (C := C) (F := F) (Payload := Payload) X ((U ∘ₘ V) B) =
        ControlPayload.setDecision false ControlPayload.ctrlPattern :: ss := by
  have hSendPref :
      IsPrefixWord
        (L := L) (C := C) (F := F) (Payload := Payload)
        sB
        (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          false) :=
    localPrefixSemantics_controlBroadcast_prefix
      (L := L) (C := C) (F := F) (Payload := Payload)
      B
      (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
      false
      sB
      hSPref
  by_cases hCont : rB ++ V B = []
  · have hVB : V B = [] := (List.eq_nil_of_append_eq_nil hCont).2
    have hrB : rB = [] := (List.eq_nil_of_append_eq_nil hCont).1
    rcases hXVisible with ⟨p, uX, hXVisible⟩
    have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
      rw [rcvCount, WordTuple.concat, hXVisible]
      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
    have hSendPos : 1 ≤ sndCount (U ∘ₘ V) B X := by
      have hNoUnmatched := hMSC.noUnmatchedRecv B X
      omega
    have hSendPos' : 1 ≤ countSends X sB := by
      simpa [sndCount, WordTuple.concat, hUB, hVB, hrB, countSends,
        AlphabetOf.mkIfFalse, Letter.isSendTo] using hSendPos
    have hNonempty : sendPayloads (C := C) (F := F) (Payload := Payload) X sB ≠ [] := by
      intro hNil
      have hLen :
          (sendPayloads (C := C) (F := F) (Payload := Payload) X sB).length = 0 := by
        simp [hNil]
      rw [sendPayloads_length] at hLen
      omega
    have hPrefPayloads :
        IsPrefixList
          (sendPayloads (C := C) (F := F) (Payload := Payload) X sB)
          [ControlPayload.setDecision false ControlPayload.ctrlPattern] := by
      simpa [sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        false (by intro Y hY; exact hY.1) hRecip] using
          (sendPayloads_prefix
            (L := L) (C := C) (F := F) (Payload := Payload) X hSendPref)
    have hPayloads :
        sendPayloads (C := C) (F := F) (Payload := Payload) X sB =
          [ControlPayload.setDecision false ControlPayload.ctrlPattern] :=
      prefix_of_singleton_nonempty hPrefPayloads hNonempty
    refine ⟨[], ?_⟩
    simp [WordTuple.concat, hUB, hPayloads, hrB, hVB, sendPayloads,
      AlphabetOf.mkIfFalse]
  · have hSWord :
      sB =
        controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          false := by
      exact localTraceSemantics_controlBroadcast_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        false
        sB
        (hSFull hCont)
    refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) X rB ++
      sendPayloads (C := C) (F := F) (Payload := Payload) X (V B), ?_⟩
    simp [WordTuple.concat, hUB, hSWord, sendPayloads_append,
      sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        false (by intro Y hY; exact hY.1) hRecip,
      List.append_assoc, sendPayloads, AlphabetOf.mkIfFalse]

private theorem if_false_prefix_send_count_pos
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hUB :
      U B =
        AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          false)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false)
          sB)
    (hXVisible :
      U X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: uX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    1 ≤ countSends X sB := by
  by_cases hCont : rB ++ V B = []
  · have hVB : V B = [] := (List.eq_nil_of_append_eq_nil hCont).2
    have hrB : rB = [] := (List.eq_nil_of_append_eq_nil hCont).1
    have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
      rw [rcvCount, WordTuple.concat, hXVisible]
      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
    have hSendPos : 1 ≤ sndCount (U ∘ₘ V) B X := by
      have hNoUnmatched := hMSC.noUnmatchedRecv B X
      omega
    simpa [sndCount, WordTuple.concat, hUB, hVB, hrB, countSends,
      AlphabetOf.mkIfFalse, Letter.isSendTo] using hSendPos
  · have hSWord :
      sB =
        controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          false := by
      exact localTraceSemantics_controlBroadcast_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        false
        sB
        (hSFull hCont)
    rw [hSWord]
    rw [← sendPayloads_length]
    simpa [sendPayloads_controlBroadcastWord_recipient
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X
      (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
      false (by intro Y hY; exact hY.1) hRecip]

private theorem if_recipient_recvPayloads_head
    (B X : L) (decision : Bool)
    (U V : WordTuple L C F Payload)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hXVisible :
      U X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision decision ControlPayload.ctrlPattern) B :: uX) :
    ∃ rs,
      recvPayloads (C := C) (F := F) (Payload := Payload) B ((U ∘ₘ V) X) =
        ControlPayload.setDecision decision ControlPayload.ctrlPattern :: rs := by
  refine ⟨recvPayloads (C := C) (F := F) (Payload := Payload) B (uX ++ V X), ?_⟩
  simp [WordTuple.concat, hXVisible, recvPayloads, recvPayloads_append,
    AlphabetOf.mkRecv, Letter.isRecvFrom]

private theorem if_true_recipient_false_impossible
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (wX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hUB :
      U B =
        AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          true)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true)
          sB)
    (hXFalse :
      U X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: wX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    False := by
  have hs :=
    if_true_decider_sendPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      c B X PTrue PFalse U V hXB hRecip sB rB hUB hSPref hSFull
      ⟨_, _, hXFalse⟩
      hMSC
  have hr :=
    if_recipient_recvPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X false U V wX hXFalse
  have hEq :=
    firstControlDecisions_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      (M := U ∘ₘ V) hMSC B X true false hs hr
  cases hEq

private theorem if_false_recipient_true_impossible
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (wX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hUB :
      U B =
        AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          false)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false)
          sB)
    (hXTrue :
      U X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: wX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    False := by
  have hs :=
    if_false_decider_sendPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      c B X PTrue PFalse U V hXB hRecip sB rB hUB hSPref hSFull
      ⟨_, _, hXTrue⟩
      hMSC
  have hr :=
    if_recipient_recvPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X true U V wX hXTrue
  have hEq :=
    firstControlDecisions_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      (M := U ∘ₘ V) hMSC B X false true hs hr
  cases hEq

private theorem if_true_recipient_prefix_cases
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (hUB :
      U B =
        AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          true)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true)
          sB)
    (hPrefX :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
        (U X))
    (hMSC : IsMSC (U ∘ₘ V)) :
    U X = [] ∨
      ∃ u,
        localPrefixSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue) u ∧
        U X =
          AlphabetOf.mkRecv (C := C) (F := F) X
            (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: u := by
  have hRecvPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (LocProg.recvIf ControlPayload.ctrlPattern B
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse))
        (U X) := by
    simpa [project, hXB, hRecip] using hPrefX
  rcases localPrefixSemantics_recvIf_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := X)
      ControlPayload.ctrlPattern B
      (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
      (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
      (U X) hRecvPref with hCases
  rcases hCases with hNil | hTrue | hFalse
  · exact Or.inl hNil
  · exact Or.inr hTrue
  · rcases hFalse with ⟨wX, _hPrefFalse, hXFalse⟩
    exact False.elim <|
      if_true_recipient_false_impossible
        (L := L) (C := C) (F := F) (Payload := Payload)
        c B X PTrue PFalse U V hXB hRecip sB rB wX wX hUB hSPref hSFull hXFalse hMSC

open Classical in
/-- In the if-true prefix, the kf-prefix of any lifeline B' has zero receives from
    a non-decider A (A ≠ B). Used in the hPrefixLe / hSendSurplusDead conditions. -/
private theorem if_true_kf_recv_count_zero
    (c : C) (B A B' : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (hAB : A ≠ B)
    (hUBeq : U B =
      AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hsBPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse) true)
        sB)
    (hsBSide :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse) true)
          sB)
    (hPref :
      ∀ X, localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (Prog.ite c B PTrue PFalse))
        (U X))
    (hSide : ∀ X, V X ≠ [] →
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
        (U X))
    (hMSC : IsMSC (U ∘ₘ V)) :
    countRecvs A
      (List.take
        (if B' = B then 1 + sB.length
         else if ifRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                  B PTrue PFalse B' ∧ U B' ≠ [] then 1 else 0)
        (U B' ++ V B')) = 0 := by
  by_cases hB'B : B' = B
  · subst B'
    have hTake :
        List.take (1 + sB.length) (U B ++ V B) =
          AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB := by
      rw [hUBeq]
      rw [List.append_assoc]
      have hLen :
          1 + sB.length ≤
            (AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB).length := by
        simpa [Nat.add_comm] using Nat.le_refl (sB.length + 1)
      have hTake' :
          List.take (1 + sB.length)
            ((AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB) ++ (rB ++ V B)) =
              List.take (1 + sB.length)
                (AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB) := by
        simpa using
          (List.take_append_of_le_length
            (l₁ := AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB)
            (l₂ := rB ++ V B)
            (i := 1 + sB.length)
            hLen)
      have hTake'' :
          List.take (1 + sB.length)
            (AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB) =
              AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB := by
        simpa [Nat.add_comm] using
          (List.take_length
            (l := AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB))
      exact hTake'.trans hTake''
    simp [hTake]
    have hsZero :
        countRecvs A sB = 0 :=
      controlBroadcast_prefix_recv_count_zero
        (L := L) (C := C) (F := F) (Payload := Payload)
        B A
        (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
        true
        sB
        hsBPref
    simpa [countRecvs, AlphabetOf.mkIfTrue, Letter.isRecvFrom, hsZero]
  · by_cases hRecip :
        ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse B' ∧
          U B' ≠ []
    · rcases if_true_recipient_prefix_cases
        (L := L) (C := C) (F := F) (Payload := Payload)
        c B B' PTrue PFalse U V hB'B hRecip.1 sB rB hUBeq hsBPref hsBSide
        (hPref B') hMSC with hNil | ⟨u, _hu, hWord⟩
      · exact False.elim (hRecip.2 hNil)
      · have hTake :
          List.take 1 (U B' ++ V B') =
            [AlphabetOf.mkRecv (C := C) (F := F) B'
              (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B] := by
          rw [hWord]
          simp
        have hBA : B ≠ A := by
          simpa using hAB.symm
        simp [hB'B, hRecip, hTake, countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom, hBA]
    · simp [hB'B, hRecip]

private theorem if_false_recipient_prefix_cases
    (c : C) (B X : L) (PTrue PFalse : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (hUB :
      U B =
        AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
          false)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false)
          sB)
    (hPrefX :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
        (U X))
    (hMSC : IsMSC (U ∘ₘ V)) :
    U X = [] ∨
      ∃ u,
        localPrefixSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse) u ∧
        U X =
          AlphabetOf.mkRecv (C := C) (F := F) X
            (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: u := by
  have hRecvPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (LocProg.recvIf ControlPayload.ctrlPattern B
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse))
        (U X) := by
    simpa [project, hXB, hRecip] using hPrefX
  rcases localPrefixSemantics_recvIf_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := X)
      ControlPayload.ctrlPattern B
      (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
      (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
      (U X) hRecvPref with hCases
  rcases hCases with hNil | hTrue | hFalse
  · exact Or.inl hNil
  · rcases hTrue with ⟨wX, _hPrefTrue, hXTrue⟩
    exact False.elim <|
      if_false_recipient_true_impossible
        (L := L) (C := C) (F := F) (Payload := Payload)
        c B X PTrue PFalse U V hXB hRecip sB rB wX wX hUB hSPref hSFull hXTrue hMSC
  · exact Or.inr hFalse

private theorem distSemantics_project_if_true_local
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
              (by
                intro X hX
                exact hX.1)
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
                    (by
                      intro Y hY
                      exact hY.1)
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
                        (by
                          intro Y hY
                          exact hY.1)
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
                    (by
                      intro Y hY
                      exact hY.1)) X =
                  [AlphabetOf.mkRecv (C := C) (F := F) X
                    (ControlPayload.setDecision true ControlPayload.ctrlPattern) B] :=
                controlBroadcastMSC_recipient (C := C) (F := F) (Payload := Payload)
                  B X
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  true
                  (by
                    intro Y hY
                    exact hY.1)
                  hRecip
              simp [WordTuple.concat, hChoice, hCtrl]⟩))
      · have hNoPartTrue : ¬ participationSet PTrue X := by
          intro hPart
          exact hRecip ⟨hXB, Or.inl hPart⟩
        have hMTrueNil : MTrue X = [] :=
          project_trace_nil_of_not_participating
            (L := L) (C := C) (F := F) (Payload := Payload) X PTrue hNoPartTrue
            (MTrue X) (hTraceTrue X)
        have hBroadcastNil :
            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              true
              (by
                intro Y hY
                exact hY.1)) X = [] :=
          controlBroadcastMSC_nonRecipient (C := C) (F := F) (Payload := Payload)
            B X
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true
            (by
              intro Y hY
              exact hY.1)
            hXB hRecip
        have hWord :
            (mscIfTrue (C := C) (F := F) (Payload := Payload) c B
              ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                    true
                    (by
                      intro Y hY
                      exact hY.1)
              ∘ₘ MTrue) X = [] := by
          have hChoice :
              mscIfTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceIfTrue (C := C) (F := F) (Payload := Payload) c B) hXB
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
          (by
            intro X hX
            exact hX.1))
    · exact hCompleteTrue

private theorem distSemantics_project_if_false_local
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
              (by
                intro X hX
                exact hX.1)
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
                    (by
                      intro Y hY
                      exact hY.1)
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
                        (by
                          intro Y hY
                          exact hY.1)
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
                    (by
                      intro Y hY
                      exact hY.1)) X =
                  [AlphabetOf.mkRecv (C := C) (F := F) X
                    (ControlPayload.setDecision false ControlPayload.ctrlPattern) B] :=
                controlBroadcastMSC_recipient (C := C) (F := F) (Payload := Payload)
                  B X
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false
                  (by
                    intro Y hY
                    exact hY.1)
                  hRecip
              simp [WordTuple.concat, hChoice, hCtrl]⟩))
      · have hNoPartFalse : ¬ participationSet PFalse X := by
          intro hPart
          exact hRecip ⟨hXB, Or.inr hPart⟩
        have hMFalseNil : MFalse X = [] :=
          project_trace_nil_of_not_participating
            (L := L) (C := C) (F := F) (Payload := Payload) X PFalse hNoPartFalse
            (MFalse X) (hTraceFalse X)
        have hBroadcastNil :
            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              false
              (by
                intro Y hY
                exact hY.1)) X = [] :=
          controlBroadcastMSC_nonRecipient (C := C) (F := F) (Payload := Payload)
            B X
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false
            (by
              intro Y hY
              exact hY.1)
            hXB hRecip
        have hWord :
            (mscIfFalse (C := C) (F := F) (Payload := Payload) c B
              ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                    false
                    (by
                      intro Y hY
                      exact hY.1)
              ∘ₘ MFalse) X = [] := by
          have hChoice :
              mscIfFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceIfFalse (C := C) (F := F) (Payload := Payload) c B) hXB
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
          (by
            intro X hX
            exact hX.1))
    · exact hCompleteFalse

private theorem distSemantics_project_while_exit_local
    (c : C) (B : L) (PBody PExit : Prog L C F Payload)
    (MExit : WordTuple L C F Payload)
    (hExit : distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PExit) MExit) :
    distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
        (.whileLoop c B PBody PExit))
      (mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
        ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              false
              (by intro X hX; exact hX.1)
        ∘ₘ MExit) := by
  rcases hExit with ⟨hTraceExit, hCompleteExit⟩
  refine ⟨?_, ?_⟩
  · intro X
    by_cases hXB : X = B
    · subst hXB
      simp only [projectDist, project]
      apply (localTraceSemantics_localWhile_unfold
        (L := L) (C := C) (F := F) (Payload := Payload) c _ _ _).mpr
      left
      refine ⟨(controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  X
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PBody PExit)
                  false)
              ++ MExit X,
              ?_, ?_⟩
      · exact localTraceSemantics_seq_intro
              (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                X
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PBody PExit)
                false)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
              _ _
              (controlBroadcast_trace (L := L) (C := C) (F := F) (Payload := Payload)
                X
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) X PBody PExit)
                false)
              (hTraceExit X)
      · simp [WordTuple.concat, mscWhileFalse, choiceWhileFalse,
              controlBroadcastMSC_decider, List.append_assoc]
    · by_cases hRecip :
        whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X
      · have hProjX : project (L := L) (C := C) (F := F) (Payload := Payload) X
              (.whileLoop c B PBody PExit) =
            LocProg.recvWhile ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit) := by
          simp [project, hXB, hRecip]
        simp only [projectDist, hProjX]
        apply (localTraceSemantics_recvWhile_unfold
          (L := L) (C := C) (F := F) (Payload := Payload)
          ControlPayload.ctrlPattern B _ _ _).mpr
        left
        refine ⟨MExit X, hTraceExit X, ?_⟩
        have hChoice :
            mscWhileFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
          mscChoice_other (C := C) (F := F) (Payload := Payload)
            B X (choiceWhileFalse (C := C) (F := F) (Payload := Payload) c B) hXB
        have hCtrl :
            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              false
              (by intro Y hY; exact hY.1)) X =
            [AlphabetOf.mkRecv (C := C) (F := F) X
              (ControlPayload.setDecision false ControlPayload.ctrlPattern) B] :=
          controlBroadcastMSC_recipient (C := C) (F := F) (Payload := Payload)
            B X
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            false
            (by intro Y hY; exact hY.1)
            hRecip
        simp [WordTuple.concat, hChoice, hCtrl]
      · have hNoPartBody : ¬ participationSet PBody X := by
          intro hPart; exact hRecip ⟨hXB, Or.inl hPart⟩
        have hNoPartExit : ¬ participationSet PExit X := by
          intro hPart; exact hRecip ⟨hXB, Or.inr hPart⟩
        have hMExitNil : MExit X = [] :=
          project_trace_nil_of_not_participating
            (L := L) (C := C) (F := F) (Payload := Payload) X PExit hNoPartExit
            (MExit X) (hTraceExit X)
        have hBroadcastNil :
            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              false
              (by intro Y hY; exact hY.1)) X = [] :=
          controlBroadcastMSC_nonRecipient (C := C) (F := F) (Payload := Payload)
            B X
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            false
            (by intro Y hY; exact hY.1)
            hXB hRecip
        have hWord :
            (mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
              ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    false
                    (by intro Y hY; exact hY.1)
              ∘ₘ MExit) X = [] := by
          have hChoice :
              mscWhileFalse (C := C) (F := F) (Payload := Payload) c B X = [] :=
            mscChoice_other (C := C) (F := F) (Payload := Payload)
              B X (choiceWhileFalse (C := C) (F := F) (Payload := Payload) c B) hXB
          simp [WordTuple.concat, hChoice, hBroadcastNil, hMExitNil]
        simpa [projectDist, project, hXB, hRecip, localTraceSemantics, hWord] using
          (localTraceSemantics_eps_nil
            (L := L) (C := C) (F := F) (Payload := Payload) (A := X))
  · apply concat_complete_complete
    · exact concat_complete_complete _ _
        (mscWhileFalse_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B)
        (controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          false
          (by intro X hX; exact hX.1))
    · exact hCompleteExit

private theorem distSemantics_project_while_body_step
    (c : C) (B : L) (PBody PExit : Prog L C F Payload)
    (MBody : WordTuple L C F Payload)
    (hBody : distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PBody) MBody)
    (MWhile : WordTuple L C F Payload)
    (hWhile : distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
        (.whileLoop c B PBody PExit)) MWhile) :
    distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
        (.whileLoop c B PBody PExit))
      (mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
        ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              true
              (by intro X hX; exact hX.1)
        ∘ₘ MBody ∘ₘ MWhile) := by
  rcases hBody with ⟨hTraceBody, hCompleteBody⟩
  rcases hWhile with ⟨hTraceWhile, hCompleteWhile⟩
  let bodyRecips :=
    whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit
  refine ⟨?_, ?_⟩
  · intro X
    by_cases hXB : X = B
    · subst hXB
      -- bodyRecips is definitionally whileRecipients X PBody PExit
      have hBodyRecips : bodyRecips = whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
          X PBody PExit := rfl
      simp only [projectDist, project, ← hBodyRecips]
      apply (localTraceSemantics_localWhile_unfold
        (L := L) (C := C) (F := F) (Payload := Payload) c _ _ _).mpr
      right
      have hTraceWhileX :
          localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (LocProg.localWhile c
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  X bodyRecips true) ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload)
                    X PBody)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  X bodyRecips false) ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload)
                    X PExit))
            (MWhile X) := by
        simpa [projectDist, project, hBodyRecips] using hTraceWhile X
      refine ⟨(controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
            X bodyRecips true) ++ MBody X,
          MWhile X, ?_, ?_, ?_⟩
      · exact localTraceSemantics_seq_intro _ _ _ _
            (controlBroadcast_trace (L := L) (C := C) (F := F) (Payload := Payload)
              X bodyRecips true)
            (hTraceBody X)
      · exact hTraceWhileX
      · simp [WordTuple.concat, mscWhileTrue, choiceWhileTrue, hBodyRecips,
          controlBroadcastMSC_decider, List.append_assoc]
    · by_cases hRecip : bodyRecips X
      · have hWhRecip : whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
              B PBody PExit X := hRecip
        have hProjX : project (L := L) (C := C) (F := F) (Payload := Payload) X
              (.whileLoop c B PBody PExit) =
            LocProg.recvWhile ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit) := by
          simp [project, dif_neg hXB, if_pos hWhRecip]
        simp only [projectDist, hProjX]
        apply (localTraceSemantics_recvWhile_unfold
          (L := L) (C := C) (F := F) (Payload := Payload)
          ControlPayload.ctrlPattern B _ _ _).mpr
        right
        have hTraceWhileX :
            localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
              (LocProg.recvWhile ControlPayload.ctrlPattern B
                (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit))
              (MWhile X) := by
          have h := hTraceWhile X
          simp only [projectDist, hProjX] at h
          exact h
        refine ⟨MBody X, MWhile X, hTraceBody X, hTraceWhileX, ?_⟩
        have hChoice : mscWhileTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
          mscChoice_other (C := C) (F := F) (Payload := Payload)
            B X (choiceWhileTrue (C := C) (F := F) (Payload := Payload) c B) hXB
        have hCtrl :
            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              true (by intro Y hY; exact hY.1)) X =
            [AlphabetOf.mkRecv (C := C) (F := F) X
              (ControlPayload.setDecision true ControlPayload.ctrlPattern) B] :=
          controlBroadcastMSC_recipient (C := C) (F := F) (Payload := Payload)
            B X (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            true (by intro Y hY; exact hY.1) hWhRecip
        simp [WordTuple.concat, hChoice, hCtrl]
      · have hNoPartBody : ¬ participationSet PBody X := fun h => hRecip ⟨hXB, Or.inl h⟩
        have hNoPartExit : ¬ participationSet PExit X := fun h => hRecip ⟨hXB, Or.inr h⟩
        have hBodyNil : MBody X = [] :=
          project_trace_nil_of_not_participating
            (L := L) (C := C) (F := F) (Payload := Payload) X PBody hNoPartBody
            (MBody X) (hTraceBody X)
        have hWhNotRecip' : ¬ whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
            B PBody PExit X := hRecip
        have hWhileNil : MWhile X = [] := by
          have hProjEps :
              projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                (.whileLoop c B PBody PExit) X = LocProg.eps := by
            simp [projectDist, project, dif_neg hXB, if_neg hWhNotRecip']
          have hTW := hTraceWhile X
          rw [hProjEps] at hTW
          simpa using hTW
        have hChoice : mscWhileTrue (C := C) (F := F) (Payload := Payload) c B X = [] :=
          mscChoice_other (C := C) (F := F) (Payload := Payload)
            B X (choiceWhileTrue (C := C) (F := F) (Payload := Payload) c B) hXB
        have hBroadcastNil :
            (controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B bodyRecips true (by intro Y hY; exact hY.1)) X = [] :=
          controlBroadcastMSC_nonRecipient (C := C) (F := F) (Payload := Payload)
            B X bodyRecips true (by intro Y hY; exact hY.1) hXB hRecip
        have hWord :
            (mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
              ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                    B bodyRecips true (by intro Y hY; exact hY.1)
              ∘ₘ MBody ∘ₘ MWhile) X = [] := by
          simp [WordTuple.concat, hChoice, hBroadcastNil, hBodyNil, hWhileNil]
        have hWhNotRecip : ¬ whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
            B PBody PExit X := hRecip
        have hProjEpsX : project (L := L) (C := C) (F := F) (Payload := Payload) X
            (.whileLoop c B PBody PExit) = LocProg.eps := by
          simp [project, dif_neg hXB, if_neg hWhNotRecip]
        rw [show projectDist (L := L) (C := C) (F := F) (Payload := Payload)
            (.whileLoop c B PBody PExit) X = LocProg.eps from hProjEpsX, hWord]
        exact localTraceSemantics_eps_nil
          (L := L) (C := C) (F := F) (Payload := Payload) (A := X)
  · apply concat_complete_complete
    · apply concat_complete_complete
      · apply concat_complete_complete
        · exact mscWhileTrue_isCompleteMSC (C := C) (F := F) (Payload := Payload) c B
        · exact controlBroadcastMSC_isCompleteMSC (C := C) (F := F) (Payload := Payload)
            B bodyRecips true (by intro X hX; exact hX.1)
      · exact hCompleteBody
    · exact hCompleteWhile

private theorem msg_recv_prefix_forces_sender_prefix
    (A : L) (xs : Payload) (B : L) (ys : Payload) (hAB : A ≠ B)
    (U V : WordTuple L C F Payload)
    (hAPref :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) A (.msg A xs B ys hAB))
        (U A))
    (hASide :
      V A ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) A (.msg A xs B ys hAB))
          (U A))
    (hBRecv : U B = [AlphabetOf.mkRecv (C := C) (F := F) B ys A])
    (hMSC : IsMSC (U ∘ₘ V)) :
    U A = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] := by
  have hACases :
      U A = [] ∨ U A = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] :=
    localPrefixSemantics_send_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := A) xs B hAB (U A) (by simpa [project] using hAPref)
  cases hACases with
  | inr hAFull =>
      exact hAFull
  | inl hANil =>
      by_cases hVA : V A = []
      · have hNoUnmatched := hMSC.noUnmatchedRecv A B
        have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) A B := by
          rw [rcvCount, WordTuple.concat, hBRecv]
          have hRecv :
              Letter.isRecvFrom A (AlphabetOf.mkRecv (C := C) (F := F) B ys A).val = true := by
            simp [AlphabetOf.mkRecv, Letter.isRecvFrom]
          simp [countRecvs, hRecv]
        have hSndZero : sndCount (U ∘ₘ V) A B = 0 := by
          simp [sndCount, WordTuple.concat, hANil, hVA, countSends]
        omega
      · have hATrace := hASide hVA
        have : U A = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] :=
          localTraceSemantics_send_eq
            (L := L) (C := C) (F := F) (Payload := Payload)
            (A := A) xs B hAB (U A) (by simpa [project] using hATrace)
        rw [hANil] at this
        simp at this

private theorem msg_send_only_forces_receiver_tail_nil
    (A : L) (xs : Payload) (B : L) (ys : Payload) (hAB : A ≠ B)
    (U V : WordTuple L C F Payload)
    (hBPref :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) B (.msg A xs B ys hAB))
        (U B))
    (hBSide :
      V B ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) B (.msg A xs B ys hAB))
          (U B))
    (hBNil : U B = []) :
    V B = [] := by
  by_cases hVB : V B = []
  · exact hVB
  · have hBTrace := hBSide hVB
    have hBA : B ≠ A := hAB.symm
    have : U B = [AlphabetOf.mkRecv (C := C) (F := F) B ys A] :=
      localTraceSemantics_recv_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        (A := B) ys A (U B) (by simpa [project, hBA] using hBTrace)
    rw [hBNil] at this
    simp at this

private theorem msg_prefix_cases
    (A : L) (xs : Payload) (B : L) (ys : Payload) (hAB : A ≠ B)
    (U V : WordTuple L C F Payload)
    (hAPref :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) A (.msg A xs B ys hAB))
        (U A))
    (hASide :
      V A ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) A (.msg A xs B ys hAB))
          (U A))
    (hBPref :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) B (.msg A xs B ys hAB))
        (U B))
    (hBSide :
      V B ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) B (.msg A xs B ys hAB))
          (U B))
    (hMSC : IsMSC (U ∘ₘ V)) :
    (U A = [] ∧ U B = []) ∨
    (U A = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] ∧ U B = [] ∧ V B = []) ∨
    (U A = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] ∧
      U B = [AlphabetOf.mkRecv (C := C) (F := F) B ys A]) := by
  have hACases :
      U A = [] ∨ U A = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] :=
    localPrefixSemantics_send_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := A) xs B hAB (U A) (by simpa [project] using hAPref)
  have hBA : B ≠ A := hAB.symm
  have hBCases :
      U B = [] ∨ U B = [AlphabetOf.mkRecv (C := C) (F := F) B ys A] :=
    localPrefixSemantics_recv_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := B) ys A (U B) (by simpa [project, hBA] using hBPref)
  cases hACases with
  | inl hANil =>
      cases hBCases with
      | inl hBNil =>
          exact Or.inl ⟨hANil, hBNil⟩
      | inr hBRecv =>
          have hAFull :=
            msg_recv_prefix_forces_sender_prefix
              (L := L) (C := C) (F := F) (Payload := Payload)
              A xs B ys hAB U V hAPref hASide hBRecv hMSC
          exact False.elim (by simpa [hANil] using hAFull)
  | inr hASend =>
      cases hBCases with
      | inl hBNil =>
          have hVBNil :=
            msg_send_only_forces_receiver_tail_nil
              (L := L) (C := C) (F := F) (Payload := Payload)
              A xs B ys hAB U V hBPref hBSide hBNil
          exact Or.inr (Or.inl ⟨hASend, hBNil, hVBNil⟩)
      | inr hBRecv =>
          exact Or.inr (Or.inr ⟨hASend, hBRecv⟩)

private theorem msg_send_only_completion_msc
    (A : L) (xs : Payload) (B : L) (ys : Payload) (hAB : A ≠ B)
    (hCompat : PayloadCompatible Payload xs ys)
    (V : WordTuple L C F Payload)
    (hVB : V B = [])
    (hMSC :
      IsMSC
        ((msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB) ∘ₘ V)) :
    IsMSC ((mscMsg (C := C) (F := F) A xs B ys hAB) ∘ₘ V) := by
  classical
  let Mold :=
    (msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB) ∘ₘ V
  let Mnew := (mscMsg (C := C) (F := F) A xs B ys hAB) ∘ₘ V
  have hSameOffB : ∀ X, X ≠ B → Mnew X = Mold X := by
    intro X hXB
    by_cases hXA : X = A
    · subst X
      simp [Mnew, Mold, WordTuple.concat, mscMsg_sender, msgSendPrefix_owner]
    · simp [Mnew, Mold, WordTuple.concat, mscMsg_other, msgSendPrefix_other, hXA, hXB]
  have hMnewB : Mnew B = [AlphabetOf.mkRecv (C := C) (F := F) B ys A] := by
    simp [Mnew, WordTuple.concat, mscMsg_receiver, hVB]
  have hMoldB : Mold B = [] := by
    have hBA : B ≠ A := by simpa [eq_comm] using hAB
    simp [Mold, WordTuple.concat, hVB, msgSendPrefix, hBA]
  refine ⟨?_, ?_, ?_⟩
  · intro S T
    by_cases hTB : T = B
    · subst T
      by_cases hSA : S = A
      · subst S
        have : 1 ≤ sndCount Mnew A B := by
          simp [Mnew, sndCount, WordTuple.concat, mscMsg_sender, countSends,
            AlphabetOf.mkSend, Letter.isSendTo]
        simpa [Mnew, rcvCount, WordTuple.concat, mscMsg_receiver, hVB, countRecvs,
          AlphabetOf.mkRecv, Letter.isRecvFrom] using this
      · have hRcvZero : rcvCount Mnew S B = 0 := by
          have hAS : A ≠ S := by simpa [eq_comm] using hSA
          simp [Mnew, rcvCount, WordTuple.concat, mscMsg_receiver, hVB, countRecvs,
            AlphabetOf.mkRecv, Letter.isRecvFrom, hAS]
        rw [hRcvZero]
        exact Nat.zero_le _
    · by_cases hSB : S = B
      · subst S
        have hRcvEq : rcvCount Mnew B T = rcvCount Mold B T := by
          rw [rcvCount, rcvCount, hSameOffB T hTB]
        have hSndNew : sndCount Mnew B T = 0 := by
          simp [sndCount, hMnewB, countSends, AlphabetOf.mkRecv, Letter.isSendTo]
        have hSndOld : sndCount Mold B T = 0 := by
          simp [sndCount, hMoldB, countSends]
        have hOld := hMSC.noUnmatchedRecv B T
        rw [hSndOld] at hOld
        rw [hRcvEq, hSndNew]
        exact hOld
      · have hSndEq : sndCount Mnew S T = sndCount Mold S T := by
          rw [sndCount, sndCount, hSameOffB S hSB]
        have hRcvEq : rcvCount Mnew S T = rcvCount Mold S T := by
          rw [rcvCount, rcvCount, hSameOffB T hTB]
        rw [hSndEq, hRcvEq]
        exact hMSC.noUnmatchedRecv S T
  · intro S T p hp
    by_cases hTB : T = B
    · subst T
      by_cases hSA : S = A
      · subst S
        have hp' :
            p ∈
              List.zip
                (sendPayloads (C := C) (F := F) (Payload := Payload) B (Mnew A))
                (recvPayloads (C := C) (F := F) (Payload := Payload) A (Mnew B)) := hp
        simp [Mnew, WordTuple.concat, mscMsg_sender, mscMsg_receiver, hVB,
          sendPayloads, recvPayloads, AlphabetOf.mkSend, AlphabetOf.mkRecv, hAB] at hp'
        rcases hp' with rfl
        exact hCompat
      · exfalso
        have hAS : A ≠ S := by simpa [eq_comm] using hSA
        simpa [Mnew, WordTuple.concat, mscMsg_receiver, hVB, recvPayloads,
          AlphabetOf.mkRecv, hAS] using hp
    · have hSendEq :
          sendPayloads (C := C) (F := F) (Payload := Payload) T (Mnew S) =
            sendPayloads (C := C) (F := F) (Payload := Payload) T (Mold S) := by
        by_cases hSB : S = B
        · subst S
          simp [hMnewB, hMoldB, sendPayloads, AlphabetOf.mkRecv]
        · rw [hSameOffB S hSB]
      have hRecvEq :
          recvPayloads (C := C) (F := F) (Payload := Payload) S (Mnew T) =
            recvPayloads (C := C) (F := F) (Payload := Payload) S (Mold T) := by
        rw [hSameOffB T hTB]
      have hp' :
          p ∈
            List.zip
              (sendPayloads (C := C) (F := F) (Payload := Payload) T (Mold S))
              (recvPayloads (C := C) (F := F) (Payload := Payload) S (Mold T)) := by
        rw [hSendEq, hRecvEq] at hp
        exact hp
      exact hMSC.labelCompat S T p hp'
  · refine ⟨?_⟩
    let R := Classical.choice hMSC.acyclic
    refine
      { rank := fun e =>
          if hB : e.lifeline = B then
            2 * R.rank ⟨A, 0⟩ + 1
          else
            2 * R.rank e
        local_mono := ?_
        fifo_mono := ?_ }
    · intro e1 e2 hSameLL hLt hpos1 hpos2
      by_cases hB1 : e1.lifeline = B
      · have hB2 : e2.lifeline = B := by simpa [hSameLL] using hB1
        have hLen1 : (Mnew e1.lifeline).length = 1 := by
          subst hB1
          simp [hMnewB]
        have hLen2 : (Mnew e2.lifeline).length = 1 := by
          subst hB2
          simp [hMnewB]
        rw [hLen1] at hpos1
        rw [hLen2] at hpos2
        omega
      · have hB2 : e2.lifeline ≠ B := by
          intro hB2
          exact hB1 (by simpa [hSameLL] using hB2)
        have hpos1' : e1.pos < (Mold e1.lifeline).length := by
          have hLenEq :
              (Mnew e1.lifeline).length = (Mold e1.lifeline).length := by
            rw [hSameOffB e1.lifeline hB1]
          exact hLenEq ▸ hpos1
        have hpos2' : e2.pos < (Mold e2.lifeline).length := by
          have hLenEq :
              (Mnew e2.lifeline).length = (Mold e2.lifeline).length := by
            rw [hSameOffB e2.lifeline hB2]
          exact hLenEq ▸ hpos2
        have hR := R.local_mono e1 e2 hSameLL hLt hpos1' hpos2'
        simp [hB1, hB2]
        omega
    · intro S T j1 j2 hj1 hj2 hSend hRecv hCntEq
      by_cases hTB : T = B
      · subst T
        have hS : S = A := by
          simpa [eq_comm, Mnew, WordTuple.concat, mscMsg_receiver, hVB,
            AlphabetOf.mkRecv, Letter.isRecvFrom] using hRecv
        subst S
        have hJ2 : j2 = 0 := by
          have hLen : (Mnew B).length = 1 := by simp [hMnewB]
          rw [hLen] at hj2
          omega
        subst hJ2
        have hJ1 : j1 = 0 := by
          cases j1 with
          | zero => rfl
          | succ k =>
              exfalso
              have hPos :
                  1 ≤ countSends B ((Mnew A).take (Nat.succ k)) := by
                simp [Mnew, WordTuple.concat, mscMsg_sender, countSends,
                  AlphabetOf.mkSend, Letter.isSendTo]
              have hCntZero :
                  countSends B ((Mnew A).take (Nat.succ k)) = 0 := by
                simpa [Mnew, WordTuple.concat, mscMsg_receiver, hVB] using hCntEq
              omega
        subst hJ1
        simp [hAB]
      · have hSB : S ≠ B := by
          intro hSB
          subst hSB
          simp [Mnew, WordTuple.concat, mscMsg_receiver, hVB,
            AlphabetOf.mkRecv, Letter.isSendTo] at hSend
        have hj1' : j1 < (Mold S).length := by
          have hLenEq : (Mnew S).length = (Mold S).length := by
            rw [hSameOffB S hSB]
          exact hLenEq ▸ hj1
        have hj2' : j2 < (Mold T).length := by
          have hLenEq : (Mnew T).length = (Mold T).length := by
            rw [hSameOffB T hTB]
          exact hLenEq ▸ hj2
        have hSend' :
            ((Mold S).get ⟨j1, hj1'⟩).val.isSendTo T = true := by
          have hEqS :
              ((mscMsg (C := C) (F := F) A xs B ys hAB) ∘ₘ V) S = Mold S := by
            simpa [Mnew] using hSameOffB S hSB
          simpa [hEqS] using hSend
        have hRecv' :
            ((Mold T).get ⟨j2, hj2'⟩).val.isRecvFrom S = true := by
          have hEqT :
              ((mscMsg (C := C) (F := F) A xs B ys hAB) ∘ₘ V) T = Mold T := by
            simpa [Mnew] using hSameOffB T hTB
          simpa [hEqT] using hRecv
        have hCntEq' :
            countSends T ((Mold S).take j1) = countRecvs S ((Mold T).take j2) := by
          have hEqS :
              ((mscMsg (C := C) (F := F) A xs B ys hAB) ∘ₘ V) S = Mold S := by
            simpa [Mnew] using hSameOffB S hSB
          have hEqT :
              ((mscMsg (C := C) (F := F) A xs B ys hAB) ∘ₘ V) T = Mold T := by
            simpa [Mnew] using hSameOffB T hTB
          simpa [hEqS, hEqT] using hCntEq
        have hR := R.fifo_mono S T j1 j2 hj1' hj2' hSend' hRecv' hCntEq'
        simp [hSB, hTB]
        omega

/-- The zipper property packaged as an induction target on global programs. -/
def UniformZipperProperty
    (P : Prog L C F Payload) : Prop :=
  WellTypedProgram P →
    ∀ U V : WordTuple L C F Payload,
      (∀ X,
        localPrefixSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X P)
          (U X)) →
      (∀ X, V X ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X P)
          (U X)) →
      IsMSC (U ∘ₘ V) →
      ∃ Ubar,
        ZipPost (L := L) (C := C) (F := F) (Payload := Payload) P U Ubar V

/-- The sequential constructor of the zipper proof: if the zipper property
    holds for `Q1` and `Q2`, then it holds for `Q1 ;; Q2`. -/
theorem uniformZipper_seq
    (Q1 Q2 : Prog L C F Payload)
    (h1 : UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) Q1)
    (h2 : UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) Q2) :
    UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) (Q1 ;; Q2) := by
  classical
  intro hWF U V hPref hSide hMSC
  rcases hWF with ⟨hWF1, hWF2⟩
  have hSplit : ∀ X : L,
      ∃ u1 u2,
        U X = u1 ++ u2 ∧
        localPrefixSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X Q1) u1 ∧
        localPrefixSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X Q2) u2 ∧
        (u2 ++ V X ≠ [] →
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X Q1) u1) ∧
        (V X ≠ [] →
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X Q2) u2) := by
    intro X
    exact localSeqBoundarySplit
      (L := L) (C := C) (F := F) (Payload := Payload)
      (project (L := L) (C := C) (F := F) (Payload := Payload) X Q1)
      (project (L := L) (C := C) (F := F) (Payload := Payload) X Q2)
      (U X) (V X)
      (by simpa [projectDist, project] using hPref X)
      (by simpa [projectDist, project] using hSide X)
  let U1 : WordTuple L C F Payload := fun X => Classical.choose (hSplit X)
  let U2 : WordTuple L C F Payload := fun X => Classical.choose (Classical.choose_spec (hSplit X))
  have hUeq : ∀ X, U X = U1 X ++ U2 X := by
    intro X
    exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).1
  have hU1Pref : ∀ X,
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X Q1)
        (U1 X) := by
    intro X
    exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.1
  have hU2Pref : ∀ X,
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X Q2)
        (U2 X) := by
    intro X
    exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.2.1
  have hU1Side : ∀ X, (U2 X ++ V X) ≠ [] →
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X Q1)
        (U1 X) := by
    intro X
    exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.2.2.1
  have hU2Side : ∀ X, V X ≠ [] →
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X Q2)
        (U2 X) := by
    intro X
    exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.2.2.2
  have hConcat : U ∘ₘ V = U1 ∘ₘ (U2 ∘ₘ V) := by
    ext X
    simp [WordTuple.concat, hUeq X, List.append_assoc]
  rcases h1 hWF1 U1 (U2 ∘ₘ V) hU1Pref
      (by
        intro X hXV
        simpa [WordTuple.concat] using hU1Side X hXV)
      (by simpa [hConcat] using hMSC) with ⟨U1bar, hZip1⟩
  have hSuffixMSC : IsMSC (U2 ∘ₘ V) := zipPost_suffix_isMSC hZip1
  rcases hZip1 with ⟨hZip1Loc, hZip1Complete, hZip1Orig, hZip1MSC⟩
  rcases h2 hWF2 U2 V hU2Pref hU2Side
      hSuffixMSC with ⟨U2bar, hZip2⟩
  have hUConcat : U = U1 ∘ₘ U2 := by
    ext X
    simp [WordTuple.concat, hUeq X]
  refine ⟨U1bar ∘ₘ U2bar, ?_⟩
  simpa [hUConcat] using
    (zipPost_seq (L := L) (C := C) (F := F) (Payload := Payload)
      Q1 Q2 U1 U1bar U2 U2bar V
      ⟨hZip1Loc, hZip1Complete, hZip1Orig, hZip1MSC⟩ hZip2)

/-- The empty program satisfies the zipper property trivially. -/
theorem uniformZipper_eps :
    UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) Prog.eps := by
  intro _hWF U V hPref _hSide hMSC
  refine ⟨mscEmpty (L := L) (C := C) (F := F) (Payload := Payload), ?_⟩
  have hU : U = mscEmpty (L := L) (C := C) (F := F) (Payload := Payload) := by
    funext X
    have hPrefX :
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (LocProg.eps : LocProg L C F Payload X) (U X) := by
      simpa [project] using hPref X
    rw [mscEmpty, WordTuple.empty]
    exact localPrefixSemantics_eps_eq_nil
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := X) (U X) hPrefX
  subst hU
  simpa [mscEmpty, WordTuple.empty, WordTuple.concat] using
    zipPost_of_complete (L := L) (C := C) (F := F) (Payload := Payload)
      Prog.eps
      (mscEmpty (L := L) (C := C) (F := F) (Payload := Payload))
      V
      (by
        intro X
        simp [project, localTraceSemantics, mscEmpty, WordTuple.empty])
      (mscEmpty_isCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload))
      (by simpa [mscEmpty, WordTuple.empty, WordTuple.concat] using hMSC)

theorem uniformZipper_act
    (A : L) (ys : Payload) (f : F) (xs : Payload) :
    UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) (.act A ys f xs) := by
  intro _hWF U V hPref hSide hMSC
  have hOwnerPref :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (LocProg.act (A := A) ys f xs) (U A) := by
    simpa [project] using hPref A
  have hOwnerCases :
      U A = [] ∨ U A = [AlphabetOf.mkAct (C := C) A ys f xs] :=
    localPrefixSemantics_act_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := A) ys f xs (U A) hOwnerPref
  have hOther : ∀ X, X ≠ A → U X = [] := by
    intro X hXA
    exact localPrefixSemantics_eps_eq_nil
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := X) (U X)
      (by simpa [project, hXA] using hPref X)
  cases hOwnerCases with
  | inl hOwnerNil =>
      have hUEmpty : U = mscEmpty (L := L) (C := C) (F := F) (Payload := Payload) := by
        funext X
        by_cases hXA : X = A
        · subst X
          simpa [mscEmpty, WordTuple.empty] using hOwnerNil
        · simpa [mscEmpty, WordTuple.empty] using hOther X hXA
      have hV : IsMSC V := by
        simpa [hUEmpty, mscEmpty, WordTuple.empty, WordTuple.concat] using hMSC
      refine ⟨mscAct (C := C) A ys f xs, ?_⟩
      refine ⟨?_, mscAct_isCompleteMSC (C := C) (F := F) A ys f xs, ?_, ?_⟩
      · intro X
        by_cases hXA : X = A
        · subst X
          refine ⟨⟨[AlphabetOf.mkAct (C := C) A ys f xs], ?_⟩, ?_, ?_⟩
          · rw [hOwnerNil]
            simp [mscAct_owner, IsPrefixWord]
          · simpa [project, localTraceSemantics, mscAct_owner]
          · intro hVX
            exfalso
            have hSideA := hSide A hVX
            simpa [project, localTraceSemantics, hOwnerNil] using hSideA
        · refine ⟨⟨[], ?_⟩, ?_, ?_⟩
          · rw [hOther X hXA, mscAct_other (C := C) A X ys f xs hXA]
            simp [IsPrefixWord]
          · simpa [project, hXA, localTraceSemantics]
          · intro _hVX
            rw [mscAct_other (C := C) A X ys f xs hXA, hOther X hXA]
      · simpa [hUEmpty, mscEmpty, WordTuple.empty, WordTuple.concat] using hMSC
      · exact concat_complete_msc _ _
          (mscAct_isCompleteMSC (C := C) (F := F) A ys f xs) hV
  | inr hOwnerFull =>
      have hUeq : U = mscAct (C := C) A ys f xs := by
        funext X
        by_cases hXA : X = A
        · subst X
          simpa [mscAct_owner] using hOwnerFull
        · rw [hOther X hXA, mscAct_other (C := C) A X ys f xs hXA]
      subst hUeq
      refine ⟨mscAct (C := C) A ys f xs, ?_⟩
      exact zipPost_of_complete
        (L := L) (C := C) (F := F) (Payload := Payload)
        (.act A ys f xs)
        (mscAct (C := C) A ys f xs)
        V
        (by
          intro X
          by_cases hXA : X = A
          · subst X
            simpa [project, localTraceSemantics, mscAct_owner]
          · simpa [project, hXA, localTraceSemantics, mscAct_other, hXA])
        (mscAct_isCompleteMSC (C := C) (F := F) A ys f xs)
        hMSC

theorem uniformZipper_msg
    (A : L) (xs : Payload) (B : L) (ys : Payload) (hAB : A ≠ B) :
    UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) (.msg A xs B ys hAB) := by
  intro hWF U V hPref hSide hMSC
  have hCompat : PayloadCompatible Payload xs ys := by
    simpa [WellTypedProgram] using hWF
  have hAPref :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) A (.msg A xs B ys hAB))
        (U A) := by
    simpa [project] using hPref A
  have hBPref :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) B (.msg A xs B ys hAB))
        (U B) := by
    simpa [project, hAB.symm] using hPref B
  have hOther : ∀ X, X ≠ A → X ≠ B → U X = [] := by
    intro X hXA hXB
    exact localPrefixSemantics_eps_eq_nil
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := X) (U X)
      (by simpa [project, hXA, hXB] using hPref X)
  have hCases :=
    msg_prefix_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      A xs B ys hAB U V hAPref (hSide A) hBPref (hSide B) hMSC
  cases hCases with
  | inl hNil =>
      rcases hNil with ⟨hANil, hBNil⟩
      have hVA : V A = [] := by
        by_cases hVAN : V A = []
        · exact hVAN
        · have hATrace := hSide A hVAN
          have : U A = [AlphabetOf.mkSend (C := C) (F := F) A xs B hAB] :=
            localTraceSemantics_send_eq
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := A) xs B hAB (U A) (by simpa [project] using hATrace)
          rw [hANil] at this
          simp at this
      have hVB : V B = [] := by
        by_cases hVBN : V B = []
        · exact hVBN
        · have hBTrace := hSide B hVBN
          have : U B = [AlphabetOf.mkRecv (C := C) (F := F) B ys A] :=
            localTraceSemantics_recv_eq
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := B) ys A (U B) (by simpa [project, hAB.symm] using hBTrace)
          rw [hBNil] at this
          simp at this
      have hUEmpty : U = mscEmpty (L := L) (C := C) (F := F) (Payload := Payload) := by
        funext X
        by_cases hXA : X = A
        · subst X
          simpa [mscEmpty, WordTuple.empty] using hANil
        · by_cases hXB : X = B
          · subst X
            simpa [mscEmpty, WordTuple.empty] using hBNil
          · simpa [mscEmpty, WordTuple.empty] using hOther X hXA hXB
      have hV : IsMSC V := by
        simpa [hUEmpty, mscEmpty, WordTuple.empty, WordTuple.concat] using hMSC
      refine ⟨mscMsg (C := C) (F := F) A xs B ys hAB, ?_⟩
      refine ⟨?_, mscMsg_isCompleteMSC (C := C) (F := F) A xs B ys hAB hCompat, hMSC, ?_⟩
      · intro X
        by_cases hXA : X = A
        · subst X
          refine ⟨⟨[AlphabetOf.mkSend (C := C) (F := F) A xs B hAB], ?_⟩, ?_, ?_⟩
          · rw [hANil]
            simp [mscMsg_sender, IsPrefixWord]
          · simpa [project, localTraceSemantics, mscMsg_sender]
          · intro hVX
            exfalso
            have hSideA := hSide A hVX
            simpa [project, localTraceSemantics, hANil] using hSideA
        · by_cases hXB : X = B
          · subst X
            refine ⟨⟨[AlphabetOf.mkRecv (C := C) (F := F) B ys A], ?_⟩, ?_, ?_⟩
            · rw [hBNil]
              simp [mscMsg_receiver, IsPrefixWord]
            · simpa [project, hAB.symm, localTraceSemantics, mscMsg_receiver]
            · intro hVX
              exfalso
              have hSideB := hSide B hVX
              simpa [project, hAB.symm, localTraceSemantics, hBNil] using hSideB
          · refine ⟨⟨[], ?_⟩, ?_, ?_⟩
            · rw [hOther X hXA hXB, mscMsg_other (C := C) (F := F) A xs B ys hAB X hXA hXB]
              simp [IsPrefixWord]
            · simpa [project, hXA, hXB, localTraceSemantics, mscMsg_other]
            · intro _
              rw [hOther X hXA hXB, mscMsg_other (C := C) (F := F) A xs B ys hAB X hXA hXB]
      · exact concat_complete_msc _ _
          (mscMsg_isCompleteMSC (C := C) (F := F) A xs B ys hAB hCompat) hV
  | inr hRest =>
      cases hRest with
      | inl hSendOnly =>
          rcases hSendOnly with ⟨hASend, hBNil, hVBNil⟩
          have hUeq :
              U =
                msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB := by
            funext X
            by_cases hXA : X = A
            · subst X
              simpa [msgSendPrefix_owner] using hASend
            · by_cases hXB : X = B
              · subst X
                have hBA : B ≠ A := by simpa [eq_comm] using hAB
                have hMsgB :
                    msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB B = [] := by
                  simp [msgSendPrefix, hBA]
                exact hBNil.trans hMsgB.symm
              · rw [hOther X hXA hXB, msgSendPrefix_other (L := L) (C := C) (F := F)
                    (Payload := Payload) A X xs B hAB hXA]
          have hOrigSendMSC :
              IsMSC
                ((msgSendPrefix (L := L) (C := C) (F := F) (Payload := Payload) A xs B hAB) ∘ₘ V) := by
            simpa [hUeq] using hMSC
          have hBarMSC :
              IsMSC ((mscMsg (C := C) (F := F) A xs B ys hAB) ∘ₘ V) := by
            exact msg_send_only_completion_msc
              (L := L) (C := C) (F := F) (Payload := Payload)
              A xs B ys hAB hCompat V hVBNil hOrigSendMSC
          have hV : IsMSC V := by
            exact suffix_msc_of_complete_prefix _ _
              (mscMsg_isCompleteMSC (C := C) (F := F) A xs B ys hAB hCompat)
              hBarMSC
          refine ⟨mscMsg (C := C) (F := F) A xs B ys hAB, ?_⟩
          refine ⟨?_, mscMsg_isCompleteMSC (C := C) (F := F) A xs B ys hAB hCompat, hMSC, hBarMSC⟩
          · intro X
            by_cases hXA : X = A
            · subst X
              refine ⟨⟨[], by simpa [hASend, mscMsg_sender, IsPrefixWord]⟩, ?_, ?_⟩
              · simpa [project, localTraceSemantics, mscMsg_sender]
              · intro _
                simpa [hASend, mscMsg_sender]
            · by_cases hXB : X = B
              · subst X
                refine ⟨⟨[AlphabetOf.mkRecv (C := C) (F := F) B ys A], ?_⟩, ?_, ?_⟩
                · rw [hBNil]
                  simp [mscMsg_receiver, IsPrefixWord]
                · simpa [project, hAB.symm, localTraceSemantics, mscMsg_receiver]
                · intro hVX
                  exfalso
                  exact hVX hVBNil
              · refine ⟨⟨[], ?_⟩, ?_, ?_⟩
                · rw [hOther X hXA hXB, mscMsg_other (C := C) (F := F) A xs B ys hAB X hXA hXB]
                  simp [IsPrefixWord]
                · simpa [project, hXA, hXB, localTraceSemantics, mscMsg_other]
                · intro _
                  rw [hOther X hXA hXB, mscMsg_other (C := C) (F := F) A xs B ys hAB X hXA hXB]
      | inr hFull =>
          rcases hFull with ⟨hASend, hBRecv⟩
          have hUeq : U = mscMsg (C := C) (F := F) A xs B ys hAB := by
            funext X
            by_cases hXA : X = A
            · subst X
              simpa [mscMsg_sender] using hASend
            · by_cases hXB : X = B
              · subst X
                simpa [mscMsg_receiver] using hBRecv
              · rw [hOther X hXA hXB, mscMsg_other (C := C) (F := F) A xs B ys hAB X hXA hXB]
          subst hUeq
          refine ⟨mscMsg (C := C) (F := F) A xs B ys hAB, ?_⟩
          exact zipPost_of_complete
            (L := L) (C := C) (F := F) (Payload := Payload)
            (.msg A xs B ys hAB)
            (mscMsg (C := C) (F := F) A xs B ys hAB)
            V
            (by
              intro X
              by_cases hXA : X = A
              · subst X
                simpa [project, localTraceSemantics, mscMsg_sender]
              · by_cases hXB : X = B
                · subst X
                  simpa [project, hAB.symm, localTraceSemantics, mscMsg_receiver]
                · simpa [project, hXA, hXB, localTraceSemantics, mscMsg_other])
            (mscMsg_isCompleteMSC (C := C) (F := F) A xs B ys hAB hCompat)
            hMSC

private theorem uniformZipper_if_nil_case
    (c : C) (B : L) (PTrue PFalse : Prog L C F Payload)
    (hFalse : UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) PFalse)
    (hWFFalse : WellTypedProgram PFalse)
    (U V : WordTuple L C F Payload)
    (hPref : ∀ X,
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
        (U X))
    (hSide : ∀ X, V X ≠ [] →
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
        (U X))
    (hMSC : IsMSC (U ∘ₘ V))
    (hBNil : U B = []) :
    ∃ Ubar,
      ZipPost (L := L) (C := C) (F := F) (Payload := Payload)
        (.ite c B PTrue PFalse) U Ubar V := by
  have hUNil : ∀ X, U X = [] := by
    intro X
    by_cases hXB : X = B
    · subst X
      exact hBNil
    · by_cases hRecip :
        ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X
      · have hXPref :
            localPrefixSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (LocProg.recvIf ControlPayload.ctrlPattern B
                (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
                (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse))
              (U X) := by
          simpa [project, hXB, hRecip] using hPref X
        rcases localPrefixSemantics_recvIf_cases
            (L := L) (C := C) (F := F) (Payload := Payload)
            (A := X)
            ControlPayload.ctrlPattern B
            (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
            (U X) hXPref with hCases
        rcases hCases with hXNil | hXTrue | hXFalse
        · exact hXNil
        · rcases hXTrue with ⟨u, _hu, hWord⟩
          have hVB : V B = [] := by
            by_cases hVB : V B = []
            · exact hVB
            · have hBTrace := hSide B hVB
              have hPartB :
                  participationSet (.ite c B PTrue PFalse) B := by
                simp
              exact False.elim <|
                (project_trace_ne_nil_of_if_participating
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  B c B PTrue PFalse (U B) hPartB hBTrace) hBNil
          have hNoUnmatched := hMSC.noUnmatchedRecv B X
          have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
            rw [rcvCount, WordTuple.concat, hWord]
            simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
          have hSndZero : sndCount (U ∘ₘ V) B X = 0 := by
            simp [sndCount, WordTuple.concat, hBNil, hVB, countSends]
          omega
        · rcases hXFalse with ⟨u, _hu, hWord⟩
          have hVB : V B = [] := by
            by_cases hVB : V B = []
            · exact hVB
            · have hBTrace := hSide B hVB
              have hPartB :
                  participationSet (.ite c B PTrue PFalse) B := by
                simp
              exact False.elim <|
                (project_trace_ne_nil_of_if_participating
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  B c B PTrue PFalse (U B) hPartB hBTrace) hBNil
          have hNoUnmatched := hMSC.noUnmatchedRecv B X
          have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
            rw [rcvCount, WordTuple.concat, hWord]
            simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
          have hSndZero : sndCount (U ∘ₘ V) B X = 0 := by
            simp [sndCount, WordTuple.concat, hBNil, hVB, countSends]
          omega
      · exact localPrefixSemantics_eps_eq_nil
          (L := L) (C := C) (F := F) (Payload := Payload)
          (A := X) (U X)
          (by simpa [project, hXB, hRecip] using hPref X)
  have hUEmpty : U = mscEmpty (L := L) (C := C) (F := F) (Payload := Payload) := by
    funext X
    simpa [mscEmpty, WordTuple.empty] using hUNil X
  rcases hFalse hWFFalse U V
      (by
        intro X
        rw [hUNil X]
        exact localPrefixSemantics_project_nil
          (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
      (by
        intro X hVX
        by_cases hPartFalse : participationSet PFalse X
        · have hPartIf : participationSet (.ite c B PTrue PFalse) X := by
            by_cases hXB : X = B
            · simp [hXB]
            · simp [hXB, hPartFalse]
          have hXTrace := hSide X hVX
          have hXNonempty :=
            project_trace_ne_nil_of_if_participating
              (L := L) (C := C) (F := F) (Payload := Payload)
              X c B PTrue PFalse (U X) hPartIf hXTrace
          rw [hUNil X] at hXNonempty
          simp at hXNonempty
        · rw [hUNil X]
          rcases exists_project_trace
              (L := L) (C := C) (F := F) (Payload := Payload) X PFalse with ⟨w, hw⟩
          have hNil :=
            project_trace_nil_of_not_participating
              (L := L) (C := C) (F := F) (Payload := Payload)
              X PFalse hPartFalse w hw
          rw [hNil] at hw
          simpa using hw)
      hMSC with ⟨Rbar, hZipFalse⟩
  have hV : IsMSC V := zipPost_suffix_isMSC hZipFalse
  rcases hZipFalse with ⟨hLocFalse, hCompleteFalse, _hOrigFalse, hMSCFalse⟩
  let Ubar :=
    mscIfFalse (C := C) (F := F) (Payload := Payload) c B
      ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false
            (by
              intro X hX
              exact hX.1)
      ∘ₘ Rbar
  have hTraceUbar :
      ∀ X,
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
          (Ubar X) := by
    have hDistFalse :
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PFalse) Rbar := by
      exact ⟨(fun X => (hLocFalse X).2.1), hCompleteFalse⟩
    intro X
    simpa [Ubar] using
      (distSemantics_project_if_false_local
        (L := L) (C := C) (F := F) (Payload := Payload)
        c B PTrue PFalse Rbar hDistFalse).1 X
  have hCompleteUbar : IsCompleteMSC Ubar := by
    have hDistFalse :
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PFalse) Rbar := by
      exact ⟨(fun X => (hLocFalse X).2.1), hCompleteFalse⟩
    simpa [Ubar] using
      (distSemantics_project_if_false_local
        (L := L) (C := C) (F := F) (Payload := Payload)
        c B PTrue PFalse Rbar hDistFalse).2
  refine ⟨Ubar, ?_⟩
  refine ⟨?_, hCompleteUbar, hMSC, concat_complete_msc _ _ hCompleteUbar hV⟩
  intro X
  refine ⟨⟨Ubar X, by simp [IsPrefixWord, hUNil X]⟩, hTraceUbar X, ?_⟩
  intro hVX
  have hNotPartIf : ¬ participationSet (.ite c B PTrue PFalse) X := by
    intro hPartIf
    have hXTrace := hSide X hVX
    have hXNonempty :=
      project_trace_ne_nil_of_if_participating
        (L := L) (C := C) (F := F) (Payload := Payload)
        X c B PTrue PFalse (U X) hPartIf hXTrace
    rw [hUNil X] at hXNonempty
    simp at hXNonempty
  have hUbarNil :
      Ubar X = [] := by
    exact project_trace_nil_of_not_participating
      (L := L) (C := C) (F := F) (Payload := Payload)
      X (.ite c B PTrue PFalse) hNotPartIf (Ubar X) (hTraceUbar X)
  rw [hUbarNil, hUNil X]

theorem uniformZipper_if
    (c : C) (B : L) (PTrue PFalse : Prog L C F Payload)
    (hTrue : UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) PTrue)
    (hFalse : UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) PFalse) :
    UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) (.ite c B PTrue PFalse) := by
  classical
  intro hWF U V hPref hSide hMSC
  rcases hWF with ⟨hWFTrue, hWFFalse⟩
  -- Get the decider's prefix semantics
  have hBPref :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (LocProg.localIf c
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
              B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse) true
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PTrue)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
              B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse) false
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PFalse))
        (U B) := by
    simpa [project] using hPref B
  -- Case analysis on the decider's prefix
  rcases localPrefixSemantics_localIf_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      c _ _ (U B) hBPref with hBNil | ⟨seqPref, hSeqPref, hUB⟩ | ⟨seqPref, hSeqPref, hUB⟩
  -- Nil case: delegate to uniformZipper_if_nil_case
  · exact uniformZipper_if_nil_case
      (L := L) (C := C) (F := F) (Payload := Payload)
      c B PTrue PFalse hFalse hWFFalse U V hPref hSide hMSC hBNil
  -- True branch
  · -- seqPref and hUB from the true case
    -- Split seqPref into broadcast prefix sB and PTrue prefix rB
    rcases localPrefixSemantics_seq_split
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse) true)
        (project (L := L) (C := C) (F := F) (Payload := Payload) B PTrue)
        seqPref hSeqPref with ⟨sB, rB, hSeqEq, hSBPref, hRBPref, hSBFull⟩
    have hUBeq : U B = AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB := by
      rw [hUB, hSeqEq, ← List.cons_append]
    have hSBTrace : rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true)
          sB := by
      intro hTail
      by_cases hrB : rB = []
      · have hVB : V B ≠ [] := by
          simpa [hrB] using hTail
        have hBTrace := hSide B hVB
        have hSeqTrace :
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  true)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PTrue)
              (sB ++ rB) := by
          simpa [project, hUBeq, localTraceSemantics,
            AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse] using hBTrace
        rw [hrB, List.append_nil] at hSeqTrace
        rcases hSeqTrace with ⟨u1, u2, hu1, hu2, hsEq⟩
        have hWord :
            u1 =
              controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                true := by
          exact localTraceSemantics_controlBroadcast_eq
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true
            u1 hu1
        rcases localPrefixSemantics_controlBroadcast_prefix
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true
            sB hSBPref with ⟨t, ht⟩
        rw [hWord, ht] at hsEq
        have hNil : [] = t ++ u2 := by
          have hEq' : sB ++ [] = sB ++ (t ++ u2) := by
            simpa [List.append_assoc] using hsEq
          exact List.append_inj_right hEq' rfl
        have htNil : t = [] := by
          cases t <;> simp at hNil <;> simp
        rw [htNil] at ht
        simpa [hWord, ht] using hu1
      · exact hSBFull hrB
    -- Define stripped tuple: drop the broadcast prefix from each lifeline
    let kIf : L → Nat := fun X =>
      if X = B then 1 + sB.length
      else if ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X ∧ U X ≠ [] then 1
      else 0
    let R : WordTuple L C F Payload := fun X => (U X).drop (kIf X)
    -- R B = rB: (mkIfTrue :: sB) ++ rB drops first (1 + |sB|) elements = rB
    have hRB : R B = rB := by
      simp only [R, kIf, ite_true]
      rw [hUBeq, show 1 + sB.length = sB.length + 1 from by omega]
      simp [List.drop_succ_cons]
    -- R has lps(project X PTrue) for each X
    have hRPref : ∀ X,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
          (R X) := by
      intro X
      by_cases hXB : X = B
      · subst X
        rw [hRB]
        exact hRBPref
      · by_cases hRecip :
            ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X
        · rcases if_true_recipient_prefix_cases
              (L := L) (C := C) (F := F) (Payload := Payload)
              c B X PTrue PFalse U V hXB hRecip sB rB hUBeq hSBPref
              hSBTrace
              (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
          · simp only [R, kIf, hXB, ite_false, hRecip, hXNil, ite_true]
            simp
            exact localPrefixSemantics_project_nil
              (L := L) (C := C) (F := F) (Payload := Payload) X PTrue
          · simp only [R, kIf, hXB, ite_false, hRecip, hXWord, ite_true]
            simp
            exact huX
        · have hXNil : U X = [] :=
            localPrefixSemantics_eps_eq_nil
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := X) (U X)
              (by simpa [project, hXB, hRecip] using hPref X)
          simp only [R, kIf, hXB, ite_false, hRecip, hXNil, ite_false]
          simp
          exact localPrefixSemantics_project_nil
            (L := L) (C := C) (F := F) (Payload := Payload) X PTrue
    -- R side condition: V X ≠ [] → lts(project X PTrue)(R X)
    have hRSide : ∀ X, V X ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
          (R X) := by
      intro X hVX
      by_cases hXB : X = B
      · subst X
        rw [hRB]
        have hBTrace := hSide B hVX
        have hSeqTrace :
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  true)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PTrue)
              (sB ++ rB) := by
          simpa [project, hUBeq, localTraceSemantics,
            AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse] using hBTrace
        have hSWord :
            sB =
              controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                true := by
          exact localTraceSemantics_controlBroadcast_eq
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true
            sB
            (hSBTrace (by
              intro hNil
              exact hVX (List.eq_nil_of_append_eq_nil hNil).2))
        rcases hSeqTrace with ⟨u1, u2, hu1, hu2, hEq⟩
        have hWord :
            u1 =
              controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                true := by
          exact localTraceSemantics_controlBroadcast_eq
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true
            u1 hu1
        rw [hWord, hSWord] at hEq
        have hRBU2 : rB = u2 := by
          exact List.append_inj_right hEq rfl
        simpa [hRBU2] using hu2
      · by_cases hRecip :
            ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X
        · rcases if_true_recipient_prefix_cases
              (L := L) (C := C) (F := F) (Payload := Payload)
              c B X PTrue PFalse U V hXB hRecip sB rB hUBeq hSBPref
              hSBTrace
              (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
          · exfalso
            exact hVX (if_recipient_nil_tail_nil
              (L := L) (C := C) (F := F) (Payload := Payload)
              c B X PTrue PFalse U V hXB hRecip hSide hXNil)
          · simp only [R, kIf, hXB, ite_false, hRecip, hXWord, ite_true]
            simp
            have hXTrace := hSide X hVX
            have hXTrace' :
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
                  uX := by
              have hXCases :
                  localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
                    uX ∨
                  (localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
                    uX ∧
                    AlphabetOf.mkRecv (C := C) (F := F) X
                      (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B =
                    AlphabetOf.mkRecv (C := C) (F := F) X
                      (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B) := by
                simpa [project, hXB, hRecip, hXWord, localTraceSemantics] using hXTrace
              rcases hXCases with hTrue | ⟨_hFalse, hContra⟩
              · exact hTrue
              · have hPayloadEq :
                    ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern =
                      ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern := by
                  simpa [AlphabetOf.mkRecv] using hContra
                exact False.elim (ctrlTrue_ne_ctrlFalse hPayloadEq)
            exact hXTrace'
        · have hXNil : U X = [] :=
            localPrefixSemantics_eps_eq_nil
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := X) (U X)
              (by simpa [project, hXB, hRecip] using hPref X)
          have hNotPartTrue : ¬ participationSet PTrue X := by
            intro hPartTrue
            exact hRecip ⟨hXB, Or.inl hPartTrue⟩
          have hNilTrace :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
                [] := by
            rcases exists_project_trace
                (L := L) (C := C) (F := F) (Payload := Payload) X PTrue with ⟨w, hw⟩
            have hwNil :=
              project_trace_nil_of_not_participating
                (L := L) (C := C) (F := F) (Payload := Payload)
                X PTrue hNotPartTrue w hw
            rw [hwNil] at hw
            simpa using hw
          simpa [R, kIf, hXB, hRecip, hXNil] using hNilTrace
    -- IsMSC(R ∘ₘ V)
    have hRMSC : IsMSC (R ∘ₘ V) := by
      have hkU : ∀ X, kIf X ≤ (U X).length := by
        intro X
        by_cases hXB : X = B
        · subst X
          simp [kIf, hUBeq]
          omega
        · by_cases hRecipX :
              ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X ∧
                U X ≠ []
          · cases hUX : U X with
            | nil => exact False.elim (hRecipX.2 hUX)
            | cons hd tl => simp [kIf, hXB, hRecipX, hUX]
          · simp [kIf, hXB, hRecipX]
      have hTakeB :
          List.take (1 + sB.length) (U B ++ V B) =
            AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB := by
        rw [hUBeq]
        rw [List.append_assoc]
        have hLen' :
            1 + sB.length ≤
              (AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB).length := by
          simpa [Nat.add_comm] using Nat.le_refl (sB.length + 1)
        have hTake' :
            List.take (1 + sB.length)
              ((AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB) ++
                (rB ++ V B)) =
              List.take (1 + sB.length)
                (AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB) := by
          simpa using
            (List.take_append_of_le_length
              (l₁ := AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB)
              (l₂ := rB ++ V B)
              (i := 1 + sB.length)
              hLen')
        have hTake'' :
            List.take (1 + sB.length)
              (AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB) =
            AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB := by
          simpa [Nat.add_comm] using
            (List.take_length
              (l := AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB))
        exact hTake'.trans hTake''
      have hTakeBM :
          ((U ∘ₘ V) B).take (kIf B) =
            AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) B c :: sB := by
        simpa [WordTuple.concat, kIf] using hTakeB
      have hDropEq :
          (fun X => ((U ∘ₘ V) X).drop (kIf X)) = R ∘ₘ V := by
        calc
          (fun X => ((U ∘ₘ V) X).drop (kIf X))
              = (fun X => (U X).drop (kIf X)) ∘ₘ V :=
                drop_concat_prefix_eq
                  (L := L) (C := C) (F := F) (Payload := Payload) U V kIf hkU
          _ = R ∘ₘ V := by rfl
      rw [← hDropEq]
      refine suffix_msc_of_safe_prefix
        (L := L) (C := C) (F := F) (Payload := Payload)
        (M := U ∘ₘ V) (k := kIf) ?hkFalse ?hPrefixLeFalse ?hSendSurplusDeadFalse hMSC
      · intro X
        have h := hkU X
        simp [WordTuple.concat]
        omega
      · intro A Y
        by_cases hAB : A = B
        · subst A
          by_cases hYB : Y = B
          · subst Y
            have hRecvZero :
                countRecvs B (((U ∘ₘ V) B).take (kIf B)) = 0 := by
              rw [hTakeBM]
              have hsZero :
                  countRecvs B sB = 0 :=
                controlBroadcast_prefix_recv_count_zero
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  B B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  true sB hSBPref
              simpa [countRecvs, AlphabetOf.mkIfTrue, Letter.isRecvFrom, hsZero]
            omega
          · by_cases hRecipY :
                ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse Y ∧
                  U Y ≠ []
            · rcases if_true_recipient_prefix_cases
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  c B Y PTrue PFalse U V hYB hRecipY.1 sB rB hUBeq hSBPref
                  hSBTrace
                  (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
              · exact False.elim (hRecipY.2 hYNil)
              · have hRecvOne :
                    countRecvs B (((U ∘ₘ V) Y).take (kIf Y)) = 1 := by
                  have hTakeY :
                      ((U ∘ₘ V) Y).take (kIf Y) =
                        [AlphabetOf.mkRecv (C := C) (F := F) Y
                          (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B] := by
                    simp [WordTuple.concat, kIf, hYB, hRecipY, hYWord]
                  rw [hTakeY]
                  simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
                have hSendPos :
                    1 ≤ countSends Y sB :=
                  if_true_prefix_send_count_pos
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B Y PTrue PFalse U V hYB hRecipY.1 sB rB uY hUBeq hSBPref
                    hSBTrace hYWord hMSC
                have hSendEq :
                    countSends Y (((U ∘ₘ V) B).take (kIf B)) = countSends Y sB := by
                  rw [hTakeBM]
                  simp [countSends, AlphabetOf.mkIfTrue, Letter.isSendTo]
                omega
            · have hRecvZero :
                  countRecvs B (((U ∘ₘ V) Y).take (kIf Y)) = 0 := by
                simp [WordTuple.concat, kIf, hYB, hRecipY]
              omega
        · have hRecvZero :
              countRecvs A (((U ∘ₘ V) Y).take (kIf Y)) = 0 := by
            simpa [WordTuple.concat, kIf] using
              if_true_kf_recv_count_zero
                (L := L) (C := C) (F := F) (Payload := Payload)
                c B A Y PTrue PFalse U V sB rB hAB hUBeq hSBPref hSBTrace hPref hSide hMSC
          omega
      · intro A Y hSurplus
        by_cases hAB : A = B
        · subst A
          by_cases hYB : Y = B
          · subst Y
            exfalso
            have hSendZero :
                countSends B (((U ∘ₘ V) B).take (kIf B)) = 0 := by
              rw [hTakeBM]
              have hsZero :
                  countSends B sB = 0 :=
                controlBroadcast_prefix_send_count_zero_nonRecipient
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  B B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  true sB hSBPref
                  (by
                    intro hSelf
                    exact (ifRecipients_no_self
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      B PTrue PFalse B hSelf) rfl)
              simpa [countSends, AlphabetOf.mkIfTrue, Letter.isSendTo, hsZero]
            omega
          · by_cases hRecipY :
                ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse Y
            · by_cases hUY : U Y = []
              · have hVY : V Y = [] :=
                  if_recipient_nil_tail_nil
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B Y PTrue PFalse U V hYB hRecipY hSide hUY
                simp [WordTuple.concat, kIf, hYB, hRecipY, hUY, hVY, countRecvs]
              · exfalso
                have hRecipY' :
                    ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse Y ∧
                      U Y ≠ [] := ⟨hRecipY, hUY⟩
                rcases if_true_recipient_prefix_cases
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B Y PTrue PFalse U V hYB hRecipY sB rB hUBeq hSBPref
                    hSBTrace
                    (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
                · exact hUY hYNil
                · have hRecvOne :
                      countRecvs B (((U ∘ₘ V) Y).take (kIf Y)) = 1 := by
                    have hTakeY :
                        ((U ∘ₘ V) Y).take (kIf Y) =
                          [AlphabetOf.mkRecv (C := C) (F := F) Y
                            (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B] := by
                      simp [WordTuple.concat, kIf, hYB, hRecipY', hYWord]
                    rw [hTakeY]
                    simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
                  have hSendLe :
                      countSends Y sB ≤ 1 :=
                    controlBroadcast_prefix_send_count_le_one_recipient
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      B Y
                      (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                      true sB hSBPref (by intro Z hZ; exact hZ.1) hRecipY
                  have hSendEq :
                      countSends Y (((U ∘ₘ V) B).take (kIf B)) = countSends Y sB := by
                    rw [hTakeBM]
                    simp [countSends, AlphabetOf.mkIfTrue, Letter.isSendTo]
                  omega
            · exfalso
              have hSendZero :
                  countSends Y (((U ∘ₘ V) B).take (kIf B)) = 0 := by
                rw [hTakeBM]
                have hsZero :
                    countSends Y sB = 0 :=
                  controlBroadcast_prefix_send_count_zero_nonRecipient
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B Y
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                    true sB hSBPref hRecipY
                simpa [countSends, AlphabetOf.mkIfTrue, Letter.isSendTo, hsZero]
              omega
        · exfalso
          have hSendZero :
              countSends Y (((U ∘ₘ V) A).take (kIf A)) = 0 := by
            by_cases hRecipA :
                ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse A ∧
                  U A ≠ []
            · rcases if_true_recipient_prefix_cases
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  c B A PTrue PFalse U V hAB hRecipA.1 sB rB hUBeq hSBPref
                  hSBTrace
                  (hPref A) hMSC with hANil | ⟨uA, _huA, hAWord⟩
              · exact False.elim (hRecipA.2 hANil)
              · have hTakeA :
                    ((U ∘ₘ V) A).take (kIf A) =
                      [AlphabetOf.mkRecv (C := C) (F := F) A
                        (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B] := by
                  simp [WordTuple.concat, kIf, hAB, hRecipA, hAWord]
                rw [hTakeA]
                simp [countSends, AlphabetOf.mkRecv, Letter.isSendTo]
            · simp [WordTuple.concat, kIf, hAB, hRecipA]
          omega
    -- Apply hTrue
    rcases hTrue hWFTrue R V hRPref hRSide hRMSC with ⟨Rbar, hZipTrue⟩
    have hSuffixTrue : IsMSC V := zipPost_suffix_isMSC hZipTrue
    rcases hZipTrue with ⟨hLocTrue, hCompleteTrue, _hOrigTrue, _hMSCTrue⟩
    -- Build Ubar = mscIfTrue ∘ₘ controlBroadcastMSC ∘ₘ Rbar
    let Ubar :=
      mscIfTrue (C := C) (F := F) (Payload := Payload) c B
        ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              true
              (by intro X hX; exact hX.1)
        ∘ₘ Rbar
    -- Trace condition for Ubar
    have hTraceUbar :
        ∀ X,
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
            (Ubar X) := by
      have hDistTrue :
          distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PTrue) Rbar :=
        ⟨(fun X => (hLocTrue X).2.1), hCompleteTrue⟩
      intro X
      simpa [Ubar] using
        (distSemantics_project_if_true_local
          (L := L) (C := C) (F := F) (Payload := Payload)
          c B PTrue PFalse Rbar hDistTrue).1 X
    have hCompleteUbar : IsCompleteMSC Ubar := by
      have hDistTrue :
          distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PTrue) Rbar :=
        ⟨(fun X => (hLocTrue X).2.1), hCompleteTrue⟩
      simpa [Ubar] using
        (distSemantics_project_if_true_local
          (L := L) (C := C) (F := F) (Payload := Payload)
          c B PTrue PFalse Rbar hDistTrue).2
    refine ⟨Ubar, ?_⟩
    refine ⟨?_, hCompleteUbar, hMSC, concat_complete_msc _ _ hCompleteUbar hSuffixTrue⟩
    intro X
    have hPrefixX :
        IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) (U X) (Ubar X) := by
      by_cases hXB : X = B
      · subst X
        rcases hLocTrue B with ⟨hRPrefB, _hTraceB, _hEqB⟩
        rcases hRPrefB with ⟨tR, htR⟩
        rcases localPrefixSemantics_controlBroadcast_prefix
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            true
            sB hSBPref with ⟨tS, htS⟩
        by_cases hrB : rB = []
        · subst hrB
          refine ⟨tS ++ tR, ?_⟩
          simp [Ubar, WordTuple.concat, mscIfTrue, hUBeq, hRB, htS, htR, List.append_assoc,
            choiceIfTrue,
            controlBroadcastMSC_decider
              (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              true
              (by intro Y hY; exact hY.1)]
        · have hSBFull' :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  true)
                sB := hSBFull hrB
          have hSWord :
              sB =
                controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  true := by
            exact localTraceSemantics_controlBroadcast_eq
              (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              true
              sB hSBFull'
          have hWordEq :
              controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                true = sB := hSWord.symm
          refine ⟨tR, ?_⟩
          simp [Ubar, WordTuple.concat, mscIfTrue, hUBeq, hRB, hWordEq, htR, List.append_assoc,
            choiceIfTrue,
            controlBroadcastMSC_decider
              (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              true
              (by intro Y hY; exact hY.1)]
      · by_cases hRecip :
            ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X
        · rcases if_true_recipient_prefix_cases
              (L := L) (C := C) (F := F) (Payload := Payload)
              c B X PTrue PFalse U V hXB hRecip sB rB hUBeq hSBPref
              hSBTrace
              (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
          · refine ⟨Ubar X, by simp [hXNil, IsPrefixWord]⟩
          · rcases hLocTrue X with ⟨hRPrefX, _hTraceX, _hEqX⟩
            rcases hRPrefX with ⟨tR, htR⟩
            have hRX : R X = uX := by
              simp [R, kIf, hXB, hRecip, hXWord]
            refine ⟨tR, ?_⟩
            simp [Ubar, WordTuple.concat, mscIfTrue, hXB, hRecip, hXWord, htR, hRX, List.append_assoc,
              controlDecisionPayload,
              controlBroadcastMSC_recipient
                (C := C) (F := F) (Payload := Payload)
                B X
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                true
                (by intro Y hY; exact hY.1)
                hRecip]
        · have hXNil : U X = [] :=
            localPrefixSemantics_eps_eq_nil
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := X) (U X)
              (by simpa [project, hXB, hRecip] using hPref X)
          refine ⟨Ubar X, by simp [hXNil, IsPrefixWord]⟩
    refine ⟨hPrefixX, hTraceUbar X, ?_⟩
    intro hVX
    exact localTraceSemantics_prefix_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := X)
      (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
      (u := U X)
      (v := Ubar X)
      (hSide X hVX)
      (hTraceUbar X)
      hPrefixX |>.symm
  -- False branch (seqPref and hUB already destructured from rcases above)
  · rcases localPrefixSemantics_seq_split
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse) false)
        (project (L := L) (C := C) (F := F) (Payload := Payload) B PFalse)
        seqPref hSeqPref with ⟨sB, rB, hSeqEq, hSBPref, hRBPref, hSBFull⟩
    have hUBeq : U B = AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB := by
      rw [hUB, hSeqEq, ← List.cons_append]
    have hSBTrace : rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false)
          sB := by
      intro hTail
      by_cases hrB : rB = []
      · have hVB : V B ≠ [] := by
          simpa [hrB] using hTail
        have hBTrace := hSide B hVB
        have hSeqTrace :
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PFalse)
              (sB ++ rB) := by
          simpa [project, hUBeq, localTraceSemantics,
            AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse] using hBTrace
        rw [hrB, List.append_nil] at hSeqTrace
        rcases hSeqTrace with ⟨u1, u2, hu1, hu2, hsEq⟩
        have hWord :
            u1 =
              controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                false := by
          exact localTraceSemantics_controlBroadcast_eq
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false
            u1 hu1
        rcases localPrefixSemantics_controlBroadcast_prefix
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false
            sB hSBPref with ⟨t, ht⟩
        rw [hWord, ht] at hsEq
        have hNil : [] = t ++ u2 := by
          have hEq' : sB ++ [] = sB ++ (t ++ u2) := by
            simpa [List.append_assoc] using hsEq
          exact List.append_inj_right hEq' rfl
        have htNil : t = [] := by
          cases t <;> simp at hNil <;> simp
        rw [htNil] at ht
        simpa [hWord, ht] using hu1
      · exact hSBFull hrB
    let kIf : L → Nat := fun X =>
      if X = B then 1 + sB.length
      else if ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X ∧ U X ≠ [] then 1
      else 0
    let R : WordTuple L C F Payload := fun X => (U X).drop (kIf X)
    have hRB : R B = rB := by
      simp only [R, kIf, ite_true]
      rw [hUBeq, show 1 + sB.length = sB.length + 1 from by omega]
      simp [List.drop_succ_cons]
    have hRPref : ∀ X,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
          (R X) := by
      intro X
      by_cases hXB : X = B
      · subst X
        rw [hRB]
        exact hRBPref
      · by_cases hRecip :
            ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X
        · rcases if_false_recipient_prefix_cases
              (L := L) (C := C) (F := F) (Payload := Payload)
              c B X PTrue PFalse U V hXB hRecip sB rB hUBeq hSBPref
              hSBTrace
              (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
          · simp only [R, kIf, hXB, ite_false, hRecip, hXNil, ite_true]
            simp
            exact localPrefixSemantics_project_nil
              (L := L) (C := C) (F := F) (Payload := Payload) X PFalse
          · simp only [R, kIf, hXB, ite_false, hRecip, hXWord, ite_true]
            simp
            exact huX
        · have hXNil : U X = [] :=
            localPrefixSemantics_eps_eq_nil
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := X) (U X)
              (by simpa [project, hXB, hRecip] using hPref X)
          simp only [R, kIf, hXB, ite_false, hRecip, hXNil, ite_false]
          simp
          exact localPrefixSemantics_project_nil
            (L := L) (C := C) (F := F) (Payload := Payload) X PFalse
    have hRSide : ∀ X, V X ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
          (R X) := by
      intro X hVX
      by_cases hXB : X = B
      · subst X
        rw [hRB]
        have hBTrace := hSide B hVX
        have hSeqTrace :
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PFalse)
              (sB ++ rB) := by
          simpa [project, hUBeq, localTraceSemantics,
            AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse] using hBTrace
        have hSWord :
            sB =
              controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                false := by
          exact localTraceSemantics_controlBroadcast_eq
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false
            sB
            (hSBTrace (by
              intro hNil
              exact hVX (List.eq_nil_of_append_eq_nil hNil).2))
        rcases hSeqTrace with ⟨u1, u2, hu1, hu2, hEq⟩
        have hWord :
            u1 =
              controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                false := by
          exact localTraceSemantics_controlBroadcast_eq
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false
            u1 hu1
        rw [hWord, hSWord] at hEq
        have hRBU2 : rB = u2 := by
          exact List.append_inj_right hEq rfl
        simpa [hRBU2] using hu2
      · by_cases hRecip :
            ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X
        · rcases if_false_recipient_prefix_cases
              (L := L) (C := C) (F := F) (Payload := Payload)
              c B X PTrue PFalse U V hXB hRecip sB rB hUBeq hSBPref
              hSBTrace
              (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
          · exfalso
            exact hVX (if_recipient_nil_tail_nil
              (L := L) (C := C) (F := F) (Payload := Payload)
              c B X PTrue PFalse U V hXB hRecip hSide hXNil)
          · simp only [R, kIf, hXB, ite_false, hRecip, hXWord, ite_true]
            simp
            have hXTrace := hSide X hVX
            have hXTrace' :
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
                  uX := by
              have hXCases :
                  (localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X PTrue)
                    uX ∧
                    AlphabetOf.mkRecv (C := C) (F := F) X
                      (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B =
                    AlphabetOf.mkRecv (C := C) (F := F) X
                      (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B) ∨
                  localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
                    uX := by
                simpa [project, hXB, hRecip, hXWord, localTraceSemantics] using hXTrace
              rcases hXCases with ⟨_hTrue, hContra⟩ | hFalse
              · have hPayloadEq :
                    ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern =
                      ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern := by
                  simpa [AlphabetOf.mkRecv] using hContra
                exact False.elim (ctrlFalse_ne_ctrlTrue hPayloadEq)
              · exact hFalse
            exact hXTrace'
        · have hXNil : U X = [] :=
            localPrefixSemantics_eps_eq_nil
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := X) (U X)
              (by simpa [project, hXB, hRecip] using hPref X)
          have hNotPartFalse : ¬ participationSet PFalse X := by
            intro hPartFalse
            exact hRecip ⟨hXB, Or.inr hPartFalse⟩
          have hNilTrace :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (project (L := L) (C := C) (F := F) (Payload := Payload) X PFalse)
                [] := by
            rcases exists_project_trace
                (L := L) (C := C) (F := F) (Payload := Payload) X PFalse with ⟨w, hw⟩
            have hwNil :=
              project_trace_nil_of_not_participating
                (L := L) (C := C) (F := F) (Payload := Payload)
                X PFalse hNotPartFalse w hw
            rw [hwNil] at hw
            simpa using hw
          simpa [R, kIf, hXB, hRecip, hXNil] using hNilTrace
    have hRMSC : IsMSC (R ∘ₘ V) := by
      have hkU : ∀ X, kIf X ≤ (U X).length := by
        intro X
        by_cases hXB : X = B
        · subst X
          simp [kIf, hUBeq]
          omega
        · by_cases hRecipX :
              ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X ∧
                U X ≠ []
          · cases hUX : U X with
            | nil => exact False.elim (hRecipX.2 hUX)
            | cons hd tl => simp [kIf, hXB, hRecipX, hUX]
          · simp [kIf, hXB, hRecipX]
      have hTakeB :
          List.take (1 + sB.length) (U B ++ V B) =
            AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB := by
        rw [hUBeq]
        rw [List.append_assoc]
        have hLen' :
            1 + sB.length ≤
              (AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB).length := by
          simpa [Nat.add_comm] using Nat.le_refl (sB.length + 1)
        have hTake' :
            List.take (1 + sB.length)
              ((AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB) ++
                (rB ++ V B)) =
              List.take (1 + sB.length)
                (AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB) := by
          simpa using
            (List.take_append_of_le_length
              (l₁ := AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB)
              (l₂ := rB ++ V B)
              (i := 1 + sB.length)
              hLen')
        have hTake'' :
            List.take (1 + sB.length)
              (AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB) =
            AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB := by
          simpa [Nat.add_comm] using
            (List.take_length
              (l := AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB))
        exact hTake'.trans hTake''
      have hTakeBM :
          ((U ∘ₘ V) B).take (kIf B) =
            AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) B c :: sB := by
        simpa [WordTuple.concat, kIf] using hTakeB
      have hDropEq :
          (fun X => ((U ∘ₘ V) X).drop (kIf X)) = R ∘ₘ V := by
        calc
          (fun X => ((U ∘ₘ V) X).drop (kIf X))
              = (fun X => (U X).drop (kIf X)) ∘ₘ V :=
                drop_concat_prefix_eq
                  (L := L) (C := C) (F := F) (Payload := Payload) U V kIf hkU
          _ = R ∘ₘ V := by rfl
      rw [← hDropEq]
      refine suffix_msc_of_safe_prefix
        (L := L) (C := C) (F := F) (Payload := Payload)
        (M := U ∘ₘ V) (k := kIf) ?hk ?hPrefixLe ?hSendSurplusDead hMSC
      · intro X
        have h := hkU X
        simp [WordTuple.concat]
        omega
      · intro A Y
        by_cases hAB : A = B
        · subst A
          by_cases hYB : Y = B
          · subst Y
            have hRecvZero :
                countRecvs B (((U ∘ₘ V) B).take (kIf B)) = 0 := by
              rw [hTakeBM]
              have hsZero :
                  countRecvs B sB = 0 :=
                controlBroadcast_prefix_recv_count_zero
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  B B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false sB hSBPref
              simpa [countRecvs, AlphabetOf.mkIfFalse, Letter.isRecvFrom, hsZero]
            omega
          · by_cases hRecipY :
                ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse Y ∧
                  U Y ≠ []
            · rcases if_false_recipient_prefix_cases
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  c B Y PTrue PFalse U V hYB hRecipY.1 sB rB hUBeq hSBPref
                  hSBTrace
                  (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
              · exact False.elim (hRecipY.2 hYNil)
              · have hRecvOne :
                    countRecvs B (((U ∘ₘ V) Y).take (kIf Y)) = 1 := by
                  have hTakeY :
                      ((U ∘ₘ V) Y).take (kIf Y) =
                        [AlphabetOf.mkRecv (C := C) (F := F) Y
                          (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B] := by
                    simp [WordTuple.concat, kIf, hYB, hRecipY, hYWord]
                  rw [hTakeY]
                  simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
                have hSendPos :
                    1 ≤ countSends Y sB :=
                  if_false_prefix_send_count_pos
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B Y PTrue PFalse U V hYB hRecipY.1 sB rB uY hUBeq hSBPref
                    hSBTrace hYWord hMSC
                have hSendEq :
                    countSends Y (((U ∘ₘ V) B).take (kIf B)) = countSends Y sB := by
                  rw [hTakeBM]
                  simp [countSends, AlphabetOf.mkIfFalse, Letter.isSendTo]
                omega
            · have hRecvZero :
                  countRecvs B (((U ∘ₘ V) Y).take (kIf Y)) = 0 := by
                simp [WordTuple.concat, kIf, hYB, hRecipY]
              omega
        · have hRecvZero :
              countRecvs A (((U ∘ₘ V) Y).take (kIf Y)) = 0 := by
            have hBA : B ≠ A := by
              intro hBA
              exact hAB hBA.symm
            by_cases hYB : Y = B
            · subst Y
              rw [hTakeBM]
              have hsZero :
                  countRecvs A sB = 0 :=
                controlBroadcast_prefix_recv_count_zero
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  B A
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false sB hSBPref
              simpa [countRecvs, AlphabetOf.mkIfFalse, Letter.isRecvFrom, hsZero, hBA]
            · by_cases hRecipY :
                  ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse Y ∧
                    U Y ≠ []
              · rcases if_false_recipient_prefix_cases
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B Y PTrue PFalse U V hYB hRecipY.1 sB rB hUBeq hSBPref
                    hSBTrace
                    (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
                · exact False.elim (hRecipY.2 hYNil)
                · have hTakeY :
                      ((U ∘ₘ V) Y).take (kIf Y) =
                        [AlphabetOf.mkRecv (C := C) (F := F) Y
                          (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B] := by
                    simp [WordTuple.concat, kIf, hYB, hRecipY, hYWord]
                  rw [hTakeY]
                  simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom, hBA]
              · simp [WordTuple.concat, kIf, hYB, hRecipY]
          omega
      · intro A Y hSurplus
        by_cases hAB : A = B
        · subst A
          by_cases hYB : Y = B
          · subst Y
            exfalso
            have hSendZero :
                countSends B (((U ∘ₘ V) B).take (kIf B)) = 0 := by
              rw [hTakeBM]
              have hsZero :
                  countSends B sB = 0 :=
                controlBroadcast_prefix_send_count_zero_nonRecipient
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  B B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false sB hSBPref
                  (by
                    intro hSelf
                    exact (ifRecipients_no_self
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      B PTrue PFalse B hSelf) rfl)
              simpa [countSends, AlphabetOf.mkIfFalse, Letter.isSendTo, hsZero]
            omega
          · by_cases hRecipY :
                ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse Y
            · by_cases hUY : U Y = []
              · have hVY : V Y = [] :=
                  if_recipient_nil_tail_nil
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B Y PTrue PFalse U V hYB hRecipY hSide hUY
                simp [WordTuple.concat, kIf, hYB, hRecipY, hUY, hVY, countRecvs]
              · exfalso
                have hRecipY' :
                    ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse Y ∧
                      U Y ≠ [] := ⟨hRecipY, hUY⟩
                rcases if_false_recipient_prefix_cases
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B Y PTrue PFalse U V hYB hRecipY sB rB hUBeq hSBPref
                    hSBTrace
                    (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
                · exact hUY hYNil
                · have hRecvOne :
                      countRecvs B (((U ∘ₘ V) Y).take (kIf Y)) = 1 := by
                    have hTakeY :
                        ((U ∘ₘ V) Y).take (kIf Y) =
                          [AlphabetOf.mkRecv (C := C) (F := F) Y
                            (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B] := by
                      simp [WordTuple.concat, kIf, hYB, hRecipY', hYWord]
                    rw [hTakeY]
                    simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
                  have hSendLe :
                      countSends Y sB ≤ 1 :=
                    controlBroadcast_prefix_send_count_le_one_recipient
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      B Y
                      (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                      false sB hSBPref (by intro Z hZ; exact hZ.1) hRecipY
                  have hSendEq :
                      countSends Y (((U ∘ₘ V) B).take (kIf B)) = countSends Y sB := by
                    rw [hTakeBM]
                    simp [countSends, AlphabetOf.mkIfFalse, Letter.isSendTo]
                  omega
            · exfalso
              have hSendZero :
                  countSends Y (((U ∘ₘ V) B).take (kIf B)) = 0 := by
                rw [hTakeBM]
                have hsZero :
                    countSends Y sB = 0 :=
                  controlBroadcast_prefix_send_count_zero_nonRecipient
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B Y
                    (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                    false sB hSBPref hRecipY
                simpa [countSends, AlphabetOf.mkIfFalse, Letter.isSendTo, hsZero]
              omega
        · exfalso
          have hSendZero :
              countSends Y (((U ∘ₘ V) A).take (kIf A)) = 0 := by
            by_cases hRecipA :
                ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse A ∧
                  U A ≠ []
            · rcases if_false_recipient_prefix_cases
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  c B A PTrue PFalse U V hAB hRecipA.1 sB rB hUBeq hSBPref
                  hSBTrace
                  (hPref A) hMSC with hANil | ⟨uA, _huA, hAWord⟩
              · exact False.elim (hRecipA.2 hANil)
              · have hTakeA :
                    ((U ∘ₘ V) A).take (kIf A) =
                      [AlphabetOf.mkRecv (C := C) (F := F) A
                        (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B] := by
                  simp [WordTuple.concat, kIf, hAB, hRecipA, hAWord]
                rw [hTakeA]
                simp [countSends, AlphabetOf.mkRecv, Letter.isSendTo]
            · simp [WordTuple.concat, kIf, hAB, hRecipA]
          omega
    rcases hFalse hWFFalse R V hRPref hRSide hRMSC with ⟨Rbar, hZipFalse⟩
    have hSuffixFalse : IsMSC V := zipPost_suffix_isMSC hZipFalse
    rcases hZipFalse with ⟨hLocFalse, hCompleteFalse, _hOrigFalse, _hMSCFalse⟩
    let Ubar :=
      mscIfFalse (C := C) (F := F) (Payload := Payload) c B
        ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              false
              (by intro X hX; exact hX.1)
        ∘ₘ Rbar
    have hTraceUbar :
        ∀ X,
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
            (Ubar X) := by
      have hDistFalse :
          distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PFalse) Rbar :=
        ⟨(fun X => (hLocFalse X).2.1), hCompleteFalse⟩
      intro X
      simpa [Ubar] using
        (distSemantics_project_if_false_local
          (L := L) (C := C) (F := F) (Payload := Payload)
          c B PTrue PFalse Rbar hDistFalse).1 X
    have hCompleteUbar : IsCompleteMSC Ubar := by
      have hDistFalse :
          distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PFalse) Rbar :=
        ⟨(fun X => (hLocFalse X).2.1), hCompleteFalse⟩
      simpa [Ubar] using
        (distSemantics_project_if_false_local
          (L := L) (C := C) (F := F) (Payload := Payload)
          c B PTrue PFalse Rbar hDistFalse).2
    refine ⟨Ubar, ?_⟩
    refine ⟨?_, hCompleteUbar, hMSC, concat_complete_msc _ _ hCompleteUbar hSuffixFalse⟩
    intro X
    have hPrefixX :
        IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) (U X) (Ubar X) := by
      by_cases hXB : X = B
      · subst X
        rcases hLocFalse B with ⟨hRPrefB, _hTraceB, _hEqB⟩
        rcases hRPrefB with ⟨tR, htR⟩
        rcases localPrefixSemantics_controlBroadcast_prefix
            (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
            false
            sB hSBPref with ⟨tS, htS⟩
        by_cases hrB : rB = []
        · subst hrB
          refine ⟨tS ++ tR, ?_⟩
          simp [Ubar, WordTuple.concat, mscIfFalse, hUBeq, hRB, htS, htR, List.append_assoc,
            choiceIfFalse,
            controlBroadcastMSC_decider
              (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              false
              (by intro Y hY; exact hY.1)]
        · have hSBFull' :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false)
                sB := hSBFull hrB
          have hSWord :
              sB =
                controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                  false := by
            exact localTraceSemantics_controlBroadcast_eq
              (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              false
              sB hSBFull'
          have hWordEq :
              controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                false = sB := hSWord.symm
          refine ⟨tR, ?_⟩
          simp [Ubar, WordTuple.concat, mscIfFalse, hUBeq, hRB, hWordEq, htR, List.append_assoc,
            choiceIfFalse,
            controlBroadcastMSC_decider
              (C := C) (F := F) (Payload := Payload)
              B
              (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
              false
              (by intro Y hY; exact hY.1)]
      · by_cases hRecip :
            ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse X
        · rcases if_false_recipient_prefix_cases
              (L := L) (C := C) (F := F) (Payload := Payload)
              c B X PTrue PFalse U V hXB hRecip sB rB hUBeq hSBPref
              hSBTrace
              (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
          · refine ⟨Ubar X, by simp [hXNil, IsPrefixWord]⟩
          · rcases hLocFalse X with ⟨hRPrefX, _hTraceX, _hEqX⟩
            rcases hRPrefX with ⟨tR, htR⟩
            have hRX : R X = uX := by
              simp [R, kIf, hXB, hRecip, hXWord]
            refine ⟨tR, ?_⟩
            simp [Ubar, WordTuple.concat, mscIfFalse, hXB, hRecip, hXWord, htR, hRX, List.append_assoc,
              controlDecisionPayload,
              controlBroadcastMSC_recipient
                (C := C) (F := F) (Payload := Payload)
                B X
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse)
                false
                (by intro Y hY; exact hY.1)
                hRecip]
        · have hXNil : U X = [] :=
            localPrefixSemantics_eps_eq_nil
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := X) (U X)
              (by simpa [project, hXB, hRecip] using hPref X)
          refine ⟨Ubar X, by simp [hXNil, IsPrefixWord]⟩
    refine ⟨hPrefixX, hTraceUbar X, ?_⟩
    intro hVX
    exact localTraceSemantics_prefix_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := X)
      (project (L := L) (C := C) (F := F) (Payload := Payload) X (.ite c B PTrue PFalse))
      (u := U X)
      (v := Ubar X)
      (hSide X hVX)
      (hTraceUbar X)
      hPrefixX |>.symm

private theorem while_recipient_nil_tail_nil
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (hSide : ∀ Y, V Y ≠ [] →
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) Y (.whileLoop c B PBody PExit))
        (U Y))
    (hXNil : U X = []) :
    V X = [] := by
  by_cases hVX : V X = []
  · exact hVX
  · have hPartX : participationSet (.whileLoop c B PBody PExit) X := by
      simp [participationSet_while, hXB, hRecip.2]
    have hXTrace := hSide X hVX
    have hXNonempty :=
      project_trace_ne_nil_of_while_participating
        (L := L) (C := C) (F := F) (Payload := Payload)
        X c B PBody PExit (U X) hPartX hXTrace
    rw [hXNil] at hXNonempty
    simp at hXNonempty

private theorem localTraceSemantics_seq_assoc_left {A : L}
    (S1 S2 S3 : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        ((S1 ;;ₗ S2) ;;ₗ S3) w) :
    localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (S1 ;;ₗ (S2 ;;ₗ S3)) w := by
  rcases h with ⟨u12, u3, h12, h3, rfl⟩
  rcases h12 with ⟨u1, u2, h1, h2, rfl⟩
  refine ⟨u1, u2 ++ u3, h1, ?_, ?_⟩
  · exact localTraceSemantics_seq_intro S2 S3 u2 u3 h2 h3
  · simp [List.append_assoc]

private theorem localPrefixSemantics_seq_assoc_left {A : L}
    (S1 S2 S3 : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        ((S1 ;;ₗ S2) ;;ₗ S3) w) :
    localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (S1 ;;ₗ (S2 ;;ₗ S3)) w := by
  rcases h with ⟨w', hw', hpref⟩
  exact ⟨w', localTraceSemantics_seq_assoc_left S1 S2 S3 w' hw', hpref⟩

private theorem while_exit_decider_sendPayloads_head
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (hUB :
      U B =
        AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          false)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            false)
          sB)
    (hXVisible :
      ∃ p uX,
        U X =
          AlphabetOf.mkRecv (C := C) (F := F) X p B :: uX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    ∃ ss,
      sendPayloads (C := C) (F := F) (Payload := Payload) X ((U ∘ₘ V) B) =
        ControlPayload.setDecision false ControlPayload.ctrlPattern :: ss := by
  have hSendPref :
      IsPrefixWord
        (L := L) (C := C) (F := F) (Payload := Payload)
        sB
        (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          false) :=
    localPrefixSemantics_controlBroadcast_prefix
      (L := L) (C := C) (F := F) (Payload := Payload)
      B
      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
      false
      sB
      hSPref
  by_cases hCont : rB ++ V B = []
  · have hVB : V B = [] := (List.eq_nil_of_append_eq_nil hCont).2
    have hrB : rB = [] := (List.eq_nil_of_append_eq_nil hCont).1
    rcases hXVisible with ⟨p, uX, hXVisible⟩
    have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
      rw [rcvCount, WordTuple.concat, hXVisible]
      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
    have hSendPos : 1 ≤ sndCount (U ∘ₘ V) B X := by
      have hNoUnmatched := hMSC.noUnmatchedRecv B X
      omega
    have hSendPos' : 1 ≤ countSends X sB := by
      simpa [sndCount, WordTuple.concat, hUB, hVB, hrB, countSends,
        AlphabetOf.mkWhileFalse, Letter.isSendTo] using hSendPos
    have hNonempty : sendPayloads (C := C) (F := F) (Payload := Payload) X sB ≠ [] := by
      intro hNil
      have hLen :
          (sendPayloads (C := C) (F := F) (Payload := Payload) X sB).length = 0 := by
        simp [hNil]
      rw [sendPayloads_length] at hLen
      omega
    have hPrefPayloads :
        IsPrefixList
          (sendPayloads (C := C) (F := F) (Payload := Payload) X sB)
          [ControlPayload.setDecision false ControlPayload.ctrlPattern] := by
      simpa [sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        false (by intro Y hY; exact hY.1) hRecip] using
          (sendPayloads_prefix
            (L := L) (C := C) (F := F) (Payload := Payload) X hSendPref)
    have hPayloads :
        sendPayloads (C := C) (F := F) (Payload := Payload) X sB =
          [ControlPayload.setDecision false ControlPayload.ctrlPattern] :=
      prefix_of_singleton_nonempty hPrefPayloads hNonempty
    refine ⟨[], ?_⟩
    simp [WordTuple.concat, hUB, hPayloads, hrB, hVB, sendPayloads,
      AlphabetOf.mkWhileFalse]
  · have hSWord :
      sB =
        controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          false := by
      exact localTraceSemantics_controlBroadcast_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        false
        sB
        (hSFull hCont)
    refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) X rB ++
      sendPayloads (C := C) (F := F) (Payload := Payload) X (V B), ?_⟩
    simp [WordTuple.concat, hUB, hSWord, sendPayloads_append,
      sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        false (by intro Y hY; exact hY.1) hRecip,
      List.append_assoc, sendPayloads, AlphabetOf.mkWhileFalse]

private theorem while_true_decider_sendPayloads_head
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (hUB :
      U B =
        AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          true)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            true)
          sB)
    (hXVisible :
      ∃ p uX,
        U X =
          AlphabetOf.mkRecv (C := C) (F := F) X p B :: uX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    ∃ ss,
      sendPayloads (C := C) (F := F) (Payload := Payload) X ((U ∘ₘ V) B) =
        ControlPayload.setDecision true ControlPayload.ctrlPattern :: ss := by
  have hSendPref :
      IsPrefixWord
        (L := L) (C := C) (F := F) (Payload := Payload)
        sB
        (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          true) :=
    localPrefixSemantics_controlBroadcast_prefix
      (L := L) (C := C) (F := F) (Payload := Payload)
      B
      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
      true
      sB
      hSPref
  by_cases hCont : rB ++ V B = []
  · have hVB : V B = [] := (List.eq_nil_of_append_eq_nil hCont).2
    have hrB : rB = [] := (List.eq_nil_of_append_eq_nil hCont).1
    rcases hXVisible with ⟨p, uX, hXVisible⟩
    have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
      rw [rcvCount, WordTuple.concat, hXVisible]
      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
    have hSendPos : 1 ≤ sndCount (U ∘ₘ V) B X := by
      have hNoUnmatched := hMSC.noUnmatchedRecv B X
      omega
    have hSendPos' : 1 ≤ countSends X sB := by
      simpa [sndCount, WordTuple.concat, hUB, hVB, hrB, countSends,
        AlphabetOf.mkWhileTrue, Letter.isSendTo] using hSendPos
    have hNonempty : sendPayloads (C := C) (F := F) (Payload := Payload) X sB ≠ [] := by
      intro hNil
      have hLen :
          (sendPayloads (C := C) (F := F) (Payload := Payload) X sB).length = 0 := by
        simp [hNil]
      rw [sendPayloads_length] at hLen
      omega
    have hPrefPayloads :
        IsPrefixList
          (sendPayloads (C := C) (F := F) (Payload := Payload) X sB)
          [ControlPayload.setDecision true ControlPayload.ctrlPattern] := by
      simpa [sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        true (by intro Y hY; exact hY.1) hRecip] using
          (sendPayloads_prefix
            (L := L) (C := C) (F := F) (Payload := Payload) X hSendPref)
    have hPayloads :
        sendPayloads (C := C) (F := F) (Payload := Payload) X sB =
          [ControlPayload.setDecision true ControlPayload.ctrlPattern] :=
      prefix_of_singleton_nonempty hPrefPayloads hNonempty
    refine ⟨[], ?_⟩
    simp [WordTuple.concat, hUB, hPayloads, hrB, hVB, sendPayloads,
      AlphabetOf.mkWhileTrue]
  · have hSWord :
      sB =
        controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          true := by
      exact localTraceSemantics_controlBroadcast_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        true
        sB
        (hSFull hCont)
    refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) X rB ++
      sendPayloads (C := C) (F := F) (Payload := Payload) X (V B), ?_⟩
    simp [WordTuple.concat, hUB, hSWord, sendPayloads_append,
      sendPayloads_controlBroadcastWord_recipient
        (L := L) (C := C) (F := F) (Payload := Payload)
        B X
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        true (by intro Y hY; exact hY.1) hRecip,
      List.append_assoc, sendPayloads, AlphabetOf.mkWhileTrue]

private theorem while_exit_prefix_send_count_pos
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hUB :
      U B =
        AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          false)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            false)
          sB)
    (hXVisible :
      U X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: uX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    1 ≤ countSends X sB := by
  by_cases hCont : rB ++ V B = []
  · have hVB : V B = [] := (List.eq_nil_of_append_eq_nil hCont).2
    have hrB : rB = [] := (List.eq_nil_of_append_eq_nil hCont).1
    have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
      rw [rcvCount, WordTuple.concat, hXVisible]
      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
    have hSendPos : 1 ≤ sndCount (U ∘ₘ V) B X := by
      have hNoUnmatched := hMSC.noUnmatchedRecv B X
      omega
    simpa [sndCount, WordTuple.concat, hUB, hVB, hrB, countSends,
      AlphabetOf.mkWhileFalse, Letter.isSendTo] using hSendPos
  · have hSWord :
      sB =
        controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          false := by
      exact localTraceSemantics_controlBroadcast_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        false
        sB
        (hSFull hCont)
    rw [hSWord]
    rw [← sendPayloads_length]
    simpa [sendPayloads_controlBroadcastWord_recipient
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X
      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
      false (by intro Y hY; exact hY.1) hRecip]

private theorem while_true_prefix_send_count_pos
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hUB :
      U B =
        AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          true)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            true)
          sB)
    (hXVisible :
      U X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: uX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    1 ≤ countSends X sB := by
  by_cases hCont : rB ++ V B = []
  · have hVB : V B = [] := (List.eq_nil_of_append_eq_nil hCont).2
    have hrB : rB = [] := (List.eq_nil_of_append_eq_nil hCont).1
    have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
      rw [rcvCount, WordTuple.concat, hXVisible]
      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
    have hSendPos : 1 ≤ sndCount (U ∘ₘ V) B X := by
      have hNoUnmatched := hMSC.noUnmatchedRecv B X
      omega
    simpa [sndCount, WordTuple.concat, hUB, hVB, hrB, countSends,
      AlphabetOf.mkWhileTrue, Letter.isSendTo] using hSendPos
  · have hSWord :
      sB =
        controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          true := by
      exact localTraceSemantics_controlBroadcast_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        B
        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
        true
        sB
        (hSFull hCont)
    rw [hSWord]
    rw [← sendPayloads_length]
    simpa [sendPayloads_controlBroadcastWord_recipient
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X
      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
      true (by intro Y hY; exact hY.1) hRecip]

private theorem while_exit_recipient_true_impossible
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (wX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hUB :
      U B =
        AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          false)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            false)
          sB)
    (hXTrue :
      U X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: wX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    False := by
  have hs :=
    while_exit_decider_sendPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      c B X PBody PExit U V hXB hRecip sB rB hUB hSPref hSFull
      ⟨_, _, hXTrue⟩
      hMSC
  have hr :=
    if_recipient_recvPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X true U V wX hXTrue
  have hEq :=
    firstControlDecisions_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      (M := U ∘ₘ V) hMSC B X false true hs hr
  cases hEq

private theorem while_true_recipient_false_impossible
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (uX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (wX : LocalWord (C := C) (F := F) (Payload := Payload) X)
    (hUB :
      U B =
        AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          true)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            true)
          sB)
    (hXFalse :
      U X =
        AlphabetOf.mkRecv (C := C) (F := F) X
          (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: wX)
    (hMSC : IsMSC (U ∘ₘ V)) :
    False := by
  have hs :=
    while_true_decider_sendPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      c B X PBody PExit U V hXB hRecip sB rB hUB hSPref hSFull
      ⟨_, _, hXFalse⟩
      hMSC
  have hr :=
    if_recipient_recvPayloads_head
      (L := L) (C := C) (F := F) (Payload := Payload)
      B X false U V wX hXFalse
  have hEq :=
    firstControlDecisions_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      (M := U ∘ₘ V) hMSC B X true false hs hr
  cases hEq

private theorem while_exit_recipient_prefix_cases
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (hUB :
      U B =
        AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          false)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            false)
          sB)
    (hPrefX :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (.whileLoop c B PBody PExit))
        (U X))
    (hMSC : IsMSC (U ∘ₘ V)) :
    U X = [] ∨
      ∃ u,
        localPrefixSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit) u ∧
        U X =
          AlphabetOf.mkRecv (C := C) (F := F) X
            (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: u := by
  have hRecvPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (LocProg.recvWhile ControlPayload.ctrlPattern B
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit))
        (U X) := by
    simpa [project, hXB, hRecip] using hPrefX
  rcases localPrefixSemantics_recvWhile_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := X)
      ControlPayload.ctrlPattern B
      (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
      (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
      (U X) hRecvPref with hCases
  rcases hCases with hNil | hExit | hBody
  · exact Or.inl hNil
  · exact Or.inr hExit
  · rcases hBody with ⟨wX, _hPrefBody, hXTrue⟩
    exact False.elim <|
      while_exit_recipient_true_impossible
        (L := L) (C := C) (F := F) (Payload := Payload)
        c B X PBody PExit U V hXB hRecip sB rB wX wX hUB hSPref hSFull hXTrue hMSC

private theorem while_body_recipient_prefix_cases
    (c : C) (B X : L) (PBody PExit : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hXB : X ≠ B)
    (hRecip :
      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X)
    (sB rB : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (hUB :
      U B =
        AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB)
    (hSPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
          B
          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
          true)
        sB)
    (hSFull :
      rB ++ V B ≠ [] →
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B
            (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
            true)
          sB)
    (hPrefX :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (.whileLoop c B PBody PExit))
        (U X))
    (hMSC : IsMSC (U ∘ₘ V)) :
    U X = [] ∨
      ∃ u,
        localPrefixSemantics
          (L := L) (C := C) (F := F) (Payload := Payload)
          ((project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) X
                (.whileLoop c B PBody PExit)) u ∧
        U X =
          AlphabetOf.mkRecv (C := C) (F := F) X
            (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: u := by
  have hRecvPref :
      localPrefixSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (LocProg.recvWhile ControlPayload.ctrlPattern B
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit))
        (U X) := by
    simpa [project, hXB, hRecip] using hPrefX
  rcases localPrefixSemantics_recvWhile_cases
      (L := L) (C := C) (F := F) (Payload := Payload)
      (A := X)
      ControlPayload.ctrlPattern B
      (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
      (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
      (U X) hRecvPref with hCases
  rcases hCases with hNil | hExit | hBody
  · exact Or.inl hNil
  · rcases hExit with ⟨wX, _hPrefExit, hXFalse⟩
    exact False.elim <|
      while_true_recipient_false_impossible
        (L := L) (C := C) (F := F) (Payload := Payload)
        c B X PBody PExit U V hXB hRecip sB rB wX wX hUB hSPref hSFull hXFalse hMSC
  · rcases hBody with ⟨u, hPrefBody, hXTrue⟩
    refine Or.inr ⟨u, ?_, hXTrue⟩
    -- hPrefBody : lps (PBody ;;ₗ recvWhile ctrlPattern B PBody PExit) u
    -- We need lps ((project X PBody) ;;ₗ project X (.whileLoop c B PBody PExit)) u
    -- Note: project X (.whileLoop c B PBody PExit) = recvWhile ctrlPattern B (project X PBody) (project X PExit)
    have hproj : project (L := L) (C := C) (F := F) (Payload := Payload) X
        (.whileLoop c B PBody PExit) =
        LocProg.recvWhile ControlPayload.ctrlPattern B
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit) := by
      simp [project, hXB, hRecip]
    rw [hproj]
    exact hPrefBody

theorem uniformZipper_while
    (c : C) (B : L) (PBody PExit : Prog L C F Payload)
    (hBody : UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) PBody)
    (hExit : UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) PExit) :
    UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) (.whileLoop c B PBody PExit) := by
  classical
  -- We prove this by strong induction on (U B).length, generalizing over U and V.
  -- The induction is set up via a sufficient lemma.
  suffices h : ∀ (n : Nat) (U V : WordTuple L C F Payload),
      WellTypedProgram (.whileLoop c B PBody PExit) →
      (U B).length ≤ n →
      (∀ X, localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X (.whileLoop c B PBody PExit))
          (U X)) →
      (∀ X, V X ≠ [] →
          localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X (.whileLoop c B PBody PExit))
            (U X)) →
      IsMSC (U ∘ₘ V) →
      ∃ Ubar, ZipPost (L := L) (C := C) (F := F) (Payload := Payload)
          (.whileLoop c B PBody PExit) U Ubar V from by
    intro hWF U V hPref hSide hMSC
    exact h (U B).length U V hWF (Nat.le_refl _) hPref hSide hMSC
  intro n
  induction n with
  | zero =>
    intro U V hWF hLen hPref hSide hMSC
    simp at hLen
    -- U B = [] (length 0)
    have hBNil : U B = [] := hLen
    rcases hWF with ⟨hWFBody, hWFExit⟩
    -- All U X = []
    have hUNil : ∀ X, U X = [] := by
      intro X
      by_cases hXB : X = B
      · subst X; exact hBNil
      · by_cases hRecip :
            whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X
        · have hXPref :
              localPrefixSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (LocProg.recvWhile ControlPayload.ctrlPattern B
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit))
                (U X) := by
            simpa [project, hXB, hRecip] using hPref X
          rcases localPrefixSemantics_recvWhile_cases
              (L := L) (C := C) (F := F) (Payload := Payload)
              (A := X) ControlPayload.ctrlPattern B
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
              (U X) hXPref with hXNil | hXExit | hXBody
          · exact hXNil
          · rcases hXExit with ⟨uX, _huX, hXWord⟩
            have hVB : V B = [] := by
              by_cases hVB : V B = []
              · exact hVB
              · have hBTrace := hSide B hVB
                have hPartB : participationSet (.whileLoop c B PBody PExit) B := by simp
                exact absurd hBNil
                  (project_trace_ne_nil_of_while_participating
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B c B PBody PExit (U B) hPartB hBTrace)
            have hNoUnmatched := hMSC.noUnmatchedRecv B X
            have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
              rw [rcvCount, WordTuple.concat, hXWord]
              simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
            have hSndZero : sndCount (U ∘ₘ V) B X = 0 := by
              simp [sndCount, WordTuple.concat, hBNil, hVB, countSends]
            omega
          · rcases hXBody with ⟨uX, _huX, hXWord⟩
            have hVB : V B = [] := by
              by_cases hVB : V B = []
              · exact hVB
              · have hBTrace := hSide B hVB
                have hPartB : participationSet (.whileLoop c B PBody PExit) B := by simp
                exact absurd hBNil
                  (project_trace_ne_nil_of_while_participating
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B c B PBody PExit (U B) hPartB hBTrace)
            have hNoUnmatched := hMSC.noUnmatchedRecv B X
            have hRcvPos : 1 ≤ rcvCount (U ∘ₘ V) B X := by
              rw [rcvCount, WordTuple.concat, hXWord]
              simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
            have hSndZero : sndCount (U ∘ₘ V) B X = 0 := by
              simp [sndCount, WordTuple.concat, hBNil, hVB, countSends]
            omega
        · exact localPrefixSemantics_eps_eq_nil
            (L := L) (C := C) (F := F) (Payload := Payload)
            (A := X) (U X)
            (by simpa [project, hXB, hRecip] using hPref X)
    -- Apply hExit with all U X = []
    rcases hExit hWFExit U V
        (by
          intro X
          rw [hUNil X]
          exact localPrefixSemantics_project_nil
            (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
        (by
          intro X hVX
          by_cases hPartExit : participationSet PExit X
          · have hPartWhile : participationSet (.whileLoop c B PBody PExit) X := by
              by_cases hXB : X = B
              · simp [hXB]
              · simp [hXB, hPartExit, whileRecipients]
            have hXTrace := hSide X hVX
            exact absurd hUNil
              (fun h => absurd
                (project_trace_ne_nil_of_while_participating
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  X c B PBody PExit (U X) hPartWhile hXTrace)
                (by rw [h X]; simp))
          · rw [hUNil X]
            rcases exists_project_trace
                (L := L) (C := C) (F := F) (Payload := Payload) X PExit with ⟨w, hw⟩
            have hNil :=
              project_trace_nil_of_not_participating
                (L := L) (C := C) (F := F) (Payload := Payload)
                X PExit hPartExit w hw
            rw [hNil] at hw
            simpa using hw)
        hMSC with ⟨Rbar, hZipExit⟩
    have hV : IsMSC V := zipPost_suffix_isMSC hZipExit
    rcases hZipExit with ⟨hLocExit, hCompleteExit, _hOrigExit, hMSCExit⟩
    let Ubar :=
      mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
        ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              false
              (by intro X hX; exact hX.1)
        ∘ₘ Rbar
    have hTraceUbar :
        ∀ X,
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X (.whileLoop c B PBody PExit))
            (Ubar X) := by
      have hDistExit :
          distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PExit) Rbar :=
        ⟨(fun X => (hLocExit X).2.1), hCompleteExit⟩
      intro X
      simpa [Ubar] using
        (distSemantics_project_while_exit_local
          (L := L) (C := C) (F := F) (Payload := Payload)
          c B PBody PExit Rbar hDistExit).1 X
    have hCompleteUbar : IsCompleteMSC Ubar := by
      have hDistExit :
          distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PExit) Rbar :=
        ⟨(fun X => (hLocExit X).2.1), hCompleteExit⟩
      simpa [Ubar] using
        (distSemantics_project_while_exit_local
          (L := L) (C := C) (F := F) (Payload := Payload)
          c B PBody PExit Rbar hDistExit).2
    refine ⟨Ubar, ?_⟩
    refine ⟨?_, hCompleteUbar, hMSC, concat_complete_msc _ _ hCompleteUbar hV⟩
    intro X
    refine ⟨⟨Ubar X, ?_⟩, hTraceUbar X, ?_⟩
    · rw [hUNil X]
      simp [IsPrefixWord]
    · intro hVX
      have hNotPartWhile : ¬ participationSet (.whileLoop c B PBody PExit) X := by
        intro hPartWhile
        have hXTrace := hSide X hVX
        have hXNonempty :=
          project_trace_ne_nil_of_while_participating
            (L := L) (C := C) (F := F) (Payload := Payload)
            X c B PBody PExit (U X) hPartWhile hXTrace
        rw [hUNil X] at hXNonempty
        simp at hXNonempty
      have hUbarNil :
          Ubar X = [] :=
        project_trace_nil_of_not_participating
          (L := L) (C := C) (F := F) (Payload := Payload)
          X (.whileLoop c B PBody PExit) hNotPartWhile (Ubar X) (hTraceUbar X)
      rw [hUbarNil, hUNil X]
  | succ n ih =>
    intro U V hWF hLen hPref hSide hMSC
    rcases hWF with ⟨hWFBody, hWFExit⟩
    -- Get the decider's prefix semantics
    have hBPref :
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (LocProg.localWhile c
            (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit) true
              ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
            (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit) false
              ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit))
          (U B) := by
      simpa [project] using hPref B
    -- Case analysis on the decider's prefix
    rcases localPrefixSemantics_localWhile_cases
        (L := L) (C := C) (F := F) (Payload := Payload)
        c _ _ (U B) hBPref with hBNil | ⟨seqPref, hSeqPref, hUB⟩ | ⟨seqPref, hSeqPref, hUB⟩
    -- Nil case: U B = [], length ≤ succ n is satisfied, recurse to base case
    · exact ih U V ⟨hWFBody, hWFExit⟩
        (by simp [hBNil])
        hPref hSide hMSC
    -- Exit (false) branch
    · rcases localPrefixSemantics_seq_split
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit) false)
          (project (L := L) (C := C) (F := F) (Payload := Payload) B PExit)
          seqPref hSeqPref with ⟨sB, rB, hSeqEq, hSBPref, hRBPref, hSBFull⟩
      have hUBeq : U B = AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB := by
        rw [hUB, hSeqEq, ← List.cons_append]
      have hSBTrace : rB ++ V B ≠ [] →
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              false)
            sB := by
        intro hTail
        by_cases hrB : rB = []
        · have hVB : V B ≠ [] := by
            simpa [hrB] using hTail
          have hBTrace := hSide B hVB
          have hWhileTrace :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (LocProg.localWhile c
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      true)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      false)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit))
                (U B) := by
            simpa [project] using hBTrace
          rcases (localTraceSemantics_localWhile_unfold
              (L := L) (C := C) (F := F) (Payload := Payload)
              c
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  true)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  false)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit)
              (U B)).mp hWhileTrace with hExitCase | hBodyCase
          · rcases hExitCase with ⟨exitWord, hExitTrace, hExitEq⟩
            have hSeqTrace :
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      false)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit)
                  (sB ++ rB) := by
              have hCons :
                  AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
                      sB ++ rB =
                    AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
                      exitWord := by
                simpa [hUBeq] using hExitEq
              have hTailEq : sB ++ rB = exitWord := (List.cons.inj hCons).2
              simpa [hTailEq] using hExitTrace
            rw [hrB, List.append_nil] at hSeqTrace
            rcases hSeqTrace with ⟨u1, u2, hu1, hu2, hsEq⟩
            have hWord :
                u1 =
                  controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    false := by
              exact localTraceSemantics_controlBroadcast_eq
                (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                  B PBody PExit)
                false
                u1 hu1
            rcases localPrefixSemantics_controlBroadcast_prefix
                (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                  B PBody PExit)
                false
                sB hSBPref with ⟨t, ht⟩
            rw [hWord, ht] at hsEq
            have hNil : [] = t ++ u2 := by
              have hEq' : sB ++ [] = sB ++ (t ++ u2) := by
                simpa [List.append_assoc] using hsEq
              exact List.append_inj_right hEq' rfl
            have htNil : t = [] := by
              cases t <;> simp at hNil <;> simp
            rw [htNil] at ht
            simpa [hWord, ht] using hu1
          · rcases hBodyCase with ⟨bodyWord, rest, _hBodyTrace, _hRestTrace, hBodyEq⟩
            have hCons :
                AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
                    sB ++ rB =
                  AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
                    bodyWord ++ rest := by
              simpa [hUBeq] using hBodyEq
            simp [AlphabetOf.mkWhileFalse, AlphabetOf.mkWhileTrue] at hCons
        · exact hSBFull hrB
      let kWhile : L → Nat := fun X =>
        if X = B then 1 + sB.length
        else if whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X ∧ U X ≠ [] then 1
        else 0
      let R : WordTuple L C F Payload := fun X => (U X).drop (kWhile X)
      have hRB : R B = rB := by
        simp only [R, kWhile, ite_true]
        rw [hUBeq, show 1 + sB.length = sB.length + 1 from by omega]
        simp [List.drop_succ_cons]
      have hRPref : ∀ X,
          localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
            (R X) := by
        intro X
        by_cases hXB : X = B
        · subst X
          rw [hRB]
          exact hRBPref
        · by_cases hRecip :
              whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X
          · rcases while_exit_recipient_prefix_cases
                (L := L) (C := C) (F := F) (Payload := Payload)
                c B X PBody PExit U V hXB hRecip sB rB hUBeq hSBPref
                hSBTrace
                (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
            · simp only [R, kWhile, hXB, ite_false, hRecip, hXNil, ite_true]
              simp
              exact localPrefixSemantics_project_nil
                (L := L) (C := C) (F := F) (Payload := Payload) X PExit
            · simp only [R, kWhile, hXB, ite_false, hRecip, hXWord, ite_true]
              simp
              exact huX
          · have hXNil : U X = [] :=
              localPrefixSemantics_eps_eq_nil
                (L := L) (C := C) (F := F) (Payload := Payload)
                (A := X) (U X)
                (by simpa [project, hXB, hRecip] using hPref X)
            simp only [R, kWhile, hXB, ite_false, hRecip, hXNil, ite_false]
            simp
            exact localPrefixSemantics_project_nil
              (L := L) (C := C) (F := F) (Payload := Payload) X PExit
      have hRSide : ∀ X, V X ≠ [] →
          localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
            (R X) := by
        intro X hVX
        by_cases hXB : X = B
        · subst X
          rw [hRB]
          have hBTrace := hSide B hVX
          have hWhileTrace :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (LocProg.localWhile c
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      true)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      false)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit))
                (U B) := by
            simpa [project] using hBTrace
          have hSeqTrace :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    false)
                  ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit)
                (sB ++ rB) := by
            rcases (localTraceSemantics_localWhile_unfold
                (L := L) (C := C) (F := F) (Payload := Payload)
                c
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    true)
                  ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    false)
                  ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit)
                (U B)).mp hWhileTrace with hExitCase | hBodyCase
            · rcases hExitCase with ⟨exitWord, hExitTrace, hExitEq⟩
              have hCons :
                  AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
                      sB ++ rB =
                    AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
                      exitWord := by
                simpa [hUBeq] using hExitEq
              have hTailEq : sB ++ rB = exitWord := (List.cons.inj hCons).2
              simpa [hTailEq] using hExitTrace
            · rcases hBodyCase with ⟨bodyWord, rest, _hBodyTrace, _hRestTrace, hBodyEq⟩
              have hCons :
                  AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
                      sB ++ rB =
                    AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
                      bodyWord ++ rest := by
                simpa [hUBeq] using hBodyEq
              simp [AlphabetOf.mkWhileFalse, AlphabetOf.mkWhileTrue] at hCons
          have hSWord :
              sB =
                controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  false := by
            exact localTraceSemantics_controlBroadcast_eq
              (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                B PBody PExit)
              false
              sB
              (hSBTrace (by
                intro hNil
                exact hVX (List.eq_nil_of_append_eq_nil hNil).2))
          rcases hSeqTrace with ⟨u1, u2, hu1, hu2, hEq⟩
          have hWord :
              u1 =
                controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  false := by
            exact localTraceSemantics_controlBroadcast_eq
              (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                B PBody PExit)
              false
              u1 hu1
          rw [hWord, hSWord] at hEq
          have hRBU2 : rB = u2 := by
            exact List.append_inj_right hEq rfl
          simpa [hRBU2] using hu2
        · by_cases hRecip :
              whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X
          · rcases while_exit_recipient_prefix_cases
                (L := L) (C := C) (F := F) (Payload := Payload)
                c B X PBody PExit U V hXB hRecip sB rB hUBeq hSBPref
                hSBTrace
                (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
            · exfalso
              exact hVX (while_recipient_nil_tail_nil
                (L := L) (C := C) (F := F) (Payload := Payload)
                c B X PBody PExit U V hXB hRecip hSide hXNil)
            · simp only [R, kWhile, hXB, ite_false, hRecip, hXWord, ite_true]
              simp
              have hXTrace := hSide X hVX
              have hRecvTrace :
                  localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    (LocProg.recvWhile ControlPayload.ctrlPattern B
                      (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                      (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit))
                    (U X) := by
                simpa [project, hXB, hRecip] using hXTrace
              rcases (localTraceSemantics_recvWhile_unfold
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  ControlPayload.ctrlPattern B
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
                  (U X)).mp hRecvTrace with hExitCase | hBodyCase
              · rcases hExitCase with ⟨exitWord, hExitTrace, hExitEq⟩
                have hCons :
                    AlphabetOf.mkRecv (C := C) (F := F) X
                        (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: uX =
                      AlphabetOf.mkRecv (C := C) (F := F) X
                        (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: exitWord := by
                  simpa [hXWord] using hExitEq
                have hTailEq : uX = exitWord := (List.cons.inj hCons).2
                simpa [hTailEq] using hExitTrace
              · rcases hBodyCase with ⟨bodyWord, rest, _hBodyTrace, _hRestTrace, hBodyEq⟩
                have hCons :
                    AlphabetOf.mkRecv (C := C) (F := F) X
                        (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: uX =
                      AlphabetOf.mkRecv (C := C) (F := F) X
                        (ControlPayload.setDecision true ControlPayload.ctrlPattern) B ::
                        bodyWord ++ rest := by
                  simpa [hXWord] using hBodyEq
                have hPayloadEq :
                    ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern =
                      ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern := by
                  simpa [AlphabetOf.mkRecv] using (List.cons.inj hCons).1
                exact False.elim (ctrlFalse_ne_ctrlTrue hPayloadEq)
          · have hXNil : U X = [] :=
              localPrefixSemantics_eps_eq_nil
                (L := L) (C := C) (F := F) (Payload := Payload)
                (A := X) (U X)
                (by simpa [project, hXB, hRecip] using hPref X)
            have hNotPartExit : ¬ participationSet PExit X := by
              intro hPartExit
              exact hRecip ⟨hXB, Or.inr hPartExit⟩
            simpa [R, kWhile, hXB, hRecip, hXNil] using
              project_empty_trace_of_not_participating
                (L := L) (C := C) (F := F) (Payload := Payload)
                X PExit hNotPartExit
      have hRMSC : IsMSC (R ∘ₘ V) := by
        have hkU : ∀ X, kWhile X ≤ (U X).length := by
          intro X
          by_cases hXB : X = B
          · subst X
            simp [kWhile, hUBeq]
            omega
          · by_cases hRecipX :
                whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X ∧
                  U X ≠ []
            · cases hUX : U X with
              | nil => exact False.elim (hRecipX.2 hUX)
              | cons hd tl => simp [kWhile, hXB, hRecipX, hUX]
            · simp [kWhile, hXB, hRecipX]
        have hTakeB :
            List.take (1 + sB.length) (U B ++ V B) =
              AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB := by
          rw [hUBeq]
          rw [List.append_assoc]
          have hLen' :
              1 + sB.length ≤
                (AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB).length := by
            simpa [Nat.add_comm] using Nat.le_refl (sB.length + 1)
          have hTake' :
              List.take (1 + sB.length)
                ((AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB) ++
                  (rB ++ V B)) =
                List.take (1 + sB.length)
                  (AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB) := by
            simpa using
              (List.take_append_of_le_length
                (l₁ := AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB)
                (l₂ := rB ++ V B)
                (i := 1 + sB.length)
                hLen')
          have hTake'' :
              List.take (1 + sB.length)
                (AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB) =
              AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB := by
            simpa [Nat.add_comm] using
              (List.take_length
                (l := AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB))
          exact hTake'.trans hTake''
        have hTakeBM :
            ((U ∘ₘ V) B).take (kWhile B) =
              AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c :: sB := by
          simpa [WordTuple.concat, kWhile] using hTakeB
        have hDropEq :
            (fun X => ((U ∘ₘ V) X).drop (kWhile X)) = R ∘ₘ V := by
          calc
            (fun X => ((U ∘ₘ V) X).drop (kWhile X))
                = (fun X => (U X).drop (kWhile X)) ∘ₘ V :=
                  drop_concat_prefix_eq
                    (L := L) (C := C) (F := F) (Payload := Payload) U V kWhile hkU
            _ = R ∘ₘ V := by rfl
        rw [← hDropEq]
        refine suffix_msc_of_safe_prefix
          (L := L) (C := C) (F := F) (Payload := Payload)
          (M := U ∘ₘ V) (k := kWhile) ?hkWhileExit ?hPrefixLeWhileExit
          ?hSendSurplusDeadWhileExit hMSC
        · intro X
          have h := hkU X
          simp [WordTuple.concat]
          omega
        · intro A Y
          by_cases hAB : A = B
          · subst A
            by_cases hYB : Y = B
            · subst Y
              have hRecvZero :
                  countRecvs B (((U ∘ₘ V) B).take (kWhile B)) = 0 := by
                rw [hTakeBM]
                have hsZero :
                    countRecvs B sB = 0 :=
                  controlBroadcast_prefix_recv_count_zero
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    false sB hSBPref
                simpa [countRecvs, AlphabetOf.mkWhileFalse, Letter.isRecvFrom, hsZero]
              omega
            · by_cases hRecipY :
                  whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit Y ∧
                    U Y ≠ []
              · rcases while_exit_recipient_prefix_cases
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B Y PBody PExit U V hYB hRecipY.1 sB rB hUBeq hSBPref
                    hSBTrace
                    (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
                · exact False.elim (hRecipY.2 hYNil)
                · have hRecvOne :
                      countRecvs B (((U ∘ₘ V) Y).take (kWhile Y)) = 1 := by
                    have hTakeY :
                        ((U ∘ₘ V) Y).take (kWhile Y) =
                          [AlphabetOf.mkRecv (C := C) (F := F) Y
                            (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B] := by
                      simp [WordTuple.concat, kWhile, hYB, hRecipY, hYWord]
                    rw [hTakeY]
                    simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
                  have hSendPos :
                      1 ≤ countSends Y sB :=
                    while_exit_prefix_send_count_pos
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      c B Y PBody PExit U V hYB hRecipY.1 sB rB uY hUBeq hSBPref
                      hSBTrace hYWord hMSC
                  have hSendEq :
                      countSends Y (((U ∘ₘ V) B).take (kWhile B)) = countSends Y sB := by
                    rw [hTakeBM]
                    simp [countSends, AlphabetOf.mkWhileFalse, Letter.isSendTo]
                  omega
              · have hRecvZero :
                    countRecvs B (((U ∘ₘ V) Y).take (kWhile Y)) = 0 := by
                  simp [WordTuple.concat, kWhile, hYB, hRecipY]
                omega
          · have hRecvZero :
                countRecvs A (((U ∘ₘ V) Y).take (kWhile Y)) = 0 := by
              have hBA : B ≠ A := by
                intro hBA
                exact hAB hBA.symm
              by_cases hYB : Y = B
              · subst Y
                rw [hTakeBM]
                have hsZero :
                    countRecvs A sB = 0 :=
                  controlBroadcast_prefix_recv_count_zero
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B A
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    false sB hSBPref
                simpa [countRecvs, AlphabetOf.mkWhileFalse, Letter.isRecvFrom, hsZero, hBA]
              · by_cases hRecipY :
                    whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit Y ∧
                      U Y ≠ []
                · rcases while_exit_recipient_prefix_cases
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      c B Y PBody PExit U V hYB hRecipY.1 sB rB hUBeq hSBPref
                      hSBTrace
                      (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
                  · exact False.elim (hRecipY.2 hYNil)
                  · have hTakeY :
                        ((U ∘ₘ V) Y).take (kWhile Y) =
                          [AlphabetOf.mkRecv (C := C) (F := F) Y
                            (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B] := by
                      simp [WordTuple.concat, kWhile, hYB, hRecipY, hYWord]
                    rw [hTakeY]
                    simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom, hBA]
                · simp [WordTuple.concat, kWhile, hYB, hRecipY]
            omega
        · intro A Y hSurplus
          by_cases hAB : A = B
          · subst A
            by_cases hYB : Y = B
            · subst Y
              exfalso
              have hSendZero :
                  countSends B (((U ∘ₘ V) B).take (kWhile B)) = 0 := by
                rw [hTakeBM]
                have hsZero :
                    countSends B sB = 0 :=
                  controlBroadcast_prefix_send_count_zero_nonRecipient
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    false sB hSBPref
                    (by
                      intro hSelf
                      exact (whileRecipients_no_self
                        (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit B hSelf) rfl)
                simpa [countSends, AlphabetOf.mkWhileFalse, Letter.isSendTo, hsZero]
              omega
            · by_cases hRecipY :
                  whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit Y
              · by_cases hUY : U Y = []
                · have hVY : V Y = [] :=
                    while_recipient_nil_tail_nil
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      c B Y PBody PExit U V hYB hRecipY hSide hUY
                  simp [WordTuple.concat, kWhile, hYB, hRecipY, hUY, hVY, countRecvs]
                · exfalso
                  have hRecipY' :
                      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit Y ∧
                        U Y ≠ [] := ⟨hRecipY, hUY⟩
                  rcases while_exit_recipient_prefix_cases
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      c B Y PBody PExit U V hYB hRecipY sB rB hUBeq hSBPref
                      hSBTrace
                      (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
                  · exact hUY hYNil
                  · have hRecvOne :
                        countRecvs B (((U ∘ₘ V) Y).take (kWhile Y)) = 1 := by
                      have hTakeY :
                          ((U ∘ₘ V) Y).take (kWhile Y) =
                            [AlphabetOf.mkRecv (C := C) (F := F) Y
                              (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B] := by
                        simp [WordTuple.concat, kWhile, hYB, hRecipY', hYWord]
                      rw [hTakeY]
                      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
                    have hSendLe :
                        countSends Y sB ≤ 1 :=
                      controlBroadcast_prefix_send_count_le_one_recipient
                        (L := L) (C := C) (F := F) (Payload := Payload)
                        B Y
                        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                        false sB hSBPref (by intro Z hZ; exact hZ.1) hRecipY
                    have hSendEq :
                        countSends Y (((U ∘ₘ V) B).take (kWhile B)) = countSends Y sB := by
                      rw [hTakeBM]
                      simp [countSends, AlphabetOf.mkWhileFalse, Letter.isSendTo]
                    omega
              · exfalso
                have hSendZero :
                    countSends Y (((U ∘ₘ V) B).take (kWhile B)) = 0 := by
                  rw [hTakeBM]
                  have hsZero :
                      countSends Y sB = 0 :=
                    controlBroadcast_prefix_send_count_zero_nonRecipient
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      B Y
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                      false sB hSBPref hRecipY
                  simpa [countSends, AlphabetOf.mkWhileFalse, Letter.isSendTo, hsZero]
                omega
          · exfalso
            have hSendZero :
                countSends Y (((U ∘ₘ V) A).take (kWhile A)) = 0 := by
              by_cases hRecipA :
                  whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit A ∧
                    U A ≠ []
              · rcases while_exit_recipient_prefix_cases
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B A PBody PExit U V hAB hRecipA.1 sB rB hUBeq hSBPref
                    hSBTrace
                    (hPref A) hMSC with hANil | ⟨uA, _huA, hAWord⟩
                · exact False.elim (hRecipA.2 hANil)
                · have hTakeA :
                      ((U ∘ₘ V) A).take (kWhile A) =
                        [AlphabetOf.mkRecv (C := C) (F := F) A
                          (ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern) B] := by
                    simp [WordTuple.concat, kWhile, hAB, hRecipA, hAWord]
                  rw [hTakeA]
                  simp [countSends, AlphabetOf.mkRecv, Letter.isSendTo]
              · simp [WordTuple.concat, kWhile, hAB, hRecipA]
            omega
      rcases hExit hWFExit R V hRPref hRSide hRMSC with ⟨Rbar, hZipExit⟩
      have hSuffixExit : IsMSC V := zipPost_suffix_isMSC hZipExit
      rcases hZipExit with ⟨hLocExit, hCompleteExit, _hOrigExit, _hMSCExit⟩
      let Ubar :=
        mscWhileFalse (C := C) (F := F) (Payload := Payload) c B
          ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                false
                (by intro X hX; exact hX.1)
          ∘ₘ Rbar
      have hTraceUbar :
          ∀ X,
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X (.whileLoop c B PBody PExit))
              (Ubar X) := by
        have hDistExit :
            distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
              (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PExit) Rbar :=
          ⟨(fun X => (hLocExit X).2.1), hCompleteExit⟩
        intro X
        simpa [Ubar] using
          (distSemantics_project_while_exit_local
            (L := L) (C := C) (F := F) (Payload := Payload)
            c B PBody PExit Rbar hDistExit).1 X
      have hCompleteUbar : IsCompleteMSC Ubar := by
        have hDistExit :
            distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
              (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PExit) Rbar :=
          ⟨(fun X => (hLocExit X).2.1), hCompleteExit⟩
        simpa [Ubar] using
          (distSemantics_project_while_exit_local
            (L := L) (C := C) (F := F) (Payload := Payload)
            c B PBody PExit Rbar hDistExit).2
      refine ⟨Ubar, ?_⟩
      refine ⟨?_, hCompleteUbar, hMSC, concat_complete_msc _ _ hCompleteUbar hSuffixExit⟩
      intro X
      have hPrefixX :
          IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) (U X) (Ubar X) := by
        by_cases hXB : X = B
        · subst X
          rcases hLocExit B with ⟨hRPrefB, _hTraceB, _hEqB⟩
          rcases hRPrefB with ⟨tR, htR⟩
          rcases localPrefixSemantics_controlBroadcast_prefix
              (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              false
              sB hSBPref with ⟨tS, htS⟩
          by_cases hrB : rB = []
          · subst hrB
            refine ⟨tS ++ tR, ?_⟩
            simp [Ubar, WordTuple.concat, mscWhileFalse, hUBeq, hRB, htS, htR, List.append_assoc,
              choiceWhileFalse,
              controlBroadcastMSC_decider
                (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                false
                (by intro Y hY; exact hY.1)]
          · have hSBFull' :
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    false)
                  sB := hSBFull hrB
            have hSWord :
                sB =
                  controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    false := by
              exact localTraceSemantics_controlBroadcast_eq
                (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                false
                sB hSBFull'
            have hWordEq :
                controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                  false = sB := hSWord.symm
            refine ⟨tR, ?_⟩
            simp [Ubar, WordTuple.concat, mscWhileFalse, hUBeq, hRB, hWordEq, htR, List.append_assoc,
              choiceWhileFalse,
              controlBroadcastMSC_decider
                (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                false
                (by intro Y hY; exact hY.1)]
        · by_cases hRecip :
              whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X
          · rcases while_exit_recipient_prefix_cases
                (L := L) (C := C) (F := F) (Payload := Payload)
                c B X PBody PExit U V hXB hRecip sB rB hUBeq hSBPref
                hSBTrace
                (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
            · refine ⟨Ubar X, by simp [hXNil, IsPrefixWord]⟩
            · rcases hLocExit X with ⟨hRPrefX, _hTraceX, _hEqX⟩
              rcases hRPrefX with ⟨tR, htR⟩
              have hRX : R X = uX := by
                simp [R, kWhile, hXB, hRecip, hXWord]
              refine ⟨tR, ?_⟩
              simp [Ubar, WordTuple.concat, mscWhileFalse, hXB, hRecip, hXWord, htR, hRX,
                List.append_assoc, controlDecisionPayload,
                controlBroadcastMSC_recipient
                  (C := C) (F := F) (Payload := Payload)
                  B X
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                  false
                  (by intro Y hY; exact hY.1)
                  hRecip]
          · have hXNil : U X = [] :=
              localPrefixSemantics_eps_eq_nil
                (L := L) (C := C) (F := F) (Payload := Payload)
                (A := X) (U X)
                (by simpa [project, hXB, hRecip] using hPref X)
            refine ⟨Ubar X, by simp [hXNil, IsPrefixWord]⟩
      refine ⟨hPrefixX, hTraceUbar X, ?_⟩
      intro hVX
      exact localTraceSemantics_prefix_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        (A := X)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (.whileLoop c B PBody PExit))
        (u := U X)
        (v := Ubar X)
        (hSide X hVX)
        (hTraceUbar X)
        hPrefixX |>.symm
    -- Body (true) loop-back branch: use IH
    · have hSeqPrefAssoc :
          localPrefixSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                  B PBody PExit) true)
              ;;ₗ ((project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B
                    (.whileLoop c B PBody PExit)))
            seqPref := by
        simpa [project] using
          localPrefixSemantics_seq_assoc_left
            (L := L) (C := C) (F := F) (Payload := Payload)
            (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                B PBody PExit)
              true)
            (project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
            (LocProg.localWhile c
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  true)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  false)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit))
            seqPref hSeqPref
      rcases localPrefixSemantics_seq_split
          (L := L) (C := C) (F := F) (Payload := Payload)
          (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
            B (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
              B PBody PExit) true)
          ((project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
            ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B
                (.whileLoop c B PBody PExit))
          seqPref hSeqPrefAssoc with ⟨sB, rB, hSeqEq, hSBPref, hRBPref, hSBFull⟩
      have hUBeq : U B = AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB ++ rB := by
        rw [hUB, hSeqEq, ← List.cons_append]
      have hSBTrace : rB ++ V B ≠ [] →
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              true)
            sB := by
        intro hTail
        by_cases hrB : rB = []
        · have hVB : V B ≠ [] := by
            simpa [hrB] using hTail
          have hBTrace := hSide B hVB
          have hWhileTrace :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (LocProg.localWhile c
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      true)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      false)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit))
                (U B) := by
            simpa [project] using hBTrace
          rcases (localTraceSemantics_localWhile_unfold
              (L := L) (C := C) (F := F) (Payload := Payload)
              c
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  true)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
              ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  false)
                ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit)
              (U B)).mp hWhileTrace with hExitCase | hBodyCase
          · rcases hExitCase with ⟨exitWord, _hExitTrace, hExitEq⟩
            have hCons :
                AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
                    sB ++ rB =
                  AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
                    exitWord := by
              simpa [hUBeq] using hExitEq
            simp [AlphabetOf.mkWhileFalse, AlphabetOf.mkWhileTrue] at hCons
          · rcases hBodyCase with ⟨bodyWord, rest, hBodyTrace, hRestTrace, hBodyEq⟩
            have hSeqTrace :
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      true)
                    ;;ₗ ((project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                      ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B
                          (.whileLoop c B PBody PExit)))
                  sB := by
              have hCons :
                  AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
                      sB =
                    AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
                      bodyWord ++ rest := by
                simpa [hUBeq, hrB] using hBodyEq
              have hTailEq : sB = bodyWord ++ rest := (List.cons.inj hCons).2
              have hLeftTrace :
                  localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    (((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                        B
                        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                        true)
                      ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                      ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B
                          (.whileLoop c B PBody PExit))
                    (bodyWord ++ rest) := by
                simpa [project] using
                  localTraceSemantics_seq_intro
                    ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                        B
                        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                        true)
                      ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                    (LocProg.localWhile c
                      ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                          B
                          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                            B PBody PExit)
                          true)
                        ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                      ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                          B
                          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                            B PBody PExit)
                          false)
                        ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit))
                    bodyWord rest hBodyTrace hRestTrace
              simpa [hTailEq, project] using
                localTraceSemantics_seq_assoc_left
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    true)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) B
                    (.whileLoop c B PBody PExit))
                  (bodyWord ++ rest) hLeftTrace
            rcases hSeqTrace with ⟨u1, u2, hu1, hu2, hsEq⟩
            have hWord :
                u1 =
                  controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    true := by
              exact localTraceSemantics_controlBroadcast_eq
                (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                  B PBody PExit)
                true
                u1 hu1
            rcases localPrefixSemantics_controlBroadcast_prefix
                (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                  B PBody PExit)
                true
                sB hSBPref with ⟨t, ht⟩
            rw [hWord, ht] at hsEq
            have hNil : [] = t ++ u2 := by
              have hEq' : sB ++ [] = sB ++ (t ++ u2) := by
                simpa [List.append_assoc] using hsEq
              exact List.append_inj_right hEq' rfl
            have htNil : t = [] := by
              cases t <;> simp at hNil <;> simp
            rw [htNil] at ht
            simpa [hWord, ht] using hu1
        · exact hSBFull hrB
      let kWhile : L → Nat := fun X =>
        if X = B then 1 + sB.length
        else if whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X ∧ U X ≠ [] then 1
        else 0
      let R : WordTuple L C F Payload := fun X => (U X).drop (kWhile X)
      have hRB : R B = rB := by
        simp only [R, kWhile, ite_true]
        rw [hUBeq, show 1 + sB.length = sB.length + 1 from by omega]
        simp [List.drop_succ_cons]
      have hRPref : ∀ X,
          localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X
              (PBody ;; .whileLoop c B PBody PExit))
            (R X) := by
        intro X
        by_cases hXB : X = B
        · subst X
          rw [hRB]
          simpa [project] using hRBPref
        · by_cases hRecip :
              whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X
          · rcases while_body_recipient_prefix_cases
                (L := L) (C := C) (F := F) (Payload := Payload)
                c B X PBody PExit U V hXB hRecip sB rB hUBeq hSBPref
                hSBTrace
                (hPref X) hMSC with hXNil | ⟨uX, huX, hXWord⟩
            · simp only [R, kWhile, hXB, ite_false, hRecip, hXNil, ite_true]
              simp
              exact localPrefixSemantics_project_nil
                (L := L) (C := C) (F := F) (Payload := Payload) X
                (PBody ;; .whileLoop c B PBody PExit)
            · simp only [R, kWhile, hXB, ite_false, hRecip, hXWord, ite_true]
              simp
              simpa [project] using huX
          · have hXNil : U X = [] :=
              localPrefixSemantics_eps_eq_nil
                (L := L) (C := C) (F := F) (Payload := Payload)
                (A := X) (U X)
                (by simpa [project, hXB, hRecip] using hPref X)
            simp only [R, kWhile, hXB, ite_false, hRecip, hXNil, ite_false]
            simp
            exact localPrefixSemantics_project_nil
              (L := L) (C := C) (F := F) (Payload := Payload) X
              (PBody ;; .whileLoop c B PBody PExit)
      have hRSide : ∀ X, V X ≠ [] →
          localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X
              (PBody ;; .whileLoop c B PBody PExit))
            (R X) := by
        intro X hVX
        by_cases hXB : X = B
        · subst X
          rw [hRB]
          have hBTrace := hSide B hVX
          have hWhileTrace :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (LocProg.localWhile c
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      true)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                  ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                      B
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit)
                      false)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit))
                (U B) := by
            simpa [project] using hBTrace
          have hSeqTrace :
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    true)
                  ;;ₗ ((project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B
                        (.whileLoop c B PBody PExit)))
                (sB ++ rB) := by
            rcases (localTraceSemantics_localWhile_unfold
                (L := L) (C := C) (F := F) (Payload := Payload)
                c
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    true)
                  ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    false)
                  ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit)
                (U B)).mp hWhileTrace with hExitCase | hBodyCase
            · rcases hExitCase with ⟨exitWord, _hExitTrace, hExitEq⟩
              have hCons :
                  AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
                      sB ++ rB =
                    AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) B c ::
                      exitWord := by
                simpa [hUBeq] using hExitEq
              simp [AlphabetOf.mkWhileFalse, AlphabetOf.mkWhileTrue] at hCons
            · rcases hBodyCase with ⟨bodyWord, rest, hBodyTrace, hRestTrace, hBodyEq⟩
              have hCons :
                  AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
                      sB ++ rB =
                    AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
                      bodyWord ++ rest := by
                simpa [hUBeq] using hBodyEq
              have hTailEq : sB ++ rB = bodyWord ++ rest := (List.cons.inj hCons).2
              have hLeftTrace :
                  localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    (((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                        B
                        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                        true)
                      ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                      ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B
                          (.whileLoop c B PBody PExit))
                    (bodyWord ++ rest) := by
                simpa [project] using
                  localTraceSemantics_seq_intro
                    ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                        B
                        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                          B PBody PExit)
                        true)
                      ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                    (LocProg.localWhile c
                      ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                          B
                          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                            B PBody PExit)
                          true)
                        ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                      ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                          B
                          (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                            B PBody PExit)
                          false)
                        ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) B PExit))
                    bodyWord rest hBodyTrace hRestTrace
              simpa [hTailEq, project] using
                localTraceSemantics_seq_assoc_left
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                      B PBody PExit)
                    true)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) B PBody)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) B
                    (.whileLoop c B PBody PExit))
                  (bodyWord ++ rest) hLeftTrace
          have hSWord :
              sB =
                controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  true := by
            exact localTraceSemantics_controlBroadcast_eq
              (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                B PBody PExit)
              true
              sB
              (hSBTrace (by
                intro hNil
                exact hVX (List.eq_nil_of_append_eq_nil hNil).2))
          rcases hSeqTrace with ⟨u1, u2, hu1, hu2, hEq⟩
          have hWord :
              u1 =
                controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                    B PBody PExit)
                  true := by
            exact localTraceSemantics_controlBroadcast_eq
              (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload)
                B PBody PExit)
              true
              u1 hu1
          rw [hWord, hSWord] at hEq
          have hRBU2 : rB = u2 := by
            exact List.append_inj_right hEq rfl
          simpa [project, hRBU2] using hu2
        · by_cases hRecip :
              whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X
          · rcases while_body_recipient_prefix_cases
                (L := L) (C := C) (F := F) (Payload := Payload)
                c B X PBody PExit U V hXB hRecip sB rB hUBeq hSBPref
                hSBTrace
                (hPref X) hMSC with hXNil | ⟨uX, _huX, hXWord⟩
            · exfalso
              exact hVX (while_recipient_nil_tail_nil
                (L := L) (C := C) (F := F) (Payload := Payload)
                c B X PBody PExit U V hXB hRecip hSide hXNil)
            · simp only [R, kWhile, hXB, ite_false, hRecip, hXWord, ite_true]
              simp
              have hXTrace := hSide X hVX
              have hRecvTrace :
                  localTraceSemantics
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    (LocProg.recvWhile ControlPayload.ctrlPattern B
                      (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                      (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit))
                    (U X) := by
                simpa [project, hXB, hRecip] using hXTrace
              rcases (localTraceSemantics_recvWhile_unfold
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  ControlPayload.ctrlPattern B
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PExit)
                  (U X)).mp hRecvTrace with hExitCase | hBodyCase
              · rcases hExitCase with ⟨exitWord, _hExitTrace, hExitEq⟩
                have hCons :
                    AlphabetOf.mkRecv (C := C) (F := F) X
                        (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: uX =
                      AlphabetOf.mkRecv (C := C) (F := F) X
                        (ControlPayload.setDecision false ControlPayload.ctrlPattern) B :: exitWord := by
                  simpa [hXWord] using hExitEq
                have hPayloadEq :
                    ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern =
                      ControlPayload.setDecision (Payload := Payload) false ControlPayload.ctrlPattern := by
                  simpa [AlphabetOf.mkRecv] using (List.cons.inj hCons).1
                exact False.elim (ctrlFalse_ne_ctrlTrue hPayloadEq.symm)
              · rcases hBodyCase with ⟨bodyWord, rest, hBodyTrace, hRestTrace, hBodyEq⟩
                have hCons :
                    AlphabetOf.mkRecv (C := C) (F := F) X
                        (ControlPayload.setDecision true ControlPayload.ctrlPattern) B :: uX =
                      AlphabetOf.mkRecv (C := C) (F := F) X
                        (ControlPayload.setDecision true ControlPayload.ctrlPattern) B ::
                        bodyWord ++ rest := by
                  simpa [hXWord] using hBodyEq
                have hTailEq : uX = bodyWord ++ rest := (List.cons.inj hCons).2
                have hRestTrace' :
                    localTraceSemantics
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      (project (L := L) (C := C) (F := F) (Payload := Payload) X
                        (.whileLoop c B PBody PExit))
                      rest := by
                  simpa [project, hXB, hRecip] using hRestTrace
                simpa [project, hTailEq] using
                  localTraceSemantics_seq_intro
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                    (project (L := L) (C := C) (F := F) (Payload := Payload) X
                      (.whileLoop c B PBody PExit))
                    bodyWord rest hBodyTrace hRestTrace'
          · have hXNil : U X = [] :=
              localPrefixSemantics_eps_eq_nil
                (L := L) (C := C) (F := F) (Payload := Payload)
                (A := X) (U X)
                (by simpa [project, hXB, hRecip] using hPref X)
            have hNotPartBody : ¬ participationSet PBody X := by
              intro hPartBody
              exact hRecip ⟨hXB, Or.inl hPartBody⟩
            have hTraceBodyNil :
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                  [] :=
              project_empty_trace_of_not_participating
                (L := L) (C := C) (F := F) (Payload := Payload)
                X PBody hNotPartBody
            have hTraceWhileNil :
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X
                    (.whileLoop c B PBody PExit))
                  [] := by
              simpa [project, hXB, hRecip] using
                (localTraceSemantics_eps_nil
                  (L := L) (C := C) (F := F) (Payload := Payload) (A := X))
            have hSeqNil :
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  ((project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                    ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) X
                        (.whileLoop c B PBody PExit))
                  ([] : LocalWord (C := C) (F := F) (Payload := Payload) X) := by
              simpa using
                localTraceSemantics_seq_intro
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
                  (project (L := L) (C := C) (F := F) (Payload := Payload) X
                    (.whileLoop c B PBody PExit))
                  [] [] hTraceBodyNil hTraceWhileNil
            simpa [R, kWhile, hXB, hRecip, hXNil, project] using hSeqNil
      have hRMSC : IsMSC (R ∘ₘ V) := by
        have hkU : ∀ X, kWhile X ≤ (U X).length := by
          intro X
          by_cases hXB : X = B
          · subst X
            simp [kWhile, hUBeq]
            omega
          · by_cases hRecipX :
                whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X ∧
                  U X ≠ []
            · cases hUX : U X with
              | nil => exact False.elim (hRecipX.2 hUX)
              | cons hd tl => simp [kWhile, hXB, hRecipX, hUX]
            · simp [kWhile, hXB, hRecipX]
        have hTakeB :
            List.take (1 + sB.length) (U B ++ V B) =
              AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB := by
          rw [hUBeq]
          rw [List.append_assoc]
          have hLen' :
              1 + sB.length ≤
                (AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB).length := by
            simpa [Nat.add_comm] using Nat.le_refl (sB.length + 1)
          have hTake' :
              List.take (1 + sB.length)
                ((AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB) ++
                  (rB ++ V B)) =
                List.take (1 + sB.length)
                  (AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB) := by
            simpa using
              (List.take_append_of_le_length
                (l₁ := AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB)
                (l₂ := rB ++ V B)
                (i := 1 + sB.length)
                hLen')
          have hTake'' :
              List.take (1 + sB.length)
                (AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB) =
              AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB := by
            simpa [Nat.add_comm] using
              (List.take_length
                (l := AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB))
          exact hTake'.trans hTake''
        have hTakeBM :
            ((U ∘ₘ V) B).take (kWhile B) =
              AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c :: sB := by
          simpa [WordTuple.concat, kWhile] using hTakeB
        have hDropEq :
            (fun X => ((U ∘ₘ V) X).drop (kWhile X)) = R ∘ₘ V := by
          calc
            (fun X => ((U ∘ₘ V) X).drop (kWhile X))
                = (fun X => (U X).drop (kWhile X)) ∘ₘ V :=
                  drop_concat_prefix_eq
                    (L := L) (C := C) (F := F) (Payload := Payload) U V kWhile hkU
            _ = R ∘ₘ V := by rfl
        rw [← hDropEq]
        refine suffix_msc_of_safe_prefix
          (L := L) (C := C) (F := F) (Payload := Payload)
          (M := U ∘ₘ V) (k := kWhile) ?hkWhileBody ?hPrefixLeWhileBody
          ?hSendSurplusDeadWhileBody hMSC
        · intro X
          have h := hkU X
          simp [WordTuple.concat]
          omega
        · intro A Y
          by_cases hAB : A = B
          · subst A
            by_cases hYB : Y = B
            · subst Y
              have hRecvZero :
                  countRecvs B (((U ∘ₘ V) B).take (kWhile B)) = 0 := by
                rw [hTakeBM]
                have hsZero :
                    countRecvs B sB = 0 :=
                  controlBroadcast_prefix_recv_count_zero
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    true sB hSBPref
                simpa [countRecvs, AlphabetOf.mkWhileTrue, Letter.isRecvFrom, hsZero]
              omega
            · by_cases hRecipY :
                  whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit Y ∧
                    U Y ≠ []
              · rcases while_body_recipient_prefix_cases
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B Y PBody PExit U V hYB hRecipY.1 sB rB hUBeq hSBPref
                    hSBTrace
                    (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
                · exact False.elim (hRecipY.2 hYNil)
                · have hRecvOne :
                      countRecvs B (((U ∘ₘ V) Y).take (kWhile Y)) = 1 := by
                    have hTakeY :
                        ((U ∘ₘ V) Y).take (kWhile Y) =
                          [AlphabetOf.mkRecv (C := C) (F := F) Y
                            (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B] := by
                      simp [WordTuple.concat, kWhile, hYB, hRecipY, hYWord]
                    rw [hTakeY]
                    simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
                  have hSendPos :
                      1 ≤ countSends Y sB :=
                    while_true_prefix_send_count_pos
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      c B Y PBody PExit U V hYB hRecipY.1 sB rB uY hUBeq hSBPref
                      hSBTrace hYWord hMSC
                  have hSendEq :
                      countSends Y (((U ∘ₘ V) B).take (kWhile B)) = countSends Y sB := by
                    rw [hTakeBM]
                    simp [countSends, AlphabetOf.mkWhileTrue, Letter.isSendTo]
                  omega
              · have hRecvZero :
                    countRecvs B (((U ∘ₘ V) Y).take (kWhile Y)) = 0 := by
                  simp [WordTuple.concat, kWhile, hYB, hRecipY]
                omega
          · have hRecvZero :
                countRecvs A (((U ∘ₘ V) Y).take (kWhile Y)) = 0 := by
              have hBA : B ≠ A := by
                intro hBA
                exact hAB hBA.symm
              by_cases hYB : Y = B
              · subst Y
                rw [hTakeBM]
                have hsZero :
                    countRecvs A sB = 0 :=
                  controlBroadcast_prefix_recv_count_zero
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B A
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    true sB hSBPref
                simpa [countRecvs, AlphabetOf.mkWhileTrue, Letter.isRecvFrom, hsZero, hBA]
              · by_cases hRecipY :
                    whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit Y ∧
                      U Y ≠ []
                · rcases while_body_recipient_prefix_cases
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      c B Y PBody PExit U V hYB hRecipY.1 sB rB hUBeq hSBPref
                      hSBTrace
                      (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
                  · exact False.elim (hRecipY.2 hYNil)
                  · have hTakeY :
                        ((U ∘ₘ V) Y).take (kWhile Y) =
                          [AlphabetOf.mkRecv (C := C) (F := F) Y
                            (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B] := by
                      simp [WordTuple.concat, kWhile, hYB, hRecipY, hYWord]
                    rw [hTakeY]
                    simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom, hBA]
                · simp [WordTuple.concat, kWhile, hYB, hRecipY]
            omega
        · intro A Y hSurplus
          by_cases hAB : A = B
          · subst A
            by_cases hYB : Y = B
            · subst Y
              exfalso
              have hSendZero :
                  countSends B (((U ∘ₘ V) B).take (kWhile B)) = 0 := by
                rw [hTakeBM]
                have hsZero :
                    countSends B sB = 0 :=
                  controlBroadcast_prefix_send_count_zero_nonRecipient
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    B B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    true sB hSBPref
                    (by
                      intro hSelf
                      exact (whileRecipients_no_self
                        (L := L) (C := C) (F := F) (Payload := Payload)
                        B PBody PExit B hSelf) rfl)
                simpa [countSends, AlphabetOf.mkWhileTrue, Letter.isSendTo, hsZero]
              omega
            · by_cases hRecipY :
                  whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit Y
              · by_cases hUY : U Y = []
                · have hVY : V Y = [] :=
                    while_recipient_nil_tail_nil
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      c B Y PBody PExit U V hYB hRecipY hSide hUY
                  simp [WordTuple.concat, kWhile, hYB, hRecipY, hUY, hVY, countRecvs]
                · exfalso
                  have hRecipY' :
                      whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit Y ∧
                        U Y ≠ [] := ⟨hRecipY, hUY⟩
                  rcases while_body_recipient_prefix_cases
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      c B Y PBody PExit U V hYB hRecipY sB rB hUBeq hSBPref
                      hSBTrace
                      (hPref Y) hMSC with hYNil | ⟨uY, _huY, hYWord⟩
                  · exact hUY hYNil
                  · have hRecvOne :
                        countRecvs B (((U ∘ₘ V) Y).take (kWhile Y)) = 1 := by
                      have hTakeY :
                          ((U ∘ₘ V) Y).take (kWhile Y) =
                            [AlphabetOf.mkRecv (C := C) (F := F) Y
                              (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B] := by
                        simp [WordTuple.concat, kWhile, hYB, hRecipY', hYWord]
                      rw [hTakeY]
                      simp [countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom]
                    have hSendLe :
                        countSends Y sB ≤ 1 :=
                      controlBroadcast_prefix_send_count_le_one_recipient
                        (L := L) (C := C) (F := F) (Payload := Payload)
                        B Y
                        (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                        true sB hSBPref (by intro Z hZ; exact hZ.1) hRecipY
                    have hSendEq :
                        countSends Y (((U ∘ₘ V) B).take (kWhile B)) = countSends Y sB := by
                      rw [hTakeBM]
                      simp [countSends, AlphabetOf.mkWhileTrue, Letter.isSendTo]
                    omega
              · exfalso
                have hSendZero :
                    countSends Y (((U ∘ₘ V) B).take (kWhile B)) = 0 := by
                  rw [hTakeBM]
                  have hsZero :
                      countSends Y sB = 0 :=
                    controlBroadcast_prefix_send_count_zero_nonRecipient
                      (L := L) (C := C) (F := F) (Payload := Payload)
                      B Y
                      (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                      true sB hSBPref hRecipY
                  simpa [countSends, AlphabetOf.mkWhileTrue, Letter.isSendTo, hsZero]
                omega
          · exfalso
            have hSendZero :
                countSends Y (((U ∘ₘ V) A).take (kWhile A)) = 0 := by
              by_cases hRecipA :
                  whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit A ∧
                    U A ≠ []
              · rcases while_body_recipient_prefix_cases
                    (L := L) (C := C) (F := F) (Payload := Payload)
                    c B A PBody PExit U V hAB hRecipA.1 sB rB hUBeq hSBPref
                    hSBTrace
                    (hPref A) hMSC with hANil | ⟨uA, _huA, hAWord⟩
                · exact False.elim (hRecipA.2 hANil)
                · have hTakeA :
                      ((U ∘ₘ V) A).take (kWhile A) =
                        [AlphabetOf.mkRecv (C := C) (F := F) A
                          (ControlPayload.setDecision (Payload := Payload) true ControlPayload.ctrlPattern) B] := by
                    simp [WordTuple.concat, kWhile, hAB, hRecipA, hAWord]
                  rw [hTakeA]
                  simp [countSends, AlphabetOf.mkRecv, Letter.isSendTo]
              · simp [WordTuple.concat, kWhile, hAB, hRecipA]
            omega
      have hSplit : ∀ X : L,
          ∃ u1 u2,
            R X = u1 ++ u2 ∧
            localPrefixSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody) u1 ∧
            localPrefixSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X
                (.whileLoop c B PBody PExit)) u2 ∧
            (u2 ++ V X ≠ [] →
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody) u1) ∧
            (V X ≠ [] →
              localTraceSemantics
                (L := L) (C := C) (F := F) (Payload := Payload)
                (project (L := L) (C := C) (F := F) (Payload := Payload) X
                  (.whileLoop c B PBody PExit)) u2) := by
        intro X
        exact localSeqBoundarySplit
          (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X
            (.whileLoop c B PBody PExit))
          (R X) (V X)
          (by simpa [project] using hRPref X)
          (by simpa [project] using hRSide X)
      let U1 : WordTuple L C F Payload := fun X => Classical.choose (hSplit X)
      let U2 : WordTuple L C F Payload := fun X => Classical.choose (Classical.choose_spec (hSplit X))
      have hReq : ∀ X, R X = U1 X ++ U2 X := by
        intro X
        exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).1
      have hU1Pref : ∀ X,
          localPrefixSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
            (U1 X) := by
        intro X
        exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.1
      have hU2Pref : ∀ X,
          localPrefixSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X
              (.whileLoop c B PBody PExit))
            (U2 X) := by
        intro X
        exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.2.1
      have hU1Side : ∀ X, (U2 X ++ V X) ≠ [] →
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X PBody)
            (U1 X) := by
        intro X
        exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.2.2.1
      have hU2Side : ∀ X, V X ≠ [] →
          localTraceSemantics
            (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X
              (.whileLoop c B PBody PExit))
            (U2 X) := by
        intro X
        exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.2.2.2
      have hConcat : R ∘ₘ V = U1 ∘ₘ (U2 ∘ₘ V) := by
        ext X
        simp [WordTuple.concat, hReq X, List.append_assoc]
      rcases hBody hWFBody U1 (U2 ∘ₘ V) hU1Pref
          (by
            intro X hXV
            simpa [WordTuple.concat] using hU1Side X hXV)
          (by simpa [hConcat] using hRMSC) with ⟨U1bar, hZip1⟩
      have hSuffixMSC : IsMSC (U2 ∘ₘ V) := zipPost_suffix_isMSC hZip1
      rcases hZip1 with ⟨hZip1Loc, hZip1Complete, hZip1Orig, hZip1MSC⟩
      have hLenU2 : (U2 B).length ≤ n := by
        have hRLen : (R B).length = rB.length := by rw [hRB]
        have hU2LeR : (U2 B).length ≤ (R B).length := by
          rw [hReq B]
          simp
        have hrBLeN : rB.length ≤ n := by
          have hLen' :
              (AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) B c ::
                sB ++ rB).length ≤ Nat.succ n := by
            simpa [hUBeq] using hLen
          simp at hLen'
          omega
        omega
      rcases ih U2 V ⟨hWFBody, hWFExit⟩ hLenU2 hU2Pref hU2Side
          hSuffixMSC with ⟨U2bar, hZip2⟩
      have hSuffixSeq : IsMSC V := zipPost_suffix_isMSC hZip2
      rcases hZip2 with ⟨hZip2Loc, hZip2Complete, hZip2Orig, hZip2MSC⟩
      have hUConcat : R = U1 ∘ₘ U2 := by
        ext X
        simp [WordTuple.concat, hReq X]
      have hZipSeq :
          ZipPost (L := L) (C := C) (F := F) (Payload := Payload)
            (PBody ;; .whileLoop c B PBody PExit) R (U1bar ∘ₘ U2bar) V := by
        simpa [hUConcat] using
          (zipPost_seq (L := L) (C := C) (F := F) (Payload := Payload)
            PBody (.whileLoop c B PBody PExit) U1 U1bar U2 U2bar V
            ⟨hZip1Loc, hZip1Complete, hZip1Orig, hZip1MSC⟩
            ⟨hZip2Loc, hZip2Complete, hZip2Orig, hZip2MSC⟩)
      have hSuffixSeq' : IsMSC V := zipPost_suffix_isMSC hZipSeq
      rcases hZipSeq with ⟨hLocSeq, hCompleteSeq, _hOrigSeq, _hMSCSeq⟩
      let Ubar :=
        mscWhileTrue (C := C) (F := F) (Payload := Payload) c B
          ∘ₘ controlBroadcastMSC (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                true
                (by intro X hX; exact hX.1)
          ∘ₘ U1bar ∘ₘ U2bar
      have hTraceUbar :
          ∀ X,
            localTraceSemantics
              (L := L) (C := C) (F := F) (Payload := Payload)
              (project (L := L) (C := C) (F := F) (Payload := Payload) X (.whileLoop c B PBody PExit))
              (Ubar X) := by
        have hDistBody :
            distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
              (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PBody) U1bar :=
          ⟨(fun X => (hZip1Loc X).2.1), hZip1Complete⟩
        have hDistWhile :
            distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
              (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                (.whileLoop c B PBody PExit)) U2bar :=
          ⟨(fun X => (hZip2Loc X).2.1), hZip2Complete⟩
        intro X
        simpa [Ubar, WordTuple.concat_assoc] using
          (distSemantics_project_while_body_step
            (L := L) (C := C) (F := F) (Payload := Payload)
            c B PBody PExit U1bar hDistBody U2bar hDistWhile).1 X
      have hCompleteUbar : IsCompleteMSC Ubar := by
        have hDistBody :
            distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
              (projectDist (L := L) (C := C) (F := F) (Payload := Payload) PBody) U1bar :=
          ⟨(fun X => (hZip1Loc X).2.1), hZip1Complete⟩
        have hDistWhile :
            distSemantics (L := L) (C := C) (F := F) (Payload := Payload)
              (projectDist (L := L) (C := C) (F := F) (Payload := Payload)
                (.whileLoop c B PBody PExit)) U2bar :=
          ⟨(fun X => (hZip2Loc X).2.1), hZip2Complete⟩
        simpa [Ubar, WordTuple.concat_assoc] using
          (distSemantics_project_while_body_step
            (L := L) (C := C) (F := F) (Payload := Payload)
            c B PBody PExit U1bar hDistBody U2bar hDistWhile).2
      refine ⟨Ubar, ?_⟩
      refine ⟨?_, hCompleteUbar, hMSC, concat_complete_msc _ _ hCompleteUbar hSuffixSeq'⟩
      intro X
      have hPrefixX :
          IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) (U X) (Ubar X) := by
        by_cases hXB : X = B
        · subst X
          rcases hLocSeq B with ⟨hRPrefB, _hTraceB, _hEqB⟩
          rcases hRPrefB with ⟨tR, htR⟩
          have hSeqbarEqB : U1bar B ++ U2bar B = rB ++ tR := by
            simpa [WordTuple.concat, hRB] using htR
          rcases localPrefixSemantics_controlBroadcast_prefix
              (L := L) (C := C) (F := F) (Payload := Payload)
              B
              (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
              true
              sB hSBPref with ⟨tS, htS⟩
          by_cases hrB : rB = []
          · subst hrB
            refine ⟨tS ++ tR, ?_⟩
            simp [Ubar, WordTuple.concat, mscWhileTrue, hUBeq, hRB, htS, hSeqbarEqB, List.append_assoc,
              choiceWhileTrue,
              controlBroadcastMSC_decider
                (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                true
                (by intro Y hY; exact hY.1)]
          · have hSBFull' :
                localTraceSemantics
                  (L := L) (C := C) (F := F) (Payload := Payload)
                  (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    true)
                  sB := hSBFull hrB
            have hSWord :
                sB =
                  controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                    B
                    (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                    true := by
              exact localTraceSemantics_controlBroadcast_eq
                (L := L) (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                true
                sB hSBFull'
            have hWordEq :
                controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload)
                  B
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                  true = sB := hSWord.symm
            refine ⟨tR, ?_⟩
            simp [Ubar, WordTuple.concat, mscWhileTrue, hUBeq, hRB, hWordEq, hSeqbarEqB, List.append_assoc,
              choiceWhileTrue,
              controlBroadcastMSC_decider
                (C := C) (F := F) (Payload := Payload)
                B
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                true
                (by intro Y hY; exact hY.1)]
        · by_cases hRecip :
              whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit X
          · rcases while_body_recipient_prefix_cases
                (L := L) (C := C) (F := F) (Payload := Payload)
                c B X PBody PExit U V hXB hRecip sB rB hUBeq hSBPref
                hSBTrace
                (hPref X) hMSC with hXNil | ⟨uX, _huX, hXWord⟩
            · refine ⟨Ubar X, by simp [hXNil, IsPrefixWord]⟩
            · rcases hLocSeq X with ⟨hRPrefX, _hTraceX, _hEqX⟩
              rcases hRPrefX with ⟨tR, htR⟩
              have hRX : R X = uX := by
                simp [R, kWhile, hXB, hRecip, hXWord]
              have hSeqbarEqX : U1bar X ++ U2bar X = uX ++ tR := by
                simpa [WordTuple.concat, hRX] using htR
              refine ⟨tR, ?_⟩
              simp [Ubar, WordTuple.concat, mscWhileTrue, hXB, hRecip, hXWord, hSeqbarEqX,
                List.append_assoc, controlDecisionPayload,
                controlBroadcastMSC_recipient
                  (C := C) (F := F) (Payload := Payload)
                  B X
                  (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit)
                  true
                  (by intro Y hY; exact hY.1)
                  hRecip]
          · have hXNil : U X = [] :=
              localPrefixSemantics_eps_eq_nil
                (L := L) (C := C) (F := F) (Payload := Payload)
                (A := X) (U X)
                (by simpa [project, hXB, hRecip] using hPref X)
            refine ⟨Ubar X, by simp [hXNil, IsPrefixWord]⟩
      refine ⟨hPrefixX, hTraceUbar X, ?_⟩
      intro hVX
      exact localTraceSemantics_prefix_eq
        (L := L) (C := C) (F := F) (Payload := Payload)
        (A := X)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X (.whileLoop c B PBody PExit))
        (u := U X)
        (v := Ubar X)
        (hSide X hVX)
        (hTraceUbar X)
        hPrefixX |>.symm

/-- The full uniform zipper theorem, obtained by structural induction from the
    constructor cases above. -/
theorem uniformZipper
    (prog : Prog L C F Payload) :
    UniformZipperProperty (L := L) (C := C) (F := F) (Payload := Payload) prog := by
  induction prog with
  | eps =>
      exact uniformZipper_eps
  | msg A xs B ys h =>
      exact uniformZipper_msg A xs B ys h
  | act A ys f xs =>
      exact uniformZipper_act A ys f xs
  | seq P1 P2 ih1 ih2 =>
      exact uniformZipper_seq P1 P2 ih1 ih2
  | ite c B PTrue PFalse ihTrue ihFalse =>
      exact uniformZipper_if c B PTrue PFalse ihTrue ihFalse
  | whileLoop c B PBody PExit ihBody ihExit =>
      exact uniformZipper_while c B PBody PExit ihBody ihExit

end ZipperLemma
