/-
  MSCAgents/ZipperPost.lean
  =========================
  Formalization of Definition 13 (def:zip-post) from sec:projection of:
  "Provable Coordination for LLM Agents via Message Sequence Charts"
-/

import MSCAgents.LocalSemantics

section ZipperPost

variable {L C F Payload : Type} [DecidableEq L] [Fintype L]
variable [PayloadCompatiblePred Payload] [ControlPayload Payload]

private def IsPrefixList {α : Type} (xs ys : List α) : Prop :=
  ∃ t, ys = xs ++ t

private theorem countSends_prefix_le {A : L} (B : L)
    {u u' : LocalWord (C := C) (F := F) (Payload := Payload) A}
    (h : IsPrefixWord u u') :
    countSends B u ≤ countSends B u' := by
  rcases h with ⟨t, rfl⟩
  simp [countSends_append]

private theorem countRecvs_prefix_le {B : L} (A : L)
    {u u' : LocalWord (C := C) (F := F) (Payload := Payload) B}
    (h : IsPrefixWord u u') :
    countRecvs A u ≤ countRecvs A u' := by
  rcases h with ⟨t, rfl⟩
  simp [countRecvs_append]

private theorem sendPayloads_prefix {A : L} (B : L)
    {u u' : LocalWord (C := C) (F := F) (Payload := Payload) A}
    (h : IsPrefixWord u u') :
    IsPrefixList
      (sendPayloads (C := C) (F := F) (Payload := Payload) B u)
      (sendPayloads (C := C) (F := F) (Payload := Payload) B u') := by
  rcases h with ⟨t, rfl⟩
  refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) B t, ?_⟩
  simp [IsPrefixList]

private theorem recvPayloads_prefix {B : L} (A : L)
    {u u' : LocalWord (C := C) (F := F) (Payload := Payload) B}
    (h : IsPrefixWord u u') :
    IsPrefixList
      (recvPayloads (C := C) (F := F) (Payload := Payload) A u)
      (recvPayloads (C := C) (F := F) (Payload := Payload) A u') := by
  rcases h with ⟨t, rfl⟩
  refine ⟨recvPayloads (C := C) (F := F) (Payload := Payload) A t, ?_⟩
  simp [IsPrefixList]

private theorem zip_prefix_of_prefix {α β : Type}
    (xs xs' : List α) (ys ys' : List β)
    (hxs : IsPrefixList xs xs') (hys : IsPrefixList ys ys') :
    IsPrefixList (List.zip xs ys) (List.zip xs' ys') := by
  rcases hxs with ⟨tx, rfl⟩
  rcases hys with ⟨ty, rfl⟩
  induction xs generalizing ys tx ty with
  | nil =>
      refine ⟨List.zip tx (ys ++ ty), ?_⟩
      simp [IsPrefixList]
  | cons x xs ih =>
      cases ys with
      | nil =>
          refine ⟨List.zip ((x :: xs) ++ tx) ty, ?_⟩
          simp
      | cons y ys =>
          rcases ih ys tx ty with ⟨t, ht⟩
          refine ⟨t, ?_⟩
          simp [ht]

private theorem mem_zip_of_prefix {α β : Type}
    {xs xs' : List α} {ys ys' : List β} {p : α × β}
    (hxs : IsPrefixList xs xs') (hys : IsPrefixList ys ys')
    (hp : p ∈ List.zip xs ys) :
    p ∈ List.zip xs' ys' := by
  rcases zip_prefix_of_prefix xs xs' ys ys' hxs hys with ⟨t, ht⟩
  rw [ht]
  exact List.mem_append.mpr (Or.inl hp)

private def leftPrefixRanking
    (U V : WordTuple L C F Payload)
    (R : CausalRanking (U ∘ₘ V)) :
    CausalRanking U where
  rank := R.rank
  local_mono := by
    intro e1 e2 hSameLL hLt hpos1 hpos2
    exact R.local_mono e1 e2 hSameLL hLt
      (by
        simp [WordTuple.concat]
        omega)
      (by
        simp [WordTuple.concat]
        omega)
  fifo_mono := by
    intro A B j1 j2 hj1 hj2 hSend hRecv hCntEq
    have hj1' : j1 < ((U ∘ₘ V) A).length := by
      simp [WordTuple.concat]
      omega
    have hj2' : j2 < ((U ∘ₘ V) B).length := by
      simp [WordTuple.concat]
      omega
    have hSend' :
        (((U ∘ₘ V) A).get ⟨j1, hj1'⟩).val.isSendTo B = true := by
      simpa [WordTuple.concat, List.getElem_append_left hj1] using hSend
    have hRecv' :
        (((U ∘ₘ V) B).get ⟨j2, hj2'⟩).val.isRecvFrom A = true := by
      simpa [WordTuple.concat, List.getElem_append_left hj2] using hRecv
    have hCntEq' :
        countSends B (((U ∘ₘ V) A).take j1) =
          countRecvs A (((U ∘ₘ V) B).take j2) := by
      rw [WordTuple.concat, WordTuple.concat]
      rw [List.take_append_of_le_length (Nat.le_of_lt hj1)]
      rw [List.take_append_of_le_length (Nat.le_of_lt hj2)]
      simpa using hCntEq
    exact R.fifo_mono A B j1 j2 hj1' hj2' hSend' hRecv' hCntEq'

/-- The zipper postcondition from Definition `def:zip-post`. -/
def ZipPost
    (P : Prog L C F Payload)
    (U Ubar V : WordTuple L C F Payload) : Prop :=
  (∀ X,
      IsPrefixWord (U X) (Ubar X) ∧
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X P)
        (Ubar X) ∧
      (V X ≠ [] → Ubar X = U X)) ∧
  IsCompleteMSC Ubar ∧
  IsMSC (U ∘ₘ V) ∧
  IsMSC (Ubar ∘ₘ V)

/-- Trivial zipper postcondition when the given left tuple is already a
    complete realization. -/
theorem zipPost_of_complete
    (P : Prog L C F Payload)
    (U V : WordTuple L C F Payload)
    (hTrace : ∀ X,
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X P)
        (U X))
    (hComplete : IsCompleteMSC U)
    (hMSC : IsMSC (U ∘ₘ V)) :
    ZipPost (L := L) (C := C) (F := F) (Payload := Payload) P U U V := by
  refine ⟨?_, hComplete, hMSC, hMSC⟩
  intro X
  refine ⟨⟨[], by simp [IsPrefixWord]⟩, hTrace X, ?_⟩
  intro _
  rfl

/-- Sequential composition of zipper postconditions. -/
theorem zipPost_seq
    (Q1 Q2 : Prog L C F Payload)
    (U1 U1bar U2 U2bar V : WordTuple L C F Payload)
    (h1 :
      ZipPost (L := L) (C := C) (F := F) (Payload := Payload)
        Q1 U1 U1bar (U2 ∘ₘ V))
    (h2 :
      ZipPost (L := L) (C := C) (F := F) (Payload := Payload)
        Q2 U2 U2bar V) :
    ZipPost (L := L) (C := C) (F := F) (Payload := Payload)
      (Q1 ;; Q2) (U1 ∘ₘ U2) (U1bar ∘ₘ U2bar) V := by
  rcases h1 with ⟨h1loc, h1complete, h1orig, _h1msc⟩
  rcases h2 with ⟨h2loc, h2complete, _h2orig, h2msc⟩
  refine ⟨?_, concat_complete_complete _ _ h1complete h2complete, ?_, ?_⟩
  · intro X
    rcases h1loc X with ⟨h1pref, h1trace, h1eq⟩
    rcases h2loc X with ⟨h2pref, h2trace, h2eq⟩
    refine ⟨?_, ?_, ?_⟩
    · by_cases hTail : U2 X ++ V X = []
      · rcases h1pref with ⟨t1, ht1⟩
        refine ⟨t1 ++ U2bar X, ?_⟩
        have hU2Nil : U2 X = [] := (List.eq_nil_of_append_eq_nil hTail).1
        simp [WordTuple.concat, ht1, hU2Nil, List.append_assoc]
      · have h1eq' : U1bar X = U1 X := h1eq hTail
        rcases h2pref with ⟨t2, ht2⟩
        refine ⟨t2, ?_⟩
        simp [WordTuple.concat, h1eq', ht2, List.append_assoc]
    · exact localTraceSemantics_seq_intro
        (project (L := L) (C := C) (F := F) (Payload := Payload) X Q1)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X Q2)
        (U1bar X) (U2bar X) h1trace h2trace
    · intro hVX
      have h1eq' : U1bar X = U1 X := by
        apply h1eq
        simp [WordTuple.concat, hVX]
      have h2eq' : U2bar X = U2 X := h2eq hVX
      simp [WordTuple.concat, h1eq', h2eq']
  · simpa [WordTuple.concat_assoc] using h1orig
  · simpa [WordTuple.concat_assoc] using
      (concat_complete_msc _ _ h1complete h2msc)

/-- Stronger sequential composition, where the first zipper postcondition is
    already expressed against the completed second prefix. -/
theorem zipPost_seq_strong
    (Q1 Q2 : Prog L C F Payload)
    (U1 U1bar U2 U2bar V : WordTuple L C F Payload)
    (hOrig : IsMSC (U1 ∘ₘ (U2 ∘ₘ V)))
    (h1 :
      ZipPost (L := L) (C := C) (F := F) (Payload := Payload)
        Q1 U1 U1bar (U2bar ∘ₘ V))
    (h2 :
      ZipPost (L := L) (C := C) (F := F) (Payload := Payload)
        Q2 U2 U2bar V) :
    ZipPost (L := L) (C := C) (F := F) (Payload := Payload)
      (Q1 ;; Q2) (U1 ∘ₘ U2) (U1bar ∘ₘ U2bar) V := by
  rcases h1 with ⟨h1loc, h1complete, _h1orig, h1msc⟩
  rcases h2 with ⟨h2loc, h2complete, _h2orig, _h2msc⟩
  refine ⟨?_, concat_complete_complete _ _ h1complete h2complete, ?_, ?_⟩
  · intro X
    rcases h1loc X with ⟨h1pref, h1trace, h1eq⟩
    rcases h2loc X with ⟨h2pref, h2trace, h2eq⟩
    refine ⟨?_, ?_, ?_⟩
    · by_cases hTail : U2bar X ++ V X = []
      · have hU2barNil : U2bar X = [] := by
          exact (List.eq_nil_of_append_eq_nil hTail).1
        have hU2Nil : U2 X = [] := by
          rcases h2pref with ⟨t, ht⟩
          rw [hU2barNil] at ht
          exact (List.eq_nil_of_append_eq_nil ht.symm).1
        simpa [WordTuple.concat, hU2barNil, hU2Nil] using h1pref
      · have h1eq' : U1bar X = U1 X := h1eq hTail
        rcases h2pref with ⟨t, ht⟩
        refine ⟨t, ?_⟩
        simp [WordTuple.concat, h1eq', ht, List.append_assoc]
    · exact localTraceSemantics_seq_intro
        (project (L := L) (C := C) (F := F) (Payload := Payload) X Q1)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X Q2)
        (U1bar X) (U2bar X) h1trace h2trace
    · intro hVX
      have h1eq' : U1bar X = U1 X := by
        apply h1eq
        simp [WordTuple.concat, hVX]
      have h2eq' : U2bar X = U2 X := h2eq hVX
      simp [WordTuple.concat, h1eq', h2eq']
  · simpa [WordTuple.concat_assoc] using hOrig
  · simpa [WordTuple.concat_assoc] using h1msc

/-- The original cut tuple carried by a zipper postcondition is an MSC. -/
theorem zipPost_orig_msc
    {P : Prog L C F Payload}
    {U Ubar V : WordTuple L C F Payload}
    (h :
      ZipPost (L := L) (C := C) (F := F) (Payload := Payload) P U Ubar V) :
    IsMSC (U ∘ₘ V) :=
  h.2.2.1

/-- The suffix tuple is an MSC, derivable from ZipPost's complete prefix
    condition and completed-cut MSC condition. -/
theorem zipPost_suffix_isMSC
    {P : Prog L C F Payload}
    {U Ubar V : WordTuple L C F Payload}
    (h :
      ZipPost (L := L) (C := C) (F := F) (Payload := Payload) P U Ubar V) :
    IsMSC V :=
  suffix_msc_of_complete_prefix Ubar V h.2.1 h.2.2.2

/-- **Remark rem:U-is-MSC**:
    every left tuple carried by a zipper postcondition is itself an MSC. -/
theorem zipPost_left_isMSC
    {P : Prog L C F Payload}
    {U Ubar V : WordTuple L C F Payload}
    (h :
      ZipPost (L := L) (C := C) (F := F) (Payload := Payload) P U Ubar V) :
    IsMSC U := by
  rcases h with ⟨hLoc, hComplete, hOrig, _hMSC⟩
  refine ⟨?_, ?_, ?_⟩
  · intro A B
    rcases hLoc A with ⟨hPrefA, _hTraceA, hEqA⟩
    rcases hLoc B with ⟨hPrefB, _hTraceB, hEqB⟩
    by_cases hVA : V A = []
    · by_cases hVB : V B = []
      · simpa [sndCount, rcvCount, WordTuple.concat, hVA, hVB] using
          hOrig.noUnmatchedRecv A B
      · have hOrigAB := hOrig.noUnmatchedRecv A B
        have hOrigAB' :
            countRecvs A (U B) + countRecvs A (V B) ≤ countSends B (U A) := by
          simpa [sndCount, rcvCount, WordTuple.concat, hVA,
            countSends_append, countRecvs_append] using hOrigAB
        calc
          rcvCount U A B = countRecvs A (U B) := rfl
          _ ≤ countRecvs A (U B) + countRecvs A (V B) := by omega
          _ ≤ countSends B (U A) := hOrigAB'
          _ = sndCount U A B := rfl
    · by_cases hVB : V B = []
      · have hAeq : Ubar A = U A := hEqA hVA
        have hBmono :
            countRecvs A (U B) ≤ countRecvs A (Ubar B) :=
          countRecvs_prefix_le (L := L) (C := C) (F := F) (Payload := Payload)
            A hPrefB
        have hCompleteAB :
            countRecvs A (Ubar B) = countSends B (Ubar A) := by
          simpa [sndCount, rcvCount] using (hComplete.complete A B).symm
        calc
          rcvCount U A B = countRecvs A (U B) := rfl
          _ ≤ countRecvs A (Ubar B) := hBmono
          _ = countSends B (Ubar A) := hCompleteAB
          _ = countSends B (U A) := by simpa [hAeq]
          _ = sndCount U A B := rfl
      · have hAeq : Ubar A = U A := hEqA hVA
        have hBeq : Ubar B = U B := hEqB hVB
        have hCompleteAB :
            countRecvs A (U B) = countSends B (U A) := by
          simpa [sndCount, rcvCount, hAeq, hBeq] using (hComplete.complete A B).symm
        rw [show rcvCount U A B = countRecvs A (U B) by rfl]
        rw [hCompleteAB]
        simp [sndCount]
  · intro A B p hp
    have hSendPref :
        IsPrefixList
          (sendPayloads (C := C) (F := F) (Payload := Payload) B (U A))
          (sendPayloads (C := C) (F := F) (Payload := Payload) B ((U ∘ₘ V) A)) := by
      refine ⟨sendPayloads (C := C) (F := F) (Payload := Payload) B (V A), ?_⟩
      simp [IsPrefixList, WordTuple.concat]
    have hRecvPref :
        IsPrefixList
          (recvPayloads (C := C) (F := F) (Payload := Payload) A (U B))
          (recvPayloads (C := C) (F := F) (Payload := Payload) A ((U ∘ₘ V) B)) := by
      refine ⟨recvPayloads (C := C) (F := F) (Payload := Payload) A (V B), ?_⟩
      simp [IsPrefixList, WordTuple.concat]
    have hp' :
        p ∈
          List.zip
            (sendPayloads (C := C) (F := F) (Payload := Payload) B ((U ∘ₘ V) A))
            (recvPayloads (C := C) (F := F) (Payload := Payload) A ((U ∘ₘ V) B)) :=
      mem_zip_of_prefix hSendPref hRecvPref hp
    exact hOrig.labelCompat A B p hp'
  · exact ⟨leftPrefixRanking U V (Classical.choice hOrig.acyclic)⟩

end ZipperPost

section ZipperPostComplete

variable {L C F Payload : Type} [DecidableEq L] [Fintype L]
variable [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]

/-- **Corollary cor:U-complete**:
    if the cut tuple itself already realizes complete local traces, then it is
    a complete MSC. -/
theorem zipPost_left_isCompleteMSC
    {P : Prog L C F Payload}
    {U Ubar V : WordTuple L C F Payload}
    (h :
      ZipPost (L := L) (C := C) (F := F) (Payload := Payload) P U Ubar V)
    (hTrace : ∀ X,
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X P)
        (U X)) :
    IsCompleteMSC U := by
  rcases h with ⟨hLoc, hComplete, _hOrig, _hMSC⟩
  have hEq : U = Ubar := by
    funext X
    rcases hLoc X with ⟨hPref, hTraceBar, _hCutEq⟩
    exact localTraceSemantics_prefix_eq
      (L := L) (C := C) (F := F) (Payload := Payload)
      (project (L := L) (C := C) (F := F) (Payload := Payload) X P)
      (hTrace X) hTraceBar hPref
  simpa [hEq] using hComplete

end ZipperPostComplete
