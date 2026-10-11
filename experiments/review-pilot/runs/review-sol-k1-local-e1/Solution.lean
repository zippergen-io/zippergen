import CompletionGoal

open SupplierAccess

namespace Direct

def agentWord (b : Bool) : Word Role.agent :=
  [AlphabetOf.mkSend Role.agent Port.plan Role.worker (by decide),
   if b then AlphabetOf.mkIfTrue Role.agent Guard.need1 else AlphabetOf.mkIfFalse Role.agent Guard.need1,
   AlphabetOf.mkSend Role.agent (Port.ctrl b) Role.finalizer (by decide),
   AlphabetOf.mkSend Role.agent (Port.ctrl b) Role.reviewer1 (by decide)] ++
  if b then [AlphabetOf.mkSend Role.agent Port.review Role.reviewer1 (by decide)] else []

def workerWord : Word Role.worker :=
  [AlphabetOf.mkRecv Role.worker Port.plan Role.agent,
   AlphabetOf.mkAct Role.worker Port.prepared Action.prepare Port.plan,
   AlphabetOf.mkSend Role.worker Port.prepared Role.finalizer (by decide)]

def finalizerWord (b : Bool) : Word Role.finalizer :=
  [AlphabetOf.mkRecv Role.finalizer (Port.ctrl b) Role.agent] ++
  (if b then [AlphabetOf.mkRecv Role.finalizer Port.verdict Role.reviewer1] else []) ++
  [AlphabetOf.mkRecv Role.finalizer Port.prepared Role.worker,
   AlphabetOf.mkAct Role.finalizer Port.unit Action.resolve Port.prepared]

def reviewerWord (b : Bool) : Word Role.reviewer1 :=
  [AlphabetOf.mkRecv Role.reviewer1 (Port.ctrl b) Role.agent] ++
  if b then
    [AlphabetOf.mkRecv Role.reviewer1 Port.review Role.agent,
     AlphabetOf.mkAct Role.reviewer1 Port.unit Action.assess Port.review,
     AlphabetOf.mkSend Role.reviewer1 Port.verdict Role.finalizer (by decide)]
  else []

def full (b : Bool) : Tuple
  | Role.agent => agentWord b
  | Role.worker => workerWord
  | Role.finalizer => finalizerWord b
  | Role.reviewer1 => reviewerWord b

private theorem trace_agent {w : Word Role.agent}
    (h : localTraceSemantics (endpoints Role.agent) w) :
    w = agentWord true ∨ w = agentWord false := by
  simp only [endpoints, localTraceSemantics] at h
  simp_all [agentWord]

private theorem trace_worker {w : Word Role.worker}
    (h : localTraceSemantics (endpoints Role.worker) w) :
    w = workerWord := by
  simp_all [endpoints, localTraceSemantics, workerWord]

private theorem trace_finalizer {w : Word Role.finalizer}
    (h : localTraceSemantics (endpoints Role.finalizer) w) :
    w = finalizerWord true ∨ w = finalizerWord false := by
  simp_all [endpoints, localTraceSemantics, finalizerWord, ControlPayload.setDecision]

private theorem trace_reviewer {w : Word Role.reviewer1}
    (h : localTraceSemantics (endpoints Role.reviewer1) w) :
    w = reviewerWord true ∨ w = reviewerWord false := by
  simp_all [endpoints, localTraceSemantics, reviewerWord, ControlPayload.setDecision]

private theorem full_traces (b : Bool) (a : Role) :
    localTraceSemantics (endpoints a) (full b a) := by
  cases b <;> cases a <;>
    simp [endpoints, full, agentWord, workerWord, finalizerWord,
      reviewerWord, localTraceSemantics, ControlPayload.setDecision]

private theorem full_balanced (b : Bool) : tupleComplete (full b) := by
  intro A B
  cases b <;> cases A <;> cases B <;> rfl

private theorem full_labels (b : Bool) : matchedLabelsCompatible (full b) := by
  intro A B
  cases b <;> cases A <;> cases B <;>
    simp [channelLabelCompatible, full, agentWord, workerWord, finalizerWord, reviewerWord,
      sendPayloads, recvPayloads, AlphabetOf.mkSend, AlphabetOf.mkRecv,
      AlphabetOf.mkAct, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse,
      PayloadCompatible, PayloadCompatiblePred.compat, compatible]

def rank (b : Bool) (e : Event Role) : Nat :=
  match e.lifeline with
  | Role.agent => e.pos
  | Role.worker => e.pos + 1
  | Role.reviewer1 => e.pos + 4
  | Role.finalizer => if e.pos = 0 then 3 else e.pos + (if b then 7 else 8)

def prefixTuple (ba bf br : Bool) (ia : Fin 6) (iw : Fin 4)
    (iff : Fin 5) (ir : Fin 5) : Tuple
  | Role.agent => (agentWord ba).take ia.val
  | Role.worker => workerWord.take iw.val
  | Role.finalizer => (finalizerWord bf).take iff.val
  | Role.reviewer1 => (reviewerWord br).take ir.val

def validCounts (M : Tuple) : Prop :=
  ∀ A B : Role, rcvCount M A B ≤ sndCount M A B

def validLabels (M : Tuple) : Prop :=
  ∀ A B : Role, ∀ p ∈ List.zip (sendPayloads B (M A)) (recvPayloads A (M B)),
    p.1 = p.2

def aligned (M : Tuple) (b : Bool) : Prop :=
  ∀ a : Role, (full b a).take (M a).length = M a

private theorem finite_alignment :
    ∀ ba bf br (ia : Fin 6) (iw : Fin 4) (iff : Fin 5) (ir : Fin 5),
      validCounts (prefixTuple ba bf br ia iw iff ir) →
      validLabels (prefixTuple ba bf br ia iw iff ir) →
      aligned (prefixTuple ba bf br ia iw iff ir) true ∨
        aligned (prefixTuple ba bf br ia iw iff ir) false := by
  unfold validCounts validLabels aligned
  decide

private theorem prefix_take {α : Type} {u v : List α}
    (h : ∃ t, v = u ++ t) : v.take u.length = u := by
  rcases h with ⟨t, rfl⟩
  simp

private theorem prefix_length {α : Type} {u v : List α}
    (h : ∃ t, v = u ++ t) : u.length ≤ v.length := by
  rcases h with ⟨t, rfl⟩
  simp

private theorem choose_prefix (M : Tuple)
    (hlocal : ∀ a, localPrefixSemantics (endpoints a) (M a)) :
    ∃ ba bf br, ∃ (ia : Fin 6) (iw : Fin 4) (iff : Fin 5) (ir : Fin 5),
      M = prefixTuple ba bf br ia iw iff ir := by
  obtain ⟨va, hva, hpa⟩ := hlocal Role.agent
  obtain ⟨vw, hvw, hpw⟩ := hlocal Role.worker
  obtain ⟨vf, hvf, hpf⟩ := hlocal Role.finalizer
  obtain ⟨vr, hvr, hpr⟩ := hlocal Role.reviewer1
  obtain ⟨ba, ha⟩ : ∃ ba, va = agentWord ba := by
    rcases trace_agent hva with h | h
    · exact ⟨true, h⟩
    · exact ⟨false, h⟩
  obtain ⟨bf, hf⟩ : ∃ bf, vf = finalizerWord bf := by
    rcases trace_finalizer hvf with h | h
    · exact ⟨true, h⟩
    · exact ⟨false, h⟩
  obtain ⟨br, hr⟩ : ∃ br, vr = reviewerWord br := by
    rcases trace_reviewer hvr with h | h
    · exact ⟨true, h⟩
    · exact ⟨false, h⟩
  have hw : vw = workerWord := trace_worker hvw
  have hla : (M Role.agent).length < 6 := by
    have := prefix_length hpa
    rw [ha] at this
    cases ba <;> simp [agentWord] at this <;> omega
  have hlw : (M Role.worker).length < 4 := by
    have := prefix_length hpw
    rw [hw] at this
    simp [workerWord] at this
    omega
  have hlf : (M Role.finalizer).length < 5 := by
    have := prefix_length hpf
    rw [hf] at this
    cases bf <;> simp [finalizerWord] at this <;> omega
  have hlr : (M Role.reviewer1).length < 5 := by
    have := prefix_length hpr
    rw [hr] at this
    cases br <;> simp [reviewerWord] at this <;> omega
  refine ⟨ba, bf, br, ⟨_, hla⟩, ⟨_, hlw⟩, ⟨_, hlf⟩, ⟨_, hlr⟩, ?_⟩
  funext a
  cases a
  · simpa [prefixTuple, ← ha] using (prefix_take hpa).symm
  · simpa [prefixTuple, ← hw] using (prefix_take hpw).symm
  · simpa [prefixTuple, ← hf] using (prefix_take hpf).symm
  · simpa [prefixTuple, ← hr] using (prefix_take hpr).symm

private theorem finite_rank :
    ∀ b A B (i j : Fin 5)
      (hi : i.val < (full b A).length) (hj : j.val < (full b B).length),
      ((full b A).get ⟨i.val, hi⟩).val.isSendTo B = true →
      ((full b B).get ⟨j.val, hj⟩).val.isRecvFrom A = true →
      countSends B ((full b A).take i.val) =
        countRecvs A ((full b B).take j.val) →
      rank b ⟨A, i.val⟩ < rank b ⟨B, j.val⟩ := by
  decide

private theorem full_rank (b : Bool) : hasAcyclicCausality (full b) := by
  refine ⟨{ rank := rank b, local_mono := ?_, fifo_mono := ?_ }⟩
  · intro e1 e2 hrole hpos h1 h2
    cases e1 with
    | mk a i =>
      cases e2 with
      | mk c j =>
        dsimp at hrole hpos h1 h2 ⊢
        subst c
        cases b <;> cases a <;>
          by_cases hi : i = 0 <;> by_cases hj : j = 0 <;>
          simp [rank, hi, hj] <;> omega
  · intro A B j1 j2 hj1 hj2 hs hr hc
    have hsmall1 : j1 < 5 := by
      cases b <;> cases A <;>
        simp [full, agentWord, workerWord, finalizerWord, reviewerWord] at hj1 <;>
        omega
    have hsmall2 : j2 < 5 := by
      cases b <;> cases B <;>
        simp [full, agentWord, workerWord, finalizerWord, reviewerWord] at hj2 <;>
        omega
    exact finite_rank b A B ⟨j1, hsmall1⟩ ⟨j2, hsmall2⟩ hj1 hj2 hs hr hc

private theorem full_complete (b : Bool) : distSemantics endpoints (full b) := by
  refine ⟨full_traces b, ?_⟩
  exact ⟨full_balanced b, full_labels b, full_rank b⟩

private theorem counts_of_msc {M : Tuple} (h : IsMSC M) : validCounts M := by
  exact h.noUnmatchedRecv

private theorem labels_of_msc {M : Tuple} (h : IsMSC M) : validLabels M := by
  intro A B p hp
  have hcompat := h.labelCompat A B p hp
  simpa [PayloadCompatible, PayloadCompatiblePred.compat, compatible] using hcompat

private theorem aligned_prefix (M : Tuple) (b : Bool)
    (h : aligned M b) : IsPrefixTuple M (full b) := by
  intro a
  refine ⟨(full b a).drop (M a).length, ?_⟩
  have hs := List.take_append_drop (M a).length (full b a)
  rw [h a] at hs
  exact hs.symm

end Direct

theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hM
  obtain ⟨ba, bf, br, ia, iw, iff, ir, hEq⟩ :=
    Direct.choose_prefix M hM.1
  have hc : Direct.validCounts (Direct.prefixTuple ba bf br ia iw iff ir) := by
    rw [← hEq]
    exact Direct.counts_of_msc hM.2
  have hl : Direct.validLabels (Direct.prefixTuple ba bf br ia iw iff ir) := by
    rw [← hEq]
    exact Direct.labels_of_msc hM.2
  rcases Direct.finite_alignment ba bf br ia iw iff ir hc hl with ht | hf
  · refine ⟨Direct.full true, Direct.full_complete true, ?_⟩
    apply Direct.aligned_prefix
    rw [hEq]
    exact ht
  · refine ⟨Direct.full false, Direct.full_complete false, ?_⟩
    apply Direct.aligned_prefix
    rw [hEq]
    exact hf
