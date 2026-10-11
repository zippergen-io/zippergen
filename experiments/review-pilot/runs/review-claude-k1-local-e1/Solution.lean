import CompletionGoal

namespace ReviewSol
open SupplierAccess

def wAT : Word Role.agent :=
  [AlphabetOf.mkSend (C := Guard) (F := Action) (Payload := Port) Role.agent Port.plan Role.worker (by decide),
   AlphabetOf.mkIfTrue (C := Guard) (F := Action) (Payload := Port) Role.agent Guard.need1,
   AlphabetOf.mkSend (C := Guard) (F := Action) (Payload := Port) Role.agent (Port.ctrl true) Role.finalizer (by decide),
   AlphabetOf.mkSend (C := Guard) (F := Action) (Payload := Port) Role.agent (Port.ctrl true) Role.reviewer1 (by decide),
   AlphabetOf.mkSend (C := Guard) (F := Action) (Payload := Port) Role.agent Port.review Role.reviewer1 (by decide)]
def wAF : Word Role.agent :=
  [AlphabetOf.mkSend (C := Guard) (F := Action) (Payload := Port) Role.agent Port.plan Role.worker (by decide),
   AlphabetOf.mkIfFalse (C := Guard) (F := Action) (Payload := Port) Role.agent Guard.need1,
   AlphabetOf.mkSend (C := Guard) (F := Action) (Payload := Port) Role.agent (Port.ctrl false) Role.finalizer (by decide),
   AlphabetOf.mkSend (C := Guard) (F := Action) (Payload := Port) Role.agent (Port.ctrl false) Role.reviewer1 (by decide)]
def wW : Word Role.worker :=
  [AlphabetOf.mkRecv (C := Guard) (F := Action) (Payload := Port) Role.worker Port.plan Role.agent,
   AlphabetOf.mkAct (C := Guard) (F := Action) (Payload := Port) Role.worker Port.prepared Action.prepare Port.plan,
   AlphabetOf.mkSend (C := Guard) (F := Action) (Payload := Port) Role.worker Port.prepared Role.finalizer (by decide)]
def wFT : Word Role.finalizer :=
  [AlphabetOf.mkRecv (C := Guard) (F := Action) (Payload := Port) Role.finalizer (Port.ctrl true) Role.agent,
   AlphabetOf.mkRecv (C := Guard) (F := Action) (Payload := Port) Role.finalizer Port.verdict Role.reviewer1,
   AlphabetOf.mkRecv (C := Guard) (F := Action) (Payload := Port) Role.finalizer Port.prepared Role.worker,
   AlphabetOf.mkAct (C := Guard) (F := Action) (Payload := Port) Role.finalizer Port.unit Action.resolve Port.prepared]
def wFF : Word Role.finalizer :=
  [AlphabetOf.mkRecv (C := Guard) (F := Action) (Payload := Port) Role.finalizer (Port.ctrl false) Role.agent,
   AlphabetOf.mkRecv (C := Guard) (F := Action) (Payload := Port) Role.finalizer Port.prepared Role.worker,
   AlphabetOf.mkAct (C := Guard) (F := Action) (Payload := Port) Role.finalizer Port.unit Action.resolve Port.prepared]
def wRT : Word Role.reviewer1 :=
  [AlphabetOf.mkRecv (C := Guard) (F := Action) (Payload := Port) Role.reviewer1 (Port.ctrl true) Role.agent,
   AlphabetOf.mkRecv (C := Guard) (F := Action) (Payload := Port) Role.reviewer1 Port.review Role.agent,
   AlphabetOf.mkAct (C := Guard) (F := Action) (Payload := Port) Role.reviewer1 Port.unit Action.assess Port.review,
   AlphabetOf.mkSend (C := Guard) (F := Action) (Payload := Port) Role.reviewer1 Port.verdict Role.finalizer (by decide)]
def wRF : Word Role.reviewer1 :=
  [AlphabetOf.mkRecv (C := Guard) (F := Action) (Payload := Port) Role.reviewer1 (Port.ctrl false) Role.agent]

def NT : Tuple
  | Role.agent => wAT
  | Role.worker => wW
  | Role.finalizer => wFT
  | Role.reviewer1 => wRT
def NF : Tuple
  | Role.agent => wAF
  | Role.worker => wW
  | Role.finalizer => wFF
  | Role.reviewer1 => wRF

theorem NT_eq : NT =
    mscMsg Role.agent Port.plan Role.worker Port.plan (by decide) ∘ₘ
    mscAct Role.worker Port.prepared Action.prepare Port.plan ∘ₘ
    mscIfTrue Guard.need1 Role.agent ∘ₘ
    mscMsg Role.agent (Port.ctrl true) Role.finalizer (Port.ctrl true) (by decide) ∘ₘ
    mscMsg Role.agent (Port.ctrl true) Role.reviewer1 (Port.ctrl true) (by decide) ∘ₘ
    mscMsg Role.agent Port.review Role.reviewer1 Port.review (by decide) ∘ₘ
    mscAct Role.reviewer1 Port.unit Action.assess Port.review ∘ₘ
    mscMsg Role.reviewer1 Port.verdict Role.finalizer Port.verdict (by decide) ∘ₘ
    mscMsg Role.worker Port.prepared Role.finalizer Port.prepared (by decide) ∘ₘ
    mscAct Role.finalizer Port.unit Action.resolve Port.prepared := by
  funext r; cases r <;> rfl

theorem NF_eq : NF =
    mscMsg Role.agent Port.plan Role.worker Port.plan (by decide) ∘ₘ
    mscAct Role.worker Port.prepared Action.prepare Port.plan ∘ₘ
    mscIfFalse Guard.need1 Role.agent ∘ₘ
    mscMsg Role.agent (Port.ctrl false) Role.finalizer (Port.ctrl false) (by decide) ∘ₘ
    mscMsg Role.agent (Port.ctrl false) Role.reviewer1 (Port.ctrl false) (by decide) ∘ₘ
    mscMsg Role.worker Port.prepared Role.finalizer Port.prepared (by decide) ∘ₘ
    mscAct Role.finalizer Port.unit Action.resolve Port.prepared := by
  funext r; cases r <;> rfl

theorem compat_refl (p : Port) : PayloadCompatible Port p p := rfl

theorem NT_complete : IsCompleteMSC NT := by
  rw [NT_eq]
  repeat' first
    | apply concat_complete_complete
    | apply mscAct_isCompleteMSC
    | apply mscIfTrue_isCompleteMSC
    | apply mscMsg_isCompleteMSC _ _ _ _ _ (compat_refl _)

theorem NF_complete : IsCompleteMSC NF := by
  rw [NF_eq]
  repeat' first
    | apply concat_complete_complete
    | apply mscAct_isCompleteMSC
    | apply mscIfFalse_isCompleteMSC
    | apply mscMsg_isCompleteMSC _ _ _ _ _ (compat_refl _)

theorem NT_local : ∀ r, localTraceSemantics (endpoints r) (NT r) := by
  intro r; cases r <;>
    simp [localTraceSemantics, endpoints, NT, wAT, wW, wFT, wRT, ControlPayload.setDecision]

theorem NF_local : ∀ r, localTraceSemantics (endpoints r) (NF r) := by
  intro r; cases r <;>
    simp [localTraceSemantics, endpoints, NF, wAF, wW, wFF, wRF, ControlPayload.setDecision]

theorem trace_cases (r : Role) (v : Word r) (h : localTraceSemantics (endpoints r) v) :
    v = NT r ∨ v = NF r := by
  cases r <;>
    simpa [localTraceSemantics, endpoints, NT, NF, wAT, wAF, wW, wFT, wFF, wRT, wRF,
      ControlPayload.setDecision] using h

theorem pre_cons {A : Role} (u : Word A) (a : AlphabetOf (C := Guard) (F := Action) (Payload := Port) A)
    (v : Word A) :
    IsPrefixWord u (a :: v) ↔ u = [] ∨ ∃ u', u = a :: u' ∧ IsPrefixWord u' v := by
  constructor
  · rintro ⟨t, ht⟩
    cases u with
    | nil => exact Or.inl rfl
    | cons b u' =>
      simp only [List.cons_append, List.cons.injEq] at ht
      exact Or.inr ⟨u', by rw [ht.1], t, ht.2⟩
  · rintro (rfl | ⟨u', rfl, t, ht⟩)
    · exact ⟨a :: v, rfl⟩
    · exact ⟨t, by rw [ht]; rfl⟩

theorem pre_nil {A : Role} (u : Word A) : IsPrefixWord u [] ↔ u = [] := by
  constructor
  · rintro ⟨t, ht⟩
    cases u with
    | nil => rfl
    | cons _ _ => simp at ht
  · rintro rfl; exact ⟨[], rfl⟩

theorem pre_empty {A : Role} (v : Word A) : IsPrefixWord [] v := ⟨v, rfl⟩


theorem finT (wA : Word Role.agent) (wX : Word Role.finalizer)
    (hA : IsPrefixWord wA wAT) (hX : IsPrefixWord wX wFF)
    (hc : countRecvs Role.agent wX ≤ countSends Role.finalizer wA)
    (hl : ∀ p ∈ List.zip (sendPayloads Role.finalizer wA) (recvPayloads Role.agent wX), p.1 = p.2) :
    IsPrefixWord wX wFT := by
  simp only [wAT, wFF, pre_cons, pre_nil] at hA hX
  rcases hX with rfl | ⟨u, rfl, -⟩
  · exact pre_empty _
  · exfalso
    rcases hA with rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl⟩⟩⟩⟩⟩ <;>
      simp [countRecvs, countSends, recvPayloads, sendPayloads, Letter.isRecvFrom, Letter.isSendTo,
        AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse] at hc hl

theorem finF (wA : Word Role.agent) (wX : Word Role.finalizer)
    (hA : IsPrefixWord wA wAF) (hX : IsPrefixWord wX wFT)
    (hc : countRecvs Role.agent wX ≤ countSends Role.finalizer wA)
    (hl : ∀ p ∈ List.zip (sendPayloads Role.finalizer wA) (recvPayloads Role.agent wX), p.1 = p.2) :
    IsPrefixWord wX wFF := by
  simp only [wAF, wFT, pre_cons, pre_nil] at hA hX
  rcases hX with rfl | ⟨u, rfl, -⟩
  · exact pre_empty _
  · exfalso
    rcases hA with rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl⟩⟩⟩⟩ <;>
      simp [countRecvs, countSends, recvPayloads, sendPayloads, Letter.isRecvFrom, Letter.isSendTo,
        AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse] at hc hl

theorem revT (wA : Word Role.agent) (wX : Word Role.reviewer1)
    (hA : IsPrefixWord wA wAT) (hX : IsPrefixWord wX wRF)
    (hc : countRecvs Role.agent wX ≤ countSends Role.reviewer1 wA)
    (hl : ∀ p ∈ List.zip (sendPayloads Role.reviewer1 wA) (recvPayloads Role.agent wX), p.1 = p.2) :
    IsPrefixWord wX wRT := by
  simp only [wAT, wRF, pre_cons, pre_nil] at hA hX
  rcases hX with rfl | ⟨u, rfl, -⟩
  · exact pre_empty _
  · exfalso
    rcases hA with rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl⟩⟩⟩⟩⟩ <;>
      simp [countRecvs, countSends, recvPayloads, sendPayloads, Letter.isRecvFrom, Letter.isSendTo,
        AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse] at hc hl

theorem revF (wA : Word Role.agent) (wX : Word Role.reviewer1)
    (hA : IsPrefixWord wA wAF) (hX : IsPrefixWord wX wRT)
    (hc : countRecvs Role.agent wX ≤ countSends Role.reviewer1 wA)
    (hl : ∀ p ∈ List.zip (sendPayloads Role.reviewer1 wA) (recvPayloads Role.agent wX), p.1 = p.2) :
    IsPrefixWord wX wRF := by
  simp only [wAF, wRT, pre_cons, pre_nil] at hA hX
  rcases hX with rfl | ⟨u, rfl, -⟩
  · exact pre_empty _
  · exfalso
    rcases hA with rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl | ⟨_, rfl, rfl⟩⟩⟩⟩ <;>
      simp [countRecvs, countSends, recvPayloads, sendPayloads, Letter.isRecvFrom, Letter.isSendTo,
        AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse] at hc hl

end ReviewSol

open SupplierAccess ReviewSol in
theorem submission : ReviewCompletion.CompletionGoal := by
  intro M ⟨hloc, hmsc⟩
  have hX : ∀ r, IsPrefixWord (M r) (NT r) ∨ IsPrefixWord (M r) (NF r) := by
    intro r
    obtain ⟨v, hv, hp⟩ := hloc r
    rcases trace_cases r v hv with rfl | rfl
    · exact Or.inl hp
    · exact Or.inr hp
  have hc : ∀ A B, countRecvs A (M B) ≤ countSends B (M A) := hmsc.noUnmatchedRecv
  have hl : ∀ A B, ∀ p ∈ List.zip (sendPayloads B (M A)) (recvPayloads A (M B)), p.1 = p.2 :=
    hmsc.labelCompat
  rcases hX Role.agent with hA | hA
  · refine ⟨NT, ⟨NT_local, NT_complete⟩, ?_⟩
    intro r
    cases r
    · exact hA
    · rcases hX Role.worker with h | h <;> exact h
    · rcases hX Role.finalizer with h | h
      · exact h
      · exact finT _ _ hA h (hc _ _) (hl _ _)
    · rcases hX Role.reviewer1 with h | h
      · exact h
      · exact revT _ _ hA h (hc _ _) (hl _ _)
  · refine ⟨NF, ⟨NF_local, NF_complete⟩, ?_⟩
    intro r
    cases r
    · exact hA
    · rcases hX Role.worker with h | h <;> exact h
    · rcases hX Role.finalizer with h | h
      · exact finF _ _ hA h (hc _ _) (hl _ _)
      · exact h
    · rcases hX Role.reviewer1 with h | h
      · exact revF _ _ hA h (hc _ _) (hl _ _)
      · exact h
