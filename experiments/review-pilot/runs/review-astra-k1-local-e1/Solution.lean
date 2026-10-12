import CompletionGoal

set_option maxRecDepth 8192
set_option maxHeartbeats 8000000

namespace EndpointCompletion
open SupplierAccess
open Role

/-- The two complete endpoint words, retaining control and local actions. -/
def full (b : Bool) : Tuple
  | agent =>
      [AlphabetOf.mkSend agent Port.plan worker (by decide)] ++
      (if b then
        [AlphabetOf.mkIfTrue agent Guard.need1,
         AlphabetOf.mkSend agent (Port.ctrl true) finalizer (by decide),
         AlphabetOf.mkSend agent (Port.ctrl true) reviewer1 (by decide),
         AlphabetOf.mkSend agent Port.review reviewer1 (by decide)]
       else
        [AlphabetOf.mkIfFalse agent Guard.need1,
         AlphabetOf.mkSend agent (Port.ctrl false) finalizer (by decide),
         AlphabetOf.mkSend agent (Port.ctrl false) reviewer1 (by decide)])
  | worker =>
      [AlphabetOf.mkRecv worker Port.plan agent,
       AlphabetOf.mkAct worker Port.prepared Action.prepare Port.plan,
       AlphabetOf.mkSend worker Port.prepared finalizer (by decide)]
  | finalizer =>
      [AlphabetOf.mkRecv finalizer (Port.ctrl b) agent] ++
      (if b then [AlphabetOf.mkRecv finalizer Port.verdict reviewer1] else []) ++
      [AlphabetOf.mkRecv finalizer Port.prepared worker,
       AlphabetOf.mkAct finalizer Port.unit Action.resolve Port.prepared]
  | reviewer1 =>
      [AlphabetOf.mkRecv reviewer1 (Port.ctrl b) agent] ++
      (if b then
        [AlphabetOf.mkRecv reviewer1 Port.review agent,
         AlphabetOf.mkAct reviewer1 Port.unit Action.assess Port.review,
         AlphabetOf.mkSend reviewer1 Port.verdict finalizer (by decide)]
       else [])

theorem trace_shape (a : Role) (w : Word a) :
    localTraceSemantics (endpoints a) w ↔ w = full true a ∨ w = full false a := by
  cases a <;>
    simp [endpoints, localTraceSemantics, full, ControlPayload.setDecision,
      or_and_right, exists_or]

theorem full_length (b : Bool) (a : Role) : (full b a).length < 6 := by
  cases b <;> cases a <;> decide

theorem prefix_shape {a : Role} {w : Word a}
    (h : localPrefixSemantics (endpoints a) w) :
    ∃ b : Bool, ∃ n : Fin 6, w = (full b a).take n.val := by
  rcases h with ⟨v, hv, t, ht⟩
  have hv' : ∃ b : Bool, v = full b a := by
    rcases (trace_shape a v).mp hv with h | h
    · exact ⟨true, h⟩
    · exact ⟨false, h⟩
  rcases hv' with ⟨b, rfl⟩
  have hlen := congrArg List.length ht
  have hbound := full_length b a
  simp only [List.length_append] at hlen
  refine ⟨b, ⟨w.length, by omega⟩, ?_⟩
  rw [ht]
  simp

theorem take_prefix {a : Role} (w : Word a) (n : Nat) :
    IsPrefixWord (w.take n) w :=
  ⟨w.drop n, (List.take_append_drop n w).symm⟩

/-- A witness ranking: all communications go from earlier to later blocks. -/
def offset : Role → Nat
  | agent => 0
  | worker => 10
  | reviewer1 => 20
  | finalizer => 30

theorem sends_forward : ∀ (b : Bool) (a c : Role)
    (i : Fin (full b a).length),
    ((full b a).get i).val.isSendTo c = true → offset a + i.val < offset c := by
  decide

def ranking (b : Bool) : CausalRanking (full b) where
  rank e := offset e.lifeline + e.pos
  local_mono := by
    intro e₁ e₂ hrole hpos _ _
    simp only [hrole]
    omega
  fifo_mono := by
    intro a c i j hi _ hs _ _
    have h := sends_forward b a c ⟨i, hi⟩ hs
    dsimp at h ⊢
    omega

theorem full_complete (b : Bool) : IsCompleteMSC (full b) where
  complete := by
    have h : ∀ b : Bool, tupleComplete (full b) := by
      unfold tupleComplete channelComplete
      decide
    exact h b
  labelCompat := by
    have h : ∀ b : Bool, matchedLabelsCompatible (full b) := by
      unfold matchedLabelsCompatible channelLabelCompatible
      change ∀ b a c, ∀ p ∈ List.zip (sendPayloads c (full b a))
        (recvPayloads a (full b c)), p.1 = p.2
      decide
    exact h b
  acyclic := ⟨ranking b⟩

theorem full_semantics (b : Bool) : distSemantics endpoints (full b) := by
  refine ⟨?_, full_complete b⟩
  intro a
  apply (trace_shape a _).mpr
  cases b <;> simp

/-- Exactly the count and payload obligations on one asynchronous FIFO. -/
def queueOK {a c : Role} (u : Word a) (v : Word c) : Prop :=
  countRecvs a v ≤ countSends c u ∧
  ∀ p ∈ List.zip (sendPayloads c u) (recvPayloads a v), p.1 = p.2

/-- A received branch notification agrees with the agent; before receipt,
    the receiver's empty word extends along either branch.  The finite check
    includes every cut in both words and includes application traffic in
    the same FIFO as the control notification. -/
theorem notification_alignment : ∀ (a : Role),
    a = finalizer ∨ a = reviewer1 →
    ∀ (b c : Bool) (i j : Fin 6),
      queueOK ((full b agent).take i.val) ((full c a).take j.val) →
      (full b a).take ((full c a).take j.val).length = (full c a).take j.val := by
  unfold queueOK
  decide

theorem receiver_prefix {M : Tuple} (hM : IsMSC M)
    {b : Bool} {i : Fin 6} (hagent : M agent = (full b agent).take i.val)
    (a : Role) (ha : a = finalizer ∨ a = reviewer1)
    (hlocal : localPrefixSemantics (endpoints a) (M a)) :
    IsPrefixWord (M a) (full b a) := by
  obtain ⟨c, j, hj⟩ := prefix_shape hlocal
  have hqueue : queueOK (M agent) (M a) :=
    ⟨hM.noUnmatchedRecv agent a, hM.labelCompat agent a⟩
  rw [hagent, hj] at hqueue
  have heq := notification_alignment a ha b c i j hqueue
  rw [← hj] at heq
  refine ⟨(full b a).drop (M a).length, ?_⟩
  simpa only [heq] using (List.take_append_drop (M a).length (full b a)).symm

end EndpointCompletion

theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hM
  obtain ⟨b, i, hi⟩ := EndpointCompletion.prefix_shape (hM.1 SupplierAccess.Role.agent)
  refine ⟨EndpointCompletion.full b, EndpointCompletion.full_semantics b, ?_⟩
  intro a
  cases a with
  | agent =>
      rw [hi]
      exact EndpointCompletion.take_prefix _ _
  | worker =>
      obtain ⟨c, j, hj⟩ := EndpointCompletion.prefix_shape (hM.1 .worker)
      rw [hj]
      have heq : EndpointCompletion.full c .worker = EndpointCompletion.full b .worker := rfl
      rw [heq]
      exact EndpointCompletion.take_prefix _ _
  | finalizer =>
      exact EndpointCompletion.receiver_prefix hM.2 hi _ (Or.inl rfl) (hM.1 _)
  | reviewer1 =>
      exact EndpointCompletion.receiver_prefix hM.2 hi _ (Or.inr rfl) (hM.1 _)

