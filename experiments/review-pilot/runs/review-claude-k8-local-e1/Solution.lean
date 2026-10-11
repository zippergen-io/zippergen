import CompletionGoal
set_option maxHeartbeats 4000000
set_option maxRecDepth 8192
set_option linter.unusedSimpArgs false
namespace RCProof
open SupplierAccess

def agentBlk (g : Guard) (r : Role) (hr : Role.agent ≠ r) : Bool → Word .agent
  | true => [AlphabetOf.mkIfTrue Role.agent g, AlphabetOf.mkSend Role.agent (Port.ctrl true) Role.finalizer (by decide),
             AlphabetOf.mkSend Role.agent (Port.ctrl true) r hr, AlphabetOf.mkSend Role.agent Port.review r hr]
  | false => [AlphabetOf.mkIfFalse Role.agent g, AlphabetOf.mkSend Role.agent (Port.ctrl false) Role.finalizer (by decide),
             AlphabetOf.mkSend Role.agent (Port.ctrl false) r hr]

def blockProg (g : Guard) (r : Role) : Local .agent :=
  LocProg.localIf g ((LocProg.send (Port.ctrl true) Role.finalizer ;;ₗ (LocProg.send (Port.ctrl true) r ;;ₗ LocProg.eps)) ;;ₗ ((LocProg.send Port.review r ;;ₗ LocProg.eps) ;;ₗ LocProg.eps)) ((LocProg.send (Port.ctrl false) Role.finalizer ;;ₗ (LocProg.send (Port.ctrl false) r ;;ₗ LocProg.eps)) ;;ₗ LocProg.eps)

theorem blockProg_iff (g : Guard) (r : Role) (hr : Role.agent ≠ r) (w : Word .agent) :
    localTraceSemantics (blockProg g r) w ↔ ∃ x, w = agentBlk g r hr x := by
  constructor
  · intro h
    simp [blockProg, localTraceSemantics, exists_prop_of_true hr,
      exists_prop_of_true (show Role.agent ≠ Role.finalizer by decide)] at h
    rcases h with h | h
    · exact ⟨true, by simp [h, agentBlk, AlphabetOf.mkSend]⟩
    · exact ⟨false, by simp [h, agentBlk, AlphabetOf.mkSend]⟩
  · rintro ⟨x, rfl⟩
    cases x
    · simp [blockProg, localTraceSemantics, agentBlk, hr]
    · simp [blockProg, localTraceSemantics, agentBlk, hr]


def agentW (d : Role → Bool) : Word .agent :=
  AlphabetOf.mkSend Role.agent Port.plan Role.worker (by decide) ::
    (agentBlk Guard.need1 Role.reviewer1 (by decide) (d Role.reviewer1) ++
     agentBlk Guard.need2 Role.reviewer2 (by decide) (d Role.reviewer2) ++
     agentBlk Guard.need3 Role.reviewer3 (by decide) (d Role.reviewer3) ++
     agentBlk Guard.need4 Role.reviewer4 (by decide) (d Role.reviewer4) ++
     agentBlk Guard.need5 Role.reviewer5 (by decide) (d Role.reviewer5) ++
     agentBlk Guard.need6 Role.reviewer6 (by decide) (d Role.reviewer6) ++
     agentBlk Guard.need7 Role.reviewer7 (by decide) (d Role.reviewer7) ++
     agentBlk Guard.need8 Role.reviewer8 (by decide) (d Role.reviewer8))

theorem agent_ep : endpoints Role.agent =
    (((((((((((LocProg.send Port.plan Role.worker ;;ₗ blockProg Guard.need1 Role.reviewer1) ;;ₗ
      blockProg Guard.need2 Role.reviewer2) ;;ₗ blockProg Guard.need3 Role.reviewer3) ;;ₗ
      blockProg Guard.need4 Role.reviewer4) ;;ₗ blockProg Guard.need5 Role.reviewer5) ;;ₗ
      blockProg Guard.need6 Role.reviewer6) ;;ₗ blockProg Guard.need7 Role.reviewer7) ;;ₗ
      blockProg Guard.need8 Role.reviewer8) ;;ₗ LocProg.eps) ;;ₗ LocProg.eps) ;;ₗ LocProg.eps) := rfl

theorem agent_sem (w : Word .agent) :
    localTraceSemantics (endpoints Role.agent) w → ∃ d, w = agentW d := by
  rw [agent_ep]
  simp only [localTraceSemantics, blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer1),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer2),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer3),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer4),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer5),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer6),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer7),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer8),
    exists_prop_of_true (show Role.agent ≠ Role.worker by decide)]
  rintro ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, rfl, ⟨x1, rfl⟩, rfl⟩,
    ⟨x2, rfl⟩, rfl⟩, ⟨x3, rfl⟩, rfl⟩, ⟨x4, rfl⟩, rfl⟩, ⟨x5, rfl⟩, rfl⟩, ⟨x6, rfl⟩, rfl⟩, ⟨x7, rfl⟩, rfl⟩,
    ⟨x8, rfl⟩, rfl⟩, rfl, rfl⟩, rfl, rfl⟩, rfl, rfl⟩
  refine ⟨fun r => match r with
    | .reviewer1 => x1 | .reviewer2 => x2 | .reviewer3 => x3 | .reviewer4 => x4
    | .reviewer5 => x5 | .reviewer6 => x6 | .reviewer7 => x7 | .reviewer8 => x8 | _ => false, ?_⟩
  simp [agentW]

theorem agent_sem_mk (d : Role → Bool) : localTraceSemantics (endpoints Role.agent) (agentW d) := by
  rw [agent_ep]
  simp only [localTraceSemantics, blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer1),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer2),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer3),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer4),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer5),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer6),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer7),
    blockProg_iff _ _ (by decide : Role.agent ≠ Role.reviewer8),
    exists_prop_of_true (show Role.agent ≠ Role.worker by decide)]
  exact ⟨_, [], ⟨_, [], ⟨_, [], ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, rfl, ⟨d Role.reviewer1, rfl⟩, rfl⟩,
    ⟨d Role.reviewer2, rfl⟩, rfl⟩, ⟨d Role.reviewer3, rfl⟩, rfl⟩, ⟨d Role.reviewer4, rfl⟩, rfl⟩,
    ⟨d Role.reviewer5, rfl⟩, rfl⟩, ⟨d Role.reviewer6, rfl⟩, rfl⟩, ⟨d Role.reviewer7, rfl⟩, rfl⟩,
    ⟨d Role.reviewer8, rfl⟩, rfl⟩, rfl, rfl⟩, rfl, rfl⟩, rfl, by simp [agentW]⟩

def workerW : Word .worker :=
  [AlphabetOf.mkRecv Role.worker Port.plan Role.agent, AlphabetOf.mkAct Role.worker Port.prepared Action.prepare Port.plan,
   AlphabetOf.mkSend Role.worker Port.prepared Role.finalizer (by decide)]

theorem worker_sem_iff (w : Word .worker) :
    localTraceSemantics (endpoints Role.worker) w ↔ w = workerW := by
  simp [endpoints, localTraceSemantics, workerW, exists_prop_of_true (show Role.worker ≠ Role.finalizer by decide)]


def revW (r : Role) (hr : r ≠ Role.finalizer) : Bool → Word r
  | true => [AlphabetOf.mkRecv r (Port.ctrl true) Role.agent, AlphabetOf.mkRecv r Port.review Role.agent,
             AlphabetOf.mkAct r Port.unit Action.assess Port.review, AlphabetOf.mkSend r Port.verdict Role.finalizer hr]
  | false => [AlphabetOf.mkRecv r (Port.ctrl false) Role.agent]

def revProg (r : Role) : Local r :=
  LocProg.recvIf (Port.ctrl false) Role.agent ((LocProg.recv Port.review Role.agent ;;ₗ LocProg.act Port.unit Action.assess Port.review) ;;ₗ LocProg.send Port.verdict Role.finalizer) LocProg.eps

theorem revProg_iff (r : Role) (hr : r ≠ Role.finalizer) (w : Word r) :
    localTraceSemantics (revProg r) w ↔ ∃ x, w = revW r hr x := by
  constructor
  · intro h
    simp [revProg, localTraceSemantics, exists_prop_of_true hr, ControlPayload.setDecision] at h
    rcases h with h | h
    · exact ⟨true, by simp [h, revW, AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkAct]⟩
    · exact ⟨false, by simp [h, revW, AlphabetOf.mkRecv]⟩
  · rintro ⟨x, rfl⟩
    cases x
    · simp [revProg, localTraceSemantics, revW, hr, ControlPayload.setDecision]
    · simp [revProg, localTraceSemantics, revW, hr, ControlPayload.setDecision]

theorem rev1_ep (w : Word .reviewer1) : localTraceSemantics (endpoints Role.reviewer1) w ↔ localTraceSemantics (revProg Role.reviewer1) w := by
  simp [endpoints, revProg, localTraceSemantics]
theorem rev2_ep (w : Word .reviewer2) : localTraceSemantics (endpoints Role.reviewer2) w ↔ localTraceSemantics (revProg Role.reviewer2) w := by
  simp [endpoints, revProg, localTraceSemantics]
theorem rev3_ep (w : Word .reviewer3) : localTraceSemantics (endpoints Role.reviewer3) w ↔ localTraceSemantics (revProg Role.reviewer3) w := by
  simp [endpoints, revProg, localTraceSemantics]
theorem rev4_ep (w : Word .reviewer4) : localTraceSemantics (endpoints Role.reviewer4) w ↔ localTraceSemantics (revProg Role.reviewer4) w := by
  simp [endpoints, revProg, localTraceSemantics]
theorem rev5_ep (w : Word .reviewer5) : localTraceSemantics (endpoints Role.reviewer5) w ↔ localTraceSemantics (revProg Role.reviewer5) w := by
  simp [endpoints, revProg, localTraceSemantics]
theorem rev6_ep (w : Word .reviewer6) : localTraceSemantics (endpoints Role.reviewer6) w ↔ localTraceSemantics (revProg Role.reviewer6) w := by
  simp [endpoints, revProg, localTraceSemantics]
theorem rev7_ep (w : Word .reviewer7) : localTraceSemantics (endpoints Role.reviewer7) w ↔ localTraceSemantics (revProg Role.reviewer7) w := by
  simp [endpoints, revProg, localTraceSemantics]
theorem rev8_ep (w : Word .reviewer8) : localTraceSemantics (endpoints Role.reviewer8) w ↔ localTraceSemantics (revProg Role.reviewer8) w := by
  simp [endpoints, revProg, localTraceSemantics]

def finTail : Word .finalizer :=
  [AlphabetOf.mkRecv Role.finalizer Port.prepared Role.worker, AlphabetOf.mkAct Role.finalizer Port.unit Action.resolve Port.prepared]

def finBlk (r : Role) : Bool → Word .finalizer
  | true => [AlphabetOf.mkRecv Role.finalizer (Port.ctrl true) Role.agent, AlphabetOf.mkRecv Role.finalizer Port.verdict r]
  | false => [AlphabetOf.mkRecv Role.finalizer (Port.ctrl false) Role.agent]

def finBlocks : List (Role × Bool) → Word .finalizer
  | [] => finTail
  | p :: ps => finBlk p.1 p.2 ++ finBlocks ps

def revList : List Role := [Role.reviewer1, Role.reviewer2, Role.reviewer3, Role.reviewer4,
  Role.reviewer5, Role.reviewer6, Role.reviewer7, Role.reviewer8]

def finW (d : Role → Bool) : Word .finalizer := finBlocks (revList.map fun r => (r, d r))

def finProg (r : Role) : Local .finalizer :=
  LocProg.recvIf (Port.ctrl false) Role.agent ((LocProg.eps ;;ₗ LocProg.eps) ;;ₗ LocProg.recv Port.verdict r) LocProg.eps

theorem finProg_iff (r : Role) (w : Word .finalizer) :
    localTraceSemantics (finProg r) w ↔ ∃ x, w = finBlk r x := by
  constructor
  · intro h
    simp [finProg, localTraceSemantics, ControlPayload.setDecision] at h
    rcases h with h | h
    · exact ⟨true, by simp [h, finBlk, AlphabetOf.mkRecv]⟩
    · exact ⟨false, by simp [h, finBlk, AlphabetOf.mkRecv]⟩
  · rintro ⟨x, rfl⟩
    cases x
    · simp [finProg, localTraceSemantics, finBlk, ControlPayload.setDecision]
    · simp [finProg, localTraceSemantics, finBlk, ControlPayload.setDecision]

theorem fin_ep : endpoints Role.finalizer =
    (((((((((((LocProg.eps ;;ₗ finProg Role.reviewer1) ;;ₗ finProg Role.reviewer2) ;;ₗ finProg Role.reviewer3) ;;ₗ
      finProg Role.reviewer4) ;;ₗ finProg Role.reviewer5) ;;ₗ finProg Role.reviewer6) ;;ₗ finProg Role.reviewer7) ;;ₗ
      finProg Role.reviewer8) ;;ₗ LocProg.eps) ;;ₗ LocProg.recv Port.prepared Role.worker) ;;ₗ
      LocProg.act Port.unit Action.resolve Port.prepared) := rfl

theorem fin_sem (w : Word .finalizer) :
    localTraceSemantics (endpoints Role.finalizer) w → ∃ d, w = finW d := by
  rw [fin_ep]
  simp only [localTraceSemantics, finProg_iff]
  rintro ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, rfl, ⟨x1, rfl⟩, rfl⟩,
    ⟨x2, rfl⟩, rfl⟩, ⟨x3, rfl⟩, rfl⟩, ⟨x4, rfl⟩, rfl⟩, ⟨x5, rfl⟩, rfl⟩, ⟨x6, rfl⟩, rfl⟩, ⟨x7, rfl⟩, rfl⟩,
    ⟨x8, rfl⟩, rfl⟩, rfl, rfl⟩, rfl, rfl⟩, rfl, rfl⟩
  refine ⟨fun r => match r with
    | .reviewer1 => x1 | .reviewer2 => x2 | .reviewer3 => x3 | .reviewer4 => x4
    | .reviewer5 => x5 | .reviewer6 => x6 | .reviewer7 => x7 | .reviewer8 => x8 | _ => false, ?_⟩
  simp [finW, finBlocks, revList, finTail]

theorem fin_sem_mk (d : Role → Bool) : localTraceSemantics (endpoints Role.finalizer) (finW d) := by
  rw [fin_ep]
  simp only [localTraceSemantics, finProg_iff]
  exact ⟨_, _, ⟨_, _, ⟨_, [], ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨_, _, ⟨[], _, rfl, ⟨d Role.reviewer1, rfl⟩, rfl⟩,
    ⟨d Role.reviewer2, rfl⟩, rfl⟩, ⟨d Role.reviewer3, rfl⟩, rfl⟩, ⟨d Role.reviewer4, rfl⟩, rfl⟩,
    ⟨d Role.reviewer5, rfl⟩, rfl⟩, ⟨d Role.reviewer6, rfl⟩, rfl⟩, ⟨d Role.reviewer7, rfl⟩, rfl⟩,
    ⟨d Role.reviewer8, rfl⟩, rfl⟩, rfl, rfl⟩, rfl, rfl⟩, rfl, by simp [finW, finBlocks, revList, finTail]⟩


def NT (d : Role → Bool) : Tuple := fun a => match a with
  | .agent => agentW d
  | .worker => workerW
  | .finalizer => finW d
  | .reviewer1 => revW Role.reviewer1 (by decide) (d Role.reviewer1)
  | .reviewer2 => revW Role.reviewer2 (by decide) (d Role.reviewer2)
  | .reviewer3 => revW Role.reviewer3 (by decide) (d Role.reviewer3)
  | .reviewer4 => revW Role.reviewer4 (by decide) (d Role.reviewer4)
  | .reviewer5 => revW Role.reviewer5 (by decide) (d Role.reviewer5)
  | .reviewer6 => revW Role.reviewer6 (by decide) (d Role.reviewer6)
  | .reviewer7 => revW Role.reviewer7 (by decide) (d Role.reviewer7)
  | .reviewer8 => revW Role.reviewer8 (by decide) (d Role.reviewer8)

def revPay : Bool → List Port
  | true => [Port.ctrl true, Port.review]
  | false => [Port.ctrl false]

def verdictPay : Bool → List Port
  | true => [Port.verdict]
  | false => []

theorem send_agentBlk (B : Role) (g : Guard) (r : Role) (hr : Role.agent ≠ r) (x : Bool) :
    sendPayloads B (agentBlk g r hr x) =
      (if Role.finalizer = B then [Port.ctrl x] else []) ++ (if r = B then revPay x else []) := by
  cases x <;> by_cases h1 : Role.finalizer = B <;> by_cases h2 : r = B <;>
    simp [agentBlk, sendPayloads, AlphabetOf.mkSend, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse, revPay, h1, h2]

theorem recv_agentBlk (A : Role) (g : Guard) (r : Role) (hr : Role.agent ≠ r) (x : Bool) :
    recvPayloads A (agentBlk g r hr x) = [] := by
  cases x <;> simp [agentBlk, recvPayloads, AlphabetOf.mkSend, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse]

theorem recv_revW (A : Role) (r : Role) (hr : r ≠ Role.finalizer) (x : Bool) :
    recvPayloads A (revW r hr x) = if Role.agent = A then revPay x else [] := by
  cases x <;> by_cases h1 : Role.agent = A <;>
    simp [revW, recvPayloads, AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkAct, revPay, h1]

theorem send_revW (B : Role) (r : Role) (hr : r ≠ Role.finalizer) (x : Bool) :
    sendPayloads B (revW r hr x) = if Role.finalizer = B then verdictPay x else [] := by
  cases x <;> by_cases h1 : Role.finalizer = B <;>
    simp [revW, sendPayloads, AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkAct, verdictPay, h1]

theorem recv_finBlk (A : Role) (r : Role) (x : Bool) :
    recvPayloads A (finBlk r x) =
      (if Role.agent = A then [Port.ctrl x] else []) ++ (if r = A then verdictPay x else []) := by
  cases x <;> by_cases h1 : Role.agent = A <;> by_cases h2 : r = A <;>
    simp [finBlk, recvPayloads, AlphabetOf.mkRecv, verdictPay, h1, h2]

theorem send_finBlk (B : Role) (r : Role) (x : Bool) :
    sendPayloads B (finBlk r x) = [] := by
  cases x <;> simp [finBlk, sendPayloads, AlphabetOf.mkRecv]

theorem channel (d : Role → Bool) (A B : Role) :
    sendPayloads B (NT d A) = recvPayloads A (NT d B) := by
  cases A <;> cases B <;>
    simp [NT, agentW, workerW, finW, finBlocks, finTail, revList, send_agentBlk, recv_agentBlk,
      recv_revW, send_revW, recv_finBlk, send_finBlk, sendPayloads, recvPayloads,
      AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkAct]


def base : Role → Nat
  | .agent => 0
  | .worker => 100
  | .finalizer => 300
  | _ => 200

theorem agentBlk_len (g : Guard) (r : Role) (hr : Role.agent ≠ r) (x : Bool) :
    (agentBlk g r hr x).length ≤ 4 := by
  cases x <;> simp [agentBlk]

theorem revW_len (r : Role) (hr : r ≠ Role.finalizer) (x : Bool) : (revW r hr x).length ≤ 4 := by
  cases x <;> simp [revW]

theorem sendBound (d : Role → Bool) (A B : Role) (h : sendPayloads B (NT d A) ≠ []) :
    base A + (NT d A).length ≤ base B := by
  rw [channel] at h
  have hl : (agentW d).length ≤ 100 := by
    simp only [agentW, List.length_cons, List.length_append]
    have := agentBlk_len Guard.need1 Role.reviewer1 (by decide) (d Role.reviewer1)
    have := agentBlk_len Guard.need2 Role.reviewer2 (by decide) (d Role.reviewer2)
    have := agentBlk_len Guard.need3 Role.reviewer3 (by decide) (d Role.reviewer3)
    have := agentBlk_len Guard.need4 Role.reviewer4 (by decide) (d Role.reviewer4)
    have := agentBlk_len Guard.need5 Role.reviewer5 (by decide) (d Role.reviewer5)
    have := agentBlk_len Guard.need6 Role.reviewer6 (by decide) (d Role.reviewer6)
    have := agentBlk_len Guard.need7 Role.reviewer7 (by decide) (d Role.reviewer7)
    have := agentBlk_len Guard.need8 Role.reviewer8 (by decide) (d Role.reviewer8)
    omega
  cases A <;> cases B <;>
    first
    | (exfalso; apply h; simp [NT, agentW, workerW, finW, finBlocks, finTail, revList, send_agentBlk, recv_agentBlk,
        recv_revW, send_revW, recv_finBlk, send_finBlk, sendPayloads, recvPayloads,
        AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkAct, verdictPay]; done)
    | (simp only [NT, base]; first | omega | (simp [workerW]; done) |
        (have := revW_len Role.reviewer1 (by decide) (d Role.reviewer1); omega) |
        (have := revW_len Role.reviewer2 (by decide) (d Role.reviewer2); omega) |
        (have := revW_len Role.reviewer3 (by decide) (d Role.reviewer3); omega) |
        (have := revW_len Role.reviewer4 (by decide) (d Role.reviewer4); omega) |
        (have := revW_len Role.reviewer5 (by decide) (d Role.reviewer5); omega) |
        (have := revW_len Role.reviewer6 (by decide) (d Role.reviewer6); omega) |
        (have := revW_len Role.reviewer7 (by decide) (d Role.reviewer7); omega) |
        (have := revW_len Role.reviewer8 (by decide) (d Role.reviewer8); omega))


theorem countSends_pos_of_get {A : Role} (B : Role) :
    ∀ (w : Word A) (j : Nat) (hj : j < w.length),
      (w.get ⟨j, hj⟩).val.isSendTo B = true → 0 < countSends B w
  | [], j, hj, _ => by simp at hj
  | a :: w, 0, _, h => by
      simp at h
      simp [countSends, h]
      omega
  | a :: w, j + 1, hj, h => by
      have := countSends_pos_of_get B w j (by simpa using hj) (by simpa using h)
      simp [countSends]
      omega

theorem zip_self_compat (xs : List Port) :
    ∀ p ∈ List.zip xs xs, PayloadCompatible Port p.1 p.2 := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      intro p hp
      simp only [List.zip_cons_cons, List.mem_cons] at hp
      rcases hp with rfl | hp
      · show compatible x x
        rfl
      · exact ih p hp

def rankNT (d : Role → Bool) : CausalRanking (NT d) where
  rank := fun e => base e.lifeline + e.pos
  local_mono := by
    intro e1 e2 h hp _ _
    rw [h]
    omega
  fifo_mono := by
    intro A B j1 j2 hj1 hj2 hs _ _
    have hpos := countSends_pos_of_get B (NT d A) j1 hj1 hs
    have hne : sendPayloads B (NT d A) ≠ [] := by
      intro he
      have := sendPayloads_length (A := A) (C := Guard) (F := Action) (Payload := Port) B (NT d A)
      rw [he] at this
      simp at this
      omega
    have := sendBound d A B hne
    show base A + j1 < base B + j2
    omega

theorem NT_complete (d : Role → Bool) : IsCompleteMSC (NT d) where
  complete := by
    intro A B
    show countSends B (NT d A) = countRecvs A (NT d B)
    rw [← sendPayloads_length, ← recvPayloads_length, channel]
  labelCompat := by
    intro A B
    unfold channelLabelCompatible
    rw [channel]
    exact zip_self_compat _
  acyclic := ⟨rankNT d⟩


theorem pw_iff {A : Role} (u v : Word A) : IsPrefixWord u v ↔ u <+: v :=
  ⟨fun ⟨t, h⟩ => ⟨t, h.symm⟩, fun ⟨t, h⟩ => ⟨t, h.symm⟩⟩

theorem sendPayloads_prefix {A : Role} (B : Role) {u v : Word A} (h : u <+: v) :
    sendPayloads B u <+: sendPayloads B v := by
  obtain ⟨t, rfl⟩ := h
  rw [sendPayloads_append]
  exact List.prefix_append _ _

theorem zip_prefix : ∀ (xs ys : List Port), ys.length ≤ xs.length →
    (∀ p ∈ List.zip xs ys, PayloadCompatible Port p.1 p.2) → ys <+: xs
  | _, [], _, _ => List.nil_prefix
  | [], _ :: _, h, _ => by simp at h
  | x :: xs, y :: ys, h, hc => by
      have hxy : x = y := hc (x, y) (by simp)
      subst hxy
      rw [List.cons_prefix_cons]
      exact ⟨rfl, zip_prefix xs ys (by simpa using h) (fun p hp => hc p (by simp [hp]))⟩

theorem recv_from_agent_prefix (M : Tuple) (hM : IsMSC M) (B : Role) :
    recvPayloads Role.agent (M B) <+: sendPayloads B (M Role.agent) := by
  apply zip_prefix
  · rw [sendPayloads_length, recvPayloads_length]
    exact hM.noUnmatchedRecv Role.agent B
  · exact hM.labelCompat Role.agent B

theorem rev_fix (r : Role) (hr : r ≠ Role.finalizer) (u : Word r) (x y : Bool)
    (hu : u <+: revW r hr x) (hp : recvPayloads Role.agent u <+: recvPayloads Role.agent (revW r hr y)) :
    u <+: revW r hr y := by
  cases x <;> cases y
  · exact hu
  · rcases List.prefix_cons_iff.1 hu with rfl | ⟨t, rfl, _⟩
    · exact List.nil_prefix
    · simp [recv_revW, revPay, recvPayloads, AlphabetOf.mkRecv] at hp
  · rcases List.prefix_cons_iff.1 hu with rfl | ⟨t, rfl, _⟩
    · exact List.nil_prefix
    · simp [recv_revW, revPay, recvPayloads, AlphabetOf.mkRecv] at hp
  · exact hu

theorem recv_finBlocks_cons (r : Role) (hr : r ≠ Role.agent) (y : Bool) (qs : List (Role × Bool)) :
    recvPayloads Role.agent (finBlocks ((r, y) :: qs)) = Port.ctrl y :: recvPayloads Role.agent (finBlocks qs) := by
  simp [finBlocks, recv_finBlk, hr]

theorem fin_fix : ∀ (ps qs : List (Role × Bool)), ps.map Prod.fst = qs.map Prod.fst →
    (∀ p ∈ ps, p.1 ≠ Role.agent) → ∀ u : Word .finalizer, u <+: finBlocks ps →
    recvPayloads Role.agent u <+: recvPayloads Role.agent (finBlocks qs) → u <+: finBlocks qs
  | [], [], _, _, u, hu, _ => hu
  | [], _ :: _, h, _, _, _, _ => by simp at h
  | _ :: _, [], h, _, _, _, _ => by simp at h
  | (r, x) :: ps, (r', y) :: qs, h, hra, u, hu, hp => by
      simp only [List.map_cons, List.cons.injEq] at h
      obtain ⟨rfl, h⟩ := h
      have hr : r ≠ Role.agent := hra (r, x) (by simp)
      have hra' : ∀ p ∈ ps, p.1 ≠ Role.agent := fun p hp => hra p (by simp [hp])
      rw [recv_finBlocks_cons r hr] at hp
      cases x <;> simp only [finBlocks, finBlk, List.cons_append, List.nil_append] at hu
      · rcases List.prefix_cons_iff.1 hu with rfl | ⟨t, rfl, ht⟩
        · exact List.nil_prefix
        · simp only [recvPayloads, AlphabetOf.mkRecv] at hp
          simp at hp
          obtain ⟨rfl, hp⟩ := hp
          simp only [finBlocks, finBlk, List.cons_append, List.nil_append]
          rw [List.cons_prefix_cons]
          exact ⟨rfl, fin_fix ps qs h hra' t ht hp⟩
      · rcases List.prefix_cons_iff.1 hu with rfl | ⟨t, rfl, ht⟩
        · exact List.nil_prefix
        · simp only [recvPayloads, AlphabetOf.mkRecv] at hp
          simp at hp
          obtain ⟨rfl, hp⟩ := hp
          simp only [finBlocks, finBlk, List.cons_append, List.nil_append]
          rw [List.cons_prefix_cons]
          refine ⟨rfl, ?_⟩
          rcases List.prefix_cons_iff.1 ht with rfl | ⟨t', rfl, ht'⟩
          · exact List.nil_prefix
          · rw [List.cons_prefix_cons]
            refine ⟨rfl, fin_fix ps qs h hra' t' ht' ?_⟩
            simpa [recvPayloads, AlphabetOf.mkRecv, hr] using hp


theorem rev_case (M : Tuple) (hM : IsMSC M) (d : Role → Bool) (hag : M Role.agent <+: agentW d)
    (r : Role) (hr : r ≠ Role.finalizer) (hN : NT d r = revW r hr (d r))
    (hep : ∀ w, localTraceSemantics (endpoints r) w ↔ localTraceSemantics (revProg r) w)
    (hpre : localPrefixSemantics (endpoints r) (M r)) :
    IsPrefixWord (M r) (NT d r) := by
  obtain ⟨v, hv, hmv⟩ := hpre
  obtain ⟨x, rfl⟩ := (revProg_iff r hr v).1 ((hep v).1 hv)
  rw [pw_iff] at hmv ⊢
  rw [hN]
  apply rev_fix r hr _ x
  · exact hmv
  · have h1 := recv_from_agent_prefix M hM r
    have h2 := sendPayloads_prefix r hag
    have h3 := channel d Role.agent r
    rw [hN] at h3
    exact h1.trans (h2.trans ⟨[], by rw [List.append_nil]; exact h3⟩)

end RCProof

open SupplierAccess RCProof in
theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hM
  obtain ⟨hloc, hmsc⟩ := hM
  obtain ⟨v, hv, hmv⟩ := hloc Role.agent
  obtain ⟨d, rfl⟩ := agent_sem v hv
  have hag : M Role.agent <+: agentW d := (pw_iff _ _).1 hmv
  refine ⟨NT d, ⟨?_, NT_complete d⟩, ?_⟩
  · intro A
    cases A
    · exact agent_sem_mk d
    · exact (worker_sem_iff _).2 rfl
    · exact fin_sem_mk d
    · exact (rev1_ep _).2 ((revProg_iff _ (by decide) _).2 ⟨_, rfl⟩)
    · exact (rev2_ep _).2 ((revProg_iff _ (by decide) _).2 ⟨_, rfl⟩)
    · exact (rev3_ep _).2 ((revProg_iff _ (by decide) _).2 ⟨_, rfl⟩)
    · exact (rev4_ep _).2 ((revProg_iff _ (by decide) _).2 ⟨_, rfl⟩)
    · exact (rev5_ep _).2 ((revProg_iff _ (by decide) _).2 ⟨_, rfl⟩)
    · exact (rev6_ep _).2 ((revProg_iff _ (by decide) _).2 ⟨_, rfl⟩)
    · exact (rev7_ep _).2 ((revProg_iff _ (by decide) _).2 ⟨_, rfl⟩)
    · exact (rev8_ep _).2 ((revProg_iff _ (by decide) _).2 ⟨_, rfl⟩)
  · intro A
    cases A
    · exact hmv
    · obtain ⟨w, hw, hmw⟩ := hloc Role.worker
      rw [(worker_sem_iff w).1 hw] at hmw
      exact hmw
    · obtain ⟨w, hw, hmw⟩ := hloc Role.finalizer
      obtain ⟨c, rfl⟩ := fin_sem w hw
      rw [pw_iff] at hmw ⊢
      apply fin_fix _ _ (by simp [revList]) (by simp [revList]) _ hmw
      have h1 := recv_from_agent_prefix M hmsc Role.finalizer
      have h2 := sendPayloads_prefix Role.finalizer hag
      have h3 := channel d Role.agent Role.finalizer
      exact h1.trans (h2.trans ⟨[], by rw [List.append_nil]; exact h3⟩)
    · exact rev_case M hmsc d hag _ (by decide) rfl rev1_ep (hloc _)
    · exact rev_case M hmsc d hag _ (by decide) rfl rev2_ep (hloc _)
    · exact rev_case M hmsc d hag _ (by decide) rfl rev3_ep (hloc _)
    · exact rev_case M hmsc d hag _ (by decide) rfl rev4_ep (hloc _)
    · exact rev_case M hmsc d hag _ (by decide) rfl rev5_ep (hloc _)
    · exact rev_case M hmsc d hag _ (by decide) rfl rev6_ep (hloc _)
    · exact rev_case M hmsc d hag _ (by decide) rfl rev7_ep (hloc _)
    · exact rev_case M hmsc d hag _ (by decide) rfl rev8_ep (hloc _)

