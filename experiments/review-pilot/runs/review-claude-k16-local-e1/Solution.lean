import CompletionGoal

/-!
Completion for the single-request review protocol, by direct endpoint / FIFO reasoning.

Route: the agent's word fixes the 16 reviewer decisions `b`.  `NW b` is the canonical complete
run for `b`; it is a complete MSC (send/receive payload lists agree on every channel, and the
ranking `level * (total length + 1) + position` is monotone since every send goes to a strictly
higher level).  Every valid prefix `M` is a prefix of `NW b`: FIFO label compatibility forces the
control bits received by reviewers and the finalizer to coincide with the agent's.
-/

namespace ReviewProof
open SupplierAccess

def sendL (A : Role) (x : Port) (B : Role) (h : A ≠ B) : AlphabetOf (C := Guard) (F := Action) (Payload := Port) A :=
  AlphabetOf.mkSend A x B h
def recvL (A : Role) (x : Port) (B : Role) : AlphabetOf (C := Guard) (F := Action) (Payload := Port) A :=
  AlphabetOf.mkRecv A x B
def actL (A : Role) (y : Port) (f : Action) (x : Port) : AlphabetOf (C := Guard) (F := Action) (Payload := Port) A :=
  AlphabetOf.mkAct A y f x

/-- Agent block for one reviewer decision. -/
def agentBlk (g : Guard) (r : Role) (h : Role.agent ≠ r) (c : Bool) : Word .agent :=
  (if c then AlphabetOf.mkIfTrue (C := Guard) (F := Action) (Payload := Port) Role.agent g
    else AlphabetOf.mkIfFalse (C := Guard) (F := Action) (Payload := Port) Role.agent g) ::
  sendL .agent (.ctrl c) .finalizer (by decide) :: sendL .agent (.ctrl c) r h ::
  (if c then [sendL .agent .review r h] else [])

@[simp] theorem exSend {A B : Role} (x : Port) (h0 : A ≠ B) (w : Word A) :
    (∃ h : A ≠ B, w = [AlphabetOf.mkSend A x B h]) ↔ w = [AlphabetOf.mkSend A x B h0] :=
  ⟨fun ⟨_, e⟩ => e, fun e => ⟨h0, e⟩⟩

theorem agentIf (g : Guard) (r : Role) (h : Role.agent ≠ r) (u : Word .agent) :
    localTraceSemantics (LocProg.localIf (A := Role.agent) g
      ((LocProg.send (Port.ctrl true) Role.finalizer ;;ₗ (LocProg.send (Port.ctrl true) r ;;ₗ LocProg.eps)) ;;ₗ
        ((LocProg.send Port.review r ;;ₗ LocProg.eps) ;;ₗ LocProg.eps))
      ((LocProg.send (Port.ctrl false) Role.finalizer ;;ₗ (LocProg.send (Port.ctrl false) r ;;ₗ LocProg.eps)) ;;ₗ LocProg.eps)) u
    ↔ ∃ c, u = agentBlk g r h c := by
  simp [localTraceSemantics, agentBlk, sendL, exSend _ h]
  constructor <;> intro hh <;> rcases hh with hh | hh <;> simp [hh]

variable {A : Role}
theorem seq_sem (S1 S2 : Local A) (w : Word A) : localTraceSemantics (S1 ;;ₗ S2) w ↔
    ∃ u1 u2, localTraceSemantics S1 u1 ∧ localTraceSemantics S2 u2 ∧ w = u1 ++ u2 := Iff.rfl
theorem eps_sem (w : Word A) : localTraceSemantics (LocProg.eps : Local A) w ↔ w = [] := Iff.rfl
theorem send_sem (x : Port) (B : Role) (w : Word A) : localTraceSemantics (LocProg.send x B : Local A) w ↔
    ∃ h : A ≠ B, w = [AlphabetOf.mkSend A x B h] := Iff.rfl
theorem recv_sem (x : Port) (B : Role) (w : Word A) : localTraceSemantics (LocProg.recv x B : Local A) w ↔
    w = [recvL A x B] := Iff.rfl
theorem act_sem (y : Port) (f : Action) (x : Port) (w : Word A) :
    localTraceSemantics (LocProg.act y f x : Local A) w ↔ w = [actL A y f x] := Iff.rfl

theorem ex_pull {α β : Type} (Q : β → α → Prop) (P : α → Prop) :
    (∃ x, (∃ c, Q c x) ∧ P x) ↔ ∃ c, ∃ x, Q c x ∧ P x :=
  ⟨fun ⟨x, ⟨c, hc⟩, hp⟩ => ⟨c, x, hc, hp⟩, fun ⟨c, x, hc, hp⟩ => ⟨x, ⟨c, hc⟩, hp⟩⟩

def finBlk (r : Role) (c : Bool) : Word .finalizer :=
  recvL .finalizer (.ctrl c) .agent :: (if c then [recvL .finalizer .verdict r] else [])

def revW (r : Role) (h : r ≠ .finalizer) (c : Bool) : Word r :=
  recvL r (.ctrl c) .agent ::
    (if c then [recvL r .review .agent, actL r .unit .assess .review, sendL r .verdict .finalizer h] else [])

def revs : List Role := [.reviewer1, .reviewer2, .reviewer3, .reviewer4, .reviewer5, .reviewer6, .reviewer7,
  .reviewer8, .reviewer9, .reviewer10, .reviewer11, .reviewer12, .reviewer13, .reviewer14, .reviewer15, .reviewer16]

def finW (b : Role → Bool) : Word .finalizer :=
  (revs.flatMap fun r => finBlk r (b r)) ++
    [recvL .finalizer .prepared .worker, actL .finalizer .unit .resolve .prepared]

def workerW : Word .worker :=
  [recvL .worker .plan .agent, actL .worker .prepared .prepare .plan, sendL .worker .prepared .finalizer (by decide)]

def agentW (b : Role → Bool) : Word .agent :=
  [sendL .agent .plan .worker (by decide)] ++
   (agentBlk .need1 .reviewer1 (by decide) (b .reviewer1) ++
   (agentBlk .need2 .reviewer2 (by decide) (b .reviewer2) ++
   (agentBlk .need3 .reviewer3 (by decide) (b .reviewer3) ++
   (agentBlk .need4 .reviewer4 (by decide) (b .reviewer4) ++
   (agentBlk .need5 .reviewer5 (by decide) (b .reviewer5) ++
   (agentBlk .need6 .reviewer6 (by decide) (b .reviewer6) ++
   (agentBlk .need7 .reviewer7 (by decide) (b .reviewer7) ++
   (agentBlk .need8 .reviewer8 (by decide) (b .reviewer8) ++
   (agentBlk .need9 .reviewer9 (by decide) (b .reviewer9) ++
   (agentBlk .need10 .reviewer10 (by decide) (b .reviewer10) ++
   (agentBlk .need11 .reviewer11 (by decide) (b .reviewer11) ++
   (agentBlk .need12 .reviewer12 (by decide) (b .reviewer12) ++
   (agentBlk .need13 .reviewer13 (by decide) (b .reviewer13) ++
   (agentBlk .need14 .reviewer14 (by decide) (b .reviewer14) ++
   (agentBlk .need15 .reviewer15 (by decide) (b .reviewer15) ++
    agentBlk .need16 .reviewer16 (by decide) (b .reviewer16))))))))))))))))

/-- The canonical complete run for the reviewer decisions `b`. -/
def NW (b : Role → Bool) : Tuple := fun A =>
  match A with
  | .agent => agentW b
  | .worker => workerW
  | .finalizer => finW b
  | .reviewer1 => revW .reviewer1 (by decide) (b .reviewer1)
  | .reviewer2 => revW .reviewer2 (by decide) (b .reviewer2)
  | .reviewer3 => revW .reviewer3 (by decide) (b .reviewer3)
  | .reviewer4 => revW .reviewer4 (by decide) (b .reviewer4)
  | .reviewer5 => revW .reviewer5 (by decide) (b .reviewer5)
  | .reviewer6 => revW .reviewer6 (by decide) (b .reviewer6)
  | .reviewer7 => revW .reviewer7 (by decide) (b .reviewer7)
  | .reviewer8 => revW .reviewer8 (by decide) (b .reviewer8)
  | .reviewer9 => revW .reviewer9 (by decide) (b .reviewer9)
  | .reviewer10 => revW .reviewer10 (by decide) (b .reviewer10)
  | .reviewer11 => revW .reviewer11 (by decide) (b .reviewer11)
  | .reviewer12 => revW .reviewer12 (by decide) (b .reviewer12)
  | .reviewer13 => revW .reviewer13 (by decide) (b .reviewer13)
  | .reviewer14 => revW .reviewer14 (by decide) (b .reviewer14)
  | .reviewer15 => revW .reviewer15 (by decide) (b .reviewer15)
  | .reviewer16 => revW .reviewer16 (by decide) (b .reviewer16)

theorem finIf (r : Role) (u : Word .finalizer) :
    localTraceSemantics (LocProg.recvIf (A := Role.finalizer) (Port.ctrl false) Role.agent
      ((LocProg.eps ;;ₗ LocProg.eps) ;;ₗ LocProg.recv Port.verdict r) LocProg.eps) u
    ↔ ∃ c, u = finBlk r c := by
  constructor
  · rintro (⟨w, hw, rfl⟩ | ⟨w, hw, rfl⟩)
    · refine ⟨true, ?_⟩
      simp only [seq_sem, eps_sem, recv_sem] at hw
      obtain ⟨_, _, ⟨_, _, rfl, rfl, rfl⟩, rfl, rfl⟩ := hw
      rfl
    · refine ⟨false, ?_⟩
      simp only [eps_sem] at hw
      subst hw; rfl
  · rintro ⟨c, rfl⟩
    cases c
    · exact Or.inr ⟨[], rfl, rfl⟩
    · exact Or.inl ⟨[recvL .finalizer .verdict r], ⟨[], _, ⟨[], [], rfl, rfl, rfl⟩, rfl, rfl⟩, rfl⟩

theorem revIf (r : Role) (h : r ≠ .finalizer) (u : Word r) :
    localTraceSemantics (LocProg.recvIf (A := r) (Port.ctrl false) Role.agent
      ((LocProg.recv Port.review Role.agent ;;ₗ LocProg.act Port.unit Action.assess Port.review) ;;ₗ
        LocProg.send Port.verdict Role.finalizer) LocProg.eps) u
    ↔ ∃ c, u = revW r h c := by
  constructor
  · rintro (⟨w, hw, rfl⟩ | ⟨w, hw, rfl⟩)
    · refine ⟨true, ?_⟩
      simp only [seq_sem, recv_sem, act_sem, send_sem] at hw
      obtain ⟨_, _, ⟨_, _, rfl, rfl, rfl⟩, ⟨_, rfl⟩, rfl⟩ := hw
      rfl
    · refine ⟨false, ?_⟩
      simp only [eps_sem] at hw
      subst hw; rfl
  · rintro ⟨c, rfl⟩
    cases c
    · exact Or.inr ⟨[], rfl, rfl⟩
    · exact Or.inl ⟨_, ⟨_, _, ⟨_, _, rfl, rfl, rfl⟩, ⟨h, rfl⟩, rfl⟩, rfl⟩

theorem agent_char (v : Word .agent) :
    localTraceSemantics (endpoints .agent) v ↔ ∃ b, v = agentW b := by
  simp only [endpoints]
  simp (disch := decide) only [seq_sem, eps_sem, send_sem, agentIf, exists_eq_left, exists_and_left,
    exists_eq_right', List.append_nil, exSend _ (show Role.agent ≠ Role.worker by decide), ex_pull,
    List.append_assoc]
  constructor
  · rintro ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, rfl⟩
    refine ⟨fun r => match r with
      | .reviewer1 => c1 | .reviewer2 => c2 | .reviewer3 => c3 | .reviewer4 => c4
      | .reviewer5 => c5 | .reviewer6 => c6 | .reviewer7 => c7 | .reviewer8 => c8
      | .reviewer9 => c9 | .reviewer10 => c10 | .reviewer11 => c11 | .reviewer12 => c12
      | .reviewer13 => c13 | .reviewer14 => c14 | .reviewer15 => c15 | .reviewer16 => c16
      | _ => false, rfl⟩
  · rintro ⟨b, rfl⟩
    exact ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, rfl⟩

theorem finW_eq (b : Role → Bool) : finW b =
    finBlk .reviewer1 (b .reviewer1) ++ (finBlk .reviewer2 (b .reviewer2) ++ (finBlk .reviewer3 (b .reviewer3) ++
    (finBlk .reviewer4 (b .reviewer4) ++ (finBlk .reviewer5 (b .reviewer5) ++ (finBlk .reviewer6 (b .reviewer6) ++
    (finBlk .reviewer7 (b .reviewer7) ++ (finBlk .reviewer8 (b .reviewer8) ++ (finBlk .reviewer9 (b .reviewer9) ++
    (finBlk .reviewer10 (b .reviewer10) ++ (finBlk .reviewer11 (b .reviewer11) ++
    (finBlk .reviewer12 (b .reviewer12) ++ (finBlk .reviewer13 (b .reviewer13) ++
    (finBlk .reviewer14 (b .reviewer14) ++ (finBlk .reviewer15 (b .reviewer15) ++
    (finBlk .reviewer16 (b .reviewer16) ++
      ([recvL .finalizer .prepared .worker] ++ [actL .finalizer .unit .resolve .prepared])))))))))))))))) := by
  simp only [finW, revs, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.append_assoc]
  rfl

theorem fin_char (v : Word .finalizer) :
    localTraceSemantics (endpoints .finalizer) v ↔ ∃ b, v = finW b := by
  simp only [endpoints]
  simp only [seq_sem, eps_sem, recv_sem, act_sem, finIf, exists_eq_left, exists_and_left,
    exists_eq_right', List.append_nil, List.nil_append, ex_pull, List.append_assoc]
  simp only [finW_eq]
  constructor
  · rintro ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, rfl⟩
    exact ⟨fun r => match r with
      | .reviewer1 => c1 | .reviewer2 => c2 | .reviewer3 => c3 | .reviewer4 => c4
      | .reviewer5 => c5 | .reviewer6 => c6 | .reviewer7 => c7 | .reviewer8 => c8
      | .reviewer9 => c9 | .reviewer10 => c10 | .reviewer11 => c11 | .reviewer12 => c12
      | .reviewer13 => c13 | .reviewer14 => c14 | .reviewer15 => c15 | .reviewer16 => c16
      | _ => false, rfl⟩
  · rintro ⟨b, rfl⟩
    exact ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, rfl⟩

theorem worker_char (v : Word .worker) :
    localTraceSemantics (endpoints .worker) v ↔ v = workerW := by
  simp only [endpoints]
  simp (disch := decide) only [seq_sem, eps_sem, send_sem, recv_sem, act_sem, exists_eq_left,
    exists_and_left, List.append_nil, List.append_assoc,
    exSend _ (show Role.worker ≠ Role.finalizer by decide)]
  rfl

theorem rev_char (A : Role) (hA : A ∈ revs) (v : Word A) :
    localTraceSemantics (endpoints A) v ↔ ∃ c, v = NW (fun _ => c) A := by
  cases A <;> (try (exfalso; revert hA; decide)) <;> simp only [endpoints, NW] <;>
  simp (disch := decide) only [seq_sem, eps_sem, revIf, exists_eq_left, exists_and_left,
    exists_eq_right', List.append_nil, List.nil_append, ex_pull]

theorem nw_sem (b : Role → Bool) (A : Role) : localTraceSemantics (endpoints A) (NW b A) := by
  cases A
  · exact (agent_char _).2 ⟨b, rfl⟩
  · exact (worker_char _).2 rfl
  · exact (fin_char _).2 ⟨b, rfl⟩
  all_goals exact (rev_char _ (by decide) _).2 ⟨_, rfl⟩

/-! Payload bookkeeping. -/

@[simp] theorem sp_nil (B : Role) : sendPayloads (A := A) (C := Guard) (F := Action) (Payload := Port) B [] = [] := rfl
@[simp] theorem rp_nil (B : Role) : recvPayloads (B := A) (C := Guard) (F := Action) (Payload := Port) B [] = [] := rfl
@[simp] theorem sp_send (B C : Role) (x : Port) (h : A ≠ C) (w : Word A) :
    sendPayloads B (sendL A x C h :: w) = if C = B then x :: sendPayloads B w else sendPayloads B w := rfl
@[simp] theorem sp_send' (B C : Role) (x : Port) (h : A ≠ C) (w : Word A) :
    sendPayloads B (AlphabetOf.mkSend A x C h :: w) = if C = B then x :: sendPayloads B w else sendPayloads B w := rfl
@[simp] theorem sp_recv (B C : Role) (x : Port) (w : Word A) :
    sendPayloads B (recvL A x C :: w) = sendPayloads B w := rfl
@[simp] theorem sp_act (B : Role) (y x : Port) (f : Action) (w : Word A) :
    sendPayloads B (actL A y f x :: w) = sendPayloads B w := rfl
@[simp] theorem rp_send (B C : Role) (x : Port) (h : A ≠ C) (w : Word A) :
    recvPayloads B (sendL A x C h :: w) = recvPayloads B w := rfl
@[simp] theorem rp_send' (B C : Role) (x : Port) (h : A ≠ C) (w : Word A) :
    recvPayloads B (AlphabetOf.mkSend A x C h :: w) = recvPayloads B w := rfl
@[simp] theorem rp_recv (B C : Role) (x : Port) (w : Word A) :
    recvPayloads B (recvL A x C :: w) = if C = B then x :: recvPayloads B w else recvPayloads B w := rfl
@[simp] theorem rp_act (B : Role) (y x : Port) (f : Action) (w : Word A) :
    recvPayloads B (actL A y f x :: w) = recvPayloads B w := rfl

@[simp] theorem sp_ifT (B : Role) (g : Guard) (w : Word A) :
    sendPayloads B (AlphabetOf.mkIfTrue (C := Guard) (F := Action) (Payload := Port) A g :: w) = sendPayloads B w := rfl
@[simp] theorem sp_ifF (B : Role) (g : Guard) (w : Word A) :
    sendPayloads B (AlphabetOf.mkIfFalse (C := Guard) (F := Action) (Payload := Port) A g :: w) = sendPayloads B w := rfl
@[simp] theorem rp_ifT (B : Role) (g : Guard) (w : Word A) :
    recvPayloads B (AlphabetOf.mkIfTrue (C := Guard) (F := Action) (Payload := Port) A g :: w) = recvPayloads B w := rfl
@[simp] theorem rp_ifF (B : Role) (g : Guard) (w : Word A) :
    recvPayloads B (AlphabetOf.mkIfFalse (C := Guard) (F := Action) (Payload := Port) A g :: w) = recvPayloads B w := rfl

theorem sp_agentBlk (B : Role) (g : Guard) (r : Role) (h : Role.agent ≠ r) (c : Bool) :
    sendPayloads B (agentBlk g r h c) =
      (if Role.finalizer = B then [Port.ctrl c] else []) ++
        (if r = B then Port.ctrl c :: (if c then [Port.review] else []) else []) := by
  cases c <;> by_cases h1 : Role.finalizer = B <;> by_cases h2 : r = B <;>
    simp [agentBlk, h1, h2]

theorem rp_agentBlk (B : Role) (g : Guard) (r : Role) (h : Role.agent ≠ r) (c : Bool) :
    recvPayloads B (agentBlk g r h c) = [] := by
  cases c <;> simp [agentBlk]

theorem sp_finBlk (B r : Role) (c : Bool) : sendPayloads B (finBlk r c) = [] := by
  cases c <;> simp [finBlk]

theorem rp_finBlk (B r : Role) (c : Bool) : recvPayloads B (finBlk r c) =
    (if Role.agent = B then [Port.ctrl c] else []) ++
      (if r = B then (if c then [Port.verdict] else []) else []) := by
  cases c <;> by_cases h1 : Role.agent = B <;> by_cases h2 : r = B <;> simp [finBlk, h1, h2]

theorem sp_revW (B r : Role) (h : r ≠ .finalizer) (c : Bool) : sendPayloads B (revW r h c) =
    if Role.finalizer = B then (if c then [Port.verdict] else []) else [] := by
  cases c <;> by_cases h1 : Role.finalizer = B <;> simp [revW, h1]

theorem rp_revW (B r : Role) (h : r ≠ .finalizer) (c : Bool) : recvPayloads B (revW r h c) =
    if Role.agent = B then Port.ctrl c :: (if c then [Port.review] else []) else [] := by
  cases c <;> by_cases h1 : Role.agent = B <;> simp [revW, h1]

theorem payload_eq (b : Role → Bool) (A B : Role) :
    sendPayloads B (NW b A) = recvPayloads A (NW b B) := by
  cases A <;> cases B <;>
    simp [NW, agentW, finW, workerW, revs, sp_agentBlk, rp_agentBlk, sp_finBlk, rp_finBlk, sp_revW, rp_revW]

def lv : Role → Nat
  | .agent => 0
  | .finalizer => 2
  | _ => 1

theorem lv_lt (b : Role → Bool) (A B : Role) : sendPayloads B (NW b A) ≠ [] → lv A < lv B := by
  cases A <;> cases B <;>
    simp [lv, NW, agentW, finW, workerW, revs, sp_agentBlk, sp_finBlk, sp_revW]

theorem sp_ne_of_get {B : Role} : ∀ (w : Word A) (i : Nat) (hi : i < w.length),
    (w.get ⟨i, hi⟩).val.isSendTo B = true → sendPayloads B w ≠ []
  | x :: w, 0, _, h => by
      rcases x with ⟨l, hl⟩
      cases l <;> simp [Letter.isSendTo] at h
      subst h
      simp [sendPayloads]
  | x :: w, i + 1, hi, h => by
      have ih := sp_ne_of_get w i (by simp at hi; omega) h
      rcases x with ⟨l, hl⟩
      cases l <;> simp [sendPayloads] <;> (try split) <;> simp_all

def totalLen (M : Tuple) : Nat :=
  ((Fintype.elems : List Role).map fun R => (M R).length).sum

theorem len_le_total (M : Tuple) (A : Role) : (M A).length ≤ totalLen M := by
  cases A <;> simp [totalLen] <;> omega

theorem acyclic_of_lv (M : Tuple) (hs : ∀ A B, sendPayloads B (M A) ≠ [] → lv A < lv B) :
    hasAcyclicCausality M := by
  refine ⟨{ rank := fun e => lv e.lifeline * (totalLen M + 1) + e.pos, local_mono := ?_, fifo_mono := ?_ }⟩
  · rintro ⟨a1, p1⟩ ⟨a2, p2⟩ h hp _ _
    simp only at h hp
    subst h
    simp only
    omega
  · intro A B j1 j2 hj1 hj2 hsend _ _
    have hlt := hs A B (sp_ne_of_get _ _ hj1 hsend)
    have hmul : (lv A + 1) * (totalLen M + 1) ≤ lv B * (totalLen M + 1) := Nat.mul_le_mul_right _ hlt
    rw [Nat.succ_mul] at hmul
    have := len_le_total M A
    simp only
    omega

theorem zip_self_mem {α : Type} : ∀ (l : List α) (p : α × α), p ∈ List.zip l l → p.1 = p.2
  | [], _, h => by simp at h
  | x :: l, p, h => by
      simp only [List.zip_cons_cons, List.mem_cons] at h
      rcases h with rfl | h
      · rfl
      · exact zip_self_mem l p h

theorem nw_complete (b : Role → Bool) : distSemantics endpoints (NW b) := by
  refine ⟨nw_sem b, ?_, ?_, acyclic_of_lv _ (lv_lt b)⟩
  · intro A B
    simp only [channelComplete, sndCount, rcvCount]
    rw [← sendPayloads_length, ← recvPayloads_length, payload_eq]
  · intro A B p hp
    rw [payload_eq] at hp
    exact zip_self_mem _ p hp

/-! From a valid prefix to the canonical completion. -/

theorem pw_nil {A : Role} (w : Word A) : IsPrefixWord [] w := ⟨w, rfl⟩

theorem pw_cons {A : Role} (x y : AlphabetOf (C := Guard) (F := Action) (Payload := Port) A) (u w : Word A) :
    IsPrefixWord (x :: u) (y :: w) ↔ x = y ∧ IsPrefixWord u w := by
  constructor
  · rintro ⟨t, ht⟩
    simp only [List.cons_append, List.cons.injEq] at ht
    exact ⟨ht.1.symm, t, ht.2⟩
  · rintro ⟨rfl, t, ht⟩
    exact ⟨t, by simp [ht]⟩

theorem zip_pfx : ∀ (xs ys : List Port), (∀ p ∈ List.zip xs ys, p.1 = p.2) →
    ys.length ≤ xs.length → ∃ t, xs = ys ++ t
  | xs, [], _, _ => ⟨xs, rfl⟩
  | [], _ :: _, _, h => absurd h (by simp)
  | x :: xs, y :: ys, hz, h => by
      have hxy : x = y := hz (x, y) (by simp)
      obtain ⟨t, ht⟩ := zip_pfx xs ys (fun p hp => hz p (by simp [hp])) (by simp at h; omega)
      exact ⟨t, by simp [hxy, ht]⟩

theorem recv_pfx_send (M : Tuple) (hM : IsMSC M) (A B : Role) :
    ∃ t, sendPayloads B (M A) = recvPayloads A (M B) ++ t := by
  apply zip_pfx
  · intro p hp
    exact hM.labelCompat A B p hp
  · have := hM.noUnmatchedRecv A B
    simp only [rcvCount, sndCount] at this
    rw [sendPayloads_length, recvPayloads_length]
    exact this

theorem rev_pref (r : Role) (h : r ≠ .finalizer) (c c' : Bool) (u : Word r)
    (hu : IsPrefixWord u (revW r h c))
    (hp : ∃ t t', Port.ctrl c' :: t' = recvPayloads .agent u ++ t) :
    IsPrefixWord u (revW r h c') := by
  cases u with
  | nil => exact pw_nil _
  | cons x u =>
    have hu' := hu
    simp only [revW, pw_cons] at hu'
    obtain ⟨rfl, _⟩ := hu'
    obtain ⟨t, t', hp⟩ := hp
    simp only [rp_recv, ↓reduceIte, List.cons_append, List.cons.injEq, Port.ctrl.injEq] at hp
    rw [hp.1]
    exact hu

theorem fin_pref (b b' : Role → Bool) (t : Word .finalizer) :
    ∀ (rs : List Role), (∀ r ∈ rs, r ≠ .agent) → ∀ (u : Word .finalizer),
    IsPrefixWord u ((rs.flatMap fun r => finBlk r (b' r)) ++ t) →
    (∃ s, rs.map (fun r => Port.ctrl (b r)) = recvPayloads .agent u ++ s) →
    IsPrefixWord u ((rs.flatMap fun r => finBlk r (b r)) ++ t)
  | [], _, u, hu, _ => by simpa using hu
  | r :: rs, hrs, u, hu, hp => by
    have hr : r ≠ .agent := hrs r (by simp)
    have hrs' : ∀ r ∈ rs, r ≠ .agent := fun r' h' => hrs r' (by simp [h'])
    simp only [List.flatMap_cons, List.append_assoc] at hu ⊢
    cases u with
    | nil => exact pw_nil _
    | cons x u =>
      simp only [finBlk, List.cons_append, pw_cons] at hu ⊢
      obtain ⟨rfl, hu⟩ := hu
      obtain ⟨s, hs⟩ := hp
      simp only [rp_recv, ↓reduceIte, List.map_cons, List.cons_append, List.cons.injEq,
        Port.ctrl.injEq] at hs
      obtain ⟨hb, hs⟩ := hs
      have hbb : b' r = b r := by rw [hb]
      simp only [hbb] at hu ⊢
      clear hb hbb
      refine ⟨by trivial, ?_⟩
      cases hc : b r
      · simp only [hc, Bool.false_eq_true, if_false, List.nil_append] at hu ⊢
        exact fin_pref b b' t rs hrs' u hu ⟨s, hs⟩
      · simp only [hc, if_true, List.cons_append, List.nil_append] at hu ⊢
        cases u with
        | nil => exact pw_nil _
        | cons y u =>
          rw [pw_cons] at hu ⊢
          obtain ⟨rfl, hu⟩ := hu
          refine ⟨rfl, ?_⟩
          simp only [rp_recv, if_neg hr] at hs
          exact fin_pref b b' t rs hrs' u hu ⟨s, hs⟩

theorem sp_agentW_fin (b : Role → Bool) :
    sendPayloads .finalizer (agentW b) = revs.map (fun r => Port.ctrl (b r)) := by
  simp [agentW, revs, sp_agentBlk]

theorem sp_agentW_rev (b : Role → Bool) (A : Role) (hA : A ∈ revs) :
    ∃ t, sendPayloads A (agentW b) = Port.ctrl (b A) :: t := by
  cases A <;> (try (exfalso; revert hA; decide)) <;> simp [agentW, sp_agentBlk]

end ReviewProof

open ReviewProof SupplierAccess in
theorem submission : ReviewCompletion.CompletionGoal := by
  intro M ⟨hloc, hmsc⟩
  obtain ⟨va, hva, hpa⟩ := hloc .agent
  obtain ⟨b, rfl⟩ := (agent_char va).1 hva
  refine ⟨NW b, nw_complete b, ?_⟩
  have hrs : ∀ B, ∃ t, sendPayloads B (agentW b) = recvPayloads .agent (M B) ++ t := by
    intro B
    obtain ⟨t1, h1⟩ := hpa
    obtain ⟨t2, h2⟩ := recv_pfx_send M hmsc .agent B
    exact ⟨t2 ++ sendPayloads B t1, by rw [h1, sendPayloads_append, h2, List.append_assoc]⟩
  have hrev : ∀ A, A ∈ revs → IsPrefixWord (M A) (NW b A) := by
    intro A hA
    obtain ⟨v, hv, hp⟩ := hloc A
    obtain ⟨c, rfl⟩ := (rev_char A hA v).1 hv
    obtain ⟨t, ht⟩ := hrs A
    obtain ⟨t', ht'⟩ := sp_agentW_rev b A hA
    rw [ht'] at ht
    cases A <;> (try (exfalso; revert hA; decide)) <;> exact rev_pref _ _ c _ _ hp ⟨t, t', ht⟩
  intro A
  cases A
  · exact hpa
  · obtain ⟨v, hv, hp⟩ := hloc .worker
    rw [worker_char] at hv
    subst hv
    exact hp
  · obtain ⟨v, hv, hp⟩ := hloc .finalizer
    obtain ⟨b', rfl⟩ := (fin_char v).1 hv
    obtain ⟨t, ht⟩ := hrs .finalizer
    rw [sp_agentW_fin] at ht
    exact fin_pref b b' _ revs (by decide) _ hp ⟨t, ht⟩
  all_goals exact hrev _ (by decide)
