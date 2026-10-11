import CompletionGoal

set_option maxRecDepth 8192
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

namespace SupplierAccess

def noLoop {A : Role} : Local A → Prop
  | .eps | .send .. | .recv .. | .act .. => True
  | .seq S T => noLoop S ∧ noLoop T
  | .recvIf _ _ S T | .localIf _ S T => noLoop S ∧ noLoop T
  | .recvWhile .. | .localWhile .. => False

def words {A : Role} : Local A → List (Word A)
  | .eps => [[]]
  | .send p B =>
      if h : A ≠ B then [[AlphabetOf.mkSend A p B h]] else []
  | .recv p B => [[AlphabetOf.mkRecv A p B]]
  | .act y f x => [[AlphabetOf.mkAct A y f x]]
  | .seq S T => (words S).flatMap (fun u => (words T).map (fun v => u ++ v))
  | .recvIf _ B S T =>
      ((words S).map (fun u => AlphabetOf.mkRecv A (.ctrl true) B :: u)) ++
      ((words T).map (fun u => AlphabetOf.mkRecv A (.ctrl false) B :: u))
  | .localIf c S T =>
      ((words S).map (fun u => AlphabetOf.mkIfTrue A c :: u)) ++
      ((words T).map (fun u => AlphabetOf.mkIfFalse A c :: u))
  | .recvWhile .. | .localWhile .. => []

theorem words_iff_trace {A : Role} (S : Local A) (h : noLoop S) (w : Word A) :
    w ∈ words S ↔ localTraceSemantics S w := by
  induction S generalizing w with
  | eps => simp [words, localTraceSemantics]
  | send p B =>
      simp only [noLoop] at h
      by_cases hAB : A ≠ B
      · simp [words, hAB, localTraceSemantics]
      · simp [words, hAB, localTraceSemantics, hAB]
  | recv p B => simp [words, localTraceSemantics]
  | act y f x => simp [words, localTraceSemantics]
  | seq S T ihS ihT =>
      rcases h with ⟨hS, hT⟩
      simp [words, localTraceSemantics, List.mem_flatMap, List.mem_map, ihS hS, ihT hT, eq_comm]
  | recvIf p B S T ihS ihT =>
      rcases h with ⟨hS, hT⟩
      simp [words, localTraceSemantics, List.mem_map, ihS hS, ihT hT,
        ControlPayload.setDecision, eq_comm]
  | localIf c S T ihS ihT =>
      rcases h with ⟨hS, hT⟩
      simp [words, localTraceSemantics, List.mem_map, ihS hS, ihT hT, eq_comm]
  | recvWhile => simp [noLoop] at h
  | localWhile => simp [noLoop] at h

theorem endpoints_noLoop (A : Role) : noLoop (endpoints A) := by
  cases A <;> simp [endpoints, noLoop]

theorem endpoint_trace_iff (A : Role) (w : Word A) :
    localTraceSemantics (endpoints A) w ↔ w ∈ words (endpoints A) :=
  (words_iff_trace (endpoints A) (endpoints_noLoop A) w).symm

def picked (A : Role) (n : Nat) : Word A :=
  (words (endpoints A)).getD n []

def completion (i : Fin 16) : Tuple
  | .agent => picked .agent i.val
  | .worker => picked .worker 0
  | .finalizer => picked .finalizer i.val
  | .reviewer1 => picked .reviewer1 (if i.val < 8 then 0 else 1)
  | .reviewer2 => picked .reviewer2 (if i.val % 8 < 4 then 0 else 1)
  | .reviewer3 => picked .reviewer3 (if i.val % 4 < 2 then 0 else 1)
  | .reviewer4 => picked .reviewer4 (if i.val % 2 < 1 then 0 else 1)

theorem completion_trace (i : Fin 16) (A : Role) :
    localTraceSemantics (endpoints A) (completion i A) := by
  apply (endpoint_trace_iff A _).2
  have h : ∀ i : Fin 16, ∀ A : Role,
      completion i A ∈ words (endpoints A) := by decide
  exact h i A

def listAll {α : Type} (xs : List α) (p : α → Prop) : Prop :=
  match xs with
  | [] => True
  | x :: rest => p x ∧ listAll rest p

def decideListAll {α : Type} (p : α → Prop) [DecidablePred p] :
    (xs : List α) → Decidable (listAll xs p)
  | [] => isTrue trivial
  | x :: xs =>
      match (inferInstance : Decidable (p x)), decideListAll p xs with
      | isTrue hx, isTrue hxs => isTrue ⟨hx, hxs⟩
      | isFalse hx, _ => isFalse (fun h => hx h.1)
      | _, isFalse hxs => isFalse (fun h => hxs h.2)

instance decidableListAll {α : Type} (xs : List α) (p : α → Prop)
    [DecidablePred p] : Decidable (listAll xs p) :=
  decideListAll p xs

theorem listAll_mem {α : Type} {xs : List α} {p : α → Prop}
    (h : listAll xs p) {x : α} (hx : x ∈ xs) : p x := by
  induction xs with
  | nil => cases hx
  | cons y ys ih =>
      rcases h with ⟨hy, hys⟩
      rcases List.mem_cons.mp hx with rfl | htail
      · exact hy
      · exact ih hys htail

theorem listAll_intro {α : Type} (xs : List α) {p : α → Prop}
    (h : ∀ x ∈ xs, p x) : listAll xs p := by
  induction xs with
  | nil => trivial
  | cons x xs ih =>
      constructor
      · exact h x (by simp)
      · exact ih (fun y hy => h y (by simp [hy]))

theorem agent_choice (w : Word .agent) (hw : w ∈ words (endpoints .agent)) :
    ∃ i : Fin 16, w = completion i .agent := by
  have hall : ∀ w ∈ words (endpoints .agent),
      ∃ i : Fin 16, w = completion i .agent := by
    have hh : listAll (words (endpoints .agent))
        (fun w => ∃ i : Fin 16, w = completion i .agent) := by decide
    exact fun w hw => listAll_mem hh hw
  exact hall w hw

def allPrefixes {A : Role} (ws : List (Word A)) : List (Word A) :=
  ws.flatMap fun w => (List.range (w.length + 1)).map w.take

theorem prefix_in_allPrefixes {A : Role} {u : Word A} {ws : List (Word A)}
    (h : ∃ w ∈ ws, IsPrefixWord u w) : u ∈ allPrefixes ws := by
  rcases h with ⟨w, hw, t, rfl⟩
  simp [allPrefixes, List.mem_flatMap, List.mem_map]
  refine ⟨u ++ t, hw, u.length, by simp; omega, ?_⟩
  simp

theorem prefix_of_take {A : Role} {u v : Word A}
    (hlen : u.length ≤ v.length) (heq : u = v.take u.length) :
    IsPrefixWord u v := by
  refine ⟨v.drop u.length, ?_⟩
  have hl : (v.take u.length).length = u.length := by simp [hlen]
  rw [heq, hl]
  exact (List.take_append_drop u.length v).symm

theorem take_of_prefix {A : Role} {u v : Word A}
    (h : IsPrefixWord u v) : u = v.take u.length := by
  rcases h with ⟨t, rfl⟩
  simp

def channelOK {A B : Role} (u : Word A) (v : Word B) : Prop :=
  (recvPayloads A v).length ≤ (sendPayloads B u).length ∧
  listAll (List.zip (sendPayloads B u) (recvPayloads A v))
    (fun p => p.1 = p.2)

instance decidableChannelOK {A B : Role} (u : Word A) (v : Word B) :
    Decidable (channelOK u v) := by
  unfold channelOK
  infer_instance

theorem channelOK_of_msc {M : Tuple} (h : IsMSC M) (A B : Role) :
    channelOK (M A) (M B) := by
  constructor
  · simpa [channelOK, sndCount, rcvCount] using h.noUnmatchedRecv A B
  · exact listAll_intro _ (fun p hp => h.labelCompat A B p hp)

def alignmentCheck (R : Role) : Prop :=
  ∀ i : Fin 16,
    listAll (allPrefixes [completion i .agent]) (fun a =>
      listAll (allPrefixes (words (endpoints R))) (fun u =>
        channelOK a u →
        u.length ≤ (completion i R).length ∧
        u = (completion i R).take u.length))


theorem alignmentCheck_target (R : Role)
    (hR : R = .finalizer ∨ R = .reviewer1 ∨ R = .reviewer2 ∨
      R = .reviewer3 ∨ R = .reviewer4) :
    alignmentCheck R := by
  rcases hR with rfl | rfl | rfl | rfl | rfl
  all_goals
    unfold alignmentCheck
    decide

theorem finalizer_alignment (i : Fin 16)
    (a : Word .agent) (u : Word .finalizer)
    (ha : IsPrefixWord a (completion i .agent))
    (hu : u ∈ allPrefixes (words (endpoints .finalizer)))
    (hc : channelOK a u) :
    IsPrefixWord u (completion i .finalizer) := by
  have hpa : a ∈ allPrefixes [completion i .agent] :=
    prefix_in_allPrefixes ⟨_, by simp, ha⟩
  have hh := alignmentCheck_target .finalizer (Or.inl rfl) i
  rcases (listAll_mem (listAll_mem hh hpa) hu) hc with ⟨hlen, heq⟩
  exact prefix_of_take hlen heq

theorem reviewer_alignment (i : Fin 16) (R : Role)
    (a : Word .agent) (u : Word R)
    (ha : IsPrefixWord a (completion i .agent))
    (hu : u ∈ allPrefixes (words (endpoints R)))
    (hc : channelOK a u)
    (hR : R = .reviewer1 ∨ R = .reviewer2 ∨ R = .reviewer3 ∨ R = .reviewer4) :
    IsPrefixWord u (completion i R) := by
  have hpa : a ∈ allPrefixes [completion i .agent] :=
    prefix_in_allPrefixes ⟨_, by simp, ha⟩
  have hRR : R = .finalizer ∨ R = .reviewer1 ∨ R = .reviewer2 ∨
      R = .reviewer3 ∨ R = .reviewer4 :=
    Or.inr hR
  have hh := alignmentCheck_target R hRR i
  rcases (listAll_mem (listAll_mem hh hpa) hu) hc with ⟨hlen, heq⟩
  exact prefix_of_take hlen heq

def stage : Role → Nat
  | .agent => 0
  | .finalizer => 2
  | _ => 1

theorem completion_short (i : Fin 16) (A : Role) :
    (completion i A).length < 100 := by
  have h : ∀ i : Fin 16, ∀ A : Role,
      (completion i A).length < 100 := by decide
  exact h i A

theorem completion_sends_forward (i : Fin 16) (A : Role)
    (l : AlphabetOf (C := Guard) (F := Action) (Payload := Port) A)
    (hl : l ∈ completion i A) (B : Role)
    (hs : l.val.isSendTo B = true) : stage A < stage B := by
  have h : ∀ i : Fin 16, ∀ A : Role,
      listAll (completion i A) (fun l =>
        ∀ B : Role, l.val.isSendTo B = true → stage A < stage B) := by decide
  exact (listAll_mem (h i A) hl) B hs

theorem completion_complete (i : Fin 16) :
    IsCompleteMSC (completion i) := by
  refine ⟨?_, ?_, ?_⟩
  · have h : ∀ i : Fin 16, ∀ A B : Role,
        sndCount (completion i) A B = rcvCount (completion i) A B := by decide
    exact h i
  · have h : ∀ i : Fin 16, ∀ A B : Role,
        listAll
          (List.zip (sendPayloads B (completion i A))
            (recvPayloads A (completion i B)))
          (fun p => p.1 = p.2) := by decide
    intro A B p hp
    exact listAll_mem (h i A B) hp
  · refine ⟨{
      rank := fun e => 100 * stage e.lifeline + e.pos
      local_mono := ?_
      fifo_mono := ?_
    }⟩
    · intro e1 e2 hrole hpos _ _
      change 100 * stage e1.lifeline + e1.pos <
        100 * stage e2.lifeline + e2.pos
      rw [hrole]
      omega
    · intro A B j1 j2 hj1 hj2 hsend _ _
      have hs : stage A < stage B :=
        completion_sends_forward i A
          ((completion i A).get ⟨j1, hj1⟩)
          (List.get_mem (completion i A) ⟨j1, hj1⟩) B hsend
      have hb := completion_short i A
      change 100 * stage A + j1 < 100 * stage B + j2
      omega

theorem completion_valid (i : Fin 16) :
    distSemantics endpoints (completion i) :=
  ⟨completion_trace i, completion_complete i⟩

theorem worker_alignment (i : Fin 16) (u : Word .worker)
    (hu : localPrefixSemantics (endpoints .worker) u) :
    IsPrefixWord u (completion i .worker) := by
  rcases hu with ⟨w, hw, hp⟩
  have hm : w ∈ words (endpoints .worker) :=
    (endpoint_trace_iff .worker w).1 hw
  have hh : listAll (words (endpoints .worker))
      (fun w => w = completion i .worker) := by
    change listAll (words (endpoints .worker))
      (fun w => w = picked .worker 0)
    decide
  rw [listAll_mem hh hm] at hp
  exact hp

end SupplierAccess

theorem submission : ReviewCompletion.CompletionGoal := by
  intro M h
  rcases h with ⟨hloc, hmsc⟩
  rcases hloc SupplierAccess.Role.agent with ⟨aw, haw, hpa⟩
  obtain ⟨i, hi⟩ :=
    SupplierAccess.agent_choice aw
      ((SupplierAccess.endpoint_trace_iff .agent aw).1 haw)
  refine ⟨SupplierAccess.completion i,
    SupplierAccess.completion_valid i, ?_⟩
  have ha : IsPrefixWord (M .agent) (SupplierAccess.completion i .agent) := by
    simpa [hi] using hpa
  intro A
  cases A with
  | agent => exact ha
  | worker => exact SupplierAccess.worker_alignment i _ (hloc .worker)
  | finalizer =>
      apply SupplierAccess.finalizer_alignment i (M .agent) (M .finalizer) ha
      · rcases hloc .finalizer with ⟨w, hw, hp⟩
        exact SupplierAccess.prefix_in_allPrefixes
          ⟨w, (SupplierAccess.endpoint_trace_iff .finalizer w).1 hw, hp⟩
      · exact SupplierAccess.channelOK_of_msc hmsc .agent .finalizer
  | reviewer1 =>
      apply SupplierAccess.reviewer_alignment i .reviewer1 (M .agent)
        (M .reviewer1) ha
      · rcases hloc .reviewer1 with ⟨w, hw, hp⟩
        exact SupplierAccess.prefix_in_allPrefixes
          ⟨w, (SupplierAccess.endpoint_trace_iff .reviewer1 w).1 hw, hp⟩
      · exact SupplierAccess.channelOK_of_msc hmsc .agent .reviewer1
      · exact Or.inl rfl
  | reviewer2 =>
      apply SupplierAccess.reviewer_alignment i .reviewer2 (M .agent)
        (M .reviewer2) ha
      · rcases hloc .reviewer2 with ⟨w, hw, hp⟩
        exact SupplierAccess.prefix_in_allPrefixes
          ⟨w, (SupplierAccess.endpoint_trace_iff .reviewer2 w).1 hw, hp⟩
      · exact SupplierAccess.channelOK_of_msc hmsc .agent .reviewer2
      · exact Or.inr (Or.inl rfl)
  | reviewer3 =>
      apply SupplierAccess.reviewer_alignment i .reviewer3 (M .agent)
        (M .reviewer3) ha
      · rcases hloc .reviewer3 with ⟨w, hw, hp⟩
        exact SupplierAccess.prefix_in_allPrefixes
          ⟨w, (SupplierAccess.endpoint_trace_iff .reviewer3 w).1 hw, hp⟩
      · exact SupplierAccess.channelOK_of_msc hmsc .agent .reviewer3
      · exact Or.inr (Or.inr (Or.inl rfl))
  | reviewer4 =>
      apply SupplierAccess.reviewer_alignment i .reviewer4 (M .agent)
        (M .reviewer4) ha
      · rcases hloc .reviewer4 with ⟨w, hw, hp⟩
        exact SupplierAccess.prefix_in_allPrefixes
          ⟨w, (SupplierAccess.endpoint_trace_iff .reviewer4 w).1 hw, hp⟩
      · exact SupplierAccess.channelOK_of_msc hmsc .agent .reviewer4
      · exact Or.inr (Or.inr (Or.inr rfl))
