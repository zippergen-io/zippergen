import CompletionGoal

namespace ReviewCompletionProof
open SupplierAccess

section Traces

variable {L C F P : Type} [DecidableEq L] [ControlPayload P]

/-- Explicit finite enumeration of the complete traces of a loop-free local program. -/
def traces {A : L} : LocProg L C F P A → List (LocalWord (C := C) (F := F) (Payload := P) A)
  | .eps => [[]]
  | .send xs B => if h : A = B then [] else [[AlphabetOf.mkSend A xs B h]]
  | .recv ys B => [[AlphabetOf.mkRecv A ys B]]
  | .act ys f xs => [[AlphabetOf.mkAct A ys f xs]]
  | .seq S1 S2 => (traces S1).flatMap (fun u => (traces S2).map (fun v => u ++ v))
  | .recvIf ys B S1 S2 =>
      (traces S1).map (fun u => AlphabetOf.mkRecv A (ControlPayload.setDecision true ys) B :: u) ++
      (traces S2).map (fun u => AlphabetOf.mkRecv A (ControlPayload.setDecision false ys) B :: u)
  | .localIf c S1 S2 =>
      (traces S1).map (fun u => AlphabetOf.mkIfTrue A c :: u) ++
      (traces S2).map (fun u => AlphabetOf.mkIfFalse A c :: u)
  | .recvWhile .. => []
  | .localWhile .. => []

/-- Loop-freeness of a local program. -/
def loopFree {A : L} : LocProg L C F P A → Bool
  | .seq S1 S2 => loopFree S1 && loopFree S2
  | .recvIf _ _ S1 S2 => loopFree S1 && loopFree S2
  | .localIf _ S1 S2 => loopFree S1 && loopFree S2
  | .recvWhile .. => false
  | .localWhile .. => false
  | _ => true

theorem traces_iff {A : L} (S : LocProg L C F P A) (hS : loopFree S = true)
    (w : LocalWord (C := C) (F := F) (Payload := P) A) :
    localTraceSemantics S w ↔ w ∈ traces S := by
  induction S generalizing w with
  | eps => simp [localTraceSemantics, traces]
  | send xs B =>
      by_cases h : A = B
      · simp [localTraceSemantics, traces, h]
      · simp [localTraceSemantics, traces, h]
  | recv ys B => simp [localTraceSemantics, traces]
  | act ys f xs => simp [localTraceSemantics, traces]
  | seq S1 S2 ih1 ih2 =>
      simp only [loopFree, Bool.and_eq_true] at hS
      simp only [localTraceSemantics, traces, List.mem_flatMap, List.mem_map, ih1 hS.1, ih2 hS.2]
      constructor
      · rintro ⟨u1, u2, h1, h2, rfl⟩; exact ⟨u1, h1, u2, h2, rfl⟩
      · rintro ⟨u1, h1, u2, h2, rfl⟩; exact ⟨u1, u2, h1, h2, rfl⟩
  | recvIf ys B S1 S2 ih1 ih2 =>
      simp only [loopFree, Bool.and_eq_true] at hS
      simp only [localTraceSemantics, traces, List.mem_append, List.mem_map, ih1 hS.1, ih2 hS.2]
      constructor
      · rintro (⟨u, h, rfl⟩ | ⟨u, h, rfl⟩)
        · exact Or.inl ⟨u, h, rfl⟩
        · exact Or.inr ⟨u, h, rfl⟩
      · rintro (⟨u, h, rfl⟩ | ⟨u, h, rfl⟩)
        · exact Or.inl ⟨u, h, rfl⟩
        · exact Or.inr ⟨u, h, rfl⟩
  | localIf c S1 S2 ih1 ih2 =>
      simp only [loopFree, Bool.and_eq_true] at hS
      simp only [localTraceSemantics, traces, List.mem_append, List.mem_map, ih1 hS.1, ih2 hS.2]
      constructor
      · rintro (⟨u, h, rfl⟩ | ⟨u, h, rfl⟩)
        · exact Or.inl ⟨u, h, rfl⟩
        · exact Or.inr ⟨u, h, rfl⟩
      · rintro (⟨u, h, rfl⟩ | ⟨u, h, rfl⟩)
        · exact Or.inl ⟨u, h, rfl⟩
        · exact Or.inr ⟨u, h, rfl⟩
  | recvWhile => simp [loopFree] at hS
  | localWhile => simp [loopFree] at hS

end Traces

/-- The receiver's completion is the unique trace whose control/application traffic
    from the agent equals what the agent's complete word sends. -/
def pick (A : Role) (v : Word Role.agent) : Word A :=
  ((traces (endpoints A)).find?
    (fun w => decide (recvPayloads Role.agent w = sendPayloads A v))).getD []

def NOf (v : Word Role.agent) : Tuple
  | .agent => v
  | .worker => pick .worker v
  | .finalizer => pick .finalizer v
  | .reviewer1 => pick .reviewer1 v
  | .reviewer2 => pick .reviewer2 v
  | .reviewer3 => pick .reviewer3 v
  | .reviewer4 => pick .reviewer4 v

def tier : Role → Nat
  | .agent => 0
  | .finalizer => 2
  | _ => 1

theorem endpoints_loopFree : ∀ A, loopFree (endpoints A) = true := by decide

theorem D1 : ∀ v ∈ traces (endpoints Role.agent), ∀ A, NOf v A ∈ traces (endpoints A) := by
  decide +kernel

theorem D2 : ∀ v ∈ traces (endpoints Role.agent), ∀ A B : Role,
    countSends B (NOf v A) = countRecvs A (NOf v B) := by
  decide +kernel

theorem D3 : ∀ v ∈ traces (endpoints Role.agent), ∀ A B : Role,
    ∀ p ∈ List.zip (sendPayloads B (NOf v A)) (recvPayloads A (NOf v B)), p.1 = p.2 := by
  decide +kernel

theorem D4 : ∀ v ∈ traces (endpoints Role.agent), ∀ A : Role,
    (NOf v A).length < 100 ∧ ∀ x ∈ NOf v A, ∀ B : Role, x.val.isSendTo B = true → tier A < tier B := by
  decide +kernel

theorem D5 : ∀ v ∈ traces (endpoints Role.agent), ∀ A : Role, A ≠ Role.agent →
    ∀ w ∈ traces (endpoints A), ∀ n ∈ List.range (w.length + 1),
      (recvPayloads Role.agent (w.take n)).isPrefixOf (sendPayloads A v) = true →
      (w.take n).isPrefixOf (NOf v A) = true := by
  decide +kernel

theorem prefix_of_zip : ∀ (R S : List Port), R.length ≤ S.length →
    (∀ p ∈ List.zip S R, p.1 = p.2) → R <+: S
  | [], S, _, _ => List.nil_prefix
  | _ :: _, [], h, _ => by simp at h
  | r :: R, s :: S, h, hz => by
      have hrs : s = r := hz (s, r) (by simp)
      subst hrs
      exact List.prefix_cons_inj _ |>.2
        (prefix_of_zip R S (by simpa using h) (fun p hp => hz p (by simp [hp])))

theorem sendPayloads_prefix {A : Role} (B : Role) {u w : Word A} (h : IsPrefixWord u w) :
    sendPayloads B u <+: sendPayloads B w := by
  obtain ⟨t, rfl⟩ := h
  exact ⟨sendPayloads B t, by simp⟩

theorem nonagent_prefix (M : Tuple) (hloc : ∀ A, localPrefixSemantics (endpoints A) (M A))
    (hmsc : IsMSC M) (v : Word Role.agent) (hv : v ∈ traces (endpoints Role.agent))
    (hpre : IsPrefixWord (M Role.agent) v) (A : Role) (hA : A ≠ Role.agent) :
    IsPrefixWord (M A) (NOf v A) := by
  obtain ⟨w, hw, t, hMw⟩ := hloc A
  rw [traces_iff _ (endpoints_loopFree A)] at hw
  have hR : recvPayloads Role.agent (M A) <+: sendPayloads A v := by
    have h1 := hmsc.noUnmatchedRecv Role.agent A
    have h2 := hmsc.labelCompat Role.agent A
    simp only [rcvCount, sndCount] at h1
    refine (prefix_of_zip _ _ ?_ ?_).trans (sendPayloads_prefix A hpre)
    · simpa using h1
    · intro p hp
      exact h2 p hp
  have htake : w.take (M A).length = M A := by
    rw [hMw]; simp
  have hn : (M A).length ∈ List.range (w.length + 1) := by
    rw [List.mem_range, hMw]; simp; omega
  have := D5 v hv A hA w hw _ hn (by rw [htake]; exact List.isPrefixOf_iff_prefix.2 hR)
  rw [htake] at this
  obtain ⟨t', ht'⟩ := List.isPrefixOf_iff_prefix.1 this
  exact ⟨t', ht'.symm⟩

theorem complete_NOf (v : Word Role.agent) (hv : v ∈ traces (endpoints Role.agent)) :
    IsCompleteMSC (NOf v) where
  complete := fun A B => D2 v hv A B
  labelCompat := fun A B p hp => D3 v hv A B p hp
  acyclic := ⟨{
    rank := fun e => tier e.lifeline * 100 + e.pos
    local_mono := by
      rintro ⟨a1, p1⟩ ⟨a2, p2⟩ h hlt _ _
      simp only at h hlt ⊢
      subst h; omega
    fifo_mono := by
      intro A B j1 j2 hj1 hj2 hsend _ _
      have htier := (D4 v hv A).2 _ (List.get_mem _ _) B hsend
      have hlen := (D4 v hv A).1
      simp only
      have : (tier A + 1) * 100 ≤ tier B * 100 := Nat.mul_le_mul_right _ htier
      rw [Nat.add_mul] at this
      omega }⟩

end ReviewCompletionProof

open ReviewCompletionProof SupplierAccess in
theorem submission : ReviewCompletion.CompletionGoal := by
  rintro M ⟨hloc, hmsc⟩
  obtain ⟨v, hv, hpre⟩ := hloc Role.agent
  rw [traces_iff _ (endpoints_loopFree _)] at hv
  refine ⟨NOf v, ⟨fun A => (traces_iff _ (endpoints_loopFree A) _).2 (D1 v hv A),
    complete_NOf v hv⟩, ?_⟩
  intro A
  cases A
  · exact hpre
  all_goals exact nonagent_prefix M hloc hmsc v hv hpre _ (by decide)
