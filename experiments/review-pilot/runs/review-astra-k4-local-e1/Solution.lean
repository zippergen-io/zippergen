import CompletionGoal

set_option maxRecDepth 8192
set_option maxHeartbeats 8000000

namespace Direct
open SupplierAccess Role

/-- A finite interpreter for local endpoints, with no source program involved. -/
def finite {a : Role} : Local a → Prop
  | .eps | .send .. | .recv .. | .act .. => True
  | .seq p q | .recvIf _ _ p q | .localIf _ p q => finite p ∧ finite q
  | .recvWhile .. | .localWhile .. => False

noncomputable instance {a : Role} (p : Local a) : Decidable (finite p) := by
  induction p <;> simp only [finite] <;> infer_instance

def words {a : Role} : Local a → List (Word a)
  | .eps => [[]]
  | .send x b => if h : a ≠ b then [[AlphabetOf.mkSend a x b h]] else []
  | .recv x b => [[AlphabetOf.mkRecv a x b]]
  | .act y f x => [[AlphabetOf.mkAct a y f x]]
  | .seq p q => (words p).flatMap fun u => (words q).map fun v => u ++ v
  | .recvIf x b p q =>
      (words p).map (AlphabetOf.mkRecv a (ControlPayload.setDecision true x) b :: ·) ++
      (words q).map (AlphabetOf.mkRecv a (ControlPayload.setDecision false x) b :: ·)
  | .localIf c p q =>
      (words p).map (AlphabetOf.mkIfTrue a c :: ·) ++
      (words q).map (AlphabetOf.mkIfFalse a c :: ·)
  | .recvWhile .. | .localWhile .. => []

theorem words_spec {a : Role} (p : Local a) (hp : finite p) (w : Word a) :
    localTraceSemantics p w ↔ w ∈ words p := by
  induction p generalizing w with
  | eps => simp [localTraceSemantics, words]
  | send x b =>
      by_cases h : a ≠ b <;> simp [localTraceSemantics, words, h]
  | recv x b => simp [localTraceSemantics, words]
  | act y f x => simp [localTraceSemantics, words]
  | seq p q ip iq =>
      simp only [finite] at hp
      simp only [localTraceSemantics, words, List.mem_flatMap, List.mem_map]
      constructor
      · rintro ⟨u,v,hu,hv,rfl⟩
        exact ⟨u, (ip hp.1 u).mp hu, v, (iq hp.2 v).mp hv, rfl⟩
      · rintro ⟨u,hu,v,hv,rfl⟩
        exact ⟨u,v,(ip hp.1 u).mpr hu,(iq hp.2 v).mpr hv,rfl⟩
  | recvIf x b p q ip iq =>
      simp only [finite] at hp
      simp [localTraceSemantics, words, ip hp.1, iq hp.2, eq_comm]
  | localIf c p q ip iq =>
      simp only [finite] at hp
      simp [localTraceSemantics, words, ip hp.1, iq hp.2, eq_comm]
  | recvWhile => contradiction
  | localWhile => contradiction

def prefixes {α : Type} (xs : List α) : List (List α) :=
  (List.range (xs.length + 1)).map (fun n => xs.take n)

theorem prefix_mem {a : Role} {u v : Word a} (h : IsPrefixWord u v) :
    u ∈ prefixes v := by
  obtain ⟨t,rfl⟩ := h
  apply List.mem_map.mpr
  refine ⟨u.length, List.mem_range.mpr ?_, ?_⟩
  · simp only [List.length_append]; omega
  · simp

def prefixWords (a : Role) : List (Word a) := (words (endpoints a)).flatMap prefixes

theorem prefix_words {a : Role} {w : Word a} (h : localPrefixSemantics (endpoints a) w) :
    w ∈ prefixWords a := by
  obtain ⟨v,hv,hp⟩ := h
  exact List.mem_flatMap.mpr ⟨v, (words_spec _ (by cases a <;> decide) _).mp hv, prefix_mem hp⟩

abbrev snd (a b : Role) (x : Port) (h : a ≠ b) : AlphabetOf (C := Guard) (F := Action) (Payload := Port) a := AlphabetOf.mkSend a x b h
abbrev rcv (a b : Role) (x : Port) : AlphabetOf (C := Guard) (F := Action) (Payload := Port) a := AlphabetOf.mkRecv a x b
abbrev act (a : Role) (y : Port) (f : Action) (x : Port) : AlphabetOf (C := Guard) (F := Action) (Payload := Port) a := AlphabetOf.mkAct a y f x

def agentPart (r : Role) (h : agent ≠ r) (g : Guard) (b : Bool) : Word agent :=
  [if b then AlphabetOf.mkIfTrue agent g else AlphabetOf.mkIfFalse agent g,
   snd agent finalizer (.ctrl b) (by decide), snd agent r (.ctrl b) h] ++
  if b then [snd agent r .review h] else []

def reviewWord (r : Role) (h : r ≠ finalizer) (b : Bool) : Word r :=
  [rcv r agent (.ctrl b)] ++ if b then
    [rcv r agent .review, act r .unit .assess .review, snd r finalizer .verdict h] else []

def finalPart (r : Role) (b : Bool) : Word finalizer :=
  [rcv finalizer agent (.ctrl b)] ++ if b then [rcv finalizer r .verdict] else []

def full (b₁ b₂ b₃ b₄ : Bool) : Tuple
  | agent => [snd agent worker .plan (by decide)] ++
      agentPart reviewer1 (by decide) .need1 b₁ ++ agentPart reviewer2 (by decide) .need2 b₂ ++
      agentPart reviewer3 (by decide) .need3 b₃ ++ agentPart reviewer4 (by decide) .need4 b₄
  | worker => [rcv worker agent .plan, act worker .prepared .prepare .plan, snd worker finalizer .prepared (by decide)]
  | finalizer => finalPart reviewer1 b₁ ++ finalPart reviewer2 b₂ ++ finalPart reviewer3 b₃ ++ finalPart reviewer4 b₄ ++
      [rcv finalizer worker .prepared, act finalizer .unit .resolve .prepared]
  | reviewer1 => reviewWord reviewer1 (by decide) b₁
  | reviewer2 => reviewWord reviewer2 (by decide) b₂
  | reviewer3 => reviewWord reviewer3 (by decide) b₃
  | reviewer4 => reviewWord reviewer4 (by decide) b₄

def choices : List (Bool × Bool × Bool × Bool) :=
  [false,true].flatMap fun a => [false,true].flatMap fun b =>
  [false,true].flatMap fun c => [false,true].map fun d => (a,b,c,d)
def chosen (bs : Bool × Bool × Bool × Bool) : Tuple := full bs.1 bs.2.1 bs.2.2.1 bs.2.2.2

theorem agent_words : ∀ w ∈ words (endpoints agent), ∃ bs ∈ choices, w = chosen bs agent := by
  decide

theorem full_local (bs : Bool × Bool × Bool × Bool) (a : Role) :
    localTraceSemantics (endpoints a) (chosen bs a) := by
  apply (words_spec _ (by cases a <;> decide) _).mpr
  rcases bs with ⟨b₁,b₂,b₃,b₄⟩
  cases b₁ <;> cases b₂ <;> cases b₃ <;> cases b₄ <;> cases a <;> decide

/-- Incoming Agent traffic fixes every branch already observed by a recipient.
The finite check includes every prefix of every local endpoint word. -/
theorem align (bs : Bool × Bool × Bool × Bool) (a : Role) (ha : a ≠ agent) :
    ∀ w ∈ prefixWords a,
      (recvPayloads agent w).IsPrefix (sendPayloads a (chosen bs agent)) →
      w.IsPrefix (chosen bs a) := by
  rcases bs with ⟨b₁,b₂,b₃,b₄⟩
  cases a <;> first | contradiction | (cases b₁ <;> cases b₂ <;> cases b₃ <;> cases b₄ <;> decide)

theorem zip_prefix (ss rs : List Port)
    (hlen : rs.length ≤ ss.length)
    (heq : ∀ p ∈ ss.zip rs, p.1 = p.2) : rs.IsPrefix ss := by
  induction rs generalizing ss with
  | nil => exact List.nil_prefix
  | cons r rs ih =>
      cases ss with
      | nil => simp at hlen
      | cons s ss =>
          have hsr : s = r := heq (s,r) (by simp)
          subst s
          exact List.cons_prefix_cons.mpr ⟨rfl, ih ss (by simpa using hlen)
            (by intro p hp; exact heq p (by simp [hp]))⟩

/-- FIFO matching and absence of unmatched receives identify the received
payload sequence as a literal prefix of the sent sequence. -/
theorem received_prefix {M : Tuple} (h : IsMSC M) (a b : Role) :
    (recvPayloads a (M b)).IsPrefix (sendPayloads b (M a)) := by
  apply zip_prefix
  · simpa only [sendPayloads_length, recvPayloads_length] using h.noUnmatchedRecv a b
  · exact h.labelCompat a b

theorem sends_prefix {a : Role} {u v : Word a} (h : IsPrefixWord u v) (b : Role) :
    (sendPayloads b u).IsPrefix (sendPayloads b v) := by
  obtain ⟨t,rfl⟩ := h
  rw [sendPayloads_append]
  exact List.prefix_append _ _

/-- All communication goes from Agent to a worker/reviewer or Finalizer,
or from a worker/reviewer to Finalizer. -/
def level : Role → Nat
  | agent => 0
  | finalizer => 2
  | _ => 1

theorem full_length (bs : Bool × Bool × Bool × Bool) : ∀ a, (chosen bs a).length ≤ 20 := by
  rcases bs with ⟨b₁,b₂,b₃,b₄⟩
  cases b₁ <;> cases b₂ <;> cases b₃ <;> cases b₄ <;> decide

theorem send_level (bs : Bool × Bool × Bool × Bool) :
    ∀ a b, ∀ x ∈ chosen bs a, x.val.isSendTo b = true → level a < level b := by
  rcases bs with ⟨b₁,b₂,b₃,b₄⟩
  cases b₁ <;> cases b₂ <;> cases b₃ <;> cases b₄ <;> decide

theorem payload_balance (bs : Bool × Bool × Bool × Bool) :
    ∀ a b, sendPayloads b (chosen bs a) = recvPayloads a (chosen bs b) := by
  rcases bs with ⟨b₁,b₂,b₃,b₄⟩
  cases b₁ <;> cases b₂ <;> cases b₃ <;> cases b₄ <;> decide

theorem zip_self (xs : List Port) : ∀ p ∈ xs.zip xs, p.1 = p.2 := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      intro p hp
      simp only [List.zip_cons_cons, List.mem_cons] at hp
      rcases hp with rfl | hp
      · rfl
      · exact ih p hp

theorem full_complete (bs : Bool × Bool × Bool × Bool) : IsCompleteMSC (chosen bs) := by
  constructor
  · intro a b
    have h := congrArg List.length (payload_balance bs a b)
    simpa only [sendPayloads_length, recvPayloads_length] using h
  · intro a b p hp
    rw [payload_balance bs a b] at hp
    exact zip_self _ p hp
  · refine ⟨{ rank := fun e => 20 * level e.lifeline + e.pos, local_mono := ?_, fifo_mono := ?_ }⟩
    · intro e₁ e₂ he hl _ _
      rw [he]
      omega
    · intro a b j₁ j₂ hj₁ _ hs _ _
      have hlev := send_level bs a b ((chosen bs a).get ⟨j₁,hj₁⟩) (List.get_mem ..) hs
      have hlen := full_length bs a
      dsimp
      omega

end Direct

theorem submission : ReviewCompletion.CompletionGoal := by
  -- Keep any complete Agent word witnessing its existing local prefix.
  intro M hM
  obtain ⟨v,hv,hpref⟩ := hM.1 SupplierAccess.Role.agent
  obtain ⟨bs,_,hbs⟩ := Direct.agent_words v
    ((Direct.words_spec _ (by decide) _).mp hv)
  subst v
  refine ⟨Direct.chosen bs, ⟨Direct.full_local bs, Direct.full_complete bs⟩, ?_⟩
  intro a
  by_cases ha : a = SupplierAccess.Role.agent
  · subst a; exact hpref
  -- FIFO agreement forces every other existing word to fit the same choices.
  · have hr := (Direct.received_prefix hM.2 SupplierAccess.Role.agent a).trans
      (Direct.sends_prefix hpref a)
    exact List.prefix_iff_exists_eq_append.mp
      (Direct.align bs a ha (M a) (Direct.prefix_words (hM.1 a)) hr)
