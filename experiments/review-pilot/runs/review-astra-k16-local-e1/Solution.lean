import CompletionGoal
set_option maxRecDepth 8192
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
open SupplierAccess SupplierAccess.Role
namespace Direct
abbrev D := (a : Role) → Local a

/-- Every complete Agent word can be filled out to a balanced, acyclic tuple.
This certificate is constructed solely from the endpoint programs. -/
def Certified (P : D) : Prop := ∀ w, localTraceSemantics (P agent) w →
  ∃ N : Tuple, distSemantics P N ∧ N agent = w

def seq (P Q : D) : D := fun a => P a ;;ₗ Q a

theorem cert_seq {P Q : D} (hP : Certified P) (hQ : Certified Q) : Certified (seq P Q) := by
  rintro w ⟨u, v, hu, hv, rfl⟩
  obtain ⟨N, hN, hn⟩ := hP u hu
  obtain ⟨K, hK, hk⟩ := hQ v hv
  refine ⟨N ∘ₘ K, ⟨?_, concat_complete_complete N K hN.2 hK.2⟩, ?_⟩
  · intro a
    exact ⟨N a, K a, hN.1 a, hK.1 a, rfl⟩
  · simp [WordTuple.concat, hn, hk]

def msg (a b : Role) (p : Port) : D := fun x =>
  if x = a then .send p b else if x = b then .recv p a else .eps

def act (a : Role) (y : Port) (f : Action) (x : Port) : D := fun b =>
  if b = a then .act y f x else .eps

theorem cert_msg (a b : Role) (p : Port) (hab : a ≠ b) : Certified (msg a b p) := by
  intro w hw
  refine ⟨mscMsg a p b p hab, ⟨?_, mscMsg_isCompleteMSC a p b p hab rfl⟩, ?_⟩
  · intro x
    by_cases ha : x = a
    · subst x; simp [msg, localTraceSemantics, hab]
    · by_cases hb : x = b
      · subst x; simp [msg, ha, localTraceSemantics, hab]
      · simp [msg, ha, hb, mscMsg_other, localTraceSemantics]
  · by_cases ha : agent = a
    · subst a
      simp [msg, localTraceSemantics] at hw
      obtain ⟨h, rfl⟩ := hw
      simp
    · by_cases hb : agent = b
      · subst b; symm; simpa [msg, ha, localTraceSemantics] using hw
      · symm; simpa [msg, ha, hb, localTraceSemantics, mscMsg_other] using hw

theorem cert_act (a : Role) (y : Port) (f : Action) (x : Port) : Certified (act a y f x) := by
  intro w hw
  refine ⟨mscAct a y f x, ⟨?_, mscAct_isCompleteMSC a y f x⟩, ?_⟩
  · intro b
    by_cases h : b = a
    · subst b; simp [act, localTraceSemantics]
    · simp [act, h, localTraceSemantics]
  · by_cases h : agent = a
    · subst a; symm; simpa [act, localTraceSemantics] using hw
    · symm; simpa [act, h, localTraceSemantics] using hw

def review (c : Guard) (r : Role) : D := fun a =>
  if a = agent then
    .localIf c
      ((.send (.ctrl true) finalizer ;;ₗ (.send (.ctrl true) r ;;ₗ .eps)) ;;ₗ
        ((.send .review r ;;ₗ .eps) ;;ₗ .eps))
      ((.send (.ctrl false) finalizer ;;ₗ (.send (.ctrl false) r ;;ₗ .eps)) ;;ₗ .eps)
  else if a = finalizer then
    .recvIf (.ctrl false) agent ((.eps ;;ₗ .eps) ;;ₗ .recv .verdict r) .eps
  else if a = r then
    .recvIf (.ctrl false) agent
      ((.recv .review agent ;;ₗ .act .unit .assess .review) ;;ₗ .send .verdict finalizer) .eps
  else .eps

/-- Explicit words for one review block, including both control sends.
The elementary tuples supply FIFO balance and causal rankings. -/
def reviewWord (c : Guard) (r : Role) (ha : agent ≠ r) (hf : r ≠ finalizer) (b : Bool) : Tuple :=
  (if b then mscIfTrue c agent else mscIfFalse c agent) ∘ₘ
  mscMsg agent (.ctrl b) finalizer (.ctrl b) (by decide) ∘ₘ
  mscMsg agent (.ctrl b) r (.ctrl b) ha ∘ₘ
  (if b then
    mscMsg agent .review r .review ha ∘ₘ
    mscAct r .unit .assess .review ∘ₘ
    mscMsg r .verdict finalizer .verdict hf
  else mscEmpty)

theorem review_complete (c r ha hf b) : IsCompleteMSC (reviewWord c r ha hf b) := by
  cases b <;> simp only [reviewWord, Bool.false_eq_true, ↓reduceIte]
  all_goals repeat first
    | apply concat_complete_complete
    | exact mscEmpty_isCompleteMSC
    | exact mscIfTrue_isCompleteMSC _ _
    | exact mscIfFalse_isCompleteMSC _ _
    | apply mscMsg_isCompleteMSC; rfl
    | apply mscAct_isCompleteMSC

theorem cert_review (c : Guard) (r : Role) (ha : agent ≠ r) (hf : r ≠ finalizer) :
    Certified (review c r) := by
  intro w hw
  have hr : r ≠ agent := Ne.symm ha
  have hfr : finalizer ≠ r := Ne.symm hf
  have ht (b : Bool) : ∀ a, localTraceSemantics (review c r a) (reviewWord c r ha hf b a) := by
    intro a
    by_cases h1 : a = agent
    · subst a
      cases b <;> simp [review, reviewWord, WordTuple.concat, localTraceSemantics,
        mscIfTrue, mscIfFalse, choiceIfTrue, choiceIfFalse, ha, hr, hf, hfr,
        mscEmpty, WordTuple.empty, ControlPayload.setDecision]
    · by_cases h2 : a = finalizer
      · subst a
        cases b <;> simp [review, reviewWord, WordTuple.concat, localTraceSemantics,
          mscIfTrue, mscIfFalse, ha, hr, hf, hfr,
          mscEmpty, WordTuple.empty, ControlPayload.setDecision]
      · by_cases h3 : a = r
        · subst a
          cases b <;> simp [review, reviewWord, WordTuple.concat, localTraceSemantics,
            mscIfTrue, mscIfFalse, ha, hr, hf, hfr,
            mscEmpty, WordTuple.empty, ControlPayload.setDecision]
        · cases b <;> simp [review, reviewWord, WordTuple.concat, localTraceSemantics,
            mscIfTrue, mscIfFalse, h1, h2, h3, mscMsg_other,
            mscEmpty, WordTuple.empty]
  simp [review, localTraceSemantics, ha] at hw
  rcases hw with rfl | rfl
  · exact ⟨reviewWord c r ha hf true, ⟨ht true, review_complete _ _ _ _ _⟩, by
      simp [reviewWord, WordTuple.concat, mscIfTrue, choiceIfTrue, ha, hr,
        mscEmpty, WordTuple.empty]⟩
  · exact ⟨reviewWord c r ha hf false, ⟨ht false, review_complete _ _ _ _ _⟩, by
      simp [reviewWord, WordTuple.concat, mscIfFalse, choiceIfFalse, ha, hr,
        mscEmpty, WordTuple.empty]⟩

def assembled : D := seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (seq (msg agent worker .plan) (review .need1 .reviewer1)) (review .need2 .reviewer2)) (review .need3 .reviewer3)) (review .need4 .reviewer4)) (review .need5 .reviewer5)) (review .need6 .reviewer6)) (review .need7 .reviewer7)) (review .need8 .reviewer8)) (review .need9 .reviewer9)) (review .need10 .reviewer10)) (review .need11 .reviewer11)) (review .need12 .reviewer12)) (review .need13 .reviewer13)) (review .need14 .reviewer14)) (review .need15 .reviewer15)) (review .need16 .reviewer16)) (act worker .prepared .prepare .plan)) (msg worker finalizer .prepared)) (act finalizer .unit .resolve .prepared)

/-- The assembled family is definitionally the literal supplied endpoints. -/
theorem assembled_eq : assembled = endpoints := by
  funext a
  cases a <;> rfl

theorem certified : Certified endpoints := by
  rw [← assembled_eq]
  unfold assembled
  repeat first
    | apply cert_seq
    | apply cert_review <;> decide
    | apply cert_msg <;> decide
    | apply cert_act

/-- A finite local trace tree. Its only nondeterministic nodes receive a
decision from Agent; every other letter is fixed. -/
inductive Tree (a : Role) where
  | done
  | dead
  | emit (l : AlphabetOf (C := Guard) (F := Action) (Payload := Port) a) (t : Tree a)
  | fork (t f : Tree a)

def Accept {a} : Tree a → Word a → Prop
  | .done, w => w = []
  | .dead, _ => False
  | .emit l t, w => ∃ u, Accept t u ∧ w = l :: u
  | .fork t f, w =>
      (∃ u, Accept t u ∧ w = AlphabetOf.mkRecv a (.ctrl true) agent :: u) ∨
      (∃ u, Accept f u ∧ w = AlphabetOf.mkRecv a (.ctrl false) agent :: u)

def bind {a} : Tree a → Tree a → Tree a
  | .done, k => k
  | .dead, _ => .dead
  | .emit l t, k => .emit l (bind t k)
  | .fork t f, k => .fork (bind t k) (bind f k)

theorem accept_bind {a} (t k : Tree a) (w : Word a) :
    Accept (bind t k) w ↔ ∃ u v, Accept t u ∧ Accept k v ∧ w = u ++ v := by
  induction t generalizing w with
  | done => simp [bind, Accept]
  | dead => simp [bind, Accept]
  | emit l t ih =>
      simp only [bind, Accept, ih]
      constructor
      · rintro ⟨_, ⟨u,v,hu,hv,rfl⟩,rfl⟩
        exact ⟨l::u,v,⟨u,hu,rfl⟩,hv,rfl⟩
      · rintro ⟨_,v,⟨u,hu,rfl⟩,hv,rfl⟩
        exact ⟨u++v,⟨u,v,hu,hv,rfl⟩,rfl⟩
  | fork t f iht ihf =>
      simp only [bind, Accept, iht, ihf]
      constructor
      · rintro (⟨_,⟨u,v,hu,hv,rfl⟩,rfl⟩ | ⟨_,⟨u,v,hu,hv,rfl⟩,rfl⟩)
        · exact ⟨_,v,Or.inl ⟨u,hu,rfl⟩,hv,rfl⟩
        · exact ⟨_,v,Or.inr ⟨u,hu,rfl⟩,hv,rfl⟩
      · rintro ⟨_,v,(⟨u,hu,rfl⟩ | ⟨u,hu,rfl⟩),hv,rfl⟩
        · exact Or.inl ⟨u++v,⟨u,v,hu,hv,rfl⟩,rfl⟩
        · exact Or.inr ⟨u++v,⟨u,v,hu,hv,rfl⟩,rfl⟩

def Represented {a} (S : Local a) : Prop :=
  ∃ t : Tree a, ∀ w, localTraceSemantics S w ↔ Accept t w

theorem rep_eps {a} : Represented (LocProg.eps : Local a) := ⟨.done, fun _ => Iff.rfl⟩
theorem rep_act {a} (y f x) : Represented (LocProg.act y f x : Local a) := by
  refine ⟨.emit (AlphabetOf.mkAct a y f x) .done, ?_⟩
  intro w; simp [localTraceSemantics, Accept]
theorem rep_recv {a} (p b) : Represented (LocProg.recv p b : Local a) := by
  refine ⟨.emit (AlphabetOf.mkRecv a p b) .done, ?_⟩
  intro w; simp [localTraceSemantics, Accept]
theorem rep_send {a} (p b) : Represented (LocProg.send p b : Local a) := by
  by_cases h : a = b
  · subst b; exact ⟨.dead, by simp [localTraceSemantics, Accept]⟩
  · refine ⟨.emit (AlphabetOf.mkSend a p b h) .done, ?_⟩
    intro w; simp [localTraceSemantics, Accept, h]
theorem rep_seq {a} {S T : Local a} (hs : Represented S) (ht : Represented T) :
    Represented (S ;;ₗ T) := by
  obtain ⟨s,hs⟩ := hs
  obtain ⟨t,ht⟩ := ht
  exact ⟨bind s t, by intro w; simp [localTraceSemantics, accept_bind, hs, ht]⟩
theorem rep_if {a} {S T : Local a} (hs : Represented S) (ht : Represented T) :
    Represented (.recvIf (.ctrl false) agent S T) := by
  obtain ⟨s,hs⟩ := hs
  obtain ⟨t,ht⟩ := ht
  exact ⟨.fork s t, by intro w; simp [localTraceSemantics, Accept, hs, ht, ControlPayload.setDecision]⟩

theorem followers (a : Role) (h : a ≠ agent) : Represented (endpoints a) := by
  cases a <;> try contradiction
  all_goals unfold endpoints
  all_goals repeat first
    | apply rep_seq
    | apply rep_if
    | exact rep_eps
    | apply rep_act
    | apply rep_recv
    | apply rep_send

theorem prefix_nil {α : Type} (w : List α) : List.IsPrefix [] w := ⟨w,rfl⟩

theorem recv_cons_cancel {a} (l : AlphabetOf (C := Guard) (F := Action) (Payload := Port) a)
    {u v : Word a}
    (h : List.IsPrefix (recvPayloads agent (l::u)) (recvPayloads agent (l::v))) :
    List.IsPrefix (recvPayloads agent u) (recvPayloads agent v) := by
  cases hl : l.val <;> simp only [recvPayloads, hl] at h
  all_goals try exact h
  split at h
  · exact (List.cons_prefix_cons.mp h).2
  · exact h

/-- Agreement of the received Agent stream rules out the first possible
divergence of local words. This preserves every letter, including actions. -/
theorem tree_align {a} (t : Tree a) {u v p : Word a}
    (hu : Accept t u) (hv : Accept t v) (hp : List.IsPrefix p u)
    (hin : List.IsPrefix (recvPayloads agent p) (recvPayloads agent v)) : List.IsPrefix p v := by
  induction t generalizing u v p with
  | done =>
      simp only [Accept] at hu hv
      subst u; subst v; exact hp
  | dead => exact False.elim hu
  | emit l t ih =>
      obtain ⟨u,hu,rfl⟩ := hu
      obtain ⟨v,hv,rfl⟩ := hv
      cases p with
      | nil => exact prefix_nil _
      | cons x p =>
          obtain ⟨s,he⟩ := hp
          have hx : x = l := (List.cons.inj he).1
          subst x
          have hp' : List.IsPrefix p u := ⟨s,(List.cons.inj he).2⟩
          exact List.cons_prefix_cons.mpr ⟨rfl, ih hu hv hp' (recv_cons_cancel l hin)⟩
  | fork t f iht ihf =>
      cases p with
      | nil => exact prefix_nil _
      | cons x p =>
          rcases hu with ⟨u,hu,rfl⟩ | ⟨u,hu,rfl⟩ <;>
            rcases hv with ⟨v,hv,rfl⟩ | ⟨v,hv,rfl⟩
          all_goals
            obtain ⟨s,he⟩ := hp
            have hx := (List.cons.inj he).1
            subst x
            have hp' : List.IsPrefix p u := ⟨s,(List.cons.inj he).2⟩
          · exact List.cons_prefix_cons.mpr ⟨rfl, iht hu hv hp' (recv_cons_cancel _ hin)⟩
          · obtain ⟨z,hz⟩ := hin
            simp [recvPayloads, AlphabetOf.mkRecv] at hz
          · obtain ⟨z,hz⟩ := hin
            simp [recvPayloads, AlphabetOf.mkRecv] at hz
          · exact List.cons_prefix_cons.mpr ⟨rfl, ihf hu hv hp' (recv_cons_cancel _ hin)⟩

/-- No unmatched receives plus exact payload compatibility says that the
received channel word is literally a prefix of the sent channel word. -/
theorem fifo_prefix (s r : List Port) (hlen : r.length ≤ s.length)
    (hlab : ∀ p ∈ s.zip r, p.1 = p.2) : List.IsPrefix r s := by
  induction r generalizing s with
  | nil => exact prefix_nil _
  | cons x r ih =>
      cases s with
      | nil => simp at hlen
      | cons y s =>
          have hy : y = x := hlab (y,x) (by simp)
          subst y
          apply List.cons_prefix_cons.mpr
          refine ⟨rfl, ih s ?_ ?_⟩
          · simpa using hlen
          · intro p hp; exact hlab p (by simp [hp])

theorem channel_prefix {M : Tuple} (hM : IsMSC M) (a b : Role) :
    List.IsPrefix (recvPayloads a (M b)) (sendPayloads b (M a)) := by
  apply fifo_prefix
  · simpa only [recvPayloads_length, sendPayloads_length] using hM.noUnmatchedRecv a b
  · intro p hp
    exact hM.labelCompat a b p hp

theorem channel_equal {N : Tuple} (hN : IsCompleteMSC N) (a b : Role) :
    sendPayloads b (N a) = recvPayloads a (N b) := by
  obtain ⟨t,ht⟩ := channel_prefix (isCompleteMSC_implies_isMSC N hN) a b
  have hl : (sendPayloads b (N a)).length = (recvPayloads a (N b)).length := by
    simpa only [sendPayloads_length, recvPayloads_length] using hN.complete a b
  have hz : t = [] := by
    have := congrArg List.length ht
    simp only [List.length_append] at this
    have : t.length = 0 := by omega
    simpa using this
  simpa [hz] using ht.symm

theorem follower_prefix {M N : Tuple} (a : Role) (ha : a ≠ agent)
    (hM : distPrefixSemantics endpoints M) (hN : distSemantics endpoints N)
    (hp : IsPrefixWord (M agent) (N agent)) : IsPrefixWord (M a) (N a) := by
  obtain ⟨t,ht⟩ := followers a ha
  obtain ⟨u,hu,s,hs⟩ := hM.1 a
  have hp' : List.IsPrefix (M a) u := ⟨s,hs.symm⟩
  have hi : List.IsPrefix (recvPayloads agent (M a)) (recvPayloads agent (N a)) := by
    obtain ⟨v,hv⟩ := channel_prefix hM.2 agent a
    obtain ⟨z,hz⟩ := hp
    refine ⟨v ++ sendPayloads a z, ?_⟩
    rw [← List.append_assoc, hv, ← sendPayloads_append, ← hz, channel_equal hN.2]
  obtain ⟨z,hz⟩ := tree_align t ((ht u).mp hu) ((ht (N a)).mp (hN.1 a)) hp' hi
  exact ⟨z,hz.symm⟩

end Direct

theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hM
  obtain ⟨w,hw,hp⟩ := hM.1 agent
  obtain ⟨N,hN,he⟩ := Direct.certified w hw
  have hpa : IsPrefixWord (M agent) (N agent) := by simpa only [he] using hp
  refine ⟨N,hN,?_⟩
  intro a
  by_cases ha : a = agent
  · subst a; exact hpa
  · exact Direct.follower_prefix a ha hM hN hpa
