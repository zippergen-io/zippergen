import CompletionGoal
set_option maxRecDepth 8192
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
open SupplierAccess SupplierAccess.Role
namespace Direct

/- Direct endpoint proof. FIFO compatibility first gives a prefix of the sent
   payload stream. A finite receive-decision tree then determines every
   non-Agent word from that stream. No source program or projection is used. -/

abbrev Pref {α : Type} (u v : List α) : Prop := ∃ t, v = u ++ t

theorem pref_trans {α : Type} {u v w : List α} (h : Pref u v) (k : Pref v w) : Pref u w := by
  rcases h with ⟨s, rfl⟩; rcases k with ⟨t, rfl⟩
  exact ⟨s ++ t, by simp [List.append_assoc]⟩

theorem pref_cons {α : Type} {x y : α} {u v : List α} (h : Pref (x::u) (y::v)) : x = y ∧ Pref u v := by
  rcases h with ⟨t, h⟩
  have hh := List.cons.inj h
  exact ⟨hh.1.symm, t, hh.2⟩

theorem pref_send {a : Role} (b : Role) {u v : Word a} (h : Pref u v) :
    Pref (sendPayloads b u) (sendPayloads b v) := by
  rcases h with ⟨t, rfl⟩; exact ⟨sendPayloads b t, sendPayloads_append _ _ _⟩

theorem zip_prefix (s r : List Port) (hlen : r.length ≤ s.length)
    (heq : ∀ p ∈ s.zip r, p.1 = p.2) : Pref r s := by
  induction r generalizing s with
  | nil => exact ⟨s, rfl⟩
  | cons x xs ih =>
    cases s with
    | nil => simp at hlen
    | cons y ys =>
      have hxy : y = x := heq (y,x) (by simp)
      obtain ⟨t, ht⟩ := ih ys (by simpa using hlen) (by
        intro p hp; exact heq p (by simp [hp]))
      exact ⟨t, by simp [hxy, ht]⟩

theorem fifo_prefix (M : Tuple) (h : IsMSC M) (a b : Role) :
    Pref (recvPayloads a (M b)) (sendPayloads b (M a)) := by
  apply zip_prefix
  · simpa [sndCount, rcvCount] using h.noUnmatchedRecv a b
  · intro p hp
    exact h.labelCompat a b p hp

/-- A local trace tree: only receives from Agent select a branch. -/
inductive Tree (a : Role) where
  | done
  | step (l : AlphabetOf (C := Guard) (F := Action) (Payload := Port) a) (next : Tree a)
  | branch (yes no : Tree a)

def accepts {a : Role} : Tree a → Word a → Prop
  | .done, w => w = []
  | .step l t, w => ∃ u, accepts t u ∧ w = l :: u
  | .branch t f, w =>
      (∃ u, accepts t u ∧ w = AlphabetOf.mkRecv a (.ctrl true) agent :: u) ∨
      (∃ u, accepts f u ∧ w = AlphabetOf.mkRecv a (.ctrl false) agent :: u)

def join {a : Role} : Tree a → Tree a → Tree a
  | .done, k => k
  | .step l t, k => .step l (join t k)
  | .branch t f, k => .branch (join t k) (join f k)

theorem accepts_join {a : Role} (t k : Tree a) (w : Word a) :
    accepts (join t k) w ↔ ∃ u v, accepts t u ∧ accepts k v ∧ w = u ++ v := by
  induction t generalizing w with
  | done => simp [join, accepts]
  | step l t ih =>
    simp only [join, accepts, ih]
    constructor
    · rintro ⟨_, ⟨u,v,hu,hv,rfl⟩,rfl⟩
      exact ⟨l::u,v,⟨u,hu,rfl⟩,hv,rfl⟩
    · rintro ⟨_,v,⟨u,hu,rfl⟩,hv,rfl⟩
      exact ⟨u++v,⟨u,v,hu,hv,rfl⟩,rfl⟩
  | branch t f iht ihf =>
    simp only [join, accepts, iht, ihf]
    constructor
    · rintro (⟨_,⟨u,v,hu,hv,rfl⟩,rfl⟩ | ⟨_,⟨u,v,hu,hv,rfl⟩,rfl⟩)
      · exact ⟨_,v,Or.inl ⟨u,hu,rfl⟩,hv,rfl⟩
      · exact ⟨_,v,Or.inr ⟨u,hu,rfl⟩,hv,rfl⟩
    · rintro ⟨_,v,(⟨u,hu,rfl⟩ | ⟨u,hu,rfl⟩),hv,rfl⟩
      · exact Or.inl ⟨u++v,⟨u,v,hu,hv,rfl⟩,rfl⟩
      · exact Or.inr ⟨u++v,⟨u,v,hu,hv,rfl⟩,rfl⟩

/-- Input-stream agreement forces literal agreement of the existing word. -/
theorem tree_align {a : Role} (t : Tree a) {u v w : Word a}
    (hu : accepts t u) (hv : accepts t v) (hw : Pref w u)
    (hi : Pref (recvPayloads agent w) (recvPayloads agent v)) : Pref w v := by
  induction t generalizing u v w with
  | done =>
    change u = [] at hu
    subst u; rcases hw with ⟨s,hs⟩
    have : w = [] := by simpa using congrArg (List.take w.length) hs
    subst w; exact ⟨v,rfl⟩
  | step l t ih =>
    obtain ⟨u,hu,rfl⟩ := hu
    obtain ⟨v,hv,rfl⟩ := hv
    cases w with
    | nil => exact ⟨_,rfl⟩
    | cons x w =>
      obtain ⟨hx,htail⟩ := pref_cons hw
      subst x
      have hi' : Pref (recvPayloads agent w) (recvPayloads agent v) := by
        cases hl : l.val <;> simp only [recvPayloads, hl] at hi
        all_goals first | exact hi | (split at hi <;> first | exact hi | exact (pref_cons hi).2)
      obtain ⟨s,hs⟩ := ih hu hv htail hi'
      exact ⟨s,by simp [hs]⟩
  | branch t f iht ihf =>
    cases w with
    | nil => exact ⟨_,rfl⟩
    | cons x w =>
      rcases hu with ⟨u,hu,rfl⟩ | ⟨u,hu,rfl⟩ <;>
        rcases hv with ⟨v,hv,rfl⟩ | ⟨v,hv,rfl⟩ <;>
        obtain ⟨hx,htail⟩ := pref_cons hw <;>
        subst x <;>
        simp only [recvPayloads, AlphabetOf.mkRecv, ↓reduceIte] at hi
      · obtain ⟨s,hs⟩ := iht hu hv htail (pref_cons hi).2
        exact ⟨s,by simp [hs]⟩
      · have := (pref_cons hi).1; cases this
      · have := (pref_cons hi).1; cases this
      · obtain ⟨s,hs⟩ := ihf hu hv htail (pref_cons hi).2
        exact ⟨s,by simp [hs]⟩
def Represented {a : Role} (p : Local a) : Prop :=
  ∃ t : Tree a, ∀ w, localTraceSemantics p w ↔ accepts t w

theorem rep_eps (a : Role) : Represented (a := a) .eps :=
  ⟨.done, fun _ => Iff.rfl⟩
theorem rep_send (a b : Role) (x : Port) (h : a ≠ b) : Represented (a := a) (.send x b) := by
  refine ⟨.step (AlphabetOf.mkSend a x b h) .done, ?_⟩
  intro w
  simp [localTraceSemantics, accepts, h]
theorem rep_recv (a b : Role) (x : Port) : Represented (a := a) (.recv x b) := by
  refine ⟨.step (AlphabetOf.mkRecv a x b) .done, ?_⟩
  intro w; simp [localTraceSemantics, accepts]
theorem rep_act (a : Role) (x y : Port) (f : Action) : Represented (a := a) (.act x f y) := by
  refine ⟨.step (AlphabetOf.mkAct a x f y) .done, ?_⟩
  intro w; simp [localTraceSemantics, accepts]
theorem rep_seq {a : Role} {p q : Local a} (hp : Represented p) (hq : Represented q) :
    Represented (p ;;ₗ q) := by
  obtain ⟨t, ht⟩ := hp; obtain ⟨k,hk⟩ := hq
  exact ⟨join t k, fun w => by simp only [accepts_join, localTraceSemantics, ht, hk]⟩
theorem rep_branch {a : Role} {p q : Local a} (hp : Represented p) (hq : Represented q) :
    Represented (.recvIf (.ctrl false) agent p q) := by
  obtain ⟨t, ht⟩ := hp; obtain ⟨k,hk⟩ := hq
  exact ⟨.branch t k, fun w => by simp only [accepts, localTraceSemantics, ht, hk]; rfl⟩

theorem endpoints_represented (a : Role) (ha : a ≠ agent) : Represented (endpoints a) := by
  cases a <;> try contradiction
  all_goals unfold endpoints
  all_goals
    repeat first
      | apply rep_seq
      | apply rep_branch
      | exact rep_eps _
      | exact rep_recv _ _ _
      | exact rep_act _ _ _ _
      | exact rep_send _ _ _ (by decide)

theorem endpoint_align (a : Role) (ha : a ≠ agent) {w v : Word a}
    (hw : localPrefixSemantics (endpoints a) w)
    (hv : localTraceSemantics (endpoints a) v)
    (hi : Pref (recvPayloads agent w) (recvPayloads agent v)) : Pref w v := by
  obtain ⟨t,ht⟩ := endpoints_represented a ha
  obtain ⟨u,hu,hwu⟩ := hw
  exact tree_align t ((ht u).mp hu) ((ht v).mp hv) hwu hi
/-- The eight independent choices of a complete Agent trace. -/
structure Bits where
  b1 : Bool
  b2 : Bool
  b3 : Bool
  b4 : Bool
  b5 : Bool
  b6 : Bool
  b7 : Bool
  b8 : Bool

def aBlock (r : Role) (g : Guard) (hr : agent ≠ r) (b : Bool) : Word agent :=
  [if b then AlphabetOf.mkIfTrue agent g else AlphabetOf.mkIfFalse agent g,
   AlphabetOf.mkSend agent (.ctrl b) finalizer (by decide),
   AlphabetOf.mkSend agent (.ctrl b) r hr] ++
  if b then [AlphabetOf.mkSend agent .review r hr] else []
def rWord (r : Role) (hr : r ≠ finalizer) (b : Bool) : Word r :=
  [AlphabetOf.mkRecv r (.ctrl b) agent] ++
  if b then [AlphabetOf.mkRecv r .review agent,
    AlphabetOf.mkAct r .unit .assess .review,
    AlphabetOf.mkSend r .verdict finalizer hr] else []
def fBlock (r : Role) (b : Bool) : Word finalizer :=
  [AlphabetOf.mkRecv finalizer (.ctrl b) agent] ++
  if b then [AlphabetOf.mkRecv finalizer .verdict r] else []
def completed (b : Bits) : Tuple
  | agent => [AlphabetOf.mkSend agent .plan worker (by decide)] ++
      aBlock reviewer1 .need1 (by decide) b.b1 ++
      aBlock reviewer2 .need2 (by decide) b.b2 ++
      aBlock reviewer3 .need3 (by decide) b.b3 ++
      aBlock reviewer4 .need4 (by decide) b.b4 ++
      aBlock reviewer5 .need5 (by decide) b.b5 ++
      aBlock reviewer6 .need6 (by decide) b.b6 ++
      aBlock reviewer7 .need7 (by decide) b.b7 ++
      aBlock reviewer8 .need8 (by decide) b.b8
  | worker => [AlphabetOf.mkRecv worker .plan agent,
      AlphabetOf.mkAct worker .prepared .prepare .plan,
      AlphabetOf.mkSend worker .prepared finalizer (by decide)]
  | finalizer =>
      fBlock reviewer1 b.b1 ++ fBlock reviewer2 b.b2 ++
      fBlock reviewer3 b.b3 ++ fBlock reviewer4 b.b4 ++
      fBlock reviewer5 b.b5 ++ fBlock reviewer6 b.b6 ++
      fBlock reviewer7 b.b7 ++ fBlock reviewer8 b.b8 ++
      [AlphabetOf.mkRecv finalizer .prepared worker] ++
      [AlphabetOf.mkAct finalizer .unit .resolve .prepared]
  | reviewer1 => rWord reviewer1 (by decide) b.b1
  | reviewer2 => rWord reviewer2 (by decide) b.b2
  | reviewer3 => rWord reviewer3 (by decide) b.b3
  | reviewer4 => rWord reviewer4 (by decide) b.b4
  | reviewer5 => rWord reviewer5 (by decide) b.b5
  | reviewer6 => rWord reviewer6 (by decide) b.b6
  | reviewer7 => rWord reviewer7 (by decide) b.b7
  | reviewer8 => rWord reviewer8 (by decide) b.b8

theorem a_trace (r : Role) (g : Guard) (hr : agent ≠ r) (w : Word agent) :
    localTraceSemantics
      (.localIf g
        ((.send (.ctrl true) finalizer ;;ₗ (.send (.ctrl true) r ;;ₗ .eps)) ;;ₗ
          ((.send .review r ;;ₗ .eps) ;;ₗ .eps))
        ((.send (.ctrl false) finalizer ;;ₗ (.send (.ctrl false) r ;;ₗ .eps)) ;;ₗ .eps)) w
      ↔ ∃ b, w = aBlock r g hr b := by
  simp [localTraceSemantics, aBlock, hr, Bool.exists_bool, or_comm]

theorem trace_seq {a : Role} (p q : Local a) (w : Word a) :
    localTraceSemantics (p ;;ₗ q) w ↔ ∃ u v, localTraceSemantics p u ∧ localTraceSemantics q v ∧ w = u ++ v := Iff.rfl
theorem trace_eps {a : Role} (w : Word a) : localTraceSemantics (.eps : Local a) w ↔ w = [] := Iff.rfl
theorem trace_send {a : Role} (x : Port) (r : Role) (w : Word a) :
    localTraceSemantics (.send x r : Local a) w ↔ ∃ h : a ≠ r, w = [AlphabetOf.mkSend a x r h] := Iff.rfl

theorem agent_trace (w : Word agent) :
    localTraceSemantics (endpoints agent) w ↔ ∃ b, w = completed b agent := by
  simp only [endpoints, trace_seq, trace_eps, trace_send,
    a_trace reviewer1 .need1 (by decide),
    a_trace reviewer2 .need2 (by decide),
    a_trace reviewer3 .need3 (by decide),
    a_trace reviewer4 .need4 (by decide),
    a_trace reviewer5 .need5 (by decide),
    a_trace reviewer6 .need6 (by decide),
    a_trace reviewer7 .need7 (by decide),
    a_trace reviewer8 .need8 (by decide)]
  simp only [ne_eq, reduceCtorEq, not_false_eq_true, exists_true_left,
    List.append_nil, List.nil_append, exists_eq_left, exists_and_left, exists_and_right]
  constructor
  · rintro ⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨b1,rfl⟩,rfl⟩,_,⟨b2,rfl⟩,rfl⟩,_,⟨b3,rfl⟩,rfl⟩,_,⟨b4,rfl⟩,rfl⟩,_,⟨b5,rfl⟩,rfl⟩,_,⟨b6,rfl⟩,rfl⟩,_,⟨b7,rfl⟩,rfl⟩,_,⟨b8,rfl⟩,rfl⟩,rfl⟩,rfl⟩,rfl⟩
    exact ⟨⟨b1,b2,b3,b4,b5,b6,b7,b8⟩,rfl⟩
  · rintro ⟨b,rfl⟩
    exact ⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨_,⟨b.b1,rfl⟩,rfl⟩,_,⟨b.b2,rfl⟩,rfl⟩,_,⟨b.b3,rfl⟩,rfl⟩,_,⟨b.b4,rfl⟩,rfl⟩,_,⟨b.b5,rfl⟩,rfl⟩,_,⟨b.b6,rfl⟩,rfl⟩,_,⟨b.b7,rfl⟩,rfl⟩,_,⟨b.b8,rfl⟩,rfl⟩,rfl⟩,rfl⟩,rfl⟩
theorem r_trace (r : Role) (hr : r ≠ finalizer) (b : Bool) :
    localTraceSemantics
      (.recvIf (.ctrl false) agent
        ((.recv .review agent ;;ₗ .act .unit .assess .review) ;;ₗ .send .verdict finalizer)
        .eps) (rWord r hr b) := by
  cases b <;> simp [rWord, localTraceSemantics, hr, ControlPayload.setDecision]
theorem f_trace (r : Role) (b : Bool) :
    localTraceSemantics
      (.recvIf (.ctrl false) agent ((.eps ;;ₗ .eps) ;;ₗ .recv .verdict r) .eps) (fBlock r b) := by
  cases b <;> simp [fBlock, localTraceSemantics, ControlPayload.setDecision]

theorem completed_local (b : Bits) (a : Role) :
    localTraceSemantics (endpoints a) (completed b a) := by
  cases a
  · exact (agent_trace _).mpr ⟨b,rfl⟩
  · simp [endpoints, completed, localTraceSemantics]
  · have h0 : localTraceSemantics (.eps : Local finalizer) [] := rfl
    have h1 := localTraceSemantics_seq_intro _ _ _ _ h0 (f_trace reviewer1 b.b1)
    have h2 := localTraceSemantics_seq_intro _ _ _ _ h1 (f_trace reviewer2 b.b2)
    have h3 := localTraceSemantics_seq_intro _ _ _ _ h2 (f_trace reviewer3 b.b3)
    have h4 := localTraceSemantics_seq_intro _ _ _ _ h3 (f_trace reviewer4 b.b4)
    have h5 := localTraceSemantics_seq_intro _ _ _ _ h4 (f_trace reviewer5 b.b5)
    have h6 := localTraceSemantics_seq_intro _ _ _ _ h5 (f_trace reviewer6 b.b6)
    have h7 := localTraceSemantics_seq_intro _ _ _ _ h6 (f_trace reviewer7 b.b7)
    have h8 := localTraceSemantics_seq_intro _ _ _ _ h7 (f_trace reviewer8 b.b8)
    have h9 := localTraceSemantics_seq_intro _ _ _ _ h8
      (show localTraceSemantics (.eps : Local finalizer) [] from rfl)
    have h10 := localTraceSemantics_seq_intro _ _ _ _ h9
      (show localTraceSemantics (.recv .prepared worker : Local finalizer)
        [AlphabetOf.mkRecv finalizer .prepared worker] from rfl)
    have h11 := localTraceSemantics_seq_intro _ _ _ _ h10
      (show localTraceSemantics (.act .unit .resolve .prepared : Local finalizer)
        [AlphabetOf.mkAct finalizer .unit .resolve .prepared] from rfl)
    simpa only [List.nil_append, List.append_nil] using h11
  · simpa [endpoints, completed, localTraceSemantics] using r_trace reviewer1 (by decide) b.b1
  · simpa [endpoints, completed, localTraceSemantics] using r_trace reviewer2 (by decide) b.b2
  · simpa [endpoints, completed, localTraceSemantics] using r_trace reviewer3 (by decide) b.b3
  · simpa [endpoints, completed, localTraceSemantics] using r_trace reviewer4 (by decide) b.b4
  · simpa [endpoints, completed, localTraceSemantics] using r_trace reviewer5 (by decide) b.b5
  · simpa [endpoints, completed, localTraceSemantics] using r_trace reviewer6 (by decide) b.b6
  · simpa [endpoints, completed, localTraceSemantics] using r_trace reviewer7 (by decide) b.b7
  · simpa [endpoints, completed, localTraceSemantics] using r_trace reviewer8 (by decide) b.b8
@[simp] theorem sends_aBlock (r : Role) (g : Guard) (h : agent ≠ r) (b : Bool) (x : Role) :
    sendPayloads x (aBlock r g h b) =
      (if finalizer = x then [Port.ctrl b] else []) ++
      (if r = x then [Port.ctrl b] else []) ++
      (if b then (if r = x then [Port.review] else []) else []) := by
  cases b <;> by_cases hf : finalizer = x <;> by_cases hr : r = x <;>
    simp [aBlock, sendPayloads, AlphabetOf.mkSend, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse, hf, hr]
@[simp] theorem recvs_aBlock (r : Role) (g : Guard) (h : agent ≠ r) (b : Bool) (x : Role) :
    recvPayloads x (aBlock r g h b) = [] := by
  cases b <;> simp [aBlock, recvPayloads, AlphabetOf.mkSend, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse]
@[simp] theorem sends_rWord (r : Role) (h : r ≠ finalizer) (b : Bool) (x : Role) :
    sendPayloads x (rWord r h b) =
      (if b then (if finalizer = x then [Port.verdict] else []) else []) := by
  cases b <;> simp [rWord, sendPayloads, AlphabetOf.mkRecv, AlphabetOf.mkAct, AlphabetOf.mkSend]
@[simp] theorem recvs_rWord (r : Role) (h : r ≠ finalizer) (b : Bool) (x : Role) :
    recvPayloads x (rWord r h b) =
      (if agent = x then [Port.ctrl b] else []) ++
      (if b then (if agent = x then [Port.review] else []) else []) := by
  cases b <;> by_cases hx : agent = x <;>
    simp [rWord, recvPayloads, AlphabetOf.mkRecv, AlphabetOf.mkAct, AlphabetOf.mkSend, hx]
@[simp] theorem sends_fBlock (r : Role) (b : Bool) (x : Role) :
    sendPayloads x (fBlock r b) = [] := by
  cases b <;> simp [fBlock, sendPayloads, AlphabetOf.mkRecv]
@[simp] theorem recvs_fBlock (r : Role) (b : Bool) (x : Role) :
    recvPayloads x (fBlock r b) =
      (if agent = x then [Port.ctrl b] else []) ++
      (if b then (if r = x then [Port.verdict] else []) else []) := by
  cases b <;> by_cases ha : agent = x <;> by_cases hr : r = x <;>
    simp [fBlock, recvPayloads, AlphabetOf.mkRecv, ha, hr]

/-- Both control and application payloads balance on each FIFO channel. -/
theorem balanced (b : Bits) (a c : Role) :
    sendPayloads c (completed b a) = recvPayloads a (completed b c) := by
  cases a <;> cases c <;>
    simp [completed, sendPayloads, recvPayloads, AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkAct]
/-- All communication goes from a lower level to a higher level. -/
def level : Role → Nat
  | agent => 0
  | finalizer => 2
  | _ => 1
def Forward {a : Role} (w : Word a) : Prop :=
  ∀ l ∈ w, ∀ c, l.val.isSendTo c = true → level a < level c

@[simp] theorem forward_append {a : Role} (u v : Word a) :
    Forward (u ++ v) ↔ Forward u ∧ Forward v := by
  simp [Forward, List.mem_append, or_imp, forall_and]

theorem forward_aBlock (r : Role) (g : Guard) (h : agent ≠ r) (b : Bool)
    (hr : 0 < level r) : Forward (aBlock r g h b) := by
  cases b <;>
    simp [Forward, aBlock, AlphabetOf.mkSend, AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse,
      Letter.isSendTo, level, hr] at * <;> assumption
theorem forward_rWord (r : Role) (h : r ≠ finalizer) (b : Bool)
    (hr : level r < 2) : Forward (rWord r h b) := by
  cases b <;>
    simp [Forward, rWord, AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkAct,
      Letter.isSendTo, level, hr] at * <;> assumption
theorem forward_fBlock (r : Role) (b : Bool) : Forward (fBlock r b) := by
  cases b <;> simp [Forward, fBlock, AlphabetOf.mkRecv, Letter.isSendTo]

theorem completed_forward (b : Bits) (a : Role) : Forward (completed b a) := by
  cases a <;> simp only [completed, forward_append]
  all_goals
    repeat first
      | constructor
      | exact forward_aBlock _ _ _ _ (by decide)
      | exact forward_rWord _ _ _ (by decide)
      | exact forward_fBlock _ _
      | (simp [Forward, AlphabetOf.mkSend, AlphabetOf.mkRecv, AlphabetOf.mkAct,
          Letter.isSendTo, level])

theorem mem_le_sum {xs : List Nat} {x : Nat} (h : x ∈ xs) : x ≤ xs.sum := by
  induction xs with
  | nil => simp at h
  | cons y ys ih =>
    simp only [List.mem_cons] at h
    rcases h with rfl | h
    · simp
    · have := ih h; simp only [List.sum_cons]; omega

def width (M : Tuple) : Nat :=
  ((Fintype.elems : List Role).map (fun a => (M a).length)).sum + 1

theorem length_lt_width (M : Tuple) (a : Role) : (M a).length < width M := by
  have hmem : (M a).length ∈ ((Fintype.elems : List Role).map (fun a => (M a).length)) := by
    apply List.mem_map.mpr
    exact ⟨a, Fintype.complete a, rfl⟩
  have := mem_le_sum hmem
  unfold width; omega

/-- A role level, followed by a local position, ranks every causal edge. -/
theorem forward_acyclic (M : Tuple) (h : ∀ a, Forward (M a)) : hasAcyclicCausality M := by
  refine ⟨{ rank := fun e => level e.lifeline * width M + e.pos
            local_mono := ?_
            fifo_mono := ?_ }⟩
  · intro e1 e2 he hp _ _
    simp only [he]; omega
  · intro a c j k hj hk hs _ _
    have hl : level a < level c := h a _ (List.get_mem _ _) c hs
    have hb := length_lt_width M a
    have hm : (level a + 1) * width M ≤ level c * width M :=
      Nat.mul_le_mul_right (width M) hl
    simp only [Nat.add_mul, Nat.one_mul] at hm
    change level a * width M + j < level c * width M + k
    omega

theorem complete (b : Bits) : IsCompleteMSC (completed b) := by
  refine ⟨?_, ?_, forward_acyclic _ (completed_forward b)⟩
  · intro a c
    have := congrArg List.length (balanced b a c)
    simpa [channelComplete, sndCount, rcvCount] using this
  · intro a c p hp
    rw [balanced b a c] at hp
    have : p.1 = p.2 := by
      have hz : ∀ (xs : List Port) (q : Port × Port), q ∈ xs.zip xs → q.1 = q.2 := by
        intro xs
        induction xs with
        | nil => simp
        | cons x xs ih =>
          intro q hq
          simp only [List.zip_cons_cons, List.mem_cons] at hq
          rcases hq with rfl | hq
          · rfl
          · exact ih q hq
      exact hz _ _ hp
    exact this
end Direct

theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hM
  obtain ⟨wa, hwa, hpa⟩ := hM.1 agent
  obtain ⟨b, rfl⟩ := (Direct.agent_trace wa).mp hwa
  refine ⟨Direct.completed b, ⟨Direct.completed_local b, Direct.complete b⟩, ?_⟩
  intro a
  by_cases ha : a = agent
  · subst a; exact hpa
  · apply Direct.endpoint_align a ha (hM.1 a) (Direct.completed_local b a)
    rw [← Direct.balanced b agent a]
    exact Direct.pref_trans (Direct.fifo_prefix M hM.2 agent a) (Direct.pref_send a hpa)
