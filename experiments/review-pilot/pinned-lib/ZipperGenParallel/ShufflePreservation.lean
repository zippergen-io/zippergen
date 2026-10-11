import ZipperGenParallel.DependencyGraph
import ZipperGenParallel.Shuffle

namespace ZipperGenParallel

universe u v w x

/--
Pre-MSC data for a shuffled family of branch component MSCs.

The event type is `Tagged (Fin n) Event`: the tag records the branch from
which the event came.  We do not package this as an `MSC` yet, because the
acyclicity argument is separate and FIFO is an explicit obligation of the
accepted shuffle.  In the paper, branch origin is determined by occurrence/path
channel names; here it is kept as a proof-level event tag carrying the same
separation information.
-/
structure ShuffledMSCData (n : Nat) (Lifeline : Type u)
    (Channel : Type v) (Event : Type w) where
  branchMSC : Fin n → MSC Lifeline Channel Event
  localIndex : Tagged (Fin n) Event → Nat
  localIndex_pos : ∀ e, 0 < localIndex e
  localIndex_inj :
    ∀ {e f},
      (branchMSC e.branch).lifeline e.val =
        (branchMSC f.branch).lifeline f.val →
      localIndex e = localIndex f →
      e = f
  local_succ_same_branch_sound :
    ∀ {i : Fin n} {e f : Tagged (Fin n) Event},
      e.branch = i →
      f.branch = i →
      LocalSucc
        (fun x => (branchMSC x.branch).lifeline x.val)
        localIndex e f →
      MSC.strictCausal (branchMSC i) e.val f.val
  local_order_same_branch_sound :
    ∀ {i : Fin n} {e f : Tagged (Fin n) Event},
      e.branch = i →
      f.branch = i →
      localIndex e < localIndex f →
      (branchMSC i).localIndex e.val < (branchMSC i).localIndex f.val
  /--
  FIFO for the accepted shuffled tuple, over ordinary channel names.

  The two messages may come from different branches; this is the
  cross-branch FIFO obligation imposed by the MSC filter.
  -/
  channel_fifo :
    ∀ {s₁ s₂ r₁ r₂ : Tagged (Fin n) Event} {A B : Lifeline}
      {ch : Channel},
      (s₁.branch = r₁.branch ∧ (branchMSC s₁.branch).msg s₁.val r₁.val) →
      (s₂.branch = r₂.branch ∧ (branchMSC s₂.branch).msg s₂.val r₂.val) →
      (branchMSC s₁.branch).lifeline s₁.val = A →
      (branchMSC s₂.branch).lifeline s₂.val = A →
      (branchMSC r₁.branch).lifeline r₁.val = B →
      (branchMSC r₂.branch).lifeline r₂.val = B →
      (branchMSC s₁.branch).kind s₁.val = EventKind.send B ch →
      (branchMSC s₂.branch).kind s₂.val = EventKind.send B ch →
      (branchMSC r₁.branch).kind r₁.val = EventKind.recv A ch →
      (branchMSC r₂.branch).kind r₂.val = EventKind.recv A ch →
      localIndex s₁ < localIndex s₂ →
      localIndex r₁ < localIndex r₂

namespace ShuffledMSCData

def lifeline {n : Nat} {Lifeline : Type u} {Channel : Type v}
    {Event : Type w} (D : ShuffledMSCData n Lifeline Channel Event)
    (e : Tagged (Fin n) Event) : Lifeline :=
  (D.branchMSC e.branch).lifeline e.val

def kind {n : Nat} {Lifeline : Type u} {Channel : Type v}
    {Event : Type w} (D : ShuffledMSCData n Lifeline Channel Event)
    (e : Tagged (Fin n) Event) :
    EventKind Lifeline Channel :=
  (D.branchMSC e.branch).kind e.val

def msg {n : Nat} {Lifeline : Type u} {Channel : Type v}
    {Event : Type w} (D : ShuffledMSCData n Lifeline Channel Event)
    (s r : Tagged (Fin n) Event) : Prop :=
  s.branch = r.branch ∧ (D.branchMSC s.branch).msg s.val r.val

def baseEdge {n : Nat} {Lifeline : Type u} {Channel : Type v}
    {Event : Type w} (D : ShuffledMSCData n Lifeline Channel Event) :
    Tagged (Fin n) Event → Tagged (Fin n) Event → Prop :=
  BaseEdge D.lifeline D.localIndex D.msg

def sameBranchEdge {n : Nat} {Lifeline : Type u} {Channel : Type v}
    {Event : Type w} (D : ShuffledMSCData n Lifeline Channel Event)
    (i : Fin n) (e f : Tagged (Fin n) Event) : Prop :=
  e.branch = i ∧ f.branch = i ∧ D.baseEdge e f

theorem msg_same_branch {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {s r : Tagged (Fin n) Event} (h : D.msg s r) :
    s.branch = r.branch :=
  h.1

theorem baseEdge_branch_switch_lifeline {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {e f : Tagged (Fin n) Event}
    (h : D.baseEdge e f) (hbranch : e.branch ≠ f.branch) :
    D.lifeline e = D.lifeline f := by
  rcases h with hlocal | hmsg
  · exact hlocal.1
  · exact False.elim (hbranch (D.msg_same_branch hmsg))

theorem baseEdge_branch_switch_localSucc {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {e f : Tagged (Fin n) Event}
    (h : D.baseEdge e f) (hbranch : e.branch ≠ f.branch) :
    LocalSucc D.lifeline D.localIndex e f := by
  rcases h with hlocal | hmsg
  · exact hlocal
  · exact False.elim (hbranch (D.msg_same_branch hmsg))

theorem baseEdge_lifeline_switch_msg {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {e f : Tagged (Fin n) Event}
    (h : D.baseEdge e f) (hline : D.lifeline e ≠ D.lifeline f) :
    D.msg e f := by
  rcases h with hlocal | hmsg
  · exact False.elim (hline hlocal.1)
  · exact hmsg

theorem baseEdge_lifeline_switch_same_branch {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {e f : Tagged (Fin n) Event}
    (h : D.baseEdge e f) (hline : D.lifeline e ≠ D.lifeline f) :
    e.branch = f.branch :=
  D.msg_same_branch (D.baseEdge_lifeline_switch_msg h hline)

theorem baseEdge_same_lifeline_localIndex_lt {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {e f : Tagged (Fin n) Event}
    (h : D.baseEdge e f) (hline : D.lifeline e = D.lifeline f) :
    D.localIndex e < D.localIndex f := by
  rcases h with hlocal | hmsg
  · rw [hlocal.2]
    exact Nat.lt_succ_self _
  · rcases e with ⟨ei, ev⟩
    rcases f with ⟨fi, fv⟩
    dsimp [msg, lifeline] at hmsg hline
    rcases hmsg with ⟨hbranch, hmsg⟩
    subst fi
    exact False.elim (((D.branchMSC ei).msg_lifeline_ne hmsg) hline)

theorem msg_to_branch_causal {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {i : Fin n} {s r : Tagged (Fin n) Event}
    (hs : s.branch = i) (h : D.msg s r) :
    MSC.strictCausal (D.branchMSC i) s.val r.val := by
  subst i
  exact MSC.msg_causal (D.branchMSC s.branch) h.2

theorem sameBranchBaseEdge_to_branch_causal {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {i : Fin n} {e f : Tagged (Fin n) Event}
    (h : D.sameBranchEdge i e f) :
    MSC.strictCausal (D.branchMSC i) e.val f.val := by
  rcases h with ⟨he, hf, hedge⟩
  rcases hedge with hlocal | hmsg
  · exact D.local_succ_same_branch_sound he hf hlocal
  · exact D.msg_to_branch_causal he hmsg

theorem sameBranchPath_to_branch_causal {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {i : Fin n} {e f : Tagged (Fin n) Event}
    (h : Relation.TransGen (D.sameBranchEdge i) e f) :
    MSC.strictCausal (D.branchMSC i) e.val f.val := by
  induction h with
  | single hrel =>
      exact D.sameBranchBaseEdge_to_branch_causal hrel
  | tail hpath hrel ih =>
      exact Relation.TransGen.trans ih
        (D.sameBranchBaseEdge_to_branch_causal hrel)

theorem msg_send_recv {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {s r : Tagged (Fin n) Event} (h : D.msg s r) :
    ∃ A B ch,
      D.lifeline s = A ∧ D.lifeline r = B ∧
      D.kind s = EventKind.send B ch ∧
      D.kind r = EventKind.recv A ch := by
  rcases s with ⟨si, sv⟩
  rcases r with ⟨ri, rv⟩
  dsimp [msg] at h
  rcases h with ⟨hbranch, hmsg⟩
  subst ri
  rcases (D.branchMSC si).msg_send_recv hmsg with
    ⟨A, B, ch, hsA, hrB, hsend, hrecv⟩
  refine ⟨A, B, ch, ?_, ?_, ?_, ?_⟩
  · simpa [lifeline] using hsA
  · simpa [lifeline] using hrB
  · simpa [kind] using hsend
  · simpa [kind] using hrecv

theorem msg_lifeline_ne {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {s r : Tagged (Fin n) Event} (h : D.msg s r) :
    D.lifeline s ≠ D.lifeline r := by
  rcases s with ⟨si, sv⟩
  rcases r with ⟨ri, rv⟩
  dsimp [msg] at h
  rcases h with ⟨hbranch, hmsg⟩
  subst ri
  simpa [lifeline] using (D.branchMSC si).msg_lifeline_ne hmsg

theorem msg_send_unique {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {s r₁ r₂ : Tagged (Fin n) Event}
    (h₁ : D.msg s r₁) (h₂ : D.msg s r₂) : r₁ = r₂ := by
  rcases s with ⟨si, sv⟩
  rcases r₁ with ⟨r₁i, r₁v⟩
  rcases r₂ with ⟨r₂i, r₂v⟩
  dsimp [msg] at h₁ h₂
  rcases h₁ with ⟨hbranch₁, hmsg₁⟩
  rcases h₂ with ⟨hbranch₂, hmsg₂⟩
  subst r₁i
  subst r₂i
  have hval : r₁v = r₂v :=
    (D.branchMSC si).msg_send_unique hmsg₁ hmsg₂
  cases hval
  rfl

theorem msg_recv_unique {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {s₁ s₂ r : Tagged (Fin n) Event}
    (h₁ : D.msg s₁ r) (h₂ : D.msg s₂ r) : s₁ = s₂ := by
  rcases s₁ with ⟨s₁i, s₁v⟩
  rcases s₂ with ⟨s₂i, s₂v⟩
  rcases r with ⟨ri, rv⟩
  dsimp [msg] at h₁ h₂
  rcases h₁ with ⟨hbranch₁, hmsg₁⟩
  rcases h₂ with ⟨hbranch₂, hmsg₂⟩
  subst s₁i
  subst s₂i
  have hval : s₁v = s₂v :=
    (D.branchMSC ri).msg_recv_unique hmsg₁ hmsg₂
  cases hval
  rfl

theorem recv_matched {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {r : Tagged (Fin n) Event}
    (hrecv :
      ∃ A ch, D.kind r = EventKind.recv A ch) :
    ∃ s, D.msg s r := by
  rcases hrecv with ⟨A, ch, hkind⟩
  have hbranchRecv :
      ∃ A' ch', (D.branchMSC r.branch).kind r.val =
        EventKind.recv A' ch' := by
    exact ⟨A, ch, by simpa [kind] using hkind⟩
  rcases (D.branchMSC r.branch).recv_matched hbranchRecv with ⟨s, hs⟩
  refine ⟨{ branch := r.branch, val := s }, ?_⟩
  exact ⟨rfl, hs⟩

theorem channel_fifo_valid {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {s₁ s₂ r₁ r₂ : Tagged (Fin n) Event}
    {A B : Lifeline} {ch : Channel}
    (hmsg₁ : D.msg s₁ r₁) (hmsg₂ : D.msg s₂ r₂)
    (hs₁A : D.lifeline s₁ = A) (hs₂A : D.lifeline s₂ = A)
    (hr₁B : D.lifeline r₁ = B) (hr₂B : D.lifeline r₂ = B)
    (hks₁ : D.kind s₁ = EventKind.send B ch)
    (hks₂ : D.kind s₂ = EventKind.send B ch)
    (hkr₁ : D.kind r₁ = EventKind.recv A ch)
    (hkr₂ : D.kind r₂ = EventKind.recv A ch)
    (hslt : D.localIndex s₁ < D.localIndex s₂) :
    D.localIndex r₁ < D.localIndex r₂ :=
  D.channel_fifo hmsg₁ hmsg₂
    (by simpa [lifeline] using hs₁A)
    (by simpa [lifeline] using hs₂A)
    (by simpa [lifeline] using hr₁B)
    (by simpa [lifeline] using hr₂B)
    (by simpa [kind] using hks₁)
    (by simpa [kind] using hks₂)
    (by simpa [kind] using hkr₁)
    (by simpa [kind] using hkr₂)
    hslt

end ShuffledMSCData

/--
A branch-local causal segment between shared lifelines.

This is the abstract object obtained by cutting a hypothetical shuffled
causal cycle at branch switches: between two consecutive switches, the path
stays inside one branch.
-/
def BranchSegment {Branch : Type u} {Lifeline : Type v}
    {Channel : Type w} {Event : Type x}
    (shared : Lifeline → Prop)
    (M : Branch → MSC Lifeline Channel Event) :
    Lifeline → Lifeline → Prop :=
  fun A B =>
    shared A ∧ shared B ∧ A ≠ B ∧
      ∃ i e f,
        (M i).lifeline e = A ∧
        (M i).lifeline f = B ∧
        MSC.strictCausal (M i) e f

theorem branchSegment_to_srg {Branch : Type u} {Lifeline : Type v}
    {Channel : Type w} {Event : Type x}
    {M : Branch → MSC Lifeline Channel Event}
    {branchDG : Branch → Lifeline → Lifeline → Prop}
    {shared : Lifeline → Prop}
    (hsound : BranchDependencySoundness M branchDG)
    {A B : Lifeline}
    (hseg : BranchSegment shared M A B) :
    SRG shared (UnionDG branchDG) A B := by
  rcases hseg with ⟨hsharedA, hsharedB, hne, i, e, f, heA, hfB, hpath⟩
  have hne' : (M i).lifeline e ≠ (M i).lifeline f := by
    intro hsame
    exact hne (by rw [← heA, ← hfB, hsame])
  have hsrg :=
    branchDependencySoundness_to_srg (M := M) (branchDG := branchDG)
      (shared := shared) hsound i hpath
      (by simpa [heA] using hsharedA)
      (by simpa [hfB] using hsharedB)
      hne'
  simpa [heA, hfB] using hsrg

theorem branchSegment_path_to_srg_path {Branch : Type u}
    {Lifeline : Type v} {Channel : Type w} {Event : Type x}
    {M : Branch → MSC Lifeline Channel Event}
    {branchDG : Branch → Lifeline → Lifeline → Prop}
    {shared : Lifeline → Prop}
    (hsound : BranchDependencySoundness M branchDG)
    {A B : Lifeline}
    (hpath : Relation.TransGen (BranchSegment shared M) A B) :
    Relation.TransGen (SRG shared (UnionDG branchDG)) A B :=
  transGen_mono (fun hseg => branchSegment_to_srg hsound hseg) hpath

/--
The acyclicity contradiction used in shuffle preservation.

Once a shuffled causal cycle has been cut into branch-local causal segments,
well-formedness rules out the resulting cycle between shared lifelines.
-/
theorem no_branchSegment_cycle {Branch : Type u} {Lifeline : Type v}
    {Channel : Type w} {Event : Type x}
    {M : Branch → MSC Lifeline Channel Event}
    {branchDG : Branch → Lifeline → Lifeline → Prop}
    {shared : Lifeline → Prop}
    (hwell : WellFormedParallel shared (UnionDG branchDG))
    (hsound : BranchDependencySoundness M branchDG)
    (A : Lifeline) :
    ¬ Relation.TransGen (BranchSegment shared M) A A := by
  intro hcycle
  exact hwell A (branchSegment_path_to_srg_path hsound hcycle)

theorem shuffled_intra_branch_causal_to_segment {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {i : Fin n} {e f : Tagged (Fin n) Event}
    (he : e.branch = i) (hf : f.branch = i)
    (hpath : Relation.TransGen (D.sameBranchEdge i) e f)
    (hshared_e : shared (D.lifeline e))
    (hshared_f : shared (D.lifeline f))
    (hne : D.lifeline e ≠ D.lifeline f) :
    BranchSegment shared D.branchMSC (D.lifeline e) (D.lifeline f) := by
  refine ⟨hshared_e, hshared_f, hne, i, e.val, f.val, ?_, ?_, ?_⟩
  · simp [ShuffledMSCData.lifeline, he]
  · simp [ShuffledMSCData.lifeline, hf]
  · exact D.sameBranchPath_to_branch_causal hpath

/--
Branch switches in a shuffled base edge occur only on shared lifelines.

The lifeline equality part follows from `baseEdge_branch_switch_lifeline`.
Whether that lifeline is shared is a property of the concrete shuffle
construction and the paper's definition of shared lifelines.
-/
def BranchSwitchShared {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    (shared : Lifeline → Prop) : Prop :=
  ∀ {e f : Tagged (Fin n) Event},
    D.baseEdge e f → e.branch ≠ f.branch → shared (D.lifeline e)

theorem BranchSwitchShared.target_shared {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    {D : ShuffledMSCData n Lifeline Channel Event}
    {shared : Lifeline → Prop}
    (hswitch : BranchSwitchShared D shared)
    {e f : Tagged (Fin n) Event}
    (h : D.baseEdge e f) (hbranch : e.branch ≠ f.branch) :
    shared (D.lifeline f) := by
  have hline := D.baseEdge_branch_switch_lifeline h hbranch
  rw [← hline]
  exact hswitch h hbranch

/-- A concrete nonempty edge path.  This is isomorphic to `Relation.TransGen`
but easier to analyze when cutting paths at branch switches. -/
inductive EdgePath {α : Type u} (r : α → α → Prop) : α → α → Prop where
  | single {a b : α} : r a b → EdgePath r a b
  | tail {a b c : α} : EdgePath r a b → r b c → EdgePath r a c

namespace EdgePath

theorem toTransGen {α : Type u} {r : α → α → Prop} {a b : α}
    (p : EdgePath r a b) : Relation.TransGen r a b := by
  induction p with
  | single h =>
      exact Relation.TransGen.single h
  | tail p h ih =>
      exact Relation.TransGen.tail ih h

def ofTransGen {α : Type u} {r : α → α → Prop} {a b : α}
    (h : Relation.TransGen r a b) : EdgePath r a b := by
  induction h with
  | single hrel =>
      exact EdgePath.single hrel
  | tail _ hrel ih =>
      exact EdgePath.tail ih hrel

/-- A possibly empty concrete edge path. -/
inductive Walk {α : Type u} (r : α → α → Prop) : α → α → Prop where
  | refl {a : α} : Walk r a a
  | tail {a b c : α} : Walk r a b → r b c → Walk r a c

namespace Walk

theorem trans {α : Type u} {r : α → α → Prop} {a b c : α}
    (p : Walk r a b) (q : Walk r b c) : Walk r a c := by
  induction q with
  | refl =>
      exact p
  | tail q h ih =>
      exact Walk.tail ih h

theorem toEdgePath_or_eq {α : Type u} {r : α → α → Prop} {a b : α}
    (w : Walk r a b) : a = b ∨ EdgePath r a b := by
  induction w with
  | refl =>
      exact Or.inl rfl
  | tail w h ih =>
      rcases ih with hEq | p
      · cases hEq
        exact Or.inr (EdgePath.single h)
      · exact Or.inr (EdgePath.tail p h)

theorem trans_edge_toEdgePath {α : Type u} {r : α → α → Prop}
    {a b c : α} (w : Walk r a b) (h : r b c) : EdgePath r a c := by
  rcases toEdgePath_or_eq w with hEq | p
  · cases hEq
    exact EdgePath.single h
  · exact EdgePath.tail p h

end Walk

theorem toWalk {α : Type u} {r : α → α → Prop} {a b : α}
    (p : EdgePath r a b) : Walk r a b := by
  induction p with
  | single h =>
      exact Walk.tail Walk.refl h
  | tail _ h ih =>
      exact Walk.tail ih h

inductive AllBranch {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop}
    (i : Fin n) : {a b : Tagged (Fin n) Event} →
      EdgePath r a b → Prop where
  | single {a b : Tagged (Fin n) Event} (h : r a b) :
      a.branch = i → b.branch = i →
      AllBranch i (EdgePath.single h)
  | tail {a b c : Tagged (Fin n) Event} {p : EdgePath r a b}
      {h : r b c} :
      AllBranch i p → c.branch = i →
      AllBranch i (EdgePath.tail p h)

inductive HasBranchSwitch {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop} :
      {a b : Tagged (Fin n) Event} → EdgePath r a b → Prop where
  | single {a b : Tagged (Fin n) Event} {h : r a b} :
      a.branch ≠ b.branch →
      HasBranchSwitch (EdgePath.single h)
  | tail_left {a b c : Tagged (Fin n) Event} {p : EdgePath r a b}
      {h : r b c} :
      HasBranchSwitch p →
      HasBranchSwitch (EdgePath.tail p h)
  | tail_last {a b c : Tagged (Fin n) Event} {p : EdgePath r a b}
      {h : r b c} :
      b.branch ≠ c.branch →
      HasBranchSwitch (EdgePath.tail p h)

inductive AllSameBy {α : Type u} {β : Type v}
    {r : α → α → Prop} (q : α → β) :
      {a b : α} → EdgePath r a b → Prop where
  | single {a b : α} {h : r a b} :
      q a = q b →
      AllSameBy q (EdgePath.single h)
  | tail {a b c : α} {p : EdgePath r a b} {h : r b c} :
      AllSameBy q p → q b = q c →
      AllSameBy q (EdgePath.tail p h)

inductive HasSwitchBy {α : Type u} {β : Type v}
    {r : α → α → Prop} (q : α → β) :
      {a b : α} → EdgePath r a b → Prop where
  | single {a b : α} {h : r a b} :
      q a ≠ q b →
      HasSwitchBy q (EdgePath.single h)
  | tail_left {a b c : α} {p : EdgePath r a b} {h : r b c} :
      HasSwitchBy q p →
      HasSwitchBy q (EdgePath.tail p h)
  | tail_last {a b c : α} {p : EdgePath r a b} {h : r b c} :
      q b ≠ q c →
      HasSwitchBy q (EdgePath.tail p h)

theorem allBranch_start {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop}
    {a b : Tagged (Fin n) Event}
    (p : EdgePath r a b) {i : Fin n} :
    AllBranch i p → a.branch = i := by
  intro h
  induction h with
  | single _ ha _ =>
      exact ha
  | tail hp _ ih =>
      exact ih

theorem allBranch_end {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop}
    {a b : Tagged (Fin n) Event}
    (p : EdgePath r a b) {i : Fin n} :
    AllBranch i p → b.branch = i := by
  intro h
  induction h with
  | single _ _ hb =>
      exact hb
  | tail _ hb _ =>
      exact hb

theorem allBranch_to_sameBranchPath {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {i : Fin n} {a b : Tagged (Fin n) Event}
    (p : EdgePath D.baseEdge a b)
    (h : AllBranch i p) :
    Relation.TransGen (D.sameBranchEdge i) a b := by
  induction h with
  | single hrel ha hb =>
      exact Relation.TransGen.single ⟨ha, hb, hrel⟩
  | tail hp hb ih =>
      have hend := EdgePath.allBranch_end _ hp
      exact Relation.TransGen.tail ih ⟨hend, hb, by assumption⟩

theorem allBranch_or_hasBranchSwitch {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop} :
    ∀ {a b : Tagged (Fin n) Event} (p : EdgePath r a b),
    (∃ i : Fin n, AllBranch i p) ∨ HasBranchSwitch p := by
  intro a b p
  induction p with
  | single h =>
      rename_i y
      by_cases hbranch : a.branch = y.branch
      · exact Or.inl ⟨a.branch, AllBranch.single h rfl hbranch.symm⟩
      · exact Or.inr (HasBranchSwitch.single (h := h) hbranch)
  | tail p h ih =>
      rename_i y z
      rcases ih with hall | hswitch
      · rcases hall with ⟨i, hall⟩
        by_cases hc : z.branch = i
        · exact Or.inl ⟨i, AllBranch.tail (h := h) hall hc⟩
        · have hb : y.branch = i := allBranch_end p hall
          have hbc : y.branch ≠ z.branch := by
            intro hsame
            exact hc (by rw [← hsame, hb])
          exact Or.inr (HasBranchSwitch.tail_last (p := p) (h := h) hbc)
      · exact Or.inr (HasBranchSwitch.tail_left (p := p) (h := h) hswitch)

theorem allSameBy_or_hasSwitchBy {α : Type u} {β : Type v}
    {r : α → α → Prop} (q : α → β) :
    ∀ {a b : α} (p : EdgePath r a b),
    AllSameBy q p ∨ HasSwitchBy q p := by
  classical
  intro a b p
  induction p with
  | single h =>
      rename_i y
      by_cases hsame : q a = q y
      · exact Or.inl (AllSameBy.single (h := h) hsame)
      · exact Or.inr (HasSwitchBy.single (h := h) hsame)
  | tail p h ih =>
      rename_i y z
      rcases ih with hall | hswitch
      · by_cases hlast : q y = q z
        · exact Or.inl (AllSameBy.tail (h := h) hall hlast)
        · exact Or.inr (HasSwitchBy.tail_last (p := p) (h := h) hlast)
      · exact Or.inr (HasSwitchBy.tail_left (p := p) (h := h) hswitch)

theorem hasBranchSwitch_exists_edge {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop}
    {a b : Tagged (Fin n) Event}
    {p : EdgePath r a b}
    (h : HasBranchSwitch p) :
    ∃ x y : Tagged (Fin n) Event, r x y ∧ x.branch ≠ y.branch := by
  induction h with
  | single hne =>
      exact ⟨_, _, by assumption, hne⟩
  | tail_left _ ih =>
      exact ih
  | tail_last hne =>
      exact ⟨_, _, by assumption, hne⟩

theorem hasSwitchBy_exists_edge {α : Type u} {β : Type v}
    {r : α → α → Prop} {q : α → β}
    {a b : α} {p : EdgePath r a b}
    (h : HasSwitchBy q p) :
    ∃ x y : α, r x y ∧ q x ≠ q y := by
  induction h with
  | single hne =>
      exact ⟨_, _, by assumption, hne⟩
  | tail_left _ ih =>
      exact ih
  | tail_last hne =>
      exact ⟨_, _, by assumption, hne⟩

theorem hasSwitchBy_decompose {α : Type u} {β : Type v}
    {r : α → α → Prop} {q : α → β}
    {a b : α} {p : EdgePath r a b}
    (h : HasSwitchBy q p) :
    ∃ x y : α,
      Walk r a x ∧ r x y ∧ q x ≠ q y ∧ Walk r y b := by
  induction h with
  | single hne =>
      exact ⟨_, _, Walk.refl, by assumption, hne, Walk.refl⟩
  | tail_left _ ih =>
      rcases ih with ⟨x, y, hpre, hedge, hne, hsuf⟩
      exact ⟨x, y, hpre, hedge, hne, Walk.tail hsuf (by assumption)⟩
  | tail_last hne =>
      rename_i y z p hrel
      exact ⟨y, z, toWalk p, hrel, hne, Walk.refl⟩

theorem hasBranchSwitch_to_hasSwitchBy {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop}
    {a b : Tagged (Fin n) Event} {p : EdgePath r a b}
    (h : HasBranchSwitch p) :
    HasSwitchBy (fun x => x.branch) p := by
  induction h with
  | single hne =>
      rename_i y hrel
      exact HasSwitchBy.single (h := hrel) hne
  | tail_left hswitch ih =>
      rename_i y z hrel
      exact HasSwitchBy.tail_left (h := hrel) ih
  | tail_last hne =>
      rename_i y z p hrel
      exact HasSwitchBy.tail_last (p := p) (h := hrel) hne

theorem hasBranchSwitch_decompose {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop}
    {a b : Tagged (Fin n) Event} {p : EdgePath r a b}
    (h : HasBranchSwitch p) :
    ∃ x y : Tagged (Fin n) Event,
      Walk r a x ∧ r x y ∧ x.branch ≠ y.branch ∧ Walk r y b := by
  exact hasSwitchBy_decompose (hasBranchSwitch_to_hasSwitchBy h)

theorem trans {α : Type u} {r : α → α → Prop} {a b c : α}
    (p : EdgePath r a b) (q : EdgePath r b c) : EdgePath r a c := by
  induction q with
  | single h =>
      exact EdgePath.tail p h
  | tail _ h ih =>
      exact EdgePath.tail ih h

theorem allBranch_trans {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop}
    {a b c : Tagged (Fin n) Event}
    {p : EdgePath r a b} {q : EdgePath r b c} {i : Fin n}
    (hp : AllBranch i p) (hq : AllBranch i q) :
    AllBranch i (trans p q) := by
  induction hq with
  | single h _ hc =>
      exact AllBranch.tail (h := h) hp hc
  | tail _ hc ih =>
      exact AllBranch.tail (h := by assumption) ih hc

theorem hasBranchSwitch_trans_left {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop}
    {a b c : Tagged (Fin n) Event}
    {p : EdgePath r a b} (q : EdgePath r b c)
    (hp : HasBranchSwitch p) :
    HasBranchSwitch (trans p q) := by
  induction q with
  | single h =>
      exact HasBranchSwitch.tail_left (h := h) hp
  | tail _ h ih =>
      exact HasBranchSwitch.tail_left (h := h) ih

theorem not_hasBranchSwitch_allBranch {n : Nat} {Event : Type u}
    {r : Tagged (Fin n) Event → Tagged (Fin n) Event → Prop}
    {a b : Tagged (Fin n) Event}
    {p : EdgePath r a b}
    (h : ¬ HasBranchSwitch p) :
    ∃ i : Fin n, AllBranch i p := by
  rcases allBranch_or_hasBranchSwitch p with hall | hswitch
  · exact hall
  · exact False.elim (h hswitch)

end EdgePath

theorem allSameLifelinePath_localIndex_lt {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {e f : Tagged (Fin n) Event}
    (p : EdgePath D.baseEdge e f)
    (hsame : EdgePath.AllSameBy D.lifeline p) :
    D.localIndex e < D.localIndex f := by
  induction hsame with
  | single hline =>
      exact D.baseEdge_same_lifeline_localIndex_lt (by assumption) hline
  | tail hsame_tail hline ih =>
      exact Nat.lt_trans ih
        (D.baseEdge_same_lifeline_localIndex_lt (by assumption) hline)

theorem cycle_has_lifeline_switch {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {e : Tagged (Fin n) Event}
    (p : EdgePath D.baseEdge e e) :
    EdgePath.HasSwitchBy D.lifeline p := by
  rcases EdgePath.allSameBy_or_hasSwitchBy D.lifeline p with hsame | hswitch
  · have hlt := allSameLifelinePath_localIndex_lt D p hsame
    exact False.elim ((Nat.lt_irrefl (D.localIndex e)) hlt)
  · exact hswitch

theorem cycle_has_lifeline_switch_edge {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {e : Tagged (Fin n) Event}
    (p : EdgePath D.baseEdge e e) :
    ∃ x y : Tagged (Fin n) Event,
      D.baseEdge x y ∧ D.lifeline x ≠ D.lifeline y := by
  exact EdgePath.hasSwitchBy_exists_edge (cycle_has_lifeline_switch D p)

theorem cycle_lifeline_switch_decompose {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {e : Tagged (Fin n) Event}
    (p : EdgePath D.baseEdge e e) :
    ∃ x y : Tagged (Fin n) Event,
      EdgePath.Walk D.baseEdge e x ∧
      D.baseEdge x y ∧
      D.lifeline x ≠ D.lifeline y ∧
      x.branch = y.branch ∧
      EdgePath.Walk D.baseEdge y e := by
  rcases EdgePath.hasSwitchBy_decompose (cycle_has_lifeline_switch D p) with
    ⟨x, y, hpre, hedge, hline, hsuf⟩
  exact ⟨x, y, hpre, hedge, hline,
    D.baseEdge_lifeline_switch_same_branch hedge hline, hsuf⟩

theorem hasBranchSwitch_exists_shared_edge {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    (hswitchShared : BranchSwitchShared D shared)
    {e f : Tagged (Fin n) Event}
    {p : EdgePath D.baseEdge e f}
    (hswitch : p.HasBranchSwitch) :
    ∃ x y : Tagged (Fin n) Event,
      D.baseEdge x y ∧
      x.branch ≠ y.branch ∧
      D.lifeline x = D.lifeline y ∧
      shared (D.lifeline x) ∧
      shared (D.lifeline y) := by
  rcases EdgePath.hasBranchSwitch_exists_edge hswitch with
    ⟨x, y, hedge, hbranch⟩
  exact ⟨x, y, hedge, hbranch,
    D.baseEdge_branch_switch_lifeline hedge hbranch,
    hswitchShared hedge hbranch,
    BranchSwitchShared.target_shared hswitchShared hedge hbranch⟩

theorem hasBranchSwitch_decompose_shared {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    (hswitchShared : BranchSwitchShared D shared)
    {e f : Tagged (Fin n) Event}
    {p : EdgePath D.baseEdge e f}
    (hswitch : p.HasBranchSwitch) :
    ∃ x y : Tagged (Fin n) Event,
      EdgePath.Walk D.baseEdge e x ∧
      D.baseEdge x y ∧
      x.branch ≠ y.branch ∧
      D.lifeline x = D.lifeline y ∧
      shared (D.lifeline x) ∧
      shared (D.lifeline y) ∧
      EdgePath.Walk D.baseEdge y f := by
  rcases EdgePath.hasBranchSwitch_decompose hswitch with
    ⟨x, y, hpre, hedge, hbranch, hsuf⟩
  exact ⟨x, y, hpre, hedge, hbranch,
    D.baseEdge_branch_switch_lifeline hedge hbranch,
    hswitchShared hedge hbranch,
    BranchSwitchShared.target_shared hswitchShared hedge hbranch,
    hsuf⟩

theorem allBranch_path_to_segment {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {i : Fin n} {e f : Tagged (Fin n) Event}
    (p : EdgePath D.baseEdge e f)
    (hall : p.AllBranch i)
    (hshared_e : shared (D.lifeline e))
    (hshared_f : shared (D.lifeline f))
    (hne : D.lifeline e ≠ D.lifeline f) :
    BranchSegment shared D.branchMSC (D.lifeline e) (D.lifeline f) := by
  exact shuffled_intra_branch_causal_to_segment D
    (EdgePath.allBranch_start p hall)
    (EdgePath.allBranch_end p hall)
    (EdgePath.allBranch_to_sameBranchPath D p hall)
    hshared_e hshared_f hne

theorem lifelineSwitchEdge_to_segment {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {e f : Tagged (Fin n) Event}
    (h : D.baseEdge e f)
    (hline : D.lifeline e ≠ D.lifeline f)
    (hshared_e : shared (D.lifeline e))
    (hshared_f : shared (D.lifeline f)) :
    BranchSegment shared D.branchMSC (D.lifeline e) (D.lifeline f) := by
  let p : EdgePath D.baseEdge e f := EdgePath.single h
  have hbranch : e.branch = f.branch :=
    D.baseEdge_lifeline_switch_same_branch h hline
  have hall : p.AllBranch e.branch :=
    EdgePath.AllBranch.single h rfl hbranch.symm
  exact allBranch_path_to_segment D p hall hshared_e hshared_f hline

/--
The remaining path-cutting obligation for shuffle preservation.

It separates an arbitrary shuffled causal cycle into either a same-branch
cycle, already impossible in the corresponding branch MSC, or a cycle of
branch segments between shared lifelines.
-/
structure ShuffledCycleCutSound {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    (shared : Lifeline → Prop) : Prop where
  cut_cycle :
    ∀ {e : Tagged (Fin n) Event},
      Relation.TransGen D.baseEdge e e →
      (∃ i : Fin n, Relation.TransGen (D.sameBranchEdge i) e e) ∨
      ∃ A : Lifeline,
        Relation.TransGen (BranchSegment shared D.branchMSC) A A

/--
Path-level version of `ShuffledCycleCutSound`.

The remaining combinatorial work should prove this predicate by cutting a
concrete cycle path at branch switches.
-/
structure ShuffledPathCutSound {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    (shared : Lifeline → Prop) : Prop where
  cut_path :
    ∀ {e : Tagged (Fin n) Event} (p : EdgePath D.baseEdge e e),
      (∃ i : Fin n, p.AllBranch i) ∨
      ∃ A : Lifeline,
        Relation.TransGen (BranchSegment shared D.branchMSC) A A

/--
The remaining branch-switch case of the path cut.

Once the all-one-branch case is discharged by
`EdgePath.allBranch_or_hasBranchSwitch`, only paths with an actual branch-switch
edge need to be converted into `BranchSegment` cycles.
-/
structure BranchSwitchCycleSound {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    (shared : Lifeline → Prop) : Prop where
  cut_switch_cycle :
    ∀ {e : Tagged (Fin n) Event} (p : EdgePath D.baseEdge e e),
      p.HasBranchSwitch →
      ∃ A : Lifeline,
        Relation.TransGen (BranchSegment shared D.branchMSC) A A

theorem BranchSwitchCycleSound.toPathCutSound {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    {D : ShuffledMSCData n Lifeline Channel Event}
    {shared : Lifeline → Prop}
    (hswitch : BranchSwitchCycleSound D shared) :
    ShuffledPathCutSound D shared where
  cut_path := by
    intro e p
    rcases EdgePath.allBranch_or_hasBranchSwitch p with hall | hhasSwitch
    · exact Or.inl hall
    · exact Or.inr (hswitch.cut_switch_cycle p hhasSwitch)

theorem ShuffledPathCutSound.toCycleCutSound {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    {D : ShuffledMSCData n Lifeline Channel Event}
    {shared : Lifeline → Prop}
    (hcut : ShuffledPathCutSound D shared) :
    ShuffledCycleCutSound D shared where
  cut_cycle := by
    intro e hcycle
    let p := EdgePath.ofTransGen hcycle
    rcases hcut.cut_path p with hsame | hsegments
    · rcases hsame with ⟨i, hall⟩
      exact Or.inl ⟨i, EdgePath.allBranch_to_sameBranchPath D p hall⟩
    · exact Or.inr hsegments

theorem shuffled_base_acyclic {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {branchDG : Fin n → Lifeline → Lifeline → Prop}
    (hwell : WellFormedParallel shared (UnionDG branchDG))
    (hsound : BranchDependencySoundness D.branchMSC branchDG)
    (hcut : ShuffledCycleCutSound D shared) :
    Acyclic D.baseEdge := by
  intro e hcycle
  rcases hcut.cut_cycle hcycle with hsame | hsegments
  · rcases hsame with ⟨i, hsameCycle⟩
    exact (D.branchMSC i).acyclic e.val
      (D.sameBranchPath_to_branch_causal hsameCycle)
  · rcases hsegments with ⟨A, hsegments⟩
    exact no_branchSegment_cycle hwell hsound A hsegments

def ShuffledMSCData.toMSC {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {branchDG : Fin n → Lifeline → Lifeline → Prop}
    (hwell : WellFormedParallel shared (UnionDG branchDG))
    (hsound : BranchDependencySoundness D.branchMSC branchDG)
    (hcut : ShuffledCycleCutSound D shared) :
    TaggedMSC (Fin n) Lifeline Channel Event where
  lifeline := D.lifeline
  kind := D.kind
  localIndex := D.localIndex
  localIndex_pos := D.localIndex_pos
  localIndex_inj := D.localIndex_inj
  msg := D.msg
  msg_send_recv := D.msg_send_recv
  msg_lifeline_ne := D.msg_lifeline_ne
  msg_send_unique := D.msg_send_unique
  msg_recv_unique := D.msg_recv_unique
  recv_matched := D.recv_matched
  acyclic := shuffled_base_acyclic D hwell hsound hcut
  channel_fifo := D.channel_fifo

def shuffle_preserves_msc {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {branchDG : Fin n → Lifeline → Lifeline → Prop}
    (hwell : WellFormedParallel shared (UnionDG branchDG))
    (hsound : BranchDependencySoundness D.branchMSC branchDG)
    (hcut : ShuffledCycleCutSound D shared) :
    TaggedMSC (Fin n) Lifeline Channel Event :=
  D.toMSC hwell hsound hcut

theorem ShuffledMSCData.taggedMessagesSameBranch_toMSC {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {branchDG : Fin n → Lifeline → Lifeline → Prop}
    (hwell : WellFormedParallel shared (UnionDG branchDG))
    (hsound : BranchDependencySoundness D.branchMSC branchDG)
    (hcut : ShuffledCycleCutSound D shared) :
    TaggedMessagesSameBranch (D.toMSC hwell hsound hcut) := by
  intro s r hmsg
  exact hmsg.1

def ShuffledMSCData.branchRestriction {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {branchDG : Fin n → Lifeline → Lifeline → Prop}
    (hwell : WellFormedParallel shared (UnionDG branchDG))
    (hsound : BranchDependencySoundness D.branchMSC branchDG)
    (hcut : ShuffledCycleCutSound D shared) (i : Fin n) :
    MSC Lifeline Channel {e : Tagged (Fin n) Event // e.branch = i} :=
  taggedBranchRestriction (D.toMSC hwell hsound hcut)
    (D.taggedMessagesSameBranch_toMSC hwell hsound hcut) i

theorem ShuffledMSCData.branchRestriction_complete {n : Nat}
    {Lifeline : Type u} {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {branchDG : Fin n → Lifeline → Lifeline → Prop}
    (hwell : WellFormedParallel shared (UnionDG branchDG))
    (hsound : BranchDependencySoundness D.branchMSC branchDG)
    (hcut : ShuffledCycleCutSound D shared) (i : Fin n)
    (hcomplete :
      MSC.IsComplete (D.toMSC hwell hsound hcut)) :
    MSC.IsComplete (D.branchRestriction hwell hsound hcut i) :=
  taggedBranchRestriction_complete (D.toMSC hwell hsound hcut)
    (D.taggedMessagesSameBranch_toMSC hwell hsound hcut) i hcomplete

theorem shuffle_preserves_complete_msc {n : Nat} {Lifeline : Type u}
    {Channel : Type v} {Event : Type w}
    (D : ShuffledMSCData n Lifeline Channel Event)
    {shared : Lifeline → Prop}
    {branchDG : Fin n → Lifeline → Lifeline → Prop}
    (hwell : WellFormedParallel shared (UnionDG branchDG))
    (hsound : BranchDependencySoundness D.branchMSC branchDG)
    (hcut : ShuffledCycleCutSound D shared)
    (hcomplete : ∀ i, MSC.IsComplete (D.branchMSC i)) :
    MSC.IsComplete (shuffle_preserves_msc D hwell hsound hcut) := by
  intro s hsend
  rcases hsend with ⟨B, ch, hkind⟩
  have hbranchSend :
      MSC.IsSend (D.branchMSC s.branch) s.val := by
    exact ⟨B, ch, by simpa [shuffle_preserves_msc, ShuffledMSCData.toMSC,
      ShuffledMSCData.kind] using hkind⟩
  rcases hcomplete s.branch s.val hbranchSend with ⟨r, hmsg⟩
  refine ⟨{ branch := s.branch, val := r }, ?_⟩
  exact ⟨rfl, hmsg⟩

end ZipperGenParallel
