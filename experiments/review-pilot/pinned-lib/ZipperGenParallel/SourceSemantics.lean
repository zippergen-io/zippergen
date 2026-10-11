import ZipperGenParallel.SourceProjection
import ZipperGenParallel.Zipper

namespace ZipperGenParallel

universe u v

namespace Source
namespace Prog

def emptyTuple (Lifeline : Type u) :
    TraceTuple Lifeline (LocalLetter Lifeline ChannelName) :=
  TraceTuple.empty Lifeline (LocalLetter Lifeline ChannelName)

def actionTuple {Lifeline : Type u} [DecidableEq Lifeline]
    (owner : Lifeline) :
    TraceTuple Lifeline (LocalLetter Lifeline ChannelName) :=
  fun A =>
    if A = owner then
      [LocalLetter.action]
    else
      []

def messageTuple {Lifeline : Type u} [DecidableEq Lifeline]
    (sender receiver : Lifeline) (ch : ChannelName) :
    TraceTuple Lifeline (LocalLetter Lifeline ChannelName) :=
  fun A =>
    if A = sender then
      [LocalLetter.send receiver ch]
    else if A = receiver then
      [LocalLetter.recv sender ch]
    else
      []

private theorem no_event_emptyTuple {Lifeline : Type u}
    (e :
      WordEvent
        (emptyTuple Lifeline :
          TraceTuple Lifeline (LocalLetter Lifeline ChannelName))) :
    False := by
  cases e with
  | mk owner index inBounds =>
      exact Nat.not_lt_zero index
        (by simpa [emptyTuple, TraceTuple.empty] using inBounds)

private theorem no_base_edge_false_msg_of_no_local_succ
    {Lifeline : Type u} {Channel : Type v}
    {words : TraceTuple Lifeline (LocalLetter Lifeline Channel)}
    (hlocal :
      ∀ e f : WordEvent words,
        ¬ LocalSucc (fun e : WordEvent words => e.owner)
            (fun e : WordEvent words => e.localIndex) e f) :
    Acyclic
      (BaseEdge
        (fun e : WordEvent words => e.owner)
        (fun e : WordEvent words => e.localIndex)
        (fun _ _ => False)) := by
  intro e hcycle
  have noEdge :
      ∀ e f : WordEvent words,
        ¬ BaseEdge (fun e : WordEvent words => e.owner)
            (fun e : WordEvent words => e.localIndex)
            (fun _ _ => False) e f := by
    intro e f h
    rcases h with hsucc | hmsg
    · exact hlocal e f hsucc
    · exact hmsg
  have noPath :
      ∀ {e f : WordEvent words},
        Relation.TransGen
          (BaseEdge (fun e : WordEvent words => e.owner)
            (fun e : WordEvent words => e.localIndex)
            (fun _ _ => False)) e f →
        False := by
    intro e f hpath
    induction hpath with
    | single h =>
        exact noEdge _ _ h
    | tail _ h ih =>
        exact noEdge _ _ h
  exact noPath hcycle

theorem emptyTuple_completeMSC {Lifeline : Type u} :
    WordIsCompleteMSC
      (emptyTuple Lifeline :
        TraceTuple Lifeline (LocalLetter Lifeline ChannelName)) := by
  refine ⟨?_, ?_⟩
  · refine
      { msg := fun _ _ => False
        msg_send_recv := ?_
        msg_letter_match := ?_
        msg_lifeline_ne := ?_
        msg_send_unique := ?_
        msg_recv_unique := ?_
        recv_matched := ?_
        acyclic := ?_
        channel_fifo := ?_ }
    · intro _ _ h
      exact False.elim h
    · intro _ _ h
      exact False.elim h
    · intro _ _ h
      exact False.elim h
    · intro _ _ _ h
      exact False.elim h
    · intro _ _ _ h
      exact False.elim h
    · intro r _hrecv
      exact False.elim (no_event_emptyTuple r)
    · intro e hcycle
      exact False.elim (no_event_emptyTuple e)
    · intro _ _ _ _ _ _ _ h
      exact False.elim h
  · intro s _hsend
    exact False.elim (no_event_emptyTuple s)

private theorem actionTuple_event_owner {Lifeline : Type u}
    [DecidableEq Lifeline] {owner : Lifeline}
    (e : WordEvent (actionTuple owner)) :
    e.owner = owner := by
  cases e with
  | mk eventOwner index inBounds =>
      by_cases howner : eventOwner = owner
      · exact howner
      · exact False.elim
          (Nat.not_lt_zero index
            (by simpa [actionTuple, howner] using inBounds))

private theorem actionTuple_event_index_zero {Lifeline : Type u}
    [DecidableEq Lifeline] {owner : Lifeline}
    (e : WordEvent (actionTuple owner)) :
    e.index = 0 := by
  cases e with
  | mk eventOwner index inBounds =>
      have howner : eventOwner = owner := by
        by_cases howner : eventOwner = owner
        · exact howner
        · exact False.elim
            (Nat.not_lt_zero index
              (by simpa [actionTuple, howner] using inBounds))
      exact Nat.lt_one_iff.mp (by simpa [actionTuple, howner] using inBounds)

private theorem actionTuple_no_local_succ {Lifeline : Type u}
    [DecidableEq Lifeline] {owner : Lifeline}
    (e f : WordEvent (actionTuple owner)) :
    ¬ LocalSucc (fun e : WordEvent (actionTuple owner) => e.owner)
        (fun e : WordEvent (actionTuple owner) => e.localIndex) e f := by
  intro hsucc
  have he : e.index = 0 := actionTuple_event_index_zero e
  have hf : f.index = 0 := actionTuple_event_index_zero f
  have hidx : f.localIndex = e.localIndex + 1 := hsucc.2
  simp [WordEvent.localIndex, he, hf] at hidx

private theorem actionTuple_not_recv {Lifeline : Type u}
    [DecidableEq Lifeline] {owner : Lifeline}
    (e : WordEvent (actionTuple owner)) :
    ¬ ∃ A ch, e.kind = EventKind.recv A ch := by
  intro hrecv
  rcases hrecv with ⟨A, ch, hkind⟩
  cases e with
  | mk eventOwner index inBounds =>
      have howner : eventOwner = owner :=
        actionTuple_event_owner
          ({ owner := eventOwner, index := index, inBounds := inBounds } :
            WordEvent (actionTuple owner))
      subst eventOwner
      have hindex : index = 0 :=
        actionTuple_event_index_zero
          ({ owner := owner, index := index, inBounds := inBounds } :
            WordEvent (actionTuple owner))
      subst index
      have hkindLocal :
          ({ owner := owner, index := 0, inBounds := inBounds } :
            WordEvent (actionTuple owner)).kind = EventKind.local := by
        simp [WordEvent.kind, WordEvent.letter, actionTuple]
      have hkind' :
          ({ owner := owner, index := 0, inBounds := inBounds } :
            WordEvent (actionTuple owner)).kind = EventKind.recv A ch := by
        exact hkind
      rw [hkindLocal] at hkind'
      cases hkind'

private theorem actionTuple_not_send {Lifeline : Type u}
    [DecidableEq Lifeline] {owner : Lifeline}
    (e : WordEvent (actionTuple owner)) :
    ¬ MSC.IsSend
        (WordMSCData.toMSC
          ({ msg := fun _ _ => False
             msg_send_recv := by intro _ _ h; exact False.elim h
             msg_letter_match := by intro _ _ h; exact False.elim h
             msg_lifeline_ne := by intro _ _ h; exact False.elim h
             msg_send_unique := by intro _ _ _ h; exact False.elim h
             msg_recv_unique := by intro _ _ _ h; exact False.elim h
             recv_matched := by
               intro r hrecv
               exact False.elim (actionTuple_not_recv r hrecv)
             acyclic :=
               no_base_edge_false_msg_of_no_local_succ
                (fun e f => actionTuple_no_local_succ e f)
             channel_fifo := by
               intro _ _ _ _ _ _ _ h
               exact False.elim h } :
            WordMSCData (actionTuple owner))) e := by
  intro hsend
  rcases hsend with ⟨B, ch, hkind⟩
  cases e with
  | mk eventOwner index inBounds =>
      have howner : eventOwner = owner :=
        actionTuple_event_owner
          ({ owner := eventOwner, index := index, inBounds := inBounds } :
            WordEvent (actionTuple owner))
      subst eventOwner
      have hindex : index = 0 :=
        actionTuple_event_index_zero
          ({ owner := owner, index := index, inBounds := inBounds } :
            WordEvent (actionTuple owner))
      subst index
      have hkindLocal :
          ({ owner := owner, index := 0, inBounds := inBounds } :
            WordEvent (actionTuple owner)).kind = EventKind.local := by
        simp [WordEvent.kind, WordEvent.letter, actionTuple]
      have hkind' :
          ({ owner := owner, index := 0, inBounds := inBounds } :
            WordEvent (actionTuple owner)).kind = EventKind.send B ch := by
        simpa [WordMSCData.toMSC] using hkind
      rw [hkindLocal] at hkind'
      cases hkind'

theorem actionTuple_completeMSC {Lifeline : Type u} [DecidableEq Lifeline]
    (owner : Lifeline) :
    WordIsCompleteMSC (actionTuple owner) := by
  let D : WordMSCData (actionTuple owner) :=
    { msg := fun _ _ => False
      msg_send_recv := by intro _ _ h; exact False.elim h
      msg_letter_match := by intro _ _ h; exact False.elim h
      msg_lifeline_ne := by intro _ _ h; exact False.elim h
      msg_send_unique := by intro _ _ _ h; exact False.elim h
      msg_recv_unique := by intro _ _ _ h; exact False.elim h
      recv_matched := by
        intro r hrecv
        exact False.elim (actionTuple_not_recv r hrecv)
      acyclic :=
        no_base_edge_false_msg_of_no_local_succ
          (fun e f => actionTuple_no_local_succ e f)
      channel_fifo := by
        intro _ _ _ _ _ _ _ h
        exact False.elim h }
  refine ⟨D, ?_⟩
  intro s hsend
  exact False.elim (actionTuple_not_send s hsend)

private theorem messageTuple_event_owner_sender_or_receiver
    {Lifeline : Type u} [DecidableEq Lifeline]
    {sender receiver : Lifeline} (_hne : sender ≠ receiver)
    {ch : ChannelName} (e : WordEvent (messageTuple sender receiver ch)) :
    e.owner = sender ∨ e.owner = receiver := by
  cases e with
  | mk eventOwner index inBounds =>
      by_cases hsender : eventOwner = sender
      · exact Or.inl hsender
      · by_cases hreceiver : eventOwner = receiver
        · exact Or.inr hreceiver
        · exact False.elim
            (Nat.not_lt_zero index
              (by
                simpa [messageTuple, hsender, hreceiver] using inBounds))

private theorem messageTuple_event_index_zero {Lifeline : Type u}
    [DecidableEq Lifeline] {sender receiver : Lifeline}
    (hne : sender ≠ receiver) {ch : ChannelName}
    (e : WordEvent (messageTuple sender receiver ch)) :
    e.index = 0 := by
  cases e with
  | mk eventOwner index inBounds =>
      by_cases hsender : eventOwner = sender
      · exact Nat.lt_one_iff.mp
          (by simpa [messageTuple, hsender] using inBounds)
      · have hreceiver : eventOwner = receiver := by
          by_cases hreceiver : eventOwner = receiver
          · exact hreceiver
          · exact False.elim
              (Nat.not_lt_zero index
                (by
                  simpa [messageTuple, hsender, hreceiver] using inBounds))
        exact Nat.lt_one_iff.mp
          (by
            have hnotSender : receiver ≠ sender := by
              intro h
              exact hne h.symm
            simpa [messageTuple, hreceiver, hnotSender] using inBounds)

private theorem messageTuple_sender_letter {Lifeline : Type u}
    [DecidableEq Lifeline] {sender receiver : Lifeline}
    (hne : sender ≠ receiver) {ch : ChannelName}
    (e : WordEvent (messageTuple sender receiver ch))
    (howner : e.owner = sender) :
    e.letter = LocalLetter.send receiver ch := by
  cases e with
  | mk eventOwner index inBounds =>
      dsimp at howner
      subst eventOwner
      have hindex : index = 0 :=
        messageTuple_event_index_zero hne
          ({ owner := sender, index := index, inBounds := inBounds } :
            WordEvent (messageTuple sender receiver ch))
      subst index
      simp [WordEvent.letter, messageTuple]

private theorem messageTuple_receiver_letter {Lifeline : Type u}
    [DecidableEq Lifeline] {sender receiver : Lifeline}
    (hne : sender ≠ receiver) {ch : ChannelName}
    (e : WordEvent (messageTuple sender receiver ch))
    (howner : e.owner = receiver) :
    e.letter = LocalLetter.recv sender ch := by
  cases e with
  | mk eventOwner index inBounds =>
      dsimp at howner
      subst eventOwner
      have hindex : index = 0 :=
        messageTuple_event_index_zero hne
          ({ owner := receiver, index := index, inBounds := inBounds } :
            WordEvent (messageTuple sender receiver ch))
      subst index
      have hnotSender : receiver ≠ sender := by
        intro h
        exact hne h.symm
      simp [WordEvent.letter, messageTuple, hnotSender]

private theorem messageTuple_no_local_succ {Lifeline : Type u}
    [DecidableEq Lifeline] {sender receiver : Lifeline}
    (hne : sender ≠ receiver) {ch : ChannelName}
    (e f : WordEvent (messageTuple sender receiver ch)) :
    ¬ LocalSucc
        (fun e : WordEvent (messageTuple sender receiver ch) => e.owner)
        (fun e : WordEvent (messageTuple sender receiver ch) => e.localIndex)
        e f := by
  intro hsucc
  have he : e.index = 0 := messageTuple_event_index_zero hne e
  have hf : f.index = 0 := messageTuple_event_index_zero hne f
  have hidx : f.localIndex = e.localIndex + 1 := hsucc.2
  simp [WordEvent.localIndex, he, hf] at hidx

private def messageTuple_sendEvent {Lifeline : Type u}
    [DecidableEq Lifeline] (sender receiver : Lifeline) (ch : ChannelName) :
    WordEvent (messageTuple sender receiver ch) :=
  { owner := sender
    index := 0
    inBounds := by simp [messageTuple] }

private def messageTuple_recvEvent {Lifeline : Type u}
    [DecidableEq Lifeline] (sender receiver : Lifeline)
    (hne : sender ≠ receiver) (ch : ChannelName) :
    WordEvent (messageTuple sender receiver ch) :=
  { owner := receiver
    index := 0
    inBounds := by
      have hnotSender : receiver ≠ sender := by
        intro h
        exact hne h.symm
      simp [messageTuple, hnotSender] }

private def messageTuple_msg {Lifeline : Type u}
    [DecidableEq Lifeline] {sender receiver : Lifeline}
    (ch : ChannelName) :
    WordEvent (messageTuple sender receiver ch) →
      WordEvent (messageTuple sender receiver ch) → Prop :=
  fun s r => s.owner = sender ∧ r.owner = receiver

private def messageTuple_level {Lifeline : Type u}
    [DecidableEq Lifeline] {sender receiver : Lifeline}
    (e : WordEvent (messageTuple sender receiver (ChannelName.main))) :
    Nat :=
  if e.owner = sender then 0 else 1

private theorem messageTuple_acyclic {Lifeline : Type u}
    [DecidableEq Lifeline] {sender receiver : Lifeline}
    (hne : sender ≠ receiver) {ch : ChannelName} :
    Acyclic
      (BaseEdge
        (fun e : WordEvent (messageTuple sender receiver ch) => e.owner)
        (fun e : WordEvent (messageTuple sender receiver ch) => e.localIndex)
        (messageTuple_msg ch)) := by
  let level : WordEvent (messageTuple sender receiver ch) → Nat :=
    fun e => if e.owner = sender then 0 else 1
  have hbase :
      ∀ e f : WordEvent (messageTuple sender receiver ch),
        BaseEdge
          (fun e : WordEvent (messageTuple sender receiver ch) => e.owner)
          (fun e : WordEvent (messageTuple sender receiver ch) => e.localIndex)
          (messageTuple_msg ch) e f →
        level e < level f := by
    intro e f h
    rcases h with hsucc | hmsg
    · exact False.elim (messageTuple_no_local_succ hne e f hsucc)
    · dsimp [messageTuple_msg] at hmsg
      have hf_not_sender : f.owner ≠ sender := by
        intro hf
        exact hne (hf.symm.trans hmsg.2)
      simp [level, hmsg.1, hf_not_sender]
  have hpath :
      ∀ {e f : WordEvent (messageTuple sender receiver ch)},
        Relation.TransGen
          (BaseEdge
            (fun e : WordEvent (messageTuple sender receiver ch) => e.owner)
            (fun e : WordEvent (messageTuple sender receiver ch) => e.localIndex)
            (messageTuple_msg ch)) e f →
        level e < level f := by
    intro e f hp
    induction hp with
    | single h =>
        exact hbase _ _ h
    | tail _ h ih =>
        exact Nat.lt_trans ih (hbase _ _ h)
  intro e hcycle
  exact Nat.lt_irrefl (level e) (hpath hcycle)

theorem messageTuple_completeMSC {Lifeline : Type u} [DecidableEq Lifeline]
    {sender receiver : Lifeline} (hne : sender ≠ receiver)
    (ch : ChannelName) :
    WordIsCompleteMSC (messageTuple sender receiver ch) := by
  let D : WordMSCData (messageTuple sender receiver ch) :=
    { msg := messageTuple_msg ch
      msg_send_recv := by
        intro s r hmsg
        refine ⟨sender, receiver, ch, hmsg.1, hmsg.2, ?_, ?_⟩
        · have hsletter := messageTuple_sender_letter hne s hmsg.1
          simpa [WordEvent.kind, hsletter]
        · have hrletter := messageTuple_receiver_letter hne r hmsg.2
          simpa [WordEvent.kind, hrletter]
      msg_letter_match := by
        intro s r hmsg
        have hsletter := messageTuple_sender_letter hne s hmsg.1
        have hrletter := messageTuple_receiver_letter hne r hmsg.2
        rw [hmsg.1, hmsg.2, hsletter, hrletter]
        simp [LocalLetter.MessageMatches]
      msg_lifeline_ne := by
        intro s r hmsg
        rw [hmsg.1, hmsg.2]
        exact hne
      msg_send_unique := by
        intro s r₁ r₂ h₁ h₂
        apply WordEvent.localIndex_inj
        · rw [h₁.2, h₂.2]
        · simp [WordEvent.localIndex,
            messageTuple_event_index_zero hne r₁,
            messageTuple_event_index_zero hne r₂]
      msg_recv_unique := by
        intro s₁ s₂ r h₁ h₂
        apply WordEvent.localIndex_inj
        · rw [h₁.1, h₂.1]
        · simp [WordEvent.localIndex,
            messageTuple_event_index_zero hne s₁,
            messageTuple_event_index_zero hne s₂]
      recv_matched := by
        intro r hrecv
        rcases messageTuple_event_owner_sender_or_receiver hne r with
          hsender | hreceiver
        · rcases hrecv with ⟨A, chRecv, hkind⟩
          have hsletter := messageTuple_sender_letter hne r hsender
          have hkindSend : r.kind = EventKind.send receiver ch := by
            simpa [WordEvent.kind, hsletter]
          rw [hkindSend] at hkind
          cases hkind
        · exact
            ⟨messageTuple_sendEvent sender receiver ch, by
              exact ⟨rfl, hreceiver⟩⟩
      acyclic := messageTuple_acyclic hne
      channel_fifo := by
        intro s₁ s₂ r₁ r₂ A B ch0 hmsg₁ hmsg₂
          _hs₁ _hs₂ _hr₁ _hr₂ _hks₁ _hks₂ _hkr₁ _hkr₂ hlt
        have hs₁_zero := messageTuple_event_index_zero hne s₁
        have hs₂_zero := messageTuple_event_index_zero hne s₂
        simp [WordEvent.localIndex, hs₁_zero, hs₂_zero] at hlt }
  refine ⟨D, ?_⟩
  intro s hsend
  rcases messageTuple_event_owner_sender_or_receiver hne s with
    hsender | hreceiver
  · exact
      ⟨messageTuple_recvEvent sender receiver hne ch, by
        change messageTuple_msg ch s
          (messageTuple_recvEvent sender receiver hne ch)
        exact ⟨hsender, rfl⟩⟩
  · rcases hsend with ⟨B, chSend, hkind⟩
    have hrletter := messageTuple_receiver_letter hne s hreceiver
    have hkindRecv : (D.toMSC).kind s = EventKind.recv sender ch := by
      simpa [WordMSCData.toMSC, WordEvent.kind, hrletter]
    rw [hkindRecv] at hkind
    cases hkind

def decisionOwnerWord {Lifeline : Type u}
    (targets : List Lifeline) (ch : ChannelName)
    (tag : ControlTag) (kind : ChoiceKind) :
    Word (LocalLetter Lifeline ChannelName) :=
  LocalLetter.choice tag kind ::
    targets.map (fun B => LocalLetter.ctrlSend B ch tag kind)

def decisionRecipientWord {Lifeline : Type u} [DecidableEq Lifeline]
    (targets : List Lifeline) (owner A : Lifeline)
    (ch : ChannelName) (tag : ControlTag) (kind : ChoiceKind) :
    Word (LocalLetter Lifeline ChannelName) :=
  if A ∈ targets then
    [LocalLetter.ctrlRecv owner ch tag kind]
  else
    []

def decisionBlock {Lifeline : Type u} [DecidableEq Lifeline]
    (owner : Lifeline) (targets : List Lifeline)
    (ch : ChannelName) (tag : ControlTag) (kind : ChoiceKind) :
    TraceTuple Lifeline (LocalLetter Lifeline ChannelName) :=
  fun A =>
    if A = owner then
      decisionOwnerWord targets ch tag kind
    else
      decisionRecipientWord targets owner A ch tag kind

/--
Internal enriched semantics for the self-contained source syntax.

This mirrors the paper-level `Enr^c(P)`: control broadcasts are present, and
the `par` clause shuffles branch executions under occurrence/path branch
channels and keeps only complete word MSCs.
-/
inductive Enriched {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) :
    ChannelName → Prog Lifeline →
      TraceTuple Lifeline (LocalLetter Lifeline ChannelName) → Prop where
  | eps {ch : ChannelName} :
      Enriched order ch Prog.eps (emptyTuple Lifeline)
  | msg {ch : ChannelName} {sender receiver : Lifeline}
      (hne : sender ≠ receiver) :
      Enriched order ch (Prog.msg sender receiver hne)
        (messageTuple sender receiver ch)
  | act {ch : ChannelName} {owner : Lifeline} :
      Enriched order ch (Prog.act owner) (actionTuple owner)
  | seq {ch : ChannelName} {first second : Prog Lifeline}
      {M₁ M₂ : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
      (hfirst : Enriched order ch first M₁)
      (hsecond : Enriched order ch second M₂) :
      Enriched order ch (Prog.seq first second) (TraceTuple.append M₁ M₂)
  | branch_true {ch : ChannelName} {tag : ControlTag} {owner : Lifeline}
      {thenBranch elseBranch : Prog Lifeline}
      {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
      (hthen : Enriched order ch thenBranch M) :
      Enriched order ch (Prog.branch tag owner thenBranch elseBranch)
        (TraceTuple.append
          (decisionBlock owner
            (recipients order owner thenBranch elseBranch)
            ch tag ChoiceKind.ifTrue)
          M)
  | branch_false {ch : ChannelName} {tag : ControlTag} {owner : Lifeline}
      {thenBranch elseBranch : Prog Lifeline}
      {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
      (helse : Enriched order ch elseBranch M) :
      Enriched order ch (Prog.branch tag owner thenBranch elseBranch)
        (TraceTuple.append
          (decisionBlock owner
            (recipients order owner thenBranch elseBranch)
            ch tag ChoiceKind.ifFalse)
          M)
  | loop_exit {ch : ChannelName} {tag : ControlTag} {owner : Lifeline}
      {body exit : Prog Lifeline}
      {Mexit : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
      (hexit : Enriched order ch exit Mexit) :
      Enriched order ch (Prog.loop tag owner body exit)
        (TraceTuple.append
          (decisionBlock owner
            (recipients order owner body exit)
            ch tag ChoiceKind.whileFalse)
          Mexit)
  | loop_iter {ch : ChannelName} {tag : ControlTag} {owner : Lifeline}
      {body exit : Prog Lifeline}
      {Mbody Mrest : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
      (hbody : Enriched order ch body Mbody)
      (hrest : Enriched order ch (Prog.loop tag owner body exit) Mrest) :
      Enriched order ch (Prog.loop tag owner body exit)
        (TraceTuple.append
          (decisionBlock owner
            (recipients order owner body exit)
            ch tag ChoiceKind.whileTrue)
          (TraceTuple.append Mbody Mrest))
  | par {n : Nat} {ch : ChannelName} {label : Nat}
      {branches : Fin n → Prog Lifeline}
      {branchWords : Fin n →
        TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
      {tagged : ParLocTaggedTrace n Lifeline ChannelName}
      (hbranches :
        ∀ i, Enriched order (ChannelName.branch ch label i)
          (branches i) (branchWords i))
      (hpar : ParLoc branchWords tagged)
      (hcomplete : WordIsCompleteMSC (TraceTuple.map Tagged.val tagged)) :
      Enriched order ch (Prog.par label branches)
        (TraceTuple.map Tagged.val tagged)

def sourceObservation {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (ch : ChannelName) (P : Prog Lifeline)
    (observed : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)) :
    Prop :=
  ErasedSemantics (Enriched order ch P) observed

def sourceObservationMain {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (P : Prog Lifeline)
    (observed : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)) :
    Prop :=
  sourceObservation order ChannelName.main P observed

namespace Enriched

/--
Lifelines not mentioned by a source program have empty enriched local words.
-/
theorem empty_of_not_mem_lifelines {Lifeline : Type u}
    [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} {P : Prog Lifeline}
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
    (h : Enriched order ch P M) {A : Lifeline}
    (hnot : A ∉ lifelines P) :
    M A = [] := by
  induction h generalizing A with
  | eps =>
      rfl
  | msg hne =>
      rename_i ch0 sender receiver
      have hsender : A ≠ sender := by
        intro hEq
        exact hnot (by simp [lifelines, hEq])
      have hreceiver : A ≠ receiver := by
        intro hEq
        exact hnot (by simp [lifelines, hEq])
      simp [messageTuple, hsender, hreceiver]
  | act =>
      rename_i ch0 owner
      have howner : A ≠ owner := by
        intro hEq
        exact hnot (by simp [lifelines, hEq])
      simp [actionTuple, howner]
  | seq hfirst hsecond ihFirst ihSecond =>
      rename_i ch0 first second M₁ M₂
      have hnotFirst : A ∉ lifelines first := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      have hnotSecond : A ∉ lifelines second := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      simp [TraceTuple.append, ihFirst hnotFirst, ihSecond hnotSecond]
  | branch_true hthen ihThen =>
      rename_i ch0 tag owner thenBranch elseBranch Mthen
      have hnotOwner : A ≠ owner := by
        intro hEq
        exact hnot (by simp [lifelines, hEq])
      have hnotThen : A ∉ lifelines thenBranch := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      have hnotElse : A ∉ lifelines elseBranch := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      have hnotTargets :
          A ∉ recipients order owner thenBranch elseBranch := by
        simp [recipients, hnotOwner, hnotThen, hnotElse]
      simp [TraceTuple.append, decisionBlock, decisionRecipientWord,
        hnotOwner, hnotTargets, ihThen hnotThen]
  | branch_false helse ihElse =>
      rename_i ch0 tag owner thenBranch elseBranch Melse
      have hnotOwner : A ≠ owner := by
        intro hEq
        exact hnot (by simp [lifelines, hEq])
      have hnotThen : A ∉ lifelines thenBranch := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      have hnotElse : A ∉ lifelines elseBranch := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      have hnotTargets :
          A ∉ recipients order owner thenBranch elseBranch := by
        simp [recipients, hnotOwner, hnotThen, hnotElse]
      simp [TraceTuple.append, decisionBlock, decisionRecipientWord,
        hnotOwner, hnotTargets, ihElse hnotElse]
  | loop_exit hexit ihExit =>
      rename_i ch0 tag owner body exit Mexit
      have hnotOwner : A ≠ owner := by
        intro hEq
        exact hnot (by simp [lifelines, hEq])
      have hnotBody : A ∉ lifelines body := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      have hnotExit : A ∉ lifelines exit := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      have hnotTargets :
          A ∉ recipients order owner body exit := by
        simp [recipients, hnotOwner, hnotBody, hnotExit]
      simp [TraceTuple.append, decisionBlock, decisionRecipientWord,
        hnotOwner, hnotTargets, ihExit hnotExit]
  | loop_iter hbody hrest ihBody ihRest =>
      rename_i ch0 tag owner body exit Mbody Mrest
      have hnotOwner : A ≠ owner := by
        intro hEq
        exact hnot (by simp [lifelines, hEq])
      have hnotBody : A ∉ lifelines body := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      have hnotExit : A ∉ lifelines exit := by
        intro hmem
        exact hnot (by simp [lifelines, hmem])
      have hnotTargets :
          A ∉ recipients order owner body exit := by
        simp [recipients, hnotOwner, hnotBody, hnotExit]
      simp [TraceTuple.append, decisionBlock, decisionRecipientWord,
        hnotOwner, hnotTargets, ihBody hnotBody, ihRest hnot]
  | par hbranches hpar _hcomplete ihBranches =>
      rename_i n ch0 branches branchWords tagged
      have herase_empty :
          ∀ i, eraseBranch i (tagged A) = [] := by
        intro i
        have hnotBranch : A ∉ lifelines (branches i) :=
          not_mem_finConcatMap hnot i
        calc
          eraseBranch i (tagged A) = branchWords i A := hpar i A
          _ = [] := ihBranches i hnotBranch
      have htagged_empty : tagged A = [] :=
        eraseBranch_all_empty_nil herase_empty
      simp [TraceTuple.map, htagged_empty]

/--
Every enriched execution is complete once decision blocks are known to be
complete word MSCs.  The only non-atomic source constructor that creates fresh
messages outside recursive subprograms is a branch/loop decision block.
-/
theorem completeMSC_of_decisionBlocks {Lifeline : Type u}
    [DecidableEq Lifeline]
    {order : List Lifeline}
    (hdecision :
      ∀ (owner : Lifeline) (targets : List Lifeline)
        (ch : ChannelName) (tag : ControlTag) (kind : ChoiceKind),
        WordIsCompleteMSC
          (decisionBlock owner targets ch tag kind))
    {ch : ChannelName} {P : Prog Lifeline}
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
    (h : Enriched order ch P M) :
    WordIsCompleteMSC M := by
  induction h with
  | eps =>
      exact emptyTuple_completeMSC
  | msg hne =>
      exact messageTuple_completeMSC hne _
  | act =>
      exact actionTuple_completeMSC _
  | seq hfirst hsecond ihFirst ihSecond =>
      exact WordMSCData.wordIsCompleteMSC_append ihFirst ihSecond
  | branch_true hthen ihThen =>
      exact
        WordMSCData.wordIsCompleteMSC_append
          (hdecision _ _ _ _ _)
          ihThen
  | branch_false helse ihElse =>
      exact
        WordMSCData.wordIsCompleteMSC_append
          (hdecision _ _ _ _ _)
          ihElse
  | loop_exit hexit ihExit =>
      exact
        WordMSCData.wordIsCompleteMSC_append
          (hdecision _ _ _ _ _)
          ihExit
  | loop_iter hbody hrest ihBody ihRest =>
      exact
        WordMSCData.wordIsCompleteMSC_append
          (hdecision _ _ _ _ _)
          (WordMSCData.wordIsCompleteMSC_append ihBody ihRest)
  | par _hbranches _hpar hcomplete _ihBranches =>
      exact hcomplete

/--
Every internal enriched source execution is a complete local trace of the
projected distributed program.

This is the syntax-directed half of internal correctness.  It deliberately
does not assert the word-MSC predicate; that remains the semantic filter
boundary.
-/
theorem to_projectedTrace {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} {P : Prog Lifeline}
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
    (h : Enriched order ch P M)
    (hcover : ∀ A, A ∈ lifelines P → A ∈ order) :
    DistProg.trace (projectDist order ch P) M := by
  induction h with
  | eps =>
      intro A
      exact LocProg.Trace.eps
  | msg hne =>
      rename_i ch0 sender receiver
      intro A
      by_cases hsender : A = sender
      · subst A
        simp [projectDist, project, messageTuple]
        exact LocProg.Trace.letter (LocalLetter.send receiver ch0)
      · by_cases hreceiver : A = receiver
        · subst A
          have hreceiver_sender : receiver ≠ sender := by
            intro h
            exact hne h.symm
          simp [projectDist, project, messageTuple, hreceiver_sender]
          exact LocProg.Trace.letter (LocalLetter.recv sender ch0)
        · simp [projectDist, project, messageTuple, hsender, hreceiver]
          exact LocProg.Trace.eps
  | act =>
      rename_i ch0 owner
      intro A
      by_cases howner : A = owner
      · subst A
        simp [projectDist, project, actionTuple]
        exact LocProg.Trace.letter LocalLetter.action
      · simp [projectDist, project, actionTuple, howner]
        exact LocProg.Trace.eps
  | seq hfirst hsecond ihFirst ihSecond =>
      rename_i ch0 first second M₁ M₂
      have hcoverFirst :
          ∀ A, A ∈ lifelines first → A ∈ order := by
        intro A hmem
        exact hcover A (by simp [lifelines, hmem])
      have hcoverSecond :
          ∀ A, A ∈ lifelines second → A ∈ order := by
        intro A hmem
        exact hcover A (by simp [lifelines, hmem])
      intro A
      exact LocProg.Trace.seq (ihFirst hcoverFirst A)
        (ihSecond hcoverSecond A)
  | branch_true hthen ihThen =>
      rename_i ch0 tag owner thenBranch elseBranch Mthen
      have hcoverThen :
          ∀ A, A ∈ lifelines thenBranch → A ∈ order := by
        intro A hmem
        exact hcover A (by simp [lifelines, hmem])
      intro A
      let targets := recipients order owner thenBranch elseBranch
      by_cases howner : A = owner
      · subst A
        simpa [projectDist, project, decisionBlock, decisionOwnerWord,
          targets, TraceTuple.append] using
          (LocProg.Trace.branch_true
            (LocProg.Trace.seq
              (controlBroadcast_trace targets ch0 tag ChoiceKind.ifTrue)
              (ihThen hcoverThen owner)))
      · by_cases hrec : A ∈ targets
        · simpa [projectDist, project, decisionBlock,
            decisionRecipientWord, howner, hrec, targets,
            TraceTuple.append] using
            (LocProg.Trace.branch_true (ihThen hcoverThen A))
        · have hnotThen : A ∉ lifelines thenBranch := by
            intro hmem
            have horder : A ∈ order :=
              hcover A (by simp [lifelines, hmem])
            have htarget :
                A ∈ recipients order owner thenBranch elseBranch := by
              simp [recipients, horder, howner, hmem]
            exact hrec htarget
          have hempty := empty_of_not_mem_lifelines hthen hnotThen
          simp [projectDist, project, decisionBlock,
            decisionRecipientWord, howner, hrec, targets,
            TraceTuple.append, hempty]
          exact LocProg.Trace.eps
  | branch_false helse ihElse =>
      rename_i ch0 tag owner thenBranch elseBranch Melse
      have hcoverElse :
          ∀ A, A ∈ lifelines elseBranch → A ∈ order := by
        intro A hmem
        exact hcover A (by simp [lifelines, hmem])
      intro A
      let targets := recipients order owner thenBranch elseBranch
      by_cases howner : A = owner
      · subst A
        simpa [projectDist, project, decisionBlock, decisionOwnerWord,
          targets, TraceTuple.append] using
          (LocProg.Trace.branch_false
            (LocProg.Trace.seq
              (controlBroadcast_trace targets ch0 tag ChoiceKind.ifFalse)
              (ihElse hcoverElse owner)))
      · by_cases hrec : A ∈ targets
        · simpa [projectDist, project, decisionBlock,
            decisionRecipientWord, howner, hrec, targets,
            TraceTuple.append] using
            (LocProg.Trace.branch_false (ihElse hcoverElse A))
        · have hnotElse : A ∉ lifelines elseBranch := by
            intro hmem
            have horder : A ∈ order :=
              hcover A (by simp [lifelines, hmem])
            have htarget :
                A ∈ recipients order owner thenBranch elseBranch := by
              simp [recipients, horder, howner, hmem]
            exact hrec htarget
          have hempty := empty_of_not_mem_lifelines helse hnotElse
          simp [projectDist, project, decisionBlock,
            decisionRecipientWord, howner, hrec, targets,
            TraceTuple.append, hempty]
          exact LocProg.Trace.eps
  | loop_exit hexit ihExit =>
      rename_i ch0 tag owner body exit Mexit
      have hcoverExit :
          ∀ A, A ∈ lifelines exit → A ∈ order := by
        intro A hmem
        exact hcover A (by simp [lifelines, hmem])
      intro A
      let targets := recipients order owner body exit
      by_cases howner : A = owner
      · subst A
        simpa [projectDist, project, decisionBlock, decisionOwnerWord,
          targets, TraceTuple.append] using
          ((LocProg.trace_loop_unfold _ _ _ _ _).2
          (Or.inl
            ⟨_,
              LocProg.Trace.seq
                (controlBroadcast_trace targets ch0 tag ChoiceKind.whileFalse)
                (ihExit hcoverExit owner),
              rfl⟩))
      · by_cases hrec : A ∈ targets
        · simpa [projectDist, project, decisionBlock,
            decisionRecipientWord, howner, hrec, targets,
            TraceTuple.append] using
            ((LocProg.trace_loop_unfold _ _ _ _ _).2
              (Or.inl ⟨_, ihExit hcoverExit A, rfl⟩))
        · have hnotExit : A ∉ lifelines exit := by
            intro hmem
            have horder : A ∈ order :=
              hcover A (by simp [lifelines, hmem])
            have htarget : A ∈ recipients order owner body exit := by
              simp [recipients, horder, howner, hmem]
            exact hrec htarget
          have hempty := empty_of_not_mem_lifelines hexit hnotExit
          simp [projectDist, project, decisionBlock,
            decisionRecipientWord, howner, hrec, targets,
            TraceTuple.append, hempty]
          exact LocProg.Trace.eps
  | loop_iter hbody hrest ihBody ihRest =>
      rename_i ch0 tag owner body exit Mbody Mrest
      have hcoverBody :
          ∀ A, A ∈ lifelines body → A ∈ order := by
        intro A hmem
        exact hcover A (by simp [lifelines, hmem])
      have hcoverRest :
          ∀ A, A ∈ lifelines (Prog.loop tag owner body exit) → A ∈ order :=
        hcover
      intro A
      let targets := recipients order owner body exit
      by_cases howner : A = owner
      · subst A
        have hrestTrace :
            LocProg.trace
              (LocProg.loop
                (LocalLetter.choice tag ChoiceKind.whileTrue)
                (LocalLetter.choice tag ChoiceKind.whileFalse)
                (LocProg.seq
                  (controlBroadcast targets ch0 tag ChoiceKind.whileTrue)
                  (project order owner ch0 body))
                (LocProg.seq
                  (controlBroadcast targets ch0 tag ChoiceKind.whileFalse)
                  (project order owner ch0 exit)))
              (Mrest owner) := by
          simpa [projectDist, project, targets] using
            (ihRest hcoverRest owner)
        simpa [projectDist, project, decisionBlock, decisionOwnerWord,
          targets, TraceTuple.append, List.append_assoc] using
          ((LocProg.trace_loop_unfold _ _ _ _ _).2
          (Or.inr
            ⟨_,
              Mrest owner,
              LocProg.Trace.seq
                (controlBroadcast_trace targets ch0 tag ChoiceKind.whileTrue)
                (ihBody hcoverBody owner),
              hrestTrace,
              by simp [targets, List.append_assoc]⟩))
      · by_cases hrec : A ∈ targets
        · have hrestTrace :
              LocProg.trace
                (LocProg.loop
                  (LocalLetter.ctrlRecv owner ch0 tag ChoiceKind.whileTrue)
                  (LocalLetter.ctrlRecv owner ch0 tag ChoiceKind.whileFalse)
                  (project order A ch0 body)
                  (project order A ch0 exit))
                (Mrest A) := by
            simpa [projectDist, project, howner, hrec, targets] using
              (ihRest hcoverRest A)
          simpa [projectDist, project, decisionBlock,
            decisionRecipientWord, howner, hrec, targets,
            TraceTuple.append, List.append_assoc] using
            ((LocProg.trace_loop_unfold _ _ _ _ _).2
              (Or.inr
                ⟨_, Mrest A, ihBody hcoverBody A,
                  hrestTrace, by simp⟩))
        · have hnotBody : A ∉ lifelines body := by
            intro hmem
            have horder : A ∈ order :=
              hcover A (by simp [lifelines, hmem])
            have htarget : A ∈ recipients order owner body exit := by
              simp [recipients, horder, howner, hmem]
            exact hrec htarget
          have hnotRest : A ∉ lifelines (Prog.loop tag owner body exit) := by
            intro hmem
            simp [lifelines] at hmem
            rcases hmem with hEq | hmem | hmem
            · exact howner hEq
            · exact hnotBody hmem
            · have horder : A ∈ order :=
                hcover A (by simp [lifelines, hmem])
              have htarget : A ∈ recipients order owner body exit := by
                simp [recipients, horder, howner, hmem]
              exact hrec htarget
          have hbodyEmpty := empty_of_not_mem_lifelines hbody hnotBody
          have hrestEmpty := empty_of_not_mem_lifelines hrest hnotRest
          simp [projectDist, project, decisionBlock,
            decisionRecipientWord, howner, hrec, targets,
            TraceTuple.append, hbodyEmpty, hrestEmpty]
          exact LocProg.Trace.eps
  | par hbranches hpar _hcomplete ihBranches =>
      rename_i n ch0 branches branchWords tagged
      intro A
      exact LocProg.Trace.par
        (tagged := tagged A)
        (branchWords := fun i => branchWords i A)
        rfl
        (fun i => hpar i A)
        (fun i =>
          ihBranches i
            (fun B hmem =>
              hcover B
                (by
                  simpa [lifelines] using
                    (mem_finConcatMap_of_mem
                      (fun j => lifelines (branches j)) i hmem)))
            A)

/--
Internal enriched executions that pass the same complete-MSC filter are
projected distributed executions.
-/
theorem to_projectedSemantics {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} {P : Prog Lifeline}
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
    (h : Enriched order ch P M)
    (hcover : ∀ A, A ∈ lifelines P → A ∈ order)
    (hcomplete : WordIsCompleteMSC M) :
    projectedSemantics WordIsCompleteMSC order ch P M :=
  ⟨to_projectedTrace h hcover, hcomplete⟩

/--
Forward internal correctness once decision blocks have concrete complete-MSC
witnesses.
-/
theorem to_projectedSemantics_of_decisionBlocks
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline}
    (hdecision :
      ∀ (owner : Lifeline) (targets : List Lifeline)
        (ch : ChannelName) (tag : ControlTag) (kind : ChoiceKind),
        WordIsCompleteMSC
          (decisionBlock owner targets ch tag kind))
    {ch : ChannelName} {P : Prog Lifeline}
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
    (h : Enriched order ch P M)
    (hcover : ∀ A, A ∈ lifelines P → A ∈ order) :
    projectedSemantics WordIsCompleteMSC order ch P M :=
  to_projectedSemantics h hcover
    (completeMSC_of_decisionBlocks hdecision h)

end Enriched

end Prog
end Source

end ZipperGenParallel
