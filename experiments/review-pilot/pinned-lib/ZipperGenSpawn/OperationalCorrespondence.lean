import ZipperGenSpawn.OperationalControlTrace

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- Number of local fork markers for one parent session and static spawn site. -/
def forkCount (parent : SessionName) (site : Occurrence) :
    List (ProjectedSpawnLetter Lifeline ChannelScope) -> Nat
  | [] => 0
  | .fork current currentSite _ :: rest =>
      (if current = parent ∧ currentSite = site then 1 else 0) +
        forkCount parent site rest
  | .event _ _ :: rest => forkCount parent site rest

@[simp] theorem forkCount_nil
    (parent : SessionName) (site : Occurrence) :
    forkCount (Lifeline := Lifeline) parent site [] = 0 := rfl

@[simp] theorem forkCount_event
    (parent : SessionName) (site : Occurrence)
    (session : SessionName) (letter : LocalLetter Lifeline ChannelScope)
    (rest : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    forkCount parent site
        (ProjectedSpawnLetter.event session letter :: rest) =
      forkCount parent site rest := rfl

@[simp] theorem forkCount_fork
    (parent current : SessionName) (site currentSite : Occurrence)
    (child : SessionName)
    (rest : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    forkCount parent site
        (ProjectedSpawnLetter.fork current currentSite child :: rest) =
      (if current = parent ∧ currentSite = site then 1 else 0) +
        forkCount parent site rest := rfl

theorem forkCount_append
    (parent : SessionName) (site : Occurrence)
    (left right : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    forkCount parent site (left ++ right) =
      forkCount parent site left + forkCount parent site right := by
  induction left with
  | nil => simp [forkCount]
  | cons head tail ih =>
      cases head <;> simp [forkCount, ih, Nat.add_assoc]

/-- The runtime counter is not an independent oracle: it is exactly the
    number of forks already recorded on that participant's physical word. -/
def CountersMatchHistory
    (configuration : Configuration Lifeline Field) : Prop :=
  ∀ A parent site,
    configuration.counters A parent site =
      forkCount parent site (configuration.history A)

theorem initial_countersMatchHistory [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    CountersMatchHistory (initialConfiguration order program) := by
  intro A parent site
  rfl

def countersAfterEmission [DecidableEq Lifeline]
    (counters : VisitCounters Lifeline)
    (A : Lifeline) (session : SessionName) :
    Emission Lifeline -> VisitCounters Lifeline
  | .fork _ site _ => incrementCounter counters A session site
  | .silent | .event _ => counters

/-- A local step changes the visit-counter store exactly when it emits a
    fork, and then only at the executing participant, parent session, and
    emitted spawn site. -/
theorem localStep_counter_effect [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {program residual : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline}
    {activation : Activation Lifeline Field}
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission activation) :
    counters' = countersAfterEmission counters A session emission := by
  induction hstep with
  | letter hletter => rfl
  | seq_done second => rfl
  | seq_left hleft ih =>
      exact ih
  | branch_true htrue => rfl
  | branch_false hfalse => rfl
  | loop_true htrue => rfl
  | loop_false hfalse => rfl
  | par_done hdone => rfl
  | par_branch i hbranch ih =>
      exact ih
  | fork site fresh childProgram => rfl

theorem step_preserves_countersMatchHistory [DecidableEq Lifeline]
    {before after : Configuration Lifeline Field}
    (hinvariant : CountersMatchHistory before)
    (hstep : Step before after) :
    CountersMatchHistory after := by
  intro B parent site
  cases hstep with
  | run A session program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      have hcounter := localStep_counter_effect hlocal
      cases emission with
      | silent =>
          simp [countersAfterEmission] at hcounter
          rw [hcounter]
          exact hinvariant B parent site
      | event letter =>
          simp [countersAfterEmission] at hcounter
          rw [hcounter]
          by_cases hBA : B = A
          · subst B
            simpa [appendEmission, forkCount_append] using
              hinvariant A parent site
          · simpa [appendEmission, hBA] using
              hinvariant B parent site
      | fork emittedParent emittedSite child =>
          have hparent : emittedParent = session :=
            localStep_fork_parent hlocal rfl
          subst emittedParent
          simp [countersAfterEmission] at hcounter
          rw [hcounter]
          by_cases hBA : B = A
          · subst B
            by_cases hp : parent = session
            · subst parent
              by_cases hs : site = emittedSite
              · subst site
                simp [incrementCounter, nextVisit, appendEmission,
                  forkCount_append, hinvariant A session emittedSite,
                  Nat.add_comm]
              · have hs' : emittedSite ≠ site := Ne.symm hs
                simp [incrementCounter, appendEmission,
                  forkCount_append, hs, hs', hinvariant A session site]
            · have hp' : session ≠ parent := Ne.symm hp
              simp [incrementCounter, appendEmission,
                forkCount_append, hp, hp', hinvariant A parent site]
          · simp [incrementCounter, appendEmission, hBA,
              hinvariant B parent site]

theorem reachable_countersMatchHistory [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    ∀ configuration,
      Reachable (initialConfiguration order program) configuration ->
        CountersMatchHistory configuration := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact initial_countersMatchHistory order program
  | tail hbefore hstep ih =>
      exact step_preserves_countersMatchHistory ih hstep

/-- At a reachable fork step, the allocated visit is one plus the number of
    earlier local forks at that participant, parent session, and site. This is
    the rank formulation used by the later cross-participant agreement proof. -/
theorem reachable_fork_uses_next_rank [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    {before after : Configuration Lifeline Field}
    (hreachable : Reachable (initialConfiguration order source) before)
    (hstep : Step before after)
    {A : Lifeline} {parent child : SessionName} {site : Occurrence}
    (hnew : after.history A =
      before.history A ++ [ProjectedSpawnLetter.fork parent site child]) :
    child = SessionName.child parent site
      (forkCount parent site (before.history A) + 1) := by
  have hcounters := reachable_countersMatchHistory order source before hreachable
  cases hstep with
  | run actor session program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      cases emission with
      | silent =>
          simp [appendEmission] at hnew
      | event letter =>
          by_cases hAA : A = actor
          · subst A
            simp [appendEmission] at hnew
          · simp [appendEmission, hAA] at hnew

      | fork emittedParent emittedSite emittedChild =>
          by_cases hAA : A = actor
          · subst A
            have heq :
                (ProjectedSpawnLetter.fork emittedParent emittedSite emittedChild :
                    ProjectedSpawnLetter Lifeline ChannelScope) =
                  ProjectedSpawnLetter.fork parent site child := by
              simpa [appendEmission] using hnew
            cases heq
            have hparent : parent = session :=
              localStep_fork_parent hlocal rfl
            have hchild : child = SessionName.child session site
                (nextVisit before.counters actor session site) :=
              localStep_fork_child hlocal rfl
            subst parent
            simpa [nextVisit, hcounters actor session site] using hchild
          · simp [appendEmission, hAA] at hnew

/-- Two participants that have observed the same number of earlier visits to
    a spawn site allocate the same canonical child session. This isolates the
    one premise that the later projection/agreement proof must establish. -/
theorem fork_children_agree_of_prior_count_eq
    {parent leftChild rightChild : SessionName} {site : Occurrence}
    {leftHistory rightHistory :
      List (ProjectedSpawnLetter Lifeline ChannelScope)}
    (hleft : leftChild = SessionName.child parent site
      (forkCount parent site leftHistory + 1))
    (hright : rightChild = SessionName.child parent site
      (forkCount parent site rightHistory + 1))
    (hcount : forkCount parent site leftHistory =
      forkCount parent site rightHistory) :
    leftChild = rightChild := by
  rw [hleft, hright, hcount]

/-- Items sent on one session-scoped FIFO route, in sender-local order. -/
def sentItems [DecidableEq Lifeline]
    (session : SessionName) (scope : ChannelScope) (receiver : Lifeline) :
    List (ProjectedSpawnLetter Lifeline ChannelScope) -> List QueueItem
  | [] => []
  | .fork _ _ _ :: rest => sentItems session scope receiver rest
  | .event current letter :: rest =>
      let tail := sentItems session scope receiver rest
      if current ≠ session then tail else
      match letter with
      | .send to currentScope =>
          if to = receiver ∧ currentScope = scope then .data :: tail else tail
      | .ctrlSend to currentScope site kind =>
          if to = receiver ∧ currentScope = scope then
            .control site kind :: tail
          else tail
      | _ => tail

/-- Items received on one session-scoped FIFO route, in receiver-local order. -/
def receivedItems [DecidableEq Lifeline]
    (session : SessionName) (scope : ChannelScope) (sender : Lifeline) :
    List (ProjectedSpawnLetter Lifeline ChannelScope) -> List QueueItem
  | [] => []
  | .fork _ _ _ :: rest => receivedItems session scope sender rest
  | .event current letter :: rest =>
      let tail := receivedItems session scope sender rest
      if current ≠ session then tail else
      match letter with
      | .recv source currentScope =>
          if source = sender ∧ currentScope = scope then .data :: tail else tail
      | .ctrlRecv source currentScope site kind =>
          if source = sender ∧ currentScope = scope then
            .control site kind :: tail
          else tail
      | _ => tail

theorem sentItems_append [DecidableEq Lifeline]
    (session : SessionName) (scope : ChannelScope) (receiver : Lifeline)
    (left right : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    sentItems session scope receiver (left ++ right) =
      sentItems session scope receiver left ++
        sentItems session scope receiver right := by
  induction left with
  | nil => rfl
  | cons head tail ih =>
      cases head with
      | fork parent site child =>
          simpa [sentItems] using ih
      | event current letter =>
          by_cases hcurrent : current ≠ session
          · simp [sentItems, hcurrent, ih]
          · have hcurrent' : current = session :=
              Decidable.not_not.mp hcurrent
            cases hcurrent'
            cases letter <;> simp [sentItems, ih] <;>
              split <;> simp_all

theorem receivedItems_append [DecidableEq Lifeline]
    (session : SessionName) (scope : ChannelScope) (sender : Lifeline)
    (left right : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    receivedItems session scope sender (left ++ right) =
      receivedItems session scope sender left ++
        receivedItems session scope sender right := by
  induction left with
  | nil => rfl
  | cons head tail ih =>
      cases head with
      | fork parent site child =>
          simpa [receivedItems] using ih
      | event current letter =>
          by_cases hcurrent : current ≠ session
          · simp [receivedItems, hcurrent, ih]
          · have hcurrent' : current = session :=
              Decidable.not_not.mp hcurrent
            cases hcurrent'
            cases letter <;> simp [receivedItems, ih] <;>
              split <;> simp_all

/-- For each logical FIFO route, sent items are precisely received items
    followed by the items still resident in the runtime queue. -/
def QueuesMatchHistory [DecidableEq Lifeline]
    (configuration : Configuration Lifeline Field) : Prop :=
  ∀ session scope sender receiver,
    sentItems session scope receiver (configuration.history sender) =
      receivedItems session scope sender (configuration.history receiver) ++
        configuration.queues session scope sender receiver

theorem initial_queuesMatchHistory [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    QueuesMatchHistory (initialConfiguration order program) := by
  intro session scope sender receiver
  rfl

/-- If a local step emits an ordinary event, its queue update is exactly the
    corresponding elementary letter step. Silent and fork steps leave queues
    unchanged. -/
theorem localStep_queue_effect [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {program residual : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline}
    {activation : Activation Lifeline Field}
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission activation) :
    (emission = .silent ∨ (∃ parent site child,
        emission = .fork parent site child)) ∧ queues' = queues ∨
      ∃ letter,
        emission = .event letter ∧ LetterStep A session letter queues queues' := by
  induction hstep with
  | letter hletter => exact Or.inr ⟨_, rfl, hletter⟩
  | seq_done second => exact Or.inl ⟨Or.inl rfl, rfl⟩
  | seq_left hleft ih => exact ih
  | branch_true htrue => exact Or.inr ⟨_, rfl, htrue⟩
  | branch_false hfalse => exact Or.inr ⟨_, rfl, hfalse⟩
  | loop_true htrue => exact Or.inr ⟨_, rfl, htrue⟩
  | loop_false hfalse => exact Or.inr ⟨_, rfl, hfalse⟩
  | par_done hdone => exact Or.inl ⟨Or.inl rfl, rfl⟩
  | par_branch i hbranch ih => exact ih
  | fork site fresh childProgram =>
      exact Or.inl ⟨Or.inr ⟨_, _, _, rfl⟩, rfl⟩

set_option maxRecDepth 10000 in
theorem step_preserves_queuesMatchHistory [DecidableEq Lifeline]
    {before after : Configuration Lifeline Field}
    (hinvariant : QueuesMatchHistory before)
    (hstep : Step before after) :
    QueuesMatchHistory after := by
  intro current route sender receiver
  cases hstep with
  | run A session program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      rcases localStep_queue_effect hlocal with hquiet | hevent
      · rcases hquiet with ⟨hemits, hqueues⟩
        subst queues'
        rcases hemits with hsilent | ⟨parent, site, child, hfork⟩
        · subst emission
          exact hinvariant current route sender receiver
        · subst emission
          by_cases hsender : sender = A
          · subst sender
            by_cases hreceiver : receiver = A
            · subst receiver
              simpa [appendEmission, sentItems_append, receivedItems_append,
                sentItems, receivedItems] using
                hinvariant current route A A
            · simpa [appendEmission, hreceiver, sentItems_append,
                receivedItems_append, sentItems, receivedItems] using
                hinvariant current route A receiver
          · by_cases hreceiver : receiver = A
            · subst receiver
              simpa [appendEmission, hsender, sentItems_append,
                receivedItems_append, sentItems, receivedItems] using
                hinvariant current route sender A
            · simpa [appendEmission, hsender, hreceiver] using
                hinvariant current route sender receiver
      · rcases hevent with ⟨letter, hemits, hletter⟩
        subst emission
        have hbase := hinvariant current route sender receiver
        cases hletter with
        | tau queues =>
            by_cases hs : sender = A <;> by_cases hr : receiver = A <;>
              simpa [appendEmission, hs, hr, sentItems_append,
                receivedItems_append, sentItems, receivedItems] using hbase
        | action queues =>
            by_cases hs : sender = A <;> by_cases hr : receiver = A <;>
              simpa [appendEmission, hs, hr, sentItems_append,
                receivedItems_append, sentItems, receivedItems] using hbase
        | choice queues tag kind =>
            by_cases hs : sender = A <;> by_cases hr : receiver = A <;>
              simpa [appendEmission, hs, hr, sentItems_append,
                receivedItems_append, sentItems, receivedItems] using hbase
        | send queues B scope =>
            by_cases hc : current = session <;>
              by_cases hroute : route = scope <;>
              by_cases hs : sender = A <;>
              by_cases hr : receiver = B <;>
              by_cases hsA : receiver = A <;>
              simp_all [appendEmission, enqueue, setQueue,
                sentItems_append, receivedItems_append, sentItems,
                receivedItems, List.append_assoc, eq_comm]
        | recv queues B scope tail hne hhead =>
            have hhead' :
                QueueItem.data :: tail =
                  before.queues session scope B A := hhead.symm
            clear hhead
            by_cases hc : current = session <;>
              by_cases hroute : route = scope <;>
              by_cases hs : sender = B <;>
              by_cases hr : receiver = A <;>
              by_cases hrA : sender = A <;>
              simp_all [appendEmission, setQueue,
                sentItems_append, receivedItems_append, sentItems,
                receivedItems, List.append_assoc, eq_comm]
        | ctrlSend queues B scope tag kind =>
            by_cases hc : current = session <;>
              by_cases hroute : route = scope <;>
              by_cases hs : sender = A <;>
              by_cases hr : receiver = B <;>
              by_cases hsA : receiver = A <;>
              simp_all [appendEmission, enqueue, setQueue,
                sentItems_append, receivedItems_append, sentItems,
                receivedItems, List.append_assoc, eq_comm]
        | ctrlRecv queues B scope tail tag kind hne hhead =>
            have hhead' :
                QueueItem.control tag kind :: tail =
                  before.queues session scope B A := hhead.symm
            clear hhead
            by_cases hc : current = session <;>
              by_cases hroute : route = scope <;>
              by_cases hs : sender = B <;>
              by_cases hr : receiver = A <;>
              by_cases hrA : sender = A <;>
              simp_all [appendEmission, setQueue,
                sentItems_append, receivedItems_append, sentItems,
                receivedItems, List.append_assoc, eq_comm]

theorem reachable_queuesMatchHistory [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    ∀ configuration,
      Reachable (initialConfiguration order program) configuration ->
        QueuesMatchHistory configuration := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact initial_queuesMatchHistory order program
  | tail hbefore hstep ih =>
      exact step_preserves_queuesMatchHistory ih hstep

/-- Runtime reception on every logical route is a prefix of transmission on
    that route; the witness suffix is exactly the current FIFO queue. -/
theorem queuesMatchHistory_received_prefix [DecidableEq Lifeline]
    {configuration : Configuration Lifeline Field}
    (hinvariant : QueuesMatchHistory configuration)
    (session : SessionName) (scope : ChannelScope)
    (sender receiver : Lifeline) :
    ListPrefix
      (receivedItems session scope sender (configuration.history receiver))
      (sentItems session scope receiver (configuration.history sender)) := by
  exact ⟨configuration.queues session scope sender receiver,
    (hinvariant session scope sender receiver).symm⟩

theorem reachable_receivedItems_prefix_sentItems [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (hreachable :
      Reachable (initialConfiguration order program) configuration)
    (session : SessionName) (scope : ChannelScope)
    (sender receiver : Lifeline) :
    ListPrefix
      (receivedItems session scope sender (configuration.history receiver))
      (sentItems session scope receiver (configuration.history sender)) := by
  exact queuesMatchHistory_received_prefix
    (reachable_queuesMatchHistory order program configuration hreachable)
    session scope sender receiver

/-- All operational invariants currently needed by the session-restriction
    correspondence argument, collected as one reusable reachable-state fact. -/
structure OperationalInvariant [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field) : Prop where
  canonicalForks : CanonicalForkHistory configuration.history
  controlsFollowForks : ControlsFollowForks configuration
  forksPrecedeEvents : ForksPrecedeEvents configuration.history
  countersMatchHistory : CountersMatchHistory configuration
  queuesMatchHistory : QueuesMatchHistory configuration
  eventsHaveControls : EventsHaveControls configuration
  controlsHaveExecution : ControlsHaveExecution order source configuration

theorem reachable_operationalInvariant [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (hreachable :
      Reachable (initialConfiguration order program) configuration) :
    OperationalInvariant order program configuration := by
  exact
    { canonicalForks :=
        reachable_canonicalForkHistory order program configuration hreachable
      controlsFollowForks :=
        reachable_controlsFollowForks order program configuration hreachable
      forksPrecedeEvents :=
        reachable_forksPrecedeEvents order program configuration hreachable
      countersMatchHistory :=
        reachable_countersMatchHistory order program configuration hreachable
      queuesMatchHistory :=
        reachable_queuesMatchHistory order program configuration hreachable
      eventsHaveControls :=
        reachable_eventsHaveControls order program configuration hreachable
      controlsHaveExecution :=
        reachable_controlsHaveExecution order program configuration hreachable }

end ZipperGenSpawn
