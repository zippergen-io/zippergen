import ZipperGenSpawn.OperationalCorrespondence
import ZipperGenSpawn.TimedMSC
import ZipperGenSpawn.FIFORank

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- One ordinary endpoint event in a scheduler linearization.  Silent
    administration and fork markers do not enter an individual session MSC. -/
structure ScheduleEntry (Lifeline : Type u) where
  owner : Lifeline
  session : SessionName
  letter : LocalLetter Lifeline ChannelScope

def emissionSchedule (A : Lifeline) (session : SessionName) :
    Emission Lifeline -> List (ScheduleEntry Lifeline)
  | .event letter => [{ owner := A, session := session, letter := letter }]
  | .silent | .fork .. => []

/-- A reachability derivation with the scheduler order retained as ghost
    proof data.  It does not change the endpoint machine. -/
inductive ScheduledReachable [DecidableEq Lifeline]
    (initial : Configuration Lifeline Field) :
    Configuration Lifeline Field -> List (ScheduleEntry Lifeline) -> Prop where
  | refl : ScheduledReachable initial initial []
  | tail {before schedule}
      (hbefore : ScheduledReachable initial before schedule)
      (A : Lifeline) (session : SessionName)
      (program residual : LocProg Lifeline Field)
      (counters' : VisitCounters Lifeline) (queues' : Queues Lifeline)
      (emission : Emission Lifeline)
      (activation : Activation Lifeline Field)
      (hcontrol : before.controls A session = some program)
      (hlocal : LocalStep A session program before.counters before.queues
        residual counters' queues' emission activation)
      (hfresh : forall child childProgram,
        activation = some (child, childProgram) ->
          before.controls A child = none) :
      ScheduledReachable initial
        { controls := setResidual before.controls A session residual activation
          counters := counters'
          queues := queues'
          history := appendEmission before.history A session emission }
        (schedule ++ emissionSchedule A session emission)

theorem scheduledReachable_reachable [DecidableEq Lifeline]
    {initial configuration : Configuration Lifeline Field}
    {schedule : List (ScheduleEntry Lifeline)}
    (h : ScheduledReachable initial configuration schedule) :
    Reachable initial configuration := by
  induction h with
  | refl => exact Reachable.refl
  | tail hbefore A session program residual counters' queues' emission
      activation hcontrol hlocal hfresh ih =>
      exact Reachable.tail ih
        (Step.run _ A session program residual counters' queues' emission
          activation hcontrol hlocal hfresh)

/-- Every ordinary reachability proof has a scheduler linearization. -/
theorem reachable_scheduled [DecidableEq Lifeline]
    {initial configuration : Configuration Lifeline Field}
    (h : Reachable initial configuration) :
    exists schedule, ScheduledReachable initial configuration schedule := by
  induction h with
  | refl => exact Exists.intro [] ScheduledReachable.refl
  | @tail before after hbefore hstep ih =>
      rcases ih with ⟨schedule, hschedule⟩
      cases hstep with
      | run A session program residual counters' queues' emission
          activation hcontrol hlocal hfresh =>
          exact Exists.intro (schedule ++ emissionSchedule A session emission)
            (ScheduledReachable.tail hschedule A session program residual
              counters' queues' emission activation hcontrol hlocal hfresh)

/-- Restrict the scheduler order to one participant and one session. -/
def scheduleWord [DecidableEq Lifeline]
    (session : SessionName) (A : Lifeline) :
    List (ScheduleEntry Lifeline) -> Word (LocalLetter Lifeline ChannelScope)
  | [] => []
  | entry :: rest =>
      if entry.owner = A ∧ entry.session = session then
        entry.letter :: scheduleWord session A rest
      else
        scheduleWord session A rest

/-- One scheduler entry contributes to exactly one participant/session word. -/
def schedulePick [DecidableEq Lifeline]
    (session : SessionName) (A : Lifeline) :
    ScheduleEntry Lifeline -> Option (LocalLetter Lifeline ChannelScope) :=
  fun entry =>
    if entry.owner = A ∧ entry.session = session then some entry.letter
    else none

theorem schedulePick_eq_some [DecidableEq Lifeline]
    {session : SessionName} {A : Lifeline}
    {entry : ScheduleEntry Lifeline}
    {letter : LocalLetter Lifeline ChannelScope}
    (h : schedulePick session A entry = some letter) :
    entry.owner = A /\ entry.session = session /\ entry.letter = letter := by
  cases entry with
  | mk owner current value =>
      simp [schedulePick] at h
      exact ⟨h.1.1, h.1.2, h.2⟩

theorem scheduleWord_eq_filterMap [DecidableEq Lifeline]
    (session : SessionName) (A : Lifeline)
    (schedule : List (ScheduleEntry Lifeline)) :
    scheduleWord session A schedule =
      schedule.filterMap (schedulePick session A) := by
  induction schedule with
  | nil => rfl
  | cons entry rest ih =>
      by_cases h : entry.owner = A ∧ entry.session = session
      · simp [scheduleWord, schedulePick, h, ih]
      · simp [scheduleWord, schedulePick, h, ih]

@[simp] theorem scheduleWord_append [DecidableEq Lifeline]
    (session : SessionName) (A : Lifeline)
    (left right : List (ScheduleEntry Lifeline)) :
    scheduleWord session A (left ++ right) =
      scheduleWord session A left ++ scheduleWord session A right := by
  induction left with
  | nil => rfl
  | cons entry rest ih =>
      by_cases howner : entry.owner = A
      · subst A
        by_cases hsession : entry.session = session
        · simp [scheduleWord, hsession, ih]
        · simp [scheduleWord, hsession, ih]
      · simp [scheduleWord, howner, ih]

theorem scheduleWord_take_prefix [DecidableEq Lifeline]
    (session : SessionName) (A : Lifeline)
    (schedule : List (ScheduleEntry Lifeline)) (count : Nat) :
    ListPrefix (scheduleWord session A (schedule.take count))
      (scheduleWord session A schedule) := by
  refine ⟨scheduleWord session A (schedule.drop count), ?_⟩
  rw [← scheduleWord_append, List.take_append_drop]

@[simp] theorem scheduleWord_emission_owner [DecidableEq Lifeline]
    (target current : SessionName) (A : Lifeline)
    (emission : Emission Lifeline) :
    scheduleWord target A (emissionSchedule A current emission) =
      if current = target then emissionWord emission else [] := by
  cases emission <;>
    simp [emissionSchedule, scheduleWord, emissionWord]

@[simp] theorem projectedSessionWord_appendEmission_owner
    [DecidableEq Lifeline]
    (history : ProjectedSpawnTuple Lifeline ChannelScope)
    (A : Lifeline) (target current : SessionName)
    (emission : Emission Lifeline) :
    projectedSessionWord target
        (appendEmission history A current emission A) =
      projectedSessionWord target (history A) ++
        if current = target then emissionWord emission else [] := by
  cases emission <;>
    simp [appendEmission, projectedSessionWord, emissionWord]

@[simp] theorem scheduleWord_emission_other [DecidableEq Lifeline]
    {A B : Lifeline} (hne : Not (A = B))
    (session current : SessionName) (emission : Emission Lifeline) :
    scheduleWord current B (emissionSchedule A session emission) = [] := by
  cases emission <;> simp [emissionSchedule, scheduleWord, hne]

/-- Per-participant session histories are exactly the corresponding
    restrictions of the scheduler linearization. -/
theorem scheduledReachable_scheduleWord [DecidableEq Lifeline]
    {initial configuration : Configuration Lifeline Field}
    {schedule : List (ScheduleEntry Lifeline)}
    (h : ScheduledReachable initial configuration schedule)
    (hempty : initial.history = emptyHistory)
    (A : Lifeline) (session : SessionName) :
    scheduleWord session A schedule =
      projectedSessionWord session (configuration.history A) := by
  induction h with
  | refl => simp [hempty, emptyHistory, scheduleWord, projectedSessionWord]
  | tail hbefore actor current program residual counters' queues' emission
      activation hcontrol hlocal hfresh ih =>
      rw [scheduleWord_append]
      by_cases hactor : A = actor
      · subst actor
        rw [scheduleWord_emission_owner]
        rw [projectedSessionWord_appendEmission_owner]
        rw [ih]
      · have hne : Not (actor = A) := fun h => hactor h.symm
        rw [scheduleWord_emission_other hne]
        simp only [List.append_nil]
        simpa [projectedSessionWord_appendEmission_other _ hactor] using ih

theorem reachable_has_scheduleWord [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (h : Reachable (initialConfiguration order source) configuration) :
    exists schedule,
      ScheduledReachable (initialConfiguration order source) configuration
        schedule /\
      forall A session,
        scheduleWord session A schedule =
          projectedSessionWord session (configuration.history A) := by
  rcases reachable_scheduled h with ⟨schedule, hschedule⟩
  refine ⟨schedule, hschedule, ?_⟩
  intro A session
  exact scheduledReachable_scheduleWord hschedule rfl A session

/-- Global positions occupied by one participant/session restriction. -/
def schedulePositions [DecidableEq Lifeline]
    (session : SessionName) (A : Lifeline) :
    Nat -> List (ScheduleEntry Lifeline) -> List Nat
  | _, [] => []
  | offset, entry :: rest =>
      if entry.owner = A ∧ entry.session = session then
        offset :: schedulePositions session A (offset + 1) rest
      else
        schedulePositions session A (offset + 1) rest

@[simp] theorem schedulePositions_length [DecidableEq Lifeline]
    (session : SessionName) (A : Lifeline)
    (offset : Nat) (schedule : List (ScheduleEntry Lifeline)) :
    (schedulePositions session A offset schedule).length =
      (scheduleWord session A schedule).length := by
  induction schedule generalizing offset with
  | nil => rfl
  | cons entry rest ih =>
      by_cases h : entry.owner = A ∧ entry.session = session
      · simp [schedulePositions, scheduleWord, h, ih]
      · simp [schedulePositions, scheduleWord, h, ih]

theorem schedulePositions_lowerBound [DecidableEq Lifeline]
    (session : SessionName) (A : Lifeline)
    (offset : Nat) (schedule : List (ScheduleEntry Lifeline)) :
    forall position,
      position ∈ schedulePositions session A offset schedule ->
        offset <= position := by
  induction schedule generalizing offset with
  | nil => simp [schedulePositions]
  | cons entry rest ih =>
      intro position hmem
      by_cases h : entry.owner = A ∧ entry.session = session
      · rw [schedulePositions, if_pos h] at hmem
        simp only [List.mem_cons] at hmem
        rcases hmem with heq | htail
        · subst position
          exact Nat.le_refl offset
        · exact Nat.le_trans (Nat.le_add_right offset 1)
            (ih (offset + 1) position htail)
      · rw [schedulePositions, if_neg h] at hmem
        change position ∈
          schedulePositions session A (offset + 1) rest at hmem
        exact Nat.le_trans (Nat.le_add_right offset 1)
          (ih (offset + 1) position hmem)

theorem schedulePositions_pairwise [DecidableEq Lifeline]
    (session : SessionName) (A : Lifeline)
    (offset : Nat) (schedule : List (ScheduleEntry Lifeline)) :
    (schedulePositions session A offset schedule).Pairwise (fun x y => x < y) := by
  induction schedule generalizing offset with
  | nil => exact List.Pairwise.nil
  | cons entry rest ih =>
      by_cases h : entry.owner = A ∧ entry.session = session
      · simp only [schedulePositions, h]
        apply List.Pairwise.cons
        · intro position hmem
          have hlower := schedulePositions_lowerBound session A
            (offset + 1) rest position hmem
          omega
        · exact ih (offset + 1)
      · simpa [schedulePositions, h] using ih (offset + 1)

/-- The scheduler position that realizes one local word event, together with
    the rank equation relating global and local positions. -/
structure ScheduleOrigin [DecidableEq Lifeline]
    (session : SessionName) (schedule : List (ScheduleEntry Lifeline))
    (words : Lifeline -> Word (LocalLetter Lifeline ChannelScope))
    (event : WordEvent words) where
  index : Nat
  inBounds : index < schedule.length
  picked :
    schedulePick session event.owner
      (schedule.get ⟨index, inBounds⟩) = some event.letter
  rank :
    selectedRank (schedulePick session event.owner) schedule index =
      event.index

theorem scheduleOrigin_exists [DecidableEq Lifeline]
    (session : SessionName) (schedule : List (ScheduleEntry Lifeline))
    (words : Lifeline -> Word (LocalLetter Lifeline ChannelScope))
    (hwords : forall A, scheduleWord session A schedule = words A)
    (event : WordEvent words) :
    Nonempty (ScheduleOrigin session schedule words event) := by
  have hlookup :
      (schedule.filterMap (schedulePick session event.owner))[event.index]? =
        some event.letter := by
    rw [← scheduleWord_eq_filterMap, hwords]
    exact List.getElem?_eq_getElem event.inBounds
  rcases exists_selected_of_filterMap_getElem?
      (schedulePick session event.owner) schedule event.index event.letter
      hlookup with ⟨index, hbound, hpick, hrank⟩
  exact ⟨⟨index, hbound, hpick, hrank⟩⟩

noncomputable def scheduleOrigin [DecidableEq Lifeline]
    (session : SessionName) (schedule : List (ScheduleEntry Lifeline))
    (words : Lifeline -> Word (LocalLetter Lifeline ChannelScope))
    (hwords : forall A, scheduleWord session A schedule = words A)
    (event : WordEvent words) : ScheduleOrigin session schedule words event :=
  Classical.choice (scheduleOrigin_exists session schedule words hwords event)

/-- Timestamp of a local word event in the scheduler linearization. -/
noncomputable def scheduleTime [DecidableEq Lifeline]
    (session : SessionName) (schedule : List (ScheduleEntry Lifeline))
    (words : Lifeline -> Word (LocalLetter Lifeline ChannelScope))
    (hwords : forall A, scheduleWord session A schedule = words A)
    (event : WordEvent words) : Nat :=
  (scheduleOrigin session schedule words hwords event).index

theorem scheduleWord_take_scheduleTime [DecidableEq Lifeline]
    (session : SessionName) (schedule : List (ScheduleEntry Lifeline))
    (words : Lifeline -> Word (LocalLetter Lifeline ChannelScope))
    (hwords : forall A, scheduleWord session A schedule = words A)
    (event : WordEvent words) :
    scheduleWord session event.owner
        (schedule.take (scheduleTime session schedule words hwords event)) =
      (words event.owner).take event.index := by
  let origin := scheduleOrigin session schedule words hwords event
  have hselected :
      (schedulePick session event.owner
        (schedule.get ⟨origin.index, origin.inBounds⟩)).isSome = true := by
    rw [origin.picked]
    rfl
  rw [scheduleWord_eq_filterMap]
  rw [show scheduleTime session schedule words hwords event = origin.index by rfl]
  rw [filterMap_take_selected_position
    (schedulePick session event.owner) schedule origin.index
      origin.inBounds hselected]
  rw [origin.rank]
  rw [← hwords]
  rw [scheduleWord_eq_filterMap]

/-- If an event already occurs in the participant word extracted from a
    strict scheduler prefix, its scheduler timestamp is below that cut. -/
theorem scheduleTime_lt_of_index_lt_prefix [DecidableEq Lifeline]
    (session : SessionName) (schedule : List (ScheduleEntry Lifeline))
    (words : Lifeline -> Word (LocalLetter Lifeline ChannelScope))
    (hwords : forall A, scheduleWord session A schedule = words A)
    (event : WordEvent words) (cut : Nat)
    (hindex : event.index <
      (scheduleWord session event.owner (schedule.take cut)).length) :
    scheduleTime session schedule words hwords event < cut := by
  let prefixWords :
      Lifeline -> Word (LocalLetter Lifeline ChannelScope) :=
    fun A => scheduleWord session A (schedule.take cut)
  let prefixEvent : WordEvent prefixWords :=
    { owner := event.owner, index := event.index, inBounds := hindex }
  let small := scheduleOrigin session (schedule.take cut) prefixWords
    (fun _ => rfl) prefixEvent
  let full := scheduleOrigin session schedule words hwords event
  have hsmallCut : small.index < cut :=
    Nat.lt_of_lt_of_le small.inBounds (List.length_take_le cut schedule)
  have hprefixSchedule : ListPrefix (schedule.take cut) schedule := by
    exact ⟨schedule.drop cut, List.take_append_drop cut schedule⟩
  have hsmallGet :
      (schedule.take cut)[small.index]? =
        some ((schedule.take cut).get ⟨small.index, small.inBounds⟩) :=
    List.getElem?_eq_getElem small.inBounds
  have hglobalGet := ListPrefix.getElem?_eq hprefixSchedule hsmallGet
  rcases List.getElem?_eq_some_iff.mp hglobalGet with
    ⟨hbound, hletter⟩
  have hwordPrefix : ListPrefix (prefixWords event.owner)
      (words event.owner) := by
    simpa [prefixWords, hwords] using
      scheduleWord_take_prefix session event.owner schedule cut
  have hprefLetter :
      (prefixWords event.owner)[event.index]? = some prefixEvent.letter :=
    List.getElem?_eq_getElem prefixEvent.inBounds
  have hfullLetter := ListPrefix.getElem?_eq hwordPrefix hprefLetter
  have hsameLetter : prefixEvent.letter = event.letter := by
    rw [List.getElem?_eq_getElem event.inBounds] at hfullLetter
    exact Option.some.inj hfullLetter.symm
  have hpickSmall :
      schedulePick session event.owner
        (schedule.get ⟨small.index, hbound⟩) = some event.letter := by
    have heq : schedule.get ⟨small.index, hbound⟩ =
        (schedule.take cut).get ⟨small.index, small.inBounds⟩ := hletter
    rw [heq]
    simpa [prefixEvent, hsameLetter] using small.picked
  have hsmallSelected :
      (schedulePick session event.owner
        (schedule.get ⟨small.index, hbound⟩)).isSome = true := by
    rw [hpickSmall]
    rfl
  have hfullSelected :
      (schedulePick session event.owner
        (schedule.get ⟨full.index, full.inBounds⟩)).isSome = true := by
    rw [full.picked]
    rfl
  have hrankSmall :
      selectedRank (schedulePick session event.owner) schedule small.index =
        event.index := by
    rw [selectedRank_of_prefix (schedulePick session event.owner)
      hprefixSchedule small.inBounds]
    exact small.rank
  have heq : small.index = full.index := by
    apply selectedRank_injective_at_selected
      (schedulePick session event.owner) schedule hbound full.inBounds
      hsmallSelected hfullSelected
    exact hrankSmall.trans full.rank.symm
  simpa [scheduleTime, full, heq] using hsmallCut

theorem scheduleTime_localSucc [DecidableEq Lifeline]
    (session : SessionName) (schedule : List (ScheduleEntry Lifeline))
    (words : Lifeline -> Word (LocalLetter Lifeline ChannelScope))
    (hwords : forall A, scheduleWord session A schedule = words A)
    {e f : WordEvent words}
    (hlocal :
      LocalSucc (fun x : WordEvent words => x.owner)
        (fun x : WordEvent words => x.localIndex) e f) :
    scheduleTime session schedule words hwords e <
      scheduleTime session schedule words hwords f := by
  rcases hlocal with ⟨howner, hindex⟩
  let horiginE := scheduleOrigin session schedule words hwords e
  let horiginF := scheduleOrigin session schedule words hwords f
  have howner' : e.owner = f.owner := howner
  have hpickE :
      (schedulePick session e.owner
        (schedule.get ⟨horiginE.index, horiginE.inBounds⟩)).isSome = true := by
    rw [horiginE.picked]
    rfl
  have hpickF :
      (schedulePick session e.owner
        (schedule.get ⟨horiginF.index, horiginF.inBounds⟩)).isSome = true := by
    rw [howner']
    rw [horiginF.picked]
    rfl
  have htime : horiginE.index < horiginF.index := by
    apply (selectedRank_lt_iff_at_selected
      (schedulePick session e.owner) schedule horiginE.inBounds
        horiginF.inBounds hpickE hpickF).1
    rw [horiginE.rank]
    have hrankF :
        selectedRank (schedulePick session e.owner) schedule horiginF.index =
          f.index := by
      simpa [howner'] using horiginF.rank
    rw [hrankF]
    dsimp [WordEvent.localIndex] at hindex
    omega
  simpa [scheduleTime] using htime

/-- Every reachable session restriction admits one natural-number timestamp
    per event that strictly increases along participant-local successor
    edges.  The timestamps are positions in an actual scheduler run. -/
theorem reachable_session_localTime [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (h : Reachable (initialConfiguration order source) configuration)
    (session : SessionName) :
    exists time : WordEvent (projectedSessionTuple session configuration.history) -> Nat,
      forall {e f},
        LocalSucc
          (fun x : WordEvent
            (projectedSessionTuple session configuration.history) => x.owner)
          (fun x : WordEvent
            (projectedSessionTuple session configuration.history) => x.localIndex)
          e f ->
        time e < time f := by
  rcases reachable_has_scheduleWord order source configuration h with
    ⟨schedule, _hscheduled, hwords⟩
  let time :
      WordEvent (projectedSessionTuple session configuration.history) -> Nat :=
    scheduleTime session schedule
      (projectedSessionTuple session configuration.history) (fun A => hwords A session)
  refine ⟨time, ?_⟩
  intro e f hlocal
  exact scheduleTime_localSucc session schedule
    (projectedSessionTuple session configuration.history)
    (fun A => hwords A session) hlocal

/-- Queue items contributed by sends in one session-local word. -/
def wordSentItems [DecidableEq Lifeline]
    (scope : ChannelScope) (receiver : Lifeline) :
    Word (LocalLetter Lifeline ChannelScope) -> List QueueItem
  | [] => []
  | letter :: rest =>
      let tail := wordSentItems scope receiver rest
      match letter with
      | .send target currentScope =>
          if target = receiver ∧ currentScope = scope then .data :: tail else tail
      | .ctrlSend target currentScope site kind =>
          if target = receiver ∧ currentScope = scope then
            .control site kind :: tail
          else tail
      | _ => tail

/-- Queue items contributed by receives in one session-local word. -/
def wordReceivedItems [DecidableEq Lifeline]
    (scope : ChannelScope) (sender : Lifeline) :
    Word (LocalLetter Lifeline ChannelScope) -> List QueueItem
  | [] => []
  | letter :: rest =>
      let tail := wordReceivedItems scope sender rest
      match letter with
      | .recv source currentScope =>
          if source = sender ∧ currentScope = scope then .data :: tail else tail
      | .ctrlRecv source currentScope site kind =>
          if source = sender ∧ currentScope = scope then
            .control site kind :: tail
          else tail
      | _ => tail

theorem sentItems_eq_wordSentItems [DecidableEq Lifeline]
    (session : SessionName) (scope : ChannelScope) (receiver : Lifeline)
    (history : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    sentItems session scope receiver history =
      wordSentItems scope receiver (projectedSessionWord session history) := by
  induction history with
  | nil => rfl
  | cons head rest ih =>
      cases head with
      | fork parent site child => simpa [sentItems, projectedSessionWord] using ih
      | event current letter =>
          by_cases hs : current = session
          · subst current
            cases letter <;>
              simp [sentItems, projectedSessionWord, wordSentItems, ih]
          · simp [sentItems, projectedSessionWord, hs, ih]

theorem receivedItems_eq_wordReceivedItems [DecidableEq Lifeline]
    (session : SessionName) (scope : ChannelScope) (sender : Lifeline)
    (history : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    receivedItems session scope sender history =
      wordReceivedItems scope sender
        (projectedSessionWord session history) := by
  induction history with
  | nil => rfl
  | cons head rest ih =>
      cases head with
      | fork parent site child =>
          simpa [receivedItems, projectedSessionWord] using ih
      | event current letter =>
          by_cases hs : current = session
          · subst current
            cases letter <;>
              simp [receivedItems, projectedSessionWord,
                wordReceivedItems, ih]
          · simp [receivedItems, projectedSessionWord, hs, ih]

/-- The extracted scheduler trace retains the exact FIFO equation, rather
    than merely send/receive counts. -/
theorem reachable_schedule_queueEquation [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (h : Reachable (initialConfiguration order source) configuration) :
    exists schedule,
      ScheduledReachable (initialConfiguration order source) configuration
        schedule /\
      (forall A session,
        scheduleWord session A schedule =
          projectedSessionWord session (configuration.history A)) /\
      forall session scope sender receiver,
        wordSentItems scope receiver (scheduleWord session sender schedule) =
          wordReceivedItems scope sender
              (scheduleWord session receiver schedule) ++
            configuration.queues session scope sender receiver := by
  rcases reachable_has_scheduleWord order source configuration h with
    ⟨schedule, hschedule, hwords⟩
  refine ⟨schedule, hschedule, hwords, ?_⟩
  intro session scope sender receiver
  rw [hwords sender session, hwords receiver session]
  rw [← sentItems_eq_wordSentItems, ← receivedItems_eq_wordReceivedItems]
  exact reachable_queuesMatchHistory order source configuration h
    session scope sender receiver

theorem scheduledReachable_full_routePrefix [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (schedule : List (ScheduleEntry Lifeline))
    (h : ScheduledReachable (initialConfiguration order source)
      configuration schedule)
    (session : SessionName) (scope : ChannelScope)
    (sender receiver : Lifeline) :
    ListPrefix
      (wordReceivedItems scope sender
        (scheduleWord session receiver schedule))
      (wordSentItems scope receiver
        (scheduleWord session sender schedule)) := by
  have hreachable := scheduledReachable_reachable h
  have hwords := scheduledReachable_scheduleWord h rfl
  refine ⟨configuration.queues session scope sender receiver, ?_⟩
  rw [hwords sender session, hwords receiver session]
  rw [← sentItems_eq_wordSentItems, ← receivedItems_eq_wordReceivedItems]
  exact (reachable_queuesMatchHistory order source configuration hreachable
    session scope sender receiver).symm

/-- FIFO feasibility holds at every event prefix of the extracted scheduler
    order, not only in the final configuration.  This is the temporal fact
    that rules out a receive being matched by a later send. -/
theorem scheduledReachable_allPrefixes_routePrefix [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    forall (configuration : Configuration Lifeline Field)
      (schedule : List (ScheduleEntry Lifeline)),
      ScheduledReachable (initialConfiguration order source)
        configuration schedule ->
      forall n, n <= schedule.length ->
        forall session scope sender receiver,
          ListPrefix
            (wordReceivedItems scope sender
              (scheduleWord session receiver (schedule.take n)))
            (wordSentItems scope receiver
              (scheduleWord session sender (schedule.take n))) := by
  intro configuration schedule hscheduled
  induction hscheduled with
  | refl =>
      intro n hn session scope sender receiver
      have hn0 : n = 0 := by simpa using hn
      subst n
      exact ⟨[], rfl⟩
  | tail hbefore actor current program residual counters' queues' emission
      activation hcontrol hlocal hfresh ih =>
      rename_i priorConfiguration prior
      intro n hn session scope sender receiver
      cases emission with
      | silent =>
          have hn' : n <= prior.length := by
            simpa [emissionSchedule] using hn
          simpa [emissionSchedule] using
            ih n hn' session scope sender receiver
      | fork parent site child =>
          have hn' : n <= prior.length := by
            simpa [emissionSchedule] using hn
          simpa [emissionSchedule] using
            ih n hn' session scope sender receiver
      | event letter =>
          by_cases hold : n <= prior.length
          · rw [List.take_append_of_le_length hold]
            exact ih n hold session scope sender receiver
          · have hlen :
                (prior ++
                  emissionSchedule actor current (.event letter)).length =
                    prior.length + 1 := by
                simp [emissionSchedule]
            have hn' : n = prior.length + 1 := by
              rw [hlen] at hn
              omega
            rw [List.take_of_length_le]
            · exact scheduledReachable_full_routePrefix order source
                _ _
                (ScheduledReachable.tail hbefore actor current program residual
                  counters' queues' (.event letter) activation
                  hcontrol hlocal hfresh)
                session scope sender receiver
            · simpa [emissionSchedule, hn']

theorem localStep_recv_queue_head [DecidableEq Lifeline]
    {A sender : Lifeline} {session : SessionName} {scope : ChannelScope}
    {program residual : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {activation : Activation Lifeline Field}
    (hstep : LocalStep A session program counters queues residual counters'
      queues' (.event (.recv sender scope)) activation) :
    sender ≠ A /\
      exists tail, queues session scope sender A = .data :: tail := by
  rcases localStep_queue_effect hstep with hsilent | ⟨letter, hemission, hletter⟩
  · rcases hsilent with ⟨hsilent | ⟨parent, site, child, hfork⟩, _⟩
    · cases hsilent
    · cases hfork
  · cases hemission
    cases hletter with
    | recv queues sender scope tail hne hhead => exact ⟨hne, tail, hhead⟩

theorem localStep_ctrlRecv_queue_head [DecidableEq Lifeline]
    {A sender : Lifeline} {session : SessionName} {scope : ChannelScope}
    {site : ControlTag} {kind : ChoiceKind}
    {program residual : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {activation : Activation Lifeline Field}
    (hstep : LocalStep A session program counters queues residual counters'
      queues' (.event (.ctrlRecv sender scope site kind)) activation) :
    sender ≠ A /\
      exists tail,
        queues session scope sender A = .control site kind :: tail := by
  rcases localStep_queue_effect hstep with hsilent | ⟨letter, hemission, hletter⟩
  · rcases hsilent with ⟨hsilent | ⟨parent, forkSite, child, hfork⟩, _⟩
    · cases hsilent
    · cases hfork
  · cases hemission
    cases hletter with
    | ctrlRecv queues sender scope tail site kind hne hhead =>
        exact ⟨hne, tail, hhead⟩

/-- Immediately before a data receive, its FIFO-rank send is already in the
    scheduler prefix.  This is stronger than final prefix accounting and is
    the fact needed for message-edge time monotonicity. -/
theorem scheduledReachable_last_recv_available [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    {before : Configuration Lifeline Field}
    {prior : List (ScheduleEntry Lifeline)}
    (hbefore : ScheduledReachable (initialConfiguration order source)
      before prior)
    (A sender : Lifeline) (session : SessionName) (scope : ChannelScope)
    (program residual : LocProg Lifeline Field)
    (counters' : VisitCounters Lifeline) (queues' : Queues Lifeline)
    (activation : Activation Lifeline Field)
    (hstep : LocalStep A session program before.counters before.queues
      residual counters' queues' (.event (.recv sender scope)) activation) :
    (wordSentItems scope A (scheduleWord session sender prior))[
        (wordReceivedItems scope sender
          (scheduleWord session A prior)).length]? = some .data := by
  rcases localStep_recv_queue_head hstep with ⟨_hne, tail, hqueue⟩
  have hreachable := scheduledReachable_reachable hbefore
  have hwords := scheduledReachable_scheduleWord hbefore rfl
  have heq := reachable_queuesMatchHistory order source before hreachable
    session scope sender A
  rw [sentItems_eq_wordSentItems, receivedItems_eq_wordReceivedItems,
    ← hwords sender session, ← hwords A session, hqueue] at heq
  rw [heq]
  simp

theorem scheduledReachable_last_ctrlRecv_available [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    {before : Configuration Lifeline Field}
    {prior : List (ScheduleEntry Lifeline)}
    (hbefore : ScheduledReachable (initialConfiguration order source)
      before prior)
    (A sender : Lifeline) (session : SessionName) (scope : ChannelScope)
    (site : ControlTag) (kind : ChoiceKind)
    (program residual : LocProg Lifeline Field)
    (counters' : VisitCounters Lifeline) (queues' : Queues Lifeline)
    (activation : Activation Lifeline Field)
    (hstep : LocalStep A session program before.counters before.queues
      residual counters' queues'
        (.event (.ctrlRecv sender scope site kind)) activation) :
    (wordSentItems scope A (scheduleWord session sender prior))[
        (wordReceivedItems scope sender
          (scheduleWord session A prior)).length]? =
      some (.control site kind) := by
  rcases localStep_ctrlRecv_queue_head hstep with ⟨_hne, tail, hqueue⟩
  have hreachable := scheduledReachable_reachable hbefore
  have hwords := scheduledReachable_scheduleWord hbefore rfl
  have heq := reachable_queuesMatchHistory order source before hreachable
    session scope sender A
  rw [sentItems_eq_wordSentItems, receivedItems_eq_wordReceivedItems,
    ← hwords sender session, ← hwords A session, hqueue] at heq
  rw [heq]
  simp

/-- Readiness of the receive stored at one global scheduler position, stated
    solely in terms of the scheduler prefix before that position. -/
def ReceiveReadyAt [DecidableEq Lifeline]
    (schedule : List (ScheduleEntry Lifeline)) (position : Nat)
    (entry : ScheduleEntry Lifeline) : Prop :=
  match entry.letter with
  | .recv sender scope =>
      sender ≠ entry.owner /\
      (wordSentItems scope entry.owner
          (scheduleWord entry.session sender (schedule.take position)))[
        (wordReceivedItems scope sender
          (scheduleWord entry.session entry.owner
            (schedule.take position))).length]? = some .data
  | .ctrlRecv sender scope site kind =>
      sender ≠ entry.owner /\
      (wordSentItems scope entry.owner
          (scheduleWord entry.session sender (schedule.take position)))[
        (wordReceivedItems scope sender
          (scheduleWord entry.session entry.owner
            (schedule.take position))).length]? =
        some (.control site kind)
  | _ => True

/-- Every receive in a scheduler trace was enabled by a matching FIFO head in
    the strict scheduler prefix before it. -/
theorem scheduledReachable_receiveReady [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    forall (configuration : Configuration Lifeline Field)
      (schedule : List (ScheduleEntry Lifeline)),
      ScheduledReachable (initialConfiguration order source)
        configuration schedule ->
      forall position entry,
        schedule[position]? = some entry ->
          ReceiveReadyAt schedule position entry := by
  intro configuration schedule hscheduled
  induction hscheduled with
  | refl =>
      intro position entry hget
      simp at hget
  | tail hbefore actor current program residual counters' queues' emission
      activation hcontrol hlocal hfresh ih =>
      rename_i priorConfiguration prior
      intro position entry hget
      cases emission with
      | silent =>
          have hget' : prior[position]? = some entry := by
            simpa [emissionSchedule] using hget
          simpa [emissionSchedule] using ih position entry hget'
      | fork parent site child =>
          have hget' : prior[position]? = some entry := by
            simpa [emissionSchedule] using hget
          simpa [emissionSchedule] using ih position entry hget'
      | event letter =>
          by_cases hpos : position < prior.length
          · have hgetPrior : prior[position]? = some entry := by
              simpa [emissionSchedule, List.getElem?_append, hpos] using hget
            have hready := ih position entry hgetPrior
            simpa [ReceiveReadyAt, emissionSchedule,
              List.take_append_of_le_length (Nat.le_of_lt hpos)] using hready
          · have hbound :
                position < (prior ++
                  emissionSchedule actor current (.event letter)).length :=
              List.getElem?_eq_some_iff.mp hget |>.1
            have hlast : position = prior.length := by
              simp [emissionSchedule] at hbound
              omega
            subst position
            have hentry :
                entry = ⟨actor, current, letter⟩ := by
              simpa [emissionSchedule] using hget.symm
            subst entry
            cases letter with
            | recv sender scope =>
                refine ⟨(localStep_recv_queue_head hlocal).1, ?_⟩
                simpa [ReceiveReadyAt, emissionSchedule] using
                  scheduledReachable_last_recv_available order source hbefore
                    actor sender current scope program residual counters'
                    queues' activation hlocal
            | ctrlRecv sender scope site kind =>
                refine ⟨(localStep_ctrlRecv_queue_head hlocal).1, ?_⟩
                simpa [ReceiveReadyAt, emissionSchedule] using
                  scheduledReachable_last_ctrlRecv_available order source
                    hbefore actor sender current scope site kind program
                    residual counters' queues' activation hlocal
            | tau | action | choice | send | ctrlSend =>
                simp [ReceiveReadyAt]

end ZipperGenSpawn
