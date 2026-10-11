import ZipperGenSpawn.OperationalSchedule

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- The item contributed by a send on one fixed FIFO route. -/
def sendPick [DecidableEq Lifeline]
    (receiver : Lifeline) (scope : ChannelScope) :
    LocalLetter Lifeline ChannelScope -> Option QueueItem
  | .send target current =>
      if target = receiver ∧ current = scope then some .data else none
  | .ctrlSend target current site kind =>
      if target = receiver ∧ current = scope then
        some (.control site kind)
      else none
  | _ => none

/-- The item contributed by a receive on one fixed FIFO route. -/
def receivePick [DecidableEq Lifeline]
    (sender : Lifeline) (scope : ChannelScope) :
    LocalLetter Lifeline ChannelScope -> Option QueueItem
  | .recv source current =>
      if source = sender ∧ current = scope then some .data else none
  | .ctrlRecv source current site kind =>
      if source = sender ∧ current = scope then
        some (.control site kind)
      else none
  | _ => none

theorem wordSentItems_eq_filterMap [DecidableEq Lifeline]
    (scope : ChannelScope) (receiver : Lifeline)
    (word : Word (LocalLetter Lifeline ChannelScope)) :
    wordSentItems scope receiver word =
      word.filterMap (sendPick receiver scope) := by
  induction word with
  | nil => rfl
  | cons letter rest ih =>
      cases letter with
      | tau | action | choice | recv | ctrlRecv =>
          simp [wordSentItems, sendPick, ih]
      | send target current =>
          by_cases h : target = receiver ∧ current = scope <;>
            simp [wordSentItems, sendPick, ih, h]
      | ctrlSend target current site kind =>
          by_cases h : target = receiver ∧ current = scope <;>
            simp [wordSentItems, sendPick, ih, h]

theorem wordReceivedItems_eq_filterMap [DecidableEq Lifeline]
    (scope : ChannelScope) (sender : Lifeline)
    (word : Word (LocalLetter Lifeline ChannelScope)) :
    wordReceivedItems scope sender word =
      word.filterMap (receivePick sender scope) := by
  induction word with
  | nil => rfl
  | cons letter rest ih =>
      cases letter with
      | tau | action | choice | send | ctrlSend =>
          simp [wordReceivedItems, receivePick, ih]
      | recv source current =>
          by_cases h : source = sender ∧ current = scope <;>
            simp [wordReceivedItems, receivePick, ih, h]
      | ctrlRecv source current site kind =>
          by_cases h : source = sender ∧ current = scope <;>
            simp [wordReceivedItems, receivePick, ih, h]

inductive SendWitness (receiver : Lifeline) (scope : ChannelScope) :
    QueueItem -> LocalLetter Lifeline ChannelScope -> Prop where
  | data : SendWitness receiver scope .data (.send receiver scope)
  | control (site : ControlTag) (kind : ChoiceKind) :
      SendWitness receiver scope (.control site kind)
        (.ctrlSend receiver scope site kind)

inductive ReceiveWitness (sender : Lifeline) (scope : ChannelScope) :
    QueueItem -> LocalLetter Lifeline ChannelScope -> Prop where
  | data : ReceiveWitness sender scope .data (.recv sender scope)
  | control (site : ControlTag) (kind : ChoiceKind) :
      ReceiveWitness sender scope (.control site kind)
        (.ctrlRecv sender scope site kind)

theorem sendWitness_pick [DecidableEq Lifeline]
    {receiver : Lifeline} {scope : ChannelScope} {item : QueueItem}
    {letter : LocalLetter Lifeline ChannelScope}
    (h : SendWitness receiver scope item letter) :
    sendPick receiver scope letter = some item := by
  cases h <;> simp [sendPick]

theorem sendPick_witness [DecidableEq Lifeline]
    {receiver : Lifeline} {scope : ChannelScope} {item : QueueItem}
    {letter : LocalLetter Lifeline ChannelScope}
    (h : sendPick receiver scope letter = some item) :
    SendWitness receiver scope item letter := by
  cases letter <;> simp [sendPick] at h
  · rcases h with ⟨⟨rfl, rfl⟩, rfl⟩
    exact SendWitness.data
  · rcases h with ⟨⟨rfl, rfl⟩, rfl⟩
    exact SendWitness.control _ _

theorem receiveWitness_pick [DecidableEq Lifeline]
    {sender : Lifeline} {scope : ChannelScope} {item : QueueItem}
    {letter : LocalLetter Lifeline ChannelScope}
    (h : ReceiveWitness sender scope item letter) :
    receivePick sender scope letter = some item := by
  cases h <;> simp [receivePick]

theorem receivePick_witness [DecidableEq Lifeline]
    {sender : Lifeline} {scope : ChannelScope} {item : QueueItem}
    {letter : LocalLetter Lifeline ChannelScope}
    (h : receivePick sender scope letter = some item) :
    ReceiveWitness sender scope item letter := by
  cases letter <;> simp [receivePick] at h
  · rcases h with ⟨⟨rfl, rfl⟩, rfl⟩
    exact ReceiveWitness.data
  · rcases h with ⟨⟨rfl, rfl⟩, rfl⟩
    exact ReceiveWitness.control _ _

theorem receiveKind_witness
    {sender : Lifeline} {scope : ChannelScope}
    {letter : LocalLetter Lifeline ChannelScope}
    (h : LocalLetter.toEventKind letter = EventKind.recv sender scope) :
    exists item, ReceiveWitness sender scope item letter := by
  cases letter <;> simp at h
  · rcases h with ⟨rfl, rfl⟩
    exact ⟨.data, ReceiveWitness.data⟩
  · rcases h with ⟨rfl, rfl⟩
    exact ⟨.control _ _, ReceiveWitness.control _ _⟩

theorem sendWitness_kind
    {receiver : Lifeline} {scope : ChannelScope} {item : QueueItem}
    {letter : LocalLetter Lifeline ChannelScope}
    (h : SendWitness receiver scope item letter) :
    LocalLetter.toEventKind letter = EventKind.send receiver scope := by
  cases h <;> rfl

theorem receiveWitness_kind
    {sender : Lifeline} {scope : ChannelScope} {item : QueueItem}
    {letter : LocalLetter Lifeline ChannelScope}
    (h : ReceiveWitness sender scope item letter) :
    LocalLetter.toEventKind letter = EventKind.recv sender scope := by
  cases h <;> rfl

theorem witness_messageMatches
    {sender receiver : Lifeline} {scope : ChannelScope} {item : QueueItem}
    {send receive : LocalLetter Lifeline ChannelScope}
    (hs : SendWitness receiver scope item send)
    (hr : ReceiveWitness sender scope item receive) :
    LocalLetter.MessageMatches sender receiver send receive := by
  cases hs <;> cases hr <;> simp [LocalLetter.MessageMatches]

theorem sendWitness_functional
    {receiver1 receiver2 : Lifeline} {scope1 scope2 : ChannelScope}
    {item1 item2 : QueueItem}
    {letter : LocalLetter Lifeline ChannelScope}
    (h1 : SendWitness receiver1 scope1 item1 letter)
    (h2 : SendWitness receiver2 scope2 item2 letter) :
    receiver1 = receiver2 /\ scope1 = scope2 /\ item1 = item2 := by
  cases h1 <;> cases h2 <;> exact ⟨rfl, rfl, rfl⟩

theorem receiveWitness_functional
    {sender1 sender2 : Lifeline} {scope1 scope2 : ChannelScope}
    {item1 item2 : QueueItem}
    {letter : LocalLetter Lifeline ChannelScope}
    (h1 : ReceiveWitness sender1 scope1 item1 letter)
    (h2 : ReceiveWitness sender2 scope2 item2 letter) :
    sender1 = sender2 /\ scope1 = scope2 /\ item1 = item2 := by
  cases h1 <;> cases h2 <;> exact ⟨rfl, rfl, rfl⟩

/-- FIFO matching pairs sends and receives with equal route rank and item.
    The inequality is supplied by the receive step that consumed the item. -/
def FIFOMessage [DecidableEq Lifeline]
    (words : Lifeline -> Word (LocalLetter Lifeline ChannelScope))
    (send receive : WordEvent words) : Prop :=
  exists scope item,
    SendWitness receive.owner scope item send.letter /\
    ReceiveWitness send.owner scope item receive.letter /\
    send.owner ≠ receive.owner /\
    selectedRank (sendPick receive.owner scope) (words send.owner)
        send.index =
      selectedRank (receivePick send.owner scope) (words receive.owner)
        receive.index

theorem fifoMessage_send_recv [DecidableEq Lifeline]
    {words : Lifeline -> Word (LocalLetter Lifeline ChannelScope)}
    {send receive : WordEvent words}
    (hmsg : FIFOMessage words send receive) :
    exists A B scope,
      send.owner = A /\ receive.owner = B /\
      send.kind = EventKind.send B scope /\
      receive.kind = EventKind.recv A scope := by
  rcases hmsg with ⟨scope, item, hsend, hreceive, _hne, _hrank⟩
  refine ⟨send.owner, receive.owner, scope, rfl, rfl, ?_, ?_⟩
  · exact sendWitness_kind hsend
  · exact receiveWitness_kind hreceive

theorem fifoMessage_letter_match [DecidableEq Lifeline]
    {words : Lifeline -> Word (LocalLetter Lifeline ChannelScope)}
    {send receive : WordEvent words}
    (hmsg : FIFOMessage words send receive) :
    LocalLetter.MessageMatches send.owner receive.owner
      send.letter receive.letter := by
  rcases hmsg with ⟨scope, item, hsend, hreceive, _hne, _hrank⟩
  exact witness_messageMatches hsend hreceive

theorem fifoMessage_send_unique [DecidableEq Lifeline]
    {words : Lifeline -> Word (LocalLetter Lifeline ChannelScope)}
    {send r1 r2 : WordEvent words}
    (h1 : FIFOMessage words send r1) (h2 : FIFOMessage words send r2) :
    r1 = r2 := by
  cases r1 with
  | mk owner1 index1 bound1 =>
    cases r2 with
    | mk owner2 index2 bound2 =>
      rcases h1 with ⟨scope1, item1, hs1, hr1, _hne1, hrank1⟩
      rcases h2 with ⟨scope2, item2, hs2, hr2, _hne2, hrank2⟩
      rcases sendWitness_functional hs1 hs2 with
        ⟨howner, hscope, hitem⟩
      change owner1 = owner2 at howner
      subst owner2
      subst scope2
      subst item2
      have hindex : index1 = index2 := by
        apply selectedRank_injective_at_selected
          (receivePick send.owner scope1) (words owner1)
          bound1 bound2
        · have hpick := receiveWitness_pick hr1
          change receivePick send.owner scope1
            ((words owner1).get ⟨index1, bound1⟩) = some item1 at hpick
          rw [hpick]
          rfl
        · have hpick := receiveWitness_pick hr2
          change receivePick send.owner scope1
            ((words owner1).get ⟨index2, bound2⟩) = some item1 at hpick
          rw [hpick]
          rfl
        · exact hrank1.symm.trans hrank2
      subst index2
      rfl

theorem fifoMessage_recv_unique [DecidableEq Lifeline]
    {words : Lifeline -> Word (LocalLetter Lifeline ChannelScope)}
    {s1 s2 receive : WordEvent words}
    (h1 : FIFOMessage words s1 receive) (h2 : FIFOMessage words s2 receive) :
    s1 = s2 := by
  cases s1 with
  | mk owner1 index1 bound1 =>
    cases s2 with
    | mk owner2 index2 bound2 =>
      rcases h1 with ⟨scope1, item1, hs1, hr1, _hne1, hrank1⟩
      rcases h2 with ⟨scope2, item2, hs2, hr2, _hne2, hrank2⟩
      rcases receiveWitness_functional hr1 hr2 with
        ⟨howner, hscope, hitem⟩
      change owner1 = owner2 at howner
      subst owner2
      subst scope2
      subst item2
      have hindex : index1 = index2 := by
        apply selectedRank_injective_at_selected
          (sendPick receive.owner scope1) (words owner1)
          bound1 bound2
        · have hpick := sendWitness_pick hs1
          change sendPick receive.owner scope1
            ((words owner1).get ⟨index1, bound1⟩) = some item1 at hpick
          rw [hpick]
          rfl
        · have hpick := sendWitness_pick hs2
          change sendPick receive.owner scope1
            ((words owner1).get ⟨index2, bound2⟩) = some item1 at hpick
          rw [hpick]
          rfl
        · exact hrank1.trans hrank2.symm
      subst index2
      rfl

theorem receiveWitness_ready [DecidableEq Lifeline]
    {schedule : List (ScheduleEntry Lifeline)} {position : Nat}
    {entry : ScheduleEntry Lifeline}
    {sender : Lifeline} {scope : ChannelScope} {item : QueueItem}
    (hwitness : ReceiveWitness sender scope item entry.letter)
    (hready : ReceiveReadyAt schedule position entry) :
    sender ≠ entry.owner /\
      (wordSentItems scope entry.owner
          (scheduleWord entry.session sender (schedule.take position)))[
      (wordReceivedItems scope sender
          (scheduleWord entry.session entry.owner
            (schedule.take position))).length]? = some item := by
  cases entry with
  | mk owner session letter =>
      dsimp at hwitness ⊢
      cases hwitness <;> simpa [ReceiveReadyAt] using hready

/-- Every receive in a reachable session word has its unique FIFO-rank send. -/
theorem scheduledReachable_recv_matched_with_time [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (schedule : List (ScheduleEntry Lifeline))
    (hscheduled : ScheduledReachable (initialConfiguration order source)
      configuration schedule)
    (session : SessionName)
    {receive : WordEvent
      (projectedSessionTuple session configuration.history)}
    (hkind : exists sender scope,
      receive.kind = EventKind.recv sender scope) :
    exists send,
      FIFOMessage
        (projectedSessionTuple session configuration.history) send receive /\
      scheduleTime session schedule
          (projectedSessionTuple session configuration.history)
          (fun A => scheduledReachable_scheduleWord hscheduled rfl A session)
          send <
        scheduleTime session schedule
          (projectedSessionTuple session configuration.history)
          (fun A => scheduledReachable_scheduleWord hscheduled rfl A session)
          receive := by
  let words := projectedSessionTuple session configuration.history
  have hwords : forall A, scheduleWord session A schedule = words A :=
    fun A => scheduledReachable_scheduleWord hscheduled rfl A session
  rcases hkind with ⟨sender, scope, hkind⟩
  rcases receiveKind_witness hkind with ⟨item, hreceive⟩
  let origin := scheduleOrigin session schedule words hwords receive
  rcases schedulePick_eq_some origin.picked with
    ⟨howner, hsession, hletter⟩
  let entry : ScheduleEntry Lifeline :=
    ⟨receive.owner, session, receive.letter⟩
  have hentry : schedule[origin.index]? = some entry := by
    rw [List.getElem?_eq_getElem origin.inBounds]
    apply congrArg some
    cases hvalue : schedule.get ⟨origin.index, origin.inBounds⟩ with
    | mk owner current letter =>
        rw [hvalue] at howner hsession hletter
        dsimp at howner hsession hletter ⊢
        subst owner
        subst current
        subst letter
        simpa [entry] using hvalue
  have hready := scheduledReachable_receiveReady order source configuration
    schedule hscheduled origin.index entry hentry
  have hwitnessEntry : ReceiveWitness sender scope item entry.letter := by
    simpa [entry] using hreceive
  rcases receiveWitness_ready hwitnessEntry hready with ⟨hne, havailable⟩
  have hreceiverPrefix := scheduleWord_take_scheduleTime session schedule words
    hwords receive
  have hrankLength :
      (wordReceivedItems scope sender
        ((words receive.owner).take receive.index)).length =
      selectedRank (receivePick sender scope) (words receive.owner)
        receive.index := by
    rw [wordReceivedItems_eq_filterMap]
    rfl
  have hsendLookup :
      (wordSentItems scope receive.owner
        (scheduleWord session sender
          (schedule.take (scheduleTime session schedule words hwords receive))))[
        selectedRank (receivePick sender scope) (words receive.owner)
          receive.index]? = some item := by
    change
      (wordSentItems scope receive.owner
        (scheduleWord session sender (schedule.take origin.index)))[
          (wordReceivedItems scope sender
            (scheduleWord session receive.owner
              (schedule.take origin.index))).length]? = some item at havailable
    rw [show origin.index =
      scheduleTime session schedule words hwords receive by rfl] at havailable
    rw [hreceiverPrefix, hrankLength] at havailable
    exact havailable
  rw [wordSentItems_eq_filterMap] at hsendLookup
  rcases exists_selected_of_filterMap_getElem?
      (sendPick receive.owner scope)
      (scheduleWord session sender
        (schedule.take (scheduleTime session schedule words hwords receive)))
      _ item hsendLookup with ⟨index, hbound, hpick, hrankPrefix⟩
  have hprefix : ListPrefix
      (scheduleWord session sender
        (schedule.take (scheduleTime session schedule words hwords receive)))
      (words sender) := by
    simpa [hwords] using scheduleWord_take_prefix session sender schedule
      (scheduleTime session schedule words hwords receive)
  let sendLetter :=
    (scheduleWord session sender
      (schedule.take (scheduleTime session schedule words hwords receive))).get
        ⟨index, hbound⟩
  have hprefixGet :
      (scheduleWord session sender
        (schedule.take (scheduleTime session schedule words hwords receive)))[
          index]? = some sendLetter :=
    List.getElem?_eq_getElem hbound
  have hfullGet := ListPrefix.getElem?_eq hprefix hprefixGet
  rcases List.getElem?_eq_some_iff.mp hfullGet with
    ⟨hfullBound, hfullLetter⟩
  let send : WordEvent words :=
    ⟨sender, index, hfullBound⟩
  have hsendWitness : SendWitness receive.owner scope item send.letter := by
    apply sendPick_witness
    change sendPick receive.owner scope
      ((words sender).get ⟨index, hfullBound⟩) = some item
    have heq : (words sender).get ⟨index, hfullBound⟩ = sendLetter :=
      hfullLetter
    rw [heq]
    exact hpick
  have hmessage : FIFOMessage words send receive := by
    refine ⟨scope, item, hsendWitness, ?_, hne, ?_⟩
    · simpa [send] using hreceive
    · have hrankFull :
        selectedRank (sendPick receive.owner scope) (words sender) index =
          selectedRank (sendPick receive.owner scope)
            (scheduleWord session sender
              (schedule.take
                (scheduleTime session schedule words hwords receive))) index :=
          selectedRank_of_prefix (sendPick receive.owner scope) hprefix hbound
      change selectedRank (sendPick receive.owner scope) (words sender) index =
        selectedRank (receivePick sender scope) (words receive.owner)
          receive.index
      rw [hrankFull]
      exact hrankPrefix
  have htime := scheduleTime_lt_of_index_lt_prefix session schedule words
    hwords send (scheduleTime session schedule words hwords receive)
    (by simpa [send] using hbound)
  exact ⟨send, hmessage, htime⟩

theorem scheduledReachable_recv_matched [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (schedule : List (ScheduleEntry Lifeline))
    (hscheduled : ScheduledReachable (initialConfiguration order source)
      configuration schedule)
    (session : SessionName)
    {receive : WordEvent
      (projectedSessionTuple session configuration.history)}
    (hkind : exists sender scope,
      receive.kind = EventKind.recv sender scope) :
    exists send, FIFOMessage
      (projectedSessionTuple session configuration.history) send receive := by
  rcases scheduledReachable_recv_matched_with_time order source configuration
      schedule hscheduled session hkind with ⟨send, hmessage, _⟩
  exact ⟨send, hmessage⟩

theorem scheduledReachable_fifoMessage_time [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (schedule : List (ScheduleEntry Lifeline))
    (hscheduled : ScheduledReachable (initialConfiguration order source)
      configuration schedule)
    (session : SessionName)
    {send receive : WordEvent
      (projectedSessionTuple session configuration.history)}
    (hmessage : FIFOMessage
      (projectedSessionTuple session configuration.history) send receive) :
    scheduleTime session schedule
        (projectedSessionTuple session configuration.history)
        (fun A => scheduledReachable_scheduleWord hscheduled rfl A session)
        send <
      scheduleTime session schedule
        (projectedSessionTuple session configuration.history)
        (fun A => scheduledReachable_scheduleWord hscheduled rfl A session)
        receive := by
  rcases fifoMessage_send_recv hmessage with
    ⟨_sender, _receiver, scope, _hs, _hr, _hks, hkr⟩
  rcases scheduledReachable_recv_matched_with_time order source configuration
      schedule hscheduled session ⟨_, scope, hkr⟩ with
    ⟨matched, hmatched, htime⟩
  have heq := fifoMessage_recv_unique hmatched hmessage
  subst matched
  exact htime

theorem fifoMessage_channel_fifo [DecidableEq Lifeline]
    {words : Lifeline -> Word (LocalLetter Lifeline ChannelScope)}
    {s1 s2 r1 r2 : WordEvent words} {A B : Lifeline}
    {scope : ChannelScope}
    (hm1 : FIFOMessage words s1 r1) (hm2 : FIFOMessage words s2 r2)
    (hs1Owner : s1.owner = A) (hs2Owner : s2.owner = A)
    (hr1Owner : r1.owner = B) (hr2Owner : r2.owner = B)
    (hks1 : s1.kind = EventKind.send B scope)
    (hks2 : s2.kind = EventKind.send B scope)
    (hkr1 : r1.kind = EventKind.recv A scope)
    (hkr2 : r2.kind = EventKind.recv A scope)
    (hlt : s1.localIndex < s2.localIndex) :
    r1.localIndex < r2.localIndex := by
  cases s1 with
  | mk sowner1 sindex1 sbound1 =>
    cases s2 with
    | mk sowner2 sindex2 sbound2 =>
      cases r1 with
      | mk rowner1 rindex1 rbound1 =>
        cases r2 with
        | mk rowner2 rindex2 rbound2 =>
          change sowner1 = A at hs1Owner
          change sowner2 = A at hs2Owner
          change rowner1 = B at hr1Owner
          change rowner2 = B at hr2Owner
          subst sowner1
          subst sowner2
          subst rowner1
          subst rowner2
          rcases hm1 with
            ⟨scope1, item1, hsend1, hrecv1, _hne1, hrank1⟩
          rcases hm2 with
            ⟨scope2, item2, hsend2, hrecv2, _hne2, hrank2⟩
          have hscope1 : scope1 = scope := by
            have hk := sendWitness_kind hsend1
            change LocalLetter.toEventKind
              ((words A).get ⟨sindex1, sbound1⟩) =
                EventKind.send B scope1 at hk
            change LocalLetter.toEventKind
              ((words A).get ⟨sindex1, sbound1⟩) =
                EventKind.send B scope at hks1
            cases hk.symm.trans hks1
            rfl
          have hscope2 : scope2 = scope := by
            have hk := sendWitness_kind hsend2
            change LocalLetter.toEventKind
              ((words A).get ⟨sindex2, sbound2⟩) =
                EventKind.send B scope2 at hk
            change LocalLetter.toEventKind
              ((words A).get ⟨sindex2, sbound2⟩) =
                EventKind.send B scope at hks2
            cases hk.symm.trans hks2
            rfl
          subst scope1
          subst scope2
          have hsendSelected1 :
              (sendPick B scope
                ((words A).get ⟨sindex1, sbound1⟩)).isSome = true := by
            have hpick := sendWitness_pick hsend1
            change sendPick B scope
              ((words A).get ⟨sindex1, sbound1⟩) = some item1 at hpick
            rw [hpick]
            rfl
          have hsendSelected2 :
              (sendPick B scope
                ((words A).get ⟨sindex2, sbound2⟩)).isSome = true := by
            have hpick := sendWitness_pick hsend2
            change sendPick B scope
              ((words A).get ⟨sindex2, sbound2⟩) = some item2 at hpick
            rw [hpick]
            rfl
          have hrecvSelected1 :
              (receivePick A scope
                ((words B).get ⟨rindex1, rbound1⟩)).isSome = true := by
            have hpick := receiveWitness_pick hrecv1
            change receivePick A scope
              ((words B).get ⟨rindex1, rbound1⟩) = some item1 at hpick
            rw [hpick]
            rfl
          have hrecvSelected2 :
              (receivePick A scope
                ((words B).get ⟨rindex2, rbound2⟩)).isSome = true := by
            have hpick := receiveWitness_pick hrecv2
            change receivePick A scope
              ((words B).get ⟨rindex2, rbound2⟩) = some item2 at hpick
            rw [hpick]
            rfl
          have hsendIndex : sindex1 < sindex2 := by
            dsimp [WordEvent.localIndex] at hlt
            omega
          have hsendRank :
              selectedRank (sendPick B scope) (words A) sindex1 <
                selectedRank (sendPick B scope) (words A) sindex2 :=
            selectedRank_strict (sendPick B scope) (words A) sbound1
              sbound2 hsendIndex hsendSelected1
          have hrecvRank :
              selectedRank (receivePick A scope) (words B) rindex1 <
                selectedRank (receivePick A scope) (words B) rindex2 := by
            rw [← hrank1, ← hrank2]
            exact hsendRank
          have hrecvIndex : rindex1 < rindex2 :=
            (selectedRank_lt_iff_at_selected (receivePick A scope)
              (words B) rbound1 rbound2 hrecvSelected1 hrecvSelected2).1
              hrecvRank
          dsimp [WordEvent.localIndex]
          omega

/-- The session restriction of a scheduled endpoint execution is an MSC. -/
noncomputable def scheduledTimedMSC [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (schedule : List (ScheduleEntry Lifeline))
    (hscheduled : ScheduledReachable (initialConfiguration order source)
      configuration schedule)
    (session : SessionName) :
    TimedWordMSCData
      (projectedSessionTuple session configuration.history) where
  msg := FIFOMessage (projectedSessionTuple session configuration.history)
  msg_send_recv := fifoMessage_send_recv
  msg_letter_match := fifoMessage_letter_match
  msg_lifeline_ne := by
    intro send receive hmessage
    exact hmessage.choose_spec.choose_spec.2.2.1
  msg_send_unique := fifoMessage_send_unique
  msg_recv_unique := fifoMessage_recv_unique
  recv_matched := by
    intro receive hkind
    exact scheduledReachable_recv_matched order source configuration schedule
      hscheduled session hkind
  channel_fifo := by
    intro s1 s2 r1 r2 A B scope hm1 hm2 hs1 hs2 hr1 hr2
      hks1 hks2 hkr1 hkr2 hlt
    exact fifoMessage_channel_fifo hm1 hm2 hs1 hs2 hr1 hr2
      hks1 hks2 hkr1 hkr2 hlt
  time := scheduleTime session schedule
    (projectedSessionTuple session configuration.history)
    (fun A => scheduledReachable_scheduleWord hscheduled rfl A session)
  local_time := by
    intro e f hlocal
    exact scheduleTime_localSucc session schedule
      (projectedSessionTuple session configuration.history)
      (fun A => scheduledReachable_scheduleWord hscheduled rfl A session)
      hlocal
  msg_time := by
    intro send receive hmessage
    exact scheduledReachable_fifoMessage_time order source configuration
      schedule hscheduled session hmessage

theorem scheduledReachable_session_wordIsMSC [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (schedule : List (ScheduleEntry Lifeline))
    (hscheduled : ScheduledReachable (initialConfiguration order source)
      configuration schedule)
    (session : SessionName) :
    WordIsMSC (projectedSessionTuple session configuration.history) :=
  (scheduledTimedMSC order source configuration schedule hscheduled session).wordIsMSC

theorem reachable_session_wordIsMSC [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (hreachable : Reachable (initialConfiguration order source) configuration)
    (session : SessionName) :
    WordIsMSC (projectedSessionTuple session configuration.history) := by
  rcases reachable_scheduled hreachable with ⟨schedule, hscheduled⟩
  exact scheduledReachable_session_wordIsMSC order source configuration
    schedule hscheduled session

/-- The operational root session is already a prefix execution of the actual
    projected source program.  This closes the full trace-plus-MSC obligation
    for unfinished root executions. -/
theorem reachable_root_prefixSemantics [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (hreachable : Reachable (initialConfiguration order source) configuration) :
    DistProg.prefixSemantics WordIsMSC
      (fun A => (Prog.projectMain order A source).eraseForks)
      (projectedSessionTuple .root configuration.history) := by
  constructor
  · intro A
    exact reachable_root_prefixTrace order source configuration hreachable A
  · exact reachable_session_wordIsMSC order source configuration hreachable
      .root

end ZipperGenSpawn
