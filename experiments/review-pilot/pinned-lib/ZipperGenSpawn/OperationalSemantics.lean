import ZipperGenSpawn.SpawnPrefixSafety
import ZipperGenSpawn.Projection

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- Symbolic queue contents. Application payloads are abstracted to `data`;
    generated control messages retain their occurrence and branch kind. -/
inductive QueueItem where
  | data
  | control (site : Occurrence) (kind : ChoiceKind)
deriving Repr, DecidableEq

abbrev ControlStore (Lifeline : Type u) (Field : Type v) :=
  Lifeline -> SessionName -> Option (LocProg Lifeline Field)

abbrev Queues (Lifeline : Type u) :=
  SessionName -> ChannelScope -> Lifeline -> Lifeline -> List QueueItem

/-- Observable output of one local symbolic step. -/
inductive Emission (Lifeline : Type u) where
  | silent
  | event (letter : LocalLetter Lifeline ChannelScope)
  | fork (parent : SessionName) (site : Occurrence) (child : SessionName)

/-- A fork creates one additional control at the same persistent participant.
    Other elementary steps create none. -/
abbrev Activation (Lifeline : Type u) (Field : Type v) :=
  Option (SessionName × LocProg Lifeline Field)

structure Configuration (Lifeline : Type u) (Field : Type v) where
  controls : ControlStore Lifeline Field
  counters : VisitCounters Lifeline
  queues : Queues Lifeline
  history : ProjectedSpawnTuple Lifeline ChannelScope

def emptyCounters : VisitCounters Lifeline :=
  fun _ _ _ => 0

def emptyQueues : Queues Lifeline :=
  fun _ _ _ _ => []

def emptyHistory : ProjectedSpawnTuple Lifeline ChannelScope :=
  fun _ => []

def setControl [DecidableEq Lifeline]
    (controls : ControlStore Lifeline Field)
    (A : Lifeline) (session : SessionName)
    (value : Option (LocProg Lifeline Field)) :
    ControlStore Lifeline Field :=
  fun B current =>
    if B = A ∧ current = session then value else controls B current

@[simp] theorem setControl_here [DecidableEq Lifeline]
    (controls : ControlStore Lifeline Field)
    (A : Lifeline) (session : SessionName)
    (value : Option (LocProg Lifeline Field)) :
    setControl controls A session value A session = value := by
  simp [setControl]

def installActivation [DecidableEq Lifeline]
    (controls : ControlStore Lifeline Field) (A : Lifeline) :
    Activation Lifeline Field -> ControlStore Lifeline Field
  | none => controls
  | some (session, program) =>
      setControl controls A session (some program)

def setResidual [DecidableEq Lifeline]
    (controls : ControlStore Lifeline Field)
    (A : Lifeline) (session : SessionName)
    (residual : LocProg Lifeline Field)
    (activation : Activation Lifeline Field) :
    ControlStore Lifeline Field :=
  installActivation
    (setControl controls A session (some residual)) A activation

def setQueue [DecidableEq Lifeline]
    (queues : Queues Lifeline)
    (session : SessionName) (scope : ChannelScope)
    (sender receiver : Lifeline) (items : List QueueItem) :
    Queues Lifeline :=
  fun current currentScope A B =>
    if current = session ∧ currentScope = scope ∧
        A = sender ∧ B = receiver then
      items
    else
      queues current currentScope A B

@[simp] theorem setQueue_here [DecidableEq Lifeline]
    (queues : Queues Lifeline)
    (session : SessionName) (scope : ChannelScope)
    (sender receiver : Lifeline) (items : List QueueItem) :
    setQueue queues session scope sender receiver items
      session scope sender receiver = items := by
  simp [setQueue]

def enqueue [DecidableEq Lifeline]
    (queues : Queues Lifeline)
    (session : SessionName) (scope : ChannelScope)
    (sender receiver : Lifeline) (item : QueueItem) : Queues Lifeline :=
  setQueue queues session scope sender receiver
    (queues session scope sender receiver ++ [item])

def appendEmission [DecidableEq Lifeline]
    (history : ProjectedSpawnTuple Lifeline ChannelScope)
    (A : Lifeline) (session : SessionName) :
    Emission Lifeline -> ProjectedSpawnTuple Lifeline ChannelScope
  | .silent => history
  | .event letter =>
      fun B => if B = A then
        history B ++ [ProjectedSpawnLetter.event session letter]
      else history B
  | .fork parent site child =>
      fun B => if B = A then
        history B ++ [ProjectedSpawnLetter.fork parent site child]
      else history B

/-- Executing one exposed local letter updates only its session-scoped queue
    and emits that same letter. -/
inductive LetterStep [DecidableEq Lifeline]
    (A : Lifeline) (session : SessionName) :
    LocalLetter Lifeline ChannelScope -> Queues Lifeline -> Queues Lifeline ->
      Prop where
  | tau (queues) :
      LetterStep A session .tau queues queues
  | action (queues) :
      LetterStep A session .action queues queues
  | choice (queues) (site : ControlTag) (kind : ChoiceKind) :
      LetterStep A session (.choice site kind) queues queues
  | send (queues) (B : Lifeline) (scope : ChannelScope) (hne : A ≠ B) :
      LetterStep A session (.send B scope) queues
        (enqueue queues session scope A B .data)
  | recv (queues) (B : Lifeline) (scope : ChannelScope) (tail : List QueueItem)
      (hne : B ≠ A)
      (hhead : queues session scope B A = .data :: tail) :
      LetterStep A session (.recv B scope) queues
        (setQueue queues session scope B A tail)
  | ctrlSend (queues) (B : Lifeline) (scope : ChannelScope)
      (site : ControlTag) (kind : ChoiceKind) (hne : A ≠ B) :
      LetterStep A session (.ctrlSend B scope site kind) queues
        (enqueue queues session scope A B (.control site kind))
  | ctrlRecv (queues) (B : Lifeline) (scope : ChannelScope)
      (tail : List QueueItem)
      (site : ControlTag) (kind : ChoiceKind)
      (hne : B ≠ A)
      (hhead : queues session scope B A = .control site kind :: tail) :
      LetterStep A session (.ctrlRecv B scope site kind) queues
        (setQueue queues session scope B A tail)

/-- One enabled position in a local residual. A step produces the replacement
    residual, updated counters and queues, one optional observation, and at
    most one child control. Parallel branches and other sessions remain
    schedulable when one position is blocked. -/
inductive LocalStep [DecidableEq Lifeline]
    (A : Lifeline) (session : SessionName) :
    LocProg Lifeline Field -> VisitCounters Lifeline -> Queues Lifeline ->
    LocProg Lifeline Field -> VisitCounters Lifeline -> Queues Lifeline ->
    Emission Lifeline -> Activation Lifeline Field -> Prop where
  | letter {letter queues queues'}
      (hstep : LetterStep A session letter queues queues') :
      LocalStep A session (.letter letter) counters queues
        .eps counters queues' (.event letter) none
  | seq_done (second) :
      LocalStep A session (.seq .eps second) counters queues
        second counters queues .silent none
  | seq_left {first counters' queues' residual emission activation}
      (hstep : LocalStep A session first counters queues
        residual counters' queues' emission activation) :
      LocalStep A session (.seq first second) counters queues
        (.seq residual second) counters' queues' emission activation
  | branch_true {trueLetter falseLetter thenBranch elseBranch queues'}
      (hstep : LetterStep A session trueLetter queues queues') :
      LocalStep A session
        (.branch trueLetter falseLetter thenBranch elseBranch) counters queues
        thenBranch counters queues' (.event trueLetter) none
  | branch_false {trueLetter falseLetter thenBranch elseBranch queues'}
      (hstep : LetterStep A session falseLetter queues queues') :
      LocalStep A session
        (.branch trueLetter falseLetter thenBranch elseBranch) counters queues
        elseBranch counters queues' (.event falseLetter) none
  | loop_true {trueLetter falseLetter body exit queues'}
      (hstep : LetterStep A session trueLetter queues queues') :
      LocalStep A session (.loop trueLetter falseLetter body exit)
        counters queues
        (.seq body (.loop trueLetter falseLetter body exit))
        counters queues' (.event trueLetter) none
  | loop_false {trueLetter falseLetter body exit queues'}
      (hstep : LetterStep A session falseLetter queues queues') :
      LocalStep A session (.loop trueLetter falseLetter body exit)
        counters queues exit counters queues' (.event falseLetter) none
  | par_done {n : Nat} {branches : Fin n -> LocProg Lifeline Field}
      (hdone : ∀ i, branches i = .eps) :
      LocalStep A session (.par branches) counters queues
        .eps counters queues .silent none
  | par_branch {n : Nat} {branches : Fin n -> LocProg Lifeline Field}
      (i : Fin n) {counters' queues' residual emission activation}
      (hstep : LocalStep A session (branches i) counters queues
        residual counters' queues' emission activation) :
      LocalStep A session (.par branches) counters queues
        (.par (fun j => if j = i then residual else branches j))
        counters' queues' emission activation
  | fork (site : Occurrence) (fresh : List Field)
      (childProgram : LocProg Lifeline Field) :
      LocalStep A session (.fork site fresh childProgram) counters queues
        .eps (incrementCounter counters A session site) queues
        (.fork session site
          (SessionName.child session site
            (nextVisit counters A session site)))
        (some
          (SessionName.child session site
            (nextVisit counters A session site), childProgram))

/-- A network step selects one participant/session control. A child control is
    installed only by the same atomic step that emits its local fork marker. -/
inductive Step [DecidableEq Lifeline] :
    Configuration Lifeline Field -> Configuration Lifeline Field -> Prop where
  | run
      (before : Configuration Lifeline Field)
      (A : Lifeline) (session : SessionName)
      (program residual : LocProg Lifeline Field)
      (counters' : VisitCounters Lifeline) (queues' : Queues Lifeline)
      (emission : Emission Lifeline)
      (activation : Activation Lifeline Field)
      (hcontrol : before.controls A session = some program)
      (hlocal : LocalStep A session program before.counters before.queues
        residual counters' queues' emission activation)
      (hfresh : ∀ child childProgram,
        activation = some (child, childProgram) ->
          before.controls A child = none) :
      Step before
        { controls := setResidual before.controls A session residual activation
          counters := counters'
          queues := queues'
          history := appendEmission before.history A session emission }

def initialConfiguration [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    Configuration Lifeline Field :=
  { controls := fun A session =>
      if session = .root then some (Prog.projectMain order A program) else none
    counters := emptyCounters
    queues := emptyQueues
    history := emptyHistory }

inductive Reachable [DecidableEq Lifeline]
    (initial : Configuration Lifeline Field) :
    Configuration Lifeline Field -> Prop where
  | refl : Reachable initial initial
  | tail {before after}
      (hbefore : Reachable initial before)
      (hstep : Step before after) : Reachable initial after

/-- Every transition extends each physical history component. -/
theorem step_history_extension [DecidableEq Lifeline]
    {before after : Configuration Lifeline Field}
    (hstep : Step before after) :
    ∀ A, ∃ suffix, after.history A = before.history A ++ suffix := by
  intro B
  cases hstep with
  | run A session program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      cases emission with
      | silent => exact ⟨[], by simp [appendEmission]⟩
      | event letter =>
          by_cases hBA : B = A
          · subst B
            exact ⟨[ProjectedSpawnLetter.event session letter],
              by simp [appendEmission]⟩
          · exact ⟨[], by simp [appendEmission, hBA]⟩
      | fork parent site child =>
          by_cases hBA : B = A
          · subst B
            exact ⟨[ProjectedSpawnLetter.fork parent site child],
              by simp [appendEmission]⟩
          · exact ⟨[], by simp [appendEmission, hBA]⟩

/-- Every fork recorded so far has the canonical form
    `child(parent, site, visit)`. -/
def CanonicalForkHistory
    (history : ProjectedSpawnTuple Lifeline ChannelScope) : Prop :=
  ∀ A parent site child,
    ProjectedSpawnLetter.fork parent site child ∈ history A ->
      ∃ visit, child = SessionName.child parent site visit

theorem canonicalForkHistory_empty :
    CanonicalForkHistory (emptyHistory (Lifeline := Lifeline)) := by
  intro A parent site child h
  cases h

theorem localStep_fork_canonical [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {program residual : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline}
    {activation : Activation Lifeline Field}
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission activation) :
    ∀ parent site child,
      emission = .fork parent site child ->
        ∃ visit, child = SessionName.child parent site visit := by
  intro parent site child hemits
  induction hstep with
  | letter hletter => cases hemits
  | seq_done second => cases hemits
  | seq_left hleft ih => exact ih hemits
  | branch_true htrue => cases hemits
  | branch_false hfalse => cases hemits
  | loop_true htrue => cases hemits
  | loop_false hfalse => cases hemits
  | par_done hdone => cases hemits
  | par_branch i hbranch ih => exact ih hemits
  | fork forkSite fresh childProgram =>
      cases hemits
      exact ⟨_, rfl⟩

/-- A local transition creates a new control only when it emits the matching
    local fork marker. -/
theorem localStep_activation_fork [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {program residual childProgram : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline}
    {activation : Activation Lifeline Field}
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission activation)
    {child : SessionName}
    (hactivation : activation = some (child, childProgram)) :
    ∃ site,
      emission = .fork session site child := by
  induction hstep with
  | letter hletter => cases hactivation
  | seq_done second => cases hactivation
  | seq_left hleft ih => exact ih hactivation
  | branch_true htrue => cases hactivation
  | branch_false hfalse => cases hactivation
  | loop_true htrue => cases hactivation
  | loop_false hfalse => cases hactivation
  | par_done hdone => cases hactivation
  | par_branch i hbranch ih => exact ih hactivation
  | fork site fresh child =>
      cases hactivation
      exact ⟨site, rfl⟩

/-- Every installed non-root control has already been activated by a local
    fork at that same physical participant. -/
def ControlsFollowForks
    (configuration : Configuration Lifeline Field) : Prop :=
  ∀ A session program,
    configuration.controls A session = some program ->
      session = .root ∨
      ∃ parent site,
        ProjectedSpawnLetter.fork parent site session ∈
          configuration.history A

theorem initial_controlsFollowForks [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    ControlsFollowForks (initialConfiguration order program) := by
  intro A session localProgram hcontrol
  by_cases hroot : session = .root
  · exact Or.inl hroot
  · simp [initialConfiguration, hroot] at hcontrol

theorem mem_appendEmission_of_mem [DecidableEq Lifeline]
    (history : ProjectedSpawnTuple Lifeline ChannelScope)
    (A B : Lifeline) (session : SessionName)
    (emission : Emission Lifeline)
    (letter : ProjectedSpawnLetter Lifeline ChannelScope)
    (hmem : letter ∈ history B) :
    letter ∈ appendEmission history A session emission B := by
  by_cases hBA : B = A
  · subst B
    cases emission <;> simp [appendEmission, hmem]
  · cases emission <;> simp [appendEmission, hBA, hmem]

theorem step_preserves_controlsFollowForks [DecidableEq Lifeline]
    {before after : Configuration Lifeline Field}
    (hcontrols : ControlsFollowForks before)
    (hstep : Step before after) :
    ControlsFollowForks after := by
  cases hstep with
  | run A parent program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      intro B current currentProgram hafter
      cases activation with
      | none =>
          by_cases htarget : B = A ∧ current = parent
          · rcases htarget with ⟨rfl, rfl⟩
            rcases hcontrols B current program hcontrol with hroot | hfork
            · exact Or.inl hroot
            · rcases hfork with ⟨origin, site, hmem⟩
              exact Or.inr ⟨origin, site,
                mem_appendEmission_of_mem before.history B B current
                  emission _ hmem⟩
          · have hold : before.controls B current = some currentProgram := by
              simpa [setResidual, installActivation, setControl, htarget]
                using hafter
            rcases hcontrols B current currentProgram hold with hroot | hfork
            · exact Or.inl hroot
            · rcases hfork with ⟨origin, site, hmem⟩
              exact Or.inr ⟨origin, site,
                mem_appendEmission_of_mem before.history A B parent
                  emission _ hmem⟩
      | some childAndProgram =>
          rcases childAndProgram with ⟨child, childProgram⟩
          by_cases hchild : B = A ∧ current = child
          · rcases hchild with ⟨rfl, rfl⟩
            rcases localStep_activation_fork hlocal
                (child := current) (childProgram := childProgram) rfl with
              ⟨site, hemits⟩
            subst emission
            exact Or.inr ⟨parent, site, by simp [appendEmission]⟩
          · have hinner :
                setControl before.controls A parent (some residual)
                    B current = some currentProgram := by
              simpa [setResidual, installActivation, setControl, hchild]
                using hafter
            by_cases hparent : B = A ∧ current = parent
            · rcases hparent with ⟨rfl, rfl⟩
              rcases hcontrols B current program hcontrol with hroot | hfork
              · exact Or.inl hroot
              · rcases hfork with ⟨origin, site, hmem⟩
                exact Or.inr ⟨origin, site,
                  mem_appendEmission_of_mem before.history B B current
                    emission _ hmem⟩
            · have hold :
                  before.controls B current = some currentProgram := by
                simpa [setControl, hparent] using hinner
              rcases hcontrols B current currentProgram hold with hroot | hfork
              · exact Or.inl hroot
              · rcases hfork with ⟨origin, site, hmem⟩
                exact Or.inr ⟨origin, site,
                  mem_appendEmission_of_mem before.history A B parent
                    emission _ hmem⟩

theorem reachable_controlsFollowForks [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    ∀ configuration,
      Reachable (initialConfiguration order program) configuration ->
        ControlsFollowForks configuration := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact initial_controlsFollowForks order program
  | tail hbefore hstep ih =>
      exact step_preserves_controlsFollowForks ih hstep

/-- The ordering half of `ProjectedForkPrecedes`: every non-root event is
    physically after that participant's own local fork marker. -/
def ForksPrecedeEvents
    (history : ProjectedSpawnTuple Lifeline ChannelScope) : Prop :=
  ∀ A session letter,
    ProjectedSpawnLetter.event session letter ∈ history A ->
      session = .root ∨
      ∃ parent site visit before middle suffix,
        session = SessionName.child parent site visit ∧
        history A =
          before ++ ProjectedSpawnLetter.fork parent site session ::
            middle ++ ProjectedSpawnLetter.event session letter :: suffix

theorem forksPrecedeEvents_empty :
    ForksPrecedeEvents (emptyHistory (Lifeline := Lifeline)) := by
  intro A session letter hmem
  cases hmem

theorem factorization_survives_append
    {word : List (ProjectedSpawnLetter Lifeline ChannelScope)}
    {parent : SessionName} {site : Occurrence} {session : SessionName}
    {letter : LocalLetter Lifeline ChannelScope}
    (tail : List (ProjectedSpawnLetter Lifeline ChannelScope))
    (hfactor : ∃ before middle suffix,
      word = before ++ ProjectedSpawnLetter.fork parent site session ::
        middle ++ ProjectedSpawnLetter.event session letter :: suffix) :
    ∃ before middle suffix,
      word ++ tail =
        before ++ ProjectedSpawnLetter.fork parent site session ::
          middle ++ ProjectedSpawnLetter.event session letter :: suffix := by
  rcases hfactor with ⟨before, middle, suffix, rfl⟩
  exact ⟨before, middle, suffix ++ tail, by simp [List.append_assoc]⟩

theorem step_preserves_forksPrecedeEvents [DecidableEq Lifeline]
    {before after : Configuration Lifeline Field}
    (hcontrols : ControlsFollowForks before)
    (hcanonical : CanonicalForkHistory before.history)
    (hprecede : ForksPrecedeEvents before.history)
    (hstep : Step before after) :
    ForksPrecedeEvents after.history := by
  intro B current letter hmem
  cases hstep with
  | run A session program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      cases emission with
      | silent =>
          exact hprecede B current letter
            (by simpa [appendEmission] using hmem)
      | fork parent site child =>
          by_cases hBA : B = A
          · subst B
            have hold :
                ProjectedSpawnLetter.event current letter ∈ before.history A := by
              simpa [appendEmission] using hmem
            rcases hprecede A current letter hold with hroot | hfactor
            · exact Or.inl hroot
            · rcases hfactor with
                ⟨origin, forkSite, visit, initial, middle, suffix,
                  hname, hword⟩
              rcases factorization_survives_append
                  [ProjectedSpawnLetter.fork parent site child]
                  ⟨initial, middle, suffix, hword⟩ with
                ⟨initial', middle', suffix', hword'⟩
              exact Or.inr ⟨origin, forkSite, visit,
                initial', middle', suffix', hname,
                by simpa [appendEmission] using hword'⟩
          · have hold := hprecede B current letter
                (by simpa [appendEmission, hBA] using hmem)
            simpa [appendEmission, hBA] using hold
      | event emitted =>
          by_cases hBA : B = A
          · subst B
            have hcases :
                ProjectedSpawnLetter.event current letter ∈ before.history A ∨
                (ProjectedSpawnLetter.event current letter :
                    ProjectedSpawnLetter Lifeline ChannelScope) =
                  ProjectedSpawnLetter.event session emitted := by
              simpa [appendEmission] using hmem
            rcases hcases with hold | hnew
            · rcases hprecede A current letter hold with hroot | hfactor
              · exact Or.inl hroot
              · rcases hfactor with
                  ⟨origin, forkSite, visit, initial, middle, suffix,
                    hname, hword⟩
                rcases factorization_survives_append
                    [ProjectedSpawnLetter.event session emitted]
                    ⟨initial, middle, suffix, hword⟩ with
                  ⟨initial', middle', suffix', hword'⟩
                exact Or.inr ⟨origin, forkSite, visit,
                  initial', middle', suffix', hname,
                  by simpa [appendEmission] using hword'⟩
            · cases hnew
              rcases hcontrols A current program hcontrol with hroot | hfork
              · exact Or.inl hroot
              · rcases hfork with ⟨parent, site, hfork⟩
                rcases List.mem_iff_append.mp hfork with
                  ⟨initial, middle, hword⟩
                rcases hcanonical A parent site current hfork with
                  ⟨visit, hname⟩
                exact Or.inr ⟨parent, site, visit,
                  initial, middle, [], hname, by
                    simp [appendEmission, hword, List.append_assoc]⟩
          · have hold := hprecede B current letter
                (by simpa [appendEmission, hBA] using hmem)
            simpa [appendEmission, hBA] using hold

theorem step_preserves_canonicalForkHistory [DecidableEq Lifeline]
    {before after : Configuration Lifeline Field}
    (hcanonical : CanonicalForkHistory before.history)
    (hstep : Step before after) :
    CanonicalForkHistory after.history := by
  intro B parent site child hmem
  cases hstep with
  | run A session program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      cases emission with
      | silent =>
          exact hcanonical B parent site child (by simpa [appendEmission] using hmem)
      | event letter =>
          by_cases hBA : B = A
          · subst B
            have hold :
                ProjectedSpawnLetter.fork parent site child ∈ before.history A := by
              simpa [appendEmission] using hmem
            exact hcanonical A parent site child hold
          · exact hcanonical B parent site child
              (by simpa [appendEmission, hBA] using hmem)
      | fork forkParent forkSite forkChild =>
          by_cases hBA : B = A
          · subst B
            have hcases :
                ProjectedSpawnLetter.fork parent site child ∈ before.history A ∨
                (ProjectedSpawnLetter.fork parent site child :
                    ProjectedSpawnLetter Lifeline ChannelScope) =
                  (ProjectedSpawnLetter.fork forkParent forkSite forkChild :
                    ProjectedSpawnLetter Lifeline ChannelScope) := by
              simpa [appendEmission] using hmem
            rcases hcases with hold | hnew
            · exact hcanonical A parent site child hold
            · have hnewCanonical :=
                localStep_fork_canonical hlocal forkParent forkSite forkChild rfl
              cases hnew
              exact hnewCanonical
          · exact hcanonical B parent site child
              (by simpa [appendEmission, hBA] using hmem)

theorem reachable_canonicalForkHistory [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    ∀ configuration,
      Reachable (initialConfiguration order program) configuration ->
        CanonicalForkHistory configuration.history := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact canonicalForkHistory_empty
  | tail hbefore hstep ih =>
      exact step_preserves_canonicalForkHistory ih hstep

theorem reachable_forksPrecedeEvents [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    ∀ configuration,
      Reachable (initialConfiguration order program) configuration ->
        ForksPrecedeEvents configuration.history := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact forksPrecedeEvents_empty
  | @tail before after hbefore hstep ih =>
      exact step_preserves_forksPrecedeEvents
        (reachable_controlsFollowForks order program before hbefore)
        (reachable_canonicalForkHistory order program before hbefore)
        ih hstep

/-- The concrete endpoint machine establishes the complete fork-ordering
    component of the multi-session prefix semantics: child names are
    canonical, and each participant's child events occur after its own local
    fork. -/
theorem reachable_projectedForkPrecedes [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (hreachable :
      Reachable (initialConfiguration order program) configuration) :
    ProjectedForkPrecedes configuration.history := by
  exact ⟨reachable_canonicalForkHistory order program configuration hreachable,
    reachable_forksPrecedeEvents order program configuration hreachable⟩

/-- The fork rule leaves the parent residual empty and creates the child
    control in the same local transition; no child completion is joined into
    the parent. -/
theorem localStep_fork_shape [DecidableEq Lifeline]
    (A : Lifeline) (session : SessionName)
    (site : Occurrence) (fresh : List Field)
    (childProgram : LocProg Lifeline Field)
    (counters : VisitCounters Lifeline) (queues : Queues Lifeline) :
    LocalStep A session (.fork site fresh childProgram) counters queues
      .eps (incrementCounter counters A session site) queues
      (.fork session site
        (SessionName.child session site
          (nextVisit counters A session site)))
      (some
        (SessionName.child session site
          (nextVisit counters A session site), childProgram)) := by
  exact LocalStep.fork site fresh childProgram

end ZipperGenSpawn
