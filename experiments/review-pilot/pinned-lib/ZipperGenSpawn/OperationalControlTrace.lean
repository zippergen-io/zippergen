import ZipperGenSpawn.OperationalLocalTrace

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- Replacing a residual and optionally installing a child never removes an
    already allocated control cell; it may only replace its program. -/
theorem setResidual_preserves_some [DecidableEq Lifeline]
    (controls : ControlStore Lifeline Field)
    (A : Lifeline) (parent : SessionName)
    (residual : LocProg Lifeline Field)
    (activation : Activation Lifeline Field)
    {B : Lifeline} {session : SessionName}
    {program : LocProg Lifeline Field}
    (hcontrol : controls B session = some program) :
    ∃ current,
      setResidual controls A parent residual activation B session =
        some current := by
  cases activation with
  | none =>
      by_cases htarget : B = A ∧ session = parent
      · exact ⟨residual, by simp [setResidual, installActivation,
          setControl, htarget]⟩
      · exact ⟨program, by simp [setResidual, installActivation,
          setControl, htarget, hcontrol]⟩
  | some childAndProgram =>
      rcases childAndProgram with ⟨child, childProgram⟩
      by_cases hchild : B = A ∧ session = child
      · exact ⟨childProgram, by simp [setResidual, installActivation,
          setControl, hchild]⟩
      · by_cases hparent : B = A ∧ session = parent
        · have hne : parent ≠ child := by
            intro heq
            exact hchild ⟨hparent.1, hparent.2.trans heq⟩
          exact ⟨residual, by simp [setResidual, installActivation,
            setControl, hparent, hne]⟩
        · exact ⟨program, by simp [setResidual, installActivation,
            setControl, hchild, hparent, hcontrol]⟩

theorem setResidual_parent_some [DecidableEq Lifeline]
    (controls : ControlStore Lifeline Field)
    (A : Lifeline) (parent : SessionName)
    (residual : LocProg Lifeline Field)
    (activation : Activation Lifeline Field) :
    ∃ current,
      setResidual controls A parent residual activation A parent =
        some current := by
  cases activation with
  | none =>
      exact ⟨residual, by simp [setResidual, installActivation, setControl]⟩
  | some childAndProgram =>
      rcases childAndProgram with ⟨child, childProgram⟩
      by_cases hchild : child = parent
      · subst child
        exact ⟨childProgram,
          by simp [setResidual, installActivation, setControl]⟩
      · exact ⟨residual, by simp [setResidual, installActivation,
          setControl, Ne.symm hchild]⟩

/-- Every ordinary event in a physical history belongs to a session whose
    local control is still allocated. Controls may reach `eps`, but are never
    deleted; this fact makes freshly installed child controls history-fresh. -/
def EventsHaveControls
    (configuration : Configuration Lifeline Field) : Prop :=
  ∀ A session letter,
    ProjectedSpawnLetter.event session letter ∈ configuration.history A ->
      ∃ program, configuration.controls A session = some program

theorem initial_eventsHaveControls [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    EventsHaveControls (initialConfiguration order program) := by
  intro A session letter hmem
  cases hmem

theorem step_preserves_eventsHaveControls [DecidableEq Lifeline]
    {before after : Configuration Lifeline Field}
    (hinvariant : EventsHaveControls before)
    (hstep : Step before after) :
    EventsHaveControls after := by
  intro B current letter hmem
  cases hstep with
  | run A session program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      cases emission with
      | silent =>
          rcases hinvariant B current letter
              (by simpa [appendEmission] using hmem) with
            ⟨oldProgram, hold⟩
          exact setResidual_preserves_some _ _ _ _ _ hold
      | fork parent site child =>
          have holdMem :
              ProjectedSpawnLetter.event current letter ∈ before.history B := by
            by_cases hBA : B = A
            · subst B
              simpa [appendEmission] using hmem
            · simpa [appendEmission, hBA] using hmem
          rcases hinvariant B current letter holdMem with
            ⟨oldProgram, hold⟩
          exact setResidual_preserves_some _ _ _ _ _ hold
      | event emitted =>
          by_cases hBA : B = A
          · subst B
            have hcases :
                ProjectedSpawnLetter.event current letter ∈ before.history A ∨
                (ProjectedSpawnLetter.event current letter :
                    ProjectedSpawnLetter Lifeline ChannelScope) =
                  ProjectedSpawnLetter.event session emitted := by
              simpa [appendEmission] using hmem
            rcases hcases with holdMem | hnew
            · rcases hinvariant A current letter holdMem with
                ⟨oldProgram, hold⟩
              exact setResidual_preserves_some _ _ _ _ _ hold
            · cases hnew
              exact setResidual_parent_some before.controls A current residual
                activation
          · have holdMem :
                ProjectedSpawnLetter.event current letter ∈ before.history B := by
              simpa [appendEmission, hBA] using hmem
            rcases hinvariant B current letter holdMem with
              ⟨oldProgram, hold⟩
            exact setResidual_preserves_some _ _ _ _ _ hold

theorem reachable_eventsHaveControls [DecidableEq Lifeline]
    (order : List Lifeline) (program : Prog Lifeline Field) :
    ∀ configuration,
      Reachable (initialConfiguration order program) configuration ->
        EventsHaveControls configuration := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact initial_eventsHaveControls order program
  | tail hbefore hstep ih =>
      exact step_preserves_eventsHaveControls ih hstep

theorem localStep_activation_child_ne_parent [DecidableEq Lifeline]
    {A : Lifeline} {session child : SessionName}
    {program residual childProgram : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline}
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission (some (child, childProgram))) :
    child ≠ session := by
  rcases localStep_activation_fork hstep rfl with ⟨site, hemits⟩
  have hchild := localStep_fork_child hstep hemits
  intro heq
  exact SessionName.child_ne_parent session site
    (nextVisit counters A session site) (heq.symm.trans hchild).symm

/-- Each allocated participant/session control retains a derivation from the
    endpoint program with which that control was created, and its emitted word
    is exactly the corresponding restriction of the physical history. Root
    controls are tied to the projection of the source choreography. -/
def ControlsHaveExecution [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field) : Prop :=
  ∀ A session residual,
    configuration.controls A session = some residual ->
      ∃ initial,
        LocalExecution A session initial
          (projectedSessionWord session (configuration.history A)) residual ∧
        (session = .root -> initial = Prog.projectMain order A source)

theorem initial_controlsHaveExecution [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    ControlsHaveExecution order source (initialConfiguration order source) := by
  intro A session residual hcontrol
  by_cases hroot : session = .root
  · subst session
    simp [initialConfiguration] at hcontrol
    subst residual
    exact ⟨Prog.projectMain order A source, LocalExecution.refl _,
      fun _ => rfl⟩
  · simp [initialConfiguration, hroot] at hcontrol

theorem step_preserves_controlsHaveExecution [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    {before after : Configuration Lifeline Field}
    (hevents : EventsHaveControls before)
    (hinvariant : ControlsHaveExecution order source before)
    (hstep : Step before after) :
    ControlsHaveExecution order source after := by
  intro B current currentProgram hafter
  cases hstep with
  | run A parent program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      cases activation with
      | none =>
          by_cases htarget : B = A ∧ current = parent
          · rcases htarget with ⟨rfl, rfl⟩
            simp [setResidual, installActivation, setControl] at hafter
            subst currentProgram
            rcases hinvariant B current program hcontrol with
              ⟨initial, hexecution, hroot⟩
            refine ⟨initial, ?_, hroot⟩
            simpa using LocalExecution.tail hexecution hlocal
          · have hold : before.controls B current = some currentProgram := by
              simpa [setResidual, installActivation, setControl, htarget]
                using hafter
            rcases hinvariant B current currentProgram hold with
              ⟨initial, hexecution, hroot⟩
            refine ⟨initial, ?_, hroot⟩
            by_cases hBA : B = A
            · subst B
              have hne : current ≠ parent := by
                intro heq
                exact htarget ⟨rfl, heq⟩
              cases emission <;>
                simpa [appendEmission, projectedSessionWord_append,
                  projectedSessionWord, hne, Ne.symm hne] using hexecution
            · simpa [projectedSessionWord_appendEmission_other _ hBA]
                using hexecution
      | some childAndProgram =>
          rcases childAndProgram with ⟨child, childProgram⟩
          have hemits := localStep_activation_fork hlocal rfl
          rcases hemits with ⟨site, hemits⟩
          subst emission
          have hchildParent : child ≠ parent :=
            localStep_activation_child_ne_parent hlocal
          have hchildName := localStep_fork_child hlocal rfl
          by_cases hchild : B = A ∧ current = child
          · rcases hchild with ⟨rfl, rfl⟩
            simp [setResidual, installActivation, setControl] at hafter
            subst currentProgram
            have hnone : before.controls B current = none :=
              hfresh current childProgram rfl
            have hnoevent : ∀ letter,
                ProjectedSpawnLetter.event current letter ∉ before.history B := by
              intro letter hmem
              rcases hevents B current letter hmem with ⟨oldProgram, hold⟩
              rw [hnone] at hold
              cases hold
            have hempty :
                projectedSessionWord current (before.history B) = [] :=
              projectedSessionWord_eq_nil_of_no_event current _ hnoevent
            exact ⟨childProgram, by simpa [appendEmission, hempty] using
              (LocalExecution.refl childProgram), by
                intro hroot
                rw [hroot] at hchildName
                cases hchildName⟩
          · by_cases hparent : B = A ∧ current = parent
            · rcases hparent with ⟨rfl, rfl⟩
              simp [setResidual, installActivation, setControl,
                Ne.symm hchildParent] at hafter
              subst currentProgram
              rcases hinvariant B current program hcontrol with
                ⟨initial, hexecution, hroot⟩
              exact ⟨initial, by simpa [appendEmission] using
                (LocalExecution.tail hexecution hlocal), hroot⟩
            · have hold :
                  before.controls B current = some currentProgram := by
                simpa [setResidual, installActivation, setControl,
                  hchild, hparent] using hafter
              rcases hinvariant B current currentProgram hold with
                ⟨initial, hexecution, hroot⟩
              exact ⟨initial, by
                by_cases hBA : B = A
                · subst B
                  simpa [appendEmission, projectedSessionWord] using hexecution
                · simpa [appendEmission, hBA] using hexecution,
                hroot⟩

theorem reachable_controlsHaveExecution [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    ∀ configuration,
      Reachable (initialConfiguration order source) configuration ->
        ControlsHaveExecution order source configuration := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact initial_controlsHaveExecution order source
  | @tail before after hbefore hstep ih =>
      exact step_preserves_controlsHaveExecution order source
        (reachable_eventsHaveControls order source before hbefore) ih hstep

/-- Every allocated control therefore exposes a finite prefix of the endpoint
    program from which that participant/session instance started. -/
theorem reachable_control_prefixTrace [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (hreachable :
      Reachable (initialConfiguration order source) configuration)
    {A : Lifeline} {session : SessionName}
    {residual : LocProg Lifeline Field}
    (hcontrol : configuration.controls A session = some residual) :
    ∃ initial,
      ZipperGenParallel.LocProg.prefixTrace initial.eraseForks
        (projectedSessionWord session (configuration.history A)) ∧
      (session = .root -> initial = Prog.projectMain order A source) := by
  rcases reachable_controlsHaveExecution order source configuration hreachable
      A session residual hcontrol with ⟨initial, hexecution, hroot⟩
  exact ⟨initial, localExecution_prefixTrace hexecution, hroot⟩

theorem reachable_root_control [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    ∀ configuration,
      Reachable (initialConfiguration order source) configuration ->
        ∀ A, ∃ residual,
          configuration.controls A .root = some residual := by
  intro configuration hreachable
  induction hreachable with
  | refl =>
      intro A
      exact ⟨Prog.projectMain order A source, by
        simp [initialConfiguration]⟩
  | @tail before after hbefore hstep ih =>
      intro A
      rcases ih A with ⟨oldResidual, hold⟩
      cases hstep with
      | run actor session program residual counters' queues' emission activation
          hcontrol hlocal hfresh =>
          exact setResidual_preserves_some before.controls actor session residual
            activation hold

/-- In particular, the root-session history at every persistent participant is
    a prefix of that participant's projected source endpoint. -/
theorem reachable_root_prefixTrace [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (hreachable :
      Reachable (initialConfiguration order source) configuration)
    (A : Lifeline) :
    ZipperGenParallel.LocProg.prefixTrace
      (Prog.projectMain order A source).eraseForks
      (projectedSessionWord .root (configuration.history A)) := by
  rcases reachable_root_control order source configuration hreachable A with
    ⟨residual, hcontrol⟩
  rcases reachable_control_prefixTrace order source configuration hreachable
      hcontrol with ⟨initial, hprefix, hroot⟩
  simpa [hroot rfl] using hprefix

end ZipperGenSpawn
