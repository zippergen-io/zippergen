import ZipperGenSpawn.ForkAgreement

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- The source choreography executed by a dynamic session.  Child identity is
    canonical, while the source body depends only on its globally unique spawn
    site; parent and visit distinguish instances of that body. -/
noncomputable def sessionSource
    (source : Prog Lifeline Field) : SessionName -> Prog Lifeline Field
  | .root => source
  | .child _ site _ => spawnBodyAt source site

/-- Projected endpoint family against which the physical multi-session trace
    is checked. Fork templates are erased because child execution is observed
    in its own session restriction. -/
noncomputable def projectedSessionPrograms [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    SessionName -> ZipperGenParallel.DistProg Lifeline ChannelScope :=
  fun session A =>
    (Prog.projectMain order A (sessionSource source session)).eraseForks

/-- Every participant word of every dynamic child session is a prefix of the
    projection of the unique source spawn body named by that session. -/
theorem reachable_child_prefixTrace [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (hfunctional : SpawnSpecsFunctional source)
    (configuration : Configuration Lifeline Field)
    (hreachable : Reachable (initialConfiguration order source) configuration)
    (A : Lifeline) (parent : SessionName) (site : Occurrence) (visit : Nat) :
    ZipperGenParallel.LocProg.prefixTrace
      ((Prog.projectMain order A (spawnBodyAt source site)).eraseForks)
      (projectedSessionWord (.child parent site visit)
        (configuration.history A)) := by
  let child := SessionName.child parent site visit
  by_cases hcontrol : exists residual,
      configuration.controls A child = some residual
  · rcases hcontrol with ⟨residual, hcontrol⟩
    rcases reachable_control_source_prefix order source configuration hreachable
        hcontrol with ⟨initial, hprefix, horigin⟩
    rcases horigin.child_info (by simp [child]) with
      ⟨originParent, originSite, originVisit, fresh, body,
        hsession, hspec, hinitial⟩
    have hindices : originParent = parent /\ originSite = site /\
        originVisit = visit :=
      SessionName.child_eq_iff.mp (by simpa [child] using hsession.symm)
    rcases hindices with ⟨rfl, rfl, rfl⟩
    rw [spawnBodyAt_spec source originSite fresh body hspec hfunctional]
    simpa [hinitial] using hprefix
  · have hnone : configuration.controls A child = none := by
      cases hvalue : configuration.controls A child with
      | none => simpa using hvalue
      | some residual => exact False.elim (hcontrol ⟨residual, hvalue⟩)
    have hnoevent : forall letter,
        ProjectedSpawnLetter.event child letter ∉ configuration.history A := by
      intro letter hmem
      rcases reachable_eventsHaveControls order source configuration hreachable
          A child letter hmem with ⟨residual, hsome⟩
      rw [hnone] at hsome
      cases hsome
    have hempty :
        projectedSessionWord child (configuration.history A) = [] :=
      projectedSessionWord_eq_nil_of_no_event child _ hnoevent
    rw [hempty]
    rcases eraseForks_trace_exists
        (Prog.projectMain order A (spawnBodyAt source site)) with
      ⟨word, hword⟩
    exact ⟨word, hword, ⟨word, rfl⟩⟩

/-- The endpoint machine realizes the full execution-side semantics of
    detached spawn: local fork causality, a projected trace prefix for every
    active root or child session, and a valid FIFO MSC for each restriction. -/
theorem reachable_projectedSpawnPrefixSemantics_of_functional
    [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (hfunctional : SpawnSpecsFunctional source)
    (configuration : Configuration Lifeline Field)
    (hreachable : Reachable (initialConfiguration order source) configuration) :
    ProjectedSpawnPrefixSemantics
      (projectedSessionPrograms order source) configuration.history := by
  constructor
  · exact reachable_projectedForkPrecedes order source configuration hreachable
  · intro session hactive
    constructor
    · intro A
      cases session with
      | root =>
          exact reachable_root_prefixTrace order source configuration hreachable A
      | child parent site visit =>
          exact reachable_child_prefixTrace order source hfunctional
            configuration hreachable A parent site visit
    · exact reachable_session_wordIsMSC order source configuration hreachable
        session

/-- The standard source well-formedness check (global uniqueness of labelled
    control occurrences) discharges the spawn-body functionality premise. -/
theorem reachable_projectedSpawnPrefixSemantics [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (hwell : source.WellLabelled)
    (configuration : Configuration Lifeline Field)
    (hreachable : Reachable (initialConfiguration order source) configuration) :
    ProjectedSpawnPrefixSemantics
      (projectedSessionPrograms order source) configuration.history :=
  reachable_projectedSpawnPrefixSemantics_of_functional order source
    (wellLabelled_spawnSpecsFunctional source hwell) configuration hreachable

/-- Reachable unfinished executions inherit any session-indexed prefix-closed
    property established for complete projected executions.  This is the
    proof API used by property-specific certificates: the operational and
    projection correspondence is discharged once here. -/
theorem reachable_active_sessions_prefix_property [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (hwell : source.WellLabelled)
    (configuration : Configuration Lifeline Field)
    (hreachable : Reachable (initialConfiguration order source) configuration)
    (property : SessionName ->
      TraceTuple Lifeline (LocalLetter Lifeline ChannelScope) -> Prop)
    (hcomplete : forall session,
      PrefixCompletable WordIsCompleteMSC WordIsMSC
        (projectedSessionPrograms order source session))
    (hclosed : forall session, ParallelTuplePrefixClosed (property session))
    (hfull : forall session words,
      DistProg.semantics WordIsCompleteMSC
        (projectedSessionPrograms order source session) words ->
        property session words) :
    forall session,
      ProjectedSessionActive .root session configuration.history ->
        property session
          (projectedSessionTuple session configuration.history) := by
  exact active_sessions_prefix_property
    (projectedSessionPrograms order source) property hcomplete hclosed hfull
    configuration.history
    (reachable_projectedSpawnPrefixSemantics order source hwell configuration
      hreachable)

end ZipperGenSpawn
