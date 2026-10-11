import ZipperGenSpawn.ParallelPrefixSafety
import ZipperGenSpawn.OperationalNames

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- A physical participant emits local fork events and session-labelled
    endpoint letters. -/
inductive ProjectedSpawnLetter
    (Lifeline : Type u) (Channel : Type v) where
  | fork
      (parent : SessionName) (site : Occurrence) (child : SessionName)
  | event
      (session : SessionName) (letter : LocalLetter Lifeline Channel)

abbrev ProjectedSpawnTuple (Lifeline : Type u) (Channel : Type v) :=
  Lifeline -> List (ProjectedSpawnLetter Lifeline Channel)

def projectedSessionWord (session : SessionName) :
    List (ProjectedSpawnLetter Lifeline Channel) ->
      Word (LocalLetter Lifeline Channel)
  | [] => []
  | .fork _ _ _ :: rest => projectedSessionWord session rest
  | .event current letter :: rest =>
      if current = session then
        letter :: projectedSessionWord session rest
      else
        projectedSessionWord session rest

def projectedSessionTuple (session : SessionName)
    (physical : ProjectedSpawnTuple Lifeline Channel) :
    TraceTuple Lifeline (LocalLetter Lifeline Channel) :=
  fun A => projectedSessionWord session (physical A)

def ProjectedSessionActive (rootSession session : SessionName)
    (physical : ProjectedSpawnTuple Lifeline Channel) : Prop :=
  session = rootSession ∨
    ∃ A parent site,
      ProjectedSpawnLetter.fork parent site session ∈ physical A

/-- Fork labels agree with canonical naming, and every child event follows
    that participant's own local fork. There is no global activation barrier. -/
def ProjectedForkPrecedes
    (physical : ProjectedSpawnTuple Lifeline Channel) : Prop :=
  (∀ A parent site child,
      ProjectedSpawnLetter.fork parent site child ∈ physical A ->
        ∃ visit,
          child = SessionName.child parent site visit) /\
  ∀ A session letter,
      ProjectedSpawnLetter.event session letter ∈ physical A ->
        session = .root ∨
        ∃ parent site visit before middle suffix,
          session = SessionName.child parent site visit /\
          physical A =
            before ++
              ProjectedSpawnLetter.fork parent site session ::
                middle ++
                  ProjectedSpawnLetter.event session letter :: suffix

/-- Every active session restriction is a valid finite prefix of its own
    projected endpoint program. -/
def ProjectedSpawnPrefixSemantics
    (program : SessionName -> ZipperGenParallel.DistProg Lifeline Channel)
    (physical : ProjectedSpawnTuple Lifeline Channel) : Prop :=
  ProjectedForkPrecedes physical /\
  ∀ session,
    ProjectedSessionActive .root session physical ->
      DistProg.prefixSemantics WordIsMSC (program session)
        (projectedSessionTuple session physical)

/-- Pointwise completion witnesses for all currently active sessions. Child
    completion is not folded into parent completion. -/
def ActiveSessionCompletions
    (program : SessionName -> ZipperGenParallel.DistProg Lifeline Channel)
    (physical : ProjectedSpawnTuple Lifeline Channel) : Prop :=
  ∀ session,
    ProjectedSessionActive .root session physical ->
      ∃ complete,
        DistProg.semantics WordIsCompleteMSC (program session) complete /\
        ∀ A,
          ListPrefix (projectedSessionTuple session physical A) (complete A)

/-- The parallel prefix-completion theorem lifts independently to every
    active spawned session, including unfinished parent and child sessions. -/
theorem active_sessions_prefix_complete
    (program : SessionName -> ZipperGenParallel.DistProg Lifeline Channel)
    (hcomplete : ∀ session,
      PrefixCompletable WordIsCompleteMSC WordIsMSC (program session))
    (physical : ProjectedSpawnTuple Lifeline Channel)
    (hphysical : ProjectedSpawnPrefixSemantics program physical) :
    ActiveSessionCompletions program physical := by
  intro session hactive
  exact hcomplete session _ (hphysical.2 session hactive)

/-- A session-indexed family of prefix-closed safety properties proved on
    complete endpoint executions already holds in a physical multi-session
    prefix. -/
theorem active_sessions_prefix_property
    (program : SessionName -> ZipperGenParallel.DistProg Lifeline Channel)
    (property : SessionName ->
      TraceTuple Lifeline (LocalLetter Lifeline Channel) -> Prop)
    (hcomplete : ∀ session,
      PrefixCompletable WordIsCompleteMSC WordIsMSC (program session))
    (hclosed : ∀ session, ParallelTuplePrefixClosed (property session))
    (hfull : ∀ session words,
      DistProg.semantics WordIsCompleteMSC (program session) words ->
        property session words)
    (physical : ProjectedSpawnTuple Lifeline Channel)
    (hphysical : ProjectedSpawnPrefixSemantics program physical) :
    ∀ session,
      ProjectedSessionActive .root session physical ->
        property session (projectedSessionTuple session physical) := by
  intro session hactive
  exact parallel_projected_prefix_property
    (property session) (hcomplete session) (hclosed session)
    (hfull session) _ (hphysical.2 session hactive)

end ZipperGenSpawn
