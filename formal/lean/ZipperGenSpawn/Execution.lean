/-
  Prefix safety for detached sessions sharing physical participants.

  A physical participant interleaves local fork markers and events from several
  sessions. Session tags are the dynamic channel namespace: restricting the
  physical words to one tag recovers that session's ordinary projected
  execution. No order is inferred between events carrying different tags.
-/

import ZipperGenSpawn.PrefixSafety

namespace ZipperGenSpawn

section Execution

variable {Session L C F Payload : Type}
variable [DecidableEq Session] [DecidableEq L] [Fintype L]
variable [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]

/-- A letter on a physical participant lifeline. Fork markers are local
    activation events; ordinary endpoint letters carry their session tag. -/
inductive SpawnLetter (Session : Type) {L C F Payload : Type} (A : L) where
  | fork (parent child : Session)
  | event (session : Session)
      (letter : AlphabetOf (C := C) (F := F) (Payload := Payload) A)

/-- One physical word per persistent participant. -/
abbrev SpawnTuple (Session L C F Payload : Type) :=
  ∀ A : L, List (SpawnLetter Session (C := C) (F := F) (Payload := Payload) A)

/-- Retain exactly the ordinary endpoint events of session `s`; fork markers
    and events of other sessions disappear. -/
def sessionWord (s : Session) {A : L} :
    List (SpawnLetter Session (C := C) (F := F) (Payload := Payload) A) →
      LocalWord (C := C) (F := F) (Payload := Payload) A
  | [] => []
  | .fork _ _ :: rest => sessionWord s rest
  | .event t letter :: rest =>
      if t = s then letter :: sessionWord s rest else sessionWord s rest

@[simp]
theorem sessionWord_append (s : Session) {A : L}
    (left right :
      List (SpawnLetter Session (C := C) (F := F) (Payload := Payload) A)) :
    sessionWord s (left ++ right) = sessionWord s left ++ sessionWord s right := by
  induction left with
  | nil => rfl
  | cons head tail ih =>
      cases head with
      | fork parent child => simp [sessionWord, ih]
      | event t letter =>
          by_cases h : t = s
          · subst t; simp [sessionWord, ih]
          · simp [sessionWord, h, ih]

/-- The projected tuple seen by one session inside a physical interleaving. -/
def sessionTuple (s : Session) (physical : SpawnTuple Session L C F Payload) :
    WordTuple L C F Payload :=
  fun A => sessionWord s (physical A)

/-- Session `s` exists if it is the root or a local fork marker has created it.
    A fork with no subsequent child event still denotes an active empty prefix. -/
def SessionActive (root s : Session)
    (physical : SpawnTuple Session L C F Payload) : Prop :=
  s = root ∨ ∃ A parent, SpawnLetter.fork parent s ∈ physical A

/-- Fork markers agree with the session-creation tree, and every event of a
    non-root session lies after that session's local fork on the same physical
    participant. There is no global fork barrier and no join. -/
def ForkPrecedes (root : Session) (parentOf : Session → Option Session)
    (physical : SpawnTuple Session L C F Payload) : Prop :=
  (∀ A parent child,
      SpawnLetter.fork parent child ∈ physical A →
        parentOf child = some parent) ∧
  ∀ A s letter,
      SpawnLetter.event s letter ∈ physical A →
        s = root ∨
        ∃ (parent : Session)
          (before middle suffix :
            List (SpawnLetter Session (C := C) (F := F) (Payload := Payload) A)),
          parentOf s = some parent ∧
          physical A =
            before ++ SpawnLetter.fork parent s :: middle ++
              SpawnLetter.event s letter :: suffix

/-- Execution-side semantics of projected spawn. Physical participants may
    interleave sessions arbitrarily. Each active session restriction must be a
    valid finite execution prefix of that session's projected choreography. -/
def SpawnPrefixSemantics
    (root : Session)
    (parentOf : Session → Option Session)
    (program : Session → Prog L C F Payload)
    (physical : SpawnTuple Session L C F Payload) : Prop :=
  ForkPrecedes root parentOf physical ∧
  ∀ s, SessionActive root s physical →
    distPrefixSemantics (projectDist (program s)) (sessionTuple s physical)

/-- Global observation of a spawned physical prefix: every active session
    restriction is a prefix of a complete global MSC of its choreography. -/
def SpawnGlobalPrefixSemantics
    (root : Session)
    (parentOf : Session → Option Session)
    (program : Session → Prog L C F Payload)
    (physical : SpawnTuple Session L C F Payload) : Prop :=
  ForkPrecedes root parentOf physical ∧
  ∀ s, SessionActive root s physical →
    ∃ M, ⟦program s⟧ M ∧
      IsPrefixTuple (eraseTuple (sessionTuple s physical)) M

/-- Soundness of finite projected executions extends pointwise to detached
    sessions sharing physical participants. -/
theorem spawned_prefix_realization
    (root : Session)
    (parentOf : Session → Option Session)
    (program : Session → Prog L C F Payload)
    (hWellTyped : ∀ s, WellTypedProgram (program s))
    (hCtrl : ∀ s, ControlDistinguishableProgram (program s))
    (physical : SpawnTuple Session L C F Payload)
    (hPhysical : SpawnPrefixSemantics root parentOf program physical) :
    SpawnGlobalPrefixSemantics root parentOf program physical := by
  refine ⟨hPhysical.1, ?_⟩
  intro s hActive
  exact projected_prefix_realization (program s) (hWellTyped s) (hCtrl s)
    _ (hPhysical.2 s hActive)

/-- Prefix-closed global properties transfer independently to every active
    session, including unfinished sessions. -/
theorem spawned_prefix_property
    (root : Session)
    (parentOf : Session → Option Session)
    (program : Session → Prog L C F Payload)
    (property : Session → WordTuple L C F Payload → Prop)
    (hWellTyped : ∀ s, WellTypedProgram (program s))
    (hCtrl : ∀ s, ControlDistinguishableProgram (program s))
    (hPrefixClosed : ∀ s, TuplePrefixClosed (property s))
    (hGlobal : ∀ s M, ⟦program s⟧ M → property s M)
    (physical : SpawnTuple Session L C F Payload)
    (hPhysical : SpawnPrefixSemantics root parentOf program physical) :
    ∀ s, SessionActive root s physical →
      property s (eraseTuple (sessionTuple s physical)) := by
  have hGlobalPrefixes := spawned_prefix_realization root parentOf program
    hWellTyped hCtrl physical hPhysical
  intro s hActive
  rcases hGlobalPrefixes.2 s hActive with ⟨M, hM, hPrefix⟩
  exact hPrefixClosed s _ _ hPrefix (hGlobal s M hM)

end Execution

end ZipperGenSpawn
