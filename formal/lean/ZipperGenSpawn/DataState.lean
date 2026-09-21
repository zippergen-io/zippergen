import ZipperGenSpawn.Execution
import ZipperGenSpawn.DataSyntax

/-!
  Participant-local data for spawned executions.

  The MSC development already retains application action names and message
  payloads in `AlphabetOf`.  This module adds the orthogonal state layer needed
  to interpret those observations when persistent participants serve several
  sessions at once.

  A cell is always owned by one participant.  Its allocation session is found
  by walking the canonical session ancestry: a field listed fresh at a child
  fork is allocated by that child; every other field inherits the allocation
  selected by its parent.  Thus a root allocation is participant-shared, while
  a fresh allocation is session-local.  No cross-participant shared memory is
  introduced.

  Stateful letters erase to the existing rich `SpawnLetter` alphabet.  The
  generic MSC/projection theorems therefore remain unchanged, and stateful
  executions inherit their global-prefix and prefix-closed-property results.
-/

namespace ZipperGenSpawn

/-- A contract for one observable event may check a participant-local cell or
    update it.  Reads make action contracts explicit; writes define the state
    visible to later events on the same physical participant. -/
inductive CellOp (Field Value : Type) where
  | expect (field : Field) (value : Value)
  | write (field : Field) (value : Value)
deriving Repr

/-- Participant-owned cells, indexed by their allocation session. -/
abbrev CellStore (Lifeline Field Value : Type) :=
  Lifeline -> SessionName -> Field -> Option Value

/-- The freshness declaration seen by a participant at a child session. -/
abbrev FreshFields (Lifeline Field : Type) :=
  Lifeline -> SessionName -> List Field

/-- Resolve the physical cell denoted by `field` in `session`.

    The definition is structural on the canonical session name.  A field not
    freshened by a child denotes exactly the cell selected by its parent. -/
def cellAllocation [DecidableEq Field]
    (fresh : FreshFields Lifeline Field) (A : Lifeline) :
    SessionName -> Field -> SessionName
  | .root, _ => .root
  | current@(.child parent _ _), field =>
      if field ∈ fresh A current then current
      else cellAllocation fresh A parent field

@[simp] theorem cellAllocation_root [DecidableEq Field]
    (fresh : FreshFields Lifeline Field) (A : Lifeline) (field : Field) :
    cellAllocation fresh A .root field = .root := rfl

@[simp] theorem cellAllocation_child_of_mem [DecidableEq Field]
    (fresh : FreshFields Lifeline Field) (A : Lifeline)
    (parent : SessionName) (site : Occurrence) (visit : Nat) (field : Field)
    (hfield : field ∈ fresh A (.child parent site visit)) :
    cellAllocation fresh A (.child parent site visit) field =
      .child parent site visit := by
  simp [cellAllocation, hfield]

@[simp] theorem cellAllocation_child_of_not_mem [DecidableEq Field]
    (fresh : FreshFields Lifeline Field) (A : Lifeline)
    (parent : SessionName) (site : Occurrence) (visit : Nat) (field : Field)
    (hfield : field ∉ fresh A (.child parent site visit)) :
    cellAllocation fresh A (.child parent site visit) field =
      cellAllocation fresh A parent field := by
  simp [cellAllocation, hfield]

def readCell [DecidableEq Field]
    (fresh : FreshFields Lifeline Field)
    (store : CellStore Lifeline Field Value)
    (A : Lifeline) (session : SessionName) (field : Field) : Option Value :=
  store A (cellAllocation fresh A session field) field

def setCell [DecidableEq Lifeline] [DecidableEq Field]
    (store : CellStore Lifeline Field Value)
    (A : Lifeline) (allocation : SessionName) (field : Field)
    (value : Option Value) : CellStore Lifeline Field Value :=
  fun B current currentField =>
    if B = A ∧ current = allocation ∧ currentField = field then value
    else store B current currentField

@[simp] theorem setCell_here [DecidableEq Lifeline] [DecidableEq Field]
    (store : CellStore Lifeline Field Value)
    (A : Lifeline) (allocation : SessionName) (field : Field)
    (value : Option Value) :
    setCell store A allocation field value A allocation field = value := by
  simp [setCell]

/-- Execute one contract operation in the cell selected by the current
    session's binding. -/
def applyCellOp [DecidableEq Lifeline] [DecidableEq Field] [DecidableEq Value]
    (fresh : FreshFields Lifeline Field)
    (A : Lifeline) (session : SessionName) :
    CellOp Field Value -> CellStore Lifeline Field Value ->
      Option (CellStore Lifeline Field Value)
  | .expect field value, store =>
      if readCell fresh store A session field = some value then some store
      else none
  | .write field value, store =>
      some <| setCell store A (cellAllocation fresh A session field)
        field (some value)

def applyCellOps [DecidableEq Lifeline] [DecidableEq Field] [DecidableEq Value]
    (fresh : FreshFields Lifeline Field)
    (A : Lifeline) (session : SessionName) :
    List (CellOp Field Value) -> CellStore Lifeline Field Value ->
      Option (CellStore Lifeline Field Value)
  | [], store => some store
  | operation :: rest, store =>
      match applyCellOp fresh A session operation store with
      | none => none
      | some next => applyCellOps fresh A session rest next

/-- Allocate one child-local field and initialize it from the cell denoted by
    the same field in the parent.  Copying `none` represents an uninitialized
    fresh cell. -/
def copyFreshField [DecidableEq Lifeline] [DecidableEq Field]
    (fresh : FreshFields Lifeline Field)
    (store : CellStore Lifeline Field Value)
    (A : Lifeline) (parent child : SessionName) (field : Field) :
    CellStore Lifeline Field Value :=
  setCell store A child field (readCell fresh store A parent field)

def copyFreshFields [DecidableEq Lifeline] [DecidableEq Field]
    (fresh : FreshFields Lifeline Field)
    (store : CellStore Lifeline Field Value)
    (A : Lifeline) (parent child : SessionName) (fields : List Field) :
    CellStore Lifeline Field Value :=
  fields.foldl (fun current field =>
    copyFreshField fresh current A parent child field) store

@[simp] theorem readCell_copyFreshField_self
    [DecidableEq Lifeline] [DecidableEq Field]
    (fresh : FreshFields Lifeline Field)
    (store : CellStore Lifeline Field Value)
    (A : Lifeline) (parent : SessionName)
    (site : Occurrence) (visit : Nat) (field : Field)
    (hfield : field ∈ fresh A (.child parent site visit)) :
    readCell fresh
        (copyFreshField fresh store A parent (.child parent site visit) field)
        A (.child parent site visit) field =
      readCell fresh store A parent field := by
  simp [readCell, copyFreshField, hfield, setCell]

@[simp] theorem readCell_copyFreshFields_singleton
    [DecidableEq Lifeline] [DecidableEq Field]
    (fresh : FreshFields Lifeline Field)
    (store : CellStore Lifeline Field Value)
    (A : Lifeline) (parent : SessionName)
    (site : Occurrence) (visit : Nat) (field : Field)
    (hfield : field ∈ fresh A (.child parent site visit)) :
    readCell fresh
        (copyFreshFields fresh store A parent (.child parent site visit) [field])
        A (.child parent site visit) field =
      readCell fresh store A parent field := by
  simpa [copyFreshFields] using
    readCell_copyFreshField_self fresh store A parent site visit field hfield

/-- A data-bearing physical letter.  Ordinary events use the existing rich
    MSC alphabet, retaining action names and payloads.  Forks additionally
    expose the participant-local fields copied at that local fork. -/
inductive DataSpawnLetter
    (Lifeline Condition Action Payload Field : Type) (A : Lifeline) where
  | fork
      (parent child : SessionName) (fresh : List Field)
  | event
      (session : SessionName)
      (letter : AlphabetOf (C := Condition) (F := Action)
        (Payload := Payload) A)

abbrev DataSpawnTuple
    (Lifeline Condition Action Payload Field : Type) :=
  ∀ A : Lifeline,
    List (DataSpawnLetter Lifeline Condition Action Payload Field A)

def DataSpawnLetter.erase
    {Lifeline Condition Action Payload Field : Type} {A : Lifeline} :
    DataSpawnLetter Lifeline Condition Action Payload Field A ->
      SpawnLetter SessionName (L := Lifeline) (C := Condition)
        (F := Action) (Payload := Payload) A
  | .fork parent child _ => .fork parent child
  | .event session letter => .event session letter

def eraseDataTuple
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field) :
    SpawnTuple SessionName Lifeline Condition Action Payload :=
  fun A => (physical A).map DataSpawnLetter.erase

/-- Event contracts attach participant-local reads and writes to the already
    observable action/message alphabet.  Generated control traffic normally
    receives the empty contract. -/
abbrev EventContract
    (Lifeline Condition Action Payload Field Value : Type) :=
  ∀ A : Lifeline,
    AlphabetOf (C := Condition) (F := Action) (Payload := Payload) A ->
      List (CellOp Field Value)

def applyDataLetter [DecidableEq Lifeline] [DecidableEq Field]
    [DecidableEq Value]
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (A : Lifeline) :
    DataSpawnLetter Lifeline Condition Action Payload Field A ->
      CellStore Lifeline Field Value -> Option (CellStore Lifeline Field Value)
  | .fork parent child fields, store =>
      if SessionName.parent child = some parent ∧ freshAt A child = fields then
        some (copyFreshFields freshAt store A parent child fields)
      else
        none
  | .event session letter, store =>
      applyCellOps freshAt A session (contract A letter) store

def replayDataWord [DecidableEq Lifeline] [DecidableEq Field]
    [DecidableEq Value]
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (A : Lifeline) :
    List (DataSpawnLetter Lifeline Condition Action Payload Field A) ->
      CellStore Lifeline Field Value -> Option (CellStore Lifeline Field Value)
  | [], store => some store
  | letter :: rest, store =>
      match applyDataLetter freshAt contract A letter store with
      | none => none
      | some next => replayDataWord freshAt contract A rest next

theorem replayDataWord_append_one
    [DecidableEq Lifeline] [DecidableEq Field] [DecidableEq Value]
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (A : Lifeline)
    (word : List (DataSpawnLetter Lifeline Condition Action Payload Field A))
    (letter : DataSpawnLetter Lifeline Condition Action Payload Field A)
    (initial current next : CellStore Lifeline Field Value)
    (hword : replayDataWord freshAt contract A word initial = some current)
    (hletter : applyDataLetter freshAt contract A letter current = some next) :
    replayDataWord freshAt contract A (word ++ [letter]) initial = some next := by
  induction word generalizing initial current with
  | nil =>
      simp only [replayDataWord] at hword
      have hcurrent : initial = current := Option.some.inj hword
      subst current
      simp only [List.nil_append, replayDataWord]
      rw [hletter]
  | cons head tail ih =>
      simp only [replayDataWord] at hword
      cases hhead : applyDataLetter freshAt contract A head initial with
      | none => simp [hhead] at hword
      | some afterHead =>
          simp only [hhead] at hword
          simp only [List.cons_append, replayDataWord]
          rw [hhead]
          exact ih afterHead current hword hletter

def appendDataLetter [DecidableEq Lifeline]
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field)
    (A : Lifeline)
    (letter : DataSpawnLetter Lifeline Condition Action Payload Field A) :
    DataSpawnTuple Lifeline Condition Action Payload Field :=
  fun B =>
    if h : B = A then
      physical B ++ [h ▸ letter]
    else
      physical B

def setParticipantStore [DecidableEq Lifeline]
    (stores : Lifeline -> CellStore Lifeline Field Value)
    (A : Lifeline) (value : CellStore Lifeline Field Value) :
    Lifeline -> CellStore Lifeline Field Value :=
  fun B => if B = A then value else stores B

@[simp] theorem setParticipantStore_here [DecidableEq Lifeline]
    (stores : Lifeline -> CellStore Lifeline Field Value)
    (A : Lifeline) (value : CellStore Lifeline Field Value) :
    setParticipantStore stores A value A = value := by
  simp [setParticipantStore]

/-- Executable-shape participant-local replay state.  The separate store for
    each scheduler participant is sound because contracts may access only cells
    owned by that participant. -/
structure DataReplayConfiguration
    (Lifeline Condition Action Payload Field Value : Type) where
  physical : DataSpawnTuple Lifeline Condition Action Payload Field
  stores : Lifeline -> CellStore Lifeline Field Value

def initialDataReplayConfiguration
    (initial : CellStore Lifeline Field Value) :
    DataReplayConfiguration Lifeline Condition Action Payload Field Value :=
  { physical := fun _ => []
    stores := fun _ => initial }

inductive DataReplayStep
    [DecidableEq Lifeline] [DecidableEq Field] [DecidableEq Value]
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value) :
    DataReplayConfiguration Lifeline Condition Action Payload Field Value ->
      DataReplayConfiguration Lifeline Condition Action Payload Field Value ->
      Prop where
  | append
      (before : DataReplayConfiguration
        Lifeline Condition Action Payload Field Value)
      (A : Lifeline)
      (letter : DataSpawnLetter Lifeline Condition Action Payload Field A)
      (next : CellStore Lifeline Field Value)
      (henabled :
        applyDataLetter freshAt contract A letter (before.stores A) = some next) :
      DataReplayStep freshAt contract before
        { physical := appendDataLetter before.physical A letter
          stores := setParticipantStore before.stores A next }

inductive DataReplayReachable
    [DecidableEq Lifeline] [DecidableEq Field] [DecidableEq Value]
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value) :
    DataReplayConfiguration Lifeline Condition Action Payload Field Value ->
      Prop where
  | refl : DataReplayReachable freshAt contract initial
      (initialDataReplayConfiguration initial)
  | tail {before after}
      (hbefore : DataReplayReachable freshAt contract initial before)
      (hstep : DataReplayStep freshAt contract before after) :
      DataReplayReachable freshAt contract initial after

/-- Standard invariant rule for the executable data replay.  Properties about
    participant memory therefore do not have to be encoded indirectly as trace
    properties: prove the initial store and one generic step, then obtain the
    invariant for every reachable multi-session configuration. -/
theorem dataReplayReachable_invariant
    [DecidableEq Lifeline] [DecidableEq Field] [DecidableEq Value]
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (invariant : DataReplayConfiguration
      Lifeline Condition Action Payload Field Value -> Prop)
    (hinitial : invariant (initialDataReplayConfiguration initial))
    (hstep : ∀ before after,
      invariant before -> DataReplayStep freshAt contract before after ->
        invariant after)
    (configuration : DataReplayConfiguration
      Lifeline Condition Action Payload Field Value)
    (hreachable : DataReplayReachable freshAt contract initial configuration) :
    invariant configuration := by
  induction hreachable with
  | refl => exact hinitial
  | tail hbefore htransition ih =>
      exact hstep _ _ ih htransition

@[simp] theorem appendDataLetter_here [DecidableEq Lifeline]
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field)
    (A : Lifeline)
    (letter : DataSpawnLetter Lifeline Condition Action Payload Field A) :
    appendDataLetter physical A letter A = physical A ++ [letter] := by
  simp [appendDataLetter]

theorem appendDataLetter_other [DecidableEq Lifeline]
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field)
    (A B : Lifeline)
    (letter : DataSpawnLetter Lifeline Condition Action Payload Field A)
    (hne : B ≠ A) :
    appendDataLetter physical A letter B = physical B := by
  simp [appendDataLetter, hne]

/-- Replay reachability records exactly the store obtained by replaying each
    participant's accumulated physical word from the initial cells. -/
theorem dataReplayReachable_replays
    [DecidableEq Lifeline] [DecidableEq Field] [DecidableEq Value]
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (configuration : DataReplayConfiguration
      Lifeline Condition Action Payload Field Value)
    (hreachable : DataReplayReachable freshAt contract initial configuration) :
    ∀ A,
      replayDataWord freshAt contract A (configuration.physical A) initial =
        some (configuration.stores A) := by
  induction hreachable with
  | refl =>
      intro A
      rfl
  | @tail before after hbefore hstep ih =>
      cases hstep with
      | append A letter next henabled =>
          intro B
          by_cases hBA : B = A
          · subst B
            simp only [setParticipantStore_here, appendDataLetter_here]
            exact replayDataWord_append_one freshAt contract A
              (before.physical A) letter initial (before.stores A) next
              (ih A) henabled
          · change replayDataWord freshAt contract B
              (appendDataLetter before.physical A letter B) initial =
                some (setParticipantStore before.stores A next B)
            rw [appendDataLetter_other _ _ _ _ hBA]
            simpa [setParticipantStore, hBA] using ih B

/-- State validity is participant-local: each physical participant word can be
    replayed from the common initial collection of participant-owned cells.
    Other participants' cells are never accessed by its contracts. -/
def CellConsistent [DecidableEq Lifeline] [DecidableEq Field]
    [DecidableEq Value]
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field) : Prop :=
  ∀ A, ∃ final,
    replayDataWord freshAt contract A (physical A) initial = some final

theorem dataReplayReachable_cellConsistent
    [DecidableEq Lifeline] [DecidableEq Field] [DecidableEq Value]
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (configuration : DataReplayConfiguration
      Lifeline Condition Action Payload Field Value)
    (hreachable : DataReplayReachable freshAt contract initial configuration) :
    CellConsistent freshAt contract initial configuration.physical := by
  intro A
  exact ⟨configuration.stores A,
    dataReplayReachable_replays freshAt contract initial configuration
      hreachable A⟩

/-- A concrete data execution combines two independent obligations:

    * the physical history is admitted by the rich projected spawn semantics;
    * every appended observation is enabled by its participant-local contract.

    Keeping the obligations separate is deliberate.  The first is the generic
    choreography/projection theorem; the second is the application-specific
    interpretation of action labels, payloads, and cells. -/
def ReachableDataExecution
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (program : SessionName -> _root_.Prog Lifeline Condition Action Payload)
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (configuration : DataReplayConfiguration
      Lifeline Condition Action Payload Field Value) : Prop :=
  SpawnPrefixSemantics .root SessionName.parent program
      (eraseDataTuple configuration.physical) ∧
    DataReplayReachable freshAt contract initial configuration

/-- Every observed fork comes from the unique data-bearing spawn declaration
    selected by the child's canonical static site. -/
def DeclaredForks [DecidableEq Lifeline]
    (source : DataProg Lifeline Field Condition Action Payload)
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field) : Prop :=
  ∀ A parent child fields,
    DataSpawnLetter.fork parent child fields ∈ physical A ->
      ∃ site visit fresh body,
        child = SessionName.child parent site visit ∧
        source.spawnBodyAt site = some (fresh, body) ∧
        fields = Prog.freshFor A fresh

/-- The conservative data-bearing extension of spawned prefix semantics. -/
def StatefulSpawnPrefixSemantics [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (program : SessionName -> _root_.Prog Lifeline Condition Action Payload)
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field) : Prop :=
  SpawnPrefixSemantics .root SessionName.parent program
      (eraseDataTuple physical) ∧
    CellConsistent freshAt contract initial physical

/-- A reachable data execution is automatically consistent with replaying its
    participant-local action contracts. -/
theorem reachableDataExecution_stateful
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (program : SessionName -> _root_.Prog Lifeline Condition Action Payload)
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (configuration : DataReplayConfiguration
      Lifeline Condition Action Payload Field Value)
    (hexecution : ReachableDataExecution program freshAt contract initial
      configuration) :
    StatefulSpawnPrefixSemantics program freshAt contract initial
      configuration.physical := by
  exact ⟨hexecution.1,
    dataReplayReachable_cellConsistent freshAt contract initial configuration
      hexecution.2⟩

/-- Erasing the state layer recovers the existing rich spawned semantics. -/
theorem StatefulSpawnPrefixSemantics.control
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    {program : SessionName -> _root_.Prog Lifeline Condition Action Payload}
    {freshAt : FreshFields Lifeline Field}
    {contract : EventContract Lifeline Condition Action Payload Field Value}
    {initial : CellStore Lifeline Field Value}
    {physical : DataSpawnTuple Lifeline Condition Action Payload Field}
    (h : StatefulSpawnPrefixSemantics program freshAt contract initial physical) :
    SpawnPrefixSemantics .root SessionName.parent program
      (eraseDataTuple physical) :=
  h.1

/-- The established global-prefix bridge applies unchanged after adding cells. -/
theorem stateful_spawned_prefix_realization
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (program : SessionName -> _root_.Prog Lifeline Condition Action Payload)
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (hWellTyped : ∀ session, WellTypedProgram (program session))
    (hControl : ∀ session, ControlDistinguishableProgram (program session))
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field)
    (hPhysical :
      StatefulSpawnPrefixSemantics program freshAt contract initial physical) :
    SpawnGlobalPrefixSemantics .root SessionName.parent program
      (eraseDataTuple physical) :=
  spawned_prefix_realization .root SessionName.parent program
    hWellTyped hControl (eraseDataTuple physical) hPhysical.1

/-- Prefix-closed properties proved on the data-rich global choreography hold
    for every active session of a stateful spawned execution. -/
theorem stateful_spawned_prefix_property
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (program : SessionName -> _root_.Prog Lifeline Condition Action Payload)
    (freshAt : FreshFields Lifeline Field)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (property : SessionName -> WordTuple Lifeline Condition Action Payload -> Prop)
    (hWellTyped : ∀ session, WellTypedProgram (program session))
    (hControl : ∀ session, ControlDistinguishableProgram (program session))
    (hPrefixClosed : ∀ session, TuplePrefixClosed (property session))
    (hGlobal : ∀ session chart, ⟦program session⟧ chart -> property session chart)
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field)
    (hPhysical :
      StatefulSpawnPrefixSemantics program freshAt contract initial physical) :
    ∀ session,
      SessionActive .root session (eraseDataTuple physical) ->
        property session
          (eraseTuple (sessionTuple session (eraseDataTuple physical))) := by
  exact spawned_prefix_property .root SessionName.parent program property
    hWellTyped hControl hPrefixClosed hGlobal (eraseDataTuple physical)
      hPhysical.1

/-- High-level semantics for one data-bearing spawn choreography.  It ties the
    total rich session family and participant freshness map back to the single
    annotated source tree and rejects forks at undeclared sites. -/
def DataSourcePrefixSemantics
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (source : DataProg Lifeline Field Condition Action Payload)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field) : Prop :=
  source.RichCoreDefined ∧
    StatefulSpawnPrefixSemantics source.programForSessionTotal
      (source.freshForSession) contract initial physical ∧
    DeclaredForks source physical

/-- Concrete execution predicate tied to one annotated source program.  This
    is the preferred entry point for application certificates: protocol
    execution, enabled action contracts, and declared spawn sites are all
    explicit, while cell consistency is derived rather than assumed. -/
def ReachableDataSourceExecution
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (source : DataProg Lifeline Field Condition Action Payload)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (configuration : DataReplayConfiguration
      Lifeline Condition Action Payload Field Value) : Prop :=
  source.RichCoreDefined ∧
    ReachableDataExecution source.programForSessionTotal source.freshForSession
      contract initial configuration ∧
    DeclaredForks source configuration.physical

theorem reachableDataSourceExecution_semantics
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (source : DataProg Lifeline Field Condition Action Payload)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (configuration : DataReplayConfiguration
      Lifeline Condition Action Payload Field Value)
    (hexecution :
      ReachableDataSourceExecution source contract initial configuration) :
    DataSourcePrefixSemantics source contract initial configuration.physical := by
  exact ⟨hexecution.1,
    reachableDataExecution_stateful source.programForSessionTotal
      source.freshForSession contract initial configuration hexecution.2.1,
    hexecution.2.2⟩

/-- Application proof API: prove a prefix-closed property once on each rich
    session body obtained from the global data choreography, then inherit it
    for every active finite distributed session. -/
theorem data_source_prefix_property
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (source : DataProg Lifeline Field Condition Action Payload)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (property : SessionName -> WordTuple Lifeline Condition Action Payload -> Prop)
    (hWellTyped : ∀ session,
      WellTypedProgram (source.programForSessionTotal session))
    (hControl : ∀ session,
      ControlDistinguishableProgram (source.programForSessionTotal session))
    (hPrefixClosed : ∀ session, TuplePrefixClosed (property session))
    (hGlobal : ∀ session chart,
      ⟦source.programForSessionTotal session⟧ chart -> property session chart)
    (physical : DataSpawnTuple Lifeline Condition Action Payload Field)
    (hPhysical : DataSourcePrefixSemantics source contract initial physical) :
    ∀ session,
      SessionActive .root session (eraseDataTuple physical) ->
        property session
          (eraseTuple (sessionTuple session (eraseDataTuple physical))) := by
  exact stateful_spawned_prefix_property source.programForSessionTotal
    source.freshForSession contract initial property hWellTyped hControl
      hPrefixClosed hGlobal physical hPhysical.2.1

/-- End-to-end application theorem for a concrete replay-reachable execution.
    The application supplies only the ordinary global proof and the two
    execution witnesses in `ReachableDataSourceExecution`; cell consistency is
    obtained from replay automatically. -/
theorem reachable_data_source_prefix_property
    [DecidableEq Lifeline] [Fintype Lifeline]
    [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    [DecidableEq Field] [DecidableEq Value]
    (source : DataProg Lifeline Field Condition Action Payload)
    (contract : EventContract Lifeline Condition Action Payload Field Value)
    (initial : CellStore Lifeline Field Value)
    (property : SessionName -> WordTuple Lifeline Condition Action Payload -> Prop)
    (hWellTyped : ∀ session,
      WellTypedProgram (source.programForSessionTotal session))
    (hControl : ∀ session,
      ControlDistinguishableProgram (source.programForSessionTotal session))
    (hPrefixClosed : ∀ session, TuplePrefixClosed (property session))
    (hGlobal : ∀ session chart,
      ⟦source.programForSessionTotal session⟧ chart -> property session chart)
    (configuration : DataReplayConfiguration
      Lifeline Condition Action Payload Field Value)
    (hexecution :
      ReachableDataSourceExecution source contract initial configuration) :
    ∀ session,
      SessionActive .root session (eraseDataTuple configuration.physical) ->
        property session
          (eraseTuple
            (sessionTuple session (eraseDataTuple configuration.physical))) := by
  exact data_source_prefix_property source contract initial property hWellTyped
    hControl hPrefixClosed hGlobal configuration.physical
      (reachableDataSourceExecution_semantics source contract initial
        configuration hexecution)

end ZipperGenSpawn
