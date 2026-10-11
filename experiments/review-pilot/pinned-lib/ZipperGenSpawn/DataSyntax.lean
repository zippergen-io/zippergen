import ZipperGenSpawn.SourceSyntax
import MSCAgents.Syntax

/-!
  Data-bearing source syntax for spawn certificates.

  The operational spawn proof deliberately uses a control-only syntax.  Proofs
  of application properties need the action names and payload patterns that the
  original MSC calculus already supports.  `DataProg` is the conservative
  annotation of the control syntax with precisely that information.

  Erasing annotations produces the control program consumed by the operational
  correspondence theorem.  Erasing detached child bodies produces the rich
  per-session MSC program consumed by `ZipperGenSpawn.Execution`.  Spawn is not
  a join: in the spawning session it contributes `eps`, while the child body is
  interpreted under its own canonical session name.
-/

namespace ZipperGenSpawn

/-- Global choreography syntax retaining application data. -/
inductive DataProg
    (Lifeline Field Condition Action Payload : Type) where
  | eps
  | msg
      (sender : Lifeline) (sent : Payload)
      (receiver : Lifeline) (received : Payload)
      (hne : sender ≠ receiver)
  | act
      (owner : Lifeline) (output : Payload) (action : Action) (input : Payload)
  | seq
      (first second : DataProg Lifeline Field Condition Action Payload)
  | branch
      (site : Occurrence) (condition : Condition) (owner : Lifeline)
      (thenBranch elseBranch : DataProg Lifeline Field Condition Action Payload)
  | loop
      (site : Occurrence) (condition : Condition) (owner : Lifeline)
      (body exit : DataProg Lifeline Field Condition Action Payload)
  | par {n : Nat}
      (site : Occurrence)
      (branches : Fin n -> DataProg Lifeline Field Condition Action Payload)
  | spawn
      (site : Occurrence) (fresh : List (Lifeline × Field))
      (body : DataProg Lifeline Field Condition Action Payload)

namespace DataProg

infixl:50 " ;;d " => DataProg.seq

/-- Forget application labels and payloads. -/
def controlErase :
    DataProg Lifeline Field Condition Action Payload -> Prog Lifeline Field
  | .eps => .eps
  | .msg sender _ receiver _ hne => .msg sender receiver hne
  | .act owner _ _ _ => .act owner
  | .seq first second => .seq first.controlErase second.controlErase
  | .branch site _ owner thenBranch elseBranch =>
      .branch site owner thenBranch.controlErase elseBranch.controlErase
  | .loop site _ owner body exit =>
      .loop site owner body.controlErase exit.controlErase
  | .par site branches => .par site (fun i => (branches i).controlErase)
  | .spawn site fresh body => .spawn site fresh body.controlErase

/-- Rich MSC behavior of one session.  Detached spawns are empty in their
    parent's behavior and are interpreted separately.  The original rich MSC
    calculus has no intra-session parallel constructor, so conversion is
    partial exactly at `par`; the parallel proof remains available through the
    control semantics. -/
def sessionCore :
    DataProg Lifeline Field Condition Action Payload ->
      Option (_root_.Prog Lifeline Condition Action Payload)
  | .eps => some .eps
  | .msg sender sent receiver received hne =>
      some (.msg sender sent receiver received hne)
  | .act owner output action input =>
      some (.act owner output action input)
  | .seq first second => do
      let firstCore <- first.sessionCore
      let secondCore <- second.sessionCore
      pure (.seq firstCore secondCore)
  | .branch _ condition owner thenBranch elseBranch => do
      let thenCore <- thenBranch.sessionCore
      let elseCore <- elseBranch.sessionCore
      pure (.ite condition owner thenCore elseCore)
  | .loop _ condition owner body exit => do
      let bodyCore <- body.sessionCore
      let exitCore <- exit.sessionCore
      pure (.whileLoop condition owner bodyCore exitCore)
  | .par _ _ => none
  | .spawn _ _ _ => some .eps

/-- All statically declared spawned bodies, including nested declarations. -/
def spawnSpecs :
    DataProg Lifeline Field Condition Action Payload ->
      List (Occurrence × List (Lifeline × Field) ×
        DataProg Lifeline Field Condition Action Payload)
  | .eps | .msg .. | .act .. => []
  | .seq first second => first.spawnSpecs ++ second.spawnSpecs
  | .branch _ _ _ thenBranch elseBranch =>
      thenBranch.spawnSpecs ++ elseBranch.spawnSpecs
  | .loop _ _ _ body exit => body.spawnSpecs ++ exit.spawnSpecs
  | .par _ branches =>
      Prog.finConcatMap (fun i => (branches i).spawnSpecs)
  | .spawn site fresh body => (site, fresh, body) :: body.spawnSpecs

def findSpawnSpec
    (site : Occurrence) :
    List (Occurrence × List (Lifeline × Field) ×
      DataProg Lifeline Field Condition Action Payload) ->
      Option (List (Lifeline × Field) ×
        DataProg Lifeline Field Condition Action Payload)
  | [] => none
  | (current, fresh, body) :: rest =>
      if current = site then some (fresh, body)
      else findSpawnSpec site rest

/-- The data-bearing body declared at a globally unique spawn occurrence. -/
def spawnBodyAt
    (source : DataProg Lifeline Field Condition Action Payload)
    (site : Occurrence) :
    Option (List (Lifeline × Field) ×
      DataProg Lifeline Field Condition Action Payload) :=
  findSpawnSpec site source.spawnSpecs

/-- The application-level program executed by a canonical session. -/
def programForSession
    (source : DataProg Lifeline Field Condition Action Payload) :
    SessionName -> Option (_root_.Prog Lifeline Condition Action Payload)
  | .root => source.sessionCore
  | .child _ site _ => do
      let (_, body) <- source.spawnBodyAt site
      body.sessionCore

/-- Total program family used by the generic spawned-execution interface.
    `RichCoreDefined` and `DeclaredForks` in `DataState` jointly rule out
    reaching the default branch in an active source-derived execution. -/
def programForSessionTotal
    (source : DataProg Lifeline Field Condition Action Payload) :
    SessionName -> _root_.Prog Lifeline Condition Action Payload :=
  fun session => (source.programForSession session).getD .eps

/-- Every session body declared by the source has a rich MSC interpretation.
    The present rich core has no intra-session parallel constructor, so this
    predicate deliberately rejects such bodies instead of letting
    `programForSessionTotal` silently interpret them as `eps`. -/
def RichCoreDefined
    (source : DataProg Lifeline Field Condition Action Payload) : Prop :=
  source.sessionCore.isSome ∧
    ∀ site fresh body,
      source.spawnBodyAt site = some (fresh, body) -> body.sessionCore.isSome

/-- Participant-local fields copied when a child session is created. -/
def freshForSession [DecidableEq Lifeline]
    (source : DataProg Lifeline Field Condition Action Payload)
    (A : Lifeline) : SessionName -> List Field
  | .root => []
  | .child _ site _ =>
      match source.spawnBodyAt site with
      | none => []
      | some (fresh, _) => Prog.freshFor A fresh

end DataProg

end ZipperGenSpawn
