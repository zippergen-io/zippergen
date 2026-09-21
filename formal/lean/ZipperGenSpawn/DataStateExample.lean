import ZipperGenSpawn.DataState

/-! Small checked examples for the data-bearing spawn interface. -/

namespace ZipperGenSpawn.DataStateExample

inductive Role where
  | writer
deriving DecidableEq

inductive Field where
  | preferences
  | draft
deriving DecidableEq

abbrev ExampleProgram := DataProg Role Field Bool String Nat

def threadBody : ExampleProgram :=
  .act .writer 2 "compose" 1

def source : ExampleProgram :=
  .spawn 7 [(.writer, .draft)] threadBody

example : source.RichCoreDefined := by
  simp [DataProg.RichCoreDefined, source, threadBody, DataProg.sessionCore,
    DataProg.spawnBodyAt, DataProg.spawnSpecs, DataProg.findSpawnSpec]

example : threadBody.sessionCore =
    some (_root_.Prog.act Role.writer 2 "compose" 1) := rfl

example : source.spawnBodyAt 7 =
    some ([ (Role.writer, Field.draft) ], threadBody) := by
  rfl

def child : SessionName := .child .root 7 1
def grandchild : SessionName := .child child 8 1

def fresh : FreshFields Role Field
  | .writer, session => if session = child then [.draft] else []

example : cellAllocation fresh .writer child .draft = child := by
  simp [fresh, child]

example : cellAllocation fresh .writer child .preferences = .root := by
  simp [fresh, child, cellAllocation]

/-- A descendant inherits the nearest ancestor allocation unless it freshens
    the field again. -/
example : cellAllocation fresh .writer grandchild .draft = child := by
  simp [fresh, grandchild, child, cellAllocation]

def initial : CellStore Role Field Nat
  | .writer, .root, .draft => some 42
  | _, _, _ => none

def noContract : EventContract Role Bool String Nat Field Nat :=
  fun _ _ => []

def draftIs42 : GuardEvaluator Role Bool Field Nat :=
  fun _ _ cells => decide (cells .draft = some 42)

def trueChoice : AlphabetOf (C := Bool) (F := String) (Payload := Nat)
    Role.writer :=
  AlphabetOf.mkIfTrue Role.writer true

def falseChoice : AlphabetOf (C := Bool) (F := String) (Payload := Nat)
    Role.writer :=
  AlphabetOf.mkIfFalse Role.writer true

example : recordedDecision trueChoice = some (true, true) := rfl

example :
    applyDataLetter fresh (guardedEventContract draftIs42 noContract)
      .writer (.event .root trueChoice) initial = some initial := by
  rfl

example :
    applyDataLetter fresh (guardedEventContract draftIs42 noContract)
      .writer (.event .root falseChoice) initial = none := by
  rfl

example :
    readCell fresh
        (copyFreshFields fresh initial .writer .root child [.draft])
        .writer child .draft = some 42 := by
  simpa [initial] using
    readCell_copyFreshFields_singleton fresh initial Role.writer
      SessionName.root 7 1 Field.draft (by simp [fresh, child])

end ZipperGenSpawn.DataStateExample
