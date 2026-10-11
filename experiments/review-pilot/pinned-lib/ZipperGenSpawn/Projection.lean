import ZipperGenSpawn.LocalProgram

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

namespace Prog

def seqLocList {Lifeline : Type u} {Field : Type v}
    (programs : List (LocProg Lifeline Field)) : LocProg Lifeline Field :=
  programs.foldr LocProg.seq LocProg.eps

def recipients {Lifeline : Type u} {Field : Type v}
    [DecidableEq Lifeline]
    (order : List Lifeline) (owner : Lifeline)
    (left right : Prog Lifeline Field) : List Lifeline :=
  order.eraseDups.filter
    (fun A =>
      decide (A ≠ owner /\ (A ∈ lifelines left ∨ A ∈ lifelines right)))

def controlBroadcast {Lifeline : Type u} {Field : Type v}
    (targets : List Lifeline) (scope : ChannelScope)
    (site : Occurrence) (kind : ChoiceKind) : LocProg Lifeline Field :=
  seqLocList <| targets.map fun B =>
    LocProg.letter (LocalLetter.ctrlSend B scope site kind)

/-- Structural endpoint projection. Parallel selects a static branch scope in
    the same session. Spawn emits one local fork template and projects its
    body at `main`; the runtime later instantiates that template with a fresh
    child session name. -/
def project {Lifeline : Type u} {Field : Type v}
    [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (scope : ChannelScope) :
    Prog Lifeline Field -> LocProg Lifeline Field
  | .eps => .eps
  | .msg sender receiver _ =>
      if A = sender then
        .letter (LocalLetter.send receiver scope)
      else if A = receiver then
        .letter (LocalLetter.recv sender scope)
      else
        .eps
  | .act owner =>
      if A = owner then .letter LocalLetter.action else .eps
  | .seq first second =>
      .seq (project order A scope first) (project order A scope second)
  | .branch site owner thenBranch elseBranch =>
      let targets := recipients order owner thenBranch elseBranch
      if A = owner then
        .branch
          (LocalLetter.choice site ChoiceKind.ifTrue)
          (LocalLetter.choice site ChoiceKind.ifFalse)
          (.seq
            (controlBroadcast targets scope site ChoiceKind.ifTrue)
            (project order A scope thenBranch))
          (.seq
            (controlBroadcast targets scope site ChoiceKind.ifFalse)
            (project order A scope elseBranch))
      else if A ∈ targets then
        .branch
          (LocalLetter.ctrlRecv owner scope site ChoiceKind.ifTrue)
          (LocalLetter.ctrlRecv owner scope site ChoiceKind.ifFalse)
          (project order A scope thenBranch)
          (project order A scope elseBranch)
      else
        .eps
  | .loop site owner body exit =>
      let targets := recipients order owner body exit
      if A = owner then
        .loop
          (LocalLetter.choice site ChoiceKind.whileTrue)
          (LocalLetter.choice site ChoiceKind.whileFalse)
          (.seq
            (controlBroadcast targets scope site ChoiceKind.whileTrue)
            (project order A scope body))
          (.seq
            (controlBroadcast targets scope site ChoiceKind.whileFalse)
            (project order A scope exit))
      else if A ∈ targets then
        .loop
          (LocalLetter.ctrlRecv owner scope site ChoiceKind.whileTrue)
          (LocalLetter.ctrlRecv owner scope site ChoiceKind.whileFalse)
          (project order A scope body)
          (project order A scope exit)
      else
        .eps
  | .par site branches =>
      .par fun i =>
        project order A (.branch site (i.val + 1)) (branches i)
  | .spawn site fresh body =>
      if A ∈ lifelines body then
        .fork site (freshFor A fresh)
          (project order A .main body)
      else
        .eps

def projectMain {Lifeline : Type u} {Field : Type v}
    [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (P : Prog Lifeline Field) :
    LocProg Lifeline Field :=
  project order A .main P

def projectDistMain {Lifeline : Type u} {Field : Type v}
    [DecidableEq Lifeline]
    (order : List Lifeline) (P : Prog Lifeline Field) :
    DistProg Lifeline Field :=
  fun A => projectMain order A P

@[simp] theorem project_spawn_of_mem
    {Lifeline : Type u} {Field : Type v} [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (scope : ChannelScope)
    (site : Occurrence) (fresh : List (Lifeline × Field))
    (body : Prog Lifeline Field) (hA : A ∈ lifelines body) :
    project order A scope (.spawn site fresh body) =
      .fork site (freshFor A fresh) (project order A .main body) := by
  simp [project, hA]

@[simp] theorem project_spawn_of_not_mem
    {Lifeline : Type u} {Field : Type v} [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (scope : ChannelScope)
    (site : Occurrence) (fresh : List (Lifeline × Field))
    (body : Prog Lifeline Field) (hA : A ∉ lifelines body) :
    project order A scope (.spawn site fresh body) = .eps := by
  simp [project, hA]

@[simp] theorem project_parallel_scope
    {Lifeline : Type u} {Field : Type v} [DecidableEq Lifeline]
    {n : Nat} (order : List Lifeline) (A : Lifeline)
    (scope : ChannelScope) (site : Occurrence)
    (branches : Fin n -> Prog Lifeline Field) :
    project order A scope (.par site branches) =
      .par (fun i =>
        project order A (.branch site (i.val + 1)) (branches i)) := rfl

end Prog

end ZipperGenSpawn
