import ZipperGenParallel.LocalProgram

namespace ZipperGenParallel

universe u v

namespace Source

/--
Occurrence/path-based channel names for projected parallel programs.

The root channel is `main`.  Entering branch `i` of a parallel occurrence
extends the current path by the occurrence label and the one-based branch index.
-/
structure ChannelName where
  path : List (Nat × Nat)
deriving Repr, DecidableEq

namespace ChannelName

def main : ChannelName :=
  { path := [] }

def extend (ch : ChannelName) (label idx : Nat) : ChannelName :=
  { path := ch.path ++ [(label, idx)] }

def branch {n : Nat} (ch : ChannelName) (label : Nat) (i : Fin n) :
    ChannelName :=
  ch.extend label (i.val + 1)

def StrictlyExtends (base child : ChannelName) : Prop :=
  ∃ suffix : List (Nat × Nat),
    suffix ≠ [] ∧ child.path = base.path ++ suffix

def Extends (base child : ChannelName) : Prop :=
  ∃ suffix : List (Nat × Nat), child.path = base.path ++ suffix

theorem extends_refl (ch : ChannelName) :
    Extends ch ch := by
  exact ⟨[], by simp⟩

theorem strictlyExtends_extends {base child : ChannelName}
    (h : StrictlyExtends base child) :
    Extends base child := by
  rcases h with ⟨suffix, _hne, hpath⟩
  exact ⟨suffix, hpath⟩

theorem extends_trans {base mid child : ChannelName}
    (h₁ : Extends base mid) (h₂ : Extends mid child) :
    Extends base child := by
  rcases h₁ with ⟨suffix₁, h₁path⟩
  rcases h₂ with ⟨suffix₂, h₂path⟩
  refine ⟨suffix₁ ++ suffix₂, ?_⟩
  rw [h₂path, h₁path, List.append_assoc]

theorem branch_strictlyExtends {n : Nat} (ch : ChannelName) (label : Nat)
    (i : Fin n) :
    StrictlyExtends ch (branch ch label i) := by
  refine ⟨[(label, i.val + 1)], ?_, rfl⟩
  intro h
  cases h

theorem branch_extends {n : Nat} (ch : ChannelName) (label : Nat)
    (i : Fin n) :
    Extends ch (branch ch label i) :=
  strictlyExtends_extends (branch_strictlyExtends ch label i)

theorem branch_injective {n : Nat} (ch : ChannelName) (label : Nat) :
    Function.Injective (fun i : Fin n => branch ch label i) := by
  intro i j h
  apply Fin.ext
  have hpath := congrArg ChannelName.path h
  dsimp [branch, extend] at hpath
  have htail : [(label, i.val + 1)] = [(label, j.val + 1)] :=
    List.append_cancel_left hpath
  injection htail with hhead
  exact Nat.succ.inj
    (by simpa [Nat.succ_eq_add_one] using congrArg Prod.snd hhead)

theorem branch_ne_of_ne {n : Nat} {ch : ChannelName} {label : Nat}
    {i j : Fin n}
    (hij : i ≠ j) :
    branch ch label i ≠ branch ch label j := by
  intro h
  exact hij (branch_injective ch label h)

theorem branch_extensions_disjoint {n : Nat} {ch child : ChannelName}
    {label : Nat} {i j : Fin n} (hij : i ≠ j)
    (hi : Extends (branch ch label i) child)
    (hj : Extends (branch ch label j) child) :
    False := by
  rcases hi with ⟨suffixI, hi⟩
  rcases hj with ⟨suffixJ, hj⟩
  have heq :
      (branch ch label i).path ++ suffixI =
        (branch ch label j).path ++ suffixJ :=
    hi.symm.trans hj
  have hbase :
      ch.path ++ ([(label, i.val + 1)] ++ suffixI) =
        ch.path ++ ([(label, j.val + 1)] ++ suffixJ) := by
    simpa [branch, extend, List.append_assoc] using heq
  have htail :
      [(label, i.val + 1)] ++ suffixI =
        [(label, j.val + 1)] ++ suffixJ :=
    List.append_cancel_left hbase
  injection htail with hhead _htail
  have hval : i.val = j.val :=
    Nat.succ.inj
      (by simpa [Nat.succ_eq_add_one] using congrArg Prod.snd hhead)
  exact hij (Fin.ext hval)

theorem branch_eq_of_common_extension {n : Nat} {ch child : ChannelName}
    {label : Nat} {i j : Fin n}
    (hi : Extends (branch ch label i) child)
    (hj : Extends (branch ch label j) child) :
    i = j := by
  by_cases hij : i = j
  · exact hij
  · exact False.elim (branch_extensions_disjoint hij hi hj)

theorem branch_occurrence_ne {n m : Nat} {ch : ChannelName}
    {label₁ label₂ : Nat} {i : Fin n} {j : Fin m}
    (hlabel : label₁ ≠ label₂) :
    branch ch label₁ i ≠ branch ch label₂ j := by
  intro h
  have hpath := congrArg ChannelName.path h
  dsimp [branch, extend] at hpath
  have htail :
      [(label₁, i.val + 1)] = [(label₂, j.val + 1)] :=
    List.append_cancel_left hpath
  injection htail with hhead
  exact hlabel (congrArg Prod.fst hhead)

end ChannelName

/--
Source-level global programs for the parallel extension.

The current local alphabet abstracts away payloads, function names, and guard
identities.  Accordingly, this source syntax records only the lifelines and
the control/message shape needed by projection.
-/
inductive Prog (Lifeline : Type u) : Type u where
  | eps : Prog Lifeline
  | msg (sender receiver : Lifeline) (h : sender ≠ receiver) : Prog Lifeline
  | act (owner : Lifeline) : Prog Lifeline
  | seq (first second : Prog Lifeline) : Prog Lifeline
  | branch
      (tag : ControlTag)
      (owner : Lifeline)
      (thenBranch elseBranch : Prog Lifeline) :
      Prog Lifeline
  | loop
      (tag : ControlTag)
      (owner : Lifeline)
      (body exit : Prog Lifeline) :
      Prog Lifeline
  | par {n : Nat}
      (label : Nat) (branches : Fin n → Prog Lifeline) :
      Prog Lifeline

namespace Prog

infixl:50 " ;;ₛ " => Prog.seq

def concatLists {α : Type u} (lists : List (List α)) : List α :=
  lists.foldr (· ++ ·) []

@[simp]
theorem concatLists_nil {α : Type u} :
    concatLists ([] : List (List α)) = [] :=
  rfl

@[simp]
theorem concatLists_cons {α : Type u} (xs : List α) (xss : List (List α)) :
    concatLists (xs :: xss) = xs ++ concatLists xss :=
  rfl

def finConcatMap {n : Nat} {α : Type u} (f : Fin n → List α) : List α :=
  concatLists (List.ofFn f)

theorem mem_concatLists_of_mem {α : Type u} {x : α}
    {lists : List (List α)} :
    ∀ {xs}, xs ∈ lists → x ∈ xs → x ∈ concatLists lists := by
  induction lists with
  | nil =>
      intro xs hmem _hx
      cases hmem
  | cons ys yss ih =>
      intro xs hmem hx
      cases hmem with
      | head =>
          simp [concatLists, hx]
      | tail _ hmem =>
          simp [concatLists]
          exact Or.inr (ih hmem hx)

theorem mem_finConcatMap_of_mem {n : Nat} {α : Type u}
    (f : Fin n → List α) (i : Fin n) {x : α}
    (hx : x ∈ f i) :
    x ∈ finConcatMap f := by
  exact mem_concatLists_of_mem (by simp) hx

theorem not_mem_finConcatMap {n : Nat} {α : Type u}
    {f : Fin n → List α} {x : α}
    (hnot : x ∉ finConcatMap f) (i : Fin n) :
    x ∉ f i := by
  intro hx
  exact hnot (mem_finConcatMap_of_mem f i hx)

/-- Lifelines mentioned syntactically by a source program. -/
def lifelines {Lifeline : Type u} : Prog Lifeline → List Lifeline
  | eps => []
  | msg sender receiver _ => [sender, receiver]
  | act owner => [owner]
  | seq first second => lifelines first ++ lifelines second
  | branch _ owner thenBranch elseBranch =>
      owner :: (lifelines thenBranch ++ lifelines elseBranch)
  | loop _ owner body exit =>
      owner :: (lifelines body ++ lifelines exit)
  | par _ branches =>
      finConcatMap (fun i => lifelines (branches i))

def mentions {Lifeline : Type u} [DecidableEq Lifeline]
    (P : Prog Lifeline) (A : Lifeline) : Bool :=
  decide (A ∈ lifelines P)

/-- Source programs whose projection contains no local `par`. -/
def NoPar {Lifeline : Type u} : Prog Lifeline → Prop
  | eps => True
  | msg .. => True
  | act _ => True
  | seq first second =>
      NoPar first ∧ NoPar second
  | branch _ _ thenBranch elseBranch =>
      NoPar thenBranch ∧ NoPar elseBranch
  | loop _ _ body exit =>
      NoPar body ∧ NoPar exit
  | par .. => False

/--
Recipients for a control construct, listed by restricting the supplied global
lifeline order.

The order parameter is the self-contained replacement for a global finite
lifeline enumeration.  Duplicate entries in the order are ignored, so each
recipient gets at most one control message for the construct.
-/
def recipients {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (owner : Lifeline)
    (left right : Prog Lifeline) : List Lifeline :=
  order.eraseDups.filter
    (fun A =>
      decide (A ≠ owner ∧ (A ∈ lifelines left ∨ A ∈ lifelines right)))

/-- Channel names generated by parallel constructs in a program. -/
def generatedChannels {Lifeline : Type u}
    (ch : ChannelName) : Prog Lifeline → List ChannelName
  | eps => []
  | msg .. => []
  | act _ => []
  | seq first second =>
      generatedChannels ch first ++ generatedChannels ch second
  | branch _ _ thenBranch elseBranch =>
      generatedChannels ch thenBranch ++ generatedChannels ch elseBranch
  | loop _ _ body exit =>
      generatedChannels ch body ++ generatedChannels ch exit
  | par label branches =>
      finConcatMap
        (fun i =>
          let chi := ChannelName.branch ch label i
          chi :: generatedChannels chi (branches i))

def seqList {Lifeline : Type u}
    (programs : List (LocProg Lifeline ChannelName)) :
    LocProg Lifeline ChannelName :=
  programs.foldr LocProg.seq LocProg.eps

@[simp]
theorem seqList_nil {Lifeline : Type u} :
    seqList ([] : List (LocProg Lifeline ChannelName)) = LocProg.eps :=
  rfl

@[simp]
theorem seqList_cons {Lifeline : Type u}
    (S : LocProg Lifeline ChannelName)
    (Ss : List (LocProg Lifeline ChannelName)) :
    seqList (S :: Ss) = LocProg.seq S (seqList Ss) :=
  rfl

def controlBroadcast {Lifeline : Type u}
    (targets : List Lifeline) (ch : ChannelName)
    (tag : ControlTag) (kind : ChoiceKind) :
    LocProg Lifeline ChannelName :=
  seqList
    (targets.map
      (fun B => LocProg.letter (LocalLetter.ctrlSend B ch tag kind)))

theorem seqList_trace_letters {Lifeline : Type u}
    (letters : List (LocalLetter Lifeline ChannelName)) :
    LocProg.trace
      (seqList (letters.map LocProg.letter))
      letters := by
  induction letters with
  | nil =>
      exact LocProg.Trace.eps
  | cons letter letters ih =>
      simpa using
        (LocProg.Trace.seq
          (LocProg.Trace.letter letter)
          ih)

theorem controlBroadcast_trace {Lifeline : Type u}
    (targets : List Lifeline) (ch : ChannelName)
    (tag : ControlTag) (kind : ChoiceKind) :
    LocProg.trace
      (controlBroadcast targets ch tag kind)
      (targets.map (fun B => LocalLetter.ctrlSend B ch tag kind)) := by
  simpa [controlBroadcast, List.map_map, Function.comp_def] using
    (seqList_trace_letters
      (Lifeline := Lifeline)
      (letters :=
        targets.map (fun B => LocalLetter.ctrlSend B ch tag kind)))

/--
Projection of a source program to the local program of one lifeline.

The `order` argument is the global lifeline order used to enumerate
control-broadcast recipients.  The current channel is kept unchanged
through sequential, branch, and loop constructs, and is extended exactly at
`par` branches.
-/
def project {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (ch : ChannelName) :
    Prog Lifeline → LocProg Lifeline ChannelName
  | eps => LocProg.eps
  | msg sender receiver _ =>
      if A = sender then
        LocProg.letter (LocalLetter.send receiver ch)
      else if A = receiver then
        LocProg.letter (LocalLetter.recv sender ch)
      else
        LocProg.eps
  | act owner =>
      if A = owner then
        LocProg.letter LocalLetter.action
      else
        LocProg.eps
  | seq first second =>
      LocProg.seq (project order A ch first) (project order A ch second)
  | branch tag owner thenBranch elseBranch =>
      let targets := recipients order owner thenBranch elseBranch
      if A = owner then
        LocProg.branch
          (LocalLetter.choice tag ChoiceKind.ifTrue)
          (LocalLetter.choice tag ChoiceKind.ifFalse)
          (LocProg.seq
            (controlBroadcast targets ch tag ChoiceKind.ifTrue)
            (project order A ch thenBranch))
          (LocProg.seq
            (controlBroadcast targets ch tag ChoiceKind.ifFalse)
            (project order A ch elseBranch))
      else if A ∈ targets then
        LocProg.branch
          (LocalLetter.ctrlRecv owner ch tag ChoiceKind.ifTrue)
          (LocalLetter.ctrlRecv owner ch tag ChoiceKind.ifFalse)
          (project order A ch thenBranch)
          (project order A ch elseBranch)
      else
        LocProg.eps
  | loop tag owner body exit =>
      let targets := recipients order owner body exit
      if A = owner then
        LocProg.loop
          (LocalLetter.choice tag ChoiceKind.whileTrue)
          (LocalLetter.choice tag ChoiceKind.whileFalse)
          (LocProg.seq
            (controlBroadcast targets ch tag ChoiceKind.whileTrue)
            (project order A ch body))
          (LocProg.seq
            (controlBroadcast targets ch tag ChoiceKind.whileFalse)
            (project order A ch exit))
      else if A ∈ targets then
        LocProg.loop
          (LocalLetter.ctrlRecv owner ch tag ChoiceKind.whileTrue)
          (LocalLetter.ctrlRecv owner ch tag ChoiceKind.whileFalse)
          (project order A ch body)
          (project order A ch exit)
      else
        LocProg.eps
  | par label branches =>
      LocProg.par
        (fun i =>
          project order A (ChannelName.branch ch label i) (branches i))

def projectDist {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (ch : ChannelName) (P : Prog Lifeline) :
    DistProg Lifeline ChannelName :=
  fun A => project order A ch P

def projectMain {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (P : Prog Lifeline) :
    LocProg Lifeline ChannelName :=
  project order A ChannelName.main P

def projectDistMain {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (P : Prog Lifeline) :
    DistProg Lifeline ChannelName :=
  projectDist order ChannelName.main P

def projectedSemantics {Lifeline : Type u} [DecidableEq Lifeline]
    (IsCompleteMSC :
      (Lifeline → Word (LocalLetter Lifeline ChannelName)) → Prop)
    (order : List Lifeline) (ch : ChannelName) (P : Prog Lifeline)
    (words : Lifeline → Word (LocalLetter Lifeline ChannelName)) :
    Prop :=
  DistProg.semantics IsCompleteMSC (projectDist order ch P) words

def projectedPrefixSemantics {Lifeline : Type u} [DecidableEq Lifeline]
    (IsMSC :
      (Lifeline → Word (LocalLetter Lifeline ChannelName)) → Prop)
    (order : List Lifeline) (ch : ChannelName) (P : Prog Lifeline)
    (words : Lifeline → Word (LocalLetter Lifeline ChannelName)) :
    Prop :=
  DistProg.prefixSemantics IsMSC (projectDist order ch P) words

def letterChannelExtends {Lifeline : Type u}
    (base : ChannelName) :
    LocalLetter Lifeline ChannelName → Prop
  | LocalLetter.send _ ch => ChannelName.Extends base ch
  | LocalLetter.recv _ ch => ChannelName.Extends base ch
  | LocalLetter.ctrlSend _ ch _ _ => ChannelName.Extends base ch
  | LocalLetter.ctrlRecv _ ch _ _ => ChannelName.Extends base ch
  | _ => True

def programChannelsExtend {Lifeline : Type u}
    (base : ChannelName) :
    LocProg Lifeline ChannelName → Prop
  | LocProg.eps => True
  | LocProg.letter letter => letterChannelExtends base letter
  | LocProg.seq first second =>
      programChannelsExtend base first ∧
        programChannelsExtend base second
  | LocProg.branch trueLetter falseLetter thenBranch elseBranch =>
      letterChannelExtends base trueLetter ∧
        letterChannelExtends base falseLetter ∧
        programChannelsExtend base thenBranch ∧
        programChannelsExtend base elseBranch
  | LocProg.loop trueLetter falseLetter body exit =>
      letterChannelExtends base trueLetter ∧
        letterChannelExtends base falseLetter ∧
        programChannelsExtend base body ∧
        programChannelsExtend base exit
  | LocProg.par branches =>
      ∀ i, programChannelsExtend base (branches i)

theorem programChannelsExtend_mono
    {Lifeline : Type u} {base mid : ChannelName}
    (hext : ChannelName.Extends base mid) :
    ∀ {program : LocProg Lifeline ChannelName},
      programChannelsExtend mid program →
      programChannelsExtend base program := by
  intro program
  induction program with
  | eps =>
      intro _h
      trivial
  | letter letter =>
      intro h
      cases letter with
      | tau =>
          trivial
      | action =>
          trivial
      | choice _ _ =>
          trivial
      | send _ ch =>
          exact ChannelName.extends_trans hext h
      | recv _ ch =>
          exact ChannelName.extends_trans hext h
      | ctrlSend _ ch _ _ =>
          exact ChannelName.extends_trans hext h
      | ctrlRecv _ ch _ _ =>
          exact ChannelName.extends_trans hext h
  | seq first second ihFirst ihSecond =>
      intro h
      exact ⟨ihFirst h.1, ihSecond h.2⟩
  | branch trueLetter falseLetter thenBranch elseBranch ihThen ihElse =>
      intro h
      rcases h with ⟨htrue, hfalse, hthen, helse⟩
      refine ⟨?_, ?_, ihThen hthen, ihElse helse⟩
      · cases trueLetter with
        | tau => trivial
        | action => trivial
        | choice _ _ => trivial
        | send _ ch => exact ChannelName.extends_trans hext htrue
        | recv _ ch => exact ChannelName.extends_trans hext htrue
        | ctrlSend _ ch _ _ => exact ChannelName.extends_trans hext htrue
        | ctrlRecv _ ch _ _ => exact ChannelName.extends_trans hext htrue
      · cases falseLetter with
        | tau => trivial
        | action => trivial
        | choice _ _ => trivial
        | send _ ch => exact ChannelName.extends_trans hext hfalse
        | recv _ ch => exact ChannelName.extends_trans hext hfalse
        | ctrlSend _ ch _ _ => exact ChannelName.extends_trans hext hfalse
        | ctrlRecv _ ch _ _ => exact ChannelName.extends_trans hext hfalse
  | loop trueLetter falseLetter body exit ihBody ihExit =>
      intro h
      rcases h with ⟨htrue, hfalse, hbody, hexit⟩
      refine ⟨?_, ?_, ihBody hbody, ihExit hexit⟩
      · cases trueLetter with
        | tau => trivial
        | action => trivial
        | choice _ _ => trivial
        | send _ ch => exact ChannelName.extends_trans hext htrue
        | recv _ ch => exact ChannelName.extends_trans hext htrue
        | ctrlSend _ ch _ _ => exact ChannelName.extends_trans hext htrue
        | ctrlRecv _ ch _ _ => exact ChannelName.extends_trans hext htrue
      · cases falseLetter with
        | tau => trivial
        | action => trivial
        | choice _ _ => trivial
        | send _ ch => exact ChannelName.extends_trans hext hfalse
        | recv _ ch => exact ChannelName.extends_trans hext hfalse
        | ctrlSend _ ch _ _ => exact ChannelName.extends_trans hext hfalse
        | ctrlRecv _ ch _ _ => exact ChannelName.extends_trans hext hfalse
  | par branches ihBranches =>
      intro h i
      exact ihBranches i (h i)

private theorem mem_concatWords_exists
    {Lifeline : Type u} {Channel : Type v}
    {x : LocalLetter Lifeline Channel}
    {words : List (Word (LocalLetter Lifeline Channel))} :
    x ∈ LocProg.concatWords words →
      ∃ word ∈ words, x ∈ word := by
  induction words with
  | nil =>
      intro h
      simp at h
  | cons word words ih =>
      intro h
      rw [LocProg.concatWords_cons] at h
      rcases List.mem_append.mp h with hx | hx
      · exact ⟨word, by simp, hx⟩
      · rcases ih hx with ⟨word', hword', hx'⟩
        exact ⟨word', by simp [hword'], hx'⟩

theorem trace_letterChannelExtends_of_programChannelsExtend
    {Lifeline : Type u}
    {base : ChannelName}
    {program : LocProg Lifeline ChannelName}
    {word : Word (LocalLetter Lifeline ChannelName)}
    (hprog : programChannelsExtend base program)
    (htrace : LocProg.trace program word) :
    ∀ {letter}, letter ∈ word → letterChannelExtends base letter := by
  induction htrace generalizing base with
  | eps =>
      intro letter hmem
      cases hmem
  | letter letter =>
      intro letter' hmem
      simp at hmem
      subst hmem
      simpa [programChannelsExtend] using hprog
  | seq hfirst hsecond ihFirst ihSecond =>
      intro letter hmem
      rcases List.mem_append.mp hmem with hmem | hmem
      · exact ihFirst hprog.1 hmem
      · exact ihSecond hprog.2 hmem
  | branch_true hrest ihRest =>
      intro letter hmem
      rcases hprog with ⟨htrue, _hfalse, hthen, _helse⟩
      simp at hmem
      rcases hmem with rfl | hmem
      · exact htrue
      · exact ihRest hthen hmem
  | branch_false hrest ihRest =>
      intro letter hmem
      rcases hprog with ⟨_htrue, hfalse, _hthen, helse⟩
      simp at hmem
      rcases hmem with rfl | hmem
      · exact hfalse
      · exact ihRest helse hmem
  | loop hbodies hexit ihBodies ihExit =>
      intro letter hmem
      rcases hprog with ⟨htrue, hfalse, hbody, hExitProg⟩
      rw [List.mem_append] at hmem
      rcases hmem with hmem | hmem
      · rcases mem_concatWords_exists hmem with ⟨segment, hsegmem, hlettermem⟩
        simp at hsegmem
        rcases hsegmem with ⟨i, rfl⟩
        simp at hlettermem
        rcases hlettermem with rfl | hbodyMem
        · exact htrue
        · exact ihBodies i hbody hbodyMem
      · simp at hmem
        rcases hmem with rfl | hexitMem
        · exact hfalse
        · exact ihExit hExitProg hexitMem
  | par htagged herase hbranches ihBranches =>
      rename_i n branches tagged branchWords
      intro letter hmem
      rw [← htagged] at hmem
      rw [List.mem_map] at hmem
      rcases hmem with ⟨taggedLetter, htaggedMem, hEq⟩
      subst hEq
      have hbranchMem :
          taggedLetter.val ∈ eraseBranch taggedLetter.branch tagged := by
        exact mem_eraseBranch_of_mem_branch htaggedMem rfl
      rw [herase taggedLetter.branch] at hbranchMem
      exact ihBranches taggedLetter.branch (hprog taggedLetter.branch) hbranchMem

theorem dtrace_letterChannelExtends_of_programChannelsExtend
    {Lifeline : Type u}
    {base : ChannelName}
    {program : LocProg Lifeline ChannelName}
    {word : Word (LocProg.DecoratedLetter Lifeline ChannelName)}
    (hprog : programChannelsExtend base program)
    (htrace : LocProg.DTrace program word) :
    ∀ {letter}, letter ∈ word →
      letterChannelExtends base (LocProg.DecoratedLetter.toLocal letter) := by
  intro letter hmem
  have hord :
      LocProg.trace program (word.map LocProg.DecoratedLetter.toLocal) :=
    LocProg.dtrace_to_trace htrace
  have hlocalMem :
      LocProg.DecoratedLetter.toLocal letter ∈
        word.map LocProg.DecoratedLetter.toLocal := by
    exact List.mem_map.mpr ⟨letter, hmem, rfl⟩
  exact trace_letterChannelExtends_of_programChannelsExtend hprog hord hlocalMem

theorem controlBroadcast_programChannelsExtend {Lifeline : Type u}
    (targets : List Lifeline) (ch : ChannelName)
    (tag : ControlTag) (kind : ChoiceKind) :
    programChannelsExtend ch (controlBroadcast targets ch tag kind) := by
  induction targets with
  | nil =>
      trivial
  | cons B targets ih =>
      exact ⟨ChannelName.extends_refl ch, ih⟩

theorem project_programChannelsExtend {Lifeline : Type u}
    [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (ch : ChannelName)
    (P : Prog Lifeline) :
    programChannelsExtend ch (project order A ch P) := by
  induction P generalizing A ch with
  | eps =>
      trivial
  | msg sender receiver hne =>
      by_cases hsender : A = sender
      · subst A
        simp [project, programChannelsExtend, letterChannelExtends,
          ChannelName.extends_refl]
      · by_cases hreceiver : A = receiver
        · subst A
          have hreceiver_sender : receiver ≠ sender := by
            intro h
            exact hne h.symm
          simp [project, hreceiver_sender, programChannelsExtend,
            letterChannelExtends, ChannelName.extends_refl]
        · simp [project, hsender, hreceiver, programChannelsExtend]
  | act owner =>
      by_cases howner : A = owner
      · subst A
        simp [project, programChannelsExtend, letterChannelExtends]
      · simp [project, howner, programChannelsExtend]
  | seq first second ihFirst ihSecond =>
      simp [project, programChannelsExtend, ihFirst, ihSecond]
  | branch tag owner thenBranch elseBranch ihThen ihElse =>
      let targets := recipients order owner thenBranch elseBranch
      by_cases howner : A = owner
      · subst A
        simpa [project, programChannelsExtend, targets] using
          (show programChannelsExtend ch
            (LocProg.branch
              (LocalLetter.choice tag ChoiceKind.ifTrue)
              (LocalLetter.choice tag ChoiceKind.ifFalse)
              (LocProg.seq
                (controlBroadcast targets ch tag ChoiceKind.ifTrue)
                (project order owner ch thenBranch))
              (LocProg.seq
                (controlBroadcast targets ch tag ChoiceKind.ifFalse)
                (project order owner ch elseBranch))) from
            ⟨trivial, trivial,
              ⟨controlBroadcast_programChannelsExtend targets ch tag ChoiceKind.ifTrue,
                ihThen owner ch⟩,
              ⟨controlBroadcast_programChannelsExtend targets ch tag ChoiceKind.ifFalse,
                ihElse owner ch⟩⟩)
      · by_cases hrec : A ∈ targets
        · simp [project, programChannelsExtend, howner, hrec, targets,
            ihThen, ihElse, letterChannelExtends, ChannelName.extends_refl]
        · simp [project, programChannelsExtend, howner, hrec, targets]
  | loop tag owner body exit ihBody ihExit =>
      let targets := recipients order owner body exit
      by_cases howner : A = owner
      · subst A
        simpa [project, programChannelsExtend, targets] using
          (show programChannelsExtend ch
            (LocProg.loop
              (LocalLetter.choice tag ChoiceKind.whileTrue)
              (LocalLetter.choice tag ChoiceKind.whileFalse)
              (LocProg.seq
                (controlBroadcast targets ch tag ChoiceKind.whileTrue)
                (project order owner ch body))
              (LocProg.seq
                (controlBroadcast targets ch tag ChoiceKind.whileFalse)
                (project order owner ch exit))) from
            ⟨trivial, trivial,
              ⟨controlBroadcast_programChannelsExtend targets ch tag ChoiceKind.whileTrue,
                ihBody owner ch⟩,
              ⟨controlBroadcast_programChannelsExtend targets ch tag ChoiceKind.whileFalse,
                ihExit owner ch⟩⟩)
      · by_cases hrec : A ∈ targets
        · simp [project, programChannelsExtend, howner, hrec, targets,
            ihBody, ihExit, letterChannelExtends, ChannelName.extends_refl]
        · simp [project, programChannelsExtend, howner, hrec, targets]
  | par label branches ihBranches =>
      intro i
      exact programChannelsExtend_mono
        (ChannelName.branch_extends ch label i)
        (ihBranches i A (ChannelName.branch ch label i))

theorem project_trace_letterChannelExtends
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {A : Lifeline} {ch : ChannelName}
    {P : Prog Lifeline}
    {word : Word (LocalLetter Lifeline ChannelName)}
    (htrace : LocProg.trace (project order A ch P) word) :
    ∀ {letter}, letter ∈ word → letterChannelExtends ch letter := by
  exact trace_letterChannelExtends_of_programChannelsExtend
    (project_programChannelsExtend order A ch P) htrace

theorem project_dtrace_letterChannelExtends
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {A : Lifeline} {ch : ChannelName}
    {P : Prog Lifeline}
    {word : Word (LocProg.DecoratedLetter Lifeline ChannelName)}
    (htrace : LocProg.DTrace (project order A ch P) word) :
    ∀ {letter}, letter ∈ word →
      letterChannelExtends ch (LocProg.DecoratedLetter.toLocal letter) := by
  exact dtrace_letterChannelExtends_of_programChannelsExtend
    (project_programChannelsExtend order A ch P) htrace

theorem seqList_guardDistinct {Lifeline : Type u}
    (programs : List (LocProg Lifeline ChannelName))
    (hprograms : ∀ S, S ∈ programs → LocProg.GuardDistinct S) :
    LocProg.GuardDistinct (seqList programs) := by
  induction programs with
  | nil =>
      trivial
  | cons S Ss ih =>
      exact
        ⟨hprograms S (by simp),
          ih (fun T hT => hprograms T (by simp [hT]))⟩

theorem controlBroadcast_guardDistinct {Lifeline : Type u}
    (targets : List Lifeline) (ch : ChannelName)
    (tag : ControlTag) (kind : ChoiceKind) :
    LocProg.GuardDistinct (controlBroadcast targets ch tag kind) := by
  apply seqList_guardDistinct
  intro S hS
  rcases List.mem_map.mp hS with ⟨B, _hB, hEq⟩
  subst hEq
  trivial

theorem seqList_noPar {Lifeline : Type u}
    (programs : List (LocProg Lifeline ChannelName))
    (hprograms : ∀ S, S ∈ programs → LocProg.NoPar S) :
    LocProg.NoPar (seqList programs) := by
  induction programs with
  | nil =>
      trivial
  | cons S Ss ih =>
      exact
        ⟨hprograms S (by simp),
          ih (fun T hT => hprograms T (by simp [hT]))⟩

theorem controlBroadcast_noPar {Lifeline : Type u}
    (targets : List Lifeline) (ch : ChannelName)
    (tag : ControlTag) (kind : ChoiceKind) :
    LocProg.NoPar (controlBroadcast targets ch tag kind) := by
  apply seqList_noPar
  intro S hS
  rcases List.mem_map.mp hS with ⟨B, _hB, hEq⟩
  subst hEq
  trivial

theorem project_guardDistinct {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (ch : ChannelName)
    (P : Prog Lifeline) :
    LocProg.GuardDistinct (project order A ch P) := by
  induction P generalizing A ch with
  | eps =>
      trivial
  | msg sender receiver hne =>
      by_cases hsender : A = sender
      · subst A
        simp [project]
        trivial
      · by_cases hreceiver : A = receiver
        · subst A
          have hreceiver_sender : receiver ≠ sender := by
            intro h
            exact hne h.symm
          simp [project, hreceiver_sender]
          trivial
        · simp [project, hsender, hreceiver]
          trivial
  | act owner =>
      by_cases howner : A = owner
      · subst A
        simp [project]
        trivial
      · simp [project, howner]
        trivial
  | seq first second ihFirst ihSecond =>
      exact LocProg.GuardDistinct.seq (ihFirst A ch) (ihSecond A ch)
  | branch tag owner thenBranch elseBranch ihThen ihElse =>
      let targets := recipients order owner thenBranch elseBranch
      by_cases howner : A = owner
      · subst A
        simp [project]
        exact LocProg.GuardDistinct.branch_choice_if tag
          (LocProg.GuardDistinct.seq
            (controlBroadcast_guardDistinct targets ch tag ChoiceKind.ifTrue)
            (ihThen owner ch))
          (LocProg.GuardDistinct.seq
            (controlBroadcast_guardDistinct targets ch tag ChoiceKind.ifFalse)
            (ihElse owner ch))
      · by_cases hrec : A ∈ targets
        · simp [project, howner, hrec, targets]
          exact LocProg.GuardDistinct.branch_ctrlRecv_if owner ch tag
            (ihThen A ch) (ihElse A ch)
        · simp [project, howner, hrec, targets]
          trivial
  | loop tag owner body exit ihBody ihExit =>
      let targets := recipients order owner body exit
      by_cases howner : A = owner
      · subst A
        simp [project]
        exact LocProg.GuardDistinct.loop_choice_while tag
          (LocProg.GuardDistinct.seq
            (controlBroadcast_guardDistinct targets ch tag ChoiceKind.whileTrue)
            (ihBody owner ch))
          (LocProg.GuardDistinct.seq
            (controlBroadcast_guardDistinct targets ch tag ChoiceKind.whileFalse)
            (ihExit owner ch))
      · by_cases hrec : A ∈ targets
        · simp [project, howner, hrec, targets]
          exact LocProg.GuardDistinct.loop_ctrlRecv_while owner ch tag
            (ihBody A ch) (ihExit A ch)
        · simp [project, howner, hrec, targets]
          trivial
  | par label branches ihBranches =>
      exact LocProg.GuardDistinct.par
        (fun i => ihBranches i A (ChannelName.branch ch label i))

theorem projectDist_guardDistinct {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (ch : ChannelName) (P : Prog Lifeline) :
    ∀ A, LocProg.GuardDistinct (projectDist order ch P A) :=
  fun A => project_guardDistinct order A ch P

theorem project_noPar {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (ch : ChannelName)
    {P : Prog Lifeline} (hP : NoPar P) :
    LocProg.NoPar (project order A ch P) := by
  induction P generalizing A ch with
  | eps =>
      trivial
  | msg sender receiver hne =>
      by_cases hsender : A = sender
      · subst A
        simp [project]
        trivial
      · by_cases hreceiver : A = receiver
        · subst A
          have hreceiver_sender : receiver ≠ sender := by
            intro h
            exact hne h.symm
          simp [project, hreceiver_sender]
          trivial
        · simp [project, hsender, hreceiver]
          trivial
  | act owner =>
      by_cases howner : A = owner
      · subst A
        simp [project]
        trivial
      · simp [project, howner]
        trivial
  | seq first second ihFirst ihSecond =>
      rcases hP with ⟨hFirst, hSecond⟩
      exact ⟨ihFirst A ch hFirst, ihSecond A ch hSecond⟩
  | branch tag owner thenBranch elseBranch ihThen ihElse =>
      rcases hP with ⟨hThen, hElse⟩
      let targets := recipients order owner thenBranch elseBranch
      by_cases howner : A = owner
      · subst A
        simp [project]
        exact
          ⟨⟨controlBroadcast_noPar targets ch tag ChoiceKind.ifTrue,
              ihThen owner ch hThen⟩,
            ⟨controlBroadcast_noPar targets ch tag ChoiceKind.ifFalse,
              ihElse owner ch hElse⟩⟩
      · by_cases hrec : A ∈ targets
        · simp [project, howner, hrec, targets]
          exact ⟨ihThen A ch hThen, ihElse A ch hElse⟩
        · simp [project, howner, hrec, targets]
          trivial
  | loop tag owner body exit ihBody ihExit =>
      rcases hP with ⟨hBody, hExit⟩
      let targets := recipients order owner body exit
      by_cases howner : A = owner
      · subst A
        simp [project]
        exact
          ⟨⟨controlBroadcast_noPar targets ch tag ChoiceKind.whileTrue,
              ihBody owner ch hBody⟩,
            ⟨controlBroadcast_noPar targets ch tag ChoiceKind.whileFalse,
              ihExit owner ch hExit⟩⟩
      · by_cases hrec : A ∈ targets
        · simp [project, howner, hrec, targets]
          exact ⟨ihBody A ch hBody, ihExit A ch hExit⟩
        · simp [project, howner, hrec, targets]
          trivial
  | par _ branches ihBranches =>
      exact False.elim hP

theorem projectDist_noPar {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (ch : ChannelName)
    {P : Prog Lifeline} (hP : NoPar P) :
    ∀ A, LocProg.NoPar (projectDist order ch P A) :=
  fun A => project_noPar order A ch hP

end Prog

end Source

end ZipperGenParallel
