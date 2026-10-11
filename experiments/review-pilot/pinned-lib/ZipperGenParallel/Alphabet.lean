import ZipperGenParallel.Basic

namespace ZipperGenParallel

universe u v

/--
Local letters of a projected participant.

The owner lifeline of the word is external to the letter.  A send records
the receiver and channel name.  A receive records the sender and channel
name.  Control sends and receives are kept separate so that the correctness
statement can erase them after comparing with the internal enriched semantics.
-/
inductive ChoiceKind where
  | ifTrue
  | ifFalse
  | whileTrue
  | whileFalse
deriving Repr, DecidableEq

abbrev ControlTag := Nat

inductive LocalLetter (Lifeline : Type u) (Channel : Type v) where
  | tau : LocalLetter Lifeline Channel
  | action : LocalLetter Lifeline Channel
  | choice (tag : ControlTag) (kind : ChoiceKind) :
      LocalLetter Lifeline Channel
  | send (to : Lifeline) (ch : Channel) : LocalLetter Lifeline Channel
  | recv (src : Lifeline) (ch : Channel) : LocalLetter Lifeline Channel
  | ctrlSend
      (to : Lifeline) (ch : Channel) (tag : ControlTag)
      (kind : ChoiceKind) :
      LocalLetter Lifeline Channel
  | ctrlRecv
      (src : Lifeline) (ch : Channel) (tag : ControlTag)
      (kind : ChoiceKind) :
      LocalLetter Lifeline Channel
deriving Repr

/--
Event kinds in the MSC semantics.

Send events carry their receiver and channel.  Receive events carry their
sender and channel.  This is the information needed both for matching and
for channel-sensitive matching in parallel regions.
-/
inductive EventKind (Lifeline : Type u) (Channel : Type v) where
  | local : EventKind Lifeline Channel
  | send (to : Lifeline) (ch : Channel) : EventKind Lifeline Channel
  | recv (src : Lifeline) (ch : Channel) : EventKind Lifeline Channel
deriving Repr

def EventKind.isSend {Lifeline : Type u} {Channel : Type v}
    : EventKind Lifeline Channel → Prop
  | EventKind.send _ _ => True
  | _ => False

def EventKind.isRecv {Lifeline : Type u} {Channel : Type v}
    : EventKind Lifeline Channel → Prop
  | EventKind.recv _ _ => True
  | _ => False

namespace LocalLetter

def MessageMatches {Lifeline : Type u} {Channel : Type v}
    (sender receiver : Lifeline) :
    LocalLetter Lifeline Channel → LocalLetter Lifeline Channel → Prop
  | LocalLetter.send to chSend, LocalLetter.recv src chRecv =>
      to = receiver ∧ src = sender ∧ chSend = chRecv
  | LocalLetter.ctrlSend to chSend tagSend kindSend,
      LocalLetter.ctrlRecv src chRecv tagRecv kindRecv =>
      to = receiver ∧ src = sender ∧ chSend = chRecv ∧
        tagSend = tagRecv ∧ kindSend = kindRecv
  | _, _ => False

theorem messageMatches_ctrl_tag_eq {Lifeline : Type u} {Channel : Type v}
    {sender receiver : Lifeline} {chSend chRecv : Channel}
    {tagSend tagRecv : ControlTag}
    {kindSend kindRecv : ChoiceKind}
    (h :
      MessageMatches sender receiver
        (LocalLetter.ctrlSend receiver chSend tagSend kindSend)
        (LocalLetter.ctrlRecv sender chRecv tagRecv kindRecv)) :
    tagSend = tagRecv :=
  h.2.2.2.1

theorem messageMatches_ctrl_kind_eq {Lifeline : Type u} {Channel : Type v}
    {sender receiver : Lifeline} {chSend chRecv : Channel}
    {tagSend tagRecv : ControlTag}
    {kindSend kindRecv : ChoiceKind}
    (h :
      MessageMatches sender receiver
        (LocalLetter.ctrlSend receiver chSend tagSend kindSend)
        (LocalLetter.ctrlRecv sender chRecv tagRecv kindRecv)) :
    kindSend = kindRecv :=
  h.2.2.2.2

/--
Convert a local projected letter into the corresponding MSC event kind.
The owner lifeline is still supplied by the local word in which the event
occurs.
-/
def toEventKind {Lifeline : Type u} {Channel : Type v} :
    LocalLetter Lifeline Channel → EventKind Lifeline Channel
  | LocalLetter.tau => EventKind.local
  | LocalLetter.action => EventKind.local
  | LocalLetter.choice _ _ => EventKind.local
  | LocalLetter.send B ch => EventKind.send B ch
  | LocalLetter.recv A ch => EventKind.recv A ch
  | LocalLetter.ctrlSend B ch _ _ => EventKind.send B ch
  | LocalLetter.ctrlRecv A ch _ _ => EventKind.recv A ch

@[simp]
theorem toEventKind_tau {Lifeline : Type u} {Channel : Type v} :
    toEventKind (LocalLetter.tau : LocalLetter Lifeline Channel) =
      EventKind.local :=
  rfl

@[simp]
theorem toEventKind_action {Lifeline : Type u} {Channel : Type v} :
    toEventKind (LocalLetter.action : LocalLetter Lifeline Channel) =
      EventKind.local :=
  rfl

@[simp]
theorem toEventKind_choice {Lifeline : Type u} {Channel : Type v}
    (tag : ControlTag) (kind : ChoiceKind) :
    toEventKind
        (LocalLetter.choice tag kind : LocalLetter Lifeline Channel) =
      EventKind.local :=
  rfl

@[simp]
theorem toEventKind_send {Lifeline : Type u} {Channel : Type v}
    (B : Lifeline) (ch : Channel) :
    toEventKind (LocalLetter.send B ch) = EventKind.send B ch :=
  rfl

@[simp]
theorem toEventKind_recv {Lifeline : Type u} {Channel : Type v}
    (A : Lifeline) (ch : Channel) :
    toEventKind (LocalLetter.recv A ch) = EventKind.recv A ch :=
  rfl

@[simp]
theorem toEventKind_ctrlSend {Lifeline : Type u} {Channel : Type v}
    (B : Lifeline) (ch : Channel) (tag : ControlTag)
    (kind : ChoiceKind) :
    toEventKind (LocalLetter.ctrlSend B ch tag kind) =
      EventKind.send B ch :=
  rfl

@[simp]
theorem toEventKind_ctrlRecv {Lifeline : Type u} {Channel : Type v}
    (A : Lifeline) (ch : Channel) (tag : ControlTag)
    (kind : ChoiceKind) :
    toEventKind (LocalLetter.ctrlRecv A ch tag kind) =
      EventKind.recv A ch :=
  rfl

@[simp]
theorem choice_ifTrue_ne_ifFalse {Lifeline : Type u} {Channel : Type v}
    (tag : ControlTag) :
    (LocalLetter.choice tag ChoiceKind.ifTrue :
      LocalLetter Lifeline Channel) ≠
    LocalLetter.choice tag ChoiceKind.ifFalse := by
  intro h
  cases h

@[simp]
theorem choice_whileTrue_ne_whileFalse {Lifeline : Type u}
    {Channel : Type v} (tag : ControlTag) :
    (LocalLetter.choice tag ChoiceKind.whileTrue :
      LocalLetter Lifeline Channel) ≠
    LocalLetter.choice tag ChoiceKind.whileFalse := by
  intro h
  cases h

@[simp]
theorem ctrlSend_ifTrue_ne_ifFalse {Lifeline : Type u} {Channel : Type v}
    (B : Lifeline) (ch : Channel) (tag : ControlTag) :
    (LocalLetter.ctrlSend B ch tag ChoiceKind.ifTrue :
      LocalLetter Lifeline Channel) ≠
    LocalLetter.ctrlSend B ch tag ChoiceKind.ifFalse := by
  intro h
  cases h

@[simp]
theorem ctrlRecv_ifTrue_ne_ifFalse {Lifeline : Type u} {Channel : Type v}
    (A : Lifeline) (ch : Channel) (tag : ControlTag) :
    (LocalLetter.ctrlRecv A ch tag ChoiceKind.ifTrue :
      LocalLetter Lifeline Channel) ≠
    LocalLetter.ctrlRecv A ch tag ChoiceKind.ifFalse := by
  intro h
  cases h

@[simp]
theorem ctrlSend_whileTrue_ne_whileFalse {Lifeline : Type u}
    {Channel : Type v} (B : Lifeline) (ch : Channel) (tag : ControlTag) :
    (LocalLetter.ctrlSend B ch tag ChoiceKind.whileTrue :
      LocalLetter Lifeline Channel) ≠
    LocalLetter.ctrlSend B ch tag ChoiceKind.whileFalse := by
  intro h
  cases h

@[simp]
theorem ctrlRecv_whileTrue_ne_whileFalse {Lifeline : Type u}
    {Channel : Type v} (A : Lifeline) (ch : Channel) (tag : ControlTag) :
    (LocalLetter.ctrlRecv A ch tag ChoiceKind.whileTrue :
      LocalLetter Lifeline Channel) ≠
    LocalLetter.ctrlRecv A ch tag ChoiceKind.whileFalse := by
  intro h
  cases h

/-- A single-letter control erasure map. -/
def eraseCtrlLetter {Lifeline : Type u} {Channel : Type v} :
    LocalLetter Lifeline Channel → Option (LocalLetter Lifeline Channel)
  | LocalLetter.ctrlSend .. => none
  | LocalLetter.ctrlRecv .. => none
  | letter => some letter

/-- Erase compiler-inserted control sends and receives from a local word. -/
def eraseCtrlWord {Lifeline : Type u} {Channel : Type v} :
    Word (LocalLetter Lifeline Channel) → Word (LocalLetter Lifeline Channel) :=
  List.filterMap eraseCtrlLetter

@[simp]
theorem eraseCtrlLetter_tau {Lifeline : Type u} {Channel : Type v} :
    eraseCtrlLetter (LocalLetter.tau : LocalLetter Lifeline Channel) =
      some LocalLetter.tau :=
  rfl

@[simp]
theorem eraseCtrlLetter_action {Lifeline : Type u} {Channel : Type v} :
    eraseCtrlLetter (LocalLetter.action : LocalLetter Lifeline Channel) =
      some LocalLetter.action :=
  rfl

@[simp]
theorem eraseCtrlLetter_choice {Lifeline : Type u} {Channel : Type v}
    (tag : ControlTag) (kind : ChoiceKind) :
    eraseCtrlLetter
        (LocalLetter.choice tag kind : LocalLetter Lifeline Channel) =
      some (LocalLetter.choice tag kind) :=
  rfl

@[simp]
theorem eraseCtrlLetter_send {Lifeline : Type u} {Channel : Type v}
    (B : Lifeline) (ch : Channel) :
    eraseCtrlLetter (LocalLetter.send B ch) = some (LocalLetter.send B ch) :=
  rfl

@[simp]
theorem eraseCtrlLetter_recv {Lifeline : Type u} {Channel : Type v}
    (A : Lifeline) (ch : Channel) :
    eraseCtrlLetter (LocalLetter.recv A ch) = some (LocalLetter.recv A ch) :=
  rfl

@[simp]
theorem eraseCtrlLetter_ctrlSend {Lifeline : Type u} {Channel : Type v}
    (B : Lifeline) (ch : Channel) (tag : ControlTag)
    (kind : ChoiceKind) :
    eraseCtrlLetter (LocalLetter.ctrlSend B ch tag kind) = none :=
  rfl

@[simp]
theorem eraseCtrlLetter_ctrlRecv {Lifeline : Type u} {Channel : Type v}
    (A : Lifeline) (ch : Channel) (tag : ControlTag)
    (kind : ChoiceKind) :
    eraseCtrlLetter (LocalLetter.ctrlRecv A ch tag kind) = none :=
  rfl

@[simp]
theorem eraseCtrlWord_nil {Lifeline : Type u} {Channel : Type v} :
    eraseCtrlWord ([] : Word (LocalLetter Lifeline Channel)) = [] :=
  rfl

@[simp]
theorem eraseCtrlWord_cons {Lifeline : Type u} {Channel : Type v}
    (letter : LocalLetter Lifeline Channel)
    (word : Word (LocalLetter Lifeline Channel)) :
    eraseCtrlWord (letter :: word) =
      match eraseCtrlLetter letter with
      | some letter' => letter' :: eraseCtrlWord word
      | none => eraseCtrlWord word := by
  cases h : eraseCtrlLetter letter <;> simp [eraseCtrlWord, h]

theorem eraseCtrlWord_append {Lifeline : Type u} {Channel : Type v}
    (left right : Word (LocalLetter Lifeline Channel)) :
    eraseCtrlWord (left ++ right) =
      eraseCtrlWord left ++ eraseCtrlWord right := by
  induction left with
  | nil =>
      rfl
  | cons letter left ih =>
      cases h : eraseCtrlLetter letter with
      | none =>
          simp [eraseCtrlWord, h]
      | some letter' =>
          simp [eraseCtrlWord, h]

@[simp]
theorem eraseCtrlWord_idem {Lifeline : Type u} {Channel : Type v}
    (word : Word (LocalLetter Lifeline Channel)) :
    eraseCtrlWord (eraseCtrlWord word) = eraseCtrlWord word := by
  induction word with
  | nil =>
      rfl
  | cons letter word ih =>
      cases letter <;> simpa [eraseCtrlWord] using ih

end LocalLetter

end ZipperGenParallel
