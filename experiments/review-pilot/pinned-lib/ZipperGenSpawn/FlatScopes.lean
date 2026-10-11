import ZipperGenSpawn.Projection
import ZipperGenParallel.SourceProjection

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- Forget the ancestry in the parallel proof's path-shaped channel names.
    Global uniqueness of parallel occurrence labels makes the final branch
    pair a sufficient static FIFO scope for generated programs. -/
def flattenChannel (channel : Source.ChannelName) : ChannelScope :=
  match channel.path.getLast? with
  | none => .main
  | some (site, index) => .branch site index

@[simp] theorem flattenChannel_main :
    flattenChannel Source.ChannelName.main = ChannelScope.main := rfl

@[simp] theorem flattenChannel_branch {n : Nat}
    (channel : Source.ChannelName) (site : Occurrence) (i : Fin n) :
    flattenChannel (Source.ChannelName.branch channel site i) =
      ChannelScope.branch site (i.val + 1) := by
  simp [flattenChannel, Source.ChannelName.branch,
    Source.ChannelName.extend]

/-- Rename the channel component of a parallel-proof letter to its flat
    static scope. -/
def flattenLetter :
    LocalLetter Lifeline Source.ChannelName ->
      LocalLetter Lifeline ChannelScope
  | .tau => .tau
  | .action => .action
  | .choice tag kind => .choice tag kind
  | .send target channel => .send target (flattenChannel channel)
  | .recv source channel => .recv source (flattenChannel channel)
  | .ctrlSend target channel tag kind =>
      .ctrlSend target (flattenChannel channel) tag kind
  | .ctrlRecv source channel tag kind =>
      .ctrlRecv source (flattenChannel channel) tag kind

/-- Rename every channel in a local parallel program. -/
def flattenLoc :
    ZipperGenParallel.LocProg Lifeline Source.ChannelName ->
      ZipperGenParallel.LocProg Lifeline ChannelScope
  | .eps => .eps
  | .letter letter => .letter (flattenLetter letter)
  | .seq first second => .seq (flattenLoc first) (flattenLoc second)
  | .branch trueLetter falseLetter thenBranch elseBranch =>
      .branch (flattenLetter trueLetter) (flattenLetter falseLetter)
        (flattenLoc thenBranch) (flattenLoc elseBranch)
  | .loop trueLetter falseLetter body exit =>
      .loop (flattenLetter trueLetter) (flattenLetter falseLetter)
        (flattenLoc body) (flattenLoc exit)
  | .par branches => .par (fun i => flattenLoc (branches i))

@[simp] theorem flattenLetter_send
    (target : Lifeline) (channel : Source.ChannelName) :
    flattenLetter (LocalLetter.send target channel) =
      LocalLetter.send target (flattenChannel channel) := rfl

@[simp] theorem flattenLetter_recv
    (source : Lifeline) (channel : Source.ChannelName) :
    flattenLetter (LocalLetter.recv source channel) =
      LocalLetter.recv source (flattenChannel channel) := rfl

@[simp] theorem flattenLetter_ctrlSend
    (target : Lifeline) (channel : Source.ChannelName)
    (tag : ControlTag) (kind : ChoiceKind) :
    flattenLetter (LocalLetter.ctrlSend target channel tag kind) =
      LocalLetter.ctrlSend target (flattenChannel channel) tag kind := rfl

@[simp] theorem flattenLetter_ctrlRecv
    (source : Lifeline) (channel : Source.ChannelName)
    (tag : ControlTag) (kind : ChoiceKind) :
    flattenLetter (LocalLetter.ctrlRecv source channel tag kind) =
      LocalLetter.ctrlRecv source (flattenChannel channel) tag kind := rfl

/-- Mapping channels commutes with a finite local sequence. -/
theorem flattenLoc_seqList
    (programs : List (ZipperGenParallel.LocProg Lifeline Source.ChannelName)) :
    flattenLoc (Source.Prog.seqList programs) =
      (programs.map flattenLoc).foldr ZipperGenParallel.LocProg.seq
        ZipperGenParallel.LocProg.eps := by
  induction programs with
  | nil => rfl
  | cons head tail ih =>
      change
        flattenLoc
            (ZipperGenParallel.LocProg.seq head
              (Source.Prog.seqList tail)) =
          ZipperGenParallel.LocProg.seq (flattenLoc head)
            ((tail.map flattenLoc).foldr
              ZipperGenParallel.LocProg.seq
              ZipperGenParallel.LocProg.eps)
      simp only [flattenLoc]
      rw [ih]

end ZipperGenSpawn
