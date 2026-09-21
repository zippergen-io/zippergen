import ZipperGenSpawn.SourceSyntax
import ZipperGenParallel.LocalProgram

namespace ZipperGenSpawn

universe u v

/-- Local endpoint programs for the spawn extension. The currently executing
    session is held by the runtime control `(A,s)`. A fork template therefore
    stores its static site and child code, while the runtime supplies the
    parent session and visit count when instantiating it. -/
inductive LocProg (Lifeline : Type u) (Field : Type v) : Type (max u v) where
  | eps
  | letter (letter : ZipperGenParallel.LocalLetter Lifeline ChannelScope)
  | seq (first second : LocProg Lifeline Field)
  | branch
      (trueLetter falseLetter :
        ZipperGenParallel.LocalLetter Lifeline ChannelScope)
      (thenBranch elseBranch : LocProg Lifeline Field)
  | loop
      (trueLetter falseLetter :
        ZipperGenParallel.LocalLetter Lifeline ChannelScope)
      (body exit : LocProg Lifeline Field)
  | par {n : Nat} (branches : Fin n -> LocProg Lifeline Field)
  | fork
      (site : Occurrence) (fresh : List Field)
      (child : LocProg Lifeline Field)

namespace LocProg

infixl:50 " ;;l " => LocProg.seq

/-- Erase spawn instructions to local marker actions without entering their
    child templates. This is the local view of the session-skeleton proof
    device used to reuse the checked parallel theorem. -/
def skeleton : LocProg Lifeline Field ->
    ZipperGenParallel.LocProg Lifeline ChannelScope
  | .eps => .eps
  | .letter value => .letter value
  | .seq first second =>
      .seq (skeleton first) (skeleton second)
  | .branch trueLetter falseLetter thenBranch elseBranch =>
      .branch trueLetter falseLetter
        (skeleton thenBranch) (skeleton elseBranch)
  | .loop trueLetter falseLetter body exit =>
      .loop trueLetter falseLetter (skeleton body) (skeleton exit)
  | .par branches => .par (fun i => skeleton (branches i))
  | .fork _ _ _ => .letter ZipperGenParallel.LocalLetter.action

theorem skeleton_seqList
    (programs : List (LocProg Lifeline Field)) :
    skeleton (programs.foldr LocProg.seq LocProg.eps) =
      (programs.map skeleton).foldr ZipperGenParallel.LocProg.seq
        ZipperGenParallel.LocProg.eps := by
  induction programs with
  | nil => rfl
  | cons head tail ih =>
      change
        ZipperGenParallel.LocProg.seq (skeleton head)
            (skeleton (tail.foldr LocProg.seq LocProg.eps)) =
          ZipperGenParallel.LocProg.seq (skeleton head)
            ((tail.map skeleton).foldr ZipperGenParallel.LocProg.seq
              ZipperGenParallel.LocProg.eps)
      rw [ih]

end LocProg

abbrev DistProg (Lifeline : Type u) (Field : Type v) :=
  Lifeline -> LocProg Lifeline Field

end ZipperGenSpawn
