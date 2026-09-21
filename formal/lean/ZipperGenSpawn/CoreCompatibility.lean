import ZipperGenSpawn.FlatScopes

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

namespace Prog

/-- The spawn-free fragment to which the checked parallel projection theorem
    applies directly. -/
def NoSpawn : Prog Lifeline Field -> Prop
  | .eps | .msg .. | .act _ => True
  | .seq first second => NoSpawn first /\ NoSpawn second
  | .branch _ _ thenBranch elseBranch =>
      NoSpawn thenBranch /\ NoSpawn elseBranch
  | .loop _ _ body exit => NoSpawn body /\ NoSpawn exit
  | .par _ branches => ∀ i, NoSpawn (branches i)
  | .spawn .. => False

/-- Forget freshness and spawn bodies. On `NoSpawn` programs this is simply a
    change of representation into the parallel paper's source language. -/
def toParallel : Prog Lifeline Field -> Source.Prog Lifeline
  | .eps => .eps
  | .msg sender receiver h => .msg sender receiver h
  | .act owner => .act owner
  | .seq first second => .seq (toParallel first) (toParallel second)
  | .branch site owner thenBranch elseBranch =>
      .branch site owner (toParallel thenBranch) (toParallel elseBranch)
  | .loop site owner body exit =>
      .loop site owner (toParallel body) (toParallel exit)
  | .par site branches => .par site (fun i => toParallel (branches i))
  | .spawn _ _ _ => .eps

theorem toParallel_lifelines_of_noSpawn :
    ∀ P : Prog Lifeline Field,
      NoSpawn P -> Source.Prog.lifelines (toParallel P) = lifelines P
  | .eps, _ => rfl
  | .msg .., _ => rfl
  | .act _, _ => rfl
  | .seq first second, h => by
      simp only [NoSpawn] at h
      simp [toParallel, Source.Prog.lifelines, lifelines,
        toParallel_lifelines_of_noSpawn first h.1,
        toParallel_lifelines_of_noSpawn second h.2]
  | .branch site owner thenBranch elseBranch, h => by
      simp only [NoSpawn] at h
      simp [toParallel, Source.Prog.lifelines, lifelines,
        toParallel_lifelines_of_noSpawn thenBranch h.1,
        toParallel_lifelines_of_noSpawn elseBranch h.2]
  | .loop site owner body exit, h => by
      simp only [NoSpawn] at h
      simp [toParallel, Source.Prog.lifelines, lifelines,
        toParallel_lifelines_of_noSpawn body h.1,
        toParallel_lifelines_of_noSpawn exit h.2]
  | .par site branches, h => by
      simp only [NoSpawn] at h
      simp only [toParallel, Source.Prog.lifelines, lifelines]
      congr 1
      funext i
      exact toParallel_lifelines_of_noSpawn (branches i) (h i)
  | .spawn .., h => False.elim h

theorem recipients_toParallel_of_noSpawn [DecidableEq Lifeline]
    (order : List Lifeline) (owner : Lifeline)
    (left right : Prog Lifeline Field)
    (hleft : NoSpawn left) (hright : NoSpawn right) :
    Source.Prog.recipients order owner
        (toParallel left) (toParallel right) =
      recipients order owner left right := by
  simp [Source.Prog.recipients, recipients,
    toParallel_lifelines_of_noSpawn left hleft,
    toParallel_lifelines_of_noSpawn right hright]

theorem flatten_controlBroadcast
    (targets : List Lifeline) (channel : Source.ChannelName)
    (site : Occurrence) (kind : ChoiceKind) :
    flattenLoc
        (Source.Prog.controlBroadcast targets channel site kind) =
      LocProg.skeleton
        (controlBroadcast (Field := Field) targets
          (flattenChannel channel) site kind) := by
  unfold Source.Prog.controlBroadcast controlBroadcast
  rw [flattenLoc_seqList]
  unfold seqLocList
  rw [LocProg.skeleton_seqList]
  simp [List.map_map, Function.comp_def, flattenLoc, flattenLetter,
    LocProg.skeleton]

/-- The flat-scope spawn projection is exactly the older path-scope parallel
    projection after channel flattening, on programs that contain no spawn.
    This is the checked compatibility bridge used by the session skeleton. -/
theorem project_noSpawn_compat [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline)
    (channel : Source.ChannelName) :
    ∀ P : Prog Lifeline Field,
      NoSpawn P ->
        flattenLoc (Source.Prog.project order A channel (toParallel P)) =
          LocProg.skeleton
            (project order A (flattenChannel channel) P)
  | .eps, _ => rfl
  | .msg sender receiver hne, _ => by
      by_cases hsender : A = sender
      · simp [toParallel, Source.Prog.project, project, hsender,
          flattenLoc, flattenLetter, LocProg.skeleton]
      · by_cases hreceiver : A = receiver
        · have hreceiverSender : receiver ≠ sender := by
            intro h
            exact hne h.symm
          simp [toParallel, Source.Prog.project, project, hreceiver,
            hreceiverSender, flattenLoc, flattenLetter, LocProg.skeleton]
        · simp [toParallel, Source.Prog.project, project, hsender, hreceiver,
            flattenLoc, LocProg.skeleton]
  | .act owner, _ => by
      by_cases howner : A = owner
      · simp [toParallel, Source.Prog.project, project, howner,
          flattenLoc, flattenLetter, LocProg.skeleton]
      · simp [toParallel, Source.Prog.project, project, howner,
          flattenLoc, LocProg.skeleton]
  | .seq first second, h => by
      simp only [NoSpawn] at h
      simp [toParallel, Source.Prog.project, project, flattenLoc,
        LocProg.skeleton,
        project_noSpawn_compat order A channel first h.1,
        project_noSpawn_compat order A channel second h.2]
  | .branch site owner thenBranch elseBranch, h => by
      simp only [NoSpawn] at h
      have htargets := recipients_toParallel_of_noSpawn
        (Field := Field) order owner thenBranch elseBranch h.1 h.2
      by_cases howner : A = owner
      · subst A
        have ihThen :=
          project_noSpawn_compat order owner channel thenBranch h.1
        have ihElse :=
          project_noSpawn_compat order owner channel elseBranch h.2
        simp [toParallel, Source.Prog.project, project, htargets,
          flattenLoc, flattenLetter, LocProg.skeleton,
          flatten_controlBroadcast (Field := Field), ihThen, ihElse]
      · by_cases hrec : A ∈ recipients order owner thenBranch elseBranch
        · simp [toParallel, Source.Prog.project, project, howner, hrec,
            htargets, flattenLoc, flattenLetter, LocProg.skeleton,
            project_noSpawn_compat order A channel thenBranch h.1,
            project_noSpawn_compat order A channel elseBranch h.2]
        · simp [toParallel, Source.Prog.project, project, howner, hrec,
            htargets, flattenLoc, LocProg.skeleton]
  | .loop site owner body exit, h => by
      simp only [NoSpawn] at h
      have htargets := recipients_toParallel_of_noSpawn
        (Field := Field) order owner body exit h.1 h.2
      by_cases howner : A = owner
      · subst A
        have ihBody :=
          project_noSpawn_compat order owner channel body h.1
        have ihExit :=
          project_noSpawn_compat order owner channel exit h.2
        simp [toParallel, Source.Prog.project, project, htargets,
          flattenLoc, flattenLetter, LocProg.skeleton,
          flatten_controlBroadcast (Field := Field), ihBody, ihExit]
      · by_cases hrec : A ∈ recipients order owner body exit
        · simp [toParallel, Source.Prog.project, project, howner, hrec,
            htargets, flattenLoc, flattenLetter, LocProg.skeleton,
            project_noSpawn_compat order A channel body h.1,
            project_noSpawn_compat order A channel exit h.2]
        · simp [toParallel, Source.Prog.project, project, howner, hrec,
            htargets, flattenLoc, LocProg.skeleton]
  | .par site branches, h => by
      simp only [NoSpawn] at h
      simp only [toParallel, Source.Prog.project, project, flattenLoc,
        LocProg.skeleton]
      congr 1
      funext i
      simpa using
        project_noSpawn_compat order A
          (Source.ChannelName.branch channel site i)
          (branches i) (h i)
  | .spawn .., h => False.elim h

end Prog

end ZipperGenSpawn
