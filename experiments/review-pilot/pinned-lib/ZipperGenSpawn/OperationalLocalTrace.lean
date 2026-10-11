import ZipperGenSpawn.OperationalSemantics

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

namespace LocProg

/-- Forget detached child templates when observing one session. A local fork
    is represented separately in the physical spawn history and contributes
    no ordinary endpoint letter to its parent session. -/
def eraseForks : ZipperGenSpawn.LocProg Lifeline Field ->
    ZipperGenParallel.LocProg Lifeline ChannelScope
  | ZipperGenSpawn.LocProg.eps => ZipperGenParallel.LocProg.eps
  | ZipperGenSpawn.LocProg.letter value =>
      ZipperGenParallel.LocProg.letter value
  | ZipperGenSpawn.LocProg.seq first second =>
      ZipperGenParallel.LocProg.seq first.eraseForks second.eraseForks
  | ZipperGenSpawn.LocProg.branch trueLetter falseLetter thenBranch elseBranch =>
      ZipperGenParallel.LocProg.branch trueLetter falseLetter
        thenBranch.eraseForks elseBranch.eraseForks
  | ZipperGenSpawn.LocProg.loop trueLetter falseLetter body exit =>
      ZipperGenParallel.LocProg.loop trueLetter falseLetter
        body.eraseForks exit.eraseForks
  | ZipperGenSpawn.LocProg.par branches =>
      ZipperGenParallel.LocProg.par (fun i => (branches i).eraseForks)
  | ZipperGenSpawn.LocProg.fork _ _ _ => ZipperGenParallel.LocProg.eps

end LocProg

/-- Fork erasure never destroys local trace existence. This supplies the
    completion witness needed to turn a residual decomposition into a prefix
    theorem. -/
theorem eraseForks_trace_exists
    (program : LocProg Lifeline Field) :
    ∃ word, ZipperGenParallel.LocProg.trace program.eraseForks word := by
  induction program with
  | eps =>
      exact ⟨[], ZipperGenParallel.LocProg.Trace.eps⟩
  | letter value =>
      exact ⟨[value], ZipperGenParallel.LocProg.Trace.letter value⟩
  | seq first second ihFirst ihSecond =>
      rcases ihFirst with ⟨firstWord, hfirst⟩
      rcases ihSecond with ⟨secondWord, hsecond⟩
      exact ⟨firstWord ++ secondWord,
        ZipperGenParallel.LocProg.Trace.seq hfirst hsecond⟩
  | branch trueLetter falseLetter thenBranch elseBranch ihThen ihElse =>
      rcases ihThen with ⟨rest, hrest⟩
      exact ⟨trueLetter :: rest,
        ZipperGenParallel.LocProg.Trace.branch_true hrest⟩
  | loop trueLetter falseLetter body exit ihBody ihExit =>
      rcases ihExit with ⟨exitWord, hexit⟩
      exact ⟨falseLetter :: exitWord,
        (ZipperGenParallel.LocProg.trace_loop_unfold _ _ _ _ _).2
          (Or.inl ⟨exitWord, hexit, rfl⟩)⟩
  | @par n branches ihBranches =>
      classical
      let branchWords : Fin n -> Word (LocalLetter Lifeline ChannelScope) :=
        fun i => Classical.choose (ihBranches i)
      let tagged := serialTaggedWordOf branchWords
      refine ⟨tagged.map Tagged.val,
        ZipperGenParallel.LocProg.Trace.par
          (tagged := tagged) (branchWords := branchWords) rfl ?_ ?_⟩
      · intro i
        exact eraseBranch_serialTaggedWordOf branchWords i
      · intro i
        exact (Classical.choose_spec (ihBranches i))
  | fork site fresh child =>
      exact ⟨[], ZipperGenParallel.LocProg.Trace.eps⟩

/-- Ordinary session letters contributed by one local transition. Fork and
    administrative transitions are silent in the session restriction. -/
def emissionWord : Emission Lifeline ->
    Word (LocalLetter Lifeline ChannelScope)
  | .event letter => [letter]
  | .silent | .fork .. => []

@[simp] theorem projectedSessionWord_append
    (session : SessionName)
    (left right : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    projectedSessionWord session (left ++ right) =
      projectedSessionWord session left ++ projectedSessionWord session right := by
  induction left with
  | nil => rfl
  | cons head tail ih =>
      cases head with
      | fork parent site child => simp [projectedSessionWord, ih]
      | event current letter =>
          by_cases hcurrent : current = session
          · subst current
            simp [projectedSessionWord, ih]
          · simp [projectedSessionWord, hcurrent, ih]

@[simp] theorem projectedSessionWord_appendEmission_self
    [DecidableEq Lifeline]
    (history : ProjectedSpawnTuple Lifeline ChannelScope)
    (A : Lifeline) (session : SessionName) (emission : Emission Lifeline) :
    projectedSessionWord session
        (appendEmission history A session emission A) =
      projectedSessionWord session (history A) ++ emissionWord emission := by
  cases emission <;>
    simp [appendEmission, projectedSessionWord, emissionWord]

@[simp] theorem projectedSessionWord_appendEmission_other
    [DecidableEq Lifeline]
    (history : ProjectedSpawnTuple Lifeline ChannelScope)
    {A B : Lifeline} (hne : B ≠ A)
    (session current : SessionName) (emission : Emission Lifeline) :
    projectedSessionWord current
        (appendEmission history A session emission B) =
      projectedSessionWord current (history B) := by
  cases emission <;> simp [appendEmission, hne]

theorem projectedSessionWord_eq_nil_of_no_event
    (session : SessionName)
    (word : List (ProjectedSpawnLetter Lifeline ChannelScope))
    (habsent : ∀ letter,
      ProjectedSpawnLetter.event session letter ∉ word) :
    projectedSessionWord session word = [] := by
  induction word with
  | nil => rfl
  | cons head tail ih =>
      cases head with
      | fork parent site child =>
          simp [projectedSessionWord, ih (fun letter hmem =>
            habsent letter (by simp [hmem]))]
      | event current letter =>
          have htail : ∀ value,
              ProjectedSpawnLetter.event session value ∉ tail := by
            intro value hmem
            exact habsent value (by simp [hmem])
          by_cases hcurrent : current = session
          · subst current
            exact False.elim (habsent letter (by simp))
          · simp [projectedSessionWord, hcurrent, ih htail]

/-- One operational local step is a leftmost trace decomposition after fork
    erasure. Any complete trace of the residual extends to a complete trace
    of the source program by prefixing the step's ordinary observation. -/
theorem localStep_trace_decomposition [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {program residual : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline}
    {activation : Activation Lifeline Field}
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission activation) :
    ∀ {rest},
      ZipperGenParallel.LocProg.trace residual.eraseForks rest ->
      ZipperGenParallel.LocProg.trace program.eraseForks
        (emissionWord emission ++ rest) := by
  induction hstep with
  | letter hletter =>
      intro rest hrest
      have : rest = [] :=
        (ZipperGenParallel.LocProg.trace_eps rest).1 hrest
      subst rest
      exact ZipperGenParallel.LocProg.Trace.letter _
  | seq_done second =>
      intro rest hrest
      exact ZipperGenParallel.LocProg.Trace.seq
        ZipperGenParallel.LocProg.Trace.eps hrest
  | seq_left hleft ih =>
      intro rest hrest
      rcases (ZipperGenParallel.LocProg.trace_seq _ _ rest).1 hrest with
        ⟨firstWord, secondWord, hfirst, hsecond, rfl⟩
      simpa [emissionWord, List.append_assoc] using
        ZipperGenParallel.LocProg.Trace.seq (ih hfirst) hsecond
  | branch_true htrue =>
      intro rest hrest
      exact ZipperGenParallel.LocProg.Trace.branch_true hrest
  | branch_false hfalse =>
      intro rest hrest
      exact ZipperGenParallel.LocProg.Trace.branch_false hrest
  | loop_true htrue =>
      intro rest hrest
      rcases (ZipperGenParallel.LocProg.trace_seq _ _ rest).1 hrest with
        ⟨bodyWord, loopWord, hbody, hloop, rfl⟩
      exact (ZipperGenParallel.LocProg.trace_loop_unfold _ _ _ _ _).2
        (Or.inr ⟨bodyWord, loopWord, hbody, hloop, rfl⟩)
  | loop_false hfalse =>
      intro rest hrest
      exact (ZipperGenParallel.LocProg.trace_loop_unfold _ _ _ _ _).2
        (Or.inl ⟨rest, hrest, rfl⟩)
  | par_done hdone =>
      intro rest hrest
      have hnil : rest = [] :=
        (ZipperGenParallel.LocProg.trace_eps rest).1 hrest
      subst rest
      refine ZipperGenParallel.LocProg.Trace.par
        (tagged := []) (branchWords := fun _ => []) rfl ?_ ?_
      · intro i
        rfl
      · intro i
        rw [hdone i]
        exact ZipperGenParallel.LocProg.Trace.eps
  | @par_branch branchField branchCounters branchQueues n branches i counters'
      queues' branchResidual branchEmission branchActivation hbranch ih =>
      intro rest hrest
      rcases ZipperGenParallel.LocProg.trace_par_elim hrest with
        ⟨tagged, htagged, branchWords, herase, hbranches⟩
      let prefixTagged := taggedWordOf i (emissionWord branchEmission)
      refine ZipperGenParallel.LocProg.Trace.par
        (tagged := prefixTagged ++ tagged)
        (branchWords := fun j =>
          if j = i then emissionWord branchEmission ++ branchWords j
          else branchWords j)
        ?_ ?_ ?_
      · simp [prefixTagged, taggedWordOf, List.map_map,
          Function.comp_def, htagged]
      · intro j
        by_cases hji : j = i
        · subst j
          simp [prefixTagged, eraseBranch_append,
            eraseBranch_taggedWordOf_same, herase]
        · simp [prefixTagged, eraseBranch_append,
            eraseBranch_taggedWordOf_ne (i := j) (j := i) hji,
            herase, hji]
      · intro j
        by_cases hji : j = i
        · subst j
          have hi := hbranches i
          simp at hi
          simpa using ih hi
        · simpa [hji] using hbranches j
  | fork site fresh childProgram =>
      intro rest hrest
      exact hrest

/-- A fork emission always names the currently executing session as parent. -/
theorem localStep_fork_parent [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {program residual : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline}
    {activation : Activation Lifeline Field}
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission activation)
    {parent child : SessionName} {site : Occurrence}
    (hemits : emission = .fork parent site child) :
    parent = session := by
  induction hstep with
  | letter hletter => cases hemits
  | seq_done second => cases hemits
  | seq_left hleft ih => exact ih hemits
  | branch_true htrue => cases hemits
  | branch_false hfalse => cases hemits
  | loop_true htrue => cases hemits
  | loop_false hfalse => cases hemits
  | par_done hdone => cases hemits
  | par_branch i hbranch ih => exact ih hemits
  | fork forkSite fresh childProgram =>
      cases hemits
      rfl

theorem localStep_fork_child [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {program residual : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline}
    {activation : Activation Lifeline Field}
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission activation)
    {parent child : SessionName} {site : Occurrence}
    (hemits : emission = .fork parent site child) :
    child = SessionName.child session site
      (nextVisit counters A session site) := by
  induction hstep with
  | letter hletter => cases hemits
  | seq_done second => cases hemits
  | seq_left hleft ih => exact ih hemits
  | branch_true htrue => cases hemits
  | branch_false hfalse => cases hemits
  | loop_true htrue => cases hemits
  | loop_false hfalse => cases hemits
  | par_done hdone => cases hemits
  | par_branch i hbranch ih => exact ih hemits
  | fork forkSite fresh childProgram =>
      cases hemits
      rfl

/-- Pure local execution evidence retained across several machine steps. -/
inductive LocalExecution [DecidableEq Lifeline]
    (A : Lifeline) (session : SessionName) :
    LocProg Lifeline Field ->
    Word (LocalLetter Lifeline ChannelScope) ->
    LocProg Lifeline Field -> Prop where
  | refl (program) : LocalExecution A session program [] program
  | tail
      {initial current residual : LocProg Lifeline Field}
      {word : Word (LocalLetter Lifeline ChannelScope)}
      {counters counters' : VisitCounters Lifeline}
      {queues queues' : Queues Lifeline}
      {emission : Emission Lifeline}
      {activation : Activation Lifeline Field}
      (hbefore : LocalExecution A session initial word current)
      (hstep : LocalStep A session current counters queues
        residual counters' queues' emission activation) :
      LocalExecution A session initial
        (word ++ emissionWord emission) residual

/-- A local operational execution can always be completed by any trace of its
    residual. This is the syntactic core of the session-prefix argument. -/
theorem localExecution_trace_decomposition [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {initial residual : LocProg Lifeline Field}
    {word : Word (LocalLetter Lifeline ChannelScope)}
    (hexecution : LocalExecution A session initial word residual) :
    ∀ {rest},
      ZipperGenParallel.LocProg.trace residual.eraseForks rest ->
      ZipperGenParallel.LocProg.trace initial.eraseForks (word ++ rest) := by
  induction hexecution with
  | refl =>
      intro rest hrest
      simpa using hrest
  | tail hbefore hstep ih =>
      intro rest hrest
      have hcurrent := localStep_trace_decomposition hstep hrest
      simpa [List.append_assoc] using ih hcurrent

/-- Every finite local machine execution is a prefix of a complete trace of
    the endpoint program from which it started. -/
theorem localExecution_prefixTrace [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {initial residual : LocProg Lifeline Field}
    {word : Word (LocalLetter Lifeline ChannelScope)}
    (hexecution : LocalExecution A session initial word residual) :
    ZipperGenParallel.LocProg.prefixTrace initial.eraseForks word := by
  rcases eraseForks_trace_exists residual with ⟨rest, hrest⟩
  exact ⟨word ++ rest,
    localExecution_trace_decomposition hexecution hrest,
    ⟨rest, rfl⟩⟩

end ZipperGenSpawn
