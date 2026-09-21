import ZipperGenSpawn.OperationalMSC

namespace ZipperGenSpawn

universe u v

/-- A spawn declaration occurring in a source choreography.  The relation
    remembers the declaration's globally unique site, freshness list, and
    body without introducing a partial lookup function. -/
inductive SpawnSpec :
    Prog Lifeline Field -> Occurrence -> List (Lifeline × Field) ->
      Prog Lifeline Field -> Prop where
  | here (site fresh body) :
      SpawnSpec (.spawn site fresh body) site fresh body
  | seq_left (h : SpawnSpec first site fresh body) :
      SpawnSpec (.seq first second) site fresh body
  | seq_right (h : SpawnSpec second site fresh body) :
      SpawnSpec (.seq first second) site fresh body
  | branch_then (h : SpawnSpec thenBranch target fresh body) :
      SpawnSpec (.branch site owner thenBranch elseBranch) target fresh body
  | branch_else (h : SpawnSpec elseBranch target fresh body) :
      SpawnSpec (.branch site owner thenBranch elseBranch) target fresh body
  | loop_body (h : SpawnSpec loopBody target fresh body) :
      SpawnSpec (.loop site owner loopBody exit) target fresh body
  | loop_exit (h : SpawnSpec exit target fresh body) :
      SpawnSpec (.loop site owner loopBody exit) target fresh body
  | par_branch (i : Fin n) (h : SpawnSpec (branches i) target fresh body) :
      SpawnSpec (.par site branches) target fresh body
  | spawn_body (h : SpawnSpec outerBody target fresh body) :
      SpawnSpec (.spawn site outerFresh outerBody) target fresh body

/-- Global uniqueness condition needed to read a spawn site as a source-body
    identifier.  `Prog.WellLabelled` is the syntactic condition intended to
    discharge it. -/
def SpawnSpecsFunctional (source : Prog Lifeline Field) : Prop :=
  forall site fresh₁ body₁ fresh₂ body₂,
    SpawnSpec source site fresh₁ body₁ ->
    SpawnSpec source site fresh₂ body₂ ->
      fresh₁ = fresh₂ /\ body₁ = body₂

/-- The source body named by a globally unique spawn occurrence.  The default
    is used only at names that do not denote a spawn in `source`. -/
noncomputable def spawnBodyAt
    (source : Prog Lifeline Field) (site : Occurrence) : Prog Lifeline Field := by
  classical
  exact if h : exists body fresh, SpawnSpec source site fresh body then
      Classical.choose h
    else
      .eps

theorem spawnBodyAt_spec
    (source : Prog Lifeline Field) (site : Occurrence)
    (fresh : List (Lifeline × Field)) (body : Prog Lifeline Field)
    (hspec : SpawnSpec source site fresh body)
    (hfunctional : SpawnSpecsFunctional source) :
    spawnBodyAt source site = body := by
  classical
  let hexists : exists candidate candidateFresh,
      SpawnSpec source site candidateFresh candidate := ⟨body, fresh, hspec⟩
  rw [spawnBodyAt, dif_pos hexists]
  let chosen := Classical.choose hexists
  have hchosenExists := Classical.choose_spec hexists
  let chosenFresh := Classical.choose hchosenExists
  have hchosen : SpawnSpec source site chosenFresh chosen :=
    Classical.choose_spec hchosenExists
  exact (hfunctional site chosenFresh chosen fresh body hchosen hspec).2

theorem mem_concatLists_of_mem
    {lists : List (List α)} {items : List α} {item : α}
    (hitems : items ∈ lists) (hitem : item ∈ items) :
    item ∈ Prog.concatLists lists := by
  induction lists with
  | nil => simp at hitems
  | cons head tail ih =>
      simp only [Prog.concatLists, List.foldr_cons, List.mem_cons] at hitems ⊢
      rcases hitems with rfl | hitems
      · exact List.mem_append_left _ hitem
      · exact List.mem_append_right _ (ih hitems)

theorem mem_finConcatMap
    {n : Nat} (items : Fin n -> List α) (i : Fin n) {item : α}
    (hitem : item ∈ items i) :
    item ∈ Prog.finConcatMap items := by
  apply mem_concatLists_of_mem
  · exact List.mem_ofFn.mpr ⟨i, rfl⟩
  · exact hitem

theorem finConcatMap_succ
    (items : Fin (n + 1) -> List α) :
    Prog.finConcatMap items =
      items 0 ++ Prog.finConcatMap (fun i : Fin n => items i.succ) := by
  simp [Prog.finConcatMap, Prog.concatLists, List.ofFn_succ]

theorem finConcatMap_nodup_each
    {n : Nat} (items : Fin n -> List α)
    (hnodup : (Prog.finConcatMap items).Nodup) (i : Fin n) :
    (items i).Nodup := by
  induction n with
  | zero => exact Fin.elim0 i
  | succ n ih =>
      rw [finConcatMap_succ] at hnodup
      have hparts := List.nodup_append.mp hnodup
      refine Fin.cases ?_ (fun j => ?_) i
      · exact hparts.1
      · exact ih (fun j : Fin n => items j.succ) hparts.2.1 j

theorem finConcatMap_nodup_cross
    {n : Nat} (items : Fin n -> List α)
    (hnodup : (Prog.finConcatMap items).Nodup)
    (i j : Fin n) (hij : i ≠ j) {item : α}
    (hi : item ∈ items i) (hj : item ∈ items j) : False := by
  induction n with
  | zero => exact Fin.elim0 i
  | succ n ih =>
      rw [finConcatMap_succ] at hnodup
      have hparts := List.nodup_append.mp hnodup
      refine Fin.cases
        (motive := fun i => forall j, i ≠ j ->
          item ∈ items i -> item ∈ items j -> False)
        ?_ (fun i' => ?_) i j hij hi hj
      · intro j hij hi hj
        refine Fin.cases
          (motive := fun j => (0 : Fin (n + 1)) ≠ j ->
            item ∈ items j -> False)
          ?_ (fun j' => ?_) j hij hj
        · intro hij hj
          exact hij rfl
        · intro hij hj
          have hjTail : item ∈
                Prog.finConcatMap (fun k : Fin n => items k.succ) :=
              mem_finConcatMap _ j' hj
          exact (hparts.2.2 item hi item hjTail) rfl
      · intro j hij hi hj
        refine Fin.cases
          (motive := fun j => i'.succ ≠ j -> item ∈ items j -> False)
          ?_ (fun j' => ?_) j hij hj
        · intro hij hj
          have hiTail : item ∈
                Prog.finConcatMap (fun k : Fin n => items k.succ) :=
              mem_finConcatMap _ i' hi
          exact (hparts.2.2 item hj item hiTail) rfl.symm
        · intro hij hj
          apply ih (fun k : Fin n => items k.succ) hparts.2.1 i' j'
              (by intro heq; subst j'; exact hij rfl) hi hj

theorem SpawnSpec.site_mem_occurrences
    (hspec : SpawnSpec source site fresh body) :
    site ∈ source.occurrences := by
  induction hspec with
  | here => simp [Prog.occurrences]
  | seq_left h ih => exact List.mem_append_left _ ih
  | seq_right h ih => exact List.mem_append_right _ ih
  | branch_then h ih =>
      simp [Prog.occurrences, ih]
  | branch_else h ih =>
      simp [Prog.occurrences, ih]
  | loop_body h ih =>
      simp [Prog.occurrences, ih]
  | loop_exit h ih =>
      simp [Prog.occurrences, ih]
  | par_branch i h ih =>
      simp only [Prog.occurrences, List.mem_cons]
      exact Or.inr (mem_finConcatMap _ i ih)
  | spawn_body h ih =>
      simp [Prog.occurrences, ih]

/-- Global uniqueness of all labelled control occurrences implies that a
    spawn label selects one freshness declaration and one body. -/
theorem wellLabelled_spawnSpecsFunctional
    (source : Prog Lifeline Field) (hwell : source.WellLabelled) :
    SpawnSpecsFunctional source := by
  intro target fresh₁ body₁ fresh₂ body₂ hleft hright
  induction source generalizing target fresh₁ body₁ fresh₂ body₂ with
  | eps => cases hleft
  | msg sender receiver hne => cases hleft
  | act owner => cases hleft
  | seq first second ihFirst ihSecond =>
      have hparts := List.nodup_append.mp hwell
      cases hleft with
      | seq_left hleft =>
          cases hright with
          | seq_left hright =>
              exact ihFirst hparts.1 target fresh₁ body₁ fresh₂ body₂ hleft hright
          | seq_right hright =>
              exact False.elim <| (hparts.2.2 target
                hleft.site_mem_occurrences target
                hright.site_mem_occurrences) rfl
      | seq_right hleft =>
          cases hright with
          | seq_left hright =>
              exact False.elim <| (hparts.2.2 target
                hright.site_mem_occurrences target
                hleft.site_mem_occurrences) rfl
          | seq_right hright =>
              exact ihSecond hparts.2.1 target fresh₁ body₁ fresh₂ body₂ hleft hright
  | branch site owner thenBranch elseBranch ihThen ihElse =>
      have htail := (List.nodup_cons.mp hwell).2
      have hparts := List.nodup_append.mp htail
      cases hleft with
      | branch_then hleft =>
          cases hright with
          | branch_then hright =>
              exact ihThen hparts.1 target fresh₁ body₁ fresh₂ body₂ hleft hright
          | branch_else hright =>
              exact False.elim <| (hparts.2.2 target
                hleft.site_mem_occurrences target
                hright.site_mem_occurrences) rfl
      | branch_else hleft =>
          cases hright with
          | branch_then hright =>
              exact False.elim <| (hparts.2.2 target
                hright.site_mem_occurrences target
                hleft.site_mem_occurrences) rfl
          | branch_else hright =>
              exact ihElse hparts.2.1 target fresh₁ body₁ fresh₂ body₂ hleft hright
  | loop site owner loopBody exit ihBody ihExit =>
      have htail := (List.nodup_cons.mp hwell).2
      have hparts := List.nodup_append.mp htail
      cases hleft with
      | loop_body hleft =>
          cases hright with
          | loop_body hright =>
              exact ihBody hparts.1 target fresh₁ body₁ fresh₂ body₂ hleft hright
          | loop_exit hright =>
              exact False.elim <| (hparts.2.2 target
                hleft.site_mem_occurrences target
                hright.site_mem_occurrences) rfl
      | loop_exit hleft =>
          cases hright with
          | loop_body hright =>
              exact False.elim <| (hparts.2.2 target
                hright.site_mem_occurrences target
                hleft.site_mem_occurrences) rfl
          | loop_exit hright =>
              exact ihExit hparts.2.1 target fresh₁ body₁ fresh₂ body₂ hleft hright
  | @par n site branches ih =>
      have htail := (List.nodup_cons.mp hwell).2
      cases hleft with
      | par_branch i hleft =>
          cases hright with
          | par_branch j hright =>
              by_cases hij : i = j
              · subst j
                exact ih i (finConcatMap_nodup_each
                    (fun k => (branches k).occurrences) htail i)
                  target fresh₁ body₁ fresh₂ body₂ hleft hright
              · exact False.elim <| finConcatMap_nodup_cross
                  (fun k => (branches k).occurrences) htail i j hij
                  hleft.site_mem_occurrences hright.site_mem_occurrences
  | spawn site outerFresh outerBody ih =>
      have hcons := List.nodup_cons.mp hwell
      cases hleft with
      | here =>
          cases hright with
          | here => exact ⟨rfl, rfl⟩
          | spawn_body hright =>
              exact False.elim (hcons.1 hright.site_mem_occurrences)
      | spawn_body hleft =>
          cases hright with
          | here => exact False.elim (hcons.1 hleft.site_mem_occurrences)
          | spawn_body hright =>
              exact ih hcons.2 target fresh₁ body₁ fresh₂ body₂ hleft hright

/-- Every local fork template in a program comes from the indicated global
    source choreography.  This property is deliberately stable under local
    reduction, so it can be attached to residual endpoint controls. -/
inductive ForkTemplatesFrom [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (source : Prog Lifeline Field) :
    LocProg Lifeline Field -> Prop where
  | eps : ForkTemplatesFrom order A source .eps
  | letter (letter) : ForkTemplatesFrom order A source (.letter letter)
  | seq (hfirst : ForkTemplatesFrom order A source first)
      (hsecond : ForkTemplatesFrom order A source second) :
      ForkTemplatesFrom order A source (.seq first second)
  | branch
      (hthen : ForkTemplatesFrom order A source thenBranch)
      (helse : ForkTemplatesFrom order A source elseBranch) :
      ForkTemplatesFrom order A source
        (.branch trueLetter falseLetter thenBranch elseBranch)
  | loop
      (hbody : ForkTemplatesFrom order A source body)
      (hexit : ForkTemplatesFrom order A source exit) :
      ForkTemplatesFrom order A source
        (.loop trueLetter falseLetter body exit)
  | par (hbranches : forall i, ForkTemplatesFrom order A source (branches i)) :
      ForkTemplatesFrom order A source (.par branches)
  | fork
      (hspec : SpawnSpec source site fresh body)
      (hchild : child = Prog.project order A .main body)
      (hfresh : localFresh = Prog.freshFor A fresh)
      (hnested : ForkTemplatesFrom order A source child) :
      ForkTemplatesFrom order A source (.fork site localFresh child)

namespace ForkTemplatesFrom

theorem liftSource [DecidableEq Lifeline]
    {order : List Lifeline} {A : Lifeline}
    {small large : Prog Lifeline Field} {program : LocProg Lifeline Field}
    (hlift : forall site fresh body,
      SpawnSpec small site fresh body -> SpawnSpec large site fresh body)
    (h : ForkTemplatesFrom order A small program) :
    ForkTemplatesFrom order A large program := by
  induction h with
  | eps => exact .eps
  | letter value => exact .letter value
  | seq hfirst hsecond ihFirst ihSecond => exact .seq ihFirst ihSecond
  | branch hthen helse ihThen ihElse => exact .branch ihThen ihElse
  | loop hbody hexit ihBody ihExit => exact .loop ihBody ihExit
  | par hbranches ih => exact .par ih
  | fork hspec hchild hfresh hnested ih =>
      exact .fork (hlift _ _ _ hspec) hchild hfresh ih

end ForkTemplatesFrom

theorem controlBroadcast_forkTemplatesFrom [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (source : Prog Lifeline Field)
    (targets : List Lifeline) (scope : ChannelScope)
    (site : Occurrence) (kind : ZipperGenParallel.ChoiceKind) :
    ForkTemplatesFrom order A source
      (Prog.controlBroadcast targets scope site kind) := by
  induction targets with
  | nil => exact .eps
  | cons head tail ih => exact .seq (.letter _) ih

theorem project_forkTemplatesFrom [DecidableEq Lifeline]
    (order : List Lifeline) (A : Lifeline) (scope : ChannelScope)
    (source : Prog Lifeline Field) :
    ForkTemplatesFrom order A source (Prog.project order A scope source) := by
  induction source generalizing scope with
  | eps => exact .eps
  | msg sender receiver hne =>
      by_cases hsender : A = sender
      · simp [Prog.project, hsender]
        exact .letter (ZipperGenParallel.LocalLetter.send receiver scope)
      · by_cases hreceiver : A = receiver
        · subst A
          simpa [Prog.project, Ne.symm hne] using
            (ForkTemplatesFrom.letter
              (order := order) (A := receiver)
              (source := Prog.msg sender receiver hne)
              (ZipperGenParallel.LocalLetter.recv sender scope))
        · simp [Prog.project, hsender, hreceiver]
          exact .eps
  | act owner =>
      simp only [Prog.project]
      split <;> first | exact .letter _ | exact .eps
  | seq first second ihFirst ihSecond =>
      exact .seq
        ((ihFirst scope).liftSource (fun _ _ _ h => .seq_left h))
        ((ihSecond scope).liftSource (fun _ _ _ h => .seq_right h))
  | branch site owner thenBranch elseBranch ihThen ihElse =>
      simp only [Prog.project]
      split
      · exact .branch
          (.seq
            (controlBroadcast_forkTemplatesFrom order A _ _ _ _ _)
            ((ihThen scope).liftSource
              (fun _ _ _ h => .branch_then h)))
          (.seq
            (controlBroadcast_forkTemplatesFrom order A _ _ _ _ _)
            ((ihElse scope).liftSource
              (fun _ _ _ h => .branch_else h)))
      · split
        · exact .branch
            ((ihThen scope).liftSource
              (fun _ _ _ h => .branch_then h))
            ((ihElse scope).liftSource
              (fun _ _ _ h => .branch_else h))
        · exact .eps
  | loop site owner body exit ihBody ihExit =>
      simp only [Prog.project]
      split
      · exact .loop
          (.seq
            (controlBroadcast_forkTemplatesFrom order A _ _ _ _ _)
            ((ihBody scope).liftSource
              (fun _ _ _ h => .loop_body h)))
          (.seq
            (controlBroadcast_forkTemplatesFrom order A _ _ _ _ _)
            ((ihExit scope).liftSource
              (fun _ _ _ h => .loop_exit h)))
      · split
        · exact .loop
            ((ihBody scope).liftSource
              (fun _ _ _ h => .loop_body h))
            ((ihExit scope).liftSource
              (fun _ _ _ h => .loop_exit h))
        · exact .eps
  | @par n site branches ih =>
      exact .par (fun i =>
        (ih i (.branch site (i.val + 1))).liftSource
          (fun _ _ _ h => .par_branch i h))
  | spawn site fresh body ih =>
      simp only [Prog.project]
      split
      · exact .fork (.here site fresh body) rfl rfl
          ((ih .main).liftSource (fun _ _ _ h => .spawn_body h))
      · exact .eps

/-- Local reduction never invents or changes a fork template. -/
theorem localStep_preserves_forkTemplatesFrom [DecidableEq Lifeline]
    {A : Lifeline} {session : SessionName}
    {program residual : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline}
    {activation : Activation Lifeline Field}
    {order : List Lifeline} {source : Prog Lifeline Field}
    (hsource : ForkTemplatesFrom order A source program)
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission activation) :
    ForkTemplatesFrom order A source residual := by
  induction hstep with
  | letter hletter => exact .eps
  | seq_done second => cases hsource with | seq _ hsecond => exact hsecond
  | seq_left hlocal ih =>
      cases hsource with
      | seq hfirst hsecond => exact .seq (ih hfirst) hsecond
  | branch_true hletter =>
      cases hsource with | branch hthen helse => exact hthen
  | branch_false hletter =>
      cases hsource with | branch hthen helse => exact helse
  | loop_true hletter =>
      cases hsource with
      | loop hbody hexit => exact .seq hbody (.loop hbody hexit)
  | loop_false hletter =>
      cases hsource with | loop hbody hexit => exact hexit
  | par_done hdone => exact .eps
  | par_branch i hlocal ih =>
      cases hsource with
      | par hbranches =>
          apply ForkTemplatesFrom.par
          intro j
          by_cases hji : j = i
          · subst j
            simpa using ih (hbranches i)
          · simp [hji]
            exact hbranches j
  | fork site fresh child => exact .eps

/-- If a local step activates a child, the installed template is exactly the
    projection of a spawn body occurring in the common source tree. -/
theorem localStep_activation_source [DecidableEq Lifeline]
    {A : Lifeline} {session childSession : SessionName}
    {program residual childProgram : LocProg Lifeline Field}
    {counters counters' : VisitCounters Lifeline}
    {queues queues' : Queues Lifeline}
    {emission : Emission Lifeline} {activation : Activation Lifeline Field}
    {order : List Lifeline} {source : Prog Lifeline Field}
    (hsource : ForkTemplatesFrom order A source program)
    (hstep : LocalStep A session program counters queues
      residual counters' queues' emission activation)
    (hactivation : activation = some (childSession, childProgram)) :
    exists site fresh body,
      emission = .fork session site childSession /\
      SpawnSpec source site fresh body /\
      childProgram = Prog.project order A .main body /\
      ForkTemplatesFrom order A source childProgram := by
  induction hstep with
  | letter hletter => cases hactivation
  | seq_done second => cases hactivation
  | seq_left hlocal ih =>
      cases hsource with | seq hfirst hsecond => exact ih hfirst hactivation
  | branch_true hletter => cases hactivation
  | branch_false hletter => cases hactivation
  | loop_true hletter => cases hactivation
  | loop_false hletter => cases hactivation
  | par_done hdone => cases hactivation
  | par_branch i hlocal ih =>
      cases hsource with | par hbranches => exact ih (hbranches i) hactivation
  | fork site localFresh child =>
      cases hactivation
      cases hsource with
      | fork hspec hchild hfresh hnested =>
          exact ⟨site, _, _, rfl, hspec, hchild, hnested⟩

/-- Every currently allocated local residual contains only fork templates
    originating in the initial source choreography. -/
def ControlsFromSource [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field) : Prop :=
  forall A session residual,
    configuration.controls A session = some residual ->
      ForkTemplatesFrom order A source residual

theorem initial_controlsFromSource [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    ControlsFromSource order source (initialConfiguration order source) := by
  intro A session residual hcontrol
  by_cases hroot : session = .root
  · subst session
    simp [initialConfiguration] at hcontrol
    subst residual
    exact project_forkTemplatesFrom order A .main source
  · simp [initialConfiguration, hroot] at hcontrol

theorem step_preserves_controlsFromSource [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    {before after : Configuration Lifeline Field}
    (hinvariant : ControlsFromSource order source before)
    (hstep : Step before after) :
    ControlsFromSource order source after := by
  intro B current currentProgram hafter
  cases hstep with
  | run A parentSession program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      have hprogram := hinvariant A parentSession program hcontrol
      cases activation with
      | none =>
          by_cases htarget : B = A /\ current = parentSession
          · rcases htarget with ⟨rfl, rfl⟩
            simp [setResidual, installActivation, setControl] at hafter
            subst currentProgram
            exact localStep_preserves_forkTemplatesFrom hprogram hlocal
          · have hold : before.controls B current = some currentProgram := by
              simpa [setResidual, installActivation, setControl, htarget]
                using hafter
            exact hinvariant B current currentProgram hold
      | some childAndProgram =>
          rcases childAndProgram with ⟨child, childProgram⟩
          have hactivation := localStep_activation_source hprogram hlocal rfl
          rcases hactivation with
            ⟨site, fresh, body, hemits, hspec, hchildProgram, hchildSource⟩
          by_cases hchild : B = A /\ current = child
          · rcases hchild with ⟨rfl, rfl⟩
            simp [setResidual, installActivation, setControl] at hafter
            subst currentProgram
            exact hchildSource
          · by_cases hparent : B = A /\ current = parentSession
            · rcases hparent with ⟨rfl, rfl⟩
              have hne := localStep_activation_child_ne_parent hlocal
              have hcurrent : current ≠ child := Ne.symm hne
              simp [setResidual, installActivation, setControl, hcurrent]
                at hafter
              subst currentProgram
              exact localStep_preserves_forkTemplatesFrom hprogram hlocal
            · have hold : before.controls B current = some currentProgram := by
                simpa [setResidual, installActivation, setControl,
                  hchild, hparent] using hafter
              exact hinvariant B current currentProgram hold

theorem reachable_controlsFromSource [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    forall configuration,
      Reachable (initialConfiguration order source) configuration ->
        ControlsFromSource order source configuration := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact initial_controlsFromSource order source
  | tail hbefore hstep ih =>
      exact step_preserves_controlsFromSource order source ih hstep

/-- Static source identity of the endpoint program with which one local
    session control was created. -/
inductive InitialControlOrigin [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) (A : Lifeline) :
    SessionName -> LocProg Lifeline Field -> Prop where
  | root : InitialControlOrigin order source A .root
      (Prog.projectMain order A source)
  | child
      (parent : SessionName) (site : Occurrence) (visit : Nat)
      (fresh : List (Lifeline × Field)) (body : Prog Lifeline Field)
      (hspec : SpawnSpec source site fresh body) :
      InitialControlOrigin order source A (.child parent site visit)
        (Prog.project order A .main body)

theorem InitialControlOrigin.child_info [DecidableEq Lifeline]
    {order : List Lifeline} {source : Prog Lifeline Field} {A : Lifeline}
    {session : SessionName} {initial : LocProg Lifeline Field}
    (horigin : InitialControlOrigin order source A session initial)
    (hroot : session ≠ .root) :
    exists parent site visit fresh body,
      session = .child parent site visit /\
      SpawnSpec source site fresh body /\
      initial = Prog.project order A .main body := by
  cases horigin with
  | root => exact False.elim (hroot rfl)
  | child parent site visit fresh body hspec =>
      exact ⟨parent, site, visit, fresh, body, rfl, hspec, rfl⟩

/-- Each allocated control is a residual of the projection selected by its
    source-level origin.  Unlike `ControlsHaveExecution`, this invariant also
    remembers which spawn body created a child session. -/
def ControlsHaveSourceExecution [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field) : Prop :=
  forall A session residual,
    configuration.controls A session = some residual ->
      exists initial,
        LocalExecution A session initial
          (projectedSessionWord session (configuration.history A)) residual /\
        InitialControlOrigin order source A session initial

theorem initial_controlsHaveSourceExecution [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    ControlsHaveSourceExecution order source
      (initialConfiguration order source) := by
  intro A session residual hcontrol
  by_cases hroot : session = .root
  · subst session
    simp [initialConfiguration] at hcontrol
    subst residual
    exact ⟨_, LocalExecution.refl _, .root⟩
  · simp [initialConfiguration, hroot] at hcontrol

theorem step_preserves_controlsHaveSourceExecution [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    {before after : Configuration Lifeline Field}
    (hevents : EventsHaveControls before)
    (htemplates : ControlsFromSource order source before)
    (hinvariant : ControlsHaveSourceExecution order source before)
    (hstep : Step before after) :
    ControlsHaveSourceExecution order source after := by
  intro B current currentProgram hafter
  cases hstep with
  | run A parent program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      have hprogramTemplates := htemplates A parent program hcontrol
      cases activation with
      | none =>
          by_cases htarget : B = A /\ current = parent
          · rcases htarget with ⟨rfl, rfl⟩
            simp [setResidual, installActivation, setControl] at hafter
            subst currentProgram
            rcases hinvariant B current program hcontrol with
              ⟨initial, hexecution, horigin⟩
            exact ⟨initial, by simpa using
              LocalExecution.tail hexecution hlocal, horigin⟩
          · have hold : before.controls B current = some currentProgram := by
              simpa [setResidual, installActivation, setControl, htarget]
                using hafter
            rcases hinvariant B current currentProgram hold with
              ⟨initial, hexecution, horigin⟩
            refine ⟨initial, ?_, horigin⟩
            by_cases hBA : B = A
            · subst B
              have hne : current ≠ parent := by
                intro heq
                exact htarget ⟨rfl, heq⟩
              cases emission <;>
                simpa [appendEmission, projectedSessionWord_append,
                  projectedSessionWord, hne, Ne.symm hne] using hexecution
            · simpa [projectedSessionWord_appendEmission_other _ hBA]
                using hexecution
      | some childAndProgram =>
          rcases childAndProgram with ⟨child, childProgram⟩
          rcases localStep_activation_source hprogramTemplates hlocal rfl with
            ⟨site, fresh, body, hemits, hspec, hchildProgram, hchildTemplates⟩
          have hchildName := localStep_fork_child hlocal hemits
          subst emission
          by_cases hchild : B = A /\ current = child
          · rcases hchild with ⟨rfl, rfl⟩
            simp [setResidual, installActivation, setControl] at hafter
            subst currentProgram
            have hnone : before.controls B current = none :=
              hfresh current childProgram rfl
            have hnoevent : forall letter,
                ProjectedSpawnLetter.event current letter ∉ before.history B := by
              intro letter hmem
              rcases hevents B current letter hmem with ⟨oldProgram, hold⟩
              rw [hnone] at hold
              cases hold
            have hempty :
                projectedSessionWord current (before.history B) = [] :=
              projectedSessionWord_eq_nil_of_no_event current _ hnoevent
            refine ⟨childProgram, ?_, ?_⟩
            · simpa [appendEmission, hempty] using
                (LocalExecution.refl childProgram)
            · rw [hchildProgram]
              rw [hchildName]
              exact .child parent site
                (nextVisit before.counters B parent site) fresh body hspec
          · by_cases hparent : B = A /\ current = parent
            · rcases hparent with ⟨rfl, rfl⟩
              have hne := localStep_activation_child_ne_parent hlocal
              have hcurrent : current ≠ child := Ne.symm hne
              simp [setResidual, installActivation, setControl, hcurrent]
                at hafter
              subst currentProgram
              rcases hinvariant B current program hcontrol with
                ⟨initial, hexecution, horigin⟩
              exact ⟨initial, by simpa [appendEmission] using
                (LocalExecution.tail hexecution hlocal), horigin⟩
            · have hold : before.controls B current = some currentProgram := by
                simpa [setResidual, installActivation, setControl,
                  hchild, hparent] using hafter
              rcases hinvariant B current currentProgram hold with
                ⟨initial, hexecution, horigin⟩
              refine ⟨initial, ?_, horigin⟩
              by_cases hBA : B = A
              · subst B
                simpa [appendEmission, projectedSessionWord] using hexecution
              · simpa [appendEmission, hBA] using hexecution

theorem reachable_controlsHaveSourceExecution [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    forall configuration,
      Reachable (initialConfiguration order source) configuration ->
        ControlsHaveSourceExecution order source configuration := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact initial_controlsHaveSourceExecution order source
  | @tail before after hbefore hstep ih =>
      exact step_preserves_controlsHaveSourceExecution order source
        (reachable_eventsHaveControls order source before hbefore)
        (reachable_controlsFromSource order source before hbefore) ih hstep

theorem reachable_control_source_prefix [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (hreachable : Reachable (initialConfiguration order source) configuration)
    {A : Lifeline} {session : SessionName}
    {residual : LocProg Lifeline Field}
    (hcontrol : configuration.controls A session = some residual) :
    exists initial,
      ZipperGenParallel.LocProg.prefixTrace initial.eraseForks
        (projectedSessionWord session (configuration.history A)) /\
      InitialControlOrigin order source A session initial := by
  rcases reachable_controlsHaveSourceExecution order source configuration
      hreachable A session residual hcontrol with
    ⟨initial, hexecution, horigin⟩
  exact ⟨initial, localExecution_prefixTrace hexecution, horigin⟩

end ZipperGenSpawn
