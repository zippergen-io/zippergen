import CompletionGoal

set_option maxRecDepth 8192
set_option maxHeartbeats 8000000

namespace ReviewDirect
open SupplierAccess
open SupplierAccess.Role

def specs : List (Guard × Role) :=
  [(Guard.need1, reviewer1), (Guard.need2, reviewer2),
   (Guard.need3, reviewer3), (Guard.need4, reviewer4),
   (Guard.need5, reviewer5), (Guard.need6, reviewer6),
   (Guard.need7, reviewer7), (Guard.need8, reviewer8),
   (Guard.need9, reviewer9), (Guard.need10, reviewer10),
   (Guard.need11, reviewer11), (Guard.need12, reviewer12),
   (Guard.need13, reviewer13), (Guard.need14, reviewer14),
   (Guard.need15, reviewer15), (Guard.need16, reviewer16)]

def agentStage (g : Guard) (r : Role) : Local agent :=
  LocProg.localIf g
    (((LocProg.send (Port.ctrl true) finalizer ;;ₗ
       (LocProg.send (Port.ctrl true) r ;;ₗ LocProg.eps)) ;;ₗ
      ((LocProg.send Port.review r ;;ₗ LocProg.eps) ;;ₗ LocProg.eps)))
    ((LocProg.send (Port.ctrl false) finalizer ;;ₗ
       (LocProg.send (Port.ctrl false) r ;;ₗ LocProg.eps)) ;;ₗ LocProg.eps)

def finalStage (r : Role) : Local finalizer :=
  LocProg.recvIf (Port.ctrl false) agent
    ((LocProg.eps ;;ₗ LocProg.eps) ;;ₗ LocProg.recv Port.verdict r)
    LocProg.eps

def agentCore : Local agent :=
  specs.foldl (fun p (g,r) => p ;;ₗ agentStage g r) (LocProg.send Port.plan worker)

def finalCore : Local finalizer :=
  specs.foldl (fun p (_,r) => p ;;ₗ finalStage r) LocProg.eps

theorem agent_endpoint : endpoints agent =
    (((agentCore ;;ₗ LocProg.eps) ;;ₗ LocProg.eps) ;;ₗ LocProg.eps) := by
  rfl

theorem final_endpoint : endpoints finalizer =
    (((finalCore ;;ₗ LocProg.eps) ;;ₗ LocProg.recv Port.prepared worker) ;;ₗ
      LocProg.act Port.unit Action.resolve Port.prepared) := by
  rfl

def agentStageWord (g : Guard) (r : Role) (hr : agent ≠ r) (b : Bool) : Word agent :=
  (if b then AlphabetOf.mkIfTrue agent g else AlphabetOf.mkIfFalse agent g) ::
    [AlphabetOf.mkSend agent (Port.ctrl b) finalizer (by decide),
     AlphabetOf.mkSend agent (Port.ctrl b) r hr] ++
    (if b then [AlphabetOf.mkSend agent Port.review r hr] else [])

def finalStageWord (r : Role) (b : Bool) : Word finalizer :=
  AlphabetOf.mkRecv finalizer (Port.ctrl b) agent ::
    (if b then [AlphabetOf.mkRecv finalizer Port.verdict r] else [])

theorem agent_stage_trace (g : Guard) (r : Role)
    (hr : r ≠ agent) (w : Word agent) :
    localTraceSemantics (agentStage g r) w ↔
      ∃ b : Bool, w = agentStageWord g r (Ne.symm hr) b := by
  constructor
  · intro h
    simp only [agentStage, localTraceSemantics] at h
    rcases h with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
    · rcases hu with ⟨u1,u2,hu1,hu2,rfl⟩
      rcases hu1 with ⟨v1,v2,hv1,hv2,rfl⟩
      rcases hv2 with ⟨x1,x2,hx1,hx2,rfl⟩
      rcases hu2 with ⟨v1,v2,hv1,hv2,rfl⟩
      rcases hv1 with ⟨x1,x2,hx1,hx2,rfl⟩
      rcases hv1 with ⟨hneq, rfl⟩
      rcases hx1 with ⟨hneq2, rfl⟩
      rcases hx1 with ⟨hneq3, rfl⟩
      exact ⟨true, by simp_all [agentStageWord]⟩
    · rcases hu with ⟨u1,u2,hu1,hu2,rfl⟩
      rcases hu1 with ⟨v1,v2,hv1,hv2,rfl⟩
      rcases hv2 with ⟨x1,x2,hx1,hx2,rfl⟩
      rcases hv1 with ⟨hneq, rfl⟩
      rcases hx1 with ⟨hneq2, rfl⟩
      exact ⟨false, by simp_all [agentStageWord]⟩
  · rintro ⟨b, rfl⟩
    cases b
    · simp [agentStage, agentStageWord, localTraceSemantics, hr]
      right
      exact Ne.symm hr
    · simp [agentStage, agentStageWord, localTraceSemantics, hr]
      refine ⟨[AlphabetOf.mkSend agent (Port.ctrl true) finalizer (by decide),
                AlphabetOf.mkSend agent (Port.ctrl true) r (Ne.symm hr)], ?_,
              [AlphabetOf.mkSend agent Port.review r (Ne.symm hr)], ?_, by simp⟩
      · refine ⟨[AlphabetOf.mkSend agent (Port.ctrl true) r (Ne.symm hr)], ?_, rfl⟩
        exact ⟨Ne.symm hr, rfl⟩
      · exact ⟨Ne.symm hr, rfl⟩

inductive StageTraces {α : Type} {a : Role} (f : α → Local a) :
    List α → List (Word a) → Prop where
  | nil : StageTraces f [] []
  | cons {x xs v vs} (hv : localTraceSemantics (f x) v)
      (hvs : StageTraces f xs vs) : StageTraces f (x :: xs) (v :: vs)

theorem fold_trace_iff {α : Type} {a : Role}
    (xs : List α) (p : Local a) (f : α → Local a) (w : Word a) :
    localTraceSemantics (xs.foldl (fun q x => q ;;ₗ f x) p) w ↔
      ∃ u vs, localTraceSemantics p u ∧
        StageTraces f xs vs ∧
        w = u ++ vs.flatten := by
  induction xs generalizing p w with
  | nil =>
      simp only [List.foldl_nil]
      constructor
      · intro h
        exact ⟨w, [], h, .nil, by simp⟩
      · rintro ⟨u, vs, hu, hvs, rfl⟩
        cases hvs
        simpa using hu
  | cons x xs ih =>
      simp only [List.foldl_cons]
      rw [ih]
      constructor
      · rintro ⟨u, vs, hu, hvs, rfl⟩
        rcases hu with ⟨u0, v0, hu0, hv0, rfl⟩
        exact ⟨u0, v0 :: vs, hu0, .cons hv0 hvs, by simp [List.append_assoc]⟩
      · rintro ⟨u, vs, hu, hvs, rfl⟩
        cases hvs with
        | cons hv hvs =>
            exact ⟨u ++ _, _, ⟨u, _, hu, hv, rfl⟩, hvs,
              by simp [List.append_assoc]⟩

def agentStageWord' (x : Guard × Role) (b : Bool) : Word agent :=
  if h : agent ≠ x.2 then agentStageWord x.1 x.2 h b else []

def agentWords (xs : List (Guard × Role)) (bs : List Bool) : List (Word agent) :=
  (xs.zip bs).map (fun (x,b) => agentStageWord' x b)

theorem agent_stage_traces_choices (xs : List (Guard × Role))
    (hvalid : ∀ x ∈ xs, x.2 ≠ agent) {vs : List (Word agent)}
    (hvs : StageTraces (fun x => agentStage x.1 x.2) xs vs) :
    ∃ bs : List Bool, bs.length = xs.length ∧ vs = agentWords xs bs := by
  induction hvs with
  | nil => exact ⟨[], rfl, rfl⟩
  | @cons x xs v vs hv hvs ih =>
      have hx : x.2 ≠ agent := hvalid x (List.mem_cons_self)
      have htail : ∀ y ∈ xs, y.2 ≠ agent := by
        intro y hy
        exact hvalid y (List.mem_cons_of_mem x hy)
      rcases agent_stage_trace x.1 x.2 hx v |>.mp hv with ⟨b, rfl⟩
      rcases ih htail with ⟨bs, hlen, rfl⟩
      exact ⟨b :: bs, by simp [hlen], by simp [agentWords, agentStageWord', Ne.symm hx]⟩

theorem specs_valid : ∀ x ∈ specs, x.2 ≠ agent := by
  intro x hx
  simp [specs] at hx
  rcases hx with h | h | h | h | h | h | h | h | h |
    h | h | h | h | h | h | h <;> subst x <;> decide

theorem agent_stage_traces_of_choices (xs : List (Guard × Role))
    (hvalid : ∀ x ∈ xs, x.2 ≠ agent) (bs : List Bool)
    (hlen : bs.length = xs.length) :
    StageTraces (fun x => agentStage x.1 x.2) xs (agentWords xs bs) := by
  induction xs generalizing bs with
  | nil =>
      cases bs with
      | nil => exact .nil
      | cons b bs => simp at hlen
  | cons x xs ih =>
      cases bs with
      | nil => simp at hlen
      | cons b bs =>
          have hx : x.2 ≠ agent := hvalid x (List.mem_cons_self)
          have ht : ∀ y ∈ xs, y.2 ≠ agent := by
            intro y hy
            exact hvalid y (List.mem_cons_of_mem x hy)
          have hlen' : bs.length = xs.length := by simpa using hlen
          simp only [agentWords, List.zip_cons_cons, List.map_cons]
          apply StageTraces.cons
          · exact (agent_stage_trace x.1 x.2 hx _).mpr
              ⟨b, by simp [agentStageWord', Ne.symm hx]⟩
          · simpa only [agentWords] using ih ht bs hlen'

def agentFull (bs : List Bool) : Word agent :=
  [AlphabetOf.mkSend agent Port.plan worker (by decide)] ++
    (agentWords specs bs).flatten

theorem agent_full_trace (bs : List Bool) (hlen : bs.length = specs.length) :
    localTraceSemantics (endpoints agent) (agentFull bs) := by
  have hcore : localTraceSemantics agentCore (agentFull bs) :=
    (fold_trace_iff specs (LocProg.send Port.plan worker)
      (fun x => agentStage x.1 x.2) (agentFull bs)).mpr
      ⟨[AlphabetOf.mkSend agent Port.plan worker (by decide)],
        agentWords specs bs, ⟨by decide, rfl⟩,
        agent_stage_traces_of_choices specs specs_valid bs hlen, rfl⟩
  rw [agent_endpoint]
  have h1 := localTraceSemantics_seq_intro agentCore LocProg.eps
    (agentFull bs) [] hcore localTraceSemantics_eps_nil
  have h2 := localTraceSemantics_seq_intro (agentCore ;;ₗ LocProg.eps) LocProg.eps
    (agentFull bs ++ []) [] h1 localTraceSemantics_eps_nil
  have h3 := localTraceSemantics_seq_intro
    ((agentCore ;;ₗ LocProg.eps) ;;ₗ LocProg.eps) LocProg.eps
    ((agentFull bs ++ []) ++ []) [] h2 localTraceSemantics_eps_nil
  simpa using h3

theorem agent_trace_choices (w : Word agent)
    (h : localTraceSemantics (endpoints agent) w) :
    ∃ bs : List Bool, bs.length = specs.length ∧ w = agentFull bs := by
  rw [agent_endpoint] at h
  rcases h with ⟨u1,u2,hu1,hu2,rfl⟩
  rcases hu1 with ⟨v1,v2,hv1,hv2,rfl⟩
  rcases hv1 with ⟨x1,x2,hx1,hx2,rfl⟩
  have hu2' : u2 = [] := hu2
  have hv2' : v2 = [] := hv2
  have hx2' : x2 = [] := hx2
  rcases (fold_trace_iff specs (LocProg.send Port.plan worker)
      (fun x => agentStage x.1 x.2) x1).mp hx1 with
    ⟨u,vs,hu,hvs,heq⟩
  rcases hu with ⟨hneq, rfl⟩
  rcases agent_stage_traces_choices specs specs_valid hvs with
    ⟨bs,hlen,rfl⟩
  exact ⟨bs, hlen, by simp [hu2',hv2',hx2',heq,agentFull]⟩

theorem final_stage_trace (r : Role) (b : Bool) :
    localTraceSemantics (finalStage r) (finalStageWord r b) := by
  cases b
  · simp [finalStage, finalStageWord, localTraceSemantics] <;> rfl
  · simp [finalStage, finalStageWord, localTraceSemantics] <;> rfl

def finalWords (xs : List (Guard × Role)) (bs : List Bool) : List (Word finalizer) :=
  (xs.zip bs).map (fun (x,b) => finalStageWord x.2 b)

theorem final_stage_traces (xs : List (Guard × Role)) (bs : List Bool)
    (hlen : bs.length = xs.length) :
    StageTraces (fun x => finalStage x.2) xs (finalWords xs bs) := by
  induction xs generalizing bs with
  | nil =>
      cases bs with
      | nil => exact .nil
      | cons b bs => simp at hlen
  | cons x xs ih =>
      cases bs with
      | nil => simp at hlen
      | cons b bs =>
          have hlen' : bs.length = xs.length := by simpa using hlen
          exact .cons (final_stage_trace x.2 b) (ih bs hlen')

def finalFull (bs : List Bool) : Word finalizer :=
  (finalWords specs bs).flatten ++
    [AlphabetOf.mkRecv finalizer Port.prepared worker,
     AlphabetOf.mkAct finalizer Port.unit Action.resolve Port.prepared]

theorem final_full_trace (bs : List Bool) (hlen : bs.length = specs.length) :
    localTraceSemantics (endpoints finalizer) (finalFull bs) := by
  have hcore : localTraceSemantics finalCore ((finalWords specs bs).flatten) :=
    (fold_trace_iff specs LocProg.eps (fun x => finalStage x.2)
      (finalWords specs bs).flatten).mpr
      ⟨[], finalWords specs bs, localTraceSemantics_eps_nil,
       final_stage_traces specs bs hlen, by simp⟩
  rw [final_endpoint]
  have h1 := localTraceSemantics_seq_intro finalCore LocProg.eps
    (finalWords specs bs).flatten [] hcore localTraceSemantics_eps_nil
  have hrecv : localTraceSemantics (LocProg.recv Port.prepared worker : Local finalizer)
      [AlphabetOf.mkRecv finalizer Port.prepared worker] := rfl
  have h2 := localTraceSemantics_seq_intro (finalCore ;;ₗ LocProg.eps)
    (LocProg.recv Port.prepared worker)
    ((finalWords specs bs).flatten ++ [])
    [AlphabetOf.mkRecv finalizer Port.prepared worker] h1 hrecv
  have hact : localTraceSemantics (LocProg.act Port.unit Action.resolve Port.prepared : Local finalizer)
      [AlphabetOf.mkAct finalizer Port.unit Action.resolve Port.prepared] := rfl
  have h3 := localTraceSemantics_seq_intro
    ((finalCore ;;ₗ LocProg.eps) ;;ₗ LocProg.recv Port.prepared worker)
    (LocProg.act Port.unit Action.resolve Port.prepared)
    (((finalWords specs bs).flatten ++ []) ++
      [AlphabetOf.mkRecv finalizer Port.prepared worker])
    [AlphabetOf.mkAct finalizer Port.unit Action.resolve Port.prepared] h2 hact
  simpa [finalFull, List.append_assoc] using h3

def reviewerFull (r : Role) (hr : r ≠ finalizer) (b : Bool) : Word r :=
  AlphabetOf.mkRecv r (Port.ctrl b) agent ::
    (if b then
      [AlphabetOf.mkRecv r Port.review agent,
       AlphabetOf.mkAct r Port.unit Action.assess Port.review,
       AlphabetOf.mkSend r Port.verdict finalizer hr]
     else [])

theorem reviewer1_full_trace (b : Bool) :
    localTraceSemantics (endpoints reviewer1)
      (reviewerFull reviewer1 (by decide) b) := by
  cases b <;> simp [endpoints, reviewerFull, localTraceSemantics] <;> rfl

def reviewerRoles : List Role :=
  [reviewer1, reviewer2, reviewer3, reviewer4, reviewer5, reviewer6,
   reviewer7, reviewer8, reviewer9, reviewer10, reviewer11, reviewer12,
   reviewer13, reviewer14, reviewer15, reviewer16]

theorem reviewers_full_trace (r : Role) (hr : r ∈ reviewerRoles)
    (hne : r ≠ finalizer) (b : Bool) :
    localTraceSemantics (endpoints r) (reviewerFull r hne b) := by
  simp [reviewerRoles] at hr
  rcases hr with h | h | h | h | h | h | h | h | h |
    h | h | h | h | h | h | h <;> subst r <;>
    cases b <;> simp [endpoints, reviewerFull, localTraceSemantics] <;> rfl

def workerFull : Word worker :=
  [AlphabetOf.mkRecv worker Port.plan agent,
   AlphabetOf.mkAct worker Port.prepared Action.prepare Port.plan,
   AlphabetOf.mkSend worker Port.prepared finalizer (by decide)]

theorem worker_full_trace :
    localTraceSemantics (endpoints worker) workerFull := by
  simp [endpoints, workerFull, localTraceSemantics]

def stageTuple (g : Guard) (r : Role) (hA : agent ≠ r)
    (hF : r ≠ finalizer) (b : Bool) : Tuple := fun a =>
  if ha : a = agent then ha ▸ agentStageWord g r hA b
  else if hf : a = finalizer then hf ▸ finalStageWord r b
  else if hr : a = r then hr ▸ reviewerFull r hF b
  else []

def stageCanonical (g : Guard) (r : Role) (hA : agent ≠ r)
    (hF : r ≠ finalizer) (b : Bool) : Tuple :=
  (if b then mscIfTrue g agent else mscIfFalse g agent) ∘ₘ
  mscMsg agent (Port.ctrl b) finalizer (Port.ctrl b) (by decide) ∘ₘ
  mscMsg agent (Port.ctrl b) r (Port.ctrl b) hA ∘ₘ
  (if b then
    mscMsg agent Port.review r Port.review hA ∘ₘ
    mscAct r Port.unit Action.assess Port.review ∘ₘ
    mscMsg r Port.verdict finalizer Port.verdict hF
   else WordTuple.empty)

theorem stage_canonical_complete (g : Guard) (r : Role)
    (hA : agent ≠ r) (hF : r ≠ finalizer) (b : Bool) :
    IsCompleteMSC (stageCanonical g r hA hF b) := by
  have hc (A B : Role) (p : Port) (h : A ≠ B) :
      IsCompleteMSC (mscMsg (C := Guard) (F := Action) A p B p h) :=
    mscMsg_isCompleteMSC A p B p h rfl
  have hchoice : IsCompleteMSC
      (if b then (mscIfTrue g agent : Tuple) else mscIfFalse g agent) := by
    cases b
    · exact mscIfFalse_isCompleteMSC g agent
    · exact mscIfTrue_isCompleteMSC g agent
  have hlast : IsCompleteMSC
      (if b then
        mscMsg agent Port.review r Port.review hA ∘ₘ
        mscAct r Port.unit Action.assess Port.review ∘ₘ
        mscMsg r Port.verdict finalizer Port.verdict hF
       else (WordTuple.empty : Tuple)) := by
    cases b
    · simpa using (mscEmpty_isCompleteMSC (L := Role) (C := Guard)
        (F := Action) (Payload := Port))
    · exact concat_complete_complete _ _
        (concat_complete_complete _ _ (hc agent r Port.review hA)
          (mscAct_isCompleteMSC r Port.unit Action.assess Port.review))
        (hc r finalizer Port.verdict hF)
  exact concat_complete_complete _ _
    (concat_complete_complete _ _
      (concat_complete_complete _ _ hchoice (hc agent finalizer (Port.ctrl b) (by decide)))
      (hc agent r (Port.ctrl b) hA)) hlast

theorem stageTuple_eq_canonical (g : Guard) (r : Role)
    (hA : agent ≠ r) (hF : r ≠ finalizer) (b : Bool) :
    stageTuple g r hA hF b = stageCanonical g r hA hF b := by
  funext a
  by_cases ha : a = agent
  · subst a
    cases b <;> simp [stageTuple, stageCanonical, agentStageWord, finalStageWord,
      reviewerFull, WordTuple.concat, WordTuple.empty, mscIfTrue,
      mscIfFalse, mscChoice, mscMsg, mscAct, singletonWord,
      choiceIfTrue, choiceIfFalse, hA, hF, List.append_assoc]
  · by_cases hf : a = finalizer
    · subst a
      cases b <;> simp [stageTuple, stageCanonical, agentStageWord, finalStageWord,
        reviewerFull, WordTuple.concat, WordTuple.empty, mscIfTrue,
        mscIfFalse, mscChoice, mscMsg, mscAct, singletonWord,
        choiceIfTrue, choiceIfFalse, hA, hF, Ne.symm hF, List.append_assoc]
    · by_cases hr : a = r
      · subst a
        cases b <;> simp [stageTuple, stageCanonical, agentStageWord, finalStageWord,
          reviewerFull, WordTuple.concat, WordTuple.empty, mscIfTrue,
          mscIfFalse, mscChoice, mscMsg, mscAct, singletonWord,
          choiceIfTrue, choiceIfFalse, hA, Ne.symm hA, hF, List.append_assoc]
      · cases b <;> simp [stageTuple, stageCanonical, agentStageWord, finalStageWord,
          reviewerFull, WordTuple.concat, WordTuple.empty, mscIfTrue,
          mscIfFalse, mscChoice, mscMsg, mscAct, singletonWord,
          choiceIfTrue, choiceIfFalse, ha, hf, hr, List.append_assoc]

theorem stageTuple_complete (g : Guard) (r : Role)
    (hA : agent ≠ r) (hF : r ≠ finalizer) (b : Bool) :
    IsCompleteMSC (stageTuple g r hA hF b) := by
  rw [stageTuple_eq_canonical]
  exact stage_canonical_complete g r hA hF b

def stageTuple' (x : Guard × Role) (b : Bool) : Tuple :=
  if hA : agent ≠ x.2 then
    if hF : x.2 ≠ finalizer then stageTuple x.1 x.2 hA hF b
    else WordTuple.empty
  else WordTuple.empty

theorem stageTuple'_complete (x : Guard × Role) (b : Bool) :
    IsCompleteMSC (stageTuple' x b) := by
  unfold stageTuple'
  split
  · split
    · exact stageTuple_complete x.1 x.2 ‹agent ≠ x.2› ‹x.2 ≠ finalizer› b
    · exact mscEmpty_isCompleteMSC
  · exact mscEmpty_isCompleteMSC

def baseMSC : Tuple :=
  mscMsg agent Port.plan worker Port.plan (by decide) ∘ₘ
    mscAct worker Port.prepared Action.prepare Port.plan

def tailMSC : Tuple :=
  mscMsg worker Port.prepared finalizer Port.prepared (by decide) ∘ₘ
    mscAct finalizer Port.unit Action.resolve Port.prepared

theorem baseMSC_complete : IsCompleteMSC baseMSC := by
  exact concat_complete_complete _ _
    (mscMsg_isCompleteMSC agent Port.plan worker Port.plan (by decide) rfl)
    (mscAct_isCompleteMSC worker Port.prepared Action.prepare Port.plan)

theorem tailMSC_complete : IsCompleteMSC tailMSC := by
  exact concat_complete_complete _ _
    (mscMsg_isCompleteMSC worker Port.prepared finalizer Port.prepared (by decide) rfl)
    (mscAct_isCompleteMSC finalizer Port.unit Action.resolve Port.prepared)

def coreMSC (bs : List Bool) : Tuple :=
  WordTuple.concatList ((specs.zip bs).map (fun (x,b) => stageTuple' x b))

def fullMSC (bs : List Bool) : Tuple := baseMSC ∘ₘ coreMSC bs ∘ₘ tailMSC

theorem coreMSC_complete (bs : List Bool) : IsCompleteMSC (coreMSC bs) := by
  unfold coreMSC
  apply concatList_complete
  intro M hM
  rcases List.mem_map.mp hM with ⟨⟨x,b⟩, _, rfl⟩
  exact stageTuple'_complete x b

theorem fullMSC_complete (bs : List Bool) : IsCompleteMSC (fullMSC bs) := by
  exact concat_complete_complete _ _
    (concat_complete_complete _ _ baseMSC_complete (coreMSC_complete bs))
    tailMSC_complete

theorem concatList_component (Ms : List Tuple) (a : Role) :
    WordTuple.concatList Ms a = (Ms.map (fun T => T a)).flatten := by
  have foldl_append {α : Type} (ws : List (List α)) (acc : List α) :
      ws.foldl (· ++ ·) acc = acc ++ ws.flatten := by
    induction ws generalizing acc with
    | nil => simp
    | cons w ws ih =>
        simp only [List.foldl_cons, List.flatten_cons]
        rw [ih]
        simp [List.append_assoc]
  induction Ms with
  | nil => rfl
  | cons T Ms ih =>
      simp [WordTuple.concatList_cons, WordTuple.concat, ih,
        concatLocalWords, foldl_append]

theorem specs_not_finalizer : ∀ x ∈ specs, x.2 ≠ finalizer := by
  intro x hx
  simp [specs] at hx
  rcases hx with h | h | h | h | h | h | h | h | h |
    h | h | h | h | h | h | h <;> subst x <;> decide

theorem coreMSC_agent (bs : List Bool) :
    coreMSC bs agent = (agentWords specs bs).flatten := by
  unfold coreMSC
  rw [concatList_component]
  simp only [List.map_map, agentWords]
  congr 1
  apply List.map_congr_left
  intro z hz
  rcases z with ⟨x,b⟩
  have hx : x ∈ specs := (List.of_mem_zip hz).1
  have hA : agent ≠ x.2 := Ne.symm (specs_valid x hx)
  have hF : x.2 ≠ finalizer := specs_not_finalizer x hx
  simp [stageTuple', stageTuple, agentStageWord', hA, hF]

theorem coreMSC_finalizer (bs : List Bool) :
    coreMSC bs finalizer = (finalWords specs bs).flatten := by
  unfold coreMSC
  rw [concatList_component]
  simp only [List.map_map, finalWords]
  congr 1
  apply List.map_congr_left
  intro z hz
  rcases z with ⟨x,b⟩
  have hx : x ∈ specs := (List.of_mem_zip hz).1
  have hA : agent ≠ x.2 := Ne.symm (specs_valid x hx)
  have hF : x.2 ≠ finalizer := specs_not_finalizer x hx
  simp [stageTuple', stageTuple, hA, hF, Ne.symm hF]
theorem bits_shape (bs : List Bool) (hlen : bs.length = specs.length) :
  ∃ b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 : Bool, bs = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16] := by
  have hn : bs.length = 16 := by simpa [specs] using hlen
  cases bs with
  | nil => simp at hn
  | cons b1 tail1 =>
      cases tail1 with
      | nil => simp at hn
      | cons b2 tail2 =>
          cases tail2 with
          | nil => simp at hn
          | cons b3 tail3 =>
              cases tail3 with
              | nil => simp at hn
              | cons b4 tail4 =>
                  cases tail4 with
                  | nil => simp at hn
                  | cons b5 tail5 =>
                      cases tail5 with
                      | nil => simp at hn
                      | cons b6 tail6 =>
                          cases tail6 with
                          | nil => simp at hn
                          | cons b7 tail7 =>
                              cases tail7 with
                              | nil => simp at hn
                              | cons b8 tail8 =>
                                  cases tail8 with
                                  | nil => simp at hn
                                  | cons b9 tail9 =>
                                      cases tail9 with
                                      | nil => simp at hn
                                      | cons b10 tail10 =>
                                          cases tail10 with
                                          | nil => simp at hn
                                          | cons b11 tail11 =>
                                              cases tail11 with
                                              | nil => simp at hn
                                              | cons b12 tail12 =>
                                                  cases tail12 with
                                                  | nil => simp at hn
                                                  | cons b13 tail13 =>
                                                      cases tail13 with
                                                      | nil => simp at hn
                                                      | cons b14 tail14 =>
                                                          cases tail14 with
                                                          | nil => simp at hn
                                                          | cons b15 tail15 =>
                                                              cases tail15 with
                                                              | nil => simp at hn
                                                              | cons b16 rest =>
                                                                  cases rest with
                                                                  | nil =>
                                                                      exact ⟨b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, rfl⟩
                                                                  | cons x xs => simp at hn

def bitAt (bs : List Bool) (i : Nat) : Bool := bs[i]?.getD false

def completeTuple (bs : List Bool) : Tuple
  | agent => agentFull bs
  | worker => workerFull
  | finalizer => finalFull bs
  | reviewer1 => reviewerFull reviewer1 (by decide) (bitAt bs 0)
  | reviewer2 => reviewerFull reviewer2 (by decide) (bitAt bs 1)
  | reviewer3 => reviewerFull reviewer3 (by decide) (bitAt bs 2)
  | reviewer4 => reviewerFull reviewer4 (by decide) (bitAt bs 3)
  | reviewer5 => reviewerFull reviewer5 (by decide) (bitAt bs 4)
  | reviewer6 => reviewerFull reviewer6 (by decide) (bitAt bs 5)
  | reviewer7 => reviewerFull reviewer7 (by decide) (bitAt bs 6)
  | reviewer8 => reviewerFull reviewer8 (by decide) (bitAt bs 7)
  | reviewer9 => reviewerFull reviewer9 (by decide) (bitAt bs 8)
  | reviewer10 => reviewerFull reviewer10 (by decide) (bitAt bs 9)
  | reviewer11 => reviewerFull reviewer11 (by decide) (bitAt bs 10)
  | reviewer12 => reviewerFull reviewer12 (by decide) (bitAt bs 11)
  | reviewer13 => reviewerFull reviewer13 (by decide) (bitAt bs 12)
  | reviewer14 => reviewerFull reviewer14 (by decide) (bitAt bs 13)
  | reviewer15 => reviewerFull reviewer15 (by decide) (bitAt bs 14)
  | reviewer16 => reviewerFull reviewer16 (by decide) (bitAt bs 15)

theorem complete_local_traces (bs : List Bool)
    (hlen : bs.length = specs.length) :
    ∀ a, localTraceSemantics (endpoints a) (completeTuple bs a) := by
  intro a
  cases a
  · exact agent_full_trace bs hlen
  · exact worker_full_trace
  · exact final_full_trace bs hlen
  · exact reviewers_full_trace reviewer1 (by decide) (by decide) (bitAt bs 0)
  · exact reviewers_full_trace reviewer2 (by decide) (by decide) (bitAt bs 1)
  · exact reviewers_full_trace reviewer3 (by decide) (by decide) (bitAt bs 2)
  · exact reviewers_full_trace reviewer4 (by decide) (by decide) (bitAt bs 3)
  · exact reviewers_full_trace reviewer5 (by decide) (by decide) (bitAt bs 4)
  · exact reviewers_full_trace reviewer6 (by decide) (by decide) (bitAt bs 5)
  · exact reviewers_full_trace reviewer7 (by decide) (by decide) (bitAt bs 6)
  · exact reviewers_full_trace reviewer8 (by decide) (by decide) (bitAt bs 7)
  · exact reviewers_full_trace reviewer9 (by decide) (by decide) (bitAt bs 8)
  · exact reviewers_full_trace reviewer10 (by decide) (by decide) (bitAt bs 9)
  · exact reviewers_full_trace reviewer11 (by decide) (by decide) (bitAt bs 10)
  · exact reviewers_full_trace reviewer12 (by decide) (by decide) (bitAt bs 11)
  · exact reviewers_full_trace reviewer13 (by decide) (by decide) (bitAt bs 12)
  · exact reviewers_full_trace reviewer14 (by decide) (by decide) (bitAt bs 13)
  · exact reviewers_full_trace reviewer15 (by decide) (by decide) (bitAt bs 14)
  · exact reviewers_full_trace reviewer16 (by decide) (by decide) (bitAt bs 15)

theorem fullMSC_eq_completeTuple (bs : List Bool)
    (hlen : bs.length = specs.length) : fullMSC bs = completeTuple bs := by
  rcases bits_shape bs hlen with ⟨b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, rfl⟩
  funext a
  cases a <;> simp [fullMSC, baseMSC, tailMSC, coreMSC,
    concatList_component, completeTuple, agentFull, finalFull,
    agentWords, finalWords, stageTuple', stageTuple, agentStageWord',
    agentStageWord, finalStageWord, reviewerFull, workerFull, bitAt,
    specs, WordTuple.concat, WordTuple.empty, concatLocalWords,
    mscMsg, mscAct, singletonWord,
    List.append_assoc]

theorem zip_eq_take {α : Type} (xs ys : List α)
    (hlen : ys.length ≤ xs.length)
    (hmatch : ∀ p ∈ xs.zip ys, p.1 = p.2) :
    ys = xs.take ys.length := by
  induction ys generalizing xs with
  | nil => simp
  | cons y ys ih =>
      cases xs with
      | nil => simp at hlen
      | cons x xs =>
          have hxy : x = y := hmatch (x,y) (by simp)
          have htail : ∀ p ∈ xs.zip ys, p.1 = p.2 := by
            intro p hp
            exact hmatch p (by simp [hp])
          have hlen' : ys.length ≤ xs.length := by simpa using hlen
          subst x
          simpa using congrArg (List.cons y) (ih xs hlen' htail)

theorem msc_payload_prefix (M : Tuple) (hM : IsMSC M) (A B : Role) :
    recvPayloads A (M B) <+: sendPayloads B (M A) := by
  have hlen : (recvPayloads A (M B)).length ≤ (sendPayloads B (M A)).length := by
    simpa [sndCount, rcvCount] using hM.noUnmatchedRecv A B
  have hmatch : ∀ p ∈ (sendPayloads B (M A)).zip (recvPayloads A (M B)),
      p.1 = p.2 := by
    intro p hp
    exact hM.labelCompat A B p hp
  have heq := zip_eq_take _ _ hlen hmatch
  exact ⟨(sendPayloads B (M A)).drop (recvPayloads A (M B)).length,
    by
      calc
        recvPayloads A (M B) ++
            (sendPayloads B (M A)).drop (recvPayloads A (M B)).length =
          (sendPayloads B (M A)).take (recvPayloads A (M B)).length ++
            (sendPayloads B (M A)).drop (recvPayloads A (M B)).length := by
          congr 1
        _ = sendPayloads B (M A) := List.take_append_drop _ _
  ⟩

theorem agent_send_finalizer (bs : List Bool)
    (hlen : bs.length = specs.length) :
    sendPayloads finalizer (agentFull bs) = bs.map Port.ctrl := by
  have hstage (g : Guard) (r : Role) (hA : agent ≠ r)
      (hF : r ≠ finalizer) (b : Bool) :
      sendPayloads finalizer (agentStageWord g r hA b) = [Port.ctrl b] := by
    cases b <;> simp [agentStageWord, sendPayloads, AlphabetOf.mkSend,
      AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse, hF]
  have hflat (ws : List (Word agent)) :
      sendPayloads finalizer ws.flatten =
        (ws.map (sendPayloads finalizer)).flatten := by
    induction ws with
    | nil => rfl
    | cons w ws ih => simp [ih]
  rcases bits_shape bs hlen with ⟨b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, rfl⟩
  simp [agentFull, sendPayloads_append, hflat, agentWords,
    agentStageWord', specs, hstage, sendPayloads, AlphabetOf.mkSend]

theorem final_inputs_prefix (M : Tuple) (hM : IsMSC M)
    (bs : List Bool) (hlen : bs.length = specs.length)
    (hagent : IsPrefixWord (M agent) (agentFull bs)) :
    recvPayloads agent (M finalizer) <+: bs.map Port.ctrl := by
  have hqueue := msc_payload_prefix M hM agent finalizer
  rcases hagent with ⟨tail, ht⟩
  have hsend : sendPayloads finalizer (M agent) <+:
      sendPayloads finalizer (agentFull bs) := by
    refine ⟨sendPayloads finalizer tail, ?_⟩
    simp [ht]
  have h := hqueue.trans hsend
  simpa [agent_send_finalizer bs hlen] using h

theorem reviewer_trace_choices (r : Role) (hr : r ∈ reviewerRoles)
    (hne : r ≠ finalizer) (w : Word r)
    (h : localTraceSemantics (endpoints r) w) :
    ∃ b : Bool, w = reviewerFull r hne b := by
  simp [reviewerRoles] at hr
  rcases hr with h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 |
    h10 | h11 | h12 | h13 | h14 | h15 | h16 <;> subst r <;>
    simp [endpoints, reviewerFull, localTraceSemantics] at h ⊢ <;>
    rcases h with ht | hf
  all_goals first | (right; simpa using ht) | (left; simpa using hf)

def reviewerIndex : Role → Nat
  | reviewer1 => 0
  | reviewer2 => 1
  | reviewer3 => 2
  | reviewer4 => 3
  | reviewer5 => 4
  | reviewer6 => 5
  | reviewer7 => 6
  | reviewer8 => 7
  | reviewer9 => 8
  | reviewer10 => 9
  | reviewer11 => 10
  | reviewer12 => 11
  | reviewer13 => 12
  | reviewer14 => 13
  | reviewer15 => 14
  | reviewer16 => 15
  | _ => 0

theorem agent_send_reviewer (bs : List Bool)
    (hlen : bs.length = specs.length) (r : Role)
    (hr : r ∈ reviewerRoles) :
    sendPayloads r (agentFull bs) =
      Port.ctrl (bitAt bs (reviewerIndex r)) ::
        (if bitAt bs (reviewerIndex r) then [Port.review] else []) := by
  have hrne : r ≠ finalizer := by
    cases r <;> simp [reviewerRoles] at hr ⊢
  have hstage (g : Guard) (q : Role) (hA : agent ≠ q)
      (b : Bool) :
      sendPayloads r (agentStageWord g q hA b) =
        (if q = r then Port.ctrl b :: (if b then [Port.review] else []) else []) := by
    cases b <;> by_cases h : q = r <;>
      simp [agentStageWord, sendPayloads, AlphabetOf.mkSend,
        AlphabetOf.mkIfTrue, AlphabetOf.mkIfFalse, h, hrne, Ne.symm hrne]
  have hflat (target : Role) (ws : List (Word agent)) :
      sendPayloads target ws.flatten =
        (ws.map (sendPayloads target)).flatten := by
    induction ws with
    | nil => rfl
    | cons w ws ih => simp [ih]
  rcases bits_shape bs hlen with ⟨b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, rfl⟩
  simp [reviewerRoles] at hr
  rcases hr with h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | h13 | h14 | h15 | h16 <;> subst r <;>
    simp [agentFull, hflat, agentWords, agentStageWord',
      specs, hstage, sendPayloads, AlphabetOf.mkSend,
      reviewerIndex, bitAt]

theorem reviewer_inputs_prefix (M : Tuple) (hM : IsMSC M)
    (bs : List Bool) (hlen : bs.length = specs.length)
    (hagent : IsPrefixWord (M agent) (agentFull bs))
    (r : Role) (hr : r ∈ reviewerRoles) :
    recvPayloads agent (M r) <+:
      Port.ctrl (bitAt bs (reviewerIndex r)) ::
        (if bitAt bs (reviewerIndex r) then [Port.review] else []) := by
  have hqueue := msc_payload_prefix M hM agent r
  rcases hagent with ⟨tail, ht⟩
  have hsend : sendPayloads r (M agent) <+:
      sendPayloads r (agentFull bs) := by
    refine ⟨sendPayloads r tail, ?_⟩
    simp [ht]
  have h := hqueue.trans hsend
  simpa [agent_send_reviewer bs hlen r hr] using h

theorem reviewer_prefix_transfer (M : Tuple) (hM : IsMSC M)
    (bs : List Bool) (hlen : bs.length = specs.length)
    (hagent : IsPrefixWord (M agent) (agentFull bs))
    (r : Role) (hr : r ∈ reviewerRoles) (hne : r ≠ finalizer)
    (hlocal : localPrefixSemantics (endpoints r) (M r)) :
    IsPrefixWord (M r) (reviewerFull r hne (bitAt bs (reviewerIndex r))) := by
  rcases hlocal with ⟨v, hv, hpref⟩
  rcases reviewer_trace_choices r hr hne v hv with ⟨c, rfl⟩
  cases hw : M r with
  | nil =>
      exact ⟨reviewerFull r hne (bitAt bs (reviewerIndex r)), by simp [hw]⟩
  | cons hd tl =>
      have hhead : hd = AlphabetOf.mkRecv r (Port.ctrl c) agent := by
        rcases hpref with ⟨suffix, heq⟩
        have hh := congrArg List.head? heq
        simpa [hw, reviewerFull] using hh.symm
      have hp := reviewer_inputs_prefix M hM bs hlen hagent r hr
      rcases hp with ⟨ptail, hpe⟩
      have hbit : c = bitAt bs (reviewerIndex r) := by
        have hh := congrArg List.head? hpe
        simp [hw, hhead, recvPayloads, AlphabetOf.mkRecv] at hh
        exact hh
      simpa [hw, hbit] using hpref

theorem final_stage_trace_choices (r : Role) (w : Word finalizer)
    (h : localTraceSemantics (finalStage r) w) :
    ∃ b : Bool, w = finalStageWord r b := by
  simp only [finalStage, localTraceSemantics] at h
  rcases h with ⟨u,hu,rfl⟩ | ⟨u,hu,rfl⟩
  · rcases hu with ⟨u1,u2,hu1,hu2,rfl⟩
    rcases hu1 with ⟨v1,v2,hv1,hv2,rfl⟩
    have hv1' : v1 = [] := hv1
    have hv2' : v2 = [] := hv2
    have hu2' : u2 = [AlphabetOf.mkRecv finalizer Port.verdict r] := hu2
    exact ⟨true, by simp [finalStageWord, hv1', hv2', hu2']; rfl⟩
  · have hu' : u = [] := hu
    exact ⟨false, by simp [finalStageWord, hu']; rfl⟩

theorem final_stage_traces_choices (xs : List (Guard × Role))
    {vs : List (Word finalizer)}
    (hvs : StageTraces (fun x => finalStage x.2) xs vs) :
    ∃ bs : List Bool, bs.length = xs.length ∧ vs = finalWords xs bs := by
  induction hvs with
  | nil => exact ⟨[], rfl, rfl⟩
  | @cons x xs v vs hv hvs ih =>
      rcases final_stage_trace_choices x.2 v hv with ⟨b, rfl⟩
      rcases ih with ⟨bs,hlen,rfl⟩
      exact ⟨b :: bs, by simp [hlen], rfl⟩

theorem final_trace_choices (w : Word finalizer)
    (h : localTraceSemantics (endpoints finalizer) w) :
    ∃ cs : List Bool, cs.length = specs.length ∧ w = finalFull cs := by
  rw [final_endpoint] at h
  rcases h with ⟨u1,u2,hu1,hu2,rfl⟩
  rcases hu1 with ⟨v1,v2,hv1,hv2,rfl⟩
  rcases hv1 with ⟨x1,x2,hx1,hx2,rfl⟩
  have hx2' : x2 = [] := hx2
  have hv2' : v2 = [AlphabetOf.mkRecv finalizer Port.prepared worker] := hv2
  have hu2' : u2 = [AlphabetOf.mkAct finalizer Port.unit Action.resolve Port.prepared] := hu2
  rcases (fold_trace_iff specs LocProg.eps (fun x => finalStage x.2) x1).mp hx1 with
    ⟨u,vs,hu,hvs,heq⟩
  have hu' : u = [] := hu
  rcases final_stage_traces_choices specs hvs with ⟨cs,hlen,rfl⟩
  exact ⟨cs, hlen, by simp [heq,hu',hx2',hv2',hu2',finalFull,List.append_assoc]⟩

def finalSeq (xs : List (Guard × Role)) (bs : List Bool) : Word finalizer :=
  (finalWords xs bs).flatten

theorem finalSeq_cons (x : Guard × Role) (xs : List (Guard × Role))
    (b : Bool) (bs : List Bool) :
    finalSeq (x :: xs) (b :: bs) =
      finalStageWord x.2 b ++ finalSeq xs bs := rfl

theorem finalStage_input (r : Role) (hne : r ≠ agent) (b : Bool) :
    recvPayloads agent (finalStageWord r b) = [Port.ctrl b] := by
  cases b <;> simp [finalStageWord, recvPayloads, AlphabetOf.mkRecv, hne]

theorem final_prefix_sync (xs : List (Guard × Role))
    (bs cs : List Bool) (tail w : Word finalizer)
    (hvalid : ∀ x ∈ xs, x.2 ≠ agent)
    (hb : bs.length = xs.length) (hc : cs.length = xs.length)
    (hw : w <+: finalSeq xs cs ++ tail)
    (hi : recvPayloads agent w <+: bs.map Port.ctrl) :
    w <+: finalSeq xs bs ++ tail := by
  induction xs generalizing bs cs w with
  | nil =>
      cases bs with
      | cons b bs => simp at hb
      | nil =>
          cases cs with
          | cons c cs => simp at hc
          | nil => simpa [finalSeq, finalWords] using hw
  | cons x xs ih =>
      cases bs with
      | nil => simp at hb
      | cons b bs =>
          cases cs with
          | nil => simp at hc
          | cons c cs =>
              have hb' : bs.length = xs.length := by simpa using hb
              have hc' : cs.length = xs.length := by simpa using hc
              rw [finalSeq_cons] at hw ⊢
              simp only [List.append_assoc] at hw ⊢
              cases hw0 : w with
              | nil => exact List.nil_prefix
              | cons hd rest =>
                  have hhead : hd = AlphabetOf.mkRecv finalizer (Port.ctrl c) agent := by
                    rcases hw with ⟨suffix, heq⟩
                    have hh := congrArg List.head? heq
                    simpa [hw0, finalStageWord] using hh
                  have hbit : c = b := by
                    rcases hi with ⟨suffix, heq⟩
                    have hh := congrArg List.head? heq
                    simp [hw0, hhead, recvPayloads, AlphabetOf.mkRecv] at hh
                    exact hh
                  subst c
                  have hcases := List.prefix_or_prefix_of_prefix hw
                    (List.prefix_append (finalStageWord x.2 b)
                      (finalSeq xs cs ++ tail))
                  rcases hcases with hshort | hlong
                  · simpa [hw0] using hshort.trans
                      (List.prefix_append _ _)
                  · rcases hlong with ⟨v, hv⟩
                    have hrest : v <+: finalSeq xs cs ++ tail := by
                      rw [← hv] at hw
                      exact (List.prefix_self_append_iff).mp hw
                    have hi' : recvPayloads agent v <+: bs.map Port.ctrl := by
                      rw [← hv] at hi
                      have hx : x.2 ≠ agent := hvalid x (List.mem_cons_self)
                      simp only [recvPayloads_append, finalStage_input x.2 hx b,
                        List.map_cons] at hi
                      exact (List.prefix_self_append_iff).mp hi
                    have htailvalid : ∀ y ∈ xs, y.2 ≠ agent := by
                      intro y hy
                      exact hvalid y (List.mem_cons_of_mem x hy)
                    have htarget := ih bs cs v htailvalid hb' hc' hrest hi'
                    have htarget' : finalStageWord x.2 b ++ v <+:
                        finalStageWord x.2 b ++ (finalSeq xs bs ++ tail) :=
                      (List.prefix_self_append_iff).mpr htarget
                    simpa [hw0, hv] using htarget'

theorem final_prefix_transfer (M : Tuple) (hM : IsMSC M)
    (bs : List Bool) (hlen : bs.length = specs.length)
    (hagent : IsPrefixWord (M agent) (agentFull bs))
    (hlocal : localPrefixSemantics (endpoints finalizer) (M finalizer)) :
    IsPrefixWord (M finalizer) (finalFull bs) := by
  rcases hlocal with ⟨v,hv,hpref⟩
  rcases final_trace_choices v hv with ⟨cs,hclen,rfl⟩
  have hw : M finalizer <+: finalSeq specs cs ++
      [AlphabetOf.mkRecv finalizer Port.prepared worker,
       AlphabetOf.mkAct finalizer Port.unit Action.resolve Port.prepared] := by
    rcases hpref with ⟨suffix, heq⟩
    exact ⟨suffix, by simpa [finalSeq, finalFull] using heq.symm⟩
  have hi := final_inputs_prefix M hM bs hlen hagent
  have hsync := final_prefix_sync specs bs cs
    [AlphabetOf.mkRecv finalizer Port.prepared worker,
     AlphabetOf.mkAct finalizer Port.unit Action.resolve Port.prepared]
    (M finalizer) specs_valid hlen hclen hw hi
  rcases hsync with ⟨suffix, heq⟩
  exact ⟨suffix, by simpa [finalFull, finalSeq] using heq.symm⟩

theorem worker_trace_unique (w : Word worker)
    (h : localTraceSemantics (endpoints worker) w) : w = workerFull := by
  simpa [endpoints, workerFull, localTraceSemantics] using h

theorem completeTuple_complete (bs : List Bool)
    (hlen : bs.length = specs.length) : IsCompleteMSC (completeTuple bs) := by
  rw [← fullMSC_eq_completeTuple bs hlen]
  exact fullMSC_complete bs

end ReviewDirect

theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hprefix
  change (∀ a, localPrefixSemantics (SupplierAccess.endpoints a) (M a)) ∧ IsMSC M at hprefix
  rcases hprefix with ⟨hlocal,hM⟩
  rcases hlocal SupplierAccess.Role.agent with ⟨v,hv,hagent⟩
  rcases ReviewDirect.agent_trace_choices v hv with ⟨bs,hlen,rfl⟩
  refine ⟨ReviewDirect.completeTuple bs,
    ⟨ReviewDirect.complete_local_traces bs hlen,
     ReviewDirect.completeTuple_complete bs hlen⟩, ?_⟩
  intro a
  cases a
  · exact hagent
  · rcases hlocal SupplierAccess.Role.worker with ⟨vw,hvw,hworker⟩
    have huniq := ReviewDirect.worker_trace_unique vw hvw
    simpa [ReviewDirect.completeTuple, huniq] using hworker
  · exact ReviewDirect.final_prefix_transfer M hM bs hlen hagent (hlocal SupplierAccess.Role.finalizer)
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer1
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer1))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer2
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer2))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer3
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer3))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer4
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer4))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer5
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer5))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer6
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer6))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer7
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer7))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer8
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer8))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer9
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer9))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer10
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer10))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer11
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer11))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer12
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer12))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer13
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer13))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer14
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer14))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer15
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer15))
  · simpa [ReviewDirect.completeTuple, ReviewDirect.reviewerIndex, ReviewDirect.bitAt] using
      (ReviewDirect.reviewer_prefix_transfer M hM bs hlen hagent SupplierAccess.Role.reviewer16
        (by decide) (by decide) (hlocal SupplierAccess.Role.reviewer16))
