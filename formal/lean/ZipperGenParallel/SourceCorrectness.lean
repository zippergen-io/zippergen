import ZipperGenParallel.SourceProjection
import ZipperGenParallel.SourceSemantics
import ZipperGenParallel.Zipper

namespace ZipperGenParallel

universe u

namespace Source
namespace Prog

/--
The branch-indexed local programs obtained by projecting the branches of a
source-level parallel region under their occurrence/path branch channels.
-/
def projectedParallelBranches {n : Nat}
    {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (ch : ChannelName) (label : Nat)
    (branches : Fin n → Prog Lifeline) :
    Fin n → DistProg Lifeline ChannelName :=
  fun i A => project order A (ChannelName.branch ch label i) (branches i)

@[simp]
theorem projectDist_par {n : Nat}
    {Lifeline : Type u} [DecidableEq Lifeline]
    (order : List Lifeline) (ch : ChannelName) (label : Nat)
    (branches : Fin n → Prog Lifeline) :
    projectDist order ch (Prog.par label branches) =
      fun A =>
        LocProg.par
          (fun i =>
            projectedParallelBranches order ch label branches i A) :=
  rfl

/--
Closed trace-image deadlock-freedom for projected source-level parallel
regions.

This is the fully discharged local/projection part of the Zipper argument:
every word-MSC prefix of the projected local `par` has a complete extension in
the decorated tagged `parloc` image.  Mapping that image to the concrete
complete-MSC predicate is a separate word-to-MSC layer.
-/
theorem projected_parallel_deadlock_free_dtrace_complete_image {n : Nat}
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} {label : Nat}
    {branches : Fin n → Prog Lifeline} :
    PrefixCompletable
      (DTraceParCompleteImage
        (projectedParallelBranches order ch label branches))
      WordIsMSC
      (projectDist order ch (Prog.par label branches)) := by
  simpa [projectDist_par, projectedParallelBranches] using
    (parallel_deadlock_free_dtrace_complete_image
      (program := projectedParallelBranches order ch label branches)
      (OrdIsMSC := WordIsMSC))

/--
Top-level specialization of
`projected_parallel_deadlock_free_dtrace_complete_image`.
-/
theorem projected_main_parallel_deadlock_free_dtrace_complete_image
    {n : Nat}
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {label : Nat}
    {branches : Fin n → Prog Lifeline} :
    PrefixCompletable
      (DTraceParCompleteImage
        (projectedParallelBranches order ChannelName.main label branches))
      WordIsMSC
      (projectDistMain order (Prog.par label branches)) := by
  simpa [projectDistMain] using
    (projected_parallel_deadlock_free_dtrace_complete_image
      (order := order)
      (ch := ChannelName.main)
      (label := label)
      (branches := branches))

/--
Forward internal-correctness direction for the source/projection layer:
complete enriched executions are projected distributed executions.
-/
theorem projected_semantics_of_enriched_complete
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} {P : Prog Lifeline}
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
    (hcover : ∀ A, A ∈ lifelines P → A ∈ order)
    (henriched : Enriched order ch P M)
    (hcomplete : WordIsCompleteMSC M) :
    projectedSemantics WordIsCompleteMSC order ch P M :=
  Enriched.to_projectedSemantics henriched hcover hcomplete

/-- Exact internal correctness for the empty source program. -/
theorem projected_semantics_eps_iff_enriched
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName}
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)} :
    projectedSemantics WordIsCompleteMSC order ch
        (Prog.eps : Prog Lifeline) M ↔
      Enriched order ch (Prog.eps : Prog Lifeline) M := by
  constructor
  · intro hsem
    have hM : M = emptyTuple Lifeline := by
      funext A
      exact (LocProg.trace_eps (M A)).1
        (by simpa [projectedSemantics, projectDist, project] using hsem.1 A)
    rw [hM]
    exact Enriched.eps
  · intro henriched
    cases henriched
    exact ⟨fun A => LocProg.Trace.eps, emptyTuple_completeMSC⟩

/-- Exact internal correctness for a source-local action. -/
theorem projected_semantics_act_iff_enriched
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} {owner : Lifeline}
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)} :
    projectedSemantics WordIsCompleteMSC order ch
        (Prog.act owner) M ↔
      Enriched order ch (Prog.act owner) M := by
  constructor
  · intro hsem
    have hM : M = actionTuple owner := by
      funext A
      by_cases howner : A = owner
      · subst A
        simpa [actionTuple] using
          (LocProg.trace_letter LocalLetter.action (M owner)).1
            (by simpa [projectedSemantics, projectDist, project] using
              hsem.1 owner)
      · simpa [actionTuple, howner] using
          (LocProg.trace_eps (M A)).1
            (by simpa [projectedSemantics, projectDist, project, howner] using
              hsem.1 A)
    rw [hM]
    exact Enriched.act
  · intro henriched
    cases henriched
    refine ⟨?_, actionTuple_completeMSC owner⟩
    intro A
    by_cases howner : A = owner
    · subst A
      simpa [projectDist, project, actionTuple] using
        (LocProg.Trace.letter
          (LocalLetter.action :
            LocalLetter Lifeline ChannelName))
    · simpa [projectDist, project, actionTuple, howner] using
        (LocProg.Trace.eps :
          LocProg.trace
            (LocProg.eps : LocProg Lifeline ChannelName) [])

/-- Exact internal correctness for a source message. -/
theorem projected_semantics_msg_iff_enriched
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName}
    {sender receiver : Lifeline} (hne : sender ≠ receiver)
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)} :
    projectedSemantics WordIsCompleteMSC order ch
        (Prog.msg sender receiver hne) M ↔
      Enriched order ch (Prog.msg sender receiver hne) M := by
  constructor
  · intro hsem
    have hM : M = messageTuple sender receiver ch := by
      funext A
      by_cases hsender : A = sender
      · subst A
        simpa [messageTuple] using
          (LocProg.trace_letter (LocalLetter.send receiver ch) (M sender)).1
            (by simpa [projectedSemantics, projectDist, project] using
              hsem.1 sender)
      · by_cases hreceiver : A = receiver
        · subst A
          have hreceiver_sender : receiver ≠ sender := by
            intro h
            exact hne h.symm
          simpa [messageTuple, hreceiver_sender] using
            (LocProg.trace_letter (LocalLetter.recv sender ch)
                (M receiver)).1
              (by
                simpa [projectedSemantics, projectDist, project,
                  hreceiver_sender] using hsem.1 receiver)
        · simpa [messageTuple, hsender, hreceiver] using
            (LocProg.trace_eps (M A)).1
              (by
                simpa [projectedSemantics, projectDist, project,
                  hsender, hreceiver] using hsem.1 A)
    rw [hM]
    exact Enriched.msg hne
  · intro henriched
    cases henriched
    refine ⟨?_, messageTuple_completeMSC hne ch⟩
    intro A
    by_cases hsender : A = sender
    · subst A
      simpa [projectDist, project, messageTuple] using
        (LocProg.Trace.letter
          (LocalLetter.send receiver ch :
            LocalLetter Lifeline ChannelName))
    · by_cases hreceiver : A = receiver
      · subst A
        have hreceiver_sender : receiver ≠ sender := by
          intro h
          exact hne h.symm
        simpa [projectDist, project, messageTuple, hreceiver_sender] using
          (LocProg.Trace.letter
            (LocalLetter.recv sender ch :
              LocalLetter Lifeline ChannelName))
      · simpa [projectDist, project, messageTuple, hsender, hreceiver] using
          (LocProg.Trace.eps :
            LocProg.trace
              (LocProg.eps : LocProg Lifeline ChannelName) [])

/-- Control-erased correctness for the empty source program. -/
theorem projected_correctness_eps
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} :
    ∀ observed,
      ErasedSemantics
          (projectedSemantics WordIsCompleteMSC order ch
            (Prog.eps : Prog Lifeline))
          observed ↔
        sourceObservation order ch (Prog.eps : Prog Lifeline) observed :=
  parallel_correctness_from_internal_equality
    (DistributedInternal :=
      projectedSemantics WordIsCompleteMSC order ch
        (Prog.eps : Prog Lifeline))
    (EnrichedInternal :=
      Enriched order ch (Prog.eps : Prog Lifeline))
    (fun _ => projected_semantics_eps_iff_enriched)

/-- Control-erased correctness for a source-local action. -/
theorem projected_correctness_act
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} {owner : Lifeline} :
    ∀ observed,
      ErasedSemantics
          (projectedSemantics WordIsCompleteMSC order ch
            (Prog.act owner))
          observed ↔
        sourceObservation order ch (Prog.act owner) observed :=
  parallel_correctness_from_internal_equality
    (DistributedInternal :=
      projectedSemantics WordIsCompleteMSC order ch
        (Prog.act owner))
    (EnrichedInternal :=
      Enriched order ch (Prog.act owner))
    (fun _ => projected_semantics_act_iff_enriched)

/-- Control-erased correctness for a source message. -/
theorem projected_correctness_msg
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName}
    {sender receiver : Lifeline} (hne : sender ≠ receiver) :
    ∀ observed,
      ErasedSemantics
          (projectedSemantics WordIsCompleteMSC order ch
            (Prog.msg sender receiver hne))
          observed ↔
        sourceObservation order ch
          (Prog.msg sender receiver hne) observed :=
  parallel_correctness_from_internal_equality
    (DistributedInternal :=
      projectedSemantics WordIsCompleteMSC order ch
        (Prog.msg sender receiver hne))
    (EnrichedInternal :=
      Enriched order ch (Prog.msg sender receiver hne))
    (fun _ => projected_semantics_msg_iff_enriched hne)

/-- Projected local languages are prefix-free for source programs without `par`. -/
theorem projected_localPrefixFree_noPar
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName}
    {P : Prog Lifeline} (hP : NoPar P) :
    LocalPrefixFree
      (fun A word => LocProg.trace (projectDist order ch P A) word) :=
  LocProg.localPrefixFree_noPar
    (projectDist_guardDistinct order ch P)
    (projectDist_noPar order ch hP)

/--
Complete projected local traces are complete word MSCs for source programs
without `par`, provided the projected program has the Zipper
prefix-completion property.
-/
theorem projected_complete_of_prefixCompletable_noPar
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName}
    {P : Prog Lifeline} (hP : NoPar P)
    (hcomplete :
      PrefixCompletable WordIsCompleteMSC WordIsMSC
        (projectDist order ch P))
    {M : TraceTuple Lifeline (LocalLetter Lifeline ChannelName)}
    (htrace : DistProg.trace (projectDist order ch P) M)
    (hmsc : WordIsMSC M) :
    WordIsCompleteMSC M :=
  complete_of_prefixCompletable_and_local_prefix_free
    hcomplete
    (projected_localPrefixFree_noPar (order := order) (ch := ch) hP)
    htrace hmsc

/--
Control-erased forward direction when the enriched witness already satisfies
the projected complete-MSC filter.
-/
theorem projected_observation_of_enriched_complete_observation
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} {P : Prog Lifeline}
    (hcover : ∀ A, A ∈ lifelines P → A ∈ order) :
    ∀ observed,
      ErasedSemantics
          (fun M => Enriched order ch P M ∧ WordIsCompleteMSC M)
          observed →
        ErasedSemantics
          (projectedSemantics WordIsCompleteMSC order ch P)
          observed := by
  intro observed hobs
  rcases hobs with ⟨internal, hinternal, herase⟩
  exact
    ⟨internal,
      projected_semantics_of_enriched_complete
        hcover hinternal.1 hinternal.2,
      herase⟩

/--
Control-erased forward direction once decision blocks have concrete complete
word-MSC witnesses.
-/
theorem projected_observation_of_enriched_observation_of_decisionBlocks
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline}
    (hdecision :
      ∀ (owner : Lifeline) (targets : List Lifeline)
        (ch : ChannelName) (tag : ControlTag) (kind : ChoiceKind),
        WordIsCompleteMSC
          (decisionBlock owner targets ch tag kind))
    {ch : ChannelName} {P : Prog Lifeline}
    (hcover : ∀ A, A ∈ lifelines P → A ∈ order) :
    ∀ observed,
      sourceObservation order ch P observed →
        ErasedSemantics
          (projectedSemantics WordIsCompleteMSC order ch P)
          observed := by
  intro observed hobs
  rcases hobs with ⟨internal, hinternal, herase⟩
  exact
    ⟨internal,
      Enriched.to_projectedSemantics_of_decisionBlocks
        hdecision hinternal hcover,
      herase⟩

/--
Top-level specialization of
`projected_observation_of_enriched_complete_observation`.
-/
theorem projected_main_observation_of_enriched_complete_observation
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {P : Prog Lifeline}
    (hcover : ∀ A, A ∈ lifelines P → A ∈ order) :
    ∀ observed,
      ErasedSemantics
          (fun M =>
            Enriched order ChannelName.main P M ∧ WordIsCompleteMSC M)
          observed →
        ErasedSemantics
          (projectedSemantics WordIsCompleteMSC order ChannelName.main P)
          observed :=
  projected_observation_of_enriched_complete_observation
    (ch := ChannelName.main)
    hcover

/--
Control-erased correctness for the self-contained source/projection layer,
from the internal equality between projected distributed semantics and the
internal enriched source semantics.
-/
theorem projected_correctness_from_internal_equality
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {ch : ChannelName} {P : Prog Lifeline}
    (h :
      ∀ M,
        projectedSemantics WordIsCompleteMSC order ch P M ↔
          Enriched order ch P M) :
    ∀ observed,
      ErasedSemantics
          (projectedSemantics WordIsCompleteMSC order ch P)
          observed ↔
        sourceObservation order ch P observed :=
  parallel_correctness_from_internal_equality
    (DistributedInternal :=
      projectedSemantics WordIsCompleteMSC order ch P)
    (EnrichedInternal := Enriched order ch P)
    h

/-- Top-level specialization of `projected_correctness_from_internal_equality`. -/
theorem projected_main_correctness_from_internal_equality
    {Lifeline : Type u} [DecidableEq Lifeline]
    {order : List Lifeline} {P : Prog Lifeline}
    (h :
      ∀ M,
        projectedSemantics WordIsCompleteMSC order ChannelName.main P M ↔
          Enriched order ChannelName.main P M) :
    ∀ observed,
      ErasedSemantics
          (projectedSemantics WordIsCompleteMSC order ChannelName.main P)
          observed ↔
        sourceObservationMain order P observed := by
  simpa [sourceObservationMain] using
    (projected_correctness_from_internal_equality
      (order := order)
      (ch := ChannelName.main)
      (P := P)
      h)

end Prog
end Source

end ZipperGenParallel
