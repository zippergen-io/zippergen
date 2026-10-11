import ZipperGenParallel.SourceCorrectness

namespace ZipperGenSpawn

open ZipperGenParallel

universe u v

/-- Componentwise prefix closure for properties of distributed words. -/
def ParallelTuplePrefixClosed
    (property : TraceTuple Lifeline (LocalLetter Lifeline Channel) -> Prop) :
    Prop :=
  ∀ U V,
    (∀ A, ListPrefix (U A) (V A)) ->
    property V -> property U

/-- Prefix-completion is enough to transfer any prefix-closed property from
    complete projected executions to every valid finite projected prefix. -/
theorem parallel_projected_prefix_property
    {program : DistProg Lifeline Channel}
    {CompleteMSC IsMSC :
      TraceTuple Lifeline (LocalLetter Lifeline Channel) -> Prop}
    (property : TraceTuple Lifeline (LocalLetter Lifeline Channel) -> Prop)
    (hcomplete : PrefixCompletable CompleteMSC IsMSC program)
    (hclosed : ParallelTuplePrefixClosed property)
    (hfull : ∀ words,
      DistProg.semantics CompleteMSC program words -> property words) :
    ∀ U,
      DistProg.prefixSemantics IsMSC program U -> property U := by
  intro U hU
  rcases hcomplete U hU with ⟨V, hsem, hpref⟩
  exact hclosed U V hpref (hfull V hsem)

end ZipperGenSpawn
