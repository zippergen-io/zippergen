import SupplierEndpoints
namespace ReviewCompletion
open SupplierAccess
/-- Every valid asynchronous prefix has a complete extension preserving all old role words. -/
def CompletionGoal : Prop :=
  ∀ M : Tuple, distPrefixSemantics endpoints M →
    ∃ N, distSemantics endpoints N ∧ IsPrefixTuple M N
end ReviewCompletion
