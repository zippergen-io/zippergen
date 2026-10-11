import CompletionGoal
import SupplierProjection
theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hPrefix
  have hProjected : distPrefixSemantics
      (projectDist SupplierAccess.protocol) M := by
    rw [← SupplierAccess.endpoints_eq_projection]
    exact hPrefix
  rw [SupplierAccess.endpoints_eq_projection]
  exact projectDist_deadlockFree
    SupplierAccess.protocol SupplierAccess.wellTyped M hProjected
