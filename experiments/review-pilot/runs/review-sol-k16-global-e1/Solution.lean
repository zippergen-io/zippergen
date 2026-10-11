import CompletionGoal
import SupplierProjection
theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hPrefix
  rw [SupplierAccess.endpoints_eq_projection] at hPrefix ⊢
  exact projectDist_deadlockFree SupplierAccess.protocol
    SupplierAccess.wellTyped M hPrefix
