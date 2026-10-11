import CompletionGoal
import SupplierProjection

theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hM
  rw [SupplierAccess.endpoints_eq_projection] at hM ⊢
  exact projectDist_deadlockFree SupplierAccess.protocol SupplierAccess.wellTyped M hM
