import CompletionGoal
import SupplierProtocol
import SupplierProjection

open SupplierAccess in
theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hM
  rw [endpoints_eq_projection] at hM ⊢
  exact projectDist_deadlockFree protocol wellTyped M hM
