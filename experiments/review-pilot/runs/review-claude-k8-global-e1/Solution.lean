import CompletionGoal
import SupplierProjection

open SupplierAccess in
theorem submission : ReviewCompletion.CompletionGoal := by
  unfold ReviewCompletion.CompletionGoal
  rw [endpoints_eq_projection]
  exact projectDist_deadlockFree protocol wellTyped
