import SupplierProtocol
import SupplierEndpoints
set_option linter.unusedSimpArgs false
set_option maxRecDepth 8192
set_option maxHeartbeats 8000000
namespace SupplierAccess
theorem endpoints_eq_projection : endpoints = projectDist protocol := by
  funext a; cases a <;>
    simp [endpoints, projectDist, project, protocol, round,
      ifRecipients, whileRecipients, participationSet,
      controlBroadcast, controlSendTargets, controlRecipients,
      lifelineOrder, Fintype.enum, seqLocList,
      ControlPayload.ctrlPattern, ControlPayload.setDecision]

end SupplierAccess
