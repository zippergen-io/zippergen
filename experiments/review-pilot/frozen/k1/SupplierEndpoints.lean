import SupplierVocabulary
set_option linter.unusedSimpArgs false
set_option maxRecDepth 8192
set_option maxHeartbeats 8000000
namespace SupplierAccess
def endpoints : (a : Role) → Local a
  | .agent => ((((LocProg.send Port.plan Role.worker ;;ₗ LocProg.localIf Guard.need1 ((LocProg.send (Port.ctrl true) Role.finalizer ;;ₗ (LocProg.send (Port.ctrl true) Role.reviewer1 ;;ₗ LocProg.eps)) ;;ₗ ((LocProg.send Port.review Role.reviewer1 ;;ₗ LocProg.eps) ;;ₗ LocProg.eps)) ((LocProg.send (Port.ctrl false) Role.finalizer ;;ₗ (LocProg.send (Port.ctrl false) Role.reviewer1 ;;ₗ LocProg.eps)) ;;ₗ LocProg.eps)) ;;ₗ LocProg.eps) ;;ₗ LocProg.eps) ;;ₗ LocProg.eps)
  | .worker => ((((LocProg.recv Port.plan Role.agent ;;ₗ LocProg.eps) ;;ₗ LocProg.act Port.prepared Action.prepare Port.plan) ;;ₗ LocProg.send Port.prepared Role.finalizer) ;;ₗ LocProg.eps)
  | .finalizer => ((((LocProg.eps ;;ₗ LocProg.recvIf (Port.ctrl false) Role.agent ((LocProg.eps ;;ₗ LocProg.eps) ;;ₗ LocProg.recv Port.verdict Role.reviewer1) LocProg.eps) ;;ₗ LocProg.eps) ;;ₗ LocProg.recv Port.prepared Role.worker) ;;ₗ LocProg.act Port.unit Action.resolve Port.prepared)
  | .reviewer1 => ((((LocProg.eps ;;ₗ LocProg.recvIf (Port.ctrl false) Role.agent ((LocProg.recv Port.review Role.agent ;;ₗ LocProg.act Port.unit Action.assess Port.review) ;;ₗ LocProg.send Port.verdict Role.finalizer) LocProg.eps) ;;ₗ LocProg.eps) ;;ₗ LocProg.eps) ;;ₗ LocProg.eps)
end SupplierAccess
