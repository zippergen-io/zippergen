import SupplierVocabulary
set_option linter.unusedSimpArgs false
namespace SupplierAccess
open Role

/-- Optional reviews are statically ordered at Finalizer. Worker may prepare
before any human review returns; its message waits in a separate FIFO channel. -/
def round : Program :=
  .msg agent .plan worker .plan (by decide) ;;
  .ite .need1 agent
    (.msg agent .review reviewer1 .review (by decide) ;;
     .act reviewer1 .unit .assess .review ;;
     .msg reviewer1 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need2 agent
    (.msg agent .review reviewer2 .review (by decide) ;;
     .act reviewer2 .unit .assess .review ;;
     .msg reviewer2 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need3 agent
    (.msg agent .review reviewer3 .review (by decide) ;;
     .act reviewer3 .unit .assess .review ;;
     .msg reviewer3 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need4 agent
    (.msg agent .review reviewer4 .review (by decide) ;;
     .act reviewer4 .unit .assess .review ;;
     .msg reviewer4 .verdict finalizer .verdict (by decide))
    .eps ;;
  .act worker .prepared .prepare .plan ;;
  .msg worker .prepared finalizer .prepared (by decide) ;;
  .act finalizer .unit .resolve .prepared
def protocol : Program := round

theorem wellTyped : WellTypedProgram protocol := by
  simp [protocol, round, WellTypedProgram,
    PayloadCompatible, PayloadCompatiblePred.compat, compatible]
theorem distinguishable : ControlDistinguishableProgram protocol := by
  simp [protocol, round, ControlDistinguishableProgram,
    isControlPayload, ControlPayloadSpec.isControl]
end SupplierAccess
