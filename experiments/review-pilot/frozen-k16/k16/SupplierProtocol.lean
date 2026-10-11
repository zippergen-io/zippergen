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
  .ite .need5 agent
    (.msg agent .review reviewer5 .review (by decide) ;;
     .act reviewer5 .unit .assess .review ;;
     .msg reviewer5 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need6 agent
    (.msg agent .review reviewer6 .review (by decide) ;;
     .act reviewer6 .unit .assess .review ;;
     .msg reviewer6 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need7 agent
    (.msg agent .review reviewer7 .review (by decide) ;;
     .act reviewer7 .unit .assess .review ;;
     .msg reviewer7 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need8 agent
    (.msg agent .review reviewer8 .review (by decide) ;;
     .act reviewer8 .unit .assess .review ;;
     .msg reviewer8 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need9 agent
    (.msg agent .review reviewer9 .review (by decide) ;;
     .act reviewer9 .unit .assess .review ;;
     .msg reviewer9 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need10 agent
    (.msg agent .review reviewer10 .review (by decide) ;;
     .act reviewer10 .unit .assess .review ;;
     .msg reviewer10 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need11 agent
    (.msg agent .review reviewer11 .review (by decide) ;;
     .act reviewer11 .unit .assess .review ;;
     .msg reviewer11 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need12 agent
    (.msg agent .review reviewer12 .review (by decide) ;;
     .act reviewer12 .unit .assess .review ;;
     .msg reviewer12 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need13 agent
    (.msg agent .review reviewer13 .review (by decide) ;;
     .act reviewer13 .unit .assess .review ;;
     .msg reviewer13 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need14 agent
    (.msg agent .review reviewer14 .review (by decide) ;;
     .act reviewer14 .unit .assess .review ;;
     .msg reviewer14 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need15 agent
    (.msg agent .review reviewer15 .review (by decide) ;;
     .act reviewer15 .unit .assess .review ;;
     .msg reviewer15 .verdict finalizer .verdict (by decide))
    .eps ;;
  .ite .need16 agent
    (.msg agent .review reviewer16 .review (by decide) ;;
     .act reviewer16 .unit .assess .review ;;
     .msg reviewer16 .verdict finalizer .verdict (by decide))
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
