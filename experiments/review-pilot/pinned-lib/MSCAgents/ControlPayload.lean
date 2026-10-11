/-
  MSCAgents/ControlPayload.lean
  =============================
  Support for the projection-generated control payloads used in sec:projection of:
  "Provable Coordination for LLM Agents via Message Sequence Charts"
-/

import MSCAgents.PayloadMatching

/-- Abstract interface for the distinguished control payloads used by
    recipient-side control constructs and projection-generated broadcasts. -/
class ControlPayload (Payload : Type) where
  ctrlPattern : Payload
  setDecision : Bool → Payload → Payload

/-- Additional laws needed for erasure and correctness:
    control payloads are recognisable, compatible with themselves, and
    compatibility preserves the control/user split. -/
class ControlPayloadSpec (Payload : Type)
    [PayloadCompatiblePred Payload] extends ControlPayload Payload where
  isControl : Payload → Bool
  isControl_true :
    isControl (ControlPayload.setDecision true ControlPayload.ctrlPattern) = true
  isControl_false :
    isControl (ControlPayload.setDecision false ControlPayload.ctrlPattern) = true
  compat_ctrl_true :
    PayloadCompatible Payload
      (ControlPayload.setDecision true ControlPayload.ctrlPattern)
      (ControlPayload.setDecision true ControlPayload.ctrlPattern)
  compat_ctrl_false :
    PayloadCompatible Payload
      (ControlPayload.setDecision false ControlPayload.ctrlPattern)
      (ControlPayload.setDecision false ControlPayload.ctrlPattern)
  compat_preserves_isControl :
    ∀ {xs ys : Payload},
      PayloadCompatible Payload xs ys →
      isControl xs = isControl ys
  compat_decision_eq :
    ∀ {b1 b2 : Bool},
      PayloadCompatible Payload
        (ControlPayload.setDecision b1 ControlPayload.ctrlPattern)
        (ControlPayload.setDecision b2 ControlPayload.ctrlPattern) →
      b1 = b2
  setDecision_injective :
    ∀ {b1 b2 : Bool} {p : Payload},
      ControlPayload.setDecision b1 p = ControlPayload.setDecision b2 p →
      b1 = b2

/-- The distinguished `⊤`-decision payload. -/
def ctrlTruePayload {Payload : Type} [ControlPayload Payload] : Payload :=
  ControlPayload.setDecision true ControlPayload.ctrlPattern

/-- The distinguished `⊥`-decision payload. -/
def ctrlFalsePayload {Payload : Type} [ControlPayload Payload] : Payload :=
  ControlPayload.setDecision false ControlPayload.ctrlPattern

/-- Predicate recognizing projection-generated control payloads. -/
def isControlPayload {Payload : Type} [PayloadCompatiblePred Payload]
    [ControlPayloadSpec Payload] (p : Payload) : Bool :=
  ControlPayloadSpec.isControl p

theorem controlPayload_compat_decision_eq
    {Payload : Type} [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    {b1 b2 : Bool}
    (h :
      PayloadCompatible Payload
        (ControlPayload.setDecision b1 ControlPayload.ctrlPattern)
        (ControlPayload.setDecision b2 ControlPayload.ctrlPattern)) :
    b1 = b2 :=
  ControlPayloadSpec.compat_decision_eq h

theorem controlPayload_setDecision_eq
    {Payload : Type} [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    {b1 b2 : Bool} {p : Payload}
    (h : ControlPayload.setDecision b1 p = ControlPayload.setDecision b2 p) :
    b1 = b2 :=
  ControlPayloadSpec.setDecision_injective h

theorem ctrlTrue_ne_ctrlFalse
    {Payload : Type} [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    {p : Payload} :
    ControlPayload.setDecision true p ≠ ControlPayload.setDecision false p := by
  intro hEq
  have : true = false := controlPayload_setDecision_eq (Payload := Payload) hEq
  cases this

theorem ctrlFalse_ne_ctrlTrue
    {Payload : Type} [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]
    {p : Payload} :
    ControlPayload.setDecision false p ≠ ControlPayload.setDecision true p := by
  intro hEq
  have : false = true := controlPayload_setDecision_eq (Payload := Payload) hEq
  cases this

theorem ctrlTrue_not_compat_ctrlFalse
    {Payload : Type} [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload] :
    ¬ PayloadCompatible Payload
        (ControlPayload.setDecision true ControlPayload.ctrlPattern)
        (ControlPayload.setDecision false ControlPayload.ctrlPattern) := by
  intro h
  have : true = false :=
    controlPayload_compat_decision_eq (Payload := Payload) h
  cases this

theorem ctrlFalse_not_compat_ctrlTrue
    {Payload : Type} [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload] :
    ¬ PayloadCompatible Payload
        (ControlPayload.setDecision false ControlPayload.ctrlPattern)
        (ControlPayload.setDecision true ControlPayload.ctrlPattern) := by
  intro h
  have : false = true :=
    controlPayload_compat_decision_eq (Payload := Payload) h
  cases this
