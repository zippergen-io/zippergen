import ZipperGenSpawn.PrefixSafety

set_option linter.unusedSimpArgs false
namespace SupplierAccess
inductive Role | agent | worker | finalizer | reviewer1 deriving DecidableEq, Repr
open Role
instance : Fintype Role where
  elems := [agent, worker, finalizer, reviewer1]
  nodup := by decide
  complete := by intro a; cases a <;> simp
@[simp] theorem role_enum : (Fintype.elems : List Role) = [agent, worker, finalizer, reviewer1] := rfl
instance (p : Role → Prop) [∀ a, Decidable (p a)] : Decidable (∀ a, p a) :=
  decidable_of_iff (p agent ∧ p worker ∧ p finalizer ∧ p reviewer1)
    ⟨by rintro ⟨h0,h1,h2,h3⟩ a; cases a <;> assumption,
     by intro h; exact ⟨h _,h _,h _,h _⟩⟩
inductive Port | unit | plan | review | verdict | prepared | ctrl (b : Bool)
  deriving DecidableEq, Repr
inductive Guard | more | need1 deriving DecidableEq, Repr
inductive Action | assess | prepare | resolve deriving DecidableEq, Repr
def compatible : Port → Port → Prop
  | a,b => a = b
instance (p q : Port) : Decidable (compatible p q) := by
  cases p <;> cases q <;> unfold compatible <;> infer_instance
instance : PayloadCompatiblePred Port := ⟨compatible⟩
instance : ControlPayload Port where
  ctrlPattern := .ctrl false
  setDecision b _ := .ctrl b
instance : ControlPayloadSpec Port where
  isControl p := match p with | .ctrl _ => true | _ => false
  isControl_true := rfl
  isControl_false := rfl
  compat_ctrl_true := rfl
  compat_ctrl_false := rfl
  compat_preserves_isControl := by
    intro a b h; cases a <;> cases b <;> simp_all [PayloadCompatible,PayloadCompatiblePred.compat,compatible]
  compat_decision_eq := by
    intro a b h; simpa [PayloadCompatible,PayloadCompatiblePred.compat,compatible,
      ControlPayload.setDecision,ControlPayload.ctrlPattern] using h
  setDecision_injective := by intro a b p h; cases h; rfl
abbrev Program := Prog Role Guard Action Port
abbrev Tuple := WordTuple Role Guard Action Port
abbrev Ltr := Letter Role Guard Action Port
abbrev Word (a : Role) := LocalWord (C := Guard) (F := Action) (Payload := Port) a
abbrev Local (a : Role) := LocProg Role Guard Action Port a
deriving instance DecidableEq, Repr for Letter

end SupplierAccess
