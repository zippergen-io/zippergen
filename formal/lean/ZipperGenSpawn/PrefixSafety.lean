/-
  Safety transfer for finite distributed prefixes.

  A projected prefix can be completed by deadlock freedom. Consequently, a
  prefix-closed property proved on the global MSC semantics holds already on
  every finite execution prefix of the projected endpoints.
-/

import MSCAgents.Correctness
import MSCAgents.DeadlockFreeness

namespace ZipperGenSpawn

section PrefixSafety

variable {L C F Payload : Type} [DecidableEq L] [Fintype L]
variable [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]

/-- A property of word tuples is prefix closed when it holds for every prefix
    of a tuple on which it holds. -/
def TuplePrefixClosed (property : WordTuple L C F Payload → Prop) : Prop :=
  ∀ M M', IsPrefixTuple M M' → property M' → property M

/-- A property may rely on causal closure of the observed cut.  Such a
    property need only be preserved by componentwise prefixes that are
    themselves valid MSCs.  This is strictly weaker than `TuplePrefixClosed`:
    an arbitrary tuple prefix may retain a receive while omitting its matching
    send, whereas an execution prefix cannot. -/
def MSCTuplePrefixClosed (property : WordTuple L C F Payload → Prop) : Prop :=
  ∀ M M', IsPrefixTuple M M' → IsMSC M → property M' → property M

/-- Erasing generated control traffic preserves the componentwise prefix
    relation. -/
theorem eraseTuple_prefix {M M' : WordTuple L C F Payload}
    (h : IsPrefixTuple M M') :
    IsPrefixTuple (eraseTuple M) (eraseTuple M') := by
  intro A
  rcases h A with ⟨suffix, hsuffix⟩
  refine ⟨eraseWord suffix, ?_⟩
  change eraseWord (M' A) = eraseWord (M A) ++ eraseWord suffix
  rw [hsuffix, eraseWord_append]

/-- Prefix soundness of projection. Every finite execution prefix of the
    projected endpoints erases to a componentwise prefix of some complete
    global MSC of the choreography. -/
theorem projected_prefix_realization
    (prog : Prog L C F Payload)
    (hWellTyped : WellTypedProgram prog)
    (hCtrl : ControlDistinguishableProgram prog) :
    ∀ Mhat,
      distPrefixSemantics (projectDist prog) Mhat →
      ∃ M,
        ⟦prog⟧ M ∧
        IsPrefixTuple (eraseTuple Mhat) M := by
  intro Mhat hMhat
  rcases projectDist_deadlockFree prog hWellTyped Mhat hMhat with
    ⟨Mcomplete, hComplete, hPrefix⟩
  exact ⟨eraseTuple Mcomplete,
    realization_sound prog hWellTyped hCtrl Mcomplete hComplete,
    eraseTuple_prefix hPrefix⟩

/-- A prefix-closed property established on the global semantics holds on
    every finite execution prefix of the projected endpoints. -/
theorem projected_prefix_property
    (prog : Prog L C F Payload)
    (hWellTyped : WellTypedProgram prog)
    (hCtrl : ControlDistinguishableProgram prog)
    (property : WordTuple L C F Payload → Prop)
    (hPrefixClosed : TuplePrefixClosed property)
    (hGlobal : ∀ M, ⟦prog⟧ M → property M) :
    ∀ Mhat,
      distPrefixSemantics (projectDist prog) Mhat →
      property (eraseTuple Mhat) := by
  intro Mhat hMhat
  rcases projected_prefix_realization prog hWellTyped hCtrl Mhat hMhat with
    ⟨M, hM, hPrefix⟩
  exact hPrefixClosed _ _ hPrefix (hGlobal M hM)

/-- A causally prefix-closed global property holds on every finite projected
    execution.  Unlike `projected_prefix_property`, the closure proof may use
    the fact that the observed tuple is an MSC. -/
theorem projected_msc_prefix_property
    (prog : Prog L C F Payload)
    (hWellTyped : WellTypedProgram prog)
    (hCtrl : ControlDistinguishableProgram prog)
    (property : WordTuple L C F Payload → Prop)
    (hPrefixClosed : MSCTuplePrefixClosed property)
    (hGlobal : ∀ M, ⟦prog⟧ M → property M) :
    ∀ Mhat,
      distPrefixSemantics (projectDist prog) Mhat →
      property (eraseTuple Mhat) := by
  intro Mhat hMhat
  rcases projected_prefix_realization prog hWellTyped hCtrl Mhat hMhat with
    ⟨M, hM, hPrefix⟩
  exact hPrefixClosed _ _ hPrefix (IsMSC_erase Mhat hMhat.2)
    (hGlobal M hM)

end PrefixSafety

end ZipperGenSpawn
