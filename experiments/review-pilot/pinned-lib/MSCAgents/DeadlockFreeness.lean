/-
  MSCAgents/DeadlockFreeness.lean
  ===============================
  Formalization of Definition 11 (def:deadlock-prefix) from sec:distributed of:
  "Provable Coordination for LLM Agents via Message Sequence Charts"
-/

import MSCAgents.ZipperLemma

section DeadlockFreeness

variable {L C F Payload : Type} [DecidableEq L] [Fintype L]
variable [PayloadCompatiblePred Payload] [ControlPayload Payload]

/-- Deadlock-freeness via prefix semantics. -/
def DeadlockFree
    (D : DistProg L C F Payload) : Prop :=
  ∀ M,
    distPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) D M →
      ∃ M',
        distSemantics (L := L) (C := C) (F := F) (Payload := Payload) D M' ∧
        IsPrefixTuple M M'

/-- Every complete distributed execution is, in particular, a valid prefix
    execution. -/
theorem distSemantics_subset_prefix
    (D : DistProg L C F Payload)
    (M : WordTuple L C F Payload)
    (hM : distSemantics (L := L) (C := C) (F := F) (Payload := Payload) D M) :
    distPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) D M := by
  rcases hM with ⟨hTrace, hComplete⟩
  refine ⟨?_, isCompleteMSC_implies_isMSC (L := L) (C := C) (F := F) (Payload := Payload) M hComplete⟩
  intro A
  refine ⟨M A, hTrace A, ?_⟩
  exact ⟨[], by simp [IsPrefixWord]⟩

end DeadlockFreeness

section DeadlockFreeCorollary

variable {L C F Payload : Type} [DecidableEq L] [Fintype L]
variable [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]

/-- Corollary `cor:deadlock-free`: the projection of every well-typed global
    program is deadlock-free in the prefix-extension sense. -/
theorem projectDist_deadlockFree
    (prog : Prog L C F Payload)
    (hWF : WellTypedProgram prog) :
    DeadlockFree (L := L) (C := C) (F := F) (Payload := Payload)
      (projectDist (L := L) (C := C) (F := F) (Payload := Payload) prog) := by
  intro M hPref
  rcases hPref with ⟨hLocalPref, hMSC⟩
  have hZip :=
    uniformZipper (L := L) (C := C) (F := F) (Payload := Payload) prog hWF
      M (WordTuple.empty (L := L) (C := C) (F := F) (Payload := Payload))
      (fun X => by
        simpa [projectDist] using hLocalPref X)
      (fun X hX => by
        simp [WordTuple.empty] at hX)
      (by
        simpa [WordTuple.concat_eps_right] using hMSC)
  rcases hZip with ⟨Mbar, hPost⟩
  rcases hPost with ⟨hLoc, hComplete, _hOrig, _hCompletedMSC, _hEmptyMSC⟩
  refine ⟨Mbar, ?_, ?_⟩
  · refine ⟨?_, hComplete⟩
    intro X
    simpa [projectDist] using (hLoc X).2.1
  · intro X
    exact (hLoc X).1

end DeadlockFreeCorollary
