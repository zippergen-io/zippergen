/-
  MSCAgents/LocalSemantics.lean
  =============================
  Formalization of the local trace semantics, prefix semantics, and
  distributed semantics from sec:distributed of:
  "Provable Coordination for LLM Agents via Message Sequence Charts"
-/

import MSCAgents.Alphabets
import MSCAgents.LocalSyntax
import MSCAgents.ControlPayload
import MSCAgents.MSC
import MSCAgents.Projection

/-- Sets of local words, represented as predicates. -/
abbrev LocalWordSet (L C F Payload : Type) (A : L) :=
  LocalWord (C := C) (F := F) (Payload := Payload) A → Prop

/-- Concatenate a list of local words. -/
def concatLocalWords {L C F Payload : Type} {A : L}
    (ws : List (LocalWord (C := C) (F := F) (Payload := Payload) A)) :
    LocalWord (C := C) (F := F) (Payload := Payload) A :=
  ws.foldl (· ++ ·) []

/-- Prefix relation on tuples of local words. -/
def IsPrefixTuple {L C F Payload : Type}
    (M1 M2 : WordTuple L C F Payload) : Prop :=
  ∀ A, IsPrefixWord (M1 A) (M2 A)

/-- Complete local trace semantics `⟪S⟫_A` from the paper. -/
def localTraceSemantics
    {L C F Payload : Type} [ControlPayload Payload] {A : L} :
    LocProg L C F Payload A → LocalWordSet L C F Payload A
  | .eps =>
      fun w => w = []
  | .send xs B =>
      fun w => ∃ h : A ≠ B, w = [AlphabetOf.mkSend (C := C) (F := F) A xs B h]
  | .recv ys B =>
      fun w => w = [AlphabetOf.mkRecv (C := C) (F := F) A ys B]
  | .act ys f xs =>
      fun w => w = [AlphabetOf.mkAct (C := C) A ys f xs]
  | .seq S1 S2 =>
      fun w => ∃ u1 u2, localTraceSemantics S1 u1 ∧ localTraceSemantics S2 u2 ∧ w = u1 ++ u2
  | .recvIf ys B STrue SFalse =>
      fun w =>
        (∃ u, localTraceSemantics STrue u ∧
          w = AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision true ys) B :: u) ∨
        (∃ u, localTraceSemantics SFalse u ∧
          w = AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision false ys) B :: u)
  | .localIf c STrue SFalse =>
      fun w =>
        (∃ u, localTraceSemantics STrue u ∧
          w = AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c :: u) ∨
        (∃ u, localTraceSemantics SFalse u ∧
          w = AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c :: u)
  | .recvWhile ys B SBody SExit =>
      fun w =>
        ∃ k : Nat,
        ∃ bodies : Fin k → LocalWord (C := C) (F := F) (Payload := Payload) A,
        (∀ i, localTraceSemantics SBody (bodies i)) ∧
        ∃ exitWord, localTraceSemantics SExit exitWord ∧
          w =
            concatLocalWords
              ((List.ofFn (fun i =>
                AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision true ys) B
                  :: bodies i)))
            ++ (AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision false ys) B
                  :: exitWord)
  | .localWhile c SBody SExit =>
      fun w =>
        ∃ k : Nat,
        ∃ bodies : Fin k → LocalWord (C := C) (F := F) (Payload := Payload) A,
        (∀ i, localTraceSemantics SBody (bodies i)) ∧
        ∃ exitWord, localTraceSemantics SExit exitWord ∧
          w =
            concatLocalWords
              ((List.ofFn (fun i =>
                AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c
                  :: bodies i)))
            ++ (AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c
                  :: exitWord)

notation "⟪" S "⟫ₗ" => localTraceSemantics S

/-- Prefix closure `⟪S⟫_A^pref` of the local trace semantics. -/
def localPrefixSemantics
    {L C F Payload : Type} [ControlPayload Payload] {A : L}
    (S : LocProg L C F Payload A) :
    LocalWordSet L C F Payload A :=
  fun u => ∃ v, localTraceSemantics S v ∧ IsPrefixWord u v

/-- Complete distributed semantics. -/
def distSemantics
    {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload]
    [ControlPayload Payload] [Fintype L]
    (D : DistProg L C F Payload) : WordTuple L C F Payload → Prop :=
  fun M => (∀ A, localTraceSemantics (D A) (M A)) ∧ IsCompleteMSC M

/-- Distributed prefix semantics. -/
def distPrefixSemantics
    {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload]
    [ControlPayload Payload] [Fintype L]
    (D : DistProg L C F Payload) : WordTuple L C F Payload → Prop :=
  fun M => (∀ A, localPrefixSemantics (D A) (M A)) ∧ IsMSC M

section LocalHelpers

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload]
  [ControlPayload Payload] [Fintype L]

private theorem concatLocalWords_foldl
    {A : L}
    (acc : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (ws : List (LocalWord (C := C) (F := F) (Payload := Payload) A)) :
    ws.foldl (· ++ ·) acc =
      acc ++ ws.foldl (· ++ ·) [] := by
  induction ws generalizing acc with
  | nil =>
      simp
  | cons w ws ih =>
      simp only [List.foldl_cons]
      rw [ih (acc ++ w)]
      simpa [List.append_assoc] using (ih w).symm

@[simp]
private theorem concatLocalWords_cons {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (ws : List (LocalWord (C := C) (F := F) (Payload := Payload) A)) :
    concatLocalWords (C := C) (F := F) (Payload := Payload) (w :: ws) =
      w ++ concatLocalWords (C := C) (F := F) (Payload := Payload) ws := by
  simpa [concatLocalWords] using
    (concatLocalWords_foldl (C := C) (F := F) (Payload := Payload) w ws)

private theorem prefix_append_split {A : L}
    (w u v : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h : IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) w (u ++ v)) :
    ∃ u' v',
      w = u' ++ v' ∧
      IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) u' u ∧
      IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) v' v ∧
      (v' ≠ [] → u' = u) := by
  rcases h with ⟨t, ht⟩
  by_cases hwu : w.length ≤ u.length
  · refine ⟨w, [], by simp, ?_, ?_, ?_⟩
    · have htake := congrArg (fun xs => xs.take w.length) ht
      have hw_eq : w = u.take w.length := by
        simpa [List.take_append_of_le_length hwu, List.take_left] using htake.symm
      refine ⟨u.drop w.length, ?_⟩
      rw [hw_eq, List.length_take_of_le hwu]
      exact (List.take_append_drop w.length u).symm
    · exact ⟨v, by simp⟩
    · intro hne
      contradiction
  · have hwu_lt : u.length < w.length := Nat.lt_of_not_ge hwu
    refine ⟨u, w.drop u.length, ?_, ⟨[], by simp⟩, ?_, ?_⟩
    · have htake := congrArg (fun xs => xs.take u.length) ht
      have hu_eq : u = w.take u.length := by
        simpa [List.take_left, List.take_append_of_le_length (Nat.le_of_lt hwu_lt)] using htake
      rw [hu_eq, List.length_take_of_le (Nat.le_of_lt hwu_lt)]
      exact (List.take_append_drop u.length w).symm
    · refine ⟨t, ?_⟩
      have hw_split : w = u ++ w.drop u.length := by
        have htake := congrArg (fun xs => xs.take u.length) ht
        have hu_eq : u = w.take u.length := by
          simpa [List.take_left, List.take_append_of_le_length (Nat.le_of_lt hwu_lt)] using htake
        rw [hu_eq, List.length_take_of_le (Nat.le_of_lt hwu_lt)]
        exact (List.take_append_drop u.length w).symm
      rw [hw_split] at ht
      have hcancel : v = w.drop u.length ++ t := by
        have hs : u ++ v = u ++ (w.drop u.length ++ t) := by
          simpa [List.append_assoc] using ht
        exact List.append_cancel_left hs
      exact hcancel
    · intro _
      rfl

private theorem prefix_cons_cancel {A : L}
    (a : AlphabetOf (C := C) (F := F) (Payload := Payload) A)
    (u v : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) (a :: u) (a :: v)) :
    IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) u v := by
  rcases h with ⟨t, ht⟩
  refine ⟨t, ?_⟩
  simpa using congrArg List.tail ht

theorem localTraceSemantics_localWhile_unfold {A : L}
    (c : C) (SBody SExit : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (.localWhile c SBody SExit) w ↔
      (∃ exitWord,
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) SExit exitWord ∧
        w = AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c :: exitWord) ∨
      (∃ bodyWord rest,
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) SBody bodyWord ∧
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (.localWhile c SBody SExit) rest ∧
        w = AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c :: bodyWord ++ rest) := by
  constructor
  · intro h
    rcases h with ⟨k, bodies, hBodies, exitWord, hExit, hw⟩
    cases k with
    | zero =>
        left
        refine ⟨exitWord, hExit, ?_⟩
        simpa [concatLocalWords] using hw
    | succ k =>
        right
        let bodiesTail : Fin k →
            LocalWord (C := C) (F := F) (Payload := Payload) A := fun i => bodies i.succ
        refine ⟨bodies 0, ?_, hBodies 0, ?_, ?_⟩
        · exact
            concatLocalWords (C := C) (F := F) (Payload := Payload)
              (List.ofFn (fun i : Fin k =>
                AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c
                  :: bodiesTail i))
            ++ (AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c
                  :: exitWord)
        · refine ⟨k, bodiesTail, ?_, exitWord, hExit, ?_⟩
          · intro i
            exact hBodies i.succ
          · rfl
        · simpa [bodiesTail, concatLocalWords_cons, List.ofFn_succ, List.append_assoc] using hw
  · intro h
    rcases h with ⟨exitWord, hExit, hw⟩ | ⟨bodyWord, rest, hBody, hRest, hw⟩
    · refine ⟨0, Fin.elim0, ?_, exitWord, hExit, ?_⟩
      · intro i
        exact Fin.elim0 i
      · simpa [concatLocalWords] using hw
    · rcases hRest with ⟨k, bodies, hBodies, exitWord, hExit, hRestEq⟩
      let bodies' : Fin (k + 1) → LocalWord (C := C) (F := F) (Payload := Payload) A :=
        Fin.cases bodyWord bodies
      refine ⟨k + 1, bodies', ?_, exitWord, hExit, ?_⟩
      · intro i
        exact Fin.cases hBody hBodies i
      · simpa [bodies', concatLocalWords_cons, List.ofFn_succ, hw, hRestEq, List.append_assoc]

theorem localTraceSemantics_recvWhile_unfold {A : L}
    (ys : Payload) (B : L) (SBody SExit : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (.recvWhile ys B SBody SExit) w ↔
      (∃ exitWord,
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) SExit exitWord ∧
        w = AlphabetOf.mkRecv (C := C) (F := F) A
              (ControlPayload.setDecision false ys) B :: exitWord) ∨
      (∃ bodyWord rest,
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) SBody bodyWord ∧
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (.recvWhile ys B SBody SExit) rest ∧
        w = AlphabetOf.mkRecv (C := C) (F := F) A
              (ControlPayload.setDecision true ys) B :: bodyWord ++ rest) := by
  constructor
  · intro h
    rcases h with ⟨k, bodies, hBodies, exitWord, hExit, hw⟩
    cases k with
    | zero =>
        left
        refine ⟨exitWord, hExit, ?_⟩
        simpa [concatLocalWords] using hw
    | succ k =>
        right
        let bodiesTail : Fin k →
            LocalWord (C := C) (F := F) (Payload := Payload) A := fun i => bodies i.succ
        refine ⟨bodies 0, ?_, hBodies 0, ?_, ?_⟩
        · exact
            concatLocalWords (C := C) (F := F) (Payload := Payload)
              (List.ofFn (fun i : Fin k =>
                AlphabetOf.mkRecv (C := C) (F := F) A
                  (ControlPayload.setDecision true ys) B :: bodiesTail i))
            ++ (AlphabetOf.mkRecv (C := C) (F := F) A
                  (ControlPayload.setDecision false ys) B :: exitWord)
        · refine ⟨k, bodiesTail, ?_, exitWord, hExit, ?_⟩
          · intro i
            exact hBodies i.succ
          · rfl
        · simpa [bodiesTail, concatLocalWords_cons, List.ofFn_succ, List.append_assoc] using hw
  · intro h
    rcases h with ⟨exitWord, hExit, hw⟩ | ⟨bodyWord, rest, hBody, hRest, hw⟩
    · refine ⟨0, Fin.elim0, ?_, exitWord, hExit, ?_⟩
      · intro i
        exact Fin.elim0 i
      · simpa [concatLocalWords] using hw
    · rcases hRest with ⟨k, bodies, hBodies, exitWord, hExit, hRestEq⟩
      let bodies' : Fin (k + 1) → LocalWord (C := C) (F := F) (Payload := Payload) A :=
        Fin.cases bodyWord bodies
      refine ⟨k + 1, bodies', ?_, exitWord, hExit, ?_⟩
      · intro i
        exact Fin.cases hBody hBodies i
      · simpa [bodies', concatLocalWords_cons, List.ofFn_succ, hw, hRestEq, List.append_assoc]

@[simp]
theorem localTraceSemantics_eps_nil {A : L} :
    localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (A := A) LocProg.eps [] := by
  simp [localTraceSemantics]

@[simp]
theorem localPrefixSemantics_eps_nil {A : L} :
    localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (A := A) LocProg.eps [] := by
  refine ⟨[], ?_, ?_⟩
  · simp [localTraceSemantics]
  · exact ⟨[] , by simp⟩

theorem localPrefixSemantics_seq_split {A : L}
    (S1 S2 : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (S1 ;;ₗ S2) w) :
    ∃ u v,
      w = u ++ v ∧
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) S1 u ∧
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) S2 v ∧
      (v ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) S1 u) := by
  rcases h with ⟨w', hw', hpref⟩
  rcases hw' with ⟨w1, w2, hw1, hw2, rfl⟩
  rcases prefix_append_split (L := L) (C := C) (F := F) (Payload := Payload) w w1 w2 hpref with
    ⟨u, v, huw, hu, hv, hfull⟩
  refine ⟨u, v, huw, ⟨w1, hw1, hu⟩, ⟨w2, hw2, hv⟩, ?_⟩
  intro hvne
  have hu_eq : u = w1 := hfull hvne
  simpa [hu_eq] using hw1

theorem localSeqBoundarySplit {A : L}
    (S1 S2 : LocProg L C F Payload A)
    (u v : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (hPref :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (S1 ;;ₗ S2) u)
    (hSide :
      v ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (S1 ;;ₗ S2) u) :
    ∃ u1 u2,
      u = u1 ++ u2 ∧
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) S1 u1 ∧
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) S2 u2 ∧
      (u2 ++ v ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) S1 u1) ∧
      (v ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) S2 u2) := by
  by_cases hv : v = []
  · rcases localPrefixSemantics_seq_split
      (L := L) (C := C) (F := F) (Payload := Payload) S1 S2 u hPref with
      ⟨u1, u2, hu, hu1, hu2, hfull⟩
    refine ⟨u1, u2, hu, hu1, hu2, ?_, ?_⟩
    · intro huv
      apply hfull
      simpa [hv] using huv
    · intro hvne
      contradiction
  · rcases hSide hv with ⟨u1, u2, hu1, hu2, hu⟩
    refine ⟨u1, u2, hu, ?_, ?_, ?_, ?_⟩
    · exact ⟨u1, hu1, ⟨[], by simp [IsPrefixWord]⟩⟩
    · exact ⟨u2, hu2, ⟨[], by simp [IsPrefixWord]⟩⟩
    · intro _
      exact hu1
    · intro _
      exact hu2

theorem localPrefixSemantics_localIf_cases {A : L}
    (c : C) (STrue SFalse : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.localIf c STrue SFalse) w) :
    w = [] ∨
      (∃ u,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) STrue u ∧
        w = AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c :: u) ∨
      (∃ u,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) SFalse u ∧
        w = AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c :: u) := by
  rcases h with ⟨w', hw', hpref⟩
  simp [localTraceSemantics] at hw'
  rcases hw' with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
  · rcases hpref with ⟨t, ht⟩
    cases w with
    | nil =>
        left
        rfl
    | cons hd tl =>
        right
        left
        refine ⟨tl, ?_, ?_⟩
        · refine ⟨u, hu, ?_⟩
          have hpair :
              AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c = hd ∧
                u = tl ++ t := by
            simpa [List.cons_append] using ht
          refine ⟨t, ?_⟩
          exact hpair.2
        · have hhd : hd = AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          have htl : u = tl ++ t := by
            simpa [hhd, List.cons_append] using ht
          subst hhd
          simp
  · rcases hpref with ⟨t, ht⟩
    cases w with
    | nil =>
        left
        rfl
    | cons hd tl =>
        right
        right
        refine ⟨tl, ?_, ?_⟩
        · refine ⟨u, hu, ?_⟩
          have hpair :
              AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c = hd ∧
                u = tl ++ t := by
            simpa [List.cons_append] using ht
          refine ⟨t, ?_⟩
          exact hpair.2
        · have hhd : hd = AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          have htl : u = tl ++ t := by
            simpa [hhd, List.cons_append] using ht
          subst hhd
          simp

theorem localPrefixSemantics_recvIf_cases {A : L}
    (ys : Payload) (B : L) (STrue SFalse : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.recvIf ys B STrue SFalse) w) :
    w = [] ∨
      (∃ u,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) STrue u ∧
        w = AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision true ys) B :: u) ∨
      (∃ u,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) SFalse u ∧
        w = AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision false ys) B :: u) := by
  rcases h with ⟨w', hw', hpref⟩
  simp [localTraceSemantics] at hw'
  rcases hw' with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
  · rcases hpref with ⟨t, ht⟩
    cases w with
    | nil =>
        left
        rfl
    | cons hd tl =>
        right
        left
        refine ⟨tl, ?_, ?_⟩
        · refine ⟨u, hu, ?_⟩
          have hpair :
              AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision true ys) B = hd ∧
                u = tl ++ t := by
            simpa [List.cons_append] using ht
          refine ⟨t, ?_⟩
          exact hpair.2
        · have hhd :
              hd =
                AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision true ys) B := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          have htl : u = tl ++ t := by
            simpa [hhd, List.cons_append] using ht
          subst hhd
          simp
  · rcases hpref with ⟨t, ht⟩
    cases w with
    | nil =>
        left
        rfl
    | cons hd tl =>
        right
        right
        refine ⟨tl, ?_, ?_⟩
        · refine ⟨u, hu, ?_⟩
          have hpair :
              AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision false ys) B = hd ∧
                u = tl ++ t := by
            simpa [List.cons_append] using ht
          refine ⟨t, ?_⟩
          exact hpair.2
        · have hhd :
              hd =
                AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision false ys) B := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          have htl : u = tl ++ t := by
            simpa [hhd, List.cons_append] using ht
          subst hhd
          simp

theorem localPrefixSemantics_localWhile_cases {A : L}
    (c : C) (SBody SExit : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.localWhile c SBody SExit) w) :
    w = [] ∨
      (∃ u,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) SExit u ∧
        w = AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c :: u) ∨
      (∃ u,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (SBody ;;ₗ .localWhile c SBody SExit) u ∧
        w = AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c :: u) := by
  rcases h with ⟨w', hw', hpref⟩
  rcases (localTraceSemantics_localWhile_unfold
      (L := L) (C := C) (F := F) (Payload := Payload) c SBody SExit w').mp hw' with
    ⟨exitWord, hExit, rfl⟩ | ⟨bodyWord, rest, hBody, hRest, rfl⟩
  · rcases hpref with ⟨t, ht⟩
    cases w with
    | nil => left; rfl
    | cons hd tl =>
        right; left
        refine ⟨tl, ?_, ?_⟩
        · refine ⟨exitWord, hExit, ?_⟩
          have hpair :
              AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c = hd ∧
                exitWord = tl ++ t := by
            simpa [List.cons_append] using ht
          exact ⟨t, hpair.2⟩
        · have hhd : hd = AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          subst hhd; simp
  · rcases hpref with ⟨t, ht⟩
    cases w with
    | nil => left; rfl
    | cons hd tl =>
        right; right
        refine ⟨tl, ?_, ?_⟩
        · have hhd : hd = AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          have htl : bodyWord ++ rest = tl ++ t := by
            simpa [hhd, List.cons_append] using ht
          refine ⟨bodyWord ++ rest, ?_, ⟨t, htl⟩⟩
          exact ⟨bodyWord, rest, hBody, hRest, rfl⟩
        · have hhd : hd = AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          subst hhd; simp

theorem localPrefixSemantics_recvWhile_cases {A : L}
    (ys : Payload) (B : L) (SBody SExit : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.recvWhile ys B SBody SExit) w) :
    w = [] ∨
      (∃ u,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload) SExit u ∧
        w = AlphabetOf.mkRecv (C := C) (F := F) A
              (ControlPayload.setDecision false ys) B :: u) ∨
      (∃ u,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (SBody ;;ₗ .recvWhile ys B SBody SExit) u ∧
        w = AlphabetOf.mkRecv (C := C) (F := F) A
              (ControlPayload.setDecision true ys) B :: u) := by
  rcases h with ⟨w', hw', hpref⟩
  rcases (localTraceSemantics_recvWhile_unfold
      (L := L) (C := C) (F := F) (Payload := Payload) ys B SBody SExit w').mp hw' with
    ⟨exitWord, hExit, rfl⟩ | ⟨bodyWord, rest, hBody, hRest, rfl⟩
  · rcases hpref with ⟨t, ht⟩
    cases w with
    | nil => left; rfl
    | cons hd tl =>
        right; left
        refine ⟨tl, ?_, ?_⟩
        · refine ⟨exitWord, hExit, ?_⟩
          have hpair :
              AlphabetOf.mkRecv (C := C) (F := F) A
                  (ControlPayload.setDecision false ys) B = hd ∧
                exitWord = tl ++ t := by
            simpa [List.cons_append] using ht
          exact ⟨t, hpair.2⟩
        · have hhd :
              hd = AlphabetOf.mkRecv (C := C) (F := F) A
                      (ControlPayload.setDecision false ys) B := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          subst hhd; simp
  · rcases hpref with ⟨t, ht⟩
    cases w with
    | nil => left; rfl
    | cons hd tl =>
        right; right
        refine ⟨tl, ?_, ?_⟩
        · have hhd :
              hd = AlphabetOf.mkRecv (C := C) (F := F) A
                      (ControlPayload.setDecision true ys) B := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          have htl : bodyWord ++ rest = tl ++ t := by
            simpa [hhd, List.cons_append] using ht
          refine ⟨bodyWord ++ rest, ?_, ⟨t, htl⟩⟩
          exact ⟨bodyWord, rest, hBody, hRest, rfl⟩
        · have hhd :
              hd = AlphabetOf.mkRecv (C := C) (F := F) A
                      (ControlPayload.setDecision true ys) B := by
            simpa [List.cons_append] using (congrArg List.head? ht).symm
          subst hhd; simp

theorem distPrefixSemantics_seq_split
    (P1 P2 : Prog L C F Payload)
    (M : WordTuple L C F Payload)
    (h :
      distPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (projectDist (L := L) (C := C) (F := F) (Payload := Payload) (P1 ;; P2)) M) :
    ∃ U V,
      M = U ∘ₘ V ∧
      (∀ X,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X P1) (U X)) ∧
      (∀ X,
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X P2) (V X)) ∧
      (∀ X, V X ≠ [] →
        localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X P1) (U X)) ∧
  IsMSC M := by
  classical
  rcases h with ⟨hPrefix, hMSC⟩
  have hSplit : ∀ X : L,
      ∃ u v,
        M X = u ++ v ∧
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X P1) u ∧
        localPrefixSemantics (L := L) (C := C) (F := F) (Payload := Payload)
          (project (L := L) (C := C) (F := F) (Payload := Payload) X P2) v ∧
        (v ≠ [] →
          localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
            (project (L := L) (C := C) (F := F) (Payload := Payload) X P1) u) := by
    intro X
    rcases localPrefixSemantics_seq_split
        (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X P1)
        (project (L := L) (C := C) (F := F) (Payload := Payload) X P2)
        (M X)
        (by simpa [projectDist] using hPrefix X) with
      ⟨u, v, huv, hu, hv, hfull⟩
    exact ⟨u, v, huv, hu, hv, hfull⟩
  let U : WordTuple L C F Payload := fun X => Classical.choose (hSplit X)
  let V : WordTuple L C F Payload := fun X => Classical.choose (Classical.choose_spec (hSplit X))
  refine ⟨U, V, ?_, ?_, ?_, ?_, hMSC⟩
  · funext X
    exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).1
  · intro X
    exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.1
  · intro X
    exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.2.1
  · intro X
    exact (Classical.choose_spec (Classical.choose_spec (hSplit X))).2.2.2

noncomputable def sendWordForTargets {A : L}
    (payload : Payload) :
    (targets : List L) →
      (∀ X, X ∈ targets → A ≠ X) →
      LocalWord (C := C) (F := F) (Payload := Payload) A
  | [], _ => []
  | X :: Xs, hTargets =>
      AlphabetOf.mkSend (C := C) (F := F) A payload X (hTargets X (by simp)) ::
        sendWordForTargets payload Xs (by
            intro Y hY
            exact hTargets Y (by simp [hY]))

theorem sendWordForTargets_eq_attach_map {A : L}
    (payload : Payload) :
    ∀ (targets : List L) (hTargets : ∀ X, X ∈ targets → A ≠ X),
      sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
        (A := A) payload targets hTargets =
      targets.attach.map
        (fun X => AlphabetOf.mkSend (C := C) (F := F) A payload X.1 (hTargets X.1 X.2))
  | [], _ => by
      simp [sendWordForTargets]
  | X :: Xs, hTargets => by
      have hTail : ∀ Y, Y ∈ Xs → A ≠ Y := by
        intro Y hY
        exact hTargets Y (by simp [hY])
      have ih := sendWordForTargets_eq_attach_map payload Xs hTail
      simpa [sendWordForTargets, ih]

theorem sendWordForTargets_filter_eq_self {A : L}
    (payload : Payload) :
    ∀ (targets : List L) (hTargets : ∀ X, X ∈ targets → A ≠ X),
      sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
        (A := A) payload
        (targets.filter (fun X => decide (A ≠ X)))
        (by
          intro X hX
          exact of_decide_eq_true ((List.mem_filter.mp hX).2)) =
      sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
        (A := A) payload targets hTargets
  | [], _ => by
      simp [sendWordForTargets]
  | X :: Xs, hTargets => by
      have hAX : A ≠ X := hTargets X (by simp)
      have hTail : ∀ Y, Y ∈ Xs → A ≠ Y := by
        intro Y hY
        exact hTargets Y (by simp [hY])
      have ih := sendWordForTargets_filter_eq_self payload Xs hTail
      simpa [sendWordForTargets, hAX, decide_not] using ih

/-- Deterministic trace of a sequential list of local sends. -/
private theorem localTraceSemantics_sendList {A : L}
    (payload : Payload) :
    ∀ (targets : List L) (hTargets : ∀ X, X ∈ targets → A ≠ X),
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload) (A := A)
        (seqLocList (targets.map (fun X => LocProg.send (A := A) payload X)))
        (sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
          (A := A) payload targets hTargets)
  | [], _ => by simp [seqLocList, localTraceSemantics, sendWordForTargets]
  | X :: Xs, hTargets => by
      refine ⟨[AlphabetOf.mkSend (C := C) (F := F) A payload X (hTargets X (by simp))],
        sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload)
          (A := A) payload Xs (by
            intro Y hY
            exact hTargets Y (by simp [hY])),
        ?_, ?_, by simp [sendWordForTargets]⟩
      · exact ⟨hTargets X (by simp), rfl⟩
      · simpa [seqLocList] using
          localTraceSemantics_sendList payload Xs (by
              intro Y hY
              exact hTargets Y (by simp [hY]))

/-- Concrete local word emitted by the decider's control broadcast. -/
noncomputable def controlBroadcastWord
    (A : L) (recips : L → Prop) (decision : Bool) :
    LocalWord (C := C) (F := F) (Payload := Payload) A := by
  classical
  exact sendWordForTargets (L := L) (C := C) (F := F) (Payload := Payload) (A := A)
    (ControlPayload.setDecision decision ControlPayload.ctrlPattern)
    (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips)
    (by
      intro X hX
      exact controlSendTargets_no_self (L := L) (C := C) (F := F) (Payload := Payload) hX)

/-- The control broadcast has the expected deterministic decider trace. -/
theorem controlBroadcast_trace
    (A : L) (recips : L → Prop) (decision : Bool) :
    localTraceSemantics
      (L := L) (C := C) (F := F) (Payload := Payload) (A := A)
      (controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload) A recips decision)
      (controlBroadcastWord (L := L) (C := C) (F := F) (Payload := Payload) A recips decision) := by
  classical
  unfold controlBroadcast controlBroadcastWord
  simpa using
    (localTraceSemantics_sendList
      (L := L) (C := C) (F := F) (Payload := Payload) (A := A)
      (ControlPayload.setDecision decision ControlPayload.ctrlPattern)
      (controlSendTargets (L := L) (C := C) (F := F) (Payload := Payload) A recips)
      (by
        intro X hX
        exact controlSendTargets_no_self (L := L) (C := C) (F := F) (Payload := Payload) hX))

theorem localTraceSemantics_seq_intro {A : L}
    (S1 S2 : LocProg L C F Payload A)
    (u1 u2 :
      LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h1 : localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) S1 u1)
    (h2 : localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) S2 u2) :
    localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
      (S1 ;;ₗ S2) (u1 ++ u2) := by
  exact ⟨u1, u2, h1, h2, rfl⟩


/-- If `A` does not participate structurally in `P`, the projected local
    program admits the empty trace. -/
theorem project_empty_trace_of_not_participating (A : L) :
    ∀ P : Prog L C F Payload, ¬ participationSet P A →
      localTraceSemantics
        (L := L) (C := C) (F := F) (Payload := Payload) (A := A)
        (project (L := L) (C := C) (F := F) (Payload := Payload) A P) []
  | .eps, _ => by simp [project, localTraceSemantics]
  | .msg X xs Y ys h, hA => by
      have hAX : A ≠ X := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      have hAY : A ≠ Y := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      simp [project, hAX, hAY, localTraceSemantics]
  | .act X ys f xs, hA => by
      have hAX : A ≠ X := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      simp [project, hAX, localTraceSemantics]
  | .seq P1 P2, hA => by
      have hA1 : ¬ participationSet P1 A := by
        intro h
        exact hA (by simp [participationSet, h])
      have hA2 : ¬ participationSet P2 A := by
        intro h
        exact hA (by simp [participationSet, h])
      refine ⟨[], [], project_empty_trace_of_not_participating A P1 hA1,
        project_empty_trace_of_not_participating A P2 hA2, by simp⟩
  | .ite c B PTrue PFalse, hA => by
      have hAB : A ≠ B := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      have hRecip : ¬ ifRecipients B PTrue PFalse A := by
        intro h
        exact hA (by
          rcases h with ⟨hAB', hBranch⟩
          simp [ifRecipients, participationSet, hAB', hBranch])
      simp [project, hAB, hRecip, localTraceSemantics]
  | .whileLoop c B PBody PExit, hA => by
      have hAB : A ≠ B := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      have hRecip : ¬ whileRecipients B PBody PExit A := by
        intro h
        exact hA (by
          rcases h with ⟨hAB', hBranch⟩
          simp [whileRecipients, participationSet, hAB', hBranch])
      simp [project, hAB, hRecip, localTraceSemantics]

/-- If `A` does not participate structurally in `P`, every projected local
    trace on `A` is empty. -/
theorem project_trace_nil_of_not_participating (A : L) :
    ∀ P : Prog L C F Payload, ¬ participationSet P A →
      ∀ w,
        localTraceSemantics
          (L := L) (C := C) (F := F) (Payload := Payload) (A := A)
          (project (L := L) (C := C) (F := F) (Payload := Payload) A P) w →
        w = []
  | .eps, _, w, hw => by simpa [project, localTraceSemantics] using hw
  | .msg X xs Y ys h, hA, w, hw => by
      have hAX : A ≠ X := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      have hAY : A ≠ Y := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      simpa [project, hAX, hAY, localTraceSemantics] using hw
  | .act X ys f xs, hA, w, hw => by
      have hAX : A ≠ X := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      simpa [project, hAX, localTraceSemantics] using hw
  | .seq P1 P2, hA, w, hw => by
      rcases hw with ⟨u1, u2, hu1, hu2, rfl⟩
      have hA1 : ¬ participationSet P1 A := by
        intro h
        exact hA (by simp [participationSet, h])
      have hA2 : ¬ participationSet P2 A := by
        intro h
        exact hA (by simp [participationSet, h])
      rw [project_trace_nil_of_not_participating A P1 hA1 _ hu1,
        project_trace_nil_of_not_participating A P2 hA2 _ hu2]
      simp
  | .ite c B PTrue PFalse, hA, w, hw => by
      have hAB : A ≠ B := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      have hRecip : ¬ ifRecipients B PTrue PFalse A := by
        intro h
        exact hA (by
          rcases h with ⟨hAB', hBranch⟩
          simp [ifRecipients, participationSet, hAB', hBranch])
      simpa [project, hAB, hRecip, localTraceSemantics] using hw
  | .whileLoop c B PBody PExit, hA, w, hw => by
      have hAB : A ≠ B := by
        intro hEq
        exact hA (by simp [participationSet, hEq])
      have hRecip : ¬ whileRecipients B PBody PExit A := by
        intro h
        exact hA (by
          rcases h with ⟨hAB', hBranch⟩
          simp [whileRecipients, participationSet, hAB', hBranch])
      simpa [project, hAB, hRecip, localTraceSemantics] using hw

theorem localTraceSemantics_localIf_ne_nil {A : L}
    (c : C) (STrue SFalse : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.localIf c STrue SFalse) w) :
    w ≠ [] := by
  simp [localTraceSemantics] at h
  rcases h with ⟨u, _, rfl⟩ | ⟨u, _, rfl⟩ <;> simp

theorem localTraceSemantics_recvIf_ne_nil {A : L}
    (ys : Payload) (B : L) (STrue SFalse : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.recvIf ys B STrue SFalse) w) :
    w ≠ [] := by
  simp [localTraceSemantics] at h
  rcases h with ⟨u, _, rfl⟩ | ⟨u, _, rfl⟩ <;> simp

theorem localTraceSemantics_localWhile_ne_nil {A : L}
    (c : C) (SBody SExit : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.localWhile c SBody SExit) w) :
    w ≠ [] := by
  rcases h with ⟨k, bodies, hBodies, exitWord, hExit, hw⟩
  intro hNil
  rw [hw] at hNil
  simp at hNil

theorem localTraceSemantics_recvWhile_ne_nil {A : L}
    (ys : Payload) (B : L) (SBody SExit : LocProg L C F Payload A)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (h :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (.recvWhile ys B SBody SExit) w) :
    w ≠ [] := by
  rcases h with ⟨k, bodies, hBodies, exitWord, hExit, hw⟩
  intro hNil
  rw [hw] at hNil
  simp at hNil

/-- Participating lifelines of a projected `if`-program have nonempty complete
    local traces because every such trace starts with the control decision. -/
theorem project_trace_ne_nil_of_if_participating
    (A : L) (c : C) (B : L) (PTrue PFalse : Prog L C F Payload)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (hPart : participationSet (.ite c B PTrue PFalse) A)
    (hTrace :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) A
          (.ite c B PTrue PFalse)) w) :
    w ≠ [] := by
  by_cases hAB : A = B
  · subst hAB
    have hProj :
        project (L := L) (C := C) (F := F) (Payload := Payload) A
          (.ite c A PTrue PFalse) =
          LocProg.localIf c
            ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                A
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                true)
              ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue)
            ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                A
                (ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PTrue PFalse)
                false)
              ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse) := by
      simp [project]
    rw [hProj] at hTrace
    exact localTraceSemantics_localIf_ne_nil
      (L := L) (C := C) (F := F) (Payload := Payload) c _ _ _ hTrace
  · have hBranch : participationSet PTrue A ∨ participationSet PFalse A := by
      simpa [participationSet_ite, hAB] using hPart
    have hRecip :
        ifRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PTrue PFalse A := by
      exact ⟨hAB, hBranch⟩
    have hProj :
        project (L := L) (C := C) (F := F) (Payload := Payload) A
          (.ite c B PTrue PFalse) =
          LocProg.recvIf ControlPayload.ctrlPattern B
            (project (L := L) (C := C) (F := F) (Payload := Payload) A PTrue)
            (project (L := L) (C := C) (F := F) (Payload := Payload) A PFalse) := by
      simp [project, hAB, hRecip]
    rw [hProj] at hTrace
    exact localTraceSemantics_recvIf_ne_nil
      (L := L) (C := C) (F := F) (Payload := Payload)
      ControlPayload.ctrlPattern B _ _ _ hTrace

/-- Participating lifelines of a projected `while`-program have nonempty
    complete local traces because every such trace starts with a control
    decision. -/
theorem project_trace_ne_nil_of_while_participating
    (A : L) (c : C) (B : L) (PBody PExit : Prog L C F Payload)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (hPart : participationSet (.whileLoop c B PBody PExit) A)
    (hTrace :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload)
        (project (L := L) (C := C) (F := F) (Payload := Payload) A
          (.whileLoop c B PBody PExit)) w) :
    w ≠ [] := by
  by_cases hAB : A = B
  · subst hAB
    have hProj :
        project (L := L) (C := C) (F := F) (Payload := Payload) A
          (.whileLoop c A PBody PExit) =
          LocProg.localWhile c
            ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                A
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)
                true)
              ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PBody)
            ((controlBroadcast (L := L) (C := C) (F := F) (Payload := Payload)
                A
                (whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) A PBody PExit)
                false)
              ;;ₗ project (L := L) (C := C) (F := F) (Payload := Payload) A PExit) := by
      simp [project]
    rw [hProj] at hTrace
    exact localTraceSemantics_localWhile_ne_nil
      (L := L) (C := C) (F := F) (Payload := Payload) c _ _ _ hTrace
  · have hBranch : participationSet PBody A ∨ participationSet PExit A := by
      simpa [participationSet_while, hAB] using hPart
    have hRecip :
        whileRecipients (L := L) (C := C) (F := F) (Payload := Payload) B PBody PExit A := by
      exact ⟨hAB, hBranch⟩
    have hProj :
        project (L := L) (C := C) (F := F) (Payload := Payload) A
          (.whileLoop c B PBody PExit) =
          LocProg.recvWhile ControlPayload.ctrlPattern B
            (project (L := L) (C := C) (F := F) (Payload := Payload) A PBody)
            (project (L := L) (C := C) (F := F) (Payload := Payload) A PExit) := by
      simp [project, hAB, hRecip]
    rw [hProj] at hTrace
    exact localTraceSemantics_recvWhile_ne_nil
      (L := L) (C := C) (F := F) (Payload := Payload)
      ControlPayload.ctrlPattern B _ _ _ hTrace

end LocalHelpers

section TracePrefix

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload] [Fintype L]
variable [ControlPayloadSpec Payload]

private theorem prefix_list_of_append_eq_left {α : Type}
    {u1 u2 v1 v2 : List α}
    (hEq : u1 ++ u2 = v1 ++ v2)
    (hLen : u1.length ≤ v1.length) :
    ∃ t, v1 = u1 ++ t := by
  refine ⟨v1.drop u1.length, ?_⟩
  have hTake := congrArg (fun xs => xs.take u1.length) hEq
  have hu1 : u1 = v1.take u1.length := by
    simpa [List.take_left, List.take_append_of_le_length hLen] using hTake
  rw [hu1, List.length_take_of_le hLen]
  exact (List.take_append_drop u1.length v1).symm

private theorem prefix_list_of_append_eq_right {α : Type}
    {u1 u2 v1 v2 : List α}
    (hEq : u1 ++ u2 = v1 ++ v2)
    (hLen : v1.length ≤ u1.length) :
    ∃ t, u1 = v1 ++ t := by
  refine ⟨u1.drop v1.length, ?_⟩
  have hTake := congrArg (fun xs => xs.take v1.length) hEq
  have hv1 : v1 = u1.take v1.length := by
    simpa [List.take_left, List.take_append_of_le_length hLen] using hTake.symm
  rw [hv1, List.length_take_of_le hLen]
  exact (List.take_append_drop v1.length u1).symm

private theorem recvDecisionHead_true_ne_false {A B : L} {ys : Payload} :
    AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision true ys) B ≠
      AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision false ys) B := by
  intro h
  have hVal :
      Letter.recvLetter A (ControlPayload.setDecision true ys) B =
        Letter.recvLetter A (ControlPayload.setDecision false ys) B :=
    congrArg Subtype.val h
  have hPayload :
      ControlPayload.setDecision true ys = ControlPayload.setDecision false ys := by
    simpa using hVal
  have : true = false := controlPayload_setDecision_eq (Payload := Payload) hPayload
  cases this

private theorem recvDecisionHead_false_ne_true {A B : L} {ys : Payload} :
    AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision false ys) B ≠
      AlphabetOf.mkRecv (C := C) (F := F) A (ControlPayload.setDecision true ys) B := by
  intro h
  have hVal :
      Letter.recvLetter A (ControlPayload.setDecision false ys) B =
        Letter.recvLetter A (ControlPayload.setDecision true ys) B :=
    congrArg Subtype.val h
  have hPayload :
      ControlPayload.setDecision false ys = ControlPayload.setDecision true ys := by
    simpa using hVal
  have : false = true := controlPayload_setDecision_eq (Payload := Payload) hPayload
  cases this

private theorem ifHead_true_ne_false {A : L} {c : C} :
    AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c ≠
      AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c := by
  intro h
  have hVal :
      Letter.ifTrueLetter c A = Letter.ifFalseLetter c A :=
    congrArg Subtype.val h
  cases hVal

private theorem ifHead_false_ne_true {A : L} {c : C} :
    AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c ≠
      AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c := by
  intro h
  have hVal :
      Letter.ifFalseLetter c A = Letter.ifTrueLetter c A :=
    congrArg Subtype.val h
  cases hVal

private theorem whileHead_true_ne_false {A : L} {c : C} :
    AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c ≠
      AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c := by
  intro h
  have hVal :
      Letter.whileTrueLetter c A = Letter.whileFalseLetter c A :=
    congrArg Subtype.val h
  cases hVal

private theorem whileHead_false_ne_true {A : L} {c : C} :
    AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c ≠
      AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c := by
  intro h
  have hVal :
      Letter.whileFalseLetter c A = Letter.whileTrueLetter c A :=
    congrArg Subtype.val h
  cases hVal

private theorem lex_decrease_of_length_le_size_lt
    {m n s t : Nat}
    (hLen : m ≤ n) (hSize : s < t) :
    Prod.Lex (fun x y => x < y) (fun x y => x < y) (m, s) (n, t) := by
  by_cases hEq : m = n
  · subst hEq
    exact Prod.Lex.right _ hSize
  · exact Prod.Lex.left _ _ (Nat.lt_of_le_of_ne hLen hEq)

/-- Complete local traces are prefix-free. -/
theorem localTraceSemantics_prefix_eq {A : L}
    (S : LocProg L C F Payload A)
    {u v : LocalWord (C := C) (F := F) (Payload := Payload) A}
    (hu :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) S u)
    (hv :
      localTraceSemantics (L := L) (C := C) (F := F) (Payload := Payload) S v)
    (hp :
      IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) u v) :
    u = v := by
  cases S with
  | eps =>
      simp [localTraceSemantics] at hu hv
      subst u
      subst v
      rfl
  | send xs B =>
      rcases hu with ⟨_, rfl⟩
      rcases hv with ⟨_, rfl⟩
      rfl
  | recv ys B =>
      simp [localTraceSemantics] at hu hv
      subst u
      subst v
      rfl
  | act ys f xs =>
      simp [localTraceSemantics] at hu hv
      subst u
      subst v
      rfl
  | seq S1 S2 =>
      rcases hu with ⟨u1, u2, hu1, hu2, rfl⟩
      rcases hv with ⟨v1, v2, hv1, hv2, rfl⟩
      rcases hp with ⟨t, ht⟩
      cases Nat.le_total u1.length v1.length with
      | inl hLen =>
          rcases prefix_list_of_append_eq_left
              (u1 := u1) (u2 := u2 ++ t) (v1 := v1) (v2 := v2)
              (by simpa [List.append_assoc] using ht.symm) hLen with
            ⟨t1, hPref1⟩
          have hEq1 : u1 = v1 :=
            localTraceSemantics_prefix_eq S1 hu1 hv1 ⟨t1, hPref1⟩
          have hPref2 :
              IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) u2 v2 := by
            refine ⟨t, ?_⟩
            simpa [hEq1, List.append_assoc] using ht
          have hEq2 : u2 = v2 :=
            localTraceSemantics_prefix_eq S2 hu2 hv2 hPref2
          simp [hEq1, hEq2]
      | inr hLen =>
          rcases prefix_list_of_append_eq_right
              (u1 := u1) (u2 := u2 ++ t) (v1 := v1) (v2 := v2)
              (by simpa [List.append_assoc] using ht.symm) hLen with
            ⟨t1, hPref1⟩
          have hEq1 : v1 = u1 :=
            localTraceSemantics_prefix_eq S1 hv1 hu1 ⟨t1, hPref1⟩
          have hPref2 :
              IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) u2 v2 := by
            refine ⟨t, ?_⟩
            simpa [hEq1, List.append_assoc] using ht
          have hEq2 : u2 = v2 :=
            localTraceSemantics_prefix_eq S2 hu2 hv2 hPref2
          simp [hEq1, hEq2]
  | recvIf ys B STrue SFalse =>
      simp [localTraceSemantics] at hu hv
      rcases hu with ⟨u', hu', rfl⟩ | ⟨u', hu', rfl⟩
      · rcases hv with ⟨v', hv', rfl⟩ | ⟨v', hv', rfl⟩
        · have hp' :=
            prefix_cons_cancel (L := L) (C := C) (F := F) (Payload := Payload)
              (AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision true ys) B)
              u' v' hp
          have hEq :=
            localTraceSemantics_prefix_eq STrue hu' hv' hp'
          simp [hEq]
        · rcases hp with ⟨t, ht⟩
          have hHead :
              AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision false ys) B =
              AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision true ys) B :=
            (List.cons.inj ht).1
          exact False.elim (recvDecisionHead_false_ne_true (ys := ys) hHead)
      · rcases hv with ⟨v', hv', rfl⟩ | ⟨v', hv', rfl⟩
        · rcases hp with ⟨t, ht⟩
          have hHead :
              AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision true ys) B =
              AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision false ys) B :=
            (List.cons.inj ht).1
          exact False.elim (recvDecisionHead_true_ne_false (ys := ys) hHead)
        · have hp' :=
            prefix_cons_cancel (L := L) (C := C) (F := F) (Payload := Payload)
              (AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision false ys) B)
              u' v' hp
          have hEq :=
            localTraceSemantics_prefix_eq SFalse hu' hv' hp'
          simp [hEq]
  | localIf c STrue SFalse =>
      simp [localTraceSemantics] at hu hv
      rcases hu with ⟨u', hu', rfl⟩ | ⟨u', hu', rfl⟩
      · rcases hv with ⟨v', hv', rfl⟩ | ⟨v', hv', rfl⟩
        · have hp' :=
            prefix_cons_cancel (L := L) (C := C) (F := F) (Payload := Payload)
              (AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c)
              u' v' hp
          have hEq :=
            localTraceSemantics_prefix_eq STrue hu' hv' hp'
          simp [hEq]
        · rcases hp with ⟨t, ht⟩
          have hHead :
              AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c =
              AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c :=
            (List.cons.inj ht).1
          exact False.elim (ifHead_false_ne_true (c := c) hHead)
      · rcases hv with ⟨v', hv', rfl⟩ | ⟨v', hv', rfl⟩
        · rcases hp with ⟨t, ht⟩
          have hHead :
              AlphabetOf.mkIfTrue (C := C) (F := F) (Payload := Payload) A c =
              AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c :=
            (List.cons.inj ht).1
          exact False.elim (ifHead_true_ne_false (c := c) hHead)
        · have hp' :=
            prefix_cons_cancel (L := L) (C := C) (F := F) (Payload := Payload)
              (AlphabetOf.mkIfFalse (C := C) (F := F) (Payload := Payload) A c)
              u' v' hp
          have hEq :=
            localTraceSemantics_prefix_eq SFalse hu' hv' hp'
          simp [hEq]
  | recvWhile ys B SBody SExit =>
      have hu' :=
        (localTraceSemantics_recvWhile_unfold
          (L := L) (C := C) (F := F) (Payload := Payload)
          (A := A) ys B SBody SExit u).mp hu
      have hv' :=
        (localTraceSemantics_recvWhile_unfold
          (L := L) (C := C) (F := F) (Payload := Payload)
          (A := A) ys B SBody SExit v).mp hv
      rcases hu' with ⟨uExit, huExit, rfl⟩ | ⟨uBody, uRest, huBody, huRest, rfl⟩
      · rcases hv' with ⟨vExit, hvExit, rfl⟩ | ⟨vBody, vRest, hvBody, hvRest, rfl⟩
        · have hp' :=
            prefix_cons_cancel (L := L) (C := C) (F := F) (Payload := Payload)
              (AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision false ys) B)
              uExit vExit hp
          have hEq :=
            localTraceSemantics_prefix_eq SExit huExit hvExit hp'
          simp [hEq]
        · rcases hp with ⟨t, ht⟩
          have hHead :
              AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision true ys) B =
              AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision false ys) B :=
            (List.cons.inj ht).1
          exact False.elim (recvDecisionHead_true_ne_false (ys := ys) hHead)
      · rcases hv' with ⟨vExit, hvExit, rfl⟩ | ⟨vBody, vRest, hvBody, hvRest, rfl⟩
        · rcases hp with ⟨t, ht⟩
          have hHead :
              AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision false ys) B =
              AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision true ys) B :=
            (List.cons.inj ht).1
          exact False.elim (recvDecisionHead_false_ne_true (ys := ys) hHead)
        · have hp' :=
            prefix_cons_cancel (L := L) (C := C) (F := F) (Payload := Payload)
              (AlphabetOf.mkRecv (C := C) (F := F) A
                (ControlPayload.setDecision true ys) B)
              (uBody ++ uRest) (vBody ++ vRest) hp
          rcases hp' with ⟨t, ht⟩
          cases Nat.le_total uBody.length vBody.length with
          | inl hLen =>
              rcases prefix_list_of_append_eq_left
                  (u1 := uBody) (u2 := uRest ++ t) (v1 := vBody) (v2 := vRest)
                  (by simpa [List.append_assoc] using ht.symm) hLen with
                ⟨t1, hPref1⟩
              have hEqBody : uBody = vBody :=
                localTraceSemantics_prefix_eq SBody huBody hvBody ⟨t1, hPref1⟩
              have hPrefRest :
                  IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) uRest vRest := by
                refine ⟨t, ?_⟩
                simpa [hEqBody, List.append_assoc] using ht
              have hEqRest :
                  uRest = vRest :=
                localTraceSemantics_prefix_eq (.recvWhile ys B SBody SExit)
                  huRest hvRest hPrefRest
              simp [hEqBody, hEqRest]
          | inr hLen =>
              rcases prefix_list_of_append_eq_right
                  (u1 := uBody) (u2 := uRest ++ t) (v1 := vBody) (v2 := vRest)
                  (by simpa [List.append_assoc] using ht.symm) hLen with
                ⟨t1, hPref1⟩
              have hEqBody : vBody = uBody :=
                localTraceSemantics_prefix_eq SBody hvBody huBody ⟨t1, hPref1⟩
              have hPrefRest :
                  IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) uRest vRest := by
                refine ⟨t, ?_⟩
                simpa [hEqBody, List.append_assoc] using ht
              have hEqRest :
                  uRest = vRest :=
                localTraceSemantics_prefix_eq (.recvWhile ys B SBody SExit)
                  huRest hvRest hPrefRest
              simp [hEqBody, hEqRest]
  | localWhile c SBody SExit =>
      have hu' :=
        (localTraceSemantics_localWhile_unfold
          (L := L) (C := C) (F := F) (Payload := Payload)
          (A := A) c SBody SExit u).mp hu
      have hv' :=
        (localTraceSemantics_localWhile_unfold
          (L := L) (C := C) (F := F) (Payload := Payload)
          (A := A) c SBody SExit v).mp hv
      rcases hu' with ⟨uExit, huExit, rfl⟩ | ⟨uBody, uRest, huBody, huRest, rfl⟩
      · rcases hv' with ⟨vExit, hvExit, rfl⟩ | ⟨vBody, vRest, hvBody, hvRest, rfl⟩
        · have hp' :=
            prefix_cons_cancel (L := L) (C := C) (F := F) (Payload := Payload)
              (AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c)
              uExit vExit hp
          have hEq :=
            localTraceSemantics_prefix_eq SExit huExit hvExit hp'
          simp [hEq]
        · rcases hp with ⟨t, ht⟩
          have hHead :
              AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c =
              AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c :=
            (List.cons.inj ht).1
          exact False.elim (whileHead_true_ne_false (c := c) hHead)
      · rcases hv' with ⟨vExit, hvExit, rfl⟩ | ⟨vBody, vRest, hvBody, hvRest, rfl⟩
        · rcases hp with ⟨t, ht⟩
          have hHead :
              AlphabetOf.mkWhileFalse (C := C) (F := F) (Payload := Payload) A c =
              AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c :=
            (List.cons.inj ht).1
          exact False.elim (whileHead_false_ne_true (c := c) hHead)
        · have hp' :=
            prefix_cons_cancel (L := L) (C := C) (F := F) (Payload := Payload)
              (AlphabetOf.mkWhileTrue (C := C) (F := F) (Payload := Payload) A c)
              (uBody ++ uRest) (vBody ++ vRest) hp
          rcases hp' with ⟨t, ht⟩
          cases Nat.le_total uBody.length vBody.length with
          | inl hLen =>
              rcases prefix_list_of_append_eq_left
                  (u1 := uBody) (u2 := uRest ++ t) (v1 := vBody) (v2 := vRest)
                  (by simpa [List.append_assoc] using ht.symm) hLen with
                ⟨t1, hPref1⟩
              have hEqBody : uBody = vBody :=
                localTraceSemantics_prefix_eq SBody huBody hvBody ⟨t1, hPref1⟩
              have hPrefRest :
                  IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) uRest vRest := by
                refine ⟨t, ?_⟩
                simpa [hEqBody, List.append_assoc] using ht
              have hEqRest :
                  uRest = vRest :=
                localTraceSemantics_prefix_eq (.localWhile c SBody SExit)
                  huRest hvRest hPrefRest
              simp [hEqBody, hEqRest]
          | inr hLen =>
              rcases prefix_list_of_append_eq_right
                  (u1 := uBody) (u2 := uRest ++ t) (v1 := vBody) (v2 := vRest)
                  (by simpa [List.append_assoc] using ht.symm) hLen with
                ⟨t1, hPref1⟩
              have hEqBody : vBody = uBody :=
                localTraceSemantics_prefix_eq SBody hvBody huBody ⟨t1, hPref1⟩
              have hPrefRest :
                  IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload) uRest vRest := by
                refine ⟨t, ?_⟩
                simpa [hEqBody, List.append_assoc] using ht
              have hEqRest :
                  uRest = vRest :=
                localTraceSemantics_prefix_eq (.localWhile c SBody SExit)
                  huRest hvRest hPrefRest
              simp [hEqBody, hEqRest]
termination_by sizeOf S + v.length
decreasing_by
  all_goals
    subst_vars
    simp_wf
    first
      | simp [List.length_append]
      | skip
    first
      | have hLenEq := congrArg List.length ht
        simp [List.length_append] at hLenEq
        omega
      | omega

end TracePrefix
