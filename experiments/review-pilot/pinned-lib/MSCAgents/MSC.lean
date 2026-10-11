/-
  MSCAgents/MSC.lean
  ==================
  Formalization of Definition 6 (def:msc) from:
  "Provable Coordination for LLM Agents via Message Sequence Charts"

  Paper section: sec:semantics

  From the paper (def:msc):
    An MSC is a tuple of local words M = (w_A)_{A ∈ 𝓛} such that:
    1. No unmatched receive events (|rcv| ≤ |snd| per channel)
    2. All matched pairs are label-compatible
    3. The causality relation (local order + FIFO) is acyclic

    M is complete if |snd| = |rcv| for every channel.

  This file defines:
    - The concrete MSC predicate and completeness predicate
    - The IsMSCPredicate instance (from MSCConcat.lean)
    - Completeness theorems for canonical MSCs
-/

import MSCAgents.Alphabets
import MSCAgents.MSCConcat
import MSCAgents.FIFORelation
import MSCAgents.CanonicalMSC
import MSCAgents.PayloadMatching
import MSCAgents.ControlPayload

/-- A local-word prefix relation.  This is used both by the local semantics and
    by the MSC-level decision-prefix stripping lemma. -/
def IsPrefixWord {L C F Payload : Type} {A : L}
    (u v : LocalWord (C := C) (F := F) (Payload := Payload) A) : Prop :=
  ∃ t, v = u ++ t

------------------------------------------------------------------------
-- The concrete MSC predicate
------------------------------------------------------------------------

section MSCPredicate

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload]

/-- Collect the payloads of all sends on channel `A → B`. -/
def sendPayloads {A : L} (B : L) :
    List (AlphabetOf (C := C) (F := F) (Payload := Payload) A) → List Payload
  | [] => []
  | hd :: tl =>
      let rest := sendPayloads B tl
      match hd.val with
      | .sendLetter _ xs tgt _ =>
          if tgt = B then xs :: rest else rest
      | _ => rest

/-- Collect the payloads of all receives on channel `A → B`. -/
def recvPayloads {B : L} (A : L) :
    List (AlphabetOf (C := C) (F := F) (Payload := Payload) B) → List Payload
  | [] => []
  | hd :: tl =>
      let rest := recvPayloads A tl
      match hd.val with
      | .recvLetter _ ys src =>
          if src = A then ys :: rest else rest
      | _ => rest

/-- Channelwise label compatibility: each FIFO-matched send/recv payload pair
    satisfies the payload-matching relation. -/
def channelLabelCompatible (M : WordTuple L C F Payload) (A B : L) : Prop :=
  ∀ p ∈ List.zip (sendPayloads B (M A)) (recvPayloads A (M B)),
    PayloadCompatible Payload p.1 p.2

/-- Condition (2) of Definition `def:msc`: all FIFO-matched pairs are
    payload-compatible. -/
def matchedLabelsCompatible (M : WordTuple L C F Payload) : Prop :=
  ∀ A B : L, channelLabelCompatible M A B

/-- Send-payload collection distributes over concatenation. -/
@[simp]
theorem sendPayloads_append {A : L} (B : L)
    (w1 w2 : List (AlphabetOf (C := C) (F := F) (Payload := Payload) A)) :
    sendPayloads B (w1 ++ w2) = sendPayloads B w1 ++ sendPayloads B w2 := by
  induction w1 with
  | nil => simp [sendPayloads]
  | cons hd tl ih =>
      cases hℓ : hd.val <;> simp [sendPayloads, ih, hℓ]
      split <;> simp [ih]

/-- Receive-payload collection distributes over concatenation. -/
@[simp]
theorem recvPayloads_append {B : L} (A : L)
    (w1 w2 : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B)) :
    recvPayloads A (w1 ++ w2) = recvPayloads A w1 ++ recvPayloads A w2 := by
  induction w1 with
  | nil => simp [recvPayloads]
  | cons hd tl ih =>
      cases hℓ : hd.val <;> simp [recvPayloads, ih, hℓ]
      split <;> simp [ih]

/-- The send-payload list has one entry per send event. -/
@[simp]
theorem sendPayloads_length {A : L} (B : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) A)) :
    (sendPayloads B w).length = countSends B w := by
  induction w with
  | nil => simp [sendPayloads, countSends]
  | cons hd tl ih =>
      cases hℓ : hd.val with
      | sendLetter owner xs tgt hneq =>
          by_cases hsend : tgt = B
          · simp [sendPayloads, countSends, ih, hℓ, hsend, Letter.isSendTo]
            omega
          · simp [sendPayloads, countSends, ih, hℓ, hsend, Letter.isSendTo]
      | recvLetter owner ys src =>
          simp [sendPayloads, countSends, ih, hℓ, Letter.isSendTo]
      | actLetter owner ys f xs =>
          simp [sendPayloads, countSends, ih, hℓ, Letter.isSendTo]
      | ifTrueLetter c owner =>
          simp [sendPayloads, countSends, ih, hℓ, Letter.isSendTo]
      | ifFalseLetter c owner =>
          simp [sendPayloads, countSends, ih, hℓ, Letter.isSendTo]
      | whileTrueLetter c owner =>
          simp [sendPayloads, countSends, ih, hℓ, Letter.isSendTo]
      | whileFalseLetter c owner =>
          simp [sendPayloads, countSends, ih, hℓ, Letter.isSendTo]

/-- The receive-payload list has one entry per receive event. -/
@[simp]
theorem recvPayloads_length {B : L} (A : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B)) :
    (recvPayloads A w).length = countRecvs A w := by
  induction w with
  | nil => simp [recvPayloads, countRecvs]
  | cons hd tl ih =>
      cases hℓ : hd.val with
      | recvLetter owner ys src =>
          by_cases hrecv : src = A
          · simp [recvPayloads, countRecvs, ih, hℓ, hrecv, Letter.isRecvFrom]
            omega
          · simp [recvPayloads, countRecvs, ih, hℓ, hrecv, Letter.isRecvFrom]
      | sendLetter owner xs tgt hneq =>
          simp [recvPayloads, countRecvs, ih, hℓ, Letter.isRecvFrom]
      | actLetter owner ys f xs =>
          simp [recvPayloads, countRecvs, ih, hℓ, Letter.isRecvFrom]
      | ifTrueLetter c owner =>
          simp [recvPayloads, countRecvs, ih, hℓ, Letter.isRecvFrom]
      | ifFalseLetter c owner =>
          simp [recvPayloads, countRecvs, ih, hℓ, Letter.isRecvFrom]
      | whileTrueLetter c owner =>
          simp [recvPayloads, countRecvs, ih, hℓ, Letter.isRecvFrom]
      | whileFalseLetter c owner =>
          simp [recvPayloads, countRecvs, ih, hℓ, Letter.isRecvFrom]

/-- Zipping concatenated lists splits across the boundary when the prefixes
    have the same length. -/
private theorem zip_append_eq_of_length_eq {α β : Type}
    (xs1 xs2 : List α) (ys1 ys2 : List β)
    (h : xs1.length = ys1.length) :
    List.zip (xs1 ++ xs2) (ys1 ++ ys2) = List.zip xs1 ys1 ++ List.zip xs2 ys2 := by
  induction xs1 generalizing ys1 with
  | nil =>
      cases ys1 <;> simp at h ⊢
  | cons x xt ih =>
      cases ys1 with
      | nil => simp at h
      | cons y yt =>
          simp at h
          simp [ih yt h]

/-- If both payload lists are empty, channel compatibility is trivial. -/
private theorem channelLabelCompatible_of_empty
    (M : WordTuple L C F Payload)
    (hSend : ∀ A B, sendPayloads B (M A) = [])
    (hRecv : ∀ A B, recvPayloads A (M B) = [])
    : matchedLabelsCompatible M := by
  intro A B p hp
  simp [channelLabelCompatible, hSend A B, hRecv A B] at hp

/-- A word tuple M is an MSC if:
    1. No unmatched receives: for each channel A→B, rcvCount ≤ sndCount
    2. All matched send/receive pairs are payload-compatible
    2. The causality relation is acyclic (witnessed by a causal ranking)
    The payload relation itself is abstracted by `PayloadCompatiblePred Payload`,
    and `PayloadMatching.lean` provides the concrete instance for
    receiver-pattern tuples. -/
structure IsMSC (M : WordTuple L C F Payload) : Prop where
  /-- Condition 1: no unmatched receive events. -/
  noUnmatchedRecv : noUnmatchedReceives M
  /-- Condition 2: matched FIFO pairs are payload-compatible. -/
  labelCompat : matchedLabelsCompatible M
  /-- Condition 3: causality is acyclic (witnessed by a ranking). -/
  acyclic : hasAcyclicCausality M

/-- A word tuple M is a complete MSC: every channel is balanced
    and all matched pairs are payload-compatible, and the causality
    relation is acyclic. -/
structure IsCompleteMSC (M : WordTuple L C F Payload) : Prop where
  /-- Every channel is complete: sndCount = rcvCount. -/
  complete : tupleComplete M
  /-- Every matched FIFO pair is payload-compatible. -/
  labelCompat : matchedLabelsCompatible M
  /-- Causality is acyclic (witnessed by a ranking). -/
  acyclic : hasAcyclicCausality M

/-- A complete MSC is an MSC. -/
theorem isCompleteMSC_implies_isMSC (M : WordTuple L C F Payload)
    (h : IsCompleteMSC M) : IsMSC M where
  noUnmatchedRecv := tupleComplete_implies_noUnmatchedReceives M h.complete
  labelCompat := h.labelCompat
  acyclic := h.acyclic

/-- If the first FIFO-visible send payload on channel `A → B` is `xs` and the
    first FIFO-visible receive payload is `ys`, then an MSC requires these
    payloads to be compatible. This packages the head case of channel label
    compatibility for later control-decision arguments. -/
theorem firstPayloads_compatible
    (M : WordTuple L C F Payload)
    (hM : IsMSC M)
    (A B : L) (xs ys : Payload)
    (hs : ∃ ss, sendPayloads (C := C) (F := F) (Payload := Payload) B (M A) = xs :: ss)
    (hr : ∃ rs, recvPayloads (C := C) (F := F) (Payload := Payload) A (M B) = ys :: rs) :
    PayloadCompatible Payload xs ys := by
  rcases hs with ⟨ss, hs⟩
  rcases hr with ⟨rs, hr⟩
  have hp :
      (xs, ys) ∈
        List.zip
          (sendPayloads (C := C) (F := F) (Payload := Payload) B (M A))
          (recvPayloads (C := C) (F := F) (Payload := Payload) A (M B)) := by
    simp [hs, hr]
  exact hM.labelCompat A B (xs, ys) hp

section ControlHeads

variable [ControlPayloadSpec Payload]

/-- If the first visible send/receive payloads on a channel are control
    decisions `b1` and `b2`, then MSC label compatibility forces the same
    decision bit on both sides. -/
theorem firstControlDecisions_eq
    (M : WordTuple L C F Payload)
    (hM : IsMSC M)
    (A B : L) (b1 b2 : Bool)
    (hs :
      ∃ ss,
        sendPayloads (C := C) (F := F) (Payload := Payload) B (M A) =
          ControlPayload.setDecision b1 ControlPayload.ctrlPattern :: ss)
    (hr :
      ∃ rs,
        recvPayloads (C := C) (F := F) (Payload := Payload) A (M B) =
          ControlPayload.setDecision b2 ControlPayload.ctrlPattern :: rs) :
    b1 = b2 := by
  exact controlPayload_compat_decision_eq (Payload := Payload)
    (firstPayloads_compatible (L := L) (C := C) (F := F) (Payload := Payload)
      M hM A B
      (ControlPayload.setDecision b1 ControlPayload.ctrlPattern)
      (ControlPayload.setDecision b2 ControlPayload.ctrlPattern)
      hs hr)

end ControlHeads

end MSCPredicate

------------------------------------------------------------------------
-- Helper: non-communicating letter lemmas
------------------------------------------------------------------------

section NonCommLemmas

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload]

/-- A send observed at position `j` forces the total send count to be positive. -/
private theorem countSends_pos_of_sendAt {A : L} (B : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) A))
    (j : Nat) (hj : j < w.length)
    (hSend : (w[j]).val.isSendTo B = true) :
    0 < countSends B w := by
  have hsplit : w = w.take j ++ [w[j]] ++ w.drop (j + 1) := by
    have hsplit0 : w = w.take j ++ w.drop j := (List.take_append_drop j w).symm
    rw [List.drop_eq_getElem_cons hj] at hsplit0
    simpa [List.append_assoc] using hsplit0
  rw [hsplit, countSends_append, countSends_append]
  simp [countSends, hSend]
  omega

/-- A receive observed at position `j` forces the total receive count to be positive. -/
private theorem countRecvs_pos_of_recvAt {B : L} (A : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B))
    (j : Nat) (hj : j < w.length)
    (hRecv : (w[j]).val.isRecvFrom A = true) :
    0 < countRecvs A w := by
  have hsplit : w = w.take j ++ [w[j]] ++ w.drop (j + 1) := by
    have hsplit0 : w = w.take j ++ w.drop j := (List.take_append_drop j w).symm
    rw [List.drop_eq_getElem_cons hj] at hsplit0
    simpa [List.append_assoc] using hsplit0
  rw [hsplit, countRecvs_append, countRecvs_append]
  simp [countRecvs, hRecv]
  omega

/-- Taking a prefix cannot increase the receive count on a channel. -/
private theorem countRecvs_take_le {B : L} (A : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B))
    (j : Nat) :
    countRecvs A (w.take j) ≤ countRecvs A w := by
  calc
    countRecvs A (w.take j) ≤ countRecvs A (w.take j) + countRecvs A (w.drop j) := by omega
    _ = countRecvs A w := by rw [← countRecvs_append, List.take_append_drop]

/-- Taking a prefix cannot increase the send count on a channel. -/
private theorem countSends_take_le {A : L} (B : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) A))
    (j : Nat) :
    countSends B (w.take j) ≤ countSends B w := by
  calc
    countSends B (w.take j) ≤ countSends B (w.take j) + countSends B (w.drop j) := by omega
    _ = countSends B w := by rw [← countSends_append, List.take_append_drop]

/-- If position `j` is a receive on channel `A → B`, then the full word has
    one more such receive than the prefix before `j`. -/
private theorem countRecvs_take_lt_of_recvAt {B : L} (A : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B))
    (j : Nat) (hj : j < w.length)
    (hRecv : (w[j]).val.isRecvFrom A = true) :
    countRecvs A (w.take j) + 1 ≤ countRecvs A w := by
  have hsplit : w = w.take j ++ [w[j]] ++ w.drop (j + 1) := by
    have hsplit0 : w = w.take j ++ w.drop j := (List.take_append_drop j w).symm
    rw [List.drop_eq_getElem_cons hj] at hsplit0
    simpa [List.append_assoc] using hsplit0
  rw [hsplit, countRecvs_append, countRecvs_append]
  simp [countRecvs, hRecv]

/-- If every letter in every word is non-communicating,
    then sndCount is 0 on every channel. -/
private theorem sndCount_zero_of_nonComm
    (M : WordTuple L C F Payload)
    (hNC : ∀ (A : L) (i : Nat) (hi : i < (M A).length),
      ((M A).get ⟨i, hi⟩).val.isNonComm = true)
    (A B : L) : sndCount M A B = 0 := by
  let w := M A
  have hNCw : ∀ (i : Nat) (hi : i < w.length), (w.get ⟨i, hi⟩).val.isNonComm = true := by
    intro i hi
    simpa [w] using hNC A i hi
  change countSends B w = 0
  revert hNCw
  induction w with
  | nil =>
      intro hNCw
      simp [countSends]
  | cons hd tl ih =>
    intro hNCw
    have hhd := hNCw 0 (by simp)
    have hs : hd.val.isSendTo B = false :=
      Letter.isNonComm_not_sendTo hd.val hhd B
    have htail : countSends B tl = 0 := by
      apply ih
      intro i hi
      exact hNCw (i + 1) (by simp [hi, Nat.succ_lt_succ_iff])
    simp [countSends, hs, htail]

/-- If every letter in every word is non-communicating,
    then rcvCount is 0 on every channel. -/
private theorem rcvCount_zero_of_nonComm
    (M : WordTuple L C F Payload)
    (hNC : ∀ (A : L) (i : Nat) (hi : i < (M A).length),
      ((M A).get ⟨i, hi⟩).val.isNonComm = true)
    (A B : L) : rcvCount M A B = 0 := by
  let w := M B
  have hNCw : ∀ (i : Nat) (hi : i < w.length), (w.get ⟨i, hi⟩).val.isNonComm = true := by
    intro i hi
    simpa [w] using hNC B i hi
  change countRecvs A w = 0
  revert hNCw
  induction w with
  | nil =>
      intro hNCw
      simp [countRecvs]
  | cons hd tl ih =>
    intro hNCw
    have hhd := hNCw 0 (by simp)
    have hr : hd.val.isRecvFrom A = false :=
      Letter.isNonComm_not_recvFrom hd.val hhd A
    have htail : countRecvs A tl = 0 := by
      apply ih
      intro i hi
      exact hNCw (i + 1) (by simp [hi, Nat.succ_lt_succ_iff])
    simp [countRecvs, hr, htail]

/-- If every letter in every word is non-communicating,
    then all send-payload collections are empty. -/
private theorem sendPayloads_nil_of_nonComm
    (M : WordTuple L C F Payload)
    (hNC : ∀ (A : L) (i : Nat) (hi : i < (M A).length),
      ((M A).get ⟨i, hi⟩).val.isNonComm = true)
    (A B : L) : sendPayloads (C := C) (F := F) (Payload := Payload) B (M A) = [] := by
  let w := M A
  have hNCw : ∀ (i : Nat) (hi : i < w.length), (w.get ⟨i, hi⟩).val.isNonComm = true := by
    intro i hi
    simpa [w] using hNC A i hi
  change sendPayloads B w = []
  revert hNCw
  induction w with
  | nil =>
      intro hNCw
      simp [sendPayloads]
  | cons hd tl ih =>
      intro hNCw
      have hhd := hNCw 0 (by simp)
      have hs : hd.val.isSendTo B = false :=
        Letter.isNonComm_not_sendTo hd.val hhd B
      have htail : sendPayloads B tl = [] := by
        apply ih
        intro i hi
        exact hNCw (i + 1) (by simp [hi, Nat.succ_lt_succ_iff])
      cases hℓ : hd.val with
      | sendLetter owner xs tgt hneq =>
          have hneq : tgt ≠ B := by
            intro hEq
            have htrue : hd.val.isSendTo B = true := by
              simpa [hℓ, Letter.isSendTo, hEq]
            rw [htrue] at hs
            contradiction
          simp [sendPayloads, hℓ, hneq, htail]
      | recvLetter owner ys src =>
          simp [sendPayloads, hℓ, htail]
      | actLetter owner ys f xs =>
          simp [sendPayloads, hℓ, htail]
      | ifTrueLetter c owner =>
          simp [sendPayloads, hℓ, htail]
      | ifFalseLetter c owner =>
          simp [sendPayloads, hℓ, htail]
      | whileTrueLetter c owner =>
          simp [sendPayloads, hℓ, htail]
      | whileFalseLetter c owner =>
          simp [sendPayloads, hℓ, htail]

/-- If every letter in every word is non-communicating,
    then all receive-payload collections are empty. -/
private theorem recvPayloads_nil_of_nonComm
    (M : WordTuple L C F Payload)
    (hNC : ∀ (A : L) (i : Nat) (hi : i < (M A).length),
      ((M A).get ⟨i, hi⟩).val.isNonComm = true)
    (A B : L) : recvPayloads (C := C) (F := F) (Payload := Payload) A (M B) = [] := by
  let w := M B
  have hNCw : ∀ (i : Nat) (hi : i < w.length), (w.get ⟨i, hi⟩).val.isNonComm = true := by
    intro i hi
    simpa [w] using hNC B i hi
  change recvPayloads A w = []
  revert hNCw
  induction w with
  | nil =>
      intro hNCw
      simp [recvPayloads]
  | cons hd tl ih =>
      intro hNCw
      have hhd := hNCw 0 (by simp)
      have hr : hd.val.isRecvFrom A = false :=
        Letter.isNonComm_not_recvFrom hd.val hhd A
      have htail : recvPayloads A tl = [] := by
        apply ih
        intro i hi
        exact hNCw (i + 1) (by simp [hi, Nat.succ_lt_succ_iff])
      cases hℓ : hd.val with
      | recvLetter owner ys src =>
          have hneq : src ≠ A := by
            intro hEq
            have htrue : hd.val.isRecvFrom A = true := by
              simpa [hℓ, Letter.isRecvFrom, hEq]
            rw [htrue] at hr
            contradiction
          simp [recvPayloads, hℓ, hneq, htail]
      | sendLetter owner xs tgt hneq =>
          simp [recvPayloads, hℓ, htail]
      | actLetter owner ys f xs =>
          simp [recvPayloads, hℓ, htail]
      | ifTrueLetter c owner =>
          simp [recvPayloads, hℓ, htail]
      | ifFalseLetter c owner =>
          simp [recvPayloads, hℓ, htail]
      | whileTrueLetter c owner =>
          simp [recvPayloads, hℓ, htail]
      | whileFalseLetter c owner =>
          simp [recvPayloads, hℓ, htail]

end NonCommLemmas

------------------------------------------------------------------------
-- Rankings for singleton non-communicating MSCs
------------------------------------------------------------------------

section SingletonRankings

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload]

/-- A word tuple with no communication and words of length ≤ 1
    admits a trivial ranking. -/
private def trivialRanking
    (M : WordTuple L C F Payload)
    (hSnd : ∀ A B, sndCount M A B = 0)
    (hShort : ∀ A, (M A).length ≤ 1) :
    CausalRanking M where
  rank := fun _ => 0
  local_mono := by
    intro e1 e2 hSameLL hLt _ _
    exfalso
    have hs := hShort e1.lifeline
    rw [hSameLL] at hs
    omega
  fifo_mono := by
    intro A B j1 _ hj1 _ hSend _ _
    exfalso
    have hcnt := hSnd A B
    simp [sndCount] at hcnt
    have hpos : 0 < countSends B (M A) :=
      countSends_pos_of_sendAt B (M A) j1 hj1 hSend
    omega

/-- A non-communicating word tuple with short words is a complete MSC. -/
private theorem nonComm_short_isCompleteMSC
    (M : WordTuple L C F Payload)
    (hNC : ∀ (A : L) (i : Nat) (hi : i < (M A).length),
      ((M A).get ⟨i, hi⟩).val.isNonComm = true)
    (hShort : ∀ A, (M A).length ≤ 1) :
    IsCompleteMSC M where
  complete := by
    intro A B
    simp [channelComplete,
          sndCount_zero_of_nonComm M hNC A B,
          rcvCount_zero_of_nonComm M hNC A B]
  labelCompat := by
    apply channelLabelCompatible_of_empty
    · intro A B
      exact sendPayloads_nil_of_nonComm M hNC A B
    · intro A B
      exact recvPayloads_nil_of_nonComm M hNC A B
  acyclic := ⟨trivialRanking M (sndCount_zero_of_nonComm M hNC) hShort⟩

end SingletonRankings

------------------------------------------------------------------------
-- Canonical MSC completeness proofs
------------------------------------------------------------------------

section CanonicalCompleteness

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload]

/-- The empty MSC is a complete MSC. -/
theorem mscEmpty_isCompleteMSC :
    IsCompleteMSC (mscEmpty (L := L) (C := C) (F := F) (Payload := Payload)) where
  complete := by
    intro A B
    simp [channelComplete, sndCount, rcvCount, mscEmpty, WordTuple.empty, countSends, countRecvs]
  labelCompat := by
    intro A B p hp
    simp [channelLabelCompatible, mscEmpty, WordTuple.empty, sendPayloads, recvPayloads] at hp
  acyclic := ⟨emptyRanking⟩

/-- Act letter is non-communicating. -/
private theorem actLetter_isNonComm (A : L) (ys : Payload) (f : F) (xs : Payload) :
    (Letter.actLetter A ys f xs : Letter L C F Payload).isNonComm = true := by
  simp [Letter.isNonComm]

/-- The canonical act MSC is a complete MSC. -/
theorem mscAct_isCompleteMSC (A : L) (ys : Payload) (f : F) (xs : Payload) :
    IsCompleteMSC (mscAct (C := C) A ys f xs) := by
  apply nonComm_short_isCompleteMSC
  · intro X i hi
    by_cases hX : X = A
    · subst hX
      simp [mscAct, singletonWord] at hi ⊢
      have : i = 0 := by omega
      subst this
      simp [AlphabetOf.mkAct, actLetter_isNonComm]
    · simp [mscAct, hX] at hi
  · intro X
    by_cases hX : X = A
    · subst hX
      simp [mscAct, singletonWord]
    · simp [mscAct, hX]

/-- ifTrue letter is non-communicating. -/
private theorem ifTrueLetter_isNonComm (c : C) (A : L) :
    (Letter.ifTrueLetter c A : Letter L C F Payload).isNonComm = true := by
  simp [Letter.isNonComm]

/-- ifFalse letter is non-communicating. -/
private theorem ifFalseLetter_isNonComm (c : C) (A : L) :
    (Letter.ifFalseLetter c A : Letter L C F Payload).isNonComm = true := by
  simp [Letter.isNonComm]

/-- whileTrue letter is non-communicating. -/
private theorem whileTrueLetter_isNonComm (c : C) (A : L) :
    (Letter.whileTrueLetter c A : Letter L C F Payload).isNonComm = true := by
  simp [Letter.isNonComm]

/-- whileFalse letter is non-communicating. -/
private theorem whileFalseLetter_isNonComm (c : C) (A : L) :
    (Letter.whileFalseLetter c A : Letter L C F Payload).isNonComm = true := by
  simp [Letter.isNonComm]

/-- Helper for choice MSC completeness. -/
private theorem mscChoice_nonComm_isCompleteMSC (B : L)
    (γ : AlphabetOf (C := C) (F := F) (Payload := Payload) B)
    (hNC : γ.val.isNonComm = true) :
    IsCompleteMSC (mscChoice B γ) := by
  apply nonComm_short_isCompleteMSC
  · intro X i hi
    by_cases hX : X = B
    · subst hX
      simp [mscChoice, singletonWord] at hi ⊢
      have : i = 0 := by omega
      subst this
      simp [hNC]
    · simp [mscChoice, hX] at hi
  · intro X
    by_cases hX : X = B
    · subst hX
      simp [mscChoice, singletonWord]
    · simp [mscChoice, hX]

/-- The canonical if-true MSC is a complete MSC. -/
theorem mscIfTrue_isCompleteMSC (c : C) (B : L) :
    IsCompleteMSC (mscIfTrue (C := C) (F := F) (Payload := Payload) c B) :=
  mscChoice_nonComm_isCompleteMSC B (choiceIfTrue c B)
    (by simp [choiceIfTrue, AlphabetOf.mkIfTrue, ifTrueLetter_isNonComm])

/-- The canonical if-false MSC is a complete MSC. -/
theorem mscIfFalse_isCompleteMSC (c : C) (B : L) :
    IsCompleteMSC (mscIfFalse (C := C) (F := F) (Payload := Payload) c B) :=
  mscChoice_nonComm_isCompleteMSC B (choiceIfFalse c B)
    (by simp [choiceIfFalse, AlphabetOf.mkIfFalse, ifFalseLetter_isNonComm])

/-- The canonical while-true MSC is a complete MSC. -/
theorem mscWhileTrue_isCompleteMSC (c : C) (B : L) :
    IsCompleteMSC (mscWhileTrue (C := C) (F := F) (Payload := Payload) c B) :=
  mscChoice_nonComm_isCompleteMSC B (choiceWhileTrue c B)
    (by simp [choiceWhileTrue, AlphabetOf.mkWhileTrue, whileTrueLetter_isNonComm])

/-- The canonical while-false MSC is a complete MSC. -/
theorem mscWhileFalse_isCompleteMSC (c : C) (B : L) :
    IsCompleteMSC (mscWhileFalse (C := C) (F := F) (Payload := Payload) c B) :=
  mscChoice_nonComm_isCompleteMSC B (choiceWhileFalse c B)
    (by simp [choiceWhileFalse, AlphabetOf.mkWhileFalse, whileFalseLetter_isNonComm])

------------------------------------------------------------------------
-- mscMsg completeness
------------------------------------------------------------------------

/-- The canonical message MSC mscMsg A xs B ys h is a complete MSC.

    Channel A→B has exactly 1 send and 1 recv.
    All other channels have 0 sends and 0 recvs.
    The ranking is: event on A gets rank 0, event on B gets rank 1.
    FIFO edge (A,0)→(B,0) is strictly increasing. No local-order edges
    (each word has ≤ 1 event). -/
private theorem sndCount_mscMsg (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B) (S T : L) :
    sndCount (mscMsg (C := C) (F := F) A xs B ys h) S T =
      if S = A ∧ T = B then 1 else 0 := by
  by_cases hSA : S = A
  · subst hSA
    by_cases hTB : T = B
    · subst hTB
      simp [sndCount, mscMsg_sender, countSends, AlphabetOf.mkSend, Letter.isSendTo]
    · have hBT : B ≠ T := by simpa [eq_comm] using hTB
      simp [sndCount, mscMsg_sender, countSends, AlphabetOf.mkSend, Letter.isSendTo, hBT, hTB]
  · by_cases hSB : S = B
    · subst hSB
      simp [sndCount, mscMsg_receiver, countSends, AlphabetOf.mkRecv, Letter.isSendTo, h, hSA]
    · simp [sndCount, mscMsg, hSA, hSB, countSends]

private theorem rcvCount_mscMsg (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B) (S T : L) :
    rcvCount (mscMsg (C := C) (F := F) A xs B ys h) S T =
      if S = A ∧ T = B then 1 else 0 := by
  by_cases hTB : T = B
  · subst hTB
    by_cases hSA : S = A
    · subst hSA
      simp [rcvCount, mscMsg_receiver, countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom, h]
    · have hAS : A ≠ S := by simpa [eq_comm] using hSA
      simp [rcvCount, mscMsg_receiver, countRecvs, AlphabetOf.mkRecv, Letter.isRecvFrom, hAS, hSA, h]
  · by_cases hTA : T = A
    · subst hTA
      simp [rcvCount, mscMsg_sender, countRecvs, AlphabetOf.mkSend, Letter.isRecvFrom, hTB]
    · simp [rcvCount, mscMsg, hTA, hTB, countRecvs]

private theorem sendPayloads_mscMsg (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B) (S T : L) :
    sendPayloads (C := C) (F := F) (Payload := Payload) T
      (mscMsg (C := C) (F := F) A xs B ys h S) =
      if S = A ∧ T = B then [xs] else [] := by
  by_cases hSA : S = A
  · subst hSA
    by_cases hTB : T = B
    · subst hTB
      simp [mscMsg_sender, sendPayloads, singletonWord, AlphabetOf.mkSend]
    · have hBT : B ≠ T := by simpa [eq_comm] using hTB
      simp [mscMsg_sender, sendPayloads, AlphabetOf.mkSend, hBT, hTB]
  · by_cases hSB : S = B
    · subst hSB
      simp [mscMsg_receiver, sendPayloads, singletonWord, AlphabetOf.mkRecv, h, hSA]
    · have hFalse : ¬ (S = A ∧ T = B) := by
        intro hST
        exact hSA hST.1
      rw [mscMsg_other (C := C) (F := F) A xs B ys h S hSA hSB]
      simp [sendPayloads, hFalse]

private theorem recvPayloads_mscMsg (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B) (S T : L) :
    recvPayloads (C := C) (F := F) (Payload := Payload) S
      (mscMsg (C := C) (F := F) A xs B ys h T) =
      if S = A ∧ T = B then [ys] else [] := by
  by_cases hTB : T = B
  · subst hTB
    by_cases hSA : S = A
    · subst hSA
      simp [mscMsg_receiver, recvPayloads, singletonWord, AlphabetOf.mkRecv, h]
    · have hAS : A ≠ S := by simpa [eq_comm] using hSA
      simp [mscMsg_receiver, recvPayloads, AlphabetOf.mkRecv, hAS, hSA, h]
  · by_cases hTA : T = A
    · subst hTA
      simp [mscMsg_sender, recvPayloads, singletonWord, AlphabetOf.mkSend, hTB]
    · have hFalse : ¬ (S = A ∧ T = B) := by
        intro hST
        exact hTB hST.2
      rw [mscMsg_other (C := C) (F := F) A xs B ys h T hTA hTB]
      simp [recvPayloads, hFalse]

theorem mscMsg_isCompleteMSC (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B) (hxy : PayloadCompatible Payload xs ys) :
    IsCompleteMSC (mscMsg (C := C) (F := F) A xs B ys h) where
  complete := by
    intro S T
    rw [channelComplete, sndCount_mscMsg A xs B ys h S T, rcvCount_mscMsg A xs B ys h S T]
  labelCompat := by
    intro S T p hp
    rw [sendPayloads_mscMsg A xs B ys h S T,
      recvPayloads_mscMsg A xs B ys h S T] at hp
    by_cases hST : S = A ∧ T = B
    · simp [hST] at hp
      rcases hp with rfl
      exact hxy
    · simp [hST] at hp
  acyclic := by
    refine ⟨⟨fun e => if e.lifeline = A then 0 else 1, ?_, ?_⟩⟩
    · intro e1 e2 hSameLL hLt _ _
      exfalso
      have hshort : (mscMsg (C := C) (F := F) A xs B ys h e1.lifeline).length ≤ 1 := by
        by_cases h1 : e1.lifeline = A
        · subst h1
          simp [mscMsg, singletonWord]
        · by_cases h2 : e1.lifeline = B
          · subst h2
            simp [mscMsg, singletonWord, h.symm]
          · simp [mscMsg, h1, h2]
      rw [hSameLL] at hshort
      omega
    · intro S T j1 j2 hj1 hj2 hSend hRecv _
      have hS : S = A := by
        by_cases hSA : S = A
        · exact hSA
        · by_cases hSB : S = B
          · subst S
            have hj1z : j1 = 0 := by
              simp [mscMsg, hSA, singletonWord] at hj1
              omega
            subst hj1z
            have : False := by
              simpa [mscMsg_receiver, singletonWord, AlphabetOf.mkRecv, Letter.isSendTo, h]
                using hSend
            contradiction
          · have : mscMsg (C := C) (F := F) A xs B ys h S = [] := by
              simp [mscMsg, hSA, hSB]
            simp [this] at hj1
      have hT : T = B := by
        by_cases hTB : T = B
        · exact hTB
        · by_cases hTA : T = A
          · subst T
            have hj2z : j2 = 0 := by
              simp [mscMsg_sender, singletonWord] at hj2
              omega
            subst hj2z
            have : False := by
              simpa [mscMsg_sender, singletonWord, AlphabetOf.mkSend, Letter.isRecvFrom]
                using hRecv
            contradiction
          · have : mscMsg (C := C) (F := F) A xs B ys h T = [] := by
              simp [mscMsg, hTA, hTB]
            simp [this] at hj2
      have hj1A := hj1
      rw [hS] at hj1A
      have hj1A' : j1 < 1 := by
        simpa [mscMsg_sender, singletonWord] using hj1A
      have hj2B := hj2
      rw [hT] at hj2B
      have hj2B' : j2 < 1 := by
        simpa [mscMsg_receiver, singletonWord, h] using hj2B
      have hj1z : j1 = 0 := by
        omega
      have hj2z : j2 = 0 := by
        omega
      subst hj1z
      subst hj2z
      simpa [hS, hT, h, eq_comm]

end CanonicalCompleteness

------------------------------------------------------------------------
-- Ranking concatenation: the key construction for IsMSCPredicate
------------------------------------------------------------------------

section RankingConcat

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload]

/-- Label compatibility is preserved by concatenation with a complete prefix. -/
theorem matchedLabelsCompatible_concat
    (M1 M2 : WordTuple L C F Payload)
    (h1 : IsCompleteMSC M1) (h2 : IsMSC M2) :
    matchedLabelsCompatible (M1 ∘ₘ M2) := by
  intro A B p hp
  rw [show sendPayloads B ((M1 ∘ₘ M2) A) =
      sendPayloads B (M1 A) ++ sendPayloads B (M2 A) by
        simp [WordTuple.concat, sendPayloads_append]]
    at hp
  rw [show recvPayloads A ((M1 ∘ₘ M2) B) =
      recvPayloads A (M1 B) ++ recvPayloads A (M2 B) by
        simp [WordTuple.concat, recvPayloads_append]]
    at hp
  have hlen :
      (sendPayloads B (M1 A)).length = (recvPayloads A (M1 B)).length := by
    simpa [sndCount, rcvCount] using h1.complete A B
  rw [zip_append_eq_of_length_eq _ _ _ _ hlen] at hp
  rcases List.mem_append.mp hp with hp | hp
  · exact h1.labelCompat A B p hp
  · exact h2.labelCompat A B p hp

/-- Given rankings for M1 and M2 where M1 is complete,
    construct a ranking for M1 ∘ₘ M2.

    Strategy: event (X, j) in M1∘M2 is in M1's portion if j < |M1 X|,
    else in M2's portion. We assign:
    - M1-events: rank from R1
    - M2-events: max(R1) + 1 + rank from R2 (shifted up)

    Local-order edges: same lifeline, pos increases.
    - Within M1: handled by R1.local_mono
    - Within M2: handled by R2.local_mono (shifted)
    - M1→M2 crossing: R1 value < shift + R2 value, so ok
    - M2→M1 crossing: impossible (pos decreases)

    FIFO edges in M1∘M2: because M1 is complete, the k-th send/recv
    on channel A→B in the concatenation decomposes:
    - k < sndCount(M1, A, B): both send and recv are in M1 (by completeness)
    - k ≥ sndCount(M1, A, B): both are in M2 (shifted)
    So FIFO edges don't cross the M1/M2 boundary when M1 is complete. -/
def concatRanking
    (M1 M2 : WordTuple L C F Payload)
    (R1 : CausalRanking M1) (R2 : CausalRanking M2)
    (hComplete : tupleComplete M1)
    (shift : Nat)
    (hShift : ∀ e, e.pos < (M1 e.lifeline).length → R1.rank e < shift) :
    CausalRanking (M1 ∘ₘ M2) where
  rank := fun e =>
    if e.pos < (M1 e.lifeline).length then
      R1.rank e
    else
      shift + R2.rank ⟨e.lifeline, e.pos - (M1 e.lifeline).length⟩
  local_mono := by
    intro e1 e2 hSameLL hLt _ _
    simp [WordTuple.concat] at *
    have hLen1 : (M1 e1.lifeline).length = (M1 e2.lifeline).length :=
      congrArg (fun X => (M1 X).length) hSameLL
    have hLen2 : (M2 e1.lifeline).length = (M2 e2.lifeline).length :=
      congrArg (fun X => (M2 X).length) hSameLL
    by_cases h1 : e1.pos < (M1 e1.lifeline).length
    · -- e1 in M1
      simp [h1]
      by_cases h2 : e2.pos < (M1 e2.lifeline).length
      · -- Both in M1
        simp [h2]
        exact R1.local_mono e1 e2 hSameLL hLt (hSameLL ▸ h1) h2
      · -- e1 in M1, e2 in M2: R1 value < shift ≤ shift + R2 value
        simp [h2]
        have := hShift e1 (hSameLL ▸ h1)
        omega
    · -- e1 in M2
      have h1' : ¬ e1.pos < (M1 e2.lifeline).length := by simpa [hLen1] using h1
      simp [h1']
      have h2 : ¬ e2.pos < (M1 e2.lifeline).length := by omega
      simp [h2]
      -- Both in M2: use R2.local_mono with shifted positions
      have hLt' : e1.pos - (M1 e2.lifeline).length < e2.pos - (M1 e2.lifeline).length := by omega
      have hpos1 : e1.pos - (M1 e2.lifeline).length < (M2 e2.lifeline).length := by
        have : (M1 e2.lifeline ++ M2 e2.lifeline).length =
               (M1 e2.lifeline).length + (M2 e2.lifeline).length := by
          simp
        omega
      have hpos2 : e2.pos - (M1 e2.lifeline).length < (M2 e2.lifeline).length := by
        have : (M1 e2.lifeline ++ M2 e2.lifeline).length =
               (M1 e2.lifeline).length + (M2 e2.lifeline).length := by
          simp
        omega
      let e1' : Event L := ⟨e1.lifeline, e1.pos - (M1 e1.lifeline).length⟩
      let e2' : Event L := ⟨e2.lifeline, e2.pos - (M1 e2.lifeline).length⟩
      have hpos1' : e1'.pos < (M2 e1'.lifeline).length := by
        simpa [e1', hLen2, hLen1] using hpos1
      have hpos2' : e2'.pos < (M2 e2'.lifeline).length := by
        simpa [e2'] using hpos2
      have hR2 : R2.rank e1' < R2.rank e2' := by
        exact R2.local_mono e1' e2' (by simp [e1', e2', hSameLL]) (by simpa [e1', e2', hLen1] using hLt') hpos1' hpos2'
      simpa [e1', e2', h1', h2, hLen1] using Nat.add_lt_add_left hR2 shift
  fifo_mono := by
    intro A B j1 j2 hj1 hj2 hSend hRecv hCntEq
    simp only [WordTuple.concat] at hj1 hj2 hSend hRecv hCntEq
    simp only [List.get_eq_getElem] at hSend hRecv
    -- Determine whether j1 is in M1 A or M2 A
    by_cases h1 : j1 < (M1 A).length
    · -- j1 in M1's portion of A's word
      simp [h1]
      -- The letter at position j1 in M1 A ++ M2 A is the letter at j1 in M1 A
      have hGet1 : (M1 A ++ M2 A)[j1]'hj1 = (M1 A)[j1]'h1 :=
        List.getElem_append_left h1
      rw [hGet1] at hSend
      -- The count of sends before j1 in M1 A ++ M2 A equals the count in M1 A
      have hTake1 : (M1 A ++ M2 A).take j1 = (M1 A).take j1 := by
        exact List.take_append_of_le_length (l₁ := M1 A) (l₂ := M2 A) (i := j1) (by omega)
      rw [hTake1] at hCntEq
      -- Now show j2 must also be in M1's portion of B's word
      -- The k-th send to B before position j1 in M1's word: k = countSends B ((M1 A).take j1)
      -- Since M1 is complete on channel A→B: sndCount M1 A B = rcvCount M1 A B
      -- k < sndCount M1 A B (since j1 is a send and there's at least k+1 sends total)
      -- So k < rcvCount M1 A B, meaning the k-th recv from A in M1 B exists
      -- The FIFO pairing in M1∘M2 matches the k-th send with the k-th recv
      -- in the concatenated word. Since M1 has rcvCount = sndCount ≥ k+1 recvs,
      -- the k-th recv in the concatenation is within M1 B's portion.
      -- Therefore j2 < (M1 B).length.
      by_cases h2 : j2 < (M1 B).length
      · -- Both in M1: use R1
        simp [h2]
        have hGet2 : (M1 B ++ M2 B)[j2]'hj2 = (M1 B)[j2]'h2 :=
          List.getElem_append_left h2
        rw [hGet2] at hRecv
        have hTake2 : (M1 B ++ M2 B).take j2 = (M1 B).take j2 := by
          exact List.take_append_of_le_length (l₁ := M1 B) (l₂ := M2 B) (i := j2) (by omega)
        rw [hTake2] at hCntEq
        exact R1.fifo_mono A B j1 j2 h1 h2 hSend hRecv hCntEq
      · -- j1 in M1, j2 in M2: rank is R1 vs shift + R2, so ok
        simp [h2]
        have := hShift ⟨A, j1⟩ h1
        omega
    · -- j1 in M2's portion of A's word
      simp [h1]
      -- The letter at j1 in M1 A ++ M2 A is from M2 A
      have hLenA : (M1 A ++ M2 A).length = (M1 A).length + (M2 A).length :=
        List.length_append
      have h1' : j1 - (M1 A).length < (M2 A).length := by omega
      have hGet1 : (M1 A ++ M2 A)[j1]'hj1 = (M2 A)[j1 - (M1 A).length]'h1' :=
        List.getElem_append_right (by omega)
      rw [hGet1] at hSend
      -- Count of sends before j1 in concatenation =
      -- countSends in M1 A + countSends in (M2 A).take (j1 - |M1 A|)
      have hTake1 : (M1 A ++ M2 A).take j1 =
          M1 A ++ (M2 A).take (j1 - (M1 A).length) := by
        rw [List.take_append]
        have hTakeM1 : (M1 A).take j1 = M1 A := by
          have hDrop : (M1 A).drop j1 = [] := List.drop_eq_nil_of_le (by omega)
          calc
            (M1 A).take j1 = (M1 A).take j1 ++ (M1 A).drop j1 := by
              simp [hDrop]
            _ = M1 A := List.take_append_drop j1 (M1 A)
        simpa [hTakeM1]
      rw [hTake1] at hCntEq
      simp [countSends_append] at hCntEq
      -- hCntEq: sndCount M1 A B + countSends B ((M2 A).take ...) = countRecvs A ((M1 B ++ M2 B).take j2)
      -- j2 must be in M2's portion too.
      -- Since sndCount M1 A B = rcvCount M1 A B (completeness),
      -- the recv count up to j2 must be ≥ sndCount M1 A B,
      -- meaning j2 ≥ (M1 B).length (all M1 B's recvs have been seen).
      have hComplete_AB := hComplete A B
      simp [channelComplete, sndCount, rcvCount] at hComplete_AB
      -- We need: j2 ≥ (M1 B).length
      -- Proof: the recv count up to j2 includes another recv (at j2 itself),
      -- so total recvs counted ≥ sndCount M1 A B + 1 > rcvCount M1 A B
      -- ... Actually, let's just check: if j2 < (M1 B).length, derive contradiction.
      by_cases h2 : j2 < (M1 B).length
      · -- j1 in M2, j2 in M1: impossible
        exfalso
        have hTake2 : (M1 B ++ M2 B).take j2 = (M1 B).take j2 := by
          exact List.take_append_of_le_length (l₁ := M1 B) (l₂ := M2 B) (i := j2) (by omega)
        rw [hTake2] at hCntEq
        have hRecvBound : countRecvs A ((M1 B).take j2) ≤ countRecvs A (M1 B) := by
          exact countRecvs_take_le A (M1 B) j2
        have hGet2 : (M1 B ++ M2 B)[j2]'hj2 = (M1 B)[j2]'h2 :=
          List.getElem_append_left h2
        rw [hGet2] at hRecv
        have hRecvAt : countRecvs A ((M1 B).take j2) + 1 ≤ countRecvs A (M1 B) := by
          exact countRecvs_take_lt_of_recvAt A (M1 B) j2 h2 hRecv
        have hRecvBound' : countRecvs A ((M1 B).take j2) ≤ countSends B (M1 A) := by
          simpa [hComplete_AB] using hRecvBound
        have hRecvAt' : countRecvs A ((M1 B).take j2) + 1 ≤ countSends B (M1 A) := by
          simpa [hComplete_AB] using hRecvAt
        omega
      · -- Both in M2
        simp [h2]
        have hLenB : (M1 B ++ M2 B).length = (M1 B).length + (M2 B).length :=
          List.length_append
        have h2' : j2 - (M1 B).length < (M2 B).length := by omega
        have hGet2 : (M1 B ++ M2 B)[j2]'hj2 = (M2 B)[j2 - (M1 B).length]'h2' :=
          List.getElem_append_right (by omega)
        rw [hGet2] at hRecv
        have hTake2 : (M1 B ++ M2 B).take j2 =
            M1 B ++ (M2 B).take (j2 - (M1 B).length) := by
          rw [List.take_append]
          have hTakeM1 : (M1 B).take j2 = M1 B := by
            have hDrop : (M1 B).drop j2 = [] := List.drop_eq_nil_of_le (by omega)
            calc
              (M1 B).take j2 = (M1 B).take j2 ++ (M1 B).drop j2 := by
                simp [hDrop]
              _ = M1 B := List.take_append_drop j2 (M1 B)
          simpa [hTakeM1]
        rw [hTake2] at hCntEq
        simp [countRecvs_append] at hCntEq
        -- hCntEq: countSends B (M1 A) + countSends B ((M2 A).take ...) =
        --         countRecvs A (M1 B) + countRecvs A ((M2 B).take ...)
        -- By completeness: countSends B (M1 A) = countRecvs A (M1 B)
        -- So: countSends B ((M2 A).take ...) = countRecvs A ((M2 B).take ...)
        have hM2Eq : countSends B ((M2 A).take (j1 - (M1 A).length)) =
            countRecvs A ((M2 B).take (j2 - (M1 B).length)) := by omega
        have hR2 := R2.fifo_mono A B
          (j1 - (M1 A).length) (j2 - (M1 B).length)
          h1' h2' hSend hRecv hM2Eq
        simpa using Nat.add_lt_add_left hR2 shift

end RankingConcat

------------------------------------------------------------------------
-- IsMSCPredicate instance
------------------------------------------------------------------------

section MSCPredicateInstance

variable {L C F Payload : Type} [DecidableEq L] [PayloadCompatiblePred Payload] [Fintype L]

------------------------------------------------------------------------
-- Finite-event bounds for rankings
------------------------------------------------------------------------

open Classical

/-- Enumerate the valid events of a word tuple. This uses the paper's
    finiteness assumption on the lifeline set. -/
private noncomputable def validEvents (M : WordTuple L C F Payload) : List (Event L) :=
  List.flatMap
    (fun A => (List.range (M A).length).map fun i => ⟨A, i⟩)
    (Fintype.enum L)

/-- Every valid event appears in `validEvents`. -/
private theorem mem_validEvents
    (M : WordTuple L C F Payload) (e : Event L)
    (he : e.pos < (M e.lifeline).length) :
    e ∈ validEvents M := by
  classical
  unfold validEvents
  refine (List.mem_flatMap).mpr ?_
  refine ⟨e.lifeline, Fintype.mem_enum e.lifeline, ?_⟩
  · refine List.mem_map.mpr ?_
    refine ⟨e.pos, ?_, ?_⟩
    · exact List.mem_range.mpr he
    · cases e
      rfl

/-- A finite upper bound on the ranks of all valid events of `M`. -/
private noncomputable def maxValidRank (M : WordTuple L C F Payload)
    (R : CausalRanking M) : Nat :=
  (validEvents M).foldl (fun acc e => max acc (R.rank e)) 0

/-- Every valid event rank is bounded by `maxValidRank`. -/
private theorem foldl_max_ge_acc
    (R : CausalRanking M) (es : List (Event L)) (acc : Nat) :
    acc ≤ es.foldl (fun acc e => max acc (R.rank e)) acc := by
  induction es generalizing acc with
  | nil =>
      simp
  | cons hd tl ih =>
      simp only [List.foldl_cons]
      exact Nat.le_trans (Nat.le_max_left _ _) (ih _)

/-- Any element occurring in the list is bounded by the fold-max. -/
private theorem rank_le_foldl_max_of_mem
    (R : CausalRanking M) :
    ∀ (es : List (Event L)) (acc : Nat) (e : Event L),
      e ∈ es →
      R.rank e ≤ es.foldl (fun acc e => max acc (R.rank e)) acc
  | [], _, _, h => False.elim (List.not_mem_nil h)
  | hd :: tl, acc, e, h => by
      simp only [List.mem_cons] at h
      simp only [List.foldl_cons]
      rcases h with rfl | htl
      · exact Nat.le_trans (Nat.le_max_right _ _) (foldl_max_ge_acc R tl _)
      · exact rank_le_foldl_max_of_mem R tl (max acc (R.rank hd)) e htl

/-- Every valid event rank is bounded by `maxValidRank`. -/
private theorem rank_le_maxValidRank
    (M : WordTuple L C F Payload) (R : CausalRanking M)
    (e : Event L) (he : e.pos < (M e.lifeline).length) :
    R.rank e ≤ maxValidRank M R := by
  classical
  unfold maxValidRank
  exact rank_le_foldl_max_of_mem R (validEvents M) 0 e (mem_validEvents M e he)

/-- Helper: extract a ranking with a known bound from a CausalRanking. -/
private noncomputable def boundedRanking (M : WordTuple L C F Payload)
    (hac : hasAcyclicCausality M) : CausalRanking M :=
  Classical.choice hac

/-- Direct construction of a causal ranking for M1 ∘ₘ M2
    when M1 is complete and both M1, M2 have rankings.

    We define:
    - For events in M1's portion (pos < |M1 X|): rank = R1.rank
    - For events in M2's portion (pos ≥ |M1 X|): rank = bound + R2.rank(shifted)

    where bound is chosen large enough that all R1 ranks are below it. -/
private noncomputable def concatRankingInstance
    (M1 M2 : WordTuple L C F Payload)
    (h1 : IsCompleteMSC M1) (h2ac : hasAcyclicCausality M2) :
    CausalRanking (M1 ∘ₘ M2) := by
  have R1 := Classical.choice h1.acyclic
  have R2 := Classical.choice h2ac
  let shift := maxValidRank M1 R1 + 1
  exact concatRanking M1 M2 R1 R2 h1.complete shift
    (by
      intro e he
      exact Nat.lt_succ_of_le (rank_le_maxValidRank M1 R1 e he))

/-- If M1 is complete and M2 is an MSC, then M1 ∘ M2 is an MSC. -/
theorem concat_complete_msc (M1 M2 : WordTuple L C F Payload)
    (h1 : IsCompleteMSC M1) (h2 : IsMSC M2) :
    IsMSC (M1 ∘ₘ M2) where
  noUnmatchedRecv := noUnmatchedReceives_concat M1 M2 h1.complete h2.noUnmatchedRecv
  labelCompat := matchedLabelsCompatible_concat M1 M2 h1 h2
  acyclic := ⟨concatRankingInstance M1 M2 h1 h2.acyclic⟩

/-- If M1 and M2 are both complete, then M1 ∘ M2 is complete. -/
theorem concat_complete_complete (M1 M2 : WordTuple L C F Payload)
    (h1 : IsCompleteMSC M1) (h2 : IsCompleteMSC M2) :
    IsCompleteMSC (M1 ∘ₘ M2) where
  complete := tupleComplete_concat M1 M2 h1.complete h2.complete
  labelCompat := matchedLabelsCompatible_concat M1 M2 h1 (isCompleteMSC_implies_isMSC M2 h2)
  acyclic := ⟨concatRankingInstance M1 M2 h1 h2.acyclic⟩

/-- Removing a complete prefix preserves the no-unmatched-receive condition. -/
private theorem noUnmatchedReceives_suffix
    (M1 M2 : WordTuple L C F Payload)
    (h1 : tupleComplete M1)
    (h : noUnmatchedReceives (M1 ∘ₘ M2)) :
  noUnmatchedReceives M2 := by
  intro A B
  specialize h A B
  have hEq : countRecvs A (M1 B) = countSends B (M1 A) := by
    simpa [sndCount, rcvCount] using (h1 A B).symm
  simp [sndCount, rcvCount, WordTuple.concat, countSends_append, countRecvs_append] at h
  rw [hEq] at h
  simpa [sndCount, rcvCount] using h

/-- Removing a complete prefix preserves channelwise payload compatibility. -/
private theorem matchedLabelsCompatible_suffix
    (M1 M2 : WordTuple L C F Payload)
    (h1 : IsCompleteMSC M1)
    (h : IsMSC (M1 ∘ₘ M2)) :
    matchedLabelsCompatible M2 := by
  intro A B p hp
  have hlen :
      (sendPayloads B (M1 A)).length = (recvPayloads A (M1 B)).length := by
    simpa [sndCount, rcvCount] using h1.complete A B
  have hp' :
      p ∈ List.zip (sendPayloads B ((M1 ∘ₘ M2) A)) (recvPayloads A ((M1 ∘ₘ M2) B)) := by
    simp [WordTuple.concat, sendPayloads_append, recvPayloads_append]
    rw [zip_append_eq_of_length_eq _ _ _ _ hlen]
    exact List.mem_append.mpr (Or.inr hp)
  exact h.labelCompat A B p hp'

/-- Dropping an initial per-lifeline prefix preserves the MSC property when
    removed receives never exceed removed sends, and any strict send surplus
    occurs only on channels with no remaining receives. -/
theorem suffix_msc_of_safe_prefix
    (M : WordTuple L C F Payload)
    (k : L → Nat)
    (hk : ∀ A, k A ≤ (M A).length)
    (hPrefixLe : ∀ A B,
      countRecvs A ((M B).take (k B)) ≤ countSends B ((M A).take (k A)))
    (hSendSurplusDead : ∀ A B,
      countSends B ((M A).take (k A)) > countRecvs A ((M B).take (k B)) →
        countRecvs A ((M B).drop (k B)) = 0)
    (h : IsMSC M) :
    IsMSC (fun A => (M A).drop (k A)) where
  noUnmatchedRecv := by
    intro A B
    let prefS := countSends B ((M A).take (k A))
    let prefR := countRecvs A ((M B).take (k B))
    let sufS := countSends B ((M A).drop (k A))
    let sufR := countRecvs A ((M B).drop (k B))
    have hTotal : prefR + sufR ≤ prefS + sufS := by
      have hOrig : countRecvs A (M B) ≤ countSends B (M A) := by
        simpa [sndCount, rcvCount] using h.noUnmatchedRecv A B
      dsimp [prefS, prefR, sufS, sufR]
      calc
        countRecvs A ((M B).take (k B)) + countRecvs A ((M B).drop (k B))
            = countRecvs A (((M B).take (k B)) ++ ((M B).drop (k B))) := by
                rw [countRecvs_append]
        _ = countRecvs A (M B) := by
              rw [List.take_append_drop]
        _ ≤ countSends B (M A) := hOrig
        _ = countSends B (((M A).take (k A)) ++ ((M A).drop (k A))) := by
              rw [List.take_append_drop]
        _ = countSends B ((M A).take (k A)) + countSends B ((M A).drop (k A)) := by
              rw [countSends_append]
    have hPrefLe : prefR ≤ prefS := by
      exact hPrefixLe A B
    by_cases hSufR : sufR = 0
    · simp [sndCount, rcvCount, sufR, hSufR]
    · have hSufRPos : 0 < sufR := Nat.pos_of_ne_zero hSufR
      have hNotSurplus : ¬ prefS > prefR := by
        intro hSurplus
        have := hSendSurplusDead A B hSurplus
        exact hSufR this
      have hPrefEq : prefS = prefR := by
        omega
      have hSufLe : sufR ≤ sufS := by
        omega
      simpa [sndCount, rcvCount, sufR] using hSufLe
  labelCompat := by
    intro A B p hp
    let prefS := sendPayloads (C := C) (F := F) (Payload := Payload) B ((M A).take (k A))
    let prefR := recvPayloads (C := C) (F := F) (Payload := Payload) A ((M B).take (k B))
    let sufS := sendPayloads (C := C) (F := F) (Payload := Payload) B ((M A).drop (k A))
    let sufR := recvPayloads (C := C) (F := F) (Payload := Payload) A ((M B).drop (k B))
    have hSufRNe : sufR ≠ [] := by
      intro hNil
      simp [sufR, hNil] at hp
    have hSufRPos :
        0 < countRecvs A ((M B).drop (k B)) := by
      have hLenPos : 0 < sufR.length := by
        cases hs : sufR with
        | nil => contradiction
        | cons hd tl => simp
      dsimp [sufR] at hLenPos ⊢
      simpa [recvPayloads_length] using hLenPos
    have hNoSurplus :
        ¬ countSends B ((M A).take (k A)) > countRecvs A ((M B).take (k B)) := by
      intro hSurplus
      have hZero := hSendSurplusDead A B hSurplus
      exact Nat.ne_of_gt hSufRPos hZero
    have hPrefEq :
        countSends B ((M A).take (k A)) = countRecvs A ((M B).take (k B)) := by
      have hPrefLe := hPrefixLe A B
      omega
    have hLenEq : prefS.length = prefR.length := by
      dsimp [prefS, prefR]
      simpa [sendPayloads_length, recvPayloads_length] using hPrefEq
    have hp' :
        p ∈
          List.zip
            (sendPayloads (C := C) (F := F) (Payload := Payload) B (M A))
            (recvPayloads (C := C) (F := F) (Payload := Payload) A (M B)) := by
      dsimp [prefS, prefR, sufS, sufR] at hp hLenEq
      have hSendSplit :
          sendPayloads (C := C) (F := F) (Payload := Payload) B (M A) =
            sendPayloads (C := C) (F := F) (Payload := Payload) B ((M A).take (k A)) ++
              sendPayloads (C := C) (F := F) (Payload := Payload) B ((M A).drop (k A)) := by
        calc
          sendPayloads (C := C) (F := F) (Payload := Payload) B (M A)
              = sendPayloads (C := C) (F := F) (Payload := Payload) B
                  (((M A).take (k A)) ++ ((M A).drop (k A))) := by
                    rw [List.take_append_drop]
          _ = sendPayloads (C := C) (F := F) (Payload := Payload) B ((M A).take (k A)) ++
                sendPayloads (C := C) (F := F) (Payload := Payload) B ((M A).drop (k A)) := by
                  rw [sendPayloads_append]
      have hRecvSplit :
          recvPayloads (C := C) (F := F) (Payload := Payload) A (M B) =
            recvPayloads (C := C) (F := F) (Payload := Payload) A ((M B).take (k B)) ++
              recvPayloads (C := C) (F := F) (Payload := Payload) A ((M B).drop (k B)) := by
        calc
          recvPayloads (C := C) (F := F) (Payload := Payload) A (M B)
              = recvPayloads (C := C) (F := F) (Payload := Payload) A
                  (((M B).take (k B)) ++ ((M B).drop (k B))) := by
                    rw [List.take_append_drop]
          _ = recvPayloads (C := C) (F := F) (Payload := Payload) A ((M B).take (k B)) ++
                recvPayloads (C := C) (F := F) (Payload := Payload) A ((M B).drop (k B)) := by
                  rw [recvPayloads_append]
      rw [hSendSplit, hRecvSplit]
      rw [zip_append_eq_of_length_eq _ _ _ _ hLenEq]
      exact List.mem_append.mpr (Or.inr hp)
    exact h.labelCompat A B p hp'
  acyclic := by
    refine ⟨{
      rank := fun e => (Classical.choice h.acyclic).rank ⟨e.lifeline, k e.lifeline + e.pos⟩
      local_mono := ?_
      fifo_mono := ?_ }⟩
    · intro e1 e2 hSameLL hLt hpos1 hpos2
      let R := Classical.choice h.acyclic
      have hpos1' : k e1.lifeline + e1.pos < (M e1.lifeline).length := by
        have hdrop1 : e1.pos < (M e1.lifeline).length - k e1.lifeline := by
          simpa [List.length_drop] using hpos1
        omega
      have hpos2' : k e2.lifeline + e2.pos < (M e2.lifeline).length := by
        have hdrop2 : e2.pos < (M e2.lifeline).length - k e2.lifeline := by
          simpa [List.length_drop] using hpos2
        omega
      exact R.local_mono
        ⟨e1.lifeline, k e1.lifeline + e1.pos⟩
        ⟨e2.lifeline, k e2.lifeline + e2.pos⟩
        hSameLL
        (by
          have hLen : k e1.lifeline = k e2.lifeline := by simpa using congrArg k hSameLL
          simpa [hLen] using Nat.add_lt_add_left hLt (k e2.lifeline))
        hpos1' hpos2'
    · intro A B j1 j2 hj1 hj2 hSend hRecv hCntEq
      let R := Classical.choice h.acyclic
      have hj1' : k A + j1 < (M A).length := by
        have hdrop1 : j1 < (M A).length - k A := by
          simpa [List.length_drop] using hj1
        omega
      have hj2' : k B + j2 < (M B).length := by
        have hdrop2 : j2 < (M B).length - k B := by
          simpa [List.length_drop] using hj2
        omega
      have hSend' :
          ((M A).get ⟨k A + j1, hj1'⟩).val.isSendTo B = true := by
        simpa using (show (((M A).drop (k A))[j1]'hj1).val.isSendTo B = true from hSend)
      have hRecv' :
          ((M B).get ⟨k B + j2, hj2'⟩).val.isRecvFrom A = true := by
        simpa using (show (((M B).drop (k B))[j2]'hj2).val.isRecvFrom A = true from hRecv)
      have hTake1 :
          (M A).take (k A + j1) = (M A).take (k A) ++ ((M A).drop (k A)).take j1 := by
        rw [← List.take_append_drop (k A) (M A), List.take_append]
        have hTakePrefix : ((M A).take (k A)).take (k A + j1) = (M A).take (k A) := by
          apply List.take_of_length_le
          simp [List.length_take, hk A]
        simpa [hTakePrefix, List.length_take, hk A, Nat.add_sub_cancel_left]
      have hTake2 :
          (M B).take (k B + j2) = (M B).take (k B) ++ ((M B).drop (k B)).take j2 := by
        rw [← List.take_append_drop (k B) (M B), List.take_append]
        have hTakePrefix : ((M B).take (k B)).take (k B + j2) = (M B).take (k B) := by
          apply List.take_of_length_le
          simp [List.length_take, hk B]
        simpa [hTakePrefix, List.length_take, hk B, Nat.add_sub_cancel_left]
      have hCntEq' :
          countSends B ((M A).take (k A + j1)) =
            countRecvs A ((M B).take (k B + j2)) := by
        rw [hTake1, hTake2]
        simp [countSends_append, countRecvs_append, hCntEq]
        have hRecvPos :
            0 < countRecvs A ((M B).drop (k B)) := by
          exact countRecvs_pos_of_recvAt
            (C := C) (F := F) (Payload := Payload)
            (A := A) ((M B).drop (k B)) j2 hj2 hRecv
        have hNoSurplus :
            ¬ countSends B ((M A).take (k A)) > countRecvs A ((M B).take (k B)) := by
          intro hSurplus
          have hZero := hSendSurplusDead A B hSurplus
          exact Nat.ne_of_gt hRecvPos hZero
        have hPrefEq :
            countSends B ((M A).take (k A)) = countRecvs A ((M B).take (k B)) := by
          have hPrefLe := hPrefixLe A B
          omega
        omega
      exact R.fifo_mono A B (k A + j1) (k B + j2) hj1' hj2' hSend' hRecv' hCntEq'

/-- The tuple obtained by stripping the available part of a decision-prefix
    tuple `D` from an MSC tuple `M`.  If `M X` has not yet reached the end of
    `D X`, the result on `X` is empty; if `M X = D X ++ s`, the result is `s`. -/
def stripDecisionPrefix
    (D M : WordTuple L C F Payload) : WordTuple L C F Payload :=
  fun X => (M X).drop (Nat.min (D X).length (M X).length)

private theorem take_min_decision_prefix_eq
    (D M : WordTuple L C F Payload)
    (hForm : ∀ X,
      IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload)
        (M X) (D X) ∨
      ∃ s, M X = D X ++ s)
    (X : L) :
    (M X).take (Nat.min (D X).length (M X).length) =
      (D X).take (Nat.min (D X).length (M X).length) := by
  rcases hForm X with hPref | hFull
  · rcases hPref with ⟨t, hD⟩
    rw [hD]
    simp [Nat.min_eq_right]
  · rcases hFull with ⟨s, hM⟩
    rw [hM]
    simp [Nat.min_eq_left]

/-- Stripping a complete prefix-like tuple from an MSC preserves the MSC
    property.  On each lifeline, `M` may either stop inside `D` or continue
    with a suffix after all of `D`; `stripDecisionPrefix` keeps exactly that
    suffix. -/
theorem strip_complete_prefix_like_msc
    (D : WordTuple L C F Payload)
    (hD : IsCompleteMSC D)
    (M : WordTuple L C F Payload)
    (hM : IsMSC M)
    (hForm : ∀ X,
      IsPrefixWord (L := L) (C := C) (F := F) (Payload := Payload)
        (M X) (D X) ∨
      ∃ s, M X = D X ++ s) :
    IsMSC (stripDecisionPrefix (L := L) (C := C) (F := F) (Payload := Payload)
      D M) := by
  let k : L → Nat := fun X => Nat.min (D X).length (M X).length
  have hkPrefix : ∀ {X : L}, IsPrefixWord (M X) (D X) → k X = (M X).length := by
    intro X hPref
    rcases hPref with ⟨t, hD_eq⟩
    dsimp [k]
    rw [hD_eq]
    simp [Nat.min_eq_right]
  have hkFull : ∀ {X : L}, (∃ s, M X = D X ++ s) → k X = (D X).length := by
    intro X hFull
    rcases hFull with ⟨s, hM_eq⟩
    dsimp [k]
    rw [hM_eq]
    simp [Nat.min_eq_left]
  have hTakeM_of_prefix :
      ∀ {X : L}, IsPrefixWord (M X) (D X) → (M X).take (k X) = M X := by
    intro X hPref
    rw [hkPrefix hPref]
    exact List.take_length (l := M X)
  have hTakeD_of_full :
      ∀ {X : L}, (∃ s, M X = D X ++ s) → (D X).take (k X) = D X := by
    intro X hFull
    rw [hkFull hFull]
    exact List.take_length (l := D X)
  have hkM : ∀ X, k X ≤ (M X).length := by
    intro X
    exact Nat.min_le_right _ _
  have hkD : ∀ X, k X ≤ (D X).length := by
    intro X
    exact Nat.min_le_left _ _
  have hTakeEq : ∀ X, (M X).take (k X) = (D X).take (k X) := by
    intro X
    exact take_min_decision_prefix_eq
      (L := L) (C := C) (F := F) (Payload := Payload) D M hForm X
  have hPrefixLe : ∀ A X,
      countRecvs A ((M X).take (k X)) ≤
        countSends X ((M A).take (k A)) := by
    intro A X
    rw [hTakeEq X, hTakeEq A]
    rcases hForm X with hXPref | hXFull
    · rcases hForm A with hAPref | hAFull
      · have hXM : (D X).take (k X) = M X := by
          rw [← hTakeEq X, hTakeM_of_prefix hXPref]
        have hAM : (D A).take (k A) = M A := by
          rw [← hTakeEq A, hTakeM_of_prefix hAPref]
        rw [hXM, hAM]
        simpa [sndCount, rcvCount] using hM.noUnmatchedRecv A X
      · have hXD_le : countRecvs A ((D X).take (k X)) ≤ countRecvs A (D X) :=
          countRecvs_take_le
            (C := C) (F := F) (Payload := Payload) A (D X) (k X)
        have hAD : (D A).take (k A) = D A := hTakeD_of_full hAFull
        rw [hAD]
        have hComplete : countRecvs A (D X) = countSends X (D A) := by
          simpa [sndCount, rcvCount] using (hD.complete A X).symm
        omega
    · rcases hForm A with hAPref | hAFull
      · rcases hXFull with ⟨sX, hMX⟩
        have hXD : (D X).take (k X) = D X := by
          exact hTakeD_of_full ⟨sX, hMX⟩
        have hAM : (D A).take (k A) = M A := by
          rw [← hTakeEq A, hTakeM_of_prefix hAPref]
        rw [hXD, hAM]
        have hTotal := hM.noUnmatchedRecv A X
        simp [sndCount, rcvCount, hMX, countRecvs_append] at hTotal
        omega
      · have hXD : (D X).take (k X) = D X := hTakeD_of_full hXFull
        have hAD : (D A).take (k A) = D A := hTakeD_of_full hAFull
        rw [hXD, hAD]
        exact Nat.le_of_eq (by
          simpa [sndCount, rcvCount] using (hD.complete A X).symm)
  have hSendSurplusDead : ∀ A X,
      countSends X ((M A).take (k A)) >
        countRecvs A ((M X).take (k X)) →
      countRecvs A ((M X).drop (k X)) = 0 := by
    intro A X hSurplus
    rcases hForm X with hPref | hFull
    · rcases hPref with ⟨t, hDX⟩
      have hkX : k X = (M X).length := by
        dsimp [k]
        rw [hDX]
        simp [Nat.min_eq_right]
      simp [hkX]
    · rcases hFull with ⟨s, hMX⟩
      have hkX : k X = (D X).length := by
        exact hkFull ⟨s, hMX⟩
      have hSurplusD :
          countSends X ((D A).take (k A)) >
            countRecvs A ((D X).take (k X)) := by
        rwa [hTakeEq A, hTakeEq X] at hSurplus
      have hRecvD : (D X).take (k X) = D X := by
        rw [hkX]
        exact List.take_length (l := D X)
      have hNoSurplus : countSends X ((D A).take (k A)) ≤ countRecvs A (D X) := by
        rcases hForm A with hAPref | hAFull
        · have hSendTakeLe :
              countSends X ((D A).take (k A)) ≤ countSends X (D A) :=
            countSends_take_le
              (C := C) (F := F) (Payload := Payload) X (D A) (k A)
          have hComplete : countSends X (D A) = countRecvs A (D X) := by
            simpa [sndCount, rcvCount] using hD.complete A X
          omega
        · have hAD : (D A).take (k A) = D A := hTakeD_of_full hAFull
          rw [hAD]
          have hComplete : countSends X (D A) = countRecvs A (D X) := by
            simpa [sndCount, rcvCount] using hD.complete A X
          omega
      rw [hRecvD] at hSurplusD
      omega
  simpa [stripDecisionPrefix, k] using
    suffix_msc_of_safe_prefix
      (L := L) (C := C) (F := F) (Payload := Payload)
      M k hkM hPrefixLe hSendSurplusDead hM

/-- Restrict a ranking on `M1 ∘ₘ M2` to the suffix `M2` by shifting event
    positions past the complete prefix `M1`. -/
private def suffixRanking
    (M1 M2 : WordTuple L C F Payload)
    (h1 : tupleComplete M1)
    (R : CausalRanking (M1 ∘ₘ M2)) :
    CausalRanking M2 where
  rank := fun e => R.rank ⟨e.lifeline, (M1 e.lifeline).length + e.pos⟩
  local_mono := by
    intro e1 e2 hSameLL hLt hpos1 hpos2
    refine R.local_mono
      ⟨e1.lifeline, (M1 e1.lifeline).length + e1.pos⟩
      ⟨e2.lifeline, (M1 e2.lifeline).length + e2.pos⟩
      ?_ ?_ ?_ ?_
    · simpa using hSameLL
    · have hLen : (M1 e1.lifeline).length = (M1 e2.lifeline).length := by
        simpa using congrArg (fun X => (M1 X).length) hSameLL
      simpa [hLen] using Nat.add_lt_add_left hLt ((M1 e2.lifeline).length)
    · simp [WordTuple.concat]
      omega
    · simp [WordTuple.concat]
      omega
  fifo_mono := by
    intro A B j1 j2 hj1 hj2 hSend hRecv hCntEq
    have hj1' : (M1 A).length + j1 < ((M1 ∘ₘ M2) A).length := by
      simp [WordTuple.concat]
      omega
    have hj2' : (M1 B).length + j2 < ((M1 ∘ₘ M2) B).length := by
      simp [WordTuple.concat]
      omega
    have hSend' :
        (((M1 ∘ₘ M2) A).get ⟨(M1 A).length + j1, hj1'⟩).val.isSendTo B = true := by
      simpa [WordTuple.concat] using
        (show (((M1 A ++ M2 A).get ⟨(M1 A).length + j1, by
            simpa [WordTuple.concat] using hj1'⟩).val.isSendTo B = true) from by
          simpa using hSend)
    have hRecv' :
        (((M1 ∘ₘ M2) B).get ⟨(M1 B).length + j2, hj2'⟩).val.isRecvFrom A = true := by
      simpa [WordTuple.concat] using
        (show (((M1 B ++ M2 B).get ⟨(M1 B).length + j2, by
            simpa [WordTuple.concat] using hj2'⟩).val.isRecvFrom A = true) from by
          simpa using hRecv)
    have hTake1 :
        ((M1 ∘ₘ M2) A).take ((M1 A).length + j1) =
          M1 A ++ (M2 A).take j1 := by
      rw [WordTuple.concat, List.take_append]
      have hTakeM1 : (M1 A).take ((M1 A).length + j1) = M1 A := by
        have hDrop : (M1 A).drop ((M1 A).length + j1) = [] := List.drop_eq_nil_of_le (by omega)
        calc
          (M1 A).take ((M1 A).length + j1) =
              (M1 A).take ((M1 A).length + j1) ++ (M1 A).drop ((M1 A).length + j1) := by
                simp [hDrop]
          _ = M1 A := List.take_append_drop ((M1 A).length + j1) (M1 A)
      simpa [hTakeM1, Nat.add_sub_cancel_left]
    have hTake2 :
        ((M1 ∘ₘ M2) B).take ((M1 B).length + j2) =
          M1 B ++ (M2 B).take j2 := by
      rw [WordTuple.concat, List.take_append]
      have hTakeM1 : (M1 B).take ((M1 B).length + j2) = M1 B := by
        have hDrop : (M1 B).drop ((M1 B).length + j2) = [] := List.drop_eq_nil_of_le (by omega)
        calc
          (M1 B).take ((M1 B).length + j2) =
              (M1 B).take ((M1 B).length + j2) ++ (M1 B).drop ((M1 B).length + j2) := by
                simp [hDrop]
          _ = M1 B := List.take_append_drop ((M1 B).length + j2) (M1 B)
      simpa [hTakeM1, Nat.add_sub_cancel_left]
    have hCntEq' :
        countSends B (((M1 ∘ₘ M2) A).take ((M1 A).length + j1)) =
          countRecvs A (((M1 ∘ₘ M2) B).take ((M1 B).length + j2)) := by
      rw [hTake1, hTake2]
      simp [countSends_append, countRecvs_append]
      have hEq := h1 A B
      simp [channelComplete, sndCount, rcvCount] at hEq
      omega
    simpa using R.fifo_mono A B ((M1 A).length + j1) ((M1 B).length + j2)
      hj1' hj2' hSend' hRecv' hCntEq'

/-- Removing a complete prefix from an MSC preserves the MSC property. -/
theorem suffix_msc_of_complete_prefix
    (M1 M2 : WordTuple L C F Payload)
    (h1 : IsCompleteMSC M1)
    (h : IsMSC (M1 ∘ₘ M2)) :
    IsMSC M2 where
  noUnmatchedRecv := noUnmatchedReceives_suffix M1 M2 h1.complete h.noUnmatchedRecv
  labelCompat := matchedLabelsCompatible_suffix M1 M2 h1 h
  acyclic := ⟨suffixRanking M1 M2 h1.complete (Classical.choice h.acyclic)⟩

/-- Removing a complete prefix from a complete MSC preserves completeness. -/
theorem suffix_completeMSC_of_complete_prefix
    (M1 M2 : WordTuple L C F Payload)
    (h1 : IsCompleteMSC M1)
    (h : IsCompleteMSC (M1 ∘ₘ M2)) :
    IsCompleteMSC M2 := by
  have hConcatMSC : IsMSC (M1 ∘ₘ M2) :=
    isCompleteMSC_implies_isMSC (M1 ∘ₘ M2) h
  have hSuffixMSC : IsMSC M2 :=
    suffix_msc_of_complete_prefix M1 M2 h1 hConcatMSC
  refine
    { complete := ?_
      labelCompat := hSuffixMSC.labelCompat
      acyclic := hSuffixMSC.acyclic }
  intro A B
  have h1AB := h1.complete A B
  have hAB := h.complete A B
  simp [channelComplete, sndCount, rcvCount, WordTuple.concat,
    countSends_append, countRecvs_append] at h1AB hAB ⊢
  omega

/-- The concrete IsMSCPredicate instance. -/
noncomputable instance concreteMSCPredicate : IsMSCPredicate L C F Payload where
  isMSC := fun M => IsMSC M
  isComplete := fun M => IsCompleteMSC M
  complete_implies_msc := fun M h => isCompleteMSC_implies_isMSC M h
  empty_is_complete := mscEmpty_isCompleteMSC
  concat_msc := fun M1 M2 h1 h2 => concat_complete_msc M1 M2 h1 h2
  concat_complete := fun M1 M2 h1 h2 => concat_complete_complete M1 M2 h1 h2

end MSCPredicateInstance
