/-
  MSCAgents/FIFORelation.lean
  ===========================
  Formalization of Definition 3 (def:tuple-fifo) from:
  "Provable Coordination for LLM Agents via Message Sequence Charts"

  Paper section: sec:semantics

  From the paper (def:tuple-fifo):
    Fix a tuple of local words M = (w_A)_{A ∈ 𝓛} with w_A ∈ (Σ_A)* for each A.
    Its event set is E_M = {(A, i) | A ∈ 𝓛, 1 ≤ i ≤ |w_A|}.

    For each channel A→B, define:
      snd_{A→B}(M) = sequence of events (A,j) where w_A[j] = send A(·) → B
      rcv_{A→B}(M) = sequence of events (B,j) where w_B[j] = recv B(·) ← A

    The FIFO relation ⊲_M ⊆ E_M × E_M contains, for each channel A→B,
    the pairs (s_i, r_i) for all 1 ≤ i ≤ min{|snd|, |rcv|}.

    M is complete if for every channel A→B, |snd_{A→B}(M)| = |rcv_{A→B}(M)|.
-/

import MSCAgents.Alphabets
import MSCAgents.MSCConcat
import MSCAgents.CanonicalMSC

------------------------------------------------------------------------
-- Events
------------------------------------------------------------------------

/-- An event in a word tuple M = (w_A)_A is a pair (A, i) where A is
    a lifeline and i is a 0-based index into w_A.
    The paper uses 1-based indexing; we use 0-based for Lean lists. -/
structure Event (L : Type) : Type where
  /-- The lifeline owning this event. -/
  lifeline : L
  /-- The 0-based position in the local word w_A. -/
  pos : Nat
  deriving DecidableEq, Repr

------------------------------------------------------------------------
-- Predicates on letters: is this a send/recv for a given channel?
------------------------------------------------------------------------

section LetterPredicates

variable {L C F Payload : Type} [DecidableEq L]

/-- Check whether a letter is a send letter targeting lifeline B. -/
def Letter.isSendTo (B : L) : Letter L C F Payload → Bool
  | .sendLetter _ _ tgt _ => decide (tgt = B)
  | _ => false

/-- Check whether a letter is a receive letter from lifeline A. -/
def Letter.isRecvFrom (A : L) : Letter L C F Payload → Bool
  | .recvLetter _ _ src => decide (src = A)
  | _ => false

/-- A letter that is not a sendLetter or recvLetter is called
    non-communicating. -/
def Letter.isNonComm : Letter L C F Payload → Bool
  | .sendLetter _ _ _ _ => false
  | .recvLetter _ _ _   => false
  | _ => true

/-- A non-communicating letter is not a send to any target. -/
theorem Letter.isNonComm_not_sendTo (ℓ : Letter L C F Payload)
    (h : ℓ.isNonComm = true) (B : L) : ℓ.isSendTo B = false := by
  cases ℓ <;> simp [Letter.isNonComm] at h <;> simp [Letter.isSendTo]

/-- A non-communicating letter is not a recv from any source. -/
theorem Letter.isNonComm_not_recvFrom (ℓ : Letter L C F Payload)
    (h : ℓ.isNonComm = true) (A : L) : ℓ.isRecvFrom A = false := by
  cases ℓ <;> simp [Letter.isNonComm] at h <;> simp [Letter.isRecvFrom]

end LetterPredicates

------------------------------------------------------------------------
-- Counting sends and receives in a local word
------------------------------------------------------------------------

section Counting

variable {L C F Payload : Type} [DecidableEq L]

/-- Count the number of send letters targeting B in a local word.
    `A` is the owner lifeline of the local word. -/
def countSends {A : L} (B : L) : List (AlphabetOf (C := C) (F := F) (Payload := Payload) A) → Nat
  | [] => 0
  | hd :: rest =>
    (if hd.val.isSendTo B then 1 else 0) + countSends B rest

/-- Count the number of receive letters from A in a local word.
    `B` is the owner lifeline of the local word. -/
def countRecvs {B : L} (A : L) : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B) → Nat
  | [] => 0
  | hd :: rest =>
    (if hd.val.isRecvFrom A then 1 else 0) + countRecvs A rest

/-- Count sends in the concatenation equals the sum of counts. -/
@[simp]
theorem countSends_append {A : L} (B : L)
    (w1 w2 : List (AlphabetOf (C := C) (F := F) (Payload := Payload) A)) :
    countSends B (w1 ++ w2) = countSends B w1 + countSends B w2 := by
  induction w1 with
  | nil => simp [countSends]
  | cons hd tl ih =>
    simp [countSends, ih]
    omega

/-- Count recvs in the concatenation equals the sum of counts. -/
@[simp]
theorem countRecvs_append {B : L} (A : L)
    (w1 w2 : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B)) :
    countRecvs A (w1 ++ w2) = countRecvs A w1 + countRecvs A w2 := by
  induction w1 with
  | nil => simp [countRecvs]
  | cons hd tl ih =>
    simp [countRecvs, ih]
    omega

/-- Send count in an empty word is 0. -/
@[simp]
theorem countSends_nil {A : L} (B : L) :
    countSends (C := C) (F := F) (Payload := Payload) B ([] : List (AlphabetOf A)) = 0 := rfl

/-- Recv count in an empty word is 0. -/
@[simp]
theorem countRecvs_nil {B : L} (A : L) :
    countRecvs (C := C) (F := F) (Payload := Payload) A ([] : List (AlphabetOf B)) = 0 := rfl

end Counting

------------------------------------------------------------------------
-- Channel send/recv counts for a word tuple
------------------------------------------------------------------------

section ChannelCounts

variable {L C F Payload : Type} [DecidableEq L]

/-- An event belongs to the event set `E_M` when its index lies inside the
    local word of its owning lifeline. This is the paper's event-set
    definition with 0-based indices. -/
def Event.InTuple (M : WordTuple L C F Payload) (e : Event L) : Prop :=
  e.pos < (M e.lifeline).length

/-- Collect the send events on channel `A → B` in increasing local order,
    starting the scan at offset `i`. -/
def sendEventsAux {A : L} (B : L) :
    Nat → List (AlphabetOf (C := C) (F := F) (Payload := Payload) A) → List (Event L)
  | _, [] => []
  | i, hd :: tl =>
      let rest := sendEventsAux B (i + 1) tl
      if hd.val.isSendTo B then
        ⟨A, i⟩ :: rest
      else
        rest

/-- Collect the receive events on channel `A → B` in increasing local order,
    starting the scan at offset `i`. -/
def recvEventsAux {B : L} (A : L) :
    Nat → List (AlphabetOf (C := C) (F := F) (Payload := Payload) B) → List (Event L)
  | _, [] => []
  | i, hd :: tl =>
      let rest := recvEventsAux A (i + 1) tl
      if hd.val.isRecvFrom A then
        ⟨B, i⟩ :: rest
      else
        rest

/-- The send-event sequence `snd_{A→B}(M)` from Definition `def:tuple-fifo`. -/
def sndEvents (M : WordTuple L C F Payload) (A B : L) : List (Event L) :=
  sendEventsAux B 0 (M A)

/-- The receive-event sequence `rcv_{A→B}(M)` from Definition `def:tuple-fifo`. -/
def rcvEvents (M : WordTuple L C F Payload) (A B : L) : List (Event L) :=
  recvEventsAux A 0 (M B)

/-- The FIFO event pairs contributed by a single channel `A → B`. -/
def fifoPairsOn (M : WordTuple L C F Payload) (A B : L) : List (Event L × Event L) :=
  List.zip (sndEvents M A B) (rcvEvents M A B)

/-- The tuple FIFO relation `⊲_M` from Definition `def:tuple-fifo`. -/
def FIFORel (M : WordTuple L C F Payload) : Event L → Event L → Prop :=
  fun e₁ e₂ => ∃ A B, (e₁, e₂) ∈ fifoPairsOn M A B

/-- The number of send events on channel A→B in word tuple M. -/
def sndCount (M : WordTuple L C F Payload) (A B : L) : Nat :=
  countSends B (M A)

/-- The number of receive events on channel A→B in word tuple M. -/
def rcvCount (M : WordTuple L C F Payload) (A B : L) : Nat :=
  countRecvs A (M B)

/-- The number of collected send events agrees with the send count. -/
@[simp]
theorem sendEventsAux_length {A : L} (B : L) (i : Nat)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) A)) :
    (sendEventsAux B i w).length = countSends B w := by
  induction w generalizing i with
  | nil => simp [sendEventsAux, countSends]
  | cons hd tl ih =>
      by_cases hsend : hd.val.isSendTo B = true
      · simp [sendEventsAux, countSends, hsend, ih (i + 1)]
        omega
      · simp [sendEventsAux, countSends, hsend, ih (i + 1)]

/-- The number of collected receive events agrees with the receive count. -/
@[simp]
theorem recvEventsAux_length {B : L} (A : L) (i : Nat)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B)) :
    (recvEventsAux A i w).length = countRecvs A w := by
  induction w generalizing i with
  | nil => simp [recvEventsAux, countRecvs]
  | cons hd tl ih =>
      by_cases hrecv : hd.val.isRecvFrom A = true
      · simp [recvEventsAux, countRecvs, hrecv, ih (i + 1)]
        omega
      · simp [recvEventsAux, countRecvs, hrecv, ih (i + 1)]

/-- The send-event sequence length is exactly `sndCount`. -/
@[simp]
theorem length_sndEvents (M : WordTuple L C F Payload) (A B : L) :
    (sndEvents M A B).length = sndCount M A B := by
  simp [sndEvents, sndCount]

/-- The receive-event sequence length is exactly `rcvCount`. -/
@[simp]
theorem length_rcvEvents (M : WordTuple L C F Payload) (A B : L) :
    (rcvEvents M A B).length = rcvCount M A B := by
  simp [rcvEvents, rcvCount]

/-- The FIFO list has one pair per matched send/receive on the channel. -/
@[simp]
theorem length_fifoPairsOn (M : WordTuple L C F Payload) (A B : L) :
    (fifoPairsOn M A B).length = min (sndCount M A B) (rcvCount M A B) := by
  simp [fifoPairsOn]

end ChannelCounts

------------------------------------------------------------------------
-- Completeness (def:tuple-fifo)
------------------------------------------------------------------------

section Completeness

variable {L C F Payload : Type} [DecidableEq L]

/-- A word tuple M is FIFO-complete on channel A→B if
    |snd_{A→B}(M)| = |rcv_{A→B}(M)|. -/
def channelComplete (M : WordTuple L C F Payload) (A B : L) : Prop :=
  sndCount M A B = rcvCount M A B

/-- A word tuple M is complete if every channel A→B is complete. -/
def tupleComplete (M : WordTuple L C F Payload) : Prop :=
  ∀ A B : L, channelComplete M A B

/-- The no-unmatched-receives condition: for each channel A→B,
    |rcv_{A→B}(M)| ≤ |snd_{A→B}(M)|.
    This is condition (1) in def:msc. -/
def noUnmatchedReceives (M : WordTuple L C F Payload) : Prop :=
  ∀ A B : L, rcvCount M A B ≤ sndCount M A B

/-- Completeness implies no unmatched receives. -/
theorem tupleComplete_implies_noUnmatchedReceives
    (M : WordTuple L C F Payload)
    (hc : tupleComplete M) :
    noUnmatchedReceives M := by
  intro A B
  exact Nat.le_of_eq (hc A B).symm

end Completeness

------------------------------------------------------------------------
-- Concatenation lemmas for send/recv counts
------------------------------------------------------------------------

section ConcatCounts

variable {L C F Payload : Type} [DecidableEq L]

/-- Send count in a concatenation is the sum of individual counts. -/
theorem sndCount_concat (M1 M2 : WordTuple L C F Payload) (A B : L) :
    sndCount (M1 ∘ₘ M2) A B = sndCount M1 A B + sndCount M2 A B := by
  simp [sndCount, WordTuple.concat, countSends_append]

/-- Recv count in a concatenation is the sum of individual counts. -/
theorem rcvCount_concat (M1 M2 : WordTuple L C F Payload) (A B : L) :
    rcvCount (M1 ∘ₘ M2) A B = rcvCount M1 A B + rcvCount M2 A B := by
  simp [rcvCount, WordTuple.concat, countRecvs_append]

/-- Concatenation of complete tuples is complete. -/
theorem tupleComplete_concat (M1 M2 : WordTuple L C F Payload)
    (h1 : tupleComplete M1) (h2 : tupleComplete M2) :
    tupleComplete (M1 ∘ₘ M2) := by
  intro A B
  simp [channelComplete, sndCount_concat, rcvCount_concat]
  have := h1 A B
  have := h2 A B
  simp [channelComplete] at *
  omega

/-- If M1 is complete and M2 has no unmatched receives, then
    M1 ∘ M2 has no unmatched receives. -/
theorem noUnmatchedReceives_concat
    (M1 M2 : WordTuple L C F Payload)
    (h1 : tupleComplete M1) (h2 : noUnmatchedReceives M2) :
    noUnmatchedReceives (M1 ∘ₘ M2) := by
  intro A B
  simp [sndCount_concat, rcvCount_concat]
  have hc := h1 A B
  have hn := h2 A B
  simp [channelComplete] at hc
  omega

end ConcatCounts

------------------------------------------------------------------------
-- Causal ranking (witness of acyclicity)
------------------------------------------------------------------------

section CausalRanking

variable {L C F Payload : Type} [DecidableEq L]

/-- A causal ranking is a function from events to natural numbers
    that is strictly monotone on both local-order edges and FIFO edges.
    The existence of a causal ranking witnesses acyclicity of the
    causality relation.

    Using rankings instead of the transitive-closure formulation makes
    concatenation preservation easy: just shift M2's ranking. -/
structure CausalRanking (M : WordTuple L C F Payload) where
  /-- The ranking function on events. -/
  rank : Event L → Nat
  /-- Local-order edges are strictly increasing. -/
  local_mono : ∀ (e1 e2 : Event L),
    e1.lifeline = e2.lifeline →
    e1.pos < e2.pos →
    e1.pos < (M e1.lifeline).length →
    e2.pos < (M e2.lifeline).length →
    rank e1 < rank e2
  /-- FIFO edges are strictly increasing.
      For channel A→B, if the k-th send is at (A,j1) and the k-th recv
      is at (B,j2), then rank (A,j1) < rank (B,j2). -/
  fifo_mono : ∀ (A B : L) (j1 j2 : Nat)
    (hj1 : j1 < (M A).length) (hj2 : j2 < (M B).length),
    ((M A).get ⟨j1, hj1⟩).val.isSendTo B = true →
    ((M B).get ⟨j2, hj2⟩).val.isRecvFrom A = true →
    countSends B ((M A).take j1) = countRecvs A ((M B).take j2) →
    rank ⟨A, j1⟩ < rank ⟨B, j2⟩

/-- A word tuple has an acyclic causality if it admits a causal ranking. -/
def hasAcyclicCausality (M : WordTuple L C F Payload) : Prop :=
  Nonempty (CausalRanking M)

end CausalRanking

------------------------------------------------------------------------
-- Ranking for empty word tuples
------------------------------------------------------------------------

section EmptyRanking

variable {L C F Payload : Type} [DecidableEq L]

/-- The empty word tuple admits the trivial ranking (any function works). -/
def emptyRanking : CausalRanking (mscEmpty (L := L) (C := C) (F := F) (Payload := Payload)) where
  rank := fun _ => 0
  local_mono := by
    intro e1 _ _ _ h1 _
    simp [mscEmpty, WordTuple.empty] at h1
  fifo_mono := by
    intro A _ j1 _ h1 _ _ _ _
    simp [mscEmpty, WordTuple.empty] at h1

end EmptyRanking
