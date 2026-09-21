/-
  MSCAgents/Erasure.lean
  ======================
  Formalization of Definition `def:erase` from sec:projection of:
  "Provable Coordination for LLM Agents via Message Sequence Charts"
-/

import MSCAgents.ControlPayload
import MSCAgents.LocalSemantics
import MSCAgents.InductiveSemantics

section Erasure

variable {L C F Payload : Type} [DecidableEq L]
variable [PayloadCompatiblePred Payload] [ControlPayloadSpec Payload]

/-- Erasure keeps non-control letters and deletes control send/receive letters. -/
def eraseLetter {A : L}
    (ℓ : AlphabetOf (C := C) (F := F) (Payload := Payload) A) :
    Option (AlphabetOf (C := C) (F := F) (Payload := Payload) A) :=
  match hℓ : ℓ.val with
  | .sendLetter owner xs tgt _ =>
      if isControlPayload xs then none else some ℓ
  | .recvLetter owner ys src =>
      if isControlPayload ys then none else some ℓ
  | _ => some ℓ

/-- Local erasure `erase_A`. -/
def eraseWord {A : L} :
    LocalWord (C := C) (F := F) (Payload := Payload) A →
    LocalWord (C := C) (F := F) (Payload := Payload) A
  | [] => []
  | hd :: tl =>
      match eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | some hd' => hd' :: eraseWord tl
      | none => eraseWord tl

/-- MSC erasure `erase(M)`. -/
def eraseTuple (M : WordTuple L C F Payload) : WordTuple L C F Payload :=
  fun A => eraseWord (M A)

notation "erase" => eraseTuple

/-- Payloads that survive erasure are exactly the non-control payloads. -/
def keepUserPayload (p : Payload) : Bool :=
  ! isControlPayload p

/-- Surviving letters paired with their original positions, starting from offset `i`. -/
def eraseTraceAux {A : L} :
    Nat →
    LocalWord (C := C) (F := F) (Payload := Payload) A →
    List (AlphabetOf (C := C) (F := F) (Payload := Payload) A × Nat)
  | _, [] => []
  | i, hd :: tl =>
      match eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | some hd' => (hd', i) :: eraseTraceAux (i + 1) tl
      | none => eraseTraceAux (i + 1) tl

/-- Surviving letters paired with their original positions. -/
def eraseTrace {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    List (AlphabetOf (C := C) (F := F) (Payload := Payload) A × Nat) :=
  eraseTraceAux (C := C) (F := F) (Payload := Payload) 0 w

@[simp]
theorem eraseWord_nil {A : L} :
    eraseWord (C := C) (F := F) (Payload := Payload) ([] :
      LocalWord (C := C) (F := F) (Payload := Payload) A) = [] := rfl

@[simp]
theorem eraseWord_cons {A : L}
    (hd : AlphabetOf (C := C) (F := F) (Payload := Payload) A)
    (tl : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    eraseWord (C := C) (F := F) (Payload := Payload) (hd :: tl) =
      match eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | some hd' => hd' :: eraseWord tl
      | none => eraseWord tl := rfl

@[simp]
theorem eraseWord_append {A : L}
    (u v : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    eraseWord (C := C) (F := F) (Payload := Payload) (u ++ v) =
      eraseWord u ++ eraseWord v := by
  induction u with
  | nil => simp [eraseWord]
  | cons hd tl ih =>
      simp [eraseWord, ih]
      split <;> simp [ih]

@[simp]
theorem eraseTraceAux_map_fst {A : L} (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    (eraseTraceAux (C := C) (F := F) (Payload := Payload) i w).map Prod.fst =
      eraseWord (C := C) (F := F) (Payload := Payload) w := by
  induction w generalizing i with
  | nil =>
      simp [eraseTraceAux, eraseWord]
  | cons hd tl ih =>
      simp [eraseTraceAux, eraseWord, ih]
      split <;> simp [ih]

@[simp]
theorem eraseTrace_map_fst {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    (eraseTrace (C := C) (F := F) (Payload := Payload) w).map Prod.fst =
      eraseWord (C := C) (F := F) (Payload := Payload) w := by
  simpa [eraseTrace] using eraseTraceAux_map_fst (C := C) (F := F) (Payload := Payload) 0 w

@[simp]
theorem eraseTraceAux_length {A : L} (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    (eraseTraceAux (C := C) (F := F) (Payload := Payload) i w).length =
      (eraseWord (C := C) (F := F) (Payload := Payload) w).length := by
  induction w generalizing i with
  | nil =>
      simp [eraseTraceAux, eraseWord]
  | cons hd tl ih =>
      simp [eraseTraceAux, eraseWord, ih]
      split <;> simp [ih]

@[simp]
theorem eraseTrace_length {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    (eraseTrace (C := C) (F := F) (Payload := Payload) w).length =
      (eraseWord (C := C) (F := F) (Payload := Payload) w).length := by
  simpa [eraseTrace] using eraseTraceAux_length (C := C) (F := F) (Payload := Payload) 0 w

private theorem eraseTraceAux_snd_ge {A : L} (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    ∀ n ∈ eraseTraceAux (C := C) (F := F) (Payload := Payload) i w, i ≤ n.2 := by
  induction w generalizing i with
  | nil =>
      intro n hn
      simp [eraseTraceAux] at hn
  | cons hd tl ih =>
      intro n hn
      cases he : eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | none =>
          simp [eraseTraceAux, he] at hn
          exact Nat.le_trans (Nat.le_succ i) (ih (i + 1) n hn)
      | some hd' =>
          simp [eraseTraceAux, he] at hn
          rcases hn with rfl | hn
          · simp
          · exact Nat.le_trans (Nat.le_succ i) (ih (i + 1) n hn)

private theorem eraseTraceAux_snd_lt {A : L} (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    ∀ n ∈ eraseTraceAux (C := C) (F := F) (Payload := Payload) i w, n.2 < i + w.length := by
  induction w generalizing i with
  | nil =>
      intro n hn
      simp [eraseTraceAux] at hn
  | cons hd tl ih =>
      intro n hn
      cases he : eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | none =>
          simp [eraseTraceAux, he] at hn
          have hlt := ih (i + 1) n hn
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hlt
      | some hd' =>
          simp [eraseTraceAux, he] at hn
          rcases hn with rfl | hn
          · simp
          · have hlt := ih (i + 1) n hn
            simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hlt

private theorem eraseTraceAux_mem_map_snd_ge {A : L} (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    ∀ n ∈ (eraseTraceAux (C := C) (F := F) (Payload := Payload) i w).map Prod.snd, i ≤ n := by
  intro n hn
  rcases List.mem_map.mp hn with ⟨p, hp, hpEq⟩
  subst hpEq
  exact eraseTraceAux_snd_ge (C := C) (F := F) (Payload := Payload) i w p hp

private theorem eraseTraceAux_pairwise_snd {A : L} (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    List.Pairwise (· < ·)
      ((eraseTraceAux (C := C) (F := F) (Payload := Payload) i w).map Prod.snd) := by
  induction w generalizing i with
  | nil =>
      simp [eraseTraceAux]
  | cons hd tl ih =>
      simp [eraseTraceAux]
      split
      · refine List.pairwise_cons.2 ?_
        refine ⟨?_, ?_⟩
        · intro n hn
          exact Nat.lt_of_lt_of_le (Nat.lt_succ_self i)
            (eraseTraceAux_mem_map_snd_ge (C := C) (F := F) (Payload := Payload) (i + 1) tl n
              (by simpa using hn))
        · simpa using ih (i + 1)
      · simpa using ih (i + 1)

private theorem eraseTrace_pairwise_snd {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    List.Pairwise (· < ·)
      ((eraseTrace (C := C) (F := F) (Payload := Payload) w).map Prod.snd) := by
  simpa [eraseTrace] using eraseTraceAux_pairwise_snd (C := C) (F := F) (Payload := Payload) 0 w

/-- Original position of the `j`-th surviving letter in `eraseWord w`. -/
def eraseOriginPos {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat) (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) : Nat :=
  (eraseTrace (C := C) (F := F) (Payload := Payload) w).get
    ⟨j, by simpa [eraseTrace_length (C := C) (F := F) (Payload := Payload) w] using hj⟩ |>.2

/-- Recursive origin position of the `j`-th surviving letter. This avoids later
    proof obligations about `eraseTrace` indexing. -/
def eraseOriginRec {A : L} :
    LocalWord (C := C) (F := F) (Payload := Payload) A → Nat → Nat
  | [], _ => 0
  | hd :: tl, j =>
      match eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | some _ =>
          match j with
          | 0 => 0
          | j + 1 => eraseOriginRec tl j + 1
      | none => eraseOriginRec tl j + 1

private theorem eraseLetter_some_eq_self {A : L}
    {hd hd' : AlphabetOf (C := C) (F := F) (Payload := Payload) A}
    (h : eraseLetter (C := C) (F := F) (Payload := Payload) hd = some hd') :
    hd' = hd := by
  cases hd with
  | mk hd hown =>
      cases hd with
      | sendLetter owner xs target hneq =>
          simp [eraseLetter] at h
          exact h.2.symm
      | recvLetter owner ys source =>
          simp [eraseLetter] at h
          exact h.2.symm
      | actLetter owner ys f xs =>
          simpa [eraseLetter] using h.symm
      | ifTrueLetter cond owner =>
          simpa [eraseLetter] using h.symm
      | ifFalseLetter cond owner =>
          simpa [eraseLetter] using h.symm
      | whileTrueLetter cond owner =>
          simpa [eraseLetter] using h.symm
      | whileFalseLetter cond owner =>
          simpa [eraseLetter] using h.symm

private theorem eraseOriginRec_lt_length {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat)
    (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) :
    eraseOriginRec (C := C) (F := F) (Payload := Payload) w j < w.length := by
  induction w generalizing j with
  | nil =>
      simp [eraseWord] at hj
  | cons hd tl ih =>
      cases he : eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | none =>
          simp [eraseWord, eraseOriginRec, he] at hj ⊢
          exact ih j hj
      | some hd' =>
          cases j with
          | zero =>
              simp [eraseWord, eraseOriginRec, he]
          | succ j =>
              simp [eraseWord, eraseOriginRec, he] at hj ⊢
              exact ih j hj

private theorem eraseOriginRec_eraseLetter_isSome {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat)
    (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) :
    (eraseLetter (C := C) (F := F) (Payload := Payload)
      (w[eraseOriginRec (C := C) (F := F) (Payload := Payload) w j]'(eraseOriginRec_lt_length
        (C := C) (F := F) (Payload := Payload) w j hj))).isSome = true := by
  induction w generalizing j with
  | nil =>
      simp [eraseWord] at hj
  | cons hd tl ih =>
      cases he : eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | none =>
          simp [eraseWord, eraseOriginRec, he] at hj ⊢
          simpa [List.getElem_cons_succ] using ih j hj
      | some hd' =>
          cases j with
          | zero =>
              have hself : hd' = hd := eraseLetter_some_eq_self (C := C) (F := F) (Payload := Payload) he
              subst hself
              simp [eraseWord, eraseOriginRec, he]
          | succ j =>
              simp [eraseWord, eraseOriginRec, he] at hj ⊢
              simpa [List.getElem_cons_succ] using ih j hj

private theorem eraseWord_get_originRec {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat)
    (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) :
    (eraseWord (C := C) (F := F) (Payload := Payload) w)[j] =
      w[eraseOriginRec (C := C) (F := F) (Payload := Payload) w j]'(eraseOriginRec_lt_length
        (C := C) (F := F) (Payload := Payload) w j hj) := by
  induction w generalizing j with
  | nil =>
      simp [eraseWord] at hj
  | cons hd tl ih =>
      cases he : eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | none =>
          simp [eraseWord, eraseOriginRec, he] at hj ⊢
          simpa [List.getElem_cons_succ] using ih j hj
      | some hd' =>
          have hself : hd' = hd := eraseLetter_some_eq_self (C := C) (F := F) (Payload := Payload) he
          subst hself
          cases j with
          | zero =>
              simp [eraseWord, eraseOriginRec, he]
          | succ j =>
              simp [eraseWord, eraseOriginRec, he] at hj ⊢
              simpa [List.getElem_cons_succ] using ih j hj

private theorem eraseOriginRec_strictMono {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    {j1 j2 : Nat}
    (hj1 : j1 < (eraseWord (C := C) (F := F) (Payload := Payload) w).length)
    (hj2 : j2 < (eraseWord (C := C) (F := F) (Payload := Payload) w).length)
    (hLt : j1 < j2) :
    eraseOriginRec (C := C) (F := F) (Payload := Payload) w j1 <
      eraseOriginRec (C := C) (F := F) (Payload := Payload) w j2 := by
  induction w generalizing j1 j2 with
  | nil =>
      simp [eraseWord] at hj1
  | cons hd tl ih =>
      cases he : eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | none =>
          simp [eraseWord, eraseOriginRec, he] at hj1 hj2 ⊢
          exact ih hj1 hj2 hLt
      | some hd' =>
          cases j1 with
          | zero =>
              cases j2 with
              | zero =>
                  cases Nat.lt_irrefl 0 hLt
              | succ j2 =>
                  simp [eraseWord, eraseOriginRec, he] at hj1 hj2 ⊢
          | succ j1 =>
              cases j2 with
              | zero =>
                  cases Nat.not_lt_of_ge (Nat.zero_le _) hLt
              | succ j2 =>
                  simp [eraseWord, eraseOriginRec, he] at hj1 hj2 ⊢
                  exact ih hj1 hj2 (Nat.lt_of_succ_lt_succ hLt)

private theorem eraseWord_take_originRec_succ {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat)
    (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) :
    eraseWord (C := C) (F := F) (Payload := Payload)
        (w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j + 1)) =
      (eraseWord (C := C) (F := F) (Payload := Payload) w).take (j + 1) := by
  induction w generalizing j with
  | nil =>
      simp [eraseWord] at hj
  | cons hd tl ih =>
      cases he : eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | none =>
          simp [eraseWord, eraseOriginRec, he] at hj ⊢
          simpa [eraseWord, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using ih j hj
      | some hd' =>
          have hself : hd' = hd := eraseLetter_some_eq_self (C := C) (F := F) (Payload := Payload) he
          subst hself
          cases j with
          | zero =>
              simp [eraseWord, eraseOriginRec, he]
          | succ j =>
              simp [eraseWord, eraseOriginRec, he] at hj ⊢
              simpa [eraseWord, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using ih j hj

private theorem eraseWord_take_originRec {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat)
    (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) :
    eraseWord (C := C) (F := F) (Payload := Payload)
        (w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j)) =
      (eraseWord (C := C) (F := F) (Payload := Payload) w).take j := by
  induction w generalizing j with
  | nil =>
      simp [eraseWord] at hj
  | cons hd tl ih =>
      cases he : eraseLetter (C := C) (F := F) (Payload := Payload) hd with
      | none =>
          simp [eraseWord, eraseOriginRec, he] at hj ⊢
          simpa [eraseWord] using ih j hj
      | some hd' =>
          have hself : hd' = hd := eraseLetter_some_eq_self (C := C) (F := F) (Payload := Payload) he
          subst hself
          cases j with
          | zero =>
              simp [eraseWord, eraseOriginRec, he]
          | succ j =>
              simp [eraseWord, eraseOriginRec, he] at hj ⊢
              simpa [eraseWord] using ih j hj

/-- Channel-specific send payloads tagged with their original local positions. -/
def sendTraceAux {A : L} (B : L) :
    Nat →
    LocalWord (C := C) (F := F) (Payload := Payload) A →
    List (Payload × Nat)
  | _, [] => []
  | i, hd :: tl =>
      let rest := sendTraceAux B (i + 1) tl
      match hd.val with
      | .sendLetter _ xs tgt _ =>
          if tgt = B then (xs, i) :: rest else rest
      | _ => rest

def sendTrace {A : L} (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    List (Payload × Nat) :=
  sendTraceAux (C := C) (F := F) (Payload := Payload) B 0 w

/-- Channel-specific receive payloads tagged with their original local positions. -/
def recvTraceAux {B : L} (A : L) :
    Nat →
    LocalWord (C := C) (F := F) (Payload := Payload) B →
    List (Payload × Nat)
  | _, [] => []
  | i, hd :: tl =>
      let rest := recvTraceAux A (i + 1) tl
      match hd.val with
      | .recvLetter _ ys src =>
          if src = A then (ys, i) :: rest else rest
      | _ => rest

def recvTrace {B : L} (A : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B) :
    List (Payload × Nat) :=
  recvTraceAux (C := C) (F := F) (Payload := Payload) A 0 w

@[simp]
private theorem sendTraceAux_append {A : L} (B : L) (i : Nat)
    (w1 w2 : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    sendTraceAux (C := C) (F := F) (Payload := Payload) B i (w1 ++ w2) =
      sendTraceAux (C := C) (F := F) (Payload := Payload) B i w1 ++
        sendTraceAux (C := C) (F := F) (Payload := Payload) B (i + w1.length) w2 := by
  induction w1 generalizing i with
  | nil =>
      simp [sendTraceAux]
  | cons hd tl ih =>
      cases hℓ : hd.val with
      | sendLetter owner xs tgt hneq =>
          by_cases htarget : tgt = B
          · simp [sendTraceAux, hℓ, htarget, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
          · simp [sendTraceAux, hℓ, htarget, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | recvLetter owner ys src =>
          simpa [sendTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | actLetter owner ys f xs =>
          simpa [sendTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | ifTrueLetter c owner =>
          simpa [sendTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | ifFalseLetter c owner =>
          simpa [sendTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | whileTrueLetter c owner =>
          simpa [sendTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | whileFalseLetter c owner =>
          simpa [sendTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

@[simp]
private theorem recvTraceAux_append {B : L} (A : L) (i : Nat)
    (w1 w2 : LocalWord (C := C) (F := F) (Payload := Payload) B) :
    recvTraceAux (C := C) (F := F) (Payload := Payload) A i (w1 ++ w2) =
      recvTraceAux (C := C) (F := F) (Payload := Payload) A i w1 ++
        recvTraceAux (C := C) (F := F) (Payload := Payload) A (i + w1.length) w2 := by
  induction w1 generalizing i with
  | nil =>
      simp [recvTraceAux]
  | cons hd tl ih =>
      cases hℓ : hd.val with
      | recvLetter owner ys src =>
          by_cases hsrc : src = A
          · simp [recvTraceAux, hℓ, hsrc, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
          · simp [recvTraceAux, hℓ, hsrc, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | sendLetter owner xs tgt hneq =>
          simpa [recvTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | actLetter owner ys f xs =>
          simpa [recvTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | ifTrueLetter c owner =>
          simpa [recvTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | ifFalseLetter c owner =>
          simpa [recvTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | whileTrueLetter c owner =>
          simpa [recvTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      | whileFalseLetter c owner =>
          simpa [recvTraceAux, hℓ, ih, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

@[simp]
private theorem sendTraceAux_map_fst {A : L} (B : L) (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    (sendTraceAux (C := C) (F := F) (Payload := Payload) B i w).map Prod.fst =
      sendPayloads (C := C) (F := F) (Payload := Payload) B w := by
  induction w generalizing i with
  | nil =>
      simp [sendTraceAux, sendPayloads]
  | cons hd tl ih =>
      cases hℓ : hd.val <;> simp [sendTraceAux, sendPayloads, ih, hℓ]
      split <;> simp [ih]

@[simp]
private theorem sendTrace_map_fst {A : L} (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    (sendTrace (C := C) (F := F) (Payload := Payload) B w).map Prod.fst =
      sendPayloads (C := C) (F := F) (Payload := Payload) B w := by
  simpa [sendTrace] using sendTraceAux_map_fst (C := C) (F := F) (Payload := Payload) B 0 w

@[simp]
private theorem recvTraceAux_map_fst {B : L} (A : L) (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B) :
    (recvTraceAux (C := C) (F := F) (Payload := Payload) A i w).map Prod.fst =
      recvPayloads (C := C) (F := F) (Payload := Payload) A w := by
  induction w generalizing i with
  | nil =>
      simp [recvTraceAux, recvPayloads]
  | cons hd tl ih =>
      cases hℓ : hd.val <;> simp [recvTraceAux, recvPayloads, ih, hℓ]
      split <;> simp [ih]

@[simp]
private theorem recvTrace_map_fst {B : L} (A : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B) :
    (recvTrace (C := C) (F := F) (Payload := Payload) A w).map Prod.fst =
      recvPayloads (C := C) (F := F) (Payload := Payload) A w := by
  simpa [recvTrace] using recvTraceAux_map_fst (C := C) (F := F) (Payload := Payload) A 0 w

@[simp]
private theorem sendTraceAux_length {A : L} (B : L) (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    (sendTraceAux (C := C) (F := F) (Payload := Payload) B i w).length =
      countSends (C := C) (F := F) (Payload := Payload) B w := by
  induction w generalizing i with
  | nil =>
      simp [sendTrace, sendTraceAux, countSends]
  | cons hd tl ih =>
      cases hℓ : hd.val with
      | sendLetter owner xs target hneq =>
          by_cases htarget : target = B
          · have hlen := ih (i + 1)
            simp [sendTrace, sendTraceAux, countSends, hℓ, Letter.isSendTo, htarget]
            omega
          · simpa [sendTrace, sendTraceAux, countSends, hℓ, Letter.isSendTo, htarget] using
              ih (i + 1)
      | recvLetter owner ys source =>
          simpa [sendTrace, sendTraceAux, countSends, hℓ, Letter.isSendTo] using ih (i + 1)
      | actLetter owner ys f xs =>
          simpa [sendTrace, sendTraceAux, countSends, hℓ, Letter.isSendTo] using ih (i + 1)
      | ifTrueLetter c owner =>
          simpa [sendTrace, sendTraceAux, countSends, hℓ, Letter.isSendTo] using ih (i + 1)
      | ifFalseLetter c owner =>
          simpa [sendTrace, sendTraceAux, countSends, hℓ, Letter.isSendTo] using ih (i + 1)
      | whileTrueLetter c owner =>
          simpa [sendTrace, sendTraceAux, countSends, hℓ, Letter.isSendTo] using ih (i + 1)
      | whileFalseLetter c owner =>
          simpa [sendTrace, sendTraceAux, countSends, hℓ, Letter.isSendTo] using ih (i + 1)

@[simp]
private theorem sendTrace_length {A : L} (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    (sendTrace (C := C) (F := F) (Payload := Payload) B w).length =
      countSends (C := C) (F := F) (Payload := Payload) B w := by
  simpa [sendTrace] using sendTraceAux_length (C := C) (F := F) (Payload := Payload) B 0 w

@[simp]
private theorem recvTraceAux_length {B : L} (A : L) (i : Nat)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B) :
    (recvTraceAux (C := C) (F := F) (Payload := Payload) A i w).length =
      countRecvs (C := C) (F := F) (Payload := Payload) A w := by
  induction w generalizing i with
  | nil =>
      simp [recvTrace, recvTraceAux, countRecvs]
  | cons hd tl ih =>
      cases hℓ : hd.val with
      | recvLetter owner ys source =>
          by_cases hsource : source = A
          · have hlen := ih (i + 1)
            simp [recvTrace, recvTraceAux, countRecvs, hℓ, Letter.isRecvFrom, hsource]
            omega
          · simpa [recvTrace, recvTraceAux, countRecvs, hℓ, Letter.isRecvFrom, hsource] using
              ih (i + 1)
      | sendLetter owner xs target hneq =>
          simpa [recvTrace, recvTraceAux, countRecvs, hℓ, Letter.isRecvFrom] using ih (i + 1)
      | actLetter owner ys f xs =>
          simpa [recvTrace, recvTraceAux, countRecvs, hℓ, Letter.isRecvFrom] using ih (i + 1)
      | ifTrueLetter c owner =>
          simpa [recvTrace, recvTraceAux, countRecvs, hℓ, Letter.isRecvFrom] using ih (i + 1)
      | ifFalseLetter c owner =>
          simpa [recvTrace, recvTraceAux, countRecvs, hℓ, Letter.isRecvFrom] using ih (i + 1)
      | whileTrueLetter c owner =>
          simpa [recvTrace, recvTraceAux, countRecvs, hℓ, Letter.isRecvFrom] using ih (i + 1)
      | whileFalseLetter c owner =>
          simpa [recvTrace, recvTraceAux, countRecvs, hℓ, Letter.isRecvFrom] using ih (i + 1)

@[simp]
private theorem recvTrace_length {B : L} (A : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B) :
    (recvTrace (C := C) (F := F) (Payload := Payload) A w).length =
      countRecvs (C := C) (F := F) (Payload := Payload) A w := by
  simpa [recvTrace] using recvTraceAux_length (C := C) (F := F) (Payload := Payload) A 0 w

private theorem countSends_take_succ_of_sendAt {A : L} (B : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) A))
    (j : Nat) (hj : j < w.length)
    (hSend : (w[j]).val.isSendTo B = true) :
    countSends (C := C) (F := F) (Payload := Payload) B (w.take (j + 1)) =
      countSends (C := C) (F := F) (Payload := Payload) B (w.take j) + 1 := by
  rw [List.take_succ_eq_append_getElem hj, countSends_append]
  simp [countSends, hSend]

private theorem countRecvs_take_succ_of_recvAt {B : L} (A : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B))
    (j : Nat) (hj : j < w.length)
    (hRecv : (w[j]).val.isRecvFrom A = true) :
    countRecvs (C := C) (F := F) (Payload := Payload) A (w.take (j + 1)) =
      countRecvs (C := C) (F := F) (Payload := Payload) A (w.take j) + 1 := by
  rw [List.take_succ_eq_append_getElem hj, countRecvs_append]
  simp [countRecvs, hRecv]

private theorem countSends_take_lt_of_sendAt {A : L} (B : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) A))
    (j : Nat) (hj : j < w.length)
    (hSend : (w[j]).val.isSendTo B = true) :
    countSends (C := C) (F := F) (Payload := Payload) B (w.take j) + 1 ≤
      countSends (C := C) (F := F) (Payload := Payload) B w := by
  have hsplit : w = w.take j ++ [w[j]] ++ w.drop (j + 1) := by
    have hsplit0 : w = w.take j ++ w.drop j := (List.take_append_drop j w).symm
    rw [List.drop_eq_getElem_cons hj] at hsplit0
    simpa [List.append_assoc] using hsplit0
  rw [hsplit, countSends_append, countSends_append]
  simp [countSends, hSend]

private theorem countRecvs_take_lt_of_recvAt {B : L} (A : L)
    (w : List (AlphabetOf (C := C) (F := F) (Payload := Payload) B))
    (j : Nat) (hj : j < w.length)
    (hRecv : (w[j]).val.isRecvFrom A = true) :
    countRecvs (C := C) (F := F) (Payload := Payload) A (w.take j) + 1 ≤
      countRecvs (C := C) (F := F) (Payload := Payload) A w := by
  have hsplit : w = w.take j ++ [w[j]] ++ w.drop (j + 1) := by
    have hsplit0 : w = w.take j ++ w.drop j := (List.take_append_drop j w).symm
    rw [List.drop_eq_getElem_cons hj] at hsplit0
    simpa [List.append_assoc] using hsplit0
  rw [hsplit, countRecvs_append, countRecvs_append]
  simp [countRecvs, hRecv]

private theorem countSends_erase_prefix_origin {A : L} (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat)
    (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) :
    countSends (C := C) (F := F) (Payload := Payload) B
        (eraseWord (C := C) (F := F) (Payload := Payload)
          (w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j))) =
      countSends (C := C) (F := F) (Payload := Payload) B
        ((eraseWord (C := C) (F := F) (Payload := Payload) w).take j) := by
  simpa [eraseWord_take_originRec (C := C) (F := F) (Payload := Payload) w j hj]

private theorem countRecvs_erase_prefix_origin {B : L} (A : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (j : Nat)
    (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) :
    countRecvs (C := C) (F := F) (Payload := Payload) A
        (eraseWord (C := C) (F := F) (Payload := Payload)
          (w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j))) =
      countRecvs (C := C) (F := F) (Payload := Payload) A
        ((eraseWord (C := C) (F := F) (Payload := Payload) w).take j) := by
  simpa [eraseWord_take_originRec (C := C) (F := F) (Payload := Payload) w j hj]

private theorem mem_zip_take_take {α β : Type} {xs : List α} {ys : List β}
    {i j : Nat} {p : α × β}
    (hp : p ∈ List.zip (xs.take i) (ys.take j)) :
    p ∈ List.zip xs ys := by
  obtain ⟨k, hkx, hky, hx, hy⟩ := List.mem_zip_get hp
  have hkx' : k < xs.length := Nat.lt_of_lt_of_le hkx (List.length_take_le' i xs)
  have hky' : k < ys.length := Nat.lt_of_lt_of_le hky (List.length_take_le' j ys)
  have hzip : k < (List.zip xs ys).length := by
    simp [List.length_zip]
    exact Nat.lt_min.mpr ⟨hkx', hky'⟩
  have hx' : xs[k] = p.1 := by
    simpa using (List.getElem_take (xs := xs) (j := i) (i := k) (h := hkx)).symm.trans hx
  have hy' : ys[k] = p.2 := by
    simpa using (List.getElem_take (xs := ys) (j := j) (i := k) (h := hky)).symm.trans hy
  have hmem : (List.zip xs ys)[k] ∈ List.zip xs ys := List.getElem_mem hzip
  simpa [List.getElem_zip, hx', hy'] using hmem



theorem eraseWord_sublist {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    List.Sublist (eraseWord (C := C) (F := F) (Payload := Payload) w) w := by
  induction w with
  | nil =>
      simp [eraseWord]
  | cons hd tl ih =>
      cases hd with
      | mk hd hown =>
          cases hd with
          | sendLetter owner xs tgt hneq =>
              by_cases hctrl : isControlPayload xs = true
              · simpa [eraseWord, eraseLetter, hctrl] using
                  (List.Sublist.cons ⟨Letter.sendLetter owner xs tgt hneq, hown⟩ ih)
              · simpa [eraseWord, eraseLetter, hctrl] using
                  (List.Sublist.cons₂ ⟨Letter.sendLetter owner xs tgt hneq, hown⟩ ih)
          | recvLetter owner ys src =>
              by_cases hctrl : isControlPayload ys = true
              · simpa [eraseWord, eraseLetter, hctrl] using
                  (List.Sublist.cons ⟨Letter.recvLetter owner ys src, hown⟩ ih)
              · simpa [eraseWord, eraseLetter, hctrl] using
                  (List.Sublist.cons₂ ⟨Letter.recvLetter owner ys src, hown⟩ ih)
          | actLetter owner ys f xs =>
              simpa [eraseWord, eraseLetter] using
                (List.Sublist.cons₂ ⟨Letter.actLetter owner ys f xs, hown⟩ ih)
          | ifTrueLetter c owner =>
              simpa [eraseWord, eraseLetter] using
                (List.Sublist.cons₂ ⟨Letter.ifTrueLetter c owner, hown⟩ ih)
          | ifFalseLetter c owner =>
              simpa [eraseWord, eraseLetter] using
                (List.Sublist.cons₂ ⟨Letter.ifFalseLetter c owner, hown⟩ ih)
          | whileTrueLetter c owner =>
              simpa [eraseWord, eraseLetter] using
                (List.Sublist.cons₂ ⟨Letter.whileTrueLetter c owner, hown⟩ ih)
          | whileFalseLetter c owner =>
              simpa [eraseWord, eraseLetter] using
                (List.Sublist.cons₂ ⟨Letter.whileFalseLetter c owner, hown⟩ ih)

theorem eraseWord_indices {A : L}
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    ∃ is : List (Fin w.length),
      eraseWord (C := C) (F := F) (Payload := Payload) w = is.map (w[·]) ∧
      is.Pairwise (· < ·) :=
  List.sublist_eq_map_getElem (eraseWord_sublist (C := C) (F := F) (Payload := Payload) w)

@[simp]
theorem sendPayloads_eraseWord {A : L} (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A) :
    sendPayloads (C := C) (F := F) (Payload := Payload) B
        (eraseWord (C := C) (F := F) (Payload := Payload) w) =
      (sendPayloads (C := C) (F := F) (Payload := Payload) B w).filter keepUserPayload := by
  induction w with
  | nil =>
      simp [eraseWord, sendPayloads, keepUserPayload]
  | cons hd tl ih =>
      cases hd with
      | mk hd hown =>
          cases hd with
          | sendLetter owner xs tgt hneq =>
              by_cases hctrl : isControlPayload xs = true
              · by_cases htgt : tgt = B
                · simp [eraseWord, eraseLetter, sendPayloads, hctrl, htgt, ih, keepUserPayload]
                · simp [eraseWord, eraseLetter, sendPayloads, hctrl, htgt, ih, keepUserPayload]
              · by_cases htgt : tgt = B
                · simp [eraseWord, eraseLetter, sendPayloads, hctrl, htgt, ih, keepUserPayload]
                · simp [eraseWord, eraseLetter, sendPayloads, hctrl, htgt, ih, keepUserPayload]
          | recvLetter owner ys src =>
              by_cases hctrl : isControlPayload ys = true
              · simp [eraseWord, eraseLetter, sendPayloads, hctrl, ih, keepUserPayload]
              · simp [eraseWord, eraseLetter, sendPayloads, hctrl, ih, keepUserPayload]
          | actLetter owner ys f xs =>
              simp [eraseWord, eraseLetter, sendPayloads, ih, keepUserPayload]
          | ifTrueLetter c owner =>
              simp [eraseWord, eraseLetter, sendPayloads, ih, keepUserPayload]
          | ifFalseLetter c owner =>
              simp [eraseWord, eraseLetter, sendPayloads, ih, keepUserPayload]
          | whileTrueLetter c owner =>
              simp [eraseWord, eraseLetter, sendPayloads, ih, keepUserPayload]
          | whileFalseLetter c owner =>
              simp [eraseWord, eraseLetter, sendPayloads, ih, keepUserPayload]

@[simp]
theorem recvPayloads_eraseWord {B : L} (A : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B) :
    recvPayloads (C := C) (F := F) (Payload := Payload) A
        (eraseWord (C := C) (F := F) (Payload := Payload) w) =
      (recvPayloads (C := C) (F := F) (Payload := Payload) A w).filter keepUserPayload := by
  induction w with
  | nil =>
      simp [eraseWord, recvPayloads, keepUserPayload]
  | cons hd tl ih =>
      cases hd with
      | mk hd hown =>
          cases hd with
          | recvLetter owner ys src =>
              by_cases hctrl : isControlPayload ys = true
              · by_cases hsrc : src = A
                · simp [eraseWord, eraseLetter, recvPayloads, hctrl, hsrc, ih, keepUserPayload]
                · simp [eraseWord, eraseLetter, recvPayloads, hctrl, hsrc, ih, keepUserPayload]
              · by_cases hsrc : src = A
                · simp [eraseWord, eraseLetter, recvPayloads, hctrl, hsrc, ih, keepUserPayload]
                · simp [eraseWord, eraseLetter, recvPayloads, hctrl, hsrc, ih, keepUserPayload]
          | sendLetter owner xs tgt hneq =>
              by_cases hctrl : isControlPayload xs = true
              · simp [eraseWord, eraseLetter, recvPayloads, hctrl, ih, keepUserPayload]
              · simp [eraseWord, eraseLetter, recvPayloads, hctrl, ih, keepUserPayload]
          | actLetter owner ys f xs =>
              simp [eraseWord, eraseLetter, recvPayloads, ih, keepUserPayload]
          | ifTrueLetter c owner =>
              simp [eraseWord, eraseLetter, recvPayloads, ih, keepUserPayload]
          | ifFalseLetter c owner =>
              simp [eraseWord, eraseLetter, recvPayloads, ih, keepUserPayload]
          | whileTrueLetter c owner =>
              simp [eraseWord, eraseLetter, recvPayloads, ih, keepUserPayload]
          | whileFalseLetter c owner =>
              simp [eraseWord, eraseLetter, recvPayloads, ih, keepUserPayload]

private theorem countSends_erase_prefix_origin_filter {A : L} (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat)
    (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) :
    countSends (C := C) (F := F) (Payload := Payload) B
        ((eraseWord (C := C) (F := F) (Payload := Payload) w).take j) =
      ((sendPayloads (C := C) (F := F) (Payload := Payload) B
          (w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j))).filter
        keepUserPayload).length := by
  rw [← countSends_erase_prefix_origin (C := C) (F := F) (Payload := Payload) B w j hj]
  rw [← sendPayloads_length (C := C) (F := F) (Payload := Payload) (B := B)
    (w := eraseWord (C := C) (F := F) (Payload := Payload)
      (w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j)))]
  simpa using congrArg List.length
    (sendPayloads_eraseWord (C := C) (F := F) (Payload := Payload) (B := B)
      (w := w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j)))

private theorem countRecvs_erase_prefix_origin_filter {B : L} (A : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (j : Nat)
    (hj : j < (eraseWord (C := C) (F := F) (Payload := Payload) w).length) :
    countRecvs (C := C) (F := F) (Payload := Payload) A
        ((eraseWord (C := C) (F := F) (Payload := Payload) w).take j) =
      ((recvPayloads (C := C) (F := F) (Payload := Payload) A
          (w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j))).filter
        keepUserPayload).length := by
  rw [← countRecvs_erase_prefix_origin (C := C) (F := F) (Payload := Payload) A w j hj]
  rw [← recvPayloads_length (C := C) (F := F) (Payload := Payload) (A := A)
    (w := eraseWord (C := C) (F := F) (Payload := Payload)
      (w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j)))]
  simpa using congrArg List.length
    (recvPayloads_eraseWord (C := C) (F := F) (Payload := Payload) (A := A)
      (w := w.take (eraseOriginRec (C := C) (F := F) (Payload := Payload) w j)))

@[simp]
theorem eraseTuple_concat (M1 M2 : WordTuple L C F Payload) :
    erase (M1 ∘ₘ M2) = erase M1 ∘ₘ erase M2 := by
  funext A
  simp [eraseTuple, WordTuple.concat, eraseWord_append]

@[simp]
theorem eraseTuple_concatList (Ms : List (WordTuple L C F Payload)) :
    erase (WordTuple.concatList Ms) = WordTuple.concatList (Ms.map erase) := by
  induction Ms with
  | nil =>
      funext A
      simp [eraseTuple, WordTuple.concatList, WordTuple.empty]
  | cons hd tl ih =>
      simp [WordTuple.concatList_cons, eraseTuple_concat, ih]

private theorem zip_filter_eq_of_flags {α β : Type}
    (keepα : α → Bool) (keepβ : β → Bool) :
    ∀ xs : List α, ∀ ys : List β,
      (∀ p ∈ List.zip xs ys, keepα p.1 = keepβ p.2) →
      List.zip (xs.filter keepα) (ys.filter keepβ) =
        (List.zip xs ys).filter (fun p => keepα p.1)
  | [], _, _ => by simp
  | _ :: _, [], _ => by simp
  | x :: xs, y :: ys, hxy => by
      have hhead : keepα x = keepβ y := by
        exact hxy (x, y) (by simp)
      have htail :
          ∀ p ∈ List.zip xs ys, keepα p.1 = keepβ p.2 := by
        intro p hp
        exact hxy p (by simp [hp])
      by_cases hx : keepα x
      · have hy : keepβ y := by simpa [hhead] using hx
        simp [hx, hy, zip_filter_eq_of_flags keepα keepβ xs ys htail]
      · have hy : ¬ keepβ y := by simpa [hhead] using hx
        simp [hx, hy, zip_filter_eq_of_flags keepα keepβ xs ys htail]

private theorem zip_filter_length_right_of_flags {α β : Type}
    (keepα : α → Bool) (keepβ : β → Bool) :
    ∀ xs : List α, ∀ ys : List β,
      ys.length ≤ xs.length →
      (∀ p ∈ List.zip xs ys, keepα p.1 = keepβ p.2) →
      ((List.zip xs ys).filter (fun p => keepα p.1)).length = (ys.filter keepβ).length
  | [], [], _, _ => by simp
  | [], _ :: _, hlen, _ => by cases hlen
  | _ :: _, [], _, _ => by simp
  | x :: xs, y :: ys, hlen, hxy => by
      have hhead : keepα x = keepβ y := by
        exact hxy (x, y) (by simp)
      have htail :
          ∀ p ∈ List.zip xs ys, keepα p.1 = keepβ p.2 := by
        intro p hp
        exact hxy p (by simp [hp])
      have hlen' : ys.length ≤ xs.length := Nat.le_of_succ_le_succ hlen
      by_cases hx : keepα x
      · have hy : keepβ y := by simpa [hhead] using hx
        simp [hx, hy, zip_filter_length_right_of_flags keepα keepβ xs ys hlen' htail]
      · have hy : ¬ keepβ y := by simpa [hhead] using hx
        simp [hx, hy, zip_filter_length_right_of_flags keepα keepβ xs ys hlen' htail]

theorem noUnmatchedReceives_erase
    (M : WordTuple L C F Payload)
    (hM : IsMSC (L := L) (C := C) (F := F) (Payload := Payload) M) :
    noUnmatchedReceives (L := L) (C := C) (F := F) (Payload := Payload) (erase M) := by
  intro A B
  let ss := sendPayloads (C := C) (F := F) (Payload := Payload) B (M A)
  let rs := recvPayloads (C := C) (F := F) (Payload := Payload) A (M B)
  have hlen : rs.length ≤ ss.length := by
    simpa [ss, rs, sndCount, rcvCount] using hM.noUnmatchedRecv A B
  have hflags :
      ∀ p ∈ List.zip ss rs, keepUserPayload p.1 = keepUserPayload p.2 := by
    intro p hp
    exact congrArg (! ·)
      (ControlPayloadSpec.compat_preserves_isControl
        (hM.labelCompat A B p (by simpa [ss, rs] using hp)))
  have hzipEq :
      List.zip (ss.filter keepUserPayload) (rs.filter keepUserPayload) =
        (List.zip ss rs).filter (fun p => keepUserPayload p.1) :=
    zip_filter_eq_of_flags keepUserPayload keepUserPayload ss rs hflags
  have hzipRight :
      ((List.zip ss rs).filter (fun p => keepUserPayload p.1)).length =
        (rs.filter keepUserPayload).length :=
    zip_filter_length_right_of_flags keepUserPayload keepUserPayload ss rs hlen hflags
  have hkeep :
      (rs.filter keepUserPayload).length ≤ (ss.filter keepUserPayload).length := by
    calc
      (rs.filter keepUserPayload).length
          = ((List.zip ss rs).filter (fun p => keepUserPayload p.1)).length := by
              symm
              exact hzipRight
      _ = (List.zip (ss.filter keepUserPayload) (rs.filter keepUserPayload)).length := by
            symm
            exact congrArg List.length hzipEq
      _ = Nat.min (ss.filter keepUserPayload).length (rs.filter keepUserPayload).length := by
            simp
      _ ≤ (ss.filter keepUserPayload).length := by
            exact Nat.min_le_left _ _
  have hcountR :
      countRecvs (C := C) (F := F) (Payload := Payload) A (eraseWord (M B)) =
        (rs.filter keepUserPayload).length := by
    rw [← recvPayloads_length (C := C) (F := F) (Payload := Payload) (A := A)
      (w := eraseWord (M B))]
    simpa [rs] using congrArg List.length
      (recvPayloads_eraseWord (C := C) (F := F) (Payload := Payload) (A := A) (w := M B))
  have hcountS :
      countSends (C := C) (F := F) (Payload := Payload) B (eraseWord (M A)) =
        (ss.filter keepUserPayload).length := by
    rw [← sendPayloads_length (C := C) (F := F) (Payload := Payload) (B := B)
      (w := eraseWord (M A))]
    simpa [ss] using congrArg List.length
      (sendPayloads_eraseWord (C := C) (F := F) (Payload := Payload) (B := B) (w := M A))
  have hcount :
      countRecvs (C := C) (F := F) (Payload := Payload) A (eraseWord (M B)) ≤
        countSends (C := C) (F := F) (Payload := Payload) B (eraseWord (M A)) := by
    rw [hcountR, hcountS]
    exact hkeep
  simpa [eraseTuple, sndCount, rcvCount] using hcount

theorem matchedLabelsCompatible_erase
    (M : WordTuple L C F Payload)
    (hM : IsMSC (L := L) (C := C) (F := F) (Payload := Payload) M) :
    matchedLabelsCompatible (L := L) (C := C) (F := F) (Payload := Payload) (erase M) := by
  intro A B p hp
  let ss := sendPayloads (C := C) (F := F) (Payload := Payload) B (M A)
  let rs := recvPayloads (C := C) (F := F) (Payload := Payload) A (M B)
  have hflags :
      ∀ q ∈ List.zip ss rs, keepUserPayload q.1 = keepUserPayload q.2 := by
    intro q hq
    exact congrArg (! ·)
      (ControlPayloadSpec.compat_preserves_isControl
        (hM.labelCompat A B q (by simpa [ss, rs] using hq)))
  have hzipEq :
      List.zip (ss.filter keepUserPayload) (rs.filter keepUserPayload) =
        (List.zip ss rs).filter (fun q => keepUserPayload q.1) :=
    zip_filter_eq_of_flags keepUserPayload keepUserPayload ss rs hflags
  have hp' : p ∈ (List.zip ss rs).filter (fun q => keepUserPayload q.1) := by
    simpa [eraseTuple, ss, rs, sendPayloads_eraseWord, recvPayloads_eraseWord, hzipEq] using hp
  have hpPair : p ∈ List.zip ss rs ∧ keepUserPayload p.1 = true := by
    simpa using hp'
  exact hM.labelCompat A B p (by simpa [ss, rs] using hpPair.1)

private theorem sendPayloads_take_prefix {A : L} (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (k : Nat) :
    sendPayloads (C := C) (F := F) (Payload := Payload) B (w.take k) =
      (sendPayloads (C := C) (F := F) (Payload := Payload) B w).take
        (countSends (C := C) (F := F) (Payload := Payload) B (w.take k)) := by
  have happ :
      sendPayloads (C := C) (F := F) (Payload := Payload) B w =
        sendPayloads (C := C) (F := F) (Payload := Payload) B (w.take k) ++
          sendPayloads (C := C) (F := F) (Payload := Payload) B (w.drop k) := by
    rw [← sendPayloads_append (C := C) (F := F) (Payload := Payload) (B := B)
      (w1 := w.take k) (w2 := w.drop k), List.take_append_drop]
  rw [happ, List.take_append_of_le_length]
  · rw [← sendPayloads_length (C := C) (F := F) (Payload := Payload) (B := B) (w := w.take k)]
    simpa using (List.take_length
      (l := sendPayloads (C := C) (F := F) (Payload := Payload) B (w.take k))).symm
  · simpa using sendPayloads_length (C := C) (F := F) (Payload := Payload)
      (B := B) (w := w.take k)

private theorem recvPayloads_take_prefix {B : L} (A : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (k : Nat) :
    recvPayloads (C := C) (F := F) (Payload := Payload) A (w.take k) =
      (recvPayloads (C := C) (F := F) (Payload := Payload) A w).take
        (countRecvs (C := C) (F := F) (Payload := Payload) A (w.take k)) := by
  have happ :
      recvPayloads (C := C) (F := F) (Payload := Payload) A w =
        recvPayloads (C := C) (F := F) (Payload := Payload) A (w.take k) ++
          recvPayloads (C := C) (F := F) (Payload := Payload) A (w.drop k) := by
    rw [← recvPayloads_append (C := C) (F := F) (Payload := Payload) (A := A)
      (w1 := w.take k) (w2 := w.drop k), List.take_append_drop]
  rw [happ, List.take_append_of_le_length]
  · rw [← recvPayloads_length (C := C) (F := F) (Payload := Payload) (A := A) (w := w.take k)]
    simpa using (List.take_length
      (l := recvPayloads (C := C) (F := F) (Payload := Payload) A (w.take k))).symm
  · simpa using recvPayloads_length (C := C) (F := F) (Payload := Payload)
      (A := A) (w := w.take k)

private theorem countSends_take_le {A : L} (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat) :
    countSends (C := C) (F := F) (Payload := Payload) B (w.take j) ≤
      countSends (C := C) (F := F) (Payload := Payload) B w := by
  calc
    countSends (C := C) (F := F) (Payload := Payload) B (w.take j) ≤
        countSends (C := C) (F := F) (Payload := Payload) B (w.take j) +
          countSends (C := C) (F := F) (Payload := Payload) B (w.drop j) := by omega
    _ = countSends (C := C) (F := F) (Payload := Payload) B w := by
      rw [← countSends_append (C := C) (F := F) (Payload := Payload) (B := B)
          (w1 := w.take j) (w2 := w.drop j)]
      rw [List.take_append_drop]

private theorem sendPayloads_get_of_sendAt {A : L} (B : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) A)
    (j : Nat) (hj : j < w.length)
    (hSend : (w[j]).val.isSendTo B = true) :
    ∃ xs owner,
      ∃ hneq : owner ≠ B,
      (w[j]).val = Letter.sendLetter owner xs B hneq ∧
      (sendPayloads (C := C) (F := F) (Payload := Payload) B w).get
        ⟨countSends (C := C) (F := F) (Payload := Payload) B (w.take j), by
          rw [sendPayloads_length (C := C) (F := F) (Payload := Payload) (B := B)]
          exact countSends_take_lt_of_sendAt (C := C) (F := F) (Payload := Payload)
            B w j hj hSend⟩ = xs := by
  cases hℓ : (w[j]).val with
  | sendLetter owner xs tgt hneq =>
      have htgt : tgt = B := by
        simpa [hℓ, Letter.isSendTo] using hSend
      have hneqB : owner ≠ B := by
        intro hownerB
        apply hneq
        rw [hownerB, htgt]
      refine ⟨xs, owner, hneqB, ?_, ?_⟩
      · cases htgt
        simpa using hℓ
      let n := countSends (C := C) (F := F) (Payload := Payload) B (w.take j)
      have htake :
          sendPayloads (C := C) (F := F) (Payload := Payload) B (w.take (j + 1)) =
            (sendPayloads (C := C) (F := F) (Payload := Payload) B w).take (n + 1) := by
        rw [sendPayloads_take_prefix (C := C) (F := F) (Payload := Payload) (A := A) B w (j + 1)]
        simp [n, countSends_take_succ_of_sendAt (C := C) (F := F) (Payload := Payload)
          B w j hj hSend]
      have hsplit :
          sendPayloads (C := C) (F := F) (Payload := Payload) B (w.take (j + 1)) =
            sendPayloads (C := C) (F := F) (Payload := Payload) B (w.take j) ++ [xs] := by
        rw [List.take_succ_eq_append_getElem hj, sendPayloads_append]
        simp [sendPayloads, hℓ, htgt]
      have hlen :
          (sendPayloads (C := C) (F := F) (Payload := Payload) B (w.take j)).length = n := by
        simp [n, sendPayloads_length]
      have htakeLen :
          ((sendPayloads (C := C) (F := F) (Payload := Payload) B w).take (n + 1)).length = n + 1 := by
        have hbound :
            n + 1 ≤ (sendPayloads (C := C) (F := F) (Payload := Payload) B w).length := by
          rw [sendPayloads_length]
          exact countSends_take_lt_of_sendAt (C := C) (F := F) (Payload := Payload)
            B w j hj hSend
        rw [List.length_take, Nat.min_eq_left hbound]
      have hnlt :
          n < ((sendPayloads (C := C) (F := F) (Payload := Payload) B w).take (n + 1)).length := by
        simpa [htakeLen]
      have hgetTake :
          (sendPayloads (C := C) (F := F) (Payload := Payload) B (w.take (j + 1))).get
            ⟨n, by rw [hsplit]; simp [hlen]⟩ = xs := by
        simpa [hsplit, hlen]
      have hgetTake' :
          ((sendPayloads (C := C) (F := F) (Payload := Payload) B w).take (n + 1)).get
            ⟨n, hnlt⟩ = xs := by
        simpa [htake] using hgetTake
      exact (List.getElem_take
        (xs := sendPayloads (C := C) (F := F) (Payload := Payload) B w)
        (j := n + 1) (i := n) (h := by omega)).symm.trans hgetTake'
  | recvLetter owner ys src =>
      simp [hℓ, Letter.isSendTo] at hSend
  | actLetter owner ys f xs =>
      simp [hℓ, Letter.isSendTo] at hSend
  | ifTrueLetter c owner =>
      simp [hℓ, Letter.isSendTo] at hSend
  | ifFalseLetter c owner =>
      simp [hℓ, Letter.isSendTo] at hSend
  | whileTrueLetter c owner =>
      simp [hℓ, Letter.isSendTo] at hSend
  | whileFalseLetter c owner =>
      simp [hℓ, Letter.isSendTo] at hSend

private theorem recvPayloads_get_of_recvAt {B : L} (A : L)
    (w : LocalWord (C := C) (F := F) (Payload := Payload) B)
    (j : Nat) (hj : j < w.length)
    (hRecv : (w[j]).val.isRecvFrom A = true) :
    ∃ ys owner,
      (w[j]).val = Letter.recvLetter owner ys A ∧
      (recvPayloads (C := C) (F := F) (Payload := Payload) A w).get
        ⟨countRecvs (C := C) (F := F) (Payload := Payload) A (w.take j), by
          rw [recvPayloads_length (C := C) (F := F) (Payload := Payload) (A := A)]
          exact countRecvs_take_lt_of_recvAt (C := C) (F := F) (Payload := Payload)
            A w j hj hRecv⟩ = ys := by
  cases hℓ : (w[j]).val with
  | recvLetter owner ys src =>
      have hsrc : src = A := by
        simpa [hℓ, Letter.isRecvFrom] using hRecv
      refine ⟨ys, owner, by simpa [hsrc] using hℓ, ?_⟩
      let n := countRecvs (C := C) (F := F) (Payload := Payload) A (w.take j)
      have htake :
          recvPayloads (C := C) (F := F) (Payload := Payload) A (w.take (j + 1)) =
            (recvPayloads (C := C) (F := F) (Payload := Payload) A w).take (n + 1) := by
        rw [recvPayloads_take_prefix (C := C) (F := F) (Payload := Payload) (B := B) A w (j + 1)]
        simp [n, countRecvs_take_succ_of_recvAt (C := C) (F := F) (Payload := Payload)
          A w j hj hRecv]
      have hsplit :
          recvPayloads (C := C) (F := F) (Payload := Payload) A (w.take (j + 1)) =
            recvPayloads (C := C) (F := F) (Payload := Payload) A (w.take j) ++ [ys] := by
        rw [List.take_succ_eq_append_getElem hj, recvPayloads_append]
        simp [recvPayloads, hℓ, hsrc]
      have hlen :
          (recvPayloads (C := C) (F := F) (Payload := Payload) A (w.take j)).length = n := by
        simp [n, recvPayloads_length]
      have htakeLen :
          ((recvPayloads (C := C) (F := F) (Payload := Payload) A w).take (n + 1)).length = n + 1 := by
        have hbound :
            n + 1 ≤ (recvPayloads (C := C) (F := F) (Payload := Payload) A w).length := by
          rw [recvPayloads_length]
          exact countRecvs_take_lt_of_recvAt (C := C) (F := F) (Payload := Payload)
            A w j hj hRecv
        rw [List.length_take, Nat.min_eq_left hbound]
      have hnlt :
          n < ((recvPayloads (C := C) (F := F) (Payload := Payload) A w).take (n + 1)).length := by
        simpa [htakeLen]
      have hgetTake :
          (recvPayloads (C := C) (F := F) (Payload := Payload) A (w.take (j + 1))).get
            ⟨n, by rw [hsplit]; simp [hlen]⟩ = ys := by
        simpa [hsplit, hlen]
      have hgetTake' :
          ((recvPayloads (C := C) (F := F) (Payload := Payload) A w).take (n + 1)).get
            ⟨n, hnlt⟩ = ys := by
        simpa [htake] using hgetTake
      exact (List.getElem_take
        (xs := recvPayloads (C := C) (F := F) (Payload := Payload) A w)
        (j := n + 1) (i := n) (h := by omega)).symm.trans hgetTake'
  | sendLetter owner xs tgt hneq =>
      simp [hℓ, Letter.isRecvFrom] at hRecv
  | actLetter owner ys f xs =>
      simp [hℓ, Letter.isRecvFrom] at hRecv
  | ifTrueLetter c owner =>
      simp [hℓ, Letter.isRecvFrom] at hRecv
  | ifFalseLetter c owner =>
      simp [hℓ, Letter.isRecvFrom] at hRecv
  | whileTrueLetter c owner =>
      simp [hℓ, Letter.isRecvFrom] at hRecv
  | whileFalseLetter c owner =>
      simp [hℓ, Letter.isRecvFrom] at hRecv


/-- Source-level programs whose user payloads never use the reserved control
    payload space. This formalizes the paper's control-message
    distinguishability assumption. -/
def ControlDistinguishableProgram : Prog L C F Payload → Prop
  | .eps => True
  | .msg _ xs _ ys _ =>
      isControlPayload xs = false ∧ isControlPayload ys = false
  | .act _ ys _ xs =>
      isControlPayload xs = false ∧ isControlPayload ys = false
  | .seq P1 P2 =>
      ControlDistinguishableProgram P1 ∧ ControlDistinguishableProgram P2
  | .ite _ _ PTrue PFalse =>
      ControlDistinguishableProgram PTrue ∧ ControlDistinguishableProgram PFalse
  | .whileLoop _ _ PBody PExit =>
      ControlDistinguishableProgram PBody ∧ ControlDistinguishableProgram PExit

@[simp]
theorem eraseTuple_mscEmpty :
    erase (mscEmpty (L := L) (C := C) (F := F) (Payload := Payload)) = mscEmpty := by
  funext A
  simp [eraseTuple, mscEmpty, WordTuple.empty, eraseWord]

@[simp]
theorem eraseTuple_mscAct (A : L) (ys : Payload) (f : F) (xs : Payload) :
    erase (mscAct (C := C) (F := F) A ys f xs) = mscAct A ys f xs := by
  funext X
  by_cases hAX : X = A
  · subst hAX
    simp [eraseTuple, mscAct_owner, eraseWord, eraseLetter, AlphabetOf.mkAct]
  · simp [eraseTuple, mscAct_other, hAX, eraseWord]

@[simp]
theorem eraseTuple_mscIfTrue (c : C) (B : L) :
    erase (mscIfTrue (C := C) (F := F) (Payload := Payload) c B) = mscIfTrue c B := by
  funext X
  by_cases hXB : X = B
  · subst hXB
    simp [eraseTuple, mscIfTrue, choiceIfTrue, mscChoice_owner, eraseWord, eraseLetter,
      AlphabetOf.mkIfTrue]
  · simp [eraseTuple, mscIfTrue, choiceIfTrue, mscChoice_other, hXB, eraseWord]

@[simp]
theorem eraseTuple_mscIfFalse (c : C) (B : L) :
    erase (mscIfFalse (C := C) (F := F) (Payload := Payload) c B) = mscIfFalse c B := by
  funext X
  by_cases hXB : X = B
  · subst hXB
    simp [eraseTuple, mscIfFalse, choiceIfFalse, mscChoice_owner, eraseWord, eraseLetter,
      AlphabetOf.mkIfFalse]
  · simp [eraseTuple, mscIfFalse, choiceIfFalse, mscChoice_other, hXB, eraseWord]

@[simp]
theorem eraseTuple_mscWhileTrue (c : C) (B : L) :
    erase (mscWhileTrue (C := C) (F := F) (Payload := Payload) c B) = mscWhileTrue c B := by
  funext X
  by_cases hXB : X = B
  · subst hXB
    simp [eraseTuple, mscWhileTrue, choiceWhileTrue, mscChoice_owner, eraseWord, eraseLetter,
      AlphabetOf.mkWhileTrue]
  · simp [eraseTuple, mscWhileTrue, choiceWhileTrue, mscChoice_other, hXB, eraseWord]

@[simp]
theorem eraseTuple_mscWhileFalse (c : C) (B : L) :
    erase (mscWhileFalse (C := C) (F := F) (Payload := Payload) c B) = mscWhileFalse c B := by
  funext X
  by_cases hXB : X = B
  · subst hXB
    simp [eraseTuple, mscWhileFalse, choiceWhileFalse, mscChoice_owner, eraseWord, eraseLetter,
      AlphabetOf.mkWhileFalse]
  · simp [eraseTuple, mscWhileFalse, choiceWhileFalse, mscChoice_other, hXB, eraseWord]

@[simp]
theorem eraseTuple_mscMsg (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B)
    (hxs : isControlPayload xs = false)
    (hys : isControlPayload ys = false) :
    erase (mscMsg (C := C) (F := F) A xs B ys h) = mscMsg A xs B ys h := by
  funext X
  by_cases hXA : X = A
  · subst hXA
    simp [eraseTuple, mscMsg_sender, eraseWord, eraseLetter, AlphabetOf.mkSend, hxs]
  · by_cases hXB : X = B
    · subst hXB
      simp [eraseTuple, mscMsg_receiver, eraseWord, eraseLetter, AlphabetOf.mkRecv, hys, h]
    · simp [eraseTuple, mscMsg_other, hXA, hXB, eraseWord]

@[simp]
theorem eraseTuple_mscMsg_control (A : L) (xs : Payload) (B : L) (ys : Payload)
    (h : A ≠ B)
    (hxs : isControlPayload xs = true)
    (hys : isControlPayload ys = true) :
    erase (mscMsg (C := C) (F := F) A xs B ys h) = mscEmpty := by
  funext X
  by_cases hXA : X = A
  · subst hXA
    simp [eraseTuple, mscMsg_sender, eraseWord, eraseLetter, AlphabetOf.mkSend, hxs, mscEmpty,
      WordTuple.empty]
  · by_cases hXB : X = B
    · subst hXB
      simp [eraseTuple, mscMsg_receiver, eraseWord, eraseLetter, AlphabetOf.mkRecv, hys, mscEmpty,
        WordTuple.empty, h]
    · simp [eraseTuple, mscMsg_other, hXA, hXB, mscEmpty, WordTuple.empty, eraseWord]

-- -----------------------------------------------------------------------
-- Helpers for the acyclicity preservation proof
-- -----------------------------------------------------------------------

/-- Filter-count is monotone in the prefix length. -/
private theorem filter_take_length_mono {α : Type} (p : α → Bool)
    (xs : List α) {k1 k2 : Nat} (h : k1 ≤ k2) :
    ((xs.take k1).filter p).length ≤ ((xs.take k2).filter p).length := by
  -- xs.take k1 = (xs.take k2).take k1 when k1 ≤ k2
  have htake : xs.take k1 = (xs.take k2).take k1 := by
    rw [List.take_take, Nat.min_eq_left h]
  rw [htake]
  calc (((xs.take k2).take k1).filter p).length
      ≤ (((xs.take k2).take k1).filter p).length +
          (((xs.take k2).drop k1).filter p).length := Nat.le_add_right _ _
    _ = ((xs.take k2).filter p).length := by
          rw [← List.length_append, ← List.filter_append, List.take_append_drop]

/-- For matched-flag pairs, same-length prefixes have equal user-count. -/
private theorem filter_prefix_count_eq_of_zip_flags
    (ss rs : List Payload)
    (hflags : ∀ p ∈ List.zip ss rs, keepUserPayload p.1 = keepUserPayload p.2) :
    ∀ k, k ≤ ss.length → k ≤ rs.length →
    ((ss.take k).filter keepUserPayload).length =
    ((rs.take k).filter keepUserPayload).length := by
  intro k
  induction k with
  | zero => intros; simp
  | succ k ih =>
    intro hks hkr
    have hk_ss : k < ss.length := Nat.lt_of_succ_le hks
    have hk_rs : k < rs.length := Nat.lt_of_succ_le hkr
    rw [List.take_succ_eq_append_getElem hk_ss,
        List.take_succ_eq_append_getElem hk_rs,
        List.filter_append, List.filter_append, List.length_append, List.length_append]
    have hzip_len : k < (List.zip ss rs).length := by
      simp [List.length_zip]; exact Nat.lt_min.mpr ⟨hk_ss, hk_rs⟩
    have hzip_get : (List.zip ss rs)[k]'hzip_len = (ss[k]'hk_ss, rs[k]'hk_rs) := by
      simp [List.getElem_zip]
    have hpair : (ss[k]'hk_ss, rs[k]'hk_rs) ∈ List.zip ss rs := by
      rw [← hzip_get]; exact List.getElem_mem hzip_len
    have hflag : keepUserPayload (ss[k]'hk_ss) = keepUserPayload (rs[k]'hk_rs) :=
      hflags _ hpair
    congr 1
    · exact ih (Nat.le_of_succ_le hks) (Nat.le_of_succ_le hkr)
    · cases hb : keepUserPayload (ss[k]'hk_ss) <;>
        simp [List.filter_cons, hb, show keepUserPayload (rs[k]'hk_rs) = _ from hflag ▸ hb]

/-- The erased-FIFO count equality implies that original send and recv
    FIFO indices coincide (nS = nR). This is the FIFO-correspondence for
    erasure: control messages are matched in pairs, so removing them
    preserves the pairing index. -/
private theorem erased_fifo_idx_eq
    (ss rs : List Payload)
    (hflags : ∀ p ∈ List.zip ss rs, keepUserPayload p.1 = keepUserPayload p.2)
    (hlen : rs.length ≤ ss.length)
    (nS nR : Nat)
    (hnS : nS < ss.length)
    (hnR : nR < rs.length)
    (hSuser : keepUserPayload (ss[nS]'hnS) = true)
    (hRuser : keepUserPayload (rs[nR]'hnR) = true)
    (heq : ((ss.take nS).filter keepUserPayload).length =
           ((rs.take nR).filter keepUserPayload).length) :
    nS = nR := by
  -- Prefix count equality for same-length prefixes
  have hprefix : ∀ k, k ≤ ss.length → k ≤ rs.length →
      ((ss.take k).filter keepUserPayload).length =
      ((rs.take k).filter keepUserPayload).length :=
    filter_prefix_count_eq_of_zip_flags ss rs hflags
  rcases Nat.lt_trichotomy nS nR with hnlt | heq | hngt
  · -- Case nS < nR: rs[nS] is user (matched with user ss[nS])
    have hnS_lt_rs : nS < rs.length := Nat.lt_trans hnlt hnR
    have hzip_len : nS < (List.zip ss rs).length := by
      simp [List.length_zip]; exact Nat.lt_min.mpr ⟨hnS, hnS_lt_rs⟩
    have hzip_get : (List.zip ss rs)[nS]'hzip_len = (ss[nS]'hnS, rs[nS]'hnS_lt_rs) := by
      simp [List.getElem_zip]
    have hpair : (ss[nS]'hnS, rs[nS]'hnS_lt_rs) ∈ List.zip ss rs := by
      rw [← hzip_get]; exact List.getElem_mem hzip_len
    have hflag_rs_nS : keepUserPayload (rs[nS]'hnS_lt_rs) = true := by
      rw [← hflags _ hpair]; exact hSuser
    -- prefix count at nS is the same on both sides
    have hpre_nS : ((ss.take nS).filter keepUserPayload).length =
                   ((rs.take nS).filter keepUserPayload).length :=
      hprefix nS (Nat.le_of_lt hnS) (Nat.le_of_lt hnS_lt_rs)
    -- prefix count at nS+1 on rs side is one more
    have hpre_nS1 : ((rs.take (nS + 1)).filter keepUserPayload).length =
                    ((rs.take nS).filter keepUserPayload).length + 1 := by
      rw [List.take_succ_eq_append_getElem hnS_lt_rs,
          List.filter_append, List.length_append]
      simp [List.filter_cons, hflag_rs_nS]
    -- prefix count at nR on rs side is ≥ nS+1 count
    have hpre_nR_ge : ((rs.take nR).filter keepUserPayload).length ≥
                      ((rs.take (nS + 1)).filter keepUserPayload).length :=
      filter_take_length_mono keepUserPayload rs hnlt
    -- Contradiction: heq + hpre_nS gives count at nS = count at nR, but nR ≥ nS+1
    omega
  · -- Case nS = nR: direct
    exact heq
  · -- Case nS > nR: ss[nR] is user (matched with user rs[nR])
    have hnR_lt_ss : nR < ss.length := Nat.lt_trans hngt hnS
    have hzip_len : nR < (List.zip ss rs).length := by
      simp [List.length_zip]; exact Nat.lt_min.mpr ⟨hnR_lt_ss, hnR⟩
    have hzip_get : (List.zip ss rs)[nR]'hzip_len = (ss[nR]'hnR_lt_ss, rs[nR]'hnR) := by
      simp [List.getElem_zip]
    have hpair : (ss[nR]'hnR_lt_ss, rs[nR]'hnR) ∈ List.zip ss rs := by
      rw [← hzip_get]; exact List.getElem_mem hzip_len
    have hflag_ss_nR : keepUserPayload (ss[nR]'hnR_lt_ss) = true := by
      rw [hflags _ hpair]; exact hRuser
    -- prefix count at nR is the same on both sides
    have hpre_nR : ((ss.take nR).filter keepUserPayload).length =
                   ((rs.take nR).filter keepUserPayload).length :=
      hprefix nR (Nat.le_of_lt hnR_lt_ss) (Nat.le_of_lt hnR)
    -- prefix count at nR+1 on ss side is one more
    have hpre_nR1 : ((ss.take (nR + 1)).filter keepUserPayload).length =
                    ((ss.take nR).filter keepUserPayload).length + 1 := by
      rw [List.take_succ_eq_append_getElem hnR_lt_ss,
          List.filter_append, List.length_append]
      simp [List.filter_cons, hflag_ss_nR]
    -- prefix count at nS on ss side is ≥ nR+1 count
    have hpre_nS_ge : ((ss.take nS).filter keepUserPayload).length ≥
                      ((ss.take (nR + 1)).filter keepUserPayload).length :=
      filter_take_length_mono keepUserPayload ss hngt
    -- Contradiction
    omega

/-- Acyclicity (existence of causal ranking) is preserved by erasure. -/
theorem hasAcyclicCausality_erase
    (M : WordTuple L C F Payload)
    (hM : IsMSC (L := L) (C := C) (F := F) (Payload := Payload) M) :
    hasAcyclicCausality (L := L) (C := C) (F := F) (Payload := Payload) (erase M) := by
  obtain ⟨R⟩ := hM.acyclic
  -- Define the erased ranking by mapping each erased-position back to its
  -- original position via eraseOriginRec, then using R.
  refine ⟨{
    rank := fun e =>
      R.rank ⟨e.lifeline,
        eraseOriginRec (C := C) (F := F) (Payload := Payload) (M e.lifeline) e.pos⟩
    local_mono := ?_
    fifo_mono := ?_
  }⟩
  · -- local_mono: eraseOriginRec is strictly monotone on surviving positions
    intro e1 e2 hll hlt hpos1 hpos2
    apply R.local_mono
    · exact hll
    · -- eraseOriginRec (M e1.lifeline) e1.pos < eraseOriginRec (M e2.lifeline) e2.pos
      -- Since hll : e1.lifeline = e2.lifeline, we need to unify the word
      have hmono : eraseOriginRec (C := C) (F := F) (Payload := Payload)
          (M e1.lifeline) e1.pos <
          eraseOriginRec (C := C) (F := F) (Payload := Payload)
          (M e1.lifeline) e2.pos :=
        eraseOriginRec_strictMono (C := C) (F := F) (Payload := Payload)
          (M e1.lifeline) hpos1 (hll ▸ hpos2) hlt
      calc eraseOriginRec (C := C) (F := F) (Payload := Payload) (M e1.lifeline) e1.pos
          < eraseOriginRec (C := C) (F := F) (Payload := Payload) (M e1.lifeline) e2.pos :=
              hmono
        _ = eraseOriginRec (C := C) (F := F) (Payload := Payload) (M e2.lifeline) e2.pos := by
              rw [hll]
    · exact eraseOriginRec_lt_length (C := C) (F := F) (Payload := Payload)
        (M e1.lifeline) e1.pos hpos1
    · exact eraseOriginRec_lt_length (C := C) (F := F) (Payload := Payload)
        (M e2.lifeline) e2.pos hpos2
  · -- fifo_mono: the FIFO pairing index is preserved by erasure
    intro A B j1 j2 hj1 hj2 hSend hRecv hFIFO
    -- Original positions
    let orig_j1 := eraseOriginRec (C := C) (F := F) (Payload := Payload) (M A) j1
    let orig_j2 := eraseOriginRec (C := C) (F := F) (Payload := Payload) (M B) j2
    have horig_j1 : orig_j1 < (M A).length :=
      eraseOriginRec_lt_length (C := C) (F := F) (Payload := Payload) (M A) j1 hj1
    have horig_j2 : orig_j2 < (M B).length :=
      eraseOriginRec_lt_length (C := C) (F := F) (Payload := Payload) (M B) j2 hj2
    -- Send/recv conditions at original positions
    -- (erase M A)[j1] = (M A)[orig_j1] by eraseWord_get_originRec
    have hget_A : (eraseWord (C := C) (F := F) (Payload := Payload) (M A))[j1]'hj1 =
        (M A)[orig_j1]'horig_j1 :=
      eraseWord_get_originRec (C := C) (F := F) (Payload := Payload) (M A) j1 hj1
    have hget_B : (eraseWord (C := C) (F := F) (Payload := Payload) (M B))[j2]'hj2 =
        (M B)[orig_j2]'horig_j2 :=
      eraseWord_get_originRec (C := C) (F := F) (Payload := Payload) (M B) j2 hj2
    -- Lift to List.get form expected by fifo_mono
    have hSend' : ((M A).get ⟨orig_j1, horig_j1⟩).val.isSendTo B = true := by
      have := hSend
      simp only [List.get_eq_getElem] at this ⊢
      rwa [← hget_A]
    have hRecv' : ((M B).get ⟨orig_j2, horig_j2⟩).val.isRecvFrom A = true := by
      have := hRecv
      simp only [List.get_eq_getElem] at this ⊢
      rwa [← hget_B]
    -- FIFO correspondence: translate erased count to original count
    -- ss = sendPayloads B (M A), nS = countSends B ((M A).take orig_j1)
    -- rs = recvPayloads A (M B), nR = countRecvs A ((M B).take orig_j2)
    let ss := sendPayloads (C := C) (F := F) (Payload := Payload) B (M A)
    let rs := recvPayloads (C := C) (F := F) (Payload := Payload) A (M B)
    let nS := countSends (C := C) (F := F) (Payload := Payload) B ((M A).take orig_j1)
    let nR := countRecvs (C := C) (F := F) (Payload := Payload) A ((M B).take orig_j2)
    have hFIFO' : nS = nR := by
      -- The erased FIFO condition gives: user-send-count prefix = user-recv-count prefix
      have hFIFO_filter_eq :
          ((ss.take nS).filter keepUserPayload).length =
          ((rs.take nR).filter keepUserPayload).length := by
        have hcs := countSends_erase_prefix_origin_filter (C := C) (F := F) (Payload := Payload)
          B (M A) j1 hj1
        have hcr := countRecvs_erase_prefix_origin_filter (C := C) (F := F) (Payload := Payload)
          A (M B) j2 hj2
        have hss_take := sendPayloads_take_prefix (C := C) (F := F) (Payload := Payload)
          (A := A) B (M A) orig_j1
        have hrs_take := recvPayloads_take_prefix (C := C) (F := F) (Payload := Payload)
          (B := B) A (M B) orig_j2
        simp only [ss, rs, nS, nR]
        rw [← hss_take, ← hrs_take]
        have := hFIFO
        simp only [eraseTuple] at this
        rw [hcs, hcr] at this
        exact this
      -- Matched pairs have the same control/user flag
      have hflags :
          ∀ p ∈ List.zip ss rs, keepUserPayload p.1 = keepUserPayload p.2 := by
        intro p hp
        exact congrArg (! ·)
          (ControlPayloadSpec.compat_preserves_isControl
            (hM.labelCompat A B p (by simpa [ss, rs] using hp)))
      -- Bound: rs.length ≤ ss.length
      have hlen : rs.length ≤ ss.length := by
        simpa [ss, rs, sendPayloads_length, recvPayloads_length, sndCount, rcvCount]
          using hM.noUnmatchedRecv A B
      -- hnS: nS < ss.length
      have hnS : nS < ss.length := by
        simp only [ss, nS, sendPayloads_length]
        have hSend_orig : ((M A)[orig_j1]'horig_j1).val.isSendTo B = true := by
          simp only [List.get_eq_getElem] at hSend'
          exact hSend'
        have h := countSends_take_lt_of_sendAt (C := C) (F := F) (Payload := Payload)
          B (M A) orig_j1 horig_j1 hSend_orig
        omega
      -- hnR: nR < rs.length
      have hnR : nR < rs.length := by
        simp only [rs, nR, recvPayloads_length]
        have hRecv_orig : ((M B)[orig_j2]'horig_j2).val.isRecvFrom A = true := by
          simp only [List.get_eq_getElem] at hRecv'
          exact hRecv'
        have h := countRecvs_take_lt_of_recvAt (C := C) (F := F) (Payload := Payload)
          A (M B) orig_j2 horig_j2 hRecv_orig
        omega
      -- ss[nS] is user-payload (orig send is non-control)
      have hSuser : keepUserPayload (ss[nS]'hnS) = true := by
        simp only [ss, nS]
        have hSend_orig : ((M A)[orig_j1]'horig_j1).val.isSendTo B = true := by
          simp only [List.get_eq_getElem] at hSend'
          exact hSend'
        have hisSome := eraseOriginRec_eraseLetter_isSome (C := C) (F := F) (Payload := Payload)
          (M A) j1 hj1
        -- eraseLetter returns some, so the letter is non-control
        -- The letter is a send (hSend_orig), so it's a non-control send
        have hIsCtrlFalse : isControlPayload
            ((sendPayloads (C := C) (F := F) (Payload := Payload) B (M A))[
              countSends (C := C) (F := F) (Payload := Payload) B ((M A).take orig_j1)]'hnS) =
            false := by
          obtain ⟨xs, owner, hneq, hval, hget⟩ :=
            sendPayloads_get_of_sendAt (C := C) (F := F) (Payload := Payload)
              B (M A) orig_j1 horig_j1 hSend_orig
          simp only [List.get_eq_getElem] at hget
          rw [hget]
          -- xs is the payload of the send; non-control because eraseLetter returns some
          have hsome : (eraseLetter (C := C) (F := F) (Payload := Payload)
              ((M A)[orig_j1]'horig_j1)).isSome = true := hisSome
          cases hctrl : isControlPayload xs
          · rfl
          · exfalso
            have hown : (Letter.sendLetter (C := C) (F := F) owner xs B hneq).owner = A := by
              have h2 := ((M A)[orig_j1]'horig_j1).2; rw [hval] at h2; exact h2
            rw [show ((M A)[orig_j1]'horig_j1) =
                    (⟨Letter.sendLetter (C := C) (F := F) owner xs B hneq, hown⟩ :
                      AlphabetOf (C := C) (F := F) (Payload := Payload) A)
                from Subtype.ext hval] at hsome
            simp [eraseLetter, hctrl] at hsome
        simp [keepUserPayload, hIsCtrlFalse]
      -- rs[nR] is user-payload (orig recv is non-control)
      have hRuser : keepUserPayload (rs[nR]'hnR) = true := by
        simp only [rs, nR]
        have hRecv_orig : ((M B)[orig_j2]'horig_j2).val.isRecvFrom A = true := by
          simp only [List.get_eq_getElem] at hRecv'
          exact hRecv'
        have hisSome := eraseOriginRec_eraseLetter_isSome (C := C) (F := F) (Payload := Payload)
          (M B) j2 hj2
        have hIsCtrlFalse : isControlPayload
            ((recvPayloads (C := C) (F := F) (Payload := Payload) A (M B))[
              countRecvs (C := C) (F := F) (Payload := Payload) A ((M B).take orig_j2)]'hnR) =
            false := by
          obtain ⟨ys, owner, hval, hget⟩ :=
            recvPayloads_get_of_recvAt (C := C) (F := F) (Payload := Payload)
              A (M B) orig_j2 horig_j2 hRecv_orig
          simp only [List.get_eq_getElem] at hget
          rw [hget]
          have hsome : (eraseLetter (C := C) (F := F) (Payload := Payload)
              ((M B)[orig_j2]'horig_j2)).isSome = true := hisSome
          cases hctrl : isControlPayload ys
          · rfl
          · exfalso
            have hown : (Letter.recvLetter (C := C) (F := F) owner ys A).owner = B := by
              have h2 := ((M B)[orig_j2]'horig_j2).2; rw [hval] at h2; exact h2
            rw [show ((M B)[orig_j2]'horig_j2) =
                    (⟨Letter.recvLetter (C := C) (F := F) owner ys A, hown⟩ :
                      AlphabetOf (C := C) (F := F) (Payload := Payload) B)
                from Subtype.ext hval] at hsome
            simp [eraseLetter, hctrl] at hsome
        simp [keepUserPayload, hIsCtrlFalse]
      -- Apply the key lemma
      exact erased_fifo_idx_eq ss rs hflags hlen nS nR hnS hnR hSuser hRuser hFIFO_filter_eq
    -- Apply R.fifo_mono with the translated conditions
    exact R.fifo_mono A B orig_j1 orig_j2 horig_j1 horig_j2 hSend' hRecv' hFIFO'

/-- If M is an MSC, then erase(M) is an MSC. -/
theorem IsMSC_erase
    (M : WordTuple L C F Payload)
    (hM : IsMSC (L := L) (C := C) (F := F) (Payload := Payload) M) :
    IsMSC (L := L) (C := C) (F := F) (Payload := Payload) (erase M) where
  noUnmatchedRecv := noUnmatchedReceives_erase M hM
  labelCompat := matchedLabelsCompatible_erase M hM
  acyclic := hasAcyclicCausality_erase M hM

/-- If M is a complete MSC, then erase(M) is a complete MSC. -/
theorem IsCompleteMSC_erase [Fintype L]
    (M : WordTuple L C F Payload)
    (hM : IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) M) :
    IsCompleteMSC (L := L) (C := C) (F := F) (Payload := Payload) (erase M) where
  complete := by
    intro A B
    simp only [channelComplete, sndCount, rcvCount, eraseTuple]
    have hbal := hM.complete A B
    simp only [channelComplete, sndCount, rcvCount] at hbal
    -- Balanced channels erase to balanced channels: control pairs cancel equally
    have hflags :
        ∀ p ∈ List.zip
          (sendPayloads (C := C) (F := F) (Payload := Payload) B (M A))
          (recvPayloads (C := C) (F := F) (Payload := Payload) A (M B)),
          keepUserPayload p.1 = keepUserPayload p.2 := by
      intro p hp
      exact congrArg (! ·)
        (ControlPayloadSpec.compat_preserves_isControl
          (hM.labelCompat A B p (by simpa using hp)))
    have hlen_eq : (recvPayloads (C := C) (F := F) (Payload := Payload) A (M B)).length =
        (sendPayloads (C := C) (F := F) (Payload := Payload) B (M A)).length := by
      simp [sendPayloads_length, recvPayloads_length, sndCount, rcvCount, hbal]
    rw [← sendPayloads_length (C := C) (F := F) (Payload := Payload) (B := B),
        ← recvPayloads_length (C := C) (F := F) (Payload := Payload) (A := A),
        sendPayloads_eraseWord, recvPayloads_eraseWord]
    -- The filtered lengths are equal by the same prefix-count argument
    have h_le : (sendPayloads (C := C) (F := F) (Payload := Payload) B (M A)).length ≤
        (recvPayloads (C := C) (F := F) (Payload := Payload) A (M B)).length := by
      omega
    have := filter_prefix_count_eq_of_zip_flags
      (sendPayloads (C := C) (F := F) (Payload := Payload) B (M A))
      (recvPayloads (C := C) (F := F) (Payload := Payload) A (M B))
      hflags
      (sendPayloads (C := C) (F := F) (Payload := Payload) B (M A)).length
      (Nat.le_refl _)
      h_le
    have key := this
    rw [List.take_length] at key
    rw [← hlen_eq, List.take_length] at key
    exact key
  labelCompat := (IsMSC_erase M (isCompleteMSC_implies_isMSC M hM)).labelCompat
  acyclic := (IsMSC_erase M (isCompleteMSC_implies_isMSC M hM)).acyclic

end Erasure
